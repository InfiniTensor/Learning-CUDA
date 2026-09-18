#include "quant_dequant/metrics.hpp"

#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string_view>
#include <utility>

namespace quant_dequant {
namespace {

/** NVFP4 的单个 FP32 global scale 在逻辑布局中的字节数。 */
constexpr std::uint64_t kNvfp4GlobalScaleBytes = sizeof(float);

/** 十进制 GB/s 中 1 GB 对应的字节数量。 */
constexpr double kBytesPerMillisecondToGigabytesPerSecond = 1.0e6;

/**
 * @brief 返回一种物理 dtype 的单元素字节数。
 *
 * @param dtype 要查询的 QDTENSOR 物理类型。
 * @param require_input_dtype true 时只接受 FP16/FP32；false 时接受 FP16/BF16/FP32。
 * @return dtype 的物理字节宽度。
 * @throws MetricsError dtype 不属于所请求的输入或输出范围时抛出。
 */
[[nodiscard]] std::uint64_t dtype_byte_width(const DType dtype,
                                              const bool require_input_dtype) {
    if (require_input_dtype && !is_supported_input_dtype(dtype)) {
        throw MetricsError{"指标计算要求 FP16/FP32 输入 dtype。"};
    }
    if (!require_input_dtype && !is_supported_output_dtype(dtype)) {
        throw MetricsError{"指标计算要求 FP16/BF16/FP32 输出 dtype。"};
    }

    switch (dtype) {
        case DType::kFloat16:
        case DType::kBFloat16:
            return 2U;
        case DType::kFloat32:
            return 4U;
        case DType::kUnknown:
            break;
    }

    throw MetricsError{"指标计算遇到了未知 dtype。"};
}

/**
 * @brief 安全计算元素数和每元素字节数的乘积。
 *
 * @param element_count 已验证的非零元素数量。
 * @param byte_width 单元素物理字节数。
 * @param field_name 用于异常诊断的结果字段名。
 * @return 未发生 uint64 溢出的总字节数。
 * @throws MetricsError 乘法无法在 uint64 中表示时抛出。
 */
[[nodiscard]] std::uint64_t checked_byte_product(
    const std::uint64_t element_count,
    const std::uint64_t byte_width,
    const std::string_view field_name) {
    constexpr std::uint64_t kMaxValue = std::numeric_limits<std::uint64_t>::max();
    if (element_count == 0U || byte_width == 0U ||
        element_count > kMaxValue / byte_width) {
        throw MetricsError{
            "无法推导 \"" + std::string{field_name} + "\" 的合法字节数。"};
    }

    return element_count * byte_width;
}

/**
 * @brief 对多个 uint64 字节字段执行不溢出的求和。
 *
 * @param first 求和初值。
 * @param second 要加入的第二项。
 * @param field_name 用于异常诊断的总和字段名。
 * @return first + second。
 * @throws MetricsError 加法溢出时抛出。
 */
[[nodiscard]] std::uint64_t checked_add(const std::uint64_t first,
                                        const std::uint64_t second,
                                        const std::string_view field_name) {
    constexpr std::uint64_t kMaxValue = std::numeric_limits<std::uint64_t>::max();
    if (first > kMaxValue - second) {
        throw MetricsError{
            "\"" + std::string{field_name} + "\" 的字节数加法溢出。"};
    }

    return first + second;
}

/**
 * @brief 把字符串安全写为 JSON 字符串内容，不写两侧双引号。
 *
 * @param value 待转义的 UTF-8 字节串。
 * @return 可嵌入 JSON 双引号中的 ASCII 转义文本。
 */
[[nodiscard]] std::string escape_json_string(const std::string_view value) {
    std::ostringstream stream{};
    for (const unsigned char character : value) {
        switch (character) {
            case '"':
                stream << "\\\"";
                break;
            case '\\':
                stream << "\\\\";
                break;
            case '\b':
                stream << "\\b";
                break;
            case '\f':
                stream << "\\f";
                break;
            case '\n':
                stream << "\\n";
                break;
            case '\r':
                stream << "\\r";
                break;
            case '\t':
                stream << "\\t";
                break;
            default:
                if (character < 0x20U) {
                    stream << "\\u00" << std::hex << std::setw(2)
                           << std::setfill('0')
                           << static_cast<unsigned int>(character) << std::dec
                           << std::setfill(' ');
                } else {
                    stream << static_cast<char>(character);
                }
                break;
        }
    }

    return stream.str();
}

/**
 * @brief 向 JSON 输出流写入一个 optional double 字段。
 *
 * @param stream 已打开的 JSON 文本输出流。
 * @param value 待写值；无值时写 JSON null。
 */
void write_optional_json_number(std::ostringstream& stream,
                                const std::optional<double>& value) {
    if (value.has_value()) {
        stream << *value;
    } else {
        stream << "null";
    }
}

/**
 * @brief 验证 RunReport 中的配置、数值和派生字段彼此一致。
 *
 * @param report 待写入 JSON 的完整报告。
 * @throws MetricsError 任一基础元数据、误差、压缩率或可选性能字段不合法时抛出。
 */
void validate_run_report(const RunReport& report) {
    if (!report.input_desc.isValid() ||
        !is_supported_input_dtype(report.input_desc.dtype) ||
        !report.quantization.isValid() || !report.dequantization.isValid() ||
        report.target_gpu.empty()) {
        throw MetricsError{"运行报告的输入描述、配置或 target_gpu 不合法。"};
    }

    const auto input_payload_bytes = compute_input_payload_bytes(report.input_desc);
    const auto logical_quantized_bytes = report.compression.logical_quantized_bytes;
    if (report.artifacts.input_payload_bytes != input_payload_bytes ||
        report.artifacts.payload_bytes == 0U ||
        report.artifacts.local_scale_bytes == 0U ||
        report.artifacts.quantized_file_bytes == 0U ||
        logical_quantized_bytes == 0U ||
        !std::isfinite(report.error.max_abs) ||
        !std::isfinite(report.error.mae) || !std::isfinite(report.error.mse) ||
        report.error.max_abs < 0.0 || report.error.mae < 0.0 || report.error.mse < 0.0 ||
        !std::isfinite(report.compression.logical_compression_ratio) ||
        !std::isfinite(report.compression.on_disk_compression_ratio) ||
        report.compression.logical_compression_ratio <= 0.0 ||
        report.compression.on_disk_compression_ratio <= 0.0) {
        throw MetricsError{"运行报告包含不合法的指标或字节统计。"};
    }

    const bool has_quant_timing = report.performance.quant_kernel_ms.has_value();
    const bool has_quant_bandwidth =
        report.performance.quant_effective_bandwidth_gbps.has_value();
    const bool has_dequant_timing = report.performance.dequant_kernel_ms.has_value();
    const bool has_dequant_bandwidth =
        report.performance.dequant_effective_bandwidth_gbps.has_value();
    if ((!has_quant_timing && has_quant_bandwidth) ||
        (!has_dequant_timing && has_dequant_bandwidth)) {
        throw MetricsError{
            "有效带宽存在时必须同时提供对应的 kernel 时间。"};
    }

    for (const std::optional<double>& value : {
             report.performance.quant_kernel_ms,
             report.performance.dequant_kernel_ms,
         }) {
        if (value.has_value() && (!std::isfinite(*value) || *value < 0.0)) {
            throw MetricsError{"运行报告的 CUDA kernel 时间必须为有限非负数。"};
        }
    }

    for (const std::optional<double>& value : {
             report.performance.quant_effective_bandwidth_gbps,
             report.performance.dequant_effective_bandwidth_gbps,
         }) {
        if (value.has_value() && (!std::isfinite(*value) || *value <= 0.0)) {
            throw MetricsError{"运行报告的 CUDA 时间或有效带宽必须为有限正数。"};
        }
    }
}

}  // namespace

MetricsError::MetricsError(std::string detail)
    : std::runtime_error(std::move(detail)) {}

ErrorMetrics compute_error_metrics(const std::span<const float> reference,
                                   const std::span<const float> actual) {
    if (reference.empty() || actual.empty() || reference.size() != actual.size()) {
        throw MetricsError{"误差统计要求两个非空且等长的 FP32 数组。"};
    }

    double max_abs = 0.0;
    double absolute_sum = 0.0;
    double squared_sum = 0.0;
    for (std::size_t index = 0U; index < reference.size(); ++index) {
        const float reference_value = reference[index];
        const float actual_value = actual[index];
        if (!std::isfinite(reference_value) || !std::isfinite(actual_value)) {
            throw MetricsError{
                "误差统计不接受 NaN 或 Inf；发现线性下标为 " +
                std::to_string(index) + " 的非有限值。"};
        }

        const double difference = static_cast<double>(reference_value) -
            static_cast<double>(actual_value);
        const double absolute_difference = std::abs(difference);
        max_abs = max_abs > absolute_difference ? max_abs : absolute_difference;
        absolute_sum += absolute_difference;
        squared_sum += difference * difference;
    }

    const double count = static_cast<double>(reference.size());
    return {
        .max_abs = max_abs,
        .mae = absolute_sum / count,
        .mse = squared_sum / count,
    };
}

std::uint64_t compute_input_payload_bytes(const TensorDesc& input_desc) {
    const auto element_count = input_desc.elementCount();
    if (!input_desc.isValid() || !element_count.has_value()) {
        throw MetricsError{"无法从非法输入 TensorDesc 推导 payload 字节数。"};
    }

    return checked_byte_product(
        *element_count, dtype_byte_width(input_desc.dtype, true), "输入 payload");
}

std::uint64_t compute_dequantized_payload_bytes(const TensorDesc& output_desc) {
    const auto element_count = output_desc.elementCount();
    if (!output_desc.isValid() || !element_count.has_value()) {
        throw MetricsError{"无法从非法输出 TensorDesc 推导 payload 字节数。"};
    }

    return checked_byte_product(
        *element_count, dtype_byte_width(output_desc.dtype, false), "反量化输出 payload");
}

std::uint64_t compute_logical_quantized_bytes(const QuantizedTensor& quantized) {
    if (!quantized.isConsistent()) {
        throw MetricsError{"逻辑量化字节统计要求自洽的 QuantizedTensor。"};
    }

    const std::uint64_t payload_bytes =
        static_cast<std::uint64_t>(quantized.payload.size());
    const std::uint64_t local_scale_bytes =
        static_cast<std::uint64_t>(quantized.local_scales.size());
    const std::uint64_t global_scale_bytes = quantized.desc.usesGlobalScale()
        ? kNvfp4GlobalScaleBytes
        : 0U;
    return checked_add(
        checked_add(payload_bytes, local_scale_bytes, "逻辑量化字节数"),
        global_scale_bytes,
        "逻辑量化字节数");
}

ArtifactMetrics make_artifact_metrics(
    const TensorDesc& input_desc,
    const QuantizedTensor& quantized,
    const TensorDesc& dequantized_output_desc,
    const std::uint64_t quantized_file_bytes,
    const std::uint64_t dequantized_file_bytes) {
    if (!quantized.isConsistent()) {
        throw MetricsError{"产物统计要求自洽的 QuantizedTensor。"};
    }

    if (input_desc.num_rows != quantized.desc.source_desc.num_rows ||
        input_desc.num_cols != quantized.desc.source_desc.num_cols ||
        input_desc.dtype != quantized.desc.source_desc.dtype ||
        dequantized_output_desc.num_rows != input_desc.num_rows ||
        dequantized_output_desc.num_cols != input_desc.num_cols ||
        !is_supported_output_dtype(dequantized_output_desc.dtype) ||
        quantized_file_bytes == 0U || dequantized_file_bytes == 0U) {
        throw MetricsError{"产物统计的输入、量化、输出描述或文件大小不匹配。"};
    }

    return {
        .input_payload_bytes = compute_input_payload_bytes(input_desc),
        .quantized_file_bytes = quantized_file_bytes,
        .dequantized_file_bytes = dequantized_file_bytes,
        .payload_bytes = static_cast<std::uint64_t>(quantized.payload.size()),
        .local_scale_bytes =
            static_cast<std::uint64_t>(quantized.local_scales.size()),
        .global_scale_bytes = quantized.desc.usesGlobalScale()
            ? kNvfp4GlobalScaleBytes
            : 0U,
    };
}

CompressionMetrics compute_compression_metrics(const ArtifactMetrics& artifacts) {
    const std::uint64_t logical_quantized_bytes = checked_add(
        checked_add(artifacts.payload_bytes,
                    artifacts.local_scale_bytes,
                    "逻辑量化字节数"),
        artifacts.global_scale_bytes,
        "逻辑量化字节数");
    if (artifacts.input_payload_bytes == 0U || logical_quantized_bytes == 0U ||
        artifacts.quantized_file_bytes == 0U) {
        throw MetricsError{"压缩率要求输入、逻辑量化与 QDWGT 文件字节数均非零。"};
    }

    return {
        .logical_quantized_bytes = logical_quantized_bytes,
        .logical_compression_ratio =
            static_cast<double>(artifacts.input_payload_bytes) /
            static_cast<double>(logical_quantized_bytes),
        .on_disk_compression_ratio =
            static_cast<double>(artifacts.input_payload_bytes) /
            static_cast<double>(artifacts.quantized_file_bytes),
    };
}

double compute_effective_bandwidth_gbps(const std::uint64_t logical_bytes,
                                        const double kernel_ms) {
    if (logical_bytes == 0U || !std::isfinite(kernel_ms) || kernel_ms <= 0.0) {
        throw MetricsError{
            "有效带宽要求非零逻辑字节数及有限正的 kernel 时间。"};
    }

    return static_cast<double>(logical_bytes) /
        (kernel_ms * kBytesPerMillisecondToGigabytesPerSecond);
}

KernelPerformance make_kernel_performance(
    const ArtifactMetrics& artifacts,
    const QuantizedTensor& quantized,
    const TensorDesc& dequantized_output_desc,
    const std::optional<double> quant_kernel_ms,
    const std::optional<double> dequant_kernel_ms) {
    const std::uint64_t logical_quantized_bytes =
        compute_logical_quantized_bytes(quantized);
    if (artifacts.input_payload_bytes == 0U) {
        throw MetricsError{
            "性能指标要求非零的输入 payload 字节统计。"};
    }

    // 反量化 kernel 的逻辑写入量是 QDTENSOR payload，而不是包含 64-byte header
    // 的实际文件长度。显式接收输出描述可避免把固定 header 大小等文件格式细节
    // 泄漏到 kernel 指标公式中。
    const std::uint64_t dequantized_payload_bytes =
        compute_dequantized_payload_bytes(dequantized_output_desc);
    const std::uint64_t quant_logical_bytes = checked_add(
        artifacts.input_payload_bytes,
        logical_quantized_bytes,
        "量化 kernel 逻辑读写字节数");
    const std::uint64_t dequant_logical_bytes = checked_add(
        logical_quantized_bytes,
        dequantized_payload_bytes,
        "反量化 kernel 逻辑读写字节数");

    const auto make_optional_bandwidth = [](
                                            const std::uint64_t logical_bytes,
                                            const std::optional<double> kernel_ms)
        -> std::optional<double> {
        if (!kernel_ms.has_value()) {
            return std::nullopt;
        }
        if (!std::isfinite(*kernel_ms) || *kernel_ms < 0.0) {
            throw MetricsError{"CUDA kernel 时间必须为有限非负数。"};
        }
        if (*kernel_ms == 0.0) {
            // CUDA Event 的有限分辨率可能把极短工作量表示为 0 ms。保留时间，
            // 但不把有限逻辑字节除以 0 伪造成无限带宽。
            return std::nullopt;
        }
        return compute_effective_bandwidth_gbps(logical_bytes, *kernel_ms);
    };

    return {
        .quant_kernel_ms = quant_kernel_ms,
        .dequant_kernel_ms = dequant_kernel_ms,
        .quant_effective_bandwidth_gbps =
            make_optional_bandwidth(quant_logical_bytes, quant_kernel_ms),
        .dequant_effective_bandwidth_gbps =
            make_optional_bandwidth(dequant_logical_bytes, dequant_kernel_ms),
    };
}

std::string serialize_run_report_json(const RunReport& report) {
    validate_run_report(report);

    std::ostringstream stream{};
    stream << std::setprecision(std::numeric_limits<double>::max_digits10);
    stream << "{\n"
           << "  \"schema_version\": 1,\n"
           << "  \"input\": {\n"
           << "    \"rows\": " << report.input_desc.num_rows << ",\n"
           << "    \"cols\": " << report.input_desc.num_cols << ",\n"
           << "    \"dtype\": \"" << to_string(report.input_desc.dtype)
           << "\"\n"
           << "  },\n"
           << "  \"config\": {\n"
           << "    \"format\": \"" << to_string(report.quantization.format)
           << "\",\n"
           << "    \"block_size\": " << report.quantization.block_size << ",\n"
           << "    \"scale_mode\": \""
           << to_string(report.quantization.scale_mode) << "\",\n"
           << "    \"rounding\": \"" << to_string(report.quantization.rounding)
           << "\",\n"
           << "    \"stochastic_seed\": "
           << report.quantization.stochastic_seed << ",\n"
           << "    \"output_type\": \""
           << to_string(report.dequantization.output_type) << "\",\n"
           << "    \"target_gpu\": \"" << escape_json_string(report.target_gpu)
           << "\"\n"
           << "  },\n"
           << "  \"artifacts\": {\n"
           << "    \"quantized_file_bytes\": "
           << report.artifacts.quantized_file_bytes << ",\n"
           << "    \"dequantized_file_bytes\": "
           << report.artifacts.dequantized_file_bytes << ",\n"
           << "    \"payload_bytes\": " << report.artifacts.payload_bytes << ",\n"
           << "    \"local_scale_bytes\": "
           << report.artifacts.local_scale_bytes << ",\n"
           << "    \"global_scale_bytes\": "
           << report.artifacts.global_scale_bytes << "\n"
           << "  },\n"
           << "  \"error\": {\n"
           << "    \"max_abs\": " << report.error.max_abs << ",\n"
           << "    \"mae\": " << report.error.mae << ",\n"
           << "    \"mse\": " << report.error.mse << "\n"
           << "  },\n"
           << "  \"compression\": {\n"
           << "    \"input_payload_bytes\": "
           << report.artifacts.input_payload_bytes << ",\n"
           << "    \"logical_quantized_bytes\": "
           << report.compression.logical_quantized_bytes << ",\n"
           << "    \"logical_compression_ratio\": "
           << report.compression.logical_compression_ratio << ",\n"
           << "    \"on_disk_compression_ratio\": "
           << report.compression.on_disk_compression_ratio << "\n"
           << "  },\n"
           << "  \"performance\": {\n"
           << "    \"quant_kernel_ms\": ";
    write_optional_json_number(stream, report.performance.quant_kernel_ms);
    stream << ",\n"
           << "    \"dequant_kernel_ms\": ";
    write_optional_json_number(stream, report.performance.dequant_kernel_ms);
    stream << ",\n"
           << "    \"quant_effective_bandwidth_gbps\": ";
    write_optional_json_number(
        stream, report.performance.quant_effective_bandwidth_gbps);
    stream << ",\n"
           << "    \"dequant_effective_bandwidth_gbps\": ";
    write_optional_json_number(
        stream, report.performance.dequant_effective_bandwidth_gbps);
    stream << "\n  }\n}\n";
    return stream.str();
}

void write_run_report_json(const std::filesystem::path& output_path,
                           const RunReport& report) {
    const std::string json = serialize_run_report_json(report);
    std::ofstream stream{output_path, std::ios::binary | std::ios::trunc};
    if (!stream.is_open()) {
        throw MetricsError{
            "无法打开运行报告输出文件：" + output_path.string()};
    }

    stream << json;
    if (!stream.good()) {
        throw MetricsError{
            "写入运行报告 JSON 失败：" + output_path.string()};
    }
}

}  // namespace quant_dequant
