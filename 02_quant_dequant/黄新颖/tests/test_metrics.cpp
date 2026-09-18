#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "quant_dequant/metrics.hpp"

namespace {

using quant_dequant::ArtifactMetrics;
using quant_dequant::CompressionMetrics;
using quant_dequant::DequantizationConfig;
using quant_dequant::DType;
using quant_dequant::ErrorMetrics;
using quant_dequant::KernelPerformance;
using quant_dequant::MetricsError;
using quant_dequant::QuantFormat;
using quant_dequant::QuantizationConfig;
using quant_dequant::QuantizedTensor;
using quant_dequant::QuantizedTensorDesc;
using quant_dequant::RoundingMode;
using quant_dequant::RunReport;
using quant_dequant::ScaleMode;
using quant_dequant::TensorDesc;
using quant_dequant::kNvfp4BlockSize;

/**
 * @brief 判断两个 double 是否在适合指标测试的绝对误差范围内相等。
 *
 * @param left 待比较的第一个数。
 * @param right 待比较的第二个数。
 * @return 差值不超过 1e-12 时返回 true。
 */
[[nodiscard]] bool nearly_equal(const double left, const double right) {
    return std::abs(left - right) <= 1.0e-12;
}

/**
 * @brief 构造一个 2×17、严格 NVFP4 block-scale 的自洽量化张量。
 *
 * 34 个元素恰好占 17 个 packed byte；每行需要两个 16 元素 local scale，因此
 * local_scales 长度为 4。数值 code 对本模块字节统计无关，只需满足数据模型。
 *
 * @return 可用于压缩率、带宽和报告 JSON 测试的 NVFP4 张量。
 */
[[nodiscard]] QuantizedTensor make_nvfp4_tensor() {
    return {
        .desc = QuantizedTensorDesc{
            .source_desc = TensorDesc{2U, 17U, DType::kFloat16},
            .format = QuantFormat::kNvfp4,
            .scale_mode = ScaleMode::kBlock,
            .rounding = RoundingMode::kNearest,
            .stochastic_seed = 0U,
            .block_size = kNvfp4BlockSize,
        },
        .payload = std::vector<std::uint8_t>(17U, 0x00U),
        .local_scales = std::vector<std::uint8_t>(4U, 0x00U),
        .global_scale = 0.5F,
    };
}

/**
 * @brief 验证误差三指标使用 double 累积、并拒绝不合法比较数组。
 *
 * @return 全部断言满足时返回 true。
 */
[[nodiscard]] bool test_error_metrics() {
    const std::vector<float> reference{1.0F, -2.0F, 3.0F};
    const std::vector<float> actual{0.0F, -4.0F, 6.0F};
    try {
        const ErrorMetrics metrics = quant_dequant::compute_error_metrics(
            reference, actual);
        if (!nearly_equal(metrics.max_abs, 3.0) ||
            !nearly_equal(metrics.mae, 2.0) ||
            !nearly_equal(metrics.mse, 14.0 / 3.0)) {
            return false;
        }
    } catch (const MetricsError& error) {
        std::cerr << "metrics 正常误差统计意外失败：" << error.what() << '\n';
        return false;
    }

    try {
        static_cast<void>(quant_dequant::compute_error_metrics(
            reference, std::vector<float>{1.0F}));
    } catch (const MetricsError&) {
        // 第二次调用必须因长度不同失败。
        return true;
    }

    return false;
}

/**
 * @brief 验证 NVFP4 逻辑字节、压缩率和由时间推导的带宽。
 *
 * @return 所有 payload、scale 和公式结果匹配时返回 true。
 */
[[nodiscard]] bool test_artifact_compression_and_performance_metrics() {
    const QuantizedTensor quantized = make_nvfp4_tensor();
    const TensorDesc output_desc{2U, 17U, DType::kBFloat16};
    try {
        const ArtifactMetrics artifacts = quant_dequant::make_artifact_metrics(
            quantized.desc.source_desc,
            quantized,
            output_desc,
            146U,
            132U);
        const CompressionMetrics compression =
            quant_dequant::compute_compression_metrics(artifacts);
        const KernelPerformance performance =
            quant_dequant::make_kernel_performance(
                artifacts, quantized, output_desc, 0.5, 0.25);

        // FP16 输入 34 项共 68 bytes；NVFP4 逻辑量化布局为 17-byte payload、
        // 4-byte local scale 和 4-byte global scale，共 25 bytes。
        const bool correct_artifacts =
            artifacts.input_payload_bytes == 68U &&
            artifacts.quantized_file_bytes == 146U &&
            artifacts.dequantized_file_bytes == 132U &&
            artifacts.payload_bytes == 17U && artifacts.local_scale_bytes == 4U &&
            artifacts.global_scale_bytes == 4U;
        const bool correct_compression =
            compression.logical_quantized_bytes == 25U &&
            nearly_equal(compression.logical_compression_ratio, 68.0 / 25.0) &&
            nearly_equal(compression.on_disk_compression_ratio, 68.0 / 146.0);
        const bool correct_performance =
            performance.quant_kernel_ms == std::optional<double>{0.5} &&
            performance.dequant_kernel_ms == std::optional<double>{0.25} &&
            performance.quant_effective_bandwidth_gbps.has_value() &&
            performance.dequant_effective_bandwidth_gbps.has_value() &&
            nearly_equal(*performance.quant_effective_bandwidth_gbps,
                         93.0 / (0.5 * 1.0e6)) &&
            nearly_equal(*performance.dequant_effective_bandwidth_gbps,
                         93.0 / (0.25 * 1.0e6));
        return correct_artifacts && correct_compression && correct_performance;
    } catch (const MetricsError& error) {
        std::cerr << "metrics 字节/压缩率/性能测试意外失败："
                  << error.what() << '\n';
        return false;
    }
}

/**
 * @brief 验证无 CUDA kernel 时间时性能 JSON 字段保留 null，且文本会正确转义。
 *
 * @return JSON 字段、转义内容和实际文件写入均正确时返回 true。
 */
[[nodiscard]] bool test_json_report_serialization_and_write() {
    const QuantizedTensor quantized = make_nvfp4_tensor();
    const TensorDesc output_desc{2U, 17U, DType::kBFloat16};
    try {
        const ArtifactMetrics artifacts = quant_dequant::make_artifact_metrics(
            quantized.desc.source_desc, quantized, output_desc, 146U, 132U);
        const CompressionMetrics compression =
            quant_dequant::compute_compression_metrics(artifacts);
        const KernelPerformance performance =
            quant_dequant::make_kernel_performance(
                artifacts, quantized, output_desc, std::nullopt, std::nullopt);
        const RunReport report{
            .input_desc = quantized.desc.source_desc,
            .quantization = QuantizationConfig{
                .format = QuantFormat::kNvfp4,
                .block_size = kNvfp4BlockSize,
                .scale_mode = ScaleMode::kBlock,
                .rounding = RoundingMode::kNearest,
                .stochastic_seed = 0U,
            },
            .dequantization = DequantizationConfig{DType::kBFloat16},
            .target_gpu = "RTX \"4060\"\nSM89",
            .artifacts = artifacts,
            .error = ErrorMetrics{.max_abs = 1.0, .mae = 0.5, .mse = 0.375},
            .compression = compression,
            .performance = performance,
        };
        const std::string json = quant_dequant::serialize_run_report_json(report);
        if (json.find("\"quant_kernel_ms\": null") == std::string::npos ||
            json.find("\"stochastic_seed\": 0") == std::string::npos ||
            json.find("RTX \\\"4060\\\"\\nSM89") == std::string::npos ||
            json.find("\"logical_quantized_bytes\": 25") == std::string::npos) {
            return false;
        }

        const std::filesystem::path report_path =
            std::filesystem::temp_directory_path() /
            "quant_dequant_metrics_report_test.json";
        std::error_code remove_error{};
        std::filesystem::remove(report_path, remove_error);
        quant_dequant::write_run_report_json(report_path, report);
        std::ifstream stream{report_path, std::ios::binary};
        const std::string file_json{
            std::istreambuf_iterator<char>{stream}, std::istreambuf_iterator<char>{}};
        std::filesystem::remove(report_path, remove_error);
        return stream.good() && file_json == json;
    } catch (const MetricsError& error) {
        std::cerr << "metrics JSON 报告测试意外失败：" << error.what() << '\n';
        return false;
    }
}

/**
 * @brief 验证极短 CUDA kernel 的 0 ms Event 时间仍可写入报告。
 *
 * Event 计时分辨率不足时，时间本身是有效观测值；只有带宽因除以零而未定义，
 * 应写为 null，不能让完整 app 因一个很小的输入失败。
 *
 * @return 时间保留为 0、两项带宽均为空时返回 true。
 */
[[nodiscard]] bool test_zero_duration_performance_metrics() {
    const QuantizedTensor quantized = make_nvfp4_tensor();
    const TensorDesc output_desc{2U, 17U, DType::kFloat32};
    try {
        const ArtifactMetrics artifacts = quant_dequant::make_artifact_metrics(
            quantized.desc.source_desc, quantized, output_desc, 146U, 200U);
        const CompressionMetrics compression =
            quant_dequant::compute_compression_metrics(artifacts);
        const KernelPerformance performance = quant_dequant::make_kernel_performance(
            artifacts, quantized, output_desc, 0.0, 0.0);
        const RunReport report{
            .input_desc = quantized.desc.source_desc,
            .quantization = QuantizationConfig{
                .format = QuantFormat::kNvfp4,
                .block_size = kNvfp4BlockSize,
                .scale_mode = ScaleMode::kBlock,
                .rounding = RoundingMode::kNearest,
                .stochastic_seed = 0U,
            },
            .dequantization = DequantizationConfig{DType::kFloat32},
            .target_gpu = "RTX 4060",
            .artifacts = artifacts,
            .error = ErrorMetrics{},
            .compression = compression,
            .performance = performance,
        };
        const std::string json = quant_dequant::serialize_run_report_json(report);
        return performance.quant_kernel_ms == std::optional<double>{0.0} &&
               performance.dequant_kernel_ms == std::optional<double>{0.0} &&
               !performance.quant_effective_bandwidth_gbps.has_value() &&
               !performance.dequant_effective_bandwidth_gbps.has_value() &&
               json.find("\"quant_kernel_ms\": 0") != std::string::npos &&
               json.find("\"quant_effective_bandwidth_gbps\": null") !=
                   std::string::npos;
    } catch (const MetricsError& error) {
        std::cerr << "metrics 0 ms CUDA Event 处理意外失败：" << error.what() << '\n';
        return false;
    }
}

}  // namespace

/**
 * @brief 运行误差、压缩率、性能与 JSON 报告的 metrics 单元测试。
 *
 * @return 所有断言通过时返回 0；否则返回 1。
 */
int run_metrics_tests() {
    if (!test_error_metrics()) {
        std::cerr << "metrics 误差统计测试失败。\n";
        return 1;
    }
    if (!test_artifact_compression_and_performance_metrics()) {
        std::cerr << "metrics 字节、压缩率或带宽测试失败。\n";
        return 1;
    }
    if (!test_json_report_serialization_and_write()) {
        std::cerr << "metrics JSON 报告测试失败。\n";
        return 1;
    }
    if (!test_zero_duration_performance_metrics()) {
        std::cerr << "metrics 0 ms CUDA Event 测试失败。\n";
        return 1;
    }

    return 0;
}
