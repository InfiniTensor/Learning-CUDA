#include <cstdint>
#include <exception>
#include <filesystem>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

#include "quant_dequant/config.hpp"
#include "quant_dequant/metrics.hpp"
#include "quant_dequant/quantize.hpp"
#include "quant_dequant/quantized_io.hpp"
#include "quant_dequant/tensor_io.hpp"
#include "quant_dequant/version.hpp"

namespace {

/** @brief `quant_dequant` 完整运行所需的全部命令行文件路径。 */
struct AppOptions {
    /** 输入用途的 QDTENSOR 文件，物理 dtype 为 FP16 或 FP32。 */
    std::filesystem::path input_path{};

    /** 包含量化、反量化与报告字段的完整 key-value 配置文件。 */
    std::filesystem::path config_path{};

    /** 要创建或覆盖的真实 bit-width QDWGT 量化权重文件。 */
    std::filesystem::path quantized_output_path{};

    /** 要创建或覆盖的 FP16/BF16/FP32 QDTENSOR 反量化输出文件。 */
    std::filesystem::path dequantized_output_path{};

    /** 要创建或覆盖的误差、压缩率和 CUDA 性能 JSON 报告。 */
    std::filesystem::path report_path{};
};

/** @brief 表示 app 命令行、输出路径或编排过程的调用方错误。 */
class AppError final : public std::runtime_error {
public:
    /**
     * @brief 构造带有面向命令行用户诊断的 app 错误。
     *
     * @param detail 失败的具体原因。
     */
    explicit AppError(std::string detail) : std::runtime_error(std::move(detail)) {}
};

/**
 * @brief 打印完整流程 app 的命令行用法。
 *
 * @param stream 要写入 usage 的标准输出或错误流。
 */
void print_usage(std::ostream& stream) {
    stream << "用法：quant_dequant \\\n  --input <input.qdtensor> \\\n  --config <app_config.toml> \\\n  --quantized-output <output.qdwgt> \\\n  --dequantized-output <output.qdtensor> \\\n  --report <report.json>\n\n"
           << "完整流程：读取 FP16/FP32 QDTENSOR → CUDA 量化并写 QDWGT → "
              "CUDA 反量化并写 FP16/BF16/FP32 QDTENSOR → 写 JSON 指标报告。\n";
}

/**
 * @brief 读取一个需要紧随值的命令行选项。
 *
 * @param argc 命令行参数数量。
 * @param argv 原始命令行参数数组。
 * @param index 当前选项在 argv 中的下标；成功时会前移到 value 下标。
 * @param option_name 当前选项名，用于诊断。
 * @return 非空的文件路径 value。
 * @throws AppError 选项缺值或 value 为空时抛出。
 */
[[nodiscard]] std::filesystem::path consume_path_argument(
    const int argc,
    char* const argv[],
    int* const index,
    const std::string_view option_name) {
    if (*index + 1 >= argc) {
        throw AppError{"选项 \"" + std::string{option_name} + "\" 缺少路径 value。"};
    }

    ++(*index);
    const std::string_view value{argv[*index]};
    if (value.empty() || value.starts_with("--")) {
        throw AppError{"选项 \"" + std::string{option_name} + "\" 缺少有效路径 value。"};
    }

    return std::filesystem::path{value};
}

/**
 * @brief 赋值一个只能出现一次的路径选项。
 *
 * @param destination 尚未设置的目标路径槽位。
 * @param value 当前解析到的路径值。
 * @param option_name 当前选项名，用于重复选项诊断。
 * @throws AppError 同一选项出现多次时抛出。
 */
void assign_unique_path(std::filesystem::path* const destination,
                        std::filesystem::path value,
                        const std::string_view option_name) {
    if (!destination->empty()) {
        throw AppError{"选项 \"" + std::string{option_name} + "\" 不能重复出现。"};
    }
    *destination = std::move(value);
}

/**
 * @brief 解析完整流程所需的五个文件路径选项。
 *
 * @param argc 命令行参数数量。
 * @param argv 原始命令行参数数组。
 * @return 已通过语法和必填字段检查的 app 选项。
 * @throws AppError 未知选项、重复选项或缺少必填选项时抛出。
 */
[[nodiscard]] AppOptions parse_options(const int argc, char* const argv[]) {
    AppOptions options{};

    for (int index = 1; index < argc; ++index) {
        const std::string_view option{argv[index]};
        if (option == "--input") {
            assign_unique_path(
                &options.input_path,
                consume_path_argument(argc, argv, &index, option), option);
        } else if (option == "--config") {
            assign_unique_path(
                &options.config_path,
                consume_path_argument(argc, argv, &index, option), option);
        } else if (option == "--quantized-output") {
            assign_unique_path(
                &options.quantized_output_path,
                consume_path_argument(argc, argv, &index, option), option);
        } else if (option == "--dequantized-output") {
            assign_unique_path(
                &options.dequantized_output_path,
                consume_path_argument(argc, argv, &index, option), option);
        } else if (option == "--report") {
            assign_unique_path(
                &options.report_path,
                consume_path_argument(argc, argv, &index, option), option);
        } else {
            throw AppError{"不支持的命令行选项：\"" + std::string{option} + "\"。"};
        }
    }

    if (options.input_path.empty() || options.config_path.empty() ||
        options.quantized_output_path.empty() ||
        options.dequantized_output_path.empty() || options.report_path.empty()) {
        throw AppError{"完整流程必须提供 --input、--config、--quantized-output、"
                       "--dequantized-output 和 --report。"};
    }

    return options;
}

/**
 * @brief 将路径转换为可比较的绝对规范化词法路径。
 *
 * 此处不解析符号链接，因为输出路径在运行前可不存在；它主要防止用户直接把同一
 * 相对/绝对路径同时传给输入和输出，导致 app 覆盖自己的输入或配置文件。
 *
 * @param path 原始用户路径。
 * @return 绝对且按词法归一化后的路径。
 * @throws AppError 无法取得绝对路径时抛出。
 */
[[nodiscard]] std::filesystem::path normalized_path(const std::filesystem::path& path) {
    std::error_code error_code{};
    const std::filesystem::path absolute_path = std::filesystem::absolute(path, error_code);
    if (error_code) {
        throw AppError{"无法规范化路径 \"" + path.string() + "\"：" +
                       error_code.message()};
    }
    return absolute_path.lexically_normal();
}

/**
 * @brief 拒绝两项逻辑上不能使用相同路径的 app 输入/输出。
 *
 * @param first_path 第一条路径。
 * @param first_name 第一条路径的命令行选项名。
 * @param second_path 第二条路径。
 * @param second_name 第二条路径的命令行选项名。
 * @throws AppError 两条规范化路径相同，可能造成覆盖时抛出。
 */
void require_distinct_paths(const std::filesystem::path& first_path,
                            const std::string_view first_name,
                            const std::filesystem::path& second_path,
                            const std::string_view second_name) {
    if (normalized_path(first_path) == normalized_path(second_path)) {
        throw AppError{"选项 \"" + std::string{first_name} + "\" 与 \"" +
                       std::string{second_name} + "\" 不能指向同一个文件。"};
    }
}

/**
 * @brief 验证输入、配置与三个输出文件不会直接相互覆盖。
 *
 * @param options 已解析的完整 app 文件路径。
 * @throws AppError 任意冲突路径存在时抛出。
 */
void validate_path_separation(const AppOptions& options) {
    require_distinct_paths(options.input_path, "--input", options.config_path, "--config");
    require_distinct_paths(
        options.input_path, "--input", options.quantized_output_path,
        "--quantized-output");
    require_distinct_paths(
        options.input_path, "--input", options.dequantized_output_path,
        "--dequantized-output");
    require_distinct_paths(options.input_path, "--input", options.report_path, "--report");
    require_distinct_paths(
        options.config_path, "--config", options.quantized_output_path,
        "--quantized-output");
    require_distinct_paths(
        options.config_path, "--config", options.dequantized_output_path,
        "--dequantized-output");
    require_distinct_paths(options.config_path, "--config", options.report_path, "--report");
    require_distinct_paths(
        options.quantized_output_path, "--quantized-output",
        options.dequantized_output_path, "--dequantized-output");
    require_distinct_paths(
        options.quantized_output_path, "--quantized-output", options.report_path,
        "--report");
    require_distinct_paths(
        options.dequantized_output_path, "--dequantized-output", options.report_path,
        "--report");
}

/**
 * @brief 在写输出前创建其父目录。
 *
 * @param output_path 即将由 app 创建或覆盖的文件路径。
 * @throws AppError 父路径存在但不是目录，或创建目录失败时抛出。
 */
void ensure_output_parent_directory(const std::filesystem::path& output_path) {
    const std::filesystem::path parent_path = output_path.parent_path();
    if (parent_path.empty()) {
        return;
    }

    std::error_code error_code{};
    const bool created = std::filesystem::create_directories(parent_path, error_code);
    if (error_code) {
        throw AppError{"无法创建输出目录 \"" + parent_path.string() + "\"：" +
                       error_code.message()};
    }
    if (!created && !std::filesystem::is_directory(parent_path, error_code)) {
        throw AppError{"输出父路径不是可用目录：\"" + parent_path.string() + "\"。"};
    }
    if (error_code) {
        throw AppError{"无法检查输出目录 \"" + parent_path.string() + "\"：" +
                       error_code.message()};
    }
}

/**
 * @brief 安全取得一个已写入产物的完整文件字节数。
 *
 * @param file_path 已由对应 I/O 写入器成功写出的文件。
 * @param artifact_name 用于错误诊断的产物名称。
 * @return 可由 metrics 层使用的 uint64 文件总字节数。
 * @throws AppError 查询失败或字节数无法以 uint64 表示时抛出。
 */
[[nodiscard]] std::uint64_t get_file_size(const std::filesystem::path& file_path,
                                          const std::string_view artifact_name) {
    std::error_code error_code{};
    const std::uintmax_t bytes = std::filesystem::file_size(file_path, error_code);
    if (error_code || bytes > std::numeric_limits<std::uint64_t>::max()) {
        throw AppError{"无法取得 " + std::string{artifact_name} + " 文件大小：\"" +
                       file_path.string() + "\"。"};
    }
    return static_cast<std::uint64_t>(bytes);
}

/**
 * @brief 向终端输出一个 JSON 风格的可选数值。
 *
 * @param stream 目标终端流。
 * @param value 可选性能数值；无值时输出 `null`。
 */
void print_optional_number(std::ostream& stream, const std::optional<double>& value) {
    if (value.has_value()) {
        stream << *value;
    } else {
        stream << "null";
    }
}

/**
 * @brief 在终端输出一次完整运行的关键结果，详细稳定记录以 JSON 为准。
 *
 * @param options 本次运行的文件路径。
 * @param report 已成功写入 JSON 的结构化报告。
 */
void print_summary(const AppOptions& options, const quant_dequant::RunReport& report) {
    std::cout << "完整 CUDA 量化/反量化流程完成。\n"
              << "  QDWGT：" << options.quantized_output_path << '\n'
              << "  反量化 QDTENSOR：" << options.dequantized_output_path << '\n'
              << "  JSON 报告：" << options.report_path << '\n'
              << "  误差：max_abs=" << report.error.max_abs
              << ", mae=" << report.error.mae << ", mse=" << report.error.mse << '\n'
              << "  压缩率：logical=" << report.compression.logical_compression_ratio
              << ", on_disk=" << report.compression.on_disk_compression_ratio << '\n'
              << "  CUDA kernel：quant=";
    print_optional_number(std::cout, report.performance.quant_kernel_ms);
    std::cout << " ms, dequant=";
    print_optional_number(std::cout, report.performance.dequant_kernel_ms);
    std::cout << " ms\n  有效带宽：quant=";
    print_optional_number(
        std::cout, report.performance.quant_effective_bandwidth_gbps);
    std::cout << " GB/s, dequant=";
    print_optional_number(
        std::cout, report.performance.dequant_effective_bandwidth_gbps);
    std::cout << " GB/s\n";
}

/**
 * @brief 执行一次从 QDTENSOR 输入到 QDWGT、反量化 QDTENSOR 与 JSON 报告的完整流程。
 *
 * @param options 已完成命令行解析和路径隔离校验的输入/输出路径。
 * @throws ConfigError、TensorIoError、QuantizedIoError、CudaPipelineError、MetricsError
 * 或 AppError 任一阶段不能继续时抛出。
 */
void run_full_pipeline(const AppOptions& options) {
    // 配置和原始 QDTENSOR 在产生任何输出前读取/验证；这样格式、dtype、NVFP4
    // block-mode 等用户错误不会留下部分输出文件。
    const quant_dequant::AppConfig config =
        quant_dequant::load_app_config(options.config_path);
    const quant_dequant::HostTensor input =
        quant_dequant::read_input_tensor(options.input_path);

    ensure_output_parent_directory(options.quantized_output_path);
    ensure_output_parent_directory(options.dequantized_output_path);
    ensure_output_parent_directory(options.report_path);

    // profile API 在同一条独占 stream 上完成 H2D、kernel 与 D2H；返回的 Event
    // 时间仅覆盖量化 kernel。QDWGT 写入器保留 payload 的真实 bit-width：NVFP4
    // 每个字节保存两个 E2M1 nibble。
    const quant_dequant::ProfiledQuantizationResult profiled_quantized =
        quant_dequant::quantize_cuda_profiled(input, config.quantization);
    quant_dequant::write_quantized_tensor(
        options.quantized_output_path, profiled_quantized.tensor);

    // 完整 app 不直接把内存中的量化对象交给反量化，而是先经 QDWGT writer/read
    // 边界。由此验证实际保存的 packed payload、local scale、NVFP4 global scale
    // 和 header 能独立重建反量化所需的 QuantizedTensor。
    const quant_dequant::QuantizedTensor stored_quantized =
        quant_dequant::read_quantized_tensor(options.quantized_output_path);

    // 反量化仍在 CUDA kernel 中完成；输出 HostTensor 的 values 统一是 FP32，
    // desc.dtype 则让 writer 按配置真正写成 FP16、BF16 或 FP32 payload。
    const quant_dequant::ProfiledDequantizationResult profiled_dequantized =
        quant_dequant::dequantize_cuda_profiled(
            stored_quantized, config.dequantization);
    quant_dequant::write_dequantized_tensor(
        options.dequantized_output_path, profiled_dequantized.tensor);

    // 重新读取实际落盘的输出。对于 FP16/BF16，这一步将文件中的窄化值扩展为
    // FP32；故误差统计精确包含物理输出 dtype 的舍入误差，而非只统计 kernel
    // 在 FP32 暂存值上的误差。
    const quant_dequant::HostTensor stored_dequantized =
        quant_dequant::read_dequantized_tensor(options.dequantized_output_path);

    const std::uint64_t quantized_file_bytes =
        get_file_size(options.quantized_output_path, "QDWGT");
    const std::uint64_t dequantized_file_bytes =
        get_file_size(options.dequantized_output_path, "反量化 QDTENSOR");
    const quant_dequant::ArtifactMetrics artifacts =
        quant_dequant::make_artifact_metrics(
            input.desc, stored_quantized, stored_dequantized.desc,
            quantized_file_bytes, dequantized_file_bytes);
    const quant_dequant::RunReport report{
        .input_desc = input.desc,
        .quantization = config.quantization,
        .dequantization = config.dequantization,
        .target_gpu = config.report.target_gpu,
        .artifacts = artifacts,
        .error = quant_dequant::compute_error_metrics(
            input.values, stored_dequantized.values),
        .compression = quant_dequant::compute_compression_metrics(artifacts),
        .performance = quant_dequant::make_kernel_performance(
            artifacts, stored_quantized, stored_dequantized.desc,
            profiled_quantized.kernel_ms, profiled_dequantized.kernel_ms),
    };
    quant_dequant::write_run_report_json(options.report_path, report);
    print_summary(options, report);
}

}  // namespace

/**
 * @brief 执行 CUDA 低精度量化项目的完整命令行流程。
 *
 * @param argc 命令行参数数量。
 * @param argv 原始命令行参数数组。
 * @return 正常完成或显示帮助时返回 0；参数、I/O、CUDA 或指标错误时返回 1。
 */
int main(const int argc, char* argv[]) {
    if (argc == 2 && std::string_view{argv[1]} == "--help") {
        print_usage(std::cout);
        return 0;
    }

    try {
        const AppOptions options = parse_options(argc, argv);
        validate_path_separation(options);
        run_full_pipeline(options);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "quant_dequant 运行失败：" << error.what() << '\n';
        print_usage(std::cerr);
        return 1;
    }
}
