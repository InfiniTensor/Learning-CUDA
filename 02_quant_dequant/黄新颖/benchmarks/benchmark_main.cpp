#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "quant_dequant/config.hpp"
#include "quant_dequant/metrics.hpp"
#include "quant_dequant/quantize.hpp"
#include "quant_dequant/tensor_io.hpp"
#include "quant_dequant/types.hpp"

namespace {

/** @brief benchmark 命令行解析或运行前参数校验失败时使用的异常。 */
class BenchmarkError final : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

/** @brief 同一类 kernel 时间样本的汇总统计。 */
struct TimingStatistics {
    /** 全部正式 repeat 中的最小 kernel 时间，单位毫秒。 */
    double min_ms{0.0};

    /** 全部正式 repeat 的算术均值，单位毫秒。 */
    double mean_ms{0.0};

    /** 排序后中位数；偶数样本取两个中间值的平均，单位毫秒。 */
    double median_ms{0.0};

    /** nearest-rank 定义的 p95，单位毫秒。 */
    double p95_ms{0.0};

    /** 全部正式 repeat 中的最大 kernel 时间，单位毫秒。 */
    double max_ms{0.0};
};

/** @brief benchmark 所需的明确命令行参数。 */
struct BenchmarkArguments {
    /** 一次读取后在整个 warmup/repeat 生命周期中复用的输入 QDTENSOR。 */
    std::filesystem::path input_path{};

    /** 完整 AppConfig 文件；量化与反量化子配置从中分别投影。 */
    std::filesystem::path config_path{};

    /** 可选 JSON 结果路径；省略时只向 stdout 写 JSON。 */
    std::optional<std::filesystem::path> output_path{};

    /** 不计入统计的同进程 warmup 次数。 */
    std::uint64_t num_warmups{10U};

    /** 计入统计的同进程正式 repeat 次数。 */
    std::uint64_t num_repeats{30U};
};

/**
 * @brief 将非负整数字符串解析为 uint64_t。
 *
 * @param text 不含前缀或空白的十进制文本。
 * @param option_name 用于错误信息的命令行选项名。
 * @return 解析成功的无符号整数。
 * @throws BenchmarkError 文本为空、含非数字或超出 uint64_t 范围时抛出。
 */
[[nodiscard]] std::uint64_t parse_unsigned_integer(
    const std::string_view text,
    const std::string_view option_name) {
    std::uint64_t value = 0U;
    const auto [end, error] = std::from_chars(
        text.data(), text.data() + text.size(), value);
    if (error != std::errc{} || end != text.data() + text.size()) {
        throw BenchmarkError("选项 " + std::string{option_name} +
                             " 必须是非负十进制整数。");
    }
    return value;
}

/**
 * @brief 解析 benchmark 命令行。
 *
 * @param argc 进程参数个数。
 * @param argv 进程参数数组。
 * @return 已校验必填路径及 warmup/repeat 约束的参数对象。
 * @throws BenchmarkError 未知选项、重复选项、缺少选项值或必填项时抛出。
 */
[[nodiscard]] BenchmarkArguments parse_arguments(const int argc, char* argv[]) {
    BenchmarkArguments arguments{};
    bool has_input = false;
    bool has_config = false;

    for (int index = 1; index < argc; ++index) {
        const std::string_view option{argv[index]};
        if (option == "--help") {
            std::cout
                << "用法：quant_dequant_bench --input <input.qdtensor> "
                   "--config <config.toml> [--warmups <N>] [--repeats <N>] "
                   "[--output <result.json>]\n";
            std::exit(0);
        }
        if (index + 1 >= argc) {
            throw BenchmarkError("选项 " + std::string{option} + " 缺少值。");
        }

        const std::string_view value{argv[++index]};
        if (option == "--input") {
            if (has_input) {
                throw BenchmarkError("--input 不能重复指定。");
            }
            arguments.input_path = std::filesystem::path{value};
            has_input = true;
        } else if (option == "--config") {
            if (has_config) {
                throw BenchmarkError("--config 不能重复指定。");
            }
            arguments.config_path = std::filesystem::path{value};
            has_config = true;
        } else if (option == "--output") {
            if (arguments.output_path.has_value()) {
                throw BenchmarkError("--output 不能重复指定。");
            }
            arguments.output_path = std::filesystem::path{value};
        } else if (option == "--warmups") {
            arguments.num_warmups = parse_unsigned_integer(value, option);
        } else if (option == "--repeats") {
            arguments.num_repeats = parse_unsigned_integer(value, option);
        } else {
            throw BenchmarkError("未知选项：" + std::string{option});
        }
    }

    if (!has_input || !has_config) {
        throw BenchmarkError("--input 和 --config 都是必填项；使用 --help 查看用法。");
    }
    if (arguments.num_repeats == 0U) {
        throw BenchmarkError("--repeats 必须大于 0。");
    }
    return arguments;
}

/**
 * @brief 计算一组非空、有限且非负的毫秒样本的统计量。
 *
 * @param samples 仅包含正式 repeat 的 CUDA Event 时间，单位毫秒。
 * @return min、mean、median、p95、max 五项稳定统计。
 * @throws BenchmarkError 样本为空或包含无效时间时抛出。
 */
[[nodiscard]] TimingStatistics compute_timing_statistics(
    const std::vector<double>& samples) {
    if (samples.empty()) {
        throw BenchmarkError("无法对空的时间样本计算统计量。");
    }

    std::vector<double> sorted_samples = samples;
    double sum_ms = 0.0;
    for (const double sample_ms : sorted_samples) {
        if (!std::isfinite(sample_ms) || sample_ms < 0.0) {
            throw BenchmarkError("CUDA Event 返回了非有限或负的 kernel 时间。");
        }
        sum_ms += sample_ms;
    }
    std::sort(sorted_samples.begin(), sorted_samples.end());

    const std::size_t count = sorted_samples.size();
    const std::size_t upper_middle = count / 2U;
    const double median_ms = (count % 2U == 0U)
                                 ? (sorted_samples[upper_middle - 1U] +
                                    sorted_samples[upper_middle]) /
                                       2.0
                                 : sorted_samples[upper_middle];
    // nearest-rank p95：第 ceil(0.95 * N) 个 1-based 样本。
    const std::size_t p95_index = (95U * count + 99U) / 100U - 1U;

    return TimingStatistics{
        .min_ms = sorted_samples.front(),
        .mean_ms = sum_ms / static_cast<double>(count),
        .median_ms = median_ms,
        .p95_ms = sorted_samples[p95_index],
        .max_ms = sorted_samples.back(),
    };
}

/**
 * @brief 将 UTF-8 字节串转为可嵌入 JSON 的最小 string literal。
 *
 * @param value 待转义的文本，例如 target_gpu。
 * @return 已含双引号的 JSON string literal。
 */
[[nodiscard]] std::string json_string(const std::string_view value) {
    std::ostringstream stream{};
    stream << '"';
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
    stream << '"';
    return stream.str();
}

/**
 * @brief 将 kernel 统计及其由均值换算的有效带宽写入 JSON。
 *
 * @param stream 目标 JSON 输出流。
 * @param statistics 已完成统计的时间对象。
 * @param logical_bytes 当前阶段按项目约定的逻辑读写字节数。
 */
void write_timing_json(
    std::ostream& stream,
    const TimingStatistics& statistics,
    const std::uint64_t logical_bytes) {
    stream << std::setprecision(12)
           << "{\n"
           << "      \"min_ms\": " << statistics.min_ms << ",\n"
           << "      \"mean_ms\": " << statistics.mean_ms << ",\n"
           << "      \"median_ms\": " << statistics.median_ms << ",\n"
           << "      \"p95_ms\": " << statistics.p95_ms << ",\n"
           << "      \"max_ms\": " << statistics.max_ms << ",\n"
           << "      \"effective_bandwidth_gbps_from_mean\": ";
    if (statistics.mean_ms == 0.0) {
        stream << "null\n";
    } else {
        stream << quant_dequant::compute_effective_bandwidth_gbps(
                      logical_bytes, statistics.mean_ms)
               << '\n';
    }
    stream << "    }";
}

/**
 * @brief 组装 benchmark 的自描述 JSON，不写文件也不访问 CUDA。
 *
 * @param input_desc 原始输入描述。
 * @param app_config 完整配置，用于记录格式和输出 dtype。
 * @param num_warmups 未纳入统计的同进程 warmup 数。
 * @param num_repeats 纳入统计的正式样本数。
 * @param quant_statistics 量化 kernel 时间统计。
 * @param dequant_statistics 反量化 kernel 时间统计。
 * @param quant_logical_bytes 量化阶段的逻辑读写字节数。
 * @param dequant_logical_bytes 反量化阶段的逻辑读写字节数。
 * @return 以换行结尾的 UTF-8 JSON 文本。
 */
[[nodiscard]] std::string make_benchmark_json(
    const quant_dequant::TensorDesc& input_desc,
    const quant_dequant::AppConfig& app_config,
    const std::uint64_t num_warmups,
    const std::uint64_t num_repeats,
    const TimingStatistics& quant_statistics,
    const TimingStatistics& dequant_statistics,
    const std::uint64_t quant_logical_bytes,
    const std::uint64_t dequant_logical_bytes) {
    std::ostringstream stream{};
    stream << "{\n"
           << "  \"schema_version\": 1,\n"
           << "  \"input\": {\n"
           << "    \"rows\": " << input_desc.num_rows << ",\n"
           << "    \"cols\": " << input_desc.num_cols << ",\n"
           << "    \"dtype\": " << json_string(quant_dequant::to_string(input_desc.dtype))
           << "\n  },\n"
           << "  \"config\": {\n"
           << "    \"format\": "
           << json_string(quant_dequant::to_string(app_config.quantization.format))
           << ",\n"
           << "    \"block_size\": " << app_config.quantization.block_size << ",\n"
           << "    \"scale_mode\": "
           << json_string(quant_dequant::to_string(app_config.quantization.scale_mode))
           << ",\n"
           << "    \"rounding\": "
           << json_string(quant_dequant::to_string(app_config.quantization.rounding))
           << ",\n"
           << "    \"output_type\": "
           << json_string(quant_dequant::to_string(app_config.dequantization.output_type))
           << ",\n"
           << "    \"target_gpu\": "
           << json_string(app_config.report.target_gpu) << "\n"
           << "  },\n"
           << "  \"benchmark\": {\n"
           << "    \"warmups\": " << num_warmups << ",\n"
           << "    \"repeats\": " << num_repeats << ",\n"
           << "    \"timing_scope\": \"CUDA Event；仅格式专用 kernel，不含 H2D、D2H、分配、文件 I/O 或 context 初始化\"\n"
           << "  },\n"
           << "  \"performance\": {\n"
           << "    \"quant\": ";
    write_timing_json(stream, quant_statistics, quant_logical_bytes);
    stream << ",\n    \"dequant\": ";
    write_timing_json(stream, dequant_statistics, dequant_logical_bytes);
    stream << "\n  }\n}\n";
    return stream.str();
}

/**
 * @brief 创建 output 路径的父目录，并写入完整 JSON。
 *
 * @param output_path 要创建或覆盖的 JSON 文件路径。
 * @param json_text 已完成序列化的 JSON 文本。
 * @throws BenchmarkError 无法创建目录或写入完整文件时抛出。
 */
void write_json_file(
    const std::filesystem::path& output_path,
    const std::string_view json_text) {
    const std::filesystem::path parent_path = output_path.parent_path();
    if (!parent_path.empty()) {
        std::error_code error{};
        std::filesystem::create_directories(parent_path, error);
        if (error) {
            throw BenchmarkError("无法创建 benchmark 输出目录：" +
                                 parent_path.string() + "：" + error.message());
        }
    }

    std::ofstream stream{output_path, std::ios::binary | std::ios::trunc};
    if (!stream) {
        throw BenchmarkError("无法写入 benchmark JSON：" + output_path.string());
    }
    stream.write(json_text.data(), static_cast<std::streamsize>(json_text.size()));
    if (!stream) {
        throw BenchmarkError("写入 benchmark JSON 时失败：" + output_path.string());
    }
}

/**
 * @brief 在同一进程中完成 CUDA warmup 和重复量化/反量化测量。
 *
 * 输入 QDTENSOR 与配置只读取一次。每轮调用 profile pipeline；该 pipeline 的
 * CUDA Event 位于 H2D 后、D2H 前，故收集的时间严格只覆盖格式专用 kernel。warmup
 * 轮完全不进入统计，用于让 CUDA context、模块加载、GPU 时钟和缓存状态趋于稳定。
 *
 * @param arguments 已校验的 benchmark 命令行参数。
 * @return 成功时返回 0。
 */
int run_benchmark(const BenchmarkArguments& arguments) {
    const quant_dequant::HostTensor input =
        quant_dequant::read_input_tensor(arguments.input_path);
    const quant_dequant::AppConfig app_config =
        quant_dequant::load_app_config(arguments.config_path);

    // warmup 与正式测量都走 quantize -> dequantize 数据依赖，保证 MXFP8 tensor
    // reduction、NVFP4 global/local scale 与 pack/unpack 等完整格式路径都被执行。
    for (std::uint64_t iteration = 0U;
         iteration < arguments.num_warmups;
         ++iteration) {
        const auto quantized = quant_dequant::quantize_cuda_profiled(
            input, app_config.quantization);
        [[maybe_unused]] const auto dequantized =
            quant_dequant::dequantize_cuda_profiled(
                quantized.tensor, app_config.dequantization);
    }

    std::vector<double> quant_samples_ms{};
    std::vector<double> dequant_samples_ms{};
    quant_samples_ms.reserve(static_cast<std::size_t>(arguments.num_repeats));
    dequant_samples_ms.reserve(static_cast<std::size_t>(arguments.num_repeats));

    // 最后一轮的量化布局与前面各轮相同；它仅用于推导报告中的逻辑读写字节数。
    std::optional<quant_dequant::QuantizedTensor> last_quantized{};
    std::optional<quant_dequant::HostTensor> last_dequantized{};
    for (std::uint64_t iteration = 0U;
         iteration < arguments.num_repeats;
         ++iteration) {
        const auto quantized = quant_dequant::quantize_cuda_profiled(
            input, app_config.quantization);
        const auto dequantized = quant_dequant::dequantize_cuda_profiled(
            quantized.tensor, app_config.dequantization);

        quant_samples_ms.push_back(quantized.kernel_ms);
        dequant_samples_ms.push_back(dequantized.kernel_ms);
        last_quantized = quantized.tensor;
        last_dequantized = dequantized.tensor;
    }

    const TimingStatistics quant_statistics =
        compute_timing_statistics(quant_samples_ms);
    const TimingStatistics dequant_statistics =
        compute_timing_statistics(dequant_samples_ms);
    const std::uint64_t input_bytes =
        quant_dequant::compute_input_payload_bytes(input.desc);
    const std::uint64_t quantized_bytes =
        quant_dequant::compute_logical_quantized_bytes(*last_quantized);
    const std::uint64_t output_bytes =
        quant_dequant::compute_dequantized_payload_bytes(last_dequantized->desc);
    const std::uint64_t quant_logical_bytes = input_bytes + quantized_bytes;
    const std::uint64_t dequant_logical_bytes = quantized_bytes + output_bytes;
    const std::string json_text = make_benchmark_json(
        input.desc,
        app_config,
        arguments.num_warmups,
        arguments.num_repeats,
        quant_statistics,
        dequant_statistics,
        quant_logical_bytes,
        dequant_logical_bytes);

    if (arguments.output_path.has_value()) {
        write_json_file(*arguments.output_path, json_text);
    }
    std::cout << json_text;
    return 0;
}

}  // namespace

int main(int argc, char* argv[]) {
    try {
        return run_benchmark(parse_arguments(argc, argv));
    } catch (const std::exception& error) {
        std::cerr << "quant_dequant_bench 失败：" << error.what() << '\n';
        return 1;
    }
}
