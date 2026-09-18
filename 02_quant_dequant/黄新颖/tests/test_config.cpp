#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <string_view>
#include <utility>

#include "quant_dequant/config.hpp"

namespace {

/**
 * @brief 删除测试过程中创建的临时配置文件。
 */
class TemporaryFileCleanup final {
public:
    /**
     * @brief 记录需要在析构时删除的文件路径。
     *
     * @param file_path 测试创建的临时文件路径。
     */
    explicit TemporaryFileCleanup(std::filesystem::path file_path)
        : mFilePath(std::move(file_path)) {}

    TemporaryFileCleanup(const TemporaryFileCleanup&) = delete;
    TemporaryFileCleanup& operator=(const TemporaryFileCleanup&) = delete;
    TemporaryFileCleanup(TemporaryFileCleanup&&) = delete;
    TemporaryFileCleanup& operator=(TemporaryFileCleanup&&) = delete;

    /**
     * @brief 尽力清理临时文件；清理失败不应覆盖原始测试结果。
     */
    ~TemporaryFileCleanup() {
        std::error_code error_code{};
        std::filesystem::remove(mFilePath, error_code);
    }

private:
    std::filesystem::path mFilePath;
};

/**
 * @brief 向指定路径写入 UTF-8 配置文本。
 *
 * @param file_path 输出文件路径。
 * @param content 要写入的完整文本。
 * @return 文件完全写入时返回 true。
 */
[[nodiscard]] bool write_text_file(const std::filesystem::path& file_path,
                                   const std::string_view content) {
    std::ofstream output_stream{file_path};
    output_stream << content;
    return output_stream.good();
}

}  // namespace

/**
 * @brief 验证完整 app 与两个单方向配置入口的解析和诊断。
 *
 * @return 所有断言通过时返回 0；失败时打印原因并返回 1。
 */
int run_config_parser_tests() {
    const auto timestamp = std::chrono::steady_clock::now().time_since_epoch().count();
    const std::filesystem::path config_path =
        std::filesystem::temp_directory_path() /
        ("quant_dequant_config_test_" + std::to_string(timestamp) + ".toml");
    const TemporaryFileCleanup cleanup{config_path};

    constexpr std::string_view kFullAppConfig = R"(
# 注释行应被忽略。
format = "nvfp4"
block_size = 16
scale_mode = "block"  # 行尾注释应被忽略。
output_type = "bf16"
rounding = "stochastic"
target_gpu = "RTX 4060 # 引号内不是注释"
stochastic_seed = 42
)";

    if (!write_text_file(config_path, kFullAppConfig)) {
        std::cerr << "无法创建配置解析测试文件。\n";
        return 1;
    }

    try {
        const quant_dequant::AppConfig app_config =
            quant_dequant::load_app_config(config_path);
        const bool valid_app_values =
            app_config.quantization.format == quant_dequant::QuantFormat::kNvfp4 &&
            app_config.quantization.block_size == quant_dequant::kNvfp4BlockSize &&
            app_config.quantization.scale_mode == quant_dequant::ScaleMode::kBlock &&
            app_config.quantization.rounding ==
                quant_dequant::RoundingMode::kStochastic &&
            app_config.quantization.stochastic_seed == 42U &&
            app_config.dequantization.output_type ==
                quant_dequant::DType::kBFloat16 &&
            app_config.report.target_gpu == "RTX 4060 # 引号内不是注释";

        if (!valid_app_values) {
            std::cerr << "完整 app 配置未被正确拆分。\n";
            return 1;
        }

        const quant_dequant::QuantizationConfig quantization_config =
            quant_dequant::load_quantization_config(config_path);
        const quant_dequant::DequantizationConfig dequantization_config =
            quant_dequant::load_dequantization_config(config_path);
        if (quantization_config.format != quant_dequant::QuantFormat::kNvfp4 ||
            dequantization_config.output_type != quant_dequant::DType::kBFloat16) {
            std::cerr << "完整配置不能正确投影为单方向配置。\n";
            return 1;
        }
    } catch (const quant_dequant::ConfigError& error) {
        std::cerr << "完整配置意外解析失败：" << error.what() << '\n';
        return 1;
    }

    constexpr std::string_view kQuantizationOnlyConfig = R"(
format = "mxfp8"
block_size = 32
scale_mode = "tensor"
rounding = "nearest"
)";

    if (!write_text_file(config_path, kQuantizationOnlyConfig)) {
        std::cerr << "无法写入单方向量化配置。\n";
        return 1;
    }

    try {
        const quant_dequant::QuantizationConfig config =
            quant_dequant::load_quantization_config(config_path);
        if (config.format != quant_dequant::QuantFormat::kMxfp8 ||
            config.block_size != quant_dequant::kMxfp8BlockSize ||
            config.scale_mode != quant_dequant::ScaleMode::kTensor ||
            config.rounding != quant_dequant::RoundingMode::kNearest ||
            config.stochastic_seed != 0U) {
            std::cerr << "单方向量化配置未被正确解析。\n";
            return 1;
        }
    } catch (const quant_dequant::ConfigError& error) {
        std::cerr << "单方向量化配置意外解析失败：" << error.what() << '\n';
        return 1;
    }

    try {
        static_cast<void>(quant_dequant::load_app_config(config_path));
        std::cerr << "缺少反量化和报告字段的配置被错误接受为完整 app 配置。\n";
        return 1;
    } catch (const quant_dequant::ConfigError&) {
    }

    constexpr std::string_view kDequantizationOnlyConfig = R"(
output_type = "fp32"
)";

    if (!write_text_file(config_path, kDequantizationOnlyConfig)) {
        std::cerr << "无法写入单方向反量化配置。\n";
        return 1;
    }

    try {
        const quant_dequant::DequantizationConfig config =
            quant_dequant::load_dequantization_config(config_path);
        if (config.output_type != quant_dequant::DType::kFloat32) {
            std::cerr << "单方向反量化配置未被正确解析。\n";
            return 1;
        }
    } catch (const quant_dequant::ConfigError& error) {
        std::cerr << "单方向反量化配置意外解析失败：" << error.what() << '\n';
        return 1;
    }

    constexpr std::string_view kInvalidNearestSeedConfig = R"(
format = "mxfp8"
block_size = 32
scale_mode = "block"
rounding = "nearest"
stochastic_seed = 7
)";

    if (!write_text_file(config_path, kInvalidNearestSeedConfig)) {
        std::cerr << "无法写入非法 seed 配置。\n";
        return 1;
    }

    try {
        static_cast<void>(quant_dequant::load_quantization_config(config_path));
        std::cerr << "nearest 配置的非零 stochastic_seed 未被拒绝。\n";
        return 1;
    } catch (const quant_dequant::ConfigError& error) {
        if (error.lineNumber() != 6U) {
            std::cerr << "非法 stochastic_seed 的错误行号不正确。\n";
            return 1;
        }
    }

    constexpr std::string_view kInvalidNvfp4TensorConfig = R"(
format = "nvfp4"
block_size = 16
scale_mode = "tensor"
rounding = "nearest"
)";

    if (!write_text_file(config_path, kInvalidNvfp4TensorConfig)) {
        std::cerr << "无法写入非法 NVFP4 tensor 配置。\n";
        return 1;
    }

    try {
        static_cast<void>(quant_dequant::load_quantization_config(config_path));
        std::cerr << "NVFP4 tensor scale 配置未被拒绝。\n";
        return 1;
    } catch (const quant_dequant::ConfigError& error) {
        if (error.lineNumber() != 4U) {
            std::cerr << "NVFP4 tensor scale 的错误行号不正确。\n";
            return 1;
        }
    }

    constexpr std::string_view kDuplicateConfig = R"(
format = "mxfp8"
format = "mxfp8"
block_size = 32
scale_mode = "block"
output_type = "fp16"
rounding = "nearest"
target_gpu = "RTX 4060"
)";

    if (!write_text_file(config_path, kDuplicateConfig)) {
        std::cerr << "无法写入重复字段测试配置。\n";
        return 1;
    }

    try {
        static_cast<void>(quant_dequant::load_app_config(config_path));
    } catch (const quant_dequant::ConfigError& error) {
        if (error.lineNumber() == 3U) {
            return 0;
        }

        std::cerr << "重复字段错误行号不正确。\n";
        return 1;
    }

    std::cerr << "重复字段未被拒绝。\n";
    return 1;
}
