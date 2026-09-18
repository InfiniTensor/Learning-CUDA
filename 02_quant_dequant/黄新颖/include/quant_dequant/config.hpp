#pragma once

#include <cstddef>
#include <filesystem>
#include <stdexcept>
#include <string>

#include "quant_dequant/types.hpp"

namespace quant_dequant {

/**
 * @brief 表示量化配置文件的读取、语法或语义校验错误。
 *
 * 异常消息已经包含文件路径和可用时的行号；调用方也可以通过访问器取得
 * 结构化信息，用于命令行程序或测试中的额外诊断。
 */
class ConfigError final : public std::runtime_error {
public:
    /**
     * @brief 构造一个配置错误。
     *
     * @param file_path 发生错误的配置文件路径。
     * @param line_number 发生错误的 1-based 行号；值为 0 表示错误不对应某一行。
     * @param detail 面向用户的具体错误说明。
     */
    ConfigError(std::filesystem::path file_path, std::size_t line_number,
                std::string detail);

    /**
     * @brief 返回发生错误的配置文件路径。
     *
     * @return 不拥有文件系统资源的常量路径引用。
     */
    [[nodiscard]] const std::filesystem::path& filePath() const noexcept;

    /**
     * @brief 返回发生错误的 1-based 行号。
     *
     * @return 行号；0 表示错误不对应单独的一行。
     */
    [[nodiscard]] std::size_t lineNumber() const noexcept;

private:
    std::filesystem::path mFilePath;
    std::size_t mLineNumber{0U};
};

/**
 * @brief 读取单方向量化所需的 `key = value` 配置。
 *
 * 必填 key 为 `format`、`block_size`、`scale_mode` 和 `rounding`；
 * `stochastic_seed` 可选，省略时为 0。文件可额外包含完整 app 配置所需的
 * `output_type` 与 `target_gpu`，但本函数只返回量化阶段的配置。
 *
 * @param config_path 输入配置文件路径。
 * @return 已完成语法和量化语义校验的 QuantizationConfig。
 * @throws ConfigError 文件无法打开，或配置内容不符合项目约定时抛出。
 */
[[nodiscard]] QuantizationConfig load_quantization_config(
    const std::filesystem::path& config_path);

/**
 * @brief 读取单方向反量化所需的 `key = value` 配置。
 *
 * 必填 key 为 `output_type`。文件可额外包含量化或报告字段，但本函数只返回
 * 反量化输出类型，因此反量化工具不需要伪造 format、scale 或 target GPU。
 *
 * @param config_path 输入配置文件路径。
 * @return 已完成语法和反量化语义校验的 DequantizationConfig。
 * @throws ConfigError 文件无法打开，或配置内容不符合项目约定时抛出。
 */
[[nodiscard]] DequantizationConfig load_dequantization_config(
    const std::filesystem::path& config_path);

/**
 * @brief 读取 `apps/main.cpp` 完整流程所需的 `key = value` 配置。
 *
 * 必填 key 为 `format`、`block_size`、`scale_mode`、`rounding`、`output_type`
 * 和 `target_gpu`；`stochastic_seed` 可选。这个聚合对象只应用于 app 编排，
 * 后续必须把其子配置分别传给量化、反量化和日志模块。
 *
 * @param config_path 输入配置文件路径。
 * @return 已完成完整流程语义校验的 AppConfig。
 * @throws ConfigError 文件无法打开，或配置内容不符合项目约定时抛出。
 */
[[nodiscard]] AppConfig load_app_config(const std::filesystem::path& config_path);

}  // namespace quant_dequant
