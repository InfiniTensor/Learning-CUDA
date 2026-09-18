#pragma once

#include <filesystem>
#include <stdexcept>
#include <string>

#include "quant_dequant/quantized_tensor.hpp"

namespace quant_dequant {

/**
 * @brief 表示 QDWGT 量化权重文件读写过程中的错误。
 */
class QuantizedIoError final : public std::runtime_error {
public:
    /**
     * @brief 构造一个量化权重 I/O 错误。
     *
     * @param file_path 发生错误的 QDWGT 文件路径。
     * @param detail 面向用户的错误说明。
     */
    QuantizedIoError(std::filesystem::path file_path, std::string detail);

    /**
     * @brief 返回发生错误的文件路径。
     *
     * @return 不拥有资源的常量路径引用。
     */
    [[nodiscard]] const std::filesystem::path& filePath() const noexcept;

private:
    std::filesystem::path mFilePath;
};

/**
 * @brief 将完整的 host 侧量化结果写为 QDWGT v1 文件。
 *
 * 写入器使用固定的 128 字节 little-endian header，紧跟 packed payload，并将
 * local scale section 向上对齐到 8 字节。它不会直接写入 C++ struct，因此不会
 * 受编译器 padding 或 host ABI 影响。
 *
 * MXFP8 在文件 header 的 `global_scale` 槽写固定 1.0F；NVFP4 则写入
 * `tensor.global_scale`。这与 `QuantizedTensor` 的内存语义保持分离。
 *
 * @param output_path 要创建或覆盖的 QDWGT 文件路径。
 * @param tensor 已完成量化且必须满足 `QuantizedTensor::isConsistent()` 的结果。
 * @throws QuantizedIoError 输出路径无法写入或量化结果不自洽时抛出。
 */
void write_quantized_tensor(const std::filesystem::path& output_path,
                            const QuantizedTensor& tensor);

/**
 * @brief 读取一个 QDWGT v1 文件，并构造统一的 host 侧量化结果。
 *
 * 读取器按固定 offset 逐字段解析 little-endian header，不依赖本机 struct
 * 的大小或任何当前配置文件。读取前会验证格式组合、shape、payload/scale
 * 长度、section offset、NVFP4 flags、global scale 和文件总长度。
 *
 * @param input_path 待读取的 QDWGT 文件路径。
 * @return 通过所有 QDWGT v1 校验的量化结果。
 * @throws QuantizedIoError 文件无法读取或不符合 QDWGT v1 格式时抛出。
 */
[[nodiscard]] QuantizedTensor read_quantized_tensor(
    const std::filesystem::path& input_path);

}  // namespace quant_dequant
