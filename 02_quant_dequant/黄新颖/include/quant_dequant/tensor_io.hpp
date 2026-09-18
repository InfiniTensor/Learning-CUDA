#pragma once

#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

#include "quant_dequant/types.hpp"

namespace quant_dequant {

/**
 * @brief 表示 QDTENSOR 文件读取或写入过程中的错误。
 */
class TensorIoError final : public std::runtime_error {
public:
    /**
     * @brief 构造一个张量 I/O 错误。
     *
     * @param file_path 发生错误的张量文件路径。
     * @param detail 面向用户的错误说明。
     */
    TensorIoError(std::filesystem::path file_path, std::string detail);

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
 * @brief 保存 host 侧 FP32 数据及其 QDTENSOR 序列化描述的张量。
 *
 * 这不是 device tensor 或通用数值抽象；它只服务于 QDTENSOR I/O 与量化
 * pipeline 的 host 端边界。`values` 始终是 row-major 的 FP32 数据；
 * `desc.dtype` 则表示其对应 QDTENSOR payload 的物理 dtype：读取输入时它是
 * 文件原始 dtype，写反量化输出时它是将要写出的 FP16、BF16 或 FP32 dtype。
 */
struct HostTensor {
    /** row-major 形状与 QDTENSOR payload 的物理 dtype。 */
    TensorDesc desc{};

    /** 按 row-major 顺序保存的 host FP32 元素。 */
    std::vector<float> values{};
};

/**
 * @brief 读取一个输入用途的 QDTENSOR 文件，并统一转换为 host FP32。
 *
 * 文件必须满足 `tensor_role = input`，且 dtype 只能是 FP16 或 FP32。
 * 函数会验证 magic、版本、字节序、header 字段、payload 长度和文件总长度。
 *
 * @param input_path 输入 QDTENSOR 文件路径。
 * @return `desc` 保留输入文件物理 dtype、元素已转换为 FP32 的 host 张量。
 * @throws TensorIoError 文件无法读取或不符合 QDTENSOR v1 格式时抛出。
 */
[[nodiscard]] HostTensor read_input_tensor(const std::filesystem::path& input_path);

/**
 * @brief 读取一个反量化输出用途的 QDTENSOR 文件，并统一转换为 host FP32。
 *
 * 文件必须满足 `tensor_role = dequantized_output`，dtype 可以是 FP16、BF16 或
 * FP32。该接口既可用于读取既有反量化结果，也可使完整 app 在写出文件后重新
 * 读取实际物理 payload，再据此统计包含 FP16/BF16 窄化误差的最终误差。
 *
 * @param output_path 反量化输出 QDTENSOR 文件路径。
 * @return `desc` 保留输出文件物理 dtype、元素已扩展为 FP32 的 host 张量。
 * @throws TensorIoError 文件无法读取或不符合 QDTENSOR v1 输出格式时抛出。
 */
[[nodiscard]] HostTensor read_dequantized_tensor(
    const std::filesystem::path& output_path);

/**
 * @brief 将 FP32 host 数据写为反量化输出用途的 QDTENSOR 文件。
 *
 * `tensor.desc.dtype` 决定文件 payload 的物理类型，可以是 FP16、BF16
 * 或 FP32；`tensor.values` 始终视为 FP32 输入，并按目标 dtype 转换。采用
 * `HostTensor` 使该接口与 `write_quantized_tensor()` 一样，以一个完整的
 * 自描述 host 对象作为写入单位。
 *
 * @param output_path 要创建或覆盖的输出文件路径。
 * @param tensor 要写出的 host 张量；`values.size()` 必须等于 `desc` 的元素数。
 * @throws TensorIoError 路径无法写入、描述非法或张量长度不匹配时抛出。
 */
void write_dequantized_tensor(const std::filesystem::path& output_path,
                              const HostTensor& tensor);
}  // namespace quant_dequant
