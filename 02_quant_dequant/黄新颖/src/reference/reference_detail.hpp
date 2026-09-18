#pragma once

#include "quant_dequant/quantize.hpp"

namespace quant_dequant::reference_detail {

/**
 * @brief 使用 MXFP8-E4M3/E8M0 规则量化一个 host FP32 张量。
 *
 * block 模式下，每行独立以 32 元素切分；尾 block 只扫描真实元素，不在
 * payload 中写入逻辑补零。tensor 模式下，整张矩阵共享一个 E8M0 scale。
 *
 * 量化结果的 payload 是一个元素一个 E4M3 byte，local_scales 则保存一个
 * tensor scale 或按 row-major block 顺序保存多个 E8M0 code。
 *
 * @param input 已通过 dispatcher 形状、dtype 和 values 长度校验的 host 张量。
 * @param config 已通过 dispatcher 校验、且 format 为 kMxfp8 的量化配置。
 * @return 与 input 具有相同逻辑形状的完整 MXFP8 QuantizedTensor。
 * @throws ReferenceError 输入包含 NaN/Inf、元数据无法构造或内部结果不自洽时抛出。
 */
[[nodiscard]] QuantizedTensor quantize_mxfp8_reference(
    const HostTensor& input,
    const QuantizationConfig& config);

/**
 * @brief 使用严格 NVFP4-E2M1/E4M3/FP32 规则量化一个 host FP32 张量。
 *
 * NVFP4 始终使用 rowwise 16 元素 block：先以整张张量 amax 构造一个 FP32
 * decode global scale，再以每个 block 的 amax 构造一个 E4M3 local scale，最后
 * 将两个 E2M1 元素按低/高 nibble 打包为一个 payload byte。
 *
 * @param input 已通过 dispatcher 形状、dtype 和 values 长度校验的 host 张量。
 * @param config 已通过 dispatcher 校验、且 format 为 kNvfp4 的量化配置。
 * @return 与 input 具有相同逻辑形状的完整 NVFP4 QuantizedTensor。
 * @throws ReferenceError 输入包含 NaN/Inf、元数据无法构造或内部结果不自洽时抛出。
 */
[[nodiscard]] QuantizedTensor quantize_nvfp4_reference(
    const HostTensor& input,
    const QuantizationConfig& config);

/**
 * @brief 使用 MXFP8 的 CPU reference 反量化一个低精度张量。
 *
 * 此接口仅供 `reference_dispatch.cpp` 调用。它解码 E8M0 local scale 与 E4M3
 * payload，生成 FP32 `HostTensor::values`。
 *
 * @param input 已通过 dispatcher 校验、且 `format == kMxfp8` 的量化结果。
 * @param config 已通过 dispatcher 校验的反量化配置。
 * @return 带有目标输出 dtype 描述的 host FP32 张量。
 * @throws ReferenceError 输入元数据、payload/scale 数量或输出配置不合法时抛出。
 */
[[nodiscard]] HostTensor dequantize_mxfp8_reference(
    const QuantizedTensor& input,
    const DequantizationConfig& config);

/**
 * @brief 使用严格 NVFP4-E2M1/E4M3/FP32 规则反量化一个 host 量化张量。
 *
 * 该实现按物理 packed payload byte 遍历：低四位解码一个偶数线性元素，高四位
 * 解码下一个奇数线性元素。即使矩阵列数为奇数、一个 byte 跨越两行，也会分别由
 * 两个元素自己的 `(row, column)` 推导 rowwise local-scale 下标。
 *
 * @param input 已通过 dispatcher 校验、且 `format == kNvfp4` 的量化结果。
 * @param config 已通过 dispatcher 校验的反量化配置。
 * @return 带有目标输出 dtype 描述的 host FP32 张量。
 * @throws ReferenceError 输入元数据、global scale、输出配置不合法时抛出。
 */
[[nodiscard]] HostTensor dequantize_nvfp4_reference(
    const QuantizedTensor& input,
    const DequantizationConfig& config);

}  // namespace quant_dequant::reference_detail
