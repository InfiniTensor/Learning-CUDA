#pragma once

#include "quant_dequant/quantize.hpp"

namespace quant_dequant::pipeline_detail {

/**
 * @brief 执行 MXFP8 CUDA 量化路径的内部入口。
 *
 * 该入口将 host 输入传输为 `DeviceQuantizationInput`，分配
 * `DeviceQuantizedTensor`，按 tensor/block scale 调用 MXFP8 E8M0/E4M3 专用
 * kernel，再 D2H 为统一 host 结果。此声明不属于公共 API；调用方必须使用
 * `quantize_cuda()` 完成格式分派。
 *
 * @param input 已由公共入口验证的 host 输入。
 * @param config 已由公共入口验证的 MXFP8 量化配置。
 * @return 已完成 D2H 的 host 侧量化张量。
 */
[[nodiscard]] QuantizedTensor quantize_mxfp8_cuda(
    const HostTensor& input,
    const QuantizationConfig& config);

/** @brief 执行 MXFP8 CUDA 量化并保留同一 stream 上的 kernel Event 时间。 */
[[nodiscard]] ProfiledQuantizationResult quantize_mxfp8_cuda_profiled(
    const HostTensor& input,
    const QuantizationConfig& config);

/**
 * @brief 执行 NVFP4 CUDA 量化路径的内部入口。
 *
 * 该入口会处理 FP4 nibble packed store、E4M3 local scale、FP32 global scale
 * 与 NVFP4 专用量化 kernel。调用方必须使用 `quantize_cuda()` 完成格式分派。
 *
 * @param input 已由公共入口验证的 host 输入。
 * @param config 已由公共入口验证的 NVFP4 量化配置。
 * @return 已完成 D2H 的 host 侧 NVFP4 量化张量。
 */
[[nodiscard]] QuantizedTensor quantize_nvfp4_cuda(
    const HostTensor& input,
    const QuantizationConfig& config);

/** @brief 执行 NVFP4 CUDA 量化并保留同一 stream 上的 kernel Event 时间。 */
[[nodiscard]] ProfiledQuantizationResult quantize_nvfp4_cuda_profiled(
    const HostTensor& input,
    const QuantizationConfig& config);

/**
 * @brief 执行 MXFP8 CUDA 反量化路径的内部入口。
 *
 * 该入口传输 MXFP8 payload 与 E8M0 scale，调用格式专用反量化 kernel，并回传
 * 统一保存为 FP32 的 `HostTensor::values`。输出描述中的 dtype 设为请求的
 * FP16、BF16 或 FP32，实际物理窄化留给 QDTENSOR I/O 边界。
 *
 * @param input 已由公共入口验证的 MXFP8 量化张量。
 * @param config 已由公共入口验证的反量化配置。
 * @return 已完成 D2H 的 host FP32 张量。
 */
[[nodiscard]] HostTensor dequantize_mxfp8_cuda(
    const QuantizedTensor& input,
    const DequantizationConfig& config);

/** @brief 执行 MXFP8 CUDA 反量化并保留同一 stream 上的 kernel Event 时间。 */
[[nodiscard]] ProfiledDequantizationResult dequantize_mxfp8_cuda_profiled(
    const QuantizedTensor& input,
    const DequantizationConfig& config);

/**
 * @brief 执行 NVFP4 CUDA 反量化路径的内部入口。
 *
 * 该入口传输 packed E2M1 payload、E4M3 local scale 和单个 FP32 global scale；
 * 格式专用 kernel 以一个线程处理一个物理 byte，分别解码两个 nibble 并为其恢复
 * 各自的 rowwise local-scale 下标，随后回传统一保存为 FP32 的 `HostTensor::values`。
 *
 * @param input 已由公共入口验证的 NVFP4 量化张量。
 * @param config 已由公共入口验证的反量化配置。
 * @return 已完成 D2H 的 host FP32 张量。
 */
[[nodiscard]] HostTensor dequantize_nvfp4_cuda(
    const QuantizedTensor& input,
    const DequantizationConfig& config);

/** @brief 执行 NVFP4 CUDA 反量化并保留同一 stream 上的 kernel Event 时间。 */
[[nodiscard]] ProfiledDequantizationResult dequantize_nvfp4_cuda_profiled(
    const QuantizedTensor& input,
    const DequantizationConfig& config);

}  // namespace quant_dequant::pipeline_detail
