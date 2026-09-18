#pragma once

#include <cstddef>

#include <cuda_runtime_api.h>

#include "quant_dequant/types.hpp"

namespace quant_dequant::pipeline {
struct DeviceQuantizationInput;
struct DeviceQuantizedTensor;
struct DeviceTensorQuantizationWorkspace;
}  // namespace quant_dequant::pipeline

namespace quant_dequant::cuda {

/**
 * @brief 启动 MXFP8 block-scale 量化的 CUDA 后端。
 *
 * 一个 warp 处理一个 rowwise 32 元素逻辑量化 block：先在 warp 内规约 `amax`，
 * 由 lane 0 写 E8M0 local scale，再使用该 scale 编码各 lane 对应的 E4M3
 * payload。固定数量 CTA 以 grid-stride 持续处理后续量化 block。它与
 * tensor-scale 路径不能共用同一个 launcher，因为 block-scale 不需要跨 CTA 的
 * 全局归约。
 *
 * @param input 已完成 H2D 的 row-major FP32 device 输入，只读。
 * @param config 已验证的 MXFP8 block-scale 量化配置。
 * @param output 已按 MXFP8 描述分配的 device 输出；kernel 写入 payload 和
 * local_scales。
 * @param stream 与本次 H2D、CUDA Event 和 D2H 共用的非拥有 stream。
 * @throws CudaPipelineError CUDA runtime 调用、kernel launch 或后续数值检查失败时抛出。
 */
void launch_mxfp8_block_quantize(
    const pipeline::DeviceQuantizationInput& input,
    const QuantizationConfig& config,
    pipeline::DeviceQuantizedTensor& output,
    cudaStream_t stream);

/**
 * @brief 启动 MXFP8 tensor-scale 量化的 CUDA 后端。
 *
 * 该路径以两个规约 kernel 求整张张量的全局 `amax`：第一阶段按固定数量 CTA
 * 写 partial 结果，第二阶段由一个 CTA 规约所有 partial 并直接写唯一的
 * `output.local_scales[0]` E8M0 scale；随后启动独立编码 kernel。普通 CUDA
 * kernel 不支持跨 CTA 同步，因此不能复用 block-scale 的单 kernel 流程。
 *
 * @param input 已完成 H2D 的 row-major FP32 device 输入，只读。
 * @param config 已验证的 MXFP8 tensor-scale 量化配置。
 * @param output 已按 MXFP8 描述分配的 device 输出；kernel 写入 payload 和唯一的
 * local scale。
 * @param workspace 生命周期覆盖所有规约与编码 kernel 的临时 partial amax /
 * 非有限标记 device 数组。
 * @param stream 与本次 H2D、CUDA Event 和 D2H 共用的非拥有 stream。
 * @throws CudaPipelineError CUDA runtime 调用、kernel launch 或后续数值检查失败时抛出。
 */
void launch_mxfp8_tensor_quantize(
    const pipeline::DeviceQuantizationInput& input,
    const QuantizationConfig& config,
    pipeline::DeviceQuantizedTensor& output,
    pipeline::DeviceTensorQuantizationWorkspace& workspace,
    cudaStream_t stream);

/**
 * @brief 返回当前 CUDA device 上 tensor-scale 第一阶段规约应使用的 CTA 数量。
 *
 * 数量使用与参考 V7 reduction 相同的 `SM 数 * 4` 持久化网格启发式，并受
 * `maxGridSize[0]` 约束。pipeline 据此分配 `DeviceTensorQuantizationWorkspace`，
 * launcher 再验证 workspace 长度与该数量一致。
 *
 * @return 大于 0 的第一阶段 CTA 数量。
 * @throws CudaPipelineError 查询当前 device 或其 grid 属性失败时抛出。
 */
[[nodiscard]] std::size_t mxfp8_tensor_partial_count();

}  // namespace quant_dequant::cuda
