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
 * @brief 返回 NVFP4 全局 amax 第一阶段规约应使用的 persistent CTA 数量。
 *
 * NVFP4 即使只支持 `scale_mode=block`，仍须先对整张张量规约 amax 以构造
 * FP32 `global_scale`。该数量使用 `SM 数 × 4` 启发式并受当前 device 的
 * `maxGridSize[0]` 限制；pipeline 依此分配临时 partial 工作区。
 *
 * @return 大于 0 的第一阶段 CTA 数量。
 * @throws CudaPipelineError 查询 CUDA device 或其属性失败时抛出。
 */
[[nodiscard]] std::size_t nvfp4_tensor_partial_count();

/**
 * @brief 启动严格 NVFP4 block-scale 量化的 CUDA 后端。
 *
 * 路径依次执行：全局两阶段 amax reduction 并写 FP32 global scale、以
 * `thread_block_tile<16>` 计算每个 rowwise 16 元素 block 的 E4M3 local scale、
 * 再以 `thread_block_tile<32>` 让相邻 lane 编码两个 E2M1 code 并由偶数 lane
 * 独占写一个 packed payload byte。后两阶段分离，避免相邻元素跨行时两个
 * quantization block 对同一 byte 产生写竞争。
 *
 * @param input 已完成 H2D 的 row-major FP32 device 输入，只读。
 * @param config 已验证的 NVFP4 block-scale 量化配置。
 * @param output 已按 NVFP4 描述分配的 device 输出；kernel 写入 payload、
 * local_scales 和唯一的 global_scale。
 * @param workspace 生命周期覆盖全局 amax 规约、scale 与编码 kernel 的临时
 * partial amax / 非有限标记 device 数组。
 * @param stream 与本次 H2D、CUDA Event 和 D2H 共用的非拥有 stream。
 * @throws CudaPipelineError CUDA runtime 调用、launch 参数或 kernel 发射失败时抛出。
 */
void launch_nvfp4_block_quantize(
    const pipeline::DeviceQuantizationInput& input,
    const QuantizationConfig& config,
    pipeline::DeviceQuantizedTensor& output,
    pipeline::DeviceTensorQuantizationWorkspace& workspace,
    cudaStream_t stream);

}  // namespace quant_dequant::cuda
