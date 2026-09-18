#pragma once

#include <cuda_runtime_api.h>

namespace quant_dequant::pipeline {
struct DeviceDequantizationOutput;
struct DeviceQuantizedTensor;
}  // namespace quant_dequant::pipeline

namespace quant_dequant::cuda {

/**
 * @brief 启动 MXFP8 E4M3/E8M0 的 CUDA 反量化后端。
 *
 * 每个线程解码一个 row-major E4M3 payload。tensor-scale 模式固定读取
 * `local_scales[0]`；block-scale 模式则由元素的 `(row, column)` 推导出其
 * rowwise 32 元素 block 的 E8M0 scale 下标。两种模式的元素级解码公式相同，
 * 因此可以安全共享同一 kernel。
 *
 * @param input 已完成 H2D 的 MXFP8 device payload 与 E8M0 local scale，只读。
 * @param output 已分配的 row-major device FP32 输出，kernel 写入每个元素。
 * @param stream 与本次 H2D、CUDA Event 和 D2H 共用的非拥有 stream。
 * @throws CudaPipelineError 描述、device buffer 或 CUDA runtime/kernel launch
 * 不合法时抛出。
 */
void launch_mxfp8_dequantize(
    const pipeline::DeviceQuantizedTensor& input,
    pipeline::DeviceDequantizationOutput& output,
    cudaStream_t stream);

}  // namespace quant_dequant::cuda
