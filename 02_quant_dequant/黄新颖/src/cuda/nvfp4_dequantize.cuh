#pragma once

#include <cuda_runtime_api.h>

namespace quant_dequant::pipeline {
struct DeviceDequantizationOutput;
struct DeviceQuantizedTensor;
}  // namespace quant_dequant::pipeline

namespace quant_dequant::cuda {

/**
 * @brief 启动 NVFP4 E2M1/E4M3/FP32 的 CUDA 反量化后端。
 *
 * 一个 CUDA 线程独占读取一个 packed payload byte，解包其中低/高两个 E2M1
 * nibble，并分别按两个元素的 row-major 坐标读取 E4M3 local scale。该映射避免
 * 相邻线程竞争同一个 byte，且能正确处理奇数列矩阵中跨行的 packed byte。
 *
 * @param input 已完成 H2D 的 NVFP4 packed payload、E4M3 local scale 与单元素
 * FP32 global scale，只读。
 * @param output 已分配的 row-major device FP32 输出，kernel 写入每个逻辑元素。
 * @param stream 与本次 H2D、CUDA Event 和 D2H 共用的非拥有 stream。
 * @throws CudaPipelineError 描述、device buffer 或 CUDA runtime/kernel launch
 * 不合法时抛出。
 */
void launch_nvfp4_dequantize(
    const pipeline::DeviceQuantizedTensor& input,
    pipeline::DeviceDequantizationOutput& output,
    cudaStream_t stream);

}  // namespace quant_dequant::cuda
