#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace nvfp4 {

constexpr int kBlockSize = 16;   // 16 个值共享一个 E4M3 block scale。
constexpr int kWarpValues = 64;  // 一个 warp 处理 64 个输入值。

inline int padded_elements(int n) {
    return ((n + kWarpValues - 1) / kWarpValues) * kWarpValues;
}

inline int packed_bytes(int n_padded) {
    return n_padded / 2;
}

inline int block_scale_count(int n_padded) {
    return n_padded / kBlockSize;
}

// 反量化：E2M1 × E4M3 block scale × FP32 global scale。
// 输入需 padding 到 64 的整数倍；workspace 可复用。
// FP32 -> NVFP4。
cudaError_t quantize(
    const float* d_input,
    int n_padded,
    uint8_t* d_packed,
    uint8_t* d_block_scales,
    float* d_global_scale,
    cudaStream_t stream = 0,
    uint32_t* d_global_amax_bits_workspace = nullptr);

// FP16 -> NVFP4，输入为原始 FP16 bit。
cudaError_t quantize_fp16(
    const uint16_t* d_input_fp16,
    int n_padded,
    uint8_t* d_packed,
    uint8_t* d_block_scales,
    float* d_global_scale,
    cudaStream_t stream = 0,
    uint32_t* d_global_amax_bits_workspace = nullptr);

// NVFP4 -> FP32。
cudaError_t dequantize(
    const uint8_t* d_packed,
    const uint8_t* d_block_scales,
    const float* d_global_scale,
    int n_padded,
    float* d_output,
    cudaStream_t stream = 0);

// NVFP4 -> FP16，输出为原始 FP16 bit。
cudaError_t dequantize_fp16(
    const uint8_t* d_packed,
    const uint8_t* d_block_scales,
    const float* d_global_scale,
    int n_padded,
    uint16_t* d_output_fp16,
    cudaStream_t stream = 0);

// NVFP4 -> BF16，输出为原始 BF16 bit。
cudaError_t dequantize_bf16(
    const uint8_t* d_packed,
    const uint8_t* d_block_scales,
    const float* d_global_scale,
    int n_padded,
    uint16_t* d_output_bf16,
    cudaStream_t stream = 0);

}  // namespace nvfp4
