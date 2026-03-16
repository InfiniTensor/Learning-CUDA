#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>

template <typename T>
__global__ void dequantize_nf4_kernel(
    const uint8_t* __restrict__ packed_weights,
    const uint8_t* __restrict__ absmax_q,
    const uint16_t* __restrict__ absmax2,
    const uint16_t* __restrict__ code2,
    float offset,
    T* __restrict__ output,
    int64_t total_elements,
    int blocksize
);

template <typename T>
void launch_dequantize_nf4(
    const uint8_t* d_packed_weights,
    const uint8_t* d_absmax_q,
    const uint16_t* d_absmax2,
    const uint16_t* d_code2,
    float offset,
    T* d_output,
    int64_t total_elements,
    int blocksize,
    cudaStream_t stream = nullptr
);