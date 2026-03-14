#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>

// 声明外部定义的 Kernel
__global__ void dequantize_nf4_kernel(
    const uint8_t* __restrict__ packed_weights, 
    const uint8_t* __restrict__ absmax_q, 
    const uint16_t* __restrict__ absmax2, 
    const uint16_t* __restrict__ code2, 
    float offset, 
    __nv_bfloat16* __restrict__ output,  // 改为 __nv_bfloat16* 更符合常规语义
    int64_t total_elements, 
    int blocksize
);

// Host 端包装函数
void launch_dequantize_nf4(
    const uint8_t* d_packed_weights, 
    const uint8_t* d_absmax_q, 
    const uint16_t* d_absmax2, 
    const uint16_t* d_code2, 
    float offset, 
    __nv_bfloat16* d_output, 
    int64_t total_elements, 
    int blocksize,
    cudaStream_t stream = nullptr
);
