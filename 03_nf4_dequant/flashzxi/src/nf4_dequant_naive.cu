//
// Created by core_dump on 2026/2/25.
//
#include <cuda_fp16.h>
#include <string>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "quant_state.h"
#include "common.cuh"
#include "nf4_dequant.h"

template <typename FP_T>
__global__ void dequant_absmax_kernel(const uint8_t* __restrict__ absmax_q,
                                      const FP_T* __restrict__ absmax2,
                                      const FP_T* __restrict__ code2, // 256
                                      int num_blocks,
                                      int group_size,   // blocks per group
                                      float offset,
                                      float* __restrict__ absmax_out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_blocks) return;

    int group = i / group_size;
    float s2 = f162float(absmax2[group]);
    float c  = f162float(code2[absmax_q[i]]);
    absmax_out[i] = c * s2 + offset;
}

// 每个block 128个线程，每个线程负责2个
template <typename OUT_T>
__global__ void dequant_nf4_kernel(const uint8_t* __restrict__ packed,
                                   const float* __restrict__ absmax,
                                   int num_elements,
                                   int block_size,
                                   OUT_T* __restrict__ out) {
    float kNF4[16] = {
        -1.0000000f, -0.6961928f, -0.5250731f, -0.3949175f,
        -0.2844414f, -0.1847734f, -0.0910500f,  0.0000000f,
         0.0795803f,  0.1609302f,  0.2461123f,  0.3379152f,
         0.4407098f,  0.5626170f,  0.7229568f,  1.0000000f
    };
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    int elem0 = t * 2;
    if (elem0 >= num_elements) return;

    uint8_t byte = packed[t];
    int lo = byte & 0x0F;
    int hi = byte >> 4;

    float s0 = absmax[elem0 / block_size];
    float v0 = s0 * kNF4[hi];
    if constexpr (std::is_same_v<OUT_T, __half>) {
        out[elem0] = __float2half(v0);
    } else {
        out[elem0] = __float2bfloat16(v0);
    }

    int elem1 = elem0 + 1;
    if (elem1 < num_elements) {
        float s1 = absmax[elem1 / block_size];
        float v1 = s1 * kNF4[lo];
        if constexpr (std::is_same_v<OUT_T, __half>) {
            out[elem1] = __float2half(v1);
        } else {
            out[elem1] = __float2bfloat16(v1);
        }
    }
}

void nf4_dequant_naive(const QuantState& quant_state, __half* output) {
    // 解码scale
    uint8_t* scale_q_s;
    __half* code2_s;
    __half* absmax2_s;

    CUDA_CHECK(cudaMalloc(&scale_q_s, quant_state.num_blocks));
    CUDA_CHECK(cudaMalloc(&code2_s, 256 * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&absmax2_s, quant_state.num_groups * sizeof(__half)));

    CUDA_CHECK(cudaMemcpy(scale_q_s, quant_state.absmax_q, quant_state.absmax_q_len_in_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(code2_s, quant_state.code2, 256 * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(absmax2_s, quant_state.absmax2, quant_state.num_groups * sizeof(__half), cudaMemcpyHostToDevice));

    dim3 dequant_scale_block_dim(128);
    dim3 dequant_scale_grid_dim((quant_state.num_blocks + 128 - 1) / 128);
    // 解码权重
    float* absmax = nullptr;
    size_t absmax_bytes = sizeof(float) * quant_state.num_blocks;
    CUDA_CHECK(cudaMalloc(&absmax, absmax_bytes));

    float* absmax_h = new float[quant_state.num_blocks];

    if (quant_state.compute_type == "bf16") {
        dequant_absmax_kernel<__nv_bfloat16><<<dequant_scale_grid_dim, dequant_scale_block_dim>>>(
            scale_q_s, (__nv_bfloat16*) absmax2_s,
            (__nv_bfloat16*) code2_s, quant_state.num_blocks, quant_state.group_size, quant_state.offset, absmax
        );
    } else if (quant_state.compute_type == "fp16") {
        dequant_absmax_kernel<__half><<<dequant_scale_grid_dim, dequant_scale_block_dim>>>(
            scale_q_s, (__half*) absmax2_s,
            (__half*) code2_s, quant_state.num_blocks, quant_state.group_size, quant_state.offset,absmax
        );
    } else {
        std::cerr << "Type Not Supported, only support bf16 | fp16" << std::endl;
        exit(-1);
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaMemcpy(absmax_h, absmax, quant_state.num_blocks * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < quant_state.num_blocks; i++) {
        std::cout << absmax_h[i] << " ";
    }
    std::cout << std::endl;

    CUDA_CHECK(cudaFree(scale_q_s));
    CUDA_CHECK(cudaFree(code2_s));
    CUDA_CHECK(cudaFree(absmax2_s));

    uint8_t* packed_weights_s;
    // output
    __half* unpacked_weights_s;

    CUDA_CHECK(cudaMalloc(&packed_weights_s, quant_state.packed_weights_len_in_bytes));
    CUDA_CHECK(cudaMalloc(&unpacked_weights_s, quant_state.num_elements * sizeof(__half)));

    CUDA_CHECK(cudaMemcpy(packed_weights_s, quant_state.packed_weights, quant_state.packed_weights_len_in_bytes, cudaMemcpyHostToDevice))

    dim3 dequant_weights_grid_dim((quant_state.packed_weights_len_in_bytes + dequant_scale_block_dim.x - 1) / dequant_scale_block_dim.x);

    if (quant_state.compute_type == "bf16") {
        dequant_nf4_kernel<__nv_bfloat16><<<dequant_weights_grid_dim, dequant_scale_block_dim>>> (
            packed_weights_s, absmax, quant_state.num_elements,
            quant_state.block_size, (__nv_bfloat16*) unpacked_weights_s
        );
    } else if (quant_state.compute_type == "fp16") {
        dequant_nf4_kernel<__half><<<dequant_weights_grid_dim, dequant_scale_block_dim>>> (
            packed_weights_s, absmax, quant_state.num_elements,
            quant_state.block_size, (__half*) unpacked_weights_s
        );
    } else {
        std::cerr << "Type Not Supported, only support bf16 | fp16" << std::endl;
        exit(-1);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(output, unpacked_weights_s, quant_state.num_elements * sizeof(__half), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(packed_weights_s));
    CUDA_CHECK(cudaFree(absmax));
    CUDA_CHECK(cudaFree(unpacked_weights_s));
}