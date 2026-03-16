//
// Created by flashzxi on 2/24/26.
//
#include <cuda_fp16.h>
#include <string>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "quant_state.h"
#include "nf4_dequant.h"
#include "common.cuh"

#define LDST32BITS(value) (reinterpret_cast<float*>(&(value))[0])
#define LDST64BITS(value) (reinterpret_cast<float2*>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4*>(&(value))[0])

// code2 为 256 * f16
// 每个线程load 2 个，需要128个线程， 故设置一个block 128个线程，每个线程处理N个计算
// 总计处理128 * N个数据, N 是2的幂 且不小于8
// 结尾不够需要padding
template<typename HFP_T, int N>
__global__ void dequant_nf4_scale_warp8_batchN_kernel(
        uint8_t* scale_q,
        HFP_T* code2,
        HFP_T* absmax2,
        int num_blocks,
        int group_size,
        float offset,
        float* output) {
    int lane_id = threadIdx.x;

    // load code2
    __shared__ float shm_code2_float[128];

    LDST32BITS(shm_code2_float[lane_id]) = LDST32BITS(code2[2 * lane_id]);
    HFP_T* shm_code2 = (HFP_T *) shm_code2_float;
    __syncthreads();

    // 一次处理8个数据
    constexpr int loop_times = N / 8;
    int g_scale_q_offset_base = blockIdx.x * 128 * N;

    alignas(16) uint8_t fragment[8];
    alignas(16) float cache_res[8];
#pragma unroll
    for (int i = 0; i < loop_times; ++i) {
        int scale_offset = g_scale_q_offset_base + i * 128 * 8 + lane_id * 8;
        HFP_T scale2 = absmax2[scale_offset / group_size];

        if (scale_offset + 7 < num_blocks) {
            LDST64BITS(fragment[0]) = LDST64BITS(*(scale_q + scale_offset));
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                cache_res[j] = f162float( shm_code2[fragment[j]] * scale2 ) + offset;
            }
            LDST128BITS(output[scale_offset]) = LDST128BITS(cache_res[0]);
            LDST128BITS(output[scale_offset + 4]) = LDST128BITS(cache_res[4]);
        } else if (scale_offset < num_blocks) {
            // 不够一组，退化为每个元素load
            int remains = num_blocks - scale_offset;
            for (int j = 0; j < remains; ++j) {
                fragment[j] = (scale_q + scale_offset)[j];
                cache_res[j] = f162float(shm_code2[fragment[j]] * scale2) + offset;
                output[scale_offset + j] = cache_res[j];
            }
        }
    }
}

// 一个block 128个线程
// 每个线程负责N个, 每个block 负责 128 * N 个数据的解码
template<typename HFP_T, int N>
__global__ void dequant_nf4_elements_warp8_batchN_kernel(uint8_t* packed_weights, float* absmax, int num_elements, int block_size, HFP_T* output) {
    float kNF4[16] = {
        -1.0000000f, -0.6961928f, -0.5250731f, -0.3949175f,
        -0.2844414f, -0.1847734f, -0.0910500f,  0.0000000f,
         0.0795803f,  0.1609302f,  0.2461123f,  0.3379152f,
         0.4407098f,  0.5626170f,  0.7229568f,  1.0000000f
    };
    uint8_t* packed_weights_end = packed_weights + (num_elements + 1) / 2;

    int bidx = blockIdx.x;
    int lane_id = threadIdx.x;

    int block_offset = bidx * 128 * N;

    // 每次处理8个，32bits
    alignas(16) uint8_t f_packed_weights[4];
    alignas(16) HFP_T cache_res[8];
    constexpr int loop_times = N / 8;
#pragma unroll
    for (int i = 0; i < loop_times; ++i) {
        int g_packed_weights_offset = block_offset + 8 * 128 * i + 8 * lane_id;
        float scale = absmax[g_packed_weights_offset / block_size];
        if (packed_weights + g_packed_weights_offset / 2 + 4 < packed_weights_end) {
            LDST32BITS(f_packed_weights[0]) = LDST32BITS(packed_weights[g_packed_weights_offset / 2]);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                uint8_t lower = f_packed_weights[j] & 0xF;
                uint8_t upper = f_packed_weights[j] >> 4;
                if constexpr (std::is_same_v<HFP_T, __half>) {
                    cache_res[2 * j] = __float2half(scale * kNF4[upper]);
                    cache_res[2 * j + 1] = __float2half(scale * kNF4[lower]);
                } else if constexpr (std::is_same_v<HFP_T, __nv_bfloat16>) {
                    cache_res[2 * j] = __float2bfloat16(scale * kNF4[upper]);
                    cache_res[2 * j + 1] = __float2bfloat16(scale * kNF4[lower]);
                }
            }
            LDST128BITS(output[g_packed_weights_offset]) = LDST128BITS(cache_res[0]);
        } else if (packed_weights + g_packed_weights_offset / 2 < packed_weights_end) {
            int remains = num_elements - g_packed_weights_offset;
            for (int j = 0; j < (remains + 1) / 2; ++j) {
                f_packed_weights[0] = packed_weights[g_packed_weights_offset / 2 + j];
                uint8_t lower = f_packed_weights[0] & 0xF;
                uint8_t upper = f_packed_weights[0] >> 4;
                if constexpr (std::is_same_v<HFP_T, __half>) {
                    cache_res[0] = __float2half(scale * kNF4[upper]);
                    cache_res[1] = __float2half(scale * kNF4[lower]);
                } else if constexpr (std::is_same_v<HFP_T, __nv_bfloat16>) {
                    cache_res[0] = __float2bfloat16(scale * kNF4[upper]);
                    cache_res[1] = __float2bfloat16(scale * kNF4[lower]);
                }
                if (g_packed_weights_offset + 2 * j >= num_elements) {
                    // 只需要写回第一个
                    output[g_packed_weights_offset + 2 * j] = cache_res[0];
                } else {
                    // 两个打包写回
                    LDST32BITS(output[g_packed_weights_offset + 2 * j]) = LDST32BITS(cache_res[0]);
                }
            }
        }
    }
}

// 一个block 128个线程
// 每个线程负责N个, 每个block 负责 128 * N 个数据的解码
template<typename HFP_T, int N>
__global__ void dequant_nf4_elements_one_phase_warp8_batchN_kernel(
        uint8_t* packed_weights,
        uint8_t* absmax_q,
        int num_elements,
        HFP_T* absmax2,
        HFP_T* code2,
        int block_size,
        int group_size,
        float offset,
        HFP_T* output) {
    constexpr float kNF4[16] = {
        -1.0000000f, -0.6961928f, -0.5250731f, -0.3949175f,
        -0.2844414f, -0.1847734f, -0.0910500f,  0.0000000f,
         0.0795803f,  0.1609302f,  0.2461123f,  0.3379152f,
         0.4407098f,  0.5626170f,  0.7229568f,  1.0000000f
    };
    uint8_t* packed_weights_end = packed_weights + (num_elements + 1) / 2;

    int bidx = blockIdx.x;
    int lane_id = threadIdx.x;

    // load code2 不用shared memory更快
//    __shared__ float shm_code2_float[128];

//    LDST32BITS(shm_code2_float[lane_id]) = LDST32BITS(code2[2 * lane_id]);
//    HFP_T* shm_code2 = (HFP_T *) shm_code2_float;
//    __syncthreads();

    int block_offset = bidx * 128 * N;

    // 每次处理8个，32bits
    alignas(16) uint8_t f_packed_weights[4];
    alignas(16) HFP_T cache_res[8];
    constexpr int loop_times = N / 8;
#pragma unroll
    for (int i = 0; i < loop_times; ++i) {
        int g_packed_weights_offset = block_offset + 8 * 128 * i + 8 * lane_id;
        int block_idx = g_packed_weights_offset / block_size;
        int group_idx = block_idx / group_size;

        HFP_T h2[2];
        uint8_t q = absmax_q[block_idx];
        LDST32BITS(h2[0]) = LDST32BITS(code2[(q >> 1) << 1]);        // 读 32-bit
        HFP_T h = (q & 1) ? h2[1] : h2[0];
        float scale = f162float(h * absmax2[group_idx]) + offset;
        if (packed_weights + g_packed_weights_offset / 2 + 4 < packed_weights_end) {
            LDST32BITS(f_packed_weights[0]) = LDST32BITS(packed_weights[g_packed_weights_offset / 2]);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                uint8_t lower = f_packed_weights[j] & 0xF;
                uint8_t upper = f_packed_weights[j] >> 4;
                if constexpr (std::is_same_v<HFP_T, __half>) {
                    cache_res[2 * j] = __float2half(scale * kNF4[upper]);
                    cache_res[2 * j + 1] = __float2half(scale * kNF4[lower]);
                } else if constexpr (std::is_same_v<HFP_T, __nv_bfloat16>) {
                    cache_res[2 * j] = __float2bfloat16(scale * kNF4[upper]);
                    cache_res[2 * j + 1] = __float2bfloat16(scale * kNF4[lower]);
                }
            }
            LDST128BITS(output[g_packed_weights_offset]) = LDST128BITS(cache_res[0]);
        } else if (packed_weights + g_packed_weights_offset / 2 < packed_weights_end) {
            int remains = num_elements - g_packed_weights_offset;
            for (int j = 0; j < (remains + 1) / 2; ++j) {
                f_packed_weights[0] = packed_weights[g_packed_weights_offset / 2 + j];
                uint8_t lower = f_packed_weights[0] & 0xF;
                uint8_t upper = f_packed_weights[0] >> 4;
                if constexpr (std::is_same_v<HFP_T, __half>) {
                    cache_res[0] = __float2half(scale * kNF4[upper]);
                    cache_res[1] = __float2half(scale * kNF4[lower]);
                } else if constexpr (std::is_same_v<HFP_T, __nv_bfloat16>) {
                    cache_res[0] = __float2bfloat16(scale * kNF4[upper]);
                    cache_res[1] = __float2bfloat16(scale * kNF4[lower]);
                }
                if (g_packed_weights_offset + 2 * j >= num_elements) {
                    // 只需要写回第一个
                    output[g_packed_weights_offset + 2 * j] = cache_res[0];
                } else {
                    // 两个打包写回
                    LDST32BITS(output[g_packed_weights_offset + 2 * j]) = LDST32BITS(cache_res[0]);
                }
            }
        }
    }
}

void nf4_dequant_warp8_batch32_two_phase(const QuantState& quant_state, __half* output) {
    constexpr int PROCESS_SIZE = 32;

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

    Tracer tracer;
    tracer.memcpy_accumulate(quant_state.num_blocks)
        .memcpy_accumulate(256 * sizeof(__half))
        .memcpy_accumulate(quant_state.num_groups * sizeof(__half))
        .memcpy_accumulate(quant_state.packed_weights_len_in_bytes)
        .memcpy_accumulate(quant_state.num_elements * sizeof(__half));

    dim3 dequant_scale_block_dim(128);
    dim3 dequant_scale_grid_dim((quant_state.num_blocks + dequant_scale_block_dim.x * PROCESS_SIZE - 1) / (dequant_scale_block_dim.x * PROCESS_SIZE));
    // 解码权重
    float* absmax = nullptr;
    size_t absmax_bytes = sizeof(float) * quant_state.num_blocks;
    CUDA_CHECK(cudaMalloc(&absmax, absmax_bytes));

    float* absmax_h = new float[quant_state.num_blocks];

    tracer.start();
    if (quant_state.compute_type == "bf16") {
        dequant_nf4_scale_warp8_batchN_kernel<__nv_bfloat16, PROCESS_SIZE><<<dequant_scale_grid_dim, dequant_scale_block_dim>>>(
            scale_q_s, (__nv_bfloat16*) code2_s,
            (__nv_bfloat16*) absmax2_s, quant_state.num_blocks, quant_state.group_size, quant_state.offset, absmax
        );
    } else if (quant_state.compute_type == "fp16") {
        dequant_nf4_scale_warp8_batchN_kernel<__half, PROCESS_SIZE><<<dequant_scale_grid_dim, dequant_scale_block_dim>>>(
            scale_q_s, (__half*) code2_s,
            (__half*) absmax2_s, quant_state.num_blocks, quant_state.group_size, quant_state.offset,absmax
        );
    } else {
        std::cerr << "Type Not Supported, only support bf16 | fp16" << std::endl;
        exit(-1);
    }
    tracer.stop();

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaMemcpy(absmax_h, absmax, quant_state.num_blocks * sizeof(float), cudaMemcpyDeviceToHost);

    CUDA_CHECK(cudaFree(scale_q_s));
    CUDA_CHECK(cudaFree(code2_s));
    CUDA_CHECK(cudaFree(absmax2_s));

    uint8_t* packed_weights_s;
    // output
    __half* unpacked_weights_s;

    CUDA_CHECK(cudaMalloc(&packed_weights_s, quant_state.packed_weights_len_in_bytes));
    CUDA_CHECK(cudaMalloc(&unpacked_weights_s, quant_state.num_elements * sizeof(__half)));
    CUDA_CHECK(cudaMemcpy(packed_weights_s, quant_state.packed_weights, quant_state.packed_weights_len_in_bytes, cudaMemcpyHostToDevice))

    dim3 dequant_weights_grid_dim((quant_state.num_elements + dequant_scale_block_dim.x * PROCESS_SIZE - 1) / (dequant_scale_block_dim.x * PROCESS_SIZE));

    tracer.start();
    if (quant_state.compute_type == "bf16") {
        dequant_nf4_elements_warp8_batchN_kernel<__nv_bfloat16, PROCESS_SIZE><<<dequant_weights_grid_dim, dequant_scale_block_dim>>> (
            packed_weights_s, absmax, quant_state.num_elements,
            quant_state.block_size, (__nv_bfloat16*) unpacked_weights_s
        );
    } else if (quant_state.compute_type == "fp16") {
        dequant_nf4_elements_warp8_batchN_kernel<__half, PROCESS_SIZE><<<dequant_weights_grid_dim, dequant_scale_block_dim>>> (
            packed_weights_s, absmax, quant_state.num_elements,
            quant_state.block_size, (__half*) unpacked_weights_s
        );
    } else {
        std::cerr << "Type Not Supported, only support bf16 | fp16" << std::endl;
        exit(-1);
    }
    tracer.stop();
    tracer.print();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(output, unpacked_weights_s, quant_state.num_elements * sizeof(__half), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(packed_weights_s));
    CUDA_CHECK(cudaFree(absmax));
    CUDA_CHECK(cudaFree(unpacked_weights_s));
}

void nf4_dequant_warp8_batch8_one_phase(const QuantState& quant_state, __half* output) {
    constexpr int PROCESS_SIZE = 8;

    uint8_t* absmax_q_s;
    __half* code2_s;
    __half* absmax2_s;
    uint8_t* packed_weights_s;
    // output
    __half* unpacked_weights_s;

    CUDA_CHECK(cudaMalloc(&absmax_q_s, quant_state.num_blocks));
    CUDA_CHECK(cudaMalloc(&code2_s, 256 * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&absmax2_s, quant_state.num_groups * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&packed_weights_s, quant_state.packed_weights_len_in_bytes));
    CUDA_CHECK(cudaMalloc(&unpacked_weights_s, quant_state.num_elements * sizeof(__half)));
    Tracer tracer;
    tracer.memcpy_accumulate(quant_state.num_blocks)
        .memcpy_accumulate(256 * sizeof(__half))
        .memcpy_accumulate(quant_state.num_groups * sizeof(__half))
        .memcpy_accumulate(quant_state.packed_weights_len_in_bytes)
        .memcpy_accumulate(quant_state.num_elements * sizeof(__half));

    CUDA_CHECK(cudaMemcpy(absmax_q_s, quant_state.absmax_q, quant_state.absmax_q_len_in_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(code2_s, quant_state.code2, 256 * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(absmax2_s, quant_state.absmax2, quant_state.num_groups * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(packed_weights_s, quant_state.packed_weights, quant_state.packed_weights_len_in_bytes, cudaMemcpyHostToDevice))

    dim3 dequant_scale_block_dim(128);
    dim3 dequant_weights_grid_dim((quant_state.num_elements + dequant_scale_block_dim.x * PROCESS_SIZE - 1) / (dequant_scale_block_dim.x * PROCESS_SIZE));

    tracer.start();
    if (quant_state.compute_type == "bf16") {
        dequant_nf4_elements_one_phase_warp8_batchN_kernel<__nv_bfloat16, PROCESS_SIZE><<<dequant_weights_grid_dim, dequant_scale_block_dim>>> (
            packed_weights_s,
            absmax_q_s,
            quant_state.num_elements,
            (__nv_bfloat16*) absmax2_s,
            (__nv_bfloat16*) code2_s,
            quant_state.block_size,
            quant_state.group_size,
            quant_state.offset,
            (__nv_bfloat16*) unpacked_weights_s
        );
    } else if (quant_state.compute_type == "fp16") {
        dequant_nf4_elements_one_phase_warp8_batchN_kernel<__half, PROCESS_SIZE><<<dequant_weights_grid_dim, dequant_scale_block_dim>>> (
            packed_weights_s,
            absmax_q_s,
            quant_state.num_elements,
            (__half*) absmax2_s,
            (__half*) code2_s,
            quant_state.block_size,
            quant_state.group_size,
            quant_state.offset,
            (__half*) unpacked_weights_s
        );
    } else {
        std::cerr << "Type Not Supported, only support bf16 | fp16" << std::endl;
        exit(-1);
    }
    tracer.stop();
    tracer.print();

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(output, unpacked_weights_s, quant_state.num_elements * sizeof(__half), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(absmax_q_s));
    CUDA_CHECK(cudaFree(code2_s));
    CUDA_CHECK(cudaFree(absmax2_s));
    CUDA_CHECK(cudaFree(packed_weights_s));
    CUDA_CHECK(cudaFree(unpacked_weights_s));

}