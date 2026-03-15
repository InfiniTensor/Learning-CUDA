#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>

// NF4 码表 (bitsandbytes create_normal_map)，kernel 启动时加载到 shared memory
__constant__ float NF4_DEQUANT_TABLE[16] = {
    -1.0f,
    -0.6961928009986877f,
    -0.5250730514526367f,
    -0.39491748809814453f,
    -0.28444138169288635f,
    -0.18477343022823334f,
    -0.09105003625154495f,
    0.0f,
    0.07958029955625534f,
    0.16093020141124725f,
    0.24611230194568634f,
    0.33791524171829224f,
    0.44070982933044434f,
    0.5626170039176941f,
    0.7229568362236023f,
    1.0f
};

// float → half / __nv_bfloat16 模板转换
template <typename OutT>
__device__ __forceinline__ OutT nf4_cast_from_float(float x);

template <>
__device__ __forceinline__ half nf4_cast_from_float<half>(float x) {
    return __float2half(x);
}

template <>
__device__ __forceinline__ __nv_bfloat16 nf4_cast_from_float<__nv_bfloat16>(float x) {
    return __float2bfloat16(x);
}

// 取 half/__nv_bfloat16 的原始 16-bit 位表示
template <typename OutT>
__device__ __forceinline__ uint16_t nf4_raw_bits(OutT v) {
    return *reinterpret_cast<uint16_t*>(&v);
}

// log2(x)，要求 x 为 2 的幂
inline int log2_pow2(int x) {
    int r = 0;
    while (x > 1) { x >>= 1; r++; }
    return r;
}

// NF4 双重量化反量化 kernel
// 线程映射: 1 thread → 4 packed bytes → 8 output elements
// absmax_real = code2[absmax_q[block_idx]] * absmax2[group_idx] + offset
// output[i]   = NF4_TABLE[index] * absmax_real

template <typename OutT>
__global__ void nf4_dequantize_kernel(
    const uint8_t*  __restrict__ packed_weights,  // [n/2] 每字节 2 个 4-bit 索引
    const uint8_t*  __restrict__ absmax_q,        // [num_blocks] 一级缩放(二次量化后)
    const half*     __restrict__ absmax2,          // [num_groups] 二级缩放因子
    const half*     __restrict__ code2,            // [256] 二级码表: uint8 → float16
    float           offset,                        // 二级量化偏移
    int             log2_blocksize,                // log2(blocksize)，用位移代替除法
    int             log2_s2_blocksize,             // log2(s2_blocksize)
    int64_t         n_elements,                    // 总元素数 M*N
    OutT*           __restrict__ output            // [n] 反量化输出
)
{
    // NF4 码表加载到 shared memory，避免 constant memory 的 warp 串行化
    __shared__ float s_nf4_table[16];
    if (threadIdx.x < 16) {
        s_nf4_table[threadIdx.x] = NF4_DEQUANT_TABLE[threadIdx.x];
    }
    __syncthreads();

    // 每线程处理 4 packed bytes = 8 输出元素
    int tid_vec = blockIdx.x * blockDim.x + threadIdx.x;
    int n_packed = (int)((n_elements + 1) / 2);

    if (tid_vec >= (n_packed + 3) / 4) return;

    // 向量化读 4 字节，尾部不足时逐字节回退
    int byte_offset = tid_vec * 4;
    uint32_t packed4;
    if (byte_offset + 4 <= n_packed) {
        packed4 = reinterpret_cast<const uint32_t*>(packed_weights)[tid_vec];
    } else {
        packed4 = 0;
        for (int b = 0; b < 4 && byte_offset + b < n_packed; b++) {
            packed4 |= ((uint32_t)packed_weights[byte_offset + b]) << (b << 3);
        }
    }

    int elem_base = tid_vec * 8;

    uint32_t out_packed[4];

    #pragma unroll
    for (int b = 0; b < 4; b++) {
        int elem0 = elem_base + b * 2;
        int elem1 = elem0 + 1;

        // 解包高 4 位 / 低 4 位索引，查 NF4 码表
        uint8_t packed_byte = (packed4 >> (b * 8)) & 0xFF;
        uint8_t idx_hi = (packed_byte >> 4) & 0x0F;
        uint8_t idx_lo = packed_byte & 0x0F;

        float val_hi = s_nf4_table[idx_hi];
        float val_lo = s_nf4_table[idx_lo];

        // 双重量化反解: absmax_real = code2[absmax_q[block_idx]] * absmax2[group_idx] + offset
        int block_idx0 = elem0 >> log2_blocksize;
        int group_idx0 = block_idx0 >> log2_s2_blocksize;

        uint8_t aq0 = absmax_q[block_idx0];
        float absmax_real0 = __half2float(code2[aq0])
                           * __half2float(absmax2[group_idx0])
                           + offset;

        OutT out0, out1;

        if (elem0 < n_elements) {
            float dq0 = val_hi * absmax_real0;
            out0 = nf4_cast_from_float<OutT>(dq0);
        } else {
            out0 = nf4_cast_from_float<OutT>(0.0f);
        }

        if (elem1 < n_elements) {
            // 相邻元素大概率同块，跨块时才重新计算 absmax
            int block_idx1 = elem1 >> log2_blocksize;
            float absmax_real1;
            if (block_idx1 == block_idx0) {
                absmax_real1 = absmax_real0;
            } else {
                uint8_t aq1 = absmax_q[block_idx1];
                int group_idx1 = block_idx1 >> log2_s2_blocksize;
                absmax_real1 = __half2float(code2[aq1])
                             * __half2float(absmax2[group_idx1])
                             + offset;
            }
            float dq1 = val_lo * absmax_real1;
            out1 = nf4_cast_from_float<OutT>(dq1);
        } else {
            out1 = nf4_cast_from_float<OutT>(0.0f);
        }

        // 两个 fp16/bf16 打包为一个 uint32_t
        uint16_t bits0 = nf4_raw_bits(out0);
        uint16_t bits1 = nf4_raw_bits(out1);
        out_packed[b] = (uint32_t)bits0 | ((uint32_t)bits1 << 16);
    }

    // 向量化写入: 完整 4-pack 用 uint4 (128-bit) 一次写出，尾部逐个写
    int out_base = tid_vec * 4;
    uint32_t* out_u32 = reinterpret_cast<uint32_t*>(output);

    int valid_packs = 0;
    for (int b = 0; b < 4; b++) {
        if (byte_offset + b < n_packed) valid_packs++;
    }

    if (valid_packs == 4) {
        reinterpret_cast<uint4*>(out_u32)[tid_vec] =
            make_uint4(out_packed[0], out_packed[1], out_packed[2], out_packed[3]);
    } else {
        for (int b = 0; b < valid_packs; b++) {
            out_u32[out_base + b] = out_packed[b];
        }
    }
}
