#pragma once

#include <cstdint>

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

inline int log2_pow2(int x) {
    int r = 0;
    while (x > 1) {
        x >>= 1;
        r++;
    }
    return r;
}

__device__ __forceinline__ uint32_t float_to_bits(float v) {
    union {
        float f;
        uint32_t u;
    } x;
    x.f = v;
    return x.u;
}

__device__ __forceinline__ float bits_to_float(uint32_t v) {
    union {
        float f;
        uint32_t u;
    } x;
    x.u = v;
    return x.f;
}

__device__ __forceinline__ float half_bits_to_float(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1Fu;
    uint32_t mant = h & 0x03FFu;

    uint32_t out;
    if (exp == 0) {
        if (mant == 0) {
            out = sign;
        } else {
            exp = 127 - 15 + 1;
            while ((mant & 0x0400u) == 0) {
                mant <<= 1;
                exp--;
            }
            mant &= 0x03FFu;
            out = sign | (exp << 23) | (mant << 13);
        }
    } else if (exp == 0x1Fu) {
        out = sign | 0x7F800000u | (mant << 13);
    } else {
        out = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
    }

    return bits_to_float(out);
}

__device__ __forceinline__ uint16_t float_to_half_bits(float v) {
    uint32_t x = float_to_bits(v);
    uint32_t sign = (x >> 16) & 0x8000u;
    int32_t exp = (int32_t)((x >> 23) & 0xFFu) - 127 + 15;
    uint32_t mant = x & 0x7FFFFFu;

    if (exp <= 0) {
        if (exp < -10) {
            return (uint16_t)sign;
        }
        mant = (mant | 0x800000u) >> (1 - exp);
        if ((mant & 0x00001000u) != 0) {
            mant += 0x00002000u;
        }
        return (uint16_t)(sign | (mant >> 13));
    }

    if (exp >= 31) {
        return (uint16_t)(sign | 0x7C00u);
    }

    uint32_t out = sign | ((uint32_t)exp << 10) | (mant >> 13);
    if ((mant & 0x00001000u) != 0) {
        out += 1;
    }
    return (uint16_t)out;
}

__device__ __forceinline__ uint16_t float_to_bf16_bits(float v) {
    uint32_t x = float_to_bits(v);
    uint32_t lsb = (x >> 16) & 1u;
    x += 0x7FFFu + lsb;
    return (uint16_t)(x >> 16);
}

template <bool OUTPUT_BF16>
__global__ void nf4_dequantize_kernel(
    const uint8_t* __restrict__ packed_weights,
    const uint8_t* __restrict__ absmax_q,
    const uint16_t* __restrict__ absmax2,
    const uint16_t* __restrict__ code2,
    float offset,
    int log2_blocksize,
    int log2_s2_blocksize,
    int64_t n_elements,
    uint16_t* __restrict__ output_bits) {
    __shared__ float s_nf4_table[16];
    if (threadIdx.x < 16) {
        s_nf4_table[threadIdx.x] = NF4_DEQUANT_TABLE[threadIdx.x];
    }
    __syncthreads();

    int tid_vec = blockIdx.x * blockDim.x + threadIdx.x;
    int n_packed = (int)((n_elements + 1) / 2);
    int n_packed_vec = (n_packed + 3) / 4;
    if (tid_vec >= n_packed_vec) {
        return;
    }

    int byte_offset = tid_vec * 4;
    uint32_t packed4 = 0;
    if (byte_offset + 4 <= n_packed) {
        packed4 = reinterpret_cast<const uint32_t*>(packed_weights)[tid_vec];
    } else {
        for (int b = 0; b < 4 && byte_offset + b < n_packed; ++b) {
            packed4 |= ((uint32_t)packed_weights[byte_offset + b]) << (b << 3);
        }
    }

    int elem_base = tid_vec * 8;
    uint32_t out_packed[4];

    #pragma unroll
    for (int b = 0; b < 4; ++b) {
        int elem0 = elem_base + b * 2;
        int elem1 = elem0 + 1;

        uint8_t packed_byte = (packed4 >> (b * 8)) & 0xFF;
        uint8_t idx_hi = (packed_byte >> 4) & 0x0F;
        uint8_t idx_lo = packed_byte & 0x0F;

        float val_hi = s_nf4_table[idx_hi];
        float val_lo = s_nf4_table[idx_lo];

        int block_idx0 = elem0 >> log2_blocksize;
        int group_idx0 = block_idx0 >> log2_s2_blocksize;
        uint8_t aq0 = absmax_q[block_idx0];

        float absmax_real0 = half_bits_to_float(code2[aq0])
                           * half_bits_to_float(absmax2[group_idx0])
                           + offset;

        uint16_t out0;
        if (elem0 < n_elements) {
            float dq0 = val_hi * absmax_real0;
            out0 = OUTPUT_BF16 ? float_to_bf16_bits(dq0) : float_to_half_bits(dq0);
        } else {
            out0 = OUTPUT_BF16 ? float_to_bf16_bits(0.0f) : float_to_half_bits(0.0f);
        }

        uint16_t out1;
        if (elem1 < n_elements) {
            int block_idx1 = elem1 >> log2_blocksize;
            float absmax_real1;
            if (block_idx1 == block_idx0) {
                absmax_real1 = absmax_real0;
            } else {
                uint8_t aq1 = absmax_q[block_idx1];
                int group_idx1 = block_idx1 >> log2_s2_blocksize;
                absmax_real1 = half_bits_to_float(code2[aq1])
                             * half_bits_to_float(absmax2[group_idx1])
                             + offset;
            }
            float dq1 = val_lo * absmax_real1;
            out1 = OUTPUT_BF16 ? float_to_bf16_bits(dq1) : float_to_half_bits(dq1);
        } else {
            out1 = OUTPUT_BF16 ? float_to_bf16_bits(0.0f) : float_to_half_bits(0.0f);
        }

        out_packed[b] = (uint32_t)out0 | ((uint32_t)out1 << 16);
    }

    int out_base = tid_vec * 4;
    uint32_t* out_u32 = reinterpret_cast<uint32_t*>(output_bits);

    int valid_packs = 0;
    for (int b = 0; b < 4; ++b) {
        if (byte_offset + b < n_packed) {
            valid_packs++;
        }
    }

    #pragma unroll
    for (int b = 0; b < 4; ++b) {
        if (b < valid_packs) {
            out_u32[out_base + b] = out_packed[b];
        }
    }
}
