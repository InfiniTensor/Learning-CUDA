#pragma once

#include "kernels.cuh"

namespace hadamard {

template <class T, bool Transform, bool Quantize>
__global__ void contiguous256_kernel(const T* input, T* output, std::uint8_t* packed,
                                     float* scales, std::size_t rows, float transform_scale) {
    const int lane = threadIdx.x % 32;
    const std::size_t row = static_cast<std::size_t>(blockIdx.x) * (blockDim.x / 32) + threadIdx.x / 32;
    // The whole warp takes this guard before any row pointer is dereferenced.
    if (row >= rows) return;
    const T* source = input + row * 256 + lane * 8;
    float values[8];
    if ((reinterpret_cast<std::uintptr_t>(source) & 15) == 0) {
        const uint4 raw = *reinterpret_cast<const uint4*>(source);
        const unsigned int words[4] = {raw.x, raw.y, raw.z, raw.w};
#pragma unroll
        for (int k = 0; k < 8; ++k)
            values[k] = as_float(storage_from_bits<T>(static_cast<unsigned short>(words[k / 2] >> (16 * (k % 2)))));
    } else {
        // Contiguous tensors may begin at a legal two-byte storage offset.
#pragma unroll
        for (int k = 0; k < 8; ++k) values[k] = as_float(source[k]);
    }
    if constexpr (Transform) {
        // Global index bits 0,1,2, followed by 3,4,5,6,7: same stage order.
#pragma unroll
        for (int stride = 1; stride < 8; stride *= 2) {
#pragma unroll
            for (int k = 0; k < 8; ++k) {
                if (!(k & stride)) {
                    const float a = values[k], b = values[k + stride];
                    values[k] = a + b;
                    values[k + stride] = a - b;
                }
            }
        }
#pragma unroll
        for (int stride = 1; stride < 32; stride *= 2) {
#pragma unroll
            for (int k = 0; k < 8; ++k) {
                const float mine = values[k];
                const float peer = __shfl_xor_sync(0xffffffff, mine, stride);
                values[k] = (lane & stride) ? peer - mine : mine + peer;
            }
        }
#pragma unroll
        for (int k = 0; k < 8; ++k)
            values[k] = as_float(as_storage<T>(values[k] * transform_scale));
    }
    if constexpr (!Quantize) {
        T* destination = output + row * 256 + lane * 8;
        if ((reinterpret_cast<std::uintptr_t>(destination) & 15) == 0) {
            unsigned int words[4];
#pragma unroll
            for (int k = 0; k < 4; ++k)
                words[k] = static_cast<unsigned int>(storage_bits(as_storage<T>(values[k * 2]))) |
                    (static_cast<unsigned int>(storage_bits(as_storage<T>(values[k * 2 + 1]))) << 16);
            *reinterpret_cast<uint4*>(destination) = make_uint4(words[0], words[1], words[2], words[3]);
        } else {
#pragma unroll
            for (int k = 0; k < 8; ++k) destination[k] = as_storage<T>(values[k]);
        }
    }
    if constexpr (Quantize) {
        float magnitude = 0;
#pragma unroll
        for (int k = 0; k < 8; ++k) magnitude = fmaxf(magnitude, fabsf(values[k]));
#pragma unroll
        for (int stride = 16; stride > 0; stride /= 2)
            magnitude = fmaxf(magnitude, __shfl_xor_sync(0xffffffff, magnitude, stride));
        const float scale = magnitude == 0 ? 1.0f : magnitude / 7.0f;
        if (lane == 0) scales[row] = scale;
        unsigned int bytes = 0;
#pragma unroll
        for (int pair = 0; pair < 4; ++pair) {
            const int low = max(-7, min(7, __float2int_rn(values[pair * 2] / scale)));
            const int high = max(-7, min(7, __float2int_rn(values[pair * 2 + 1] / scale)));
            bytes |= static_cast<unsigned int>((low & 15) | ((high & 15) << 4)) << (pair * 8);
        }
        std::uint8_t* destination = packed + row * 128 + lane * 4;
        if ((reinterpret_cast<std::uintptr_t>(destination) & 3) == 0) {
            *reinterpret_cast<unsigned int*>(destination) = bytes;
        } else {
#pragma unroll
            for (int k = 0; k < 4; ++k) destination[k] = static_cast<std::uint8_t>(bytes >> (k * 8));
        }
    }
}

}  // namespace hadamard
