#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cstddef>
#include <cstdint>

namespace hadamard {

template <class T> __host__ __device__ float as_float(T value);
template <> __host__ __device__ inline float as_float(__half value) { return __half2float(value); }
template <> __host__ __device__ inline float as_float(__nv_bfloat16 value) { return __bfloat162float(value); }
template <class T> __host__ __device__ T as_storage(float value);
template <> __host__ __device__ inline __half as_storage(float value) { return __float2half_rn(value); }
template <> __host__ __device__ inline __nv_bfloat16 as_storage(float value) { return __float2bfloat16_rn(value); }

template <class T> __device__ inline T storage_from_bits(unsigned short bits);
template <> __device__ inline __half storage_from_bits<__half>(unsigned short bits) {
    return __ushort_as_half(bits);
}
template <> __device__ inline __nv_bfloat16 storage_from_bits<__nv_bfloat16>(unsigned short bits) {
    return __ushort_as_bfloat16(bits);
}
template <class T> __device__ inline unsigned short storage_bits(T value);
template <> __device__ inline unsigned short storage_bits(__half value) {
    return __half_as_ushort(value);
}
template <> __device__ inline unsigned short storage_bits(__nv_bfloat16 value) {
    return __bfloat16_as_ushort(value);
}

template <class T>
__global__ void to_float_kernel(const T* input, float* output, std::size_t size) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < size) output[i] = as_float(input[i]);
}

__global__ void butterfly_stage(const float* input, float* output, std::size_t size, int stride) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < size) {
        const float mine = input[i], peer = input[i ^ stride];
        output[i] = (i & stride) ? peer - mine : mine + peer;
    }
}

template <class T>
__global__ void from_float_kernel(const float* input, T* output, std::size_t size, float scale) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < size) output[i] = as_storage<T>(input[i] * scale);
}

inline constexpr std::size_t wide_pair_min_rows = 4096;

template <int N, bool Transform, bool Quantize>
__host__ __device__ inline int warp_row_lanes(std::size_t rows, const void* input, const void* output) {
    if constexpr (Transform && N >= 32) {
        const bool aligned_input = (reinterpret_cast<std::uintptr_t>(input) & 3) == 0;
        const bool aligned_output = Quantize || (reinterpret_cast<std::uintptr_t>(output) & 3) == 0;
        return rows >= wide_pair_min_rows && aligned_input && aligned_output ? 16 : 32;
    }
    return 32;
}

// Called only by launch_warp after the row-count and pointer-alignment checks.
template <class T, int N, bool Quantize>
__global__ void warp_pairs_kernel(const T* input, T* output, std::uint8_t* packed,
                                  float* scales, std::size_t rows, float transform_scale) {
    constexpr int Items = N / 32;
    static_assert(N >= 32 && N <= 256 && (N & (N - 1)) == 0);
    const int lane = threadIdx.x % 16;
    const std::size_t row = static_cast<std::size_t>(blockIdx.x) * (blockDim.x / 16) + threadIdx.x / 16;
    // Only a whole physical warp exits before subgroup shuffles.
    if ((row / 2) * 2 >= rows) return;
    const bool valid = row < rows;
    float low[Items], high[Items];
    // Bit zero stays in registers; both half-warps retain all active lanes.
    // Tail groups must participate in the width-16 shuffles below.
#pragma unroll
    for (int k = 0; k < Items; ++k) {
        const std::size_t index = row * N + lane * 2 + 32 * k;
        float a = 0.0f, b = 0.0f;
        if (valid) {
            // The common launch predicate proves four-byte alignment.
            const auto raw = *reinterpret_cast<const unsigned int*>(input + index);
            a = as_float(storage_from_bits<T>(static_cast<unsigned short>(raw)));
            b = as_float(storage_from_bits<T>(static_cast<unsigned short>(raw >> 16)));
        }
        low[k] = a + b;
        high[k] = a - b;
    }
#pragma unroll
    for (int stride = 1; stride < 16; stride *= 2) {
#pragma unroll
        for (int k = 0; k < Items; ++k) {
            const float a = low[k], b = high[k];
            const float peer_a = __shfl_xor_sync(0xffffffff, a, stride, 16);
            const float peer_b = __shfl_xor_sync(0xffffffff, b, stride, 16);
            low[k] = (lane & stride) ? peer_a - a : a + peer_a;
            high[k] = (lane & stride) ? peer_b - b : b + peer_b;
        }
    }
#pragma unroll
    for (int stride = 1; stride < Items; stride *= 2) {
#pragma unroll
        for (int k = 0; k < Items; ++k) {
            if (!(k & stride)) {
                const float a = low[k], b = low[k + stride];
                const float c = high[k], d = high[k + stride];
                low[k] = a + b;
                low[k + stride] = a - b;
                high[k] = c + d;
                high[k + stride] = c - d;
            }
        }
    }
#pragma unroll
    for (int k = 0; k < Items; ++k) {
        low[k] = as_float(as_storage<T>(low[k] * transform_scale));
        high[k] = as_float(as_storage<T>(high[k] * transform_scale));
        if constexpr (!Quantize) {
            if (valid) {
                const std::size_t index = row * N + lane * 2 + 32 * k;
                const auto a = static_cast<unsigned int>(storage_bits(as_storage<T>(low[k])));
                const auto b = static_cast<unsigned int>(storage_bits(as_storage<T>(high[k])));
                *reinterpret_cast<unsigned int*>(output + index) = a | (b << 16);
            }
        }
    }
    if constexpr (Quantize) {
        float magnitude = 0.0f;
#pragma unroll
        for (int k = 0; k < Items; ++k)
            magnitude = fmaxf(magnitude, fmaxf(fabsf(low[k]), fabsf(high[k])));
#pragma unroll
        for (int stride = 8; stride > 0; stride /= 2)
            magnitude = fmaxf(magnitude, __shfl_xor_sync(0xffffffff, magnitude, stride, 16));
        const float scale = magnitude == 0 ? 1.0f : magnitude / 7.0f;
        if (valid && lane == 0) scales[row] = scale;
#pragma unroll
        for (int k = 0; k < Items; ++k) {
            const int a = max(-7, min(7, __float2int_rn(low[k] / scale)));
            const int b = max(-7, min(7, __float2int_rn(high[k] / scale)));
            if (valid)
                packed[row * (N / 2) + lane + 16 * k] = static_cast<std::uint8_t>((a & 15) | ((b & 15) << 4));
        }
    }
}

template <class T, int N, bool Transform, bool Quantize>
__global__ void warp_kernel(const T* input, T* output, std::uint8_t* packed,
                            float* scales, std::size_t rows, float transform_scale) {
    constexpr int Items = (N + 31) / 32;
    const int lane = threadIdx.x % 32;
    const std::size_t row = static_cast<std::size_t>(blockIdx.x) * (blockDim.x / 32) + threadIdx.x / 32;
    // An entire warp exits together: every shuffle below has a full active warp.
    if (row >= rows) return;
    float values[Items];
#pragma unroll
    for (int k = 0; k < Items; ++k)
        values[k] = lane + 32 * k < N ? as_float(input[row * N + lane + 32 * k]) : 0;
    if constexpr (Transform) {
#pragma unroll
        for (int stride = 1; stride < (N < 32 ? N : 32); stride *= 2) {
#pragma unroll
            for (int k = 0; k < Items; ++k) {
                const float mine = values[k];
                const float peer = __shfl_xor_sync(0xffffffff, mine, stride);
                values[k] = (lane & stride) ? peer - mine : mine + peer;
            }
        }
#pragma unroll
        for (int stride = 1; stride < Items; stride *= 2) {
#pragma unroll
            for (int k = 0; k < Items; ++k) {
                if (!(k & stride)) {
                    const float a = values[k], b = values[k + stride];
                    values[k] = a + b;
                    values[k + stride] = a - b;
                }
            }
        }
    }
    // Preserve the exact transform-then-quantize contract, including output rounding.
#pragma unroll
    for (int k = 0; k < Items; ++k) {
        if (lane + 32 * k < N) {
            if constexpr (Transform) values[k] = as_float(as_storage<T>(values[k] * transform_scale));
            if constexpr (!Quantize) output[row * N + lane + 32 * k] = as_storage<T>(values[k]);
        } else values[k] = 0;
    }
    if constexpr (Quantize) {
        float magnitude = 0;
#pragma unroll
        for (int k = 0; k < Items; ++k) magnitude = fmaxf(magnitude, fabsf(values[k]));
#pragma unroll
        for (int stride = 16; stride > 0; stride /= 2)
            magnitude = fmaxf(magnitude, __shfl_xor_sync(0xffffffff, magnitude, stride));
        const float scale = magnitude == 0 ? 1.0f : magnitude / 7.0f;
        if (lane == 0) scales[row] = scale;
#pragma unroll
        for (int k = 0; k < Items; ++k) {
            const int q = max(-7, min(7, __float2int_rn(values[k] / scale)));
            const int next = __shfl_down_sync(0xffffffff, q, 1);
            const int index = lane + 32 * k;
            if (!(lane & 1) && index < N) {
                const int high = index + 1 < N ? (next & 15) : 0;
                packed[row * ((N + 1) / 2) + index / 2] = static_cast<std::uint8_t>((q & 15) | (high << 4));
            }
        }
    }
}

// Keep the original kernel separate so fallback launches retain its register footprint.
template <class T, int N, bool Transform, bool Quantize>
inline void launch_warp(const T* input, T* output, std::uint8_t* packed, float* scales,
                        std::size_t rows, float scale, int threads, cudaStream_t stream = nullptr) {
    if constexpr (Transform && N >= 32) {
        if (warp_row_lanes<N, Transform, Quantize>(rows, input, output) == 16) {
            const auto blocks = static_cast<unsigned int>((rows - 1) / (threads / 16) + 1);
            warp_pairs_kernel<T, N, Quantize><<<blocks, threads, 0, stream>>>(input, output, packed, scales, rows, scale);
            return;
        }
    }
    const auto blocks = static_cast<unsigned int>((rows - 1) / (threads / 32) + 1);
    warp_kernel<T, N, Transform, Quantize><<<blocks, threads, 0, stream>>>(input, output, packed, scales, rows, scale);
}

// Dense H_N multiplication is an intentionally distinct Tensor Core algorithm.
// Compare it to FWHT experimentally; using Tensor Cores does not imply a speedup.
template <class T, int N>
__global__ void tensor_core_kernel(const T* input, const T* matrix, T* output,
                                   std::size_t rows, float scale) {
    using namespace nvcuda;
    __shared__ __align__(32) T a[16 * N];
    __shared__ __align__(32) float c[16 * 16];
    const std::size_t row_start = static_cast<std::size_t>(blockIdx.x) * 16;
    const int column_start = blockIdx.y * 16;
    for (int i = threadIdx.x; i < 16 * N; i += blockDim.x) {
        const std::size_t row = row_start + i / N;
        a[i] = row < rows ? input[row * N + i % N] : as_storage<T>(0);
    }
    __syncthreads();
    if (threadIdx.x < 32) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, T, wmma::row_major> af;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, T, wmma::row_major> bf;
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> cf;
        wmma::fill_fragment(cf, 0.0f);
#pragma unroll
        for (int k = 0; k < N; k += 16) {
            wmma::load_matrix_sync(af, a + k, N);
            wmma::load_matrix_sync(bf, matrix + k * N + column_start, N);
            wmma::mma_sync(cf, af, bf, cf);
        }
        wmma::store_matrix_sync(c, cf, 16, wmma::mem_row_major);
    }
    __syncthreads();
    for (int i = threadIdx.x; i < 256; i += blockDim.x) {
        const std::size_t row = row_start + i / 16;
        if (row < rows) output[row * N + column_start + i % 16] = as_storage<T>(c[i] * scale);
    }
}

}  // namespace hadamard
