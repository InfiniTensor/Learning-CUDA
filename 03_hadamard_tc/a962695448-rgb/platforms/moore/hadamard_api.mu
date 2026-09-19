// Ported from the verified MetaX implementation at cb82ea6.
// MUSA results require independent validation on Moore Threads hardware.
#include "hadamard_api.h"
#define HADAMARD_QUANT_HD __host__ __device__
#include "exact_int4.hpp"
#undef HADAMARD_QUANT_HD

#include <cmath>
#include <limits>

namespace hadamard::moore {
namespace {

template<class T> __device__ float read_value(T value);
template<> __device__ float read_value(__half value) { return __half2float(value); }
template<> __device__ float read_value(__mt_bfloat16 value) {
    return __bfloat162float(value);
}
template<class T> __device__ T store_value(float value);
template<> __device__ __half store_value(float value) { return __float2half_rn(value); }
template<> __device__ __mt_bfloat16 store_value(float value) {
    return __float2bfloat16_rn(value);
}

// The baseline retains native RNE division. The optimized paths compare exact
// integer significands at the composite float32/integer rounding boundaries.
template<bool IntegerCompare>
__device__ std::uint8_t quantize_pair(const float* values, int pair, int n,
                                    float row_scale) {
    std::uint8_t byte = 0;
    for (int k = 0; k < 2 && pair * 2 + k < n; ++k) {
        int q;
        if constexpr (IntegerCompare) {
            q = detail::exact_int4_from_bits(__float_as_uint(values[pair * 2 + k]),
                                            __float_as_uint(row_scale));
        } else {
            const float x = __fdiv_rn(values[pair * 2 + k], row_scale);
            const float lower = floorf(x);
            const float fraction = x - lower;
            q = static_cast<int>(lower);
            if (fraction > 0.5f || (fraction == 0.5f && q % 2 != 0)) ++q;
            q = q < -7 ? -7 : (q > 7 ? 7 : q);
        }
        byte |= static_cast<std::uint8_t>((q & 15) << (4 * k));
    }
    return byte;
}

// 沿用固定来源版本的基线算法：一元素一线程、每层两次屏障、线程 0 串行归约。
// 与 optimized 独立保留蝶形计算，供同一运行环境下公平比较。
template<class T, bool Transform, bool Quantize>
__global__ void baseline_kernel(const T* input, T* output, std::uint8_t* packed,
                                float* scales, std::size_t rows, int n, float scale) {
    __shared__ float values[256];
    __shared__ float row_scale;
    const int i = threadIdx.x;
    for (std::size_t row = blockIdx.x; row < rows; row += gridDim.x) {
        const std::size_t offset = row * static_cast<std::size_t>(n);
        if (i < n) values[i] = read_value(input[offset + i]);
        __syncthreads();
        if constexpr (Transform) {
            for (int stride = 1; stride < n; stride *= 2) {
                float next = 0;
                if (i < n) {
                    const float a = values[i], b = values[i ^ stride];
                    next = (i & stride) ? b - a : a + b;
                }
                __syncthreads();
                if (i < n) values[i] = next;
                __syncthreads();
            }
            if (i < n) values[i] = read_value(store_value<T>(values[i] * scale));
            __syncthreads();
        }
        if constexpr (!Quantize) {
            if (i < n) output[offset + i] = store_value<T>(values[i]);
        } else {
            if (i == 0) {
                float magnitude = 0;
                for (int j = 0; j < n; ++j)
                    magnitude = fmaxf(magnitude, fabsf(values[j]));
                row_scale = magnitude == 0 ? 1.0f : __fdiv_rn(magnitude, 7.0f);
                scales[row] = row_scale;
            }
            __syncthreads();
            const int bytes = (n + 1) / 2;
            if (i < bytes) packed[row * bytes + i] = quantize_pair<false>(values, i, n, row_scale);
        }
        // 多行复用同一 block 时，防止下行加载覆盖仍在打包/写出的上一行。
        if (rows - row > gridDim.x) __syncthreads();
    }
}

// 每个线程独占一个蝶形的两个输入/输出，因此单层没有线程间读写冲突，
// 只需在本层结束后同步。此算法不假定 warp 宽度，不使用 warp 级隐式同步。
template<class T, bool Transform, bool Quantize>
__global__ void optimized_kernel(const T* input, T* output, std::uint8_t* packed,
                                 float* scales, std::size_t rows, int n, float scale) {
    __shared__ float values[256];
    __shared__ float maxima[128];
    __shared__ float row_scale;
    const int tid = threadIdx.x;
    for (std::size_t row = blockIdx.x; row < rows; row += gridDim.x) {
        const std::size_t offset = row * static_cast<std::size_t>(n);
        for (int i = tid; i < n; i += blockDim.x) values[i] = read_value(input[offset + i]);
        __syncthreads();
        if constexpr (Transform) {
            for (int stride = 1; stride < n; stride *= 2) {
                if (tid < n / 2) {
                    const int low = ((tid & ~(stride - 1)) << 1) | (tid & (stride - 1));
                    const float a = values[low], b = values[low + stride];
                    values[low] = a + b;
                    values[low + stride] = a - b;
                }
                __syncthreads();
            }
        }
        if constexpr (!Quantize) {
            for (int i = tid; i < n; i += blockDim.x)
                output[offset + i] = store_value<T>(values[i] * scale);
        } else {
            float magnitude = 0;
            for (int i = tid; i < n; i += blockDim.x) {
                float value = values[i];
                if constexpr (Transform) {
                    value = read_value(store_value<T>(value * scale));
                    values[i] = value;
                }
                magnitude = fmaxf(magnitude, fabsf(value));
            }
            maxima[tid] = magnitude;
            __syncthreads();
            for (int stride = blockDim.x / 2; stride > 1; stride /= 2) {
                if (tid < stride) maxima[tid] = fmaxf(maxima[tid], maxima[tid + stride]);
                __syncthreads();
            }
            if (tid == 0) {
                // 最后两个值只由线程 0 消费，与 scale 写入合并，少一次屏障。
                const float maximum = fmaxf(maxima[0], maxima[1]);
                row_scale = maximum == 0 ? 1.0f : __fdiv_rn(maximum, 7.0f);
                scales[row] = row_scale;
            }
            __syncthreads();
            const int bytes = (n + 1) / 2;
            for (int i = tid; i < bytes; i += blockDim.x)
                packed[row * bytes + i] = quantize_pair<true>(values, i, n, row_scale);
        }
        if (rows - row > gridDim.x) __syncthreads();
    }
}

#if defined(HADAMARD_MOORE_SHUFFLE32)

// Only select this candidate after the native MUSA width=32 shuffle probe passes.
// A CTA has eight logical 32-lane row groups. Tail groups execute with zero values;
// all 256 threads participate in each exchange, including the two reported 128-lane units.
__device__ int shuffle32_quantized_nibble(float value, float scale) {
    return detail::exact_int4_from_bits(__float_as_uint(value), __float_as_uint(scale)) & 15;
}

template<class T, bool Transform, bool Quantize, int N>
__global__ void shuffle32_kernel(const T* input, T* output, std::uint8_t* packed,
                              float* scales, std::size_t rows, float scale) {
    constexpr int width = 32;
    constexpr int rows_per_block = 8;
    constexpr int registers = N > width ? N / width : 1;
    const int lane = threadIdx.x % width;
    const int warp = threadIdx.x / width;
    const std::size_t first_row = static_cast<std::size_t>(blockIdx.x) * rows_per_block;
    const std::size_t row_stride = static_cast<std::size_t>(gridDim.x) * rows_per_block;
    // Every thread in a CTA executes the same number of row-batch iterations.
    for (std::size_t row_base = first_row; row_base < rows; row_base += row_stride) {
        const std::size_t row = row_base + warp;
        const bool active_row = row < rows;
        const std::size_t offset = row * N;
        float values[registers];
        #pragma unroll
        for (int r = 0; r < registers; ++r) {
            const int i = lane + r * width;
            values[r] = active_row && i < N ? read_value(input[offset + i]) : 0.0f;
        }
        if constexpr (Transform) {
            #pragma unroll
            for (int stride = 1; stride < (N < width ? N : width); stride *= 2) {
                #pragma unroll
                for (int r = 0; r < registers; ++r) {
                    const float current = values[r];
                    const float peer = __shfl_xor_sync(0xffffffffu, current, stride, width);
                    values[r] = (lane & stride) ? peer - current : current + peer;
                }
            }
            // Higher butterfly stages exchange registers owned by the same lane.
            #pragma unroll
            for (int stride = 1; stride < registers; stride *= 2) {
                #pragma unroll
                for (int r = 0; r < registers; ++r) {
                    if (!(r & stride)) {
                        const float x = values[r], y = values[r + stride];
                        values[r] = x + y;
                        values[r + stride] = x - y;
                    }
                }
            }
        }
        if constexpr (!Quantize) {
            #pragma unroll
            for (int r = 0; r < registers; ++r) {
                const int i = lane + r * width;
                if (active_row && i < N) output[offset + i] = store_value<T>(values[r] * scale);
            }
        } else {
            float magnitude = 0.0f;
            #pragma unroll
            for (int r = 0; r < registers; ++r) {
                if constexpr (Transform) values[r] = read_value(store_value<T>(values[r] * scale));
                magnitude = fmaxf(magnitude, fabsf(values[r]));
            }
            #pragma unroll
            for (int stride = width / 2; stride > 0; stride /= 2)
                magnitude = fmaxf(magnitude, __shfl_xor_sync(0xffffffffu, magnitude, stride, width));
            const float row_scale = magnitude == 0.0f ? 1.0f : __fdiv_rn(magnitude, 7.0f);
            if (active_row && lane == 0) scales[row] = row_scale;
            #pragma unroll
            for (int r = 0; r < registers; ++r) {
                // shuffle 在分支之前：奇数 lane 同样必须提供其相邻元素。
                const float peer = __shfl_xor_sync(0xffffffffu, values[r], 1, width);
                const int i = lane + r * width;
                if (active_row && (lane & 1) == 0 && i < N) {
                    const int low = shuffle32_quantized_nibble(values[r], row_scale);
                    const int high = i + 1 < N ? shuffle32_quantized_nibble(peer, row_scale) : 0;
                    packed[row * ((N + 1) / 2) + i / 2]
                        = static_cast<std::uint8_t>(low | (high << 4));
                }
            }
        }
    }
}

template<class T, bool Transform, bool Quantize>
musaError_t launch_shuffle32(const T* input, T* output, std::uint8_t* packed, float* scales,
                          std::size_t rows, int n, float scale, musaStream_t stream) {
    // launch() checks rows <= SIZE_MAX / sizeof(float), so rows+7 cannot overflow.
    const std::size_t requested = (rows + 7) / 8;
    const unsigned int blocks = static_cast<unsigned int>(requested < 65535 ? requested : 65535);
    #define LAUNCH_SHUFFLE32(N) case N: \
        shuffle32_kernel<T, Transform, Quantize, N><<<blocks, 256, 0, stream>>>( \
            input, output, packed, scales, rows, scale); \
        break
    switch (n) {
        LAUNCH_SHUFFLE32(1);
        LAUNCH_SHUFFLE32(2);
        LAUNCH_SHUFFLE32(4);
        LAUNCH_SHUFFLE32(8);
        LAUNCH_SHUFFLE32(16);
        LAUNCH_SHUFFLE32(32);
        LAUNCH_SHUFFLE32(64);
        LAUNCH_SHUFFLE32(128);
        LAUNCH_SHUFFLE32(256);
        default: return musaErrorInvalidValue;
    }
    #undef LAUNCH_SHUFFLE32
    return musaGetLastError();
}

#endif  // HADAMARD_MOORE_SHUFFLE32

bool valid_range(const void* pointer, std::size_t bytes, std::size_t alignment) {
    const auto address = reinterpret_cast<std::uintptr_t>(pointer);
    return pointer != nullptr && address % alignment == 0
        && bytes <= std::numeric_limits<std::uintptr_t>::max() - address;
}

bool overlaps(const void* left, std::size_t left_size,
              const void* right, std::size_t right_size) {
    // valid_range 已确保两个区间端点的加法不会溢出。
    const auto a = reinterpret_cast<std::uintptr_t>(left);
    const auto b = reinterpret_cast<std::uintptr_t>(right);
    return a < b + right_size && b < a + left_size;
}

template<class T, bool Transform, bool Quantize>
musaError_t launch(const T* input, T* output, std::uint8_t* packed, float* scales,
                   std::size_t rows, int n, float scale, musaStream_t stream, Method method) {
    if (n < 1 || n > 256 || (n & (n - 1)) != 0
        || !std::isfinite(scale) || scale <= 0
        || (method != Method::Baseline && method != Method::Optimized && method != Method::Shuffle32))
        return musaErrorInvalidValue;
    if (rows == 0) return musaSuccess;

    constexpr auto maximum = std::numeric_limits<std::size_t>::max();
    const std::size_t row_bytes = static_cast<std::size_t>(n) * sizeof(T);
    if (rows > maximum / row_bytes || rows > maximum / sizeof(float))
        return musaErrorInvalidValue;
    const std::size_t input_bytes = rows * row_bytes;
    if (!valid_range(input, input_bytes, alignof(T))) return musaErrorInvalidValue;
    if constexpr (Quantize) {
        const std::size_t packed_bytes = rows * static_cast<std::size_t>((n + 1) / 2);
        const std::size_t scale_bytes = rows * sizeof(float);
        if (!valid_range(packed, packed_bytes, alignof(std::uint8_t))
            || !valid_range(scales, scale_bytes, alignof(float))
            || overlaps(input, input_bytes, packed, packed_bytes)
            || overlaps(input, input_bytes, scales, scale_bytes)
            || overlaps(packed, packed_bytes, scales, scale_bytes)) return musaErrorInvalidValue;
    } else {
        if (!valid_range(output, input_bytes, alignof(T))
            || (input != output && overlaps(input, input_bytes, output, input_bytes)))
            return musaErrorInvalidValue;
    }

    if (method == Method::Shuffle32) {
        #if defined(HADAMARD_MOORE_SHUFFLE32)
        return launch_shuffle32<T, Transform, Quantize>(input, output, packed, scales, rows, n, scale, stream);
        #else
        return musaErrorNotSupported;
        #endif
    }

    // 保守上限兼容不同设备的 grid.x 限制，超过上限由 block 顺序处理多行。
    const unsigned int blocks = static_cast<unsigned int>(rows < 65535 ? rows : 65535);
    if (method == Method::Baseline) {
        const int threads = n < 128 ? 128 : n;
        baseline_kernel<T, Transform, Quantize><<<blocks, threads, 0, stream>>>(
            input, output, packed, scales, rows, n, scale);
    } else {
        const int threads = 128;
        optimized_kernel<T, Transform, Quantize><<<blocks, threads, 0, stream>>>(
            input, output, packed, scales, rows, n, scale);
    }
    return musaGetLastError();
}

}  // namespace

#define DEFINE_TYPED_API(T) \
musaError_t transform(const T* input, T* output, std::size_t rows, int n, float scale, \
                      musaStream_t stream, Method method) { \
    return launch<T, true, false>(input, output, nullptr, nullptr, rows, n, scale, stream, method); \
} \
musaError_t quantize_int4(const T* input, std::uint8_t* packed, float* scales, \
                          std::size_t rows, int n, musaStream_t stream, Method method) { \
    return launch<T, false, true>(input, nullptr, packed, scales, rows, n, 1.0f, stream, method); \
} \
musaError_t transform_int4(const T* input, std::uint8_t* packed, float* scales, \
                           std::size_t rows, int n, float scale, musaStream_t stream, Method method) { \
    return launch<T, true, true>(input, nullptr, packed, scales, rows, n, scale, stream, method); \
}

DEFINE_TYPED_API(__half)
DEFINE_TYPED_API(__mt_bfloat16)
#undef DEFINE_TYPED_API

}  // namespace hadamard::moore

