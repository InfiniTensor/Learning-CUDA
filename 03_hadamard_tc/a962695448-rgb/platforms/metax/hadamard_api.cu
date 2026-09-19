// 算法来源：本项目天数后端固定提交 a387db3332c6f9b01f128dd681848260c9691281。
// 沐曦后端必须独立完成正确性与性能验证，不继承其他平台的验收结果。
#include "hadamard_api.h"

#include <cmath>
#include <limits>

namespace hadamard::metax {
namespace {

template<class T> __device__ float read_value(T value);
template<> __device__ float read_value(__half value) { return __half2float(value); }
template<> __device__ float read_value(__nv_bfloat16 value) {
    return __bfloat162float(value);
}
template<class T> __device__ T store_value(float value);
template<> __device__ __half store_value(float value) { return __float2half_rn(value); }
template<> __device__ __nv_bfloat16 store_value(float value) {
    return __float2bfloat16_rn(value);
}

__device__ std::uint8_t quantize_pair(const float* values, int pair, int n,
                                    float row_scale) {
    std::uint8_t byte = 0;
    for (int k = 0; k < 2 && pair * 2 + k < n; ++k) {
        const float x = values[pair * 2 + k] / row_scale;
        // 明确实现最近偶数舍入，不依赖编译器默认取整模式。
        const float lower = floorf(x);
        const float fraction = x - lower;
        int q = static_cast<int>(lower);
        if (fraction > 0.5f || (fraction == 0.5f && q % 2 != 0)) ++q;
        q = q < -7 ? -7 : (q > 7 ? 7 : q);
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
                row_scale = magnitude == 0 ? 1.0f : magnitude / 7.0f;
                scales[row] = row_scale;
            }
            __syncthreads();
            const int bytes = (n + 1) / 2;
            if (i < bytes) packed[row * bytes + i] = quantize_pair(values, i, n, row_scale);
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
                row_scale = maximum == 0 ? 1.0f : maximum / 7.0f;
                scales[row] = row_scale;
            }
            __syncthreads();
            const int bytes = (n + 1) / 2;
            for (int i = tid; i < bytes; i += blockDim.x)
                packed[row * bytes + i] = quantize_pair(values, i, n, row_scale);
        }
        if (rows - row > gridDim.x) __syncthreads();
    }
}

#if defined(HADAMARD_METAX_WARP64) && defined(__MACACC__)

// 仅在沐曦 MACA/cu-bridge 构建中启用。legacy shuffle 的 width=64 已由沐曦实机独立探针验证。
// 这里不使用 NVIDIA 的 32 位 active mask，也不将 32-lane 测试当作正确性证据。
__device__ int warp64_quantized_nibble(float value, float scale) {
    const float x = value / scale;
    const float lower = floorf(x);
    const float fraction = x - lower;
    int q = static_cast<int>(lower);
    if (fraction > 0.5f || (fraction == 0.5f && q % 2 != 0)) ++q;
    q = q < -7 ? -7 : (q > 7 ? 7 : q);
    return q & 15;
}

template<class T, bool Transform, bool Quantize, int N>
__global__ void warp64_kernel(const T* input, T* output, std::uint8_t* packed,
                              float* scales, std::size_t rows, float scale) {
    constexpr int width = 64;
    constexpr int rows_per_block = 4;
    constexpr int registers = N > width ? N / width : 1;
    const int lane = threadIdx.x % width;
    const int warp = threadIdx.x / width;
    const std::size_t first_row = static_cast<std::size_t>(blockIdx.x) * rows_per_block + warp;
    const std::size_t row_stride = static_cast<std::size_t>(gridDim.x) * rows_per_block;
    // 同一 warp 的 64 个 lane 具有相同 row 与循环次数。尾行只跳过完整 warp，
    // 所有参与行均有完整的 64 个 lane 执行每次 shuffle。
    for (std::size_t row = first_row; row < rows; row += row_stride) {
        const std::size_t offset = row * N;
        float values[registers];
        #pragma unroll
        for (int r = 0; r < registers; ++r) {
            const int i = lane + r * width;
            values[r] = i < N ? read_value(input[offset + i]) : 0.0f;
        }
        if constexpr (Transform) {
            #pragma unroll
            for (int stride = 1; stride < (N < width ? N : width); stride *= 2) {
                #pragma unroll
                for (int r = 0; r < registers; ++r) {
                    const float current = values[r];
                    const float peer = __shfl_xor(current, stride, width);
                    values[r] = (lane & stride) ? peer - current : current + peer;
                }
            }
            // stride=64/128 的搭档位于同一 lane 的其他寄存器中，
            // 顺序与标量共享内存 FWHT 的从低位到高位阶段一致。
            if constexpr (N >= 128) {
                const float a = values[0], b = values[1];
                values[0] = a + b;
                values[1] = a - b;
            }
            if constexpr (N == 256) {
                const float c = values[2], d = values[3];
                values[2] = c + d;
                values[3] = c - d;
                const float a = values[0], b = values[2];
                const float e = values[1], f = values[3];
                values[0] = a + b;
                values[2] = a - b;
                values[1] = e + f;
                values[3] = e - f;
            }
        }
        if constexpr (!Quantize) {
            #pragma unroll
            for (int r = 0; r < registers; ++r) {
                const int i = lane + r * width;
                if (i < N) output[offset + i] = store_value<T>(values[r] * scale);
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
                magnitude = fmaxf(magnitude, __shfl_xor(magnitude, stride, width));
            const float row_scale = magnitude == 0.0f ? 1.0f : magnitude / 7.0f;
            if (lane == 0) scales[row] = row_scale;
            #pragma unroll
            for (int r = 0; r < registers; ++r) {
                // shuffle 在分支之前：奇数 lane 同样必须提供其相邻元素。
                const float peer = __shfl_xor(values[r], 1, width);
                const int i = lane + r * width;
                if ((lane & 1) == 0 && i < N) {
                    const int low = warp64_quantized_nibble(values[r], row_scale);
                    const int high = i + 1 < N ? warp64_quantized_nibble(peer, row_scale) : 0;
                    packed[row * ((N + 1) / 2) + i / 2]
                        = static_cast<std::uint8_t>(low | (high << 4));
                }
            }
        }
    }
}

template<class T, bool Transform, bool Quantize>
cudaError_t launch_warp64(const T* input, T* output, std::uint8_t* packed, float* scales,
                          std::size_t rows, int n, float scale, cudaStream_t stream) {
    // rows <= SIZE_MAX/4 已由 launch 校验，因此 (rows+3)/4 不会溢出。
    const std::size_t requested = (rows + 3) / 4;
    const unsigned int blocks = static_cast<unsigned int>(requested < 65535 ? requested : 65535);
    #define LAUNCH_WARP64(N) case N: \
        warp64_kernel<T, Transform, Quantize, N><<<blocks, 256, 0, stream>>>( \
            input, output, packed, scales, rows, scale); \
        break
    switch (n) {
        LAUNCH_WARP64(1);
        LAUNCH_WARP64(2);
        LAUNCH_WARP64(4);
        LAUNCH_WARP64(8);
        LAUNCH_WARP64(16);
        LAUNCH_WARP64(32);
        LAUNCH_WARP64(64);
        LAUNCH_WARP64(128);
        LAUNCH_WARP64(256);
        default: return cudaErrorInvalidValue;
    }
    #undef LAUNCH_WARP64
    return cudaGetLastError();
}

#endif  // HADAMARD_METAX_WARP64

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
cudaError_t launch(const T* input, T* output, std::uint8_t* packed, float* scales,
                   std::size_t rows, int n, float scale, cudaStream_t stream, Method method) {
    if (n < 1 || n > 256 || (n & (n - 1)) != 0
        || !std::isfinite(scale) || scale <= 0
        || (method != Method::Baseline && method != Method::Optimized && method != Method::Warp64))
        return cudaErrorInvalidValue;
    if (rows == 0) return cudaSuccess;

    constexpr auto maximum = std::numeric_limits<std::size_t>::max();
    const std::size_t row_bytes = static_cast<std::size_t>(n) * sizeof(T);
    if (rows > maximum / row_bytes || rows > maximum / sizeof(float))
        return cudaErrorInvalidValue;
    const std::size_t input_bytes = rows * row_bytes;
    if (!valid_range(input, input_bytes, alignof(T))) return cudaErrorInvalidValue;
    if constexpr (Quantize) {
        const std::size_t packed_bytes = rows * static_cast<std::size_t>((n + 1) / 2);
        const std::size_t scale_bytes = rows * sizeof(float);
        if (!valid_range(packed, packed_bytes, alignof(std::uint8_t))
            || !valid_range(scales, scale_bytes, alignof(float))
            || overlaps(input, input_bytes, packed, packed_bytes)
            || overlaps(input, input_bytes, scales, scale_bytes)
            || overlaps(packed, packed_bytes, scales, scale_bytes)) return cudaErrorInvalidValue;
    } else {
        if (!valid_range(output, input_bytes, alignof(T))
            || (input != output && overlaps(input, input_bytes, output, input_bytes)))
            return cudaErrorInvalidValue;
    }

    if (method == Method::Warp64) {
        #if defined(HADAMARD_METAX_WARP64) && defined(__MACACC__)
        return launch_warp64<T, Transform, Quantize>(input, output, packed, scales, rows, n, scale, stream);
        #else
        return cudaErrorNotSupported;
        #endif
    }

    // 保守上限兼容不同设备的 grid.x 限制，超过上限由 block 顺序处理多行。
    const unsigned int blocks = static_cast<unsigned int>(rows < 65535 ? rows : 65535);
    if (method == Method::Baseline) {
        const int threads = n < 64 ? 64 : n;
        baseline_kernel<T, Transform, Quantize><<<blocks, threads, 0, stream>>>(
            input, output, packed, scales, rows, n, scale);
    } else {
        const int threads = n / 2 < 64 ? 64 : n / 2;
        optimized_kernel<T, Transform, Quantize><<<blocks, threads, 0, stream>>>(
            input, output, packed, scales, rows, n, scale);
    }
    return cudaGetLastError();
}

}  // namespace

#define DEFINE_TYPED_API(T) \
cudaError_t transform(const T* input, T* output, std::size_t rows, int n, float scale, \
                      cudaStream_t stream, Method method) { \
    return launch<T, true, false>(input, output, nullptr, nullptr, rows, n, scale, stream, method); \
} \
cudaError_t quantize_int4(const T* input, std::uint8_t* packed, float* scales, \
                          std::size_t rows, int n, cudaStream_t stream, Method method) { \
    return launch<T, false, true>(input, nullptr, packed, scales, rows, n, 1.0f, stream, method); \
} \
cudaError_t transform_int4(const T* input, std::uint8_t* packed, float* scales, \
                           std::size_t rows, int n, float scale, cudaStream_t stream, Method method) { \
    return launch<T, true, true>(input, nullptr, packed, scales, rows, n, scale, stream, method); \
}

DEFINE_TYPED_API(__half)
DEFINE_TYPED_API(__nv_bfloat16)
#undef DEFINE_TYPED_API

}  // namespace hadamard::metax
