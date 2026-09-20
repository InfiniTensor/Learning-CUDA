#include "hadamard.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <cmath>
#include <cstdint>
#include <type_traits>

namespace hadamard {
namespace {

// 一个 block 固定包含 8 个 warp；每个 warp 独立处理一行。
constexpr int kWarpSize = 32;
constexpr int kWarpsPerBlock = 8;

// 读取时把 FP16/BF16 转成 FP32，避免在蝶形累加过程中反复进行低精度运算。
template <typename T>
__device__ __forceinline__ float load_as_float(const T* ptr) {
  return static_cast<float>(*ptr);
}

template <>
__device__ __forceinline__ float load_as_float<__half>(const __half* ptr) {
  return __half2float(*ptr);
}

template <>
__device__ __forceinline__ float load_as_float<__nv_bfloat16>(
    const __nv_bfloat16* ptr) {
  return __bfloat162float(*ptr);
}

template <typename T>
__device__ __forceinline__ T store_from_float(float value);

template <>
__device__ __forceinline__ __half store_from_float<__half>(float value) {
  return __float2half_rn(value);
}

template <>
__device__ __forceinline__ __nv_bfloat16
store_from_float<__nv_bfloat16>(float value) {
  return __float2bfloat16_rn(value);
}

template <typename T>
__device__ __forceinline__ float round_to_storage(float value) {
  // 融合量化前显式模拟一次写入目标存储类型的舍入，
  // 这样才能与“变换结果落地后再量化”的路径逐字节一致。
  const T rounded = store_from_float<T>(value);
  return load_as_float(&rounded);
}

// 一个轻量的整数混合函数，用于根据 column 和 seed 生成稳定的符号。
__host__ __device__ __forceinline__ std::uint32_t mix32(std::uint32_t x) {
  x ^= x >> 16;
  x *= 0x7feb352du;
  x ^= x >> 15;
  x *= 0x846ca68bu;
  return x ^ (x >> 16);
}

__device__ __forceinline__ float random_sign(int column,
                                              std::uint32_t seed) {
  // seed 为 0 时关闭随机符号；否则同一列在所有 row 上使用相同的 ±1。
  if (seed == 0) return 1.0f;
  return (mix32(static_cast<std::uint32_t>(column) ^ seed) & 1u) ? -1.0f
                                                                 : 1.0f;
}

template <typename T, int D, bool kQuantize, int kMode>
__global__ __launch_bounds__(kWarpsPerBlock * kWarpSize)
void warp_fht_kernel(const T* __restrict__ input, T* __restrict__ output,
                     void* __restrict__ quantized,
                     float* __restrict__ scales, std::size_t rows,
                     std::uint32_t sign_seed) {
  constexpr int kItems = D / kWarpSize;
  const int lane = threadIdx.x & (kWarpSize - 1);
  const int warp_in_block = threadIdx.x / kWarpSize;
  const std::size_t row =
      static_cast<std::size_t>(blockIdx.x) * kWarpsPerBlock + warp_in_block;
  if (row >= rows) return;

  // 每个 lane 负责 column = lane + 32 * item，因此一个 warp 可以覆盖
  // D 个元素：D=64/128/256 时每个线程分别持有 2/4/8 个寄存器值。
  const T* row_input = input + row * D;
  float values[kItems];
#pragma unroll
  for (int item = 0; item < kItems; ++item) {
    const int column = lane + item * kWarpSize;
    values[item] = load_as_float(row_input + column) *
                   random_sign(column, sign_seed);
  }

  // 前五级（stride=1..16）通过 warp shuffle 在 lane 之间交换数据；
  // stride>=32 的级别只需要交换同一线程的寄存器。整个 row 不访问共享内存，
  // 因而只需要一次全局读和一次全局写。
#pragma unroll
  for (int stride = 1; stride < kWarpSize; stride <<= 1) {
#pragma unroll
    for (int item = 0; item < kItems; ++item) {
      const float other = __shfl_xor_sync(0xffffffffu, values[item], stride);
      values[item] = (lane & stride) ? other - values[item]
                                     : values[item] + other;
    }
  }

#pragma unroll
  for (int register_stride = 1; register_stride < kItems;
       register_stride <<= 1) {
#pragma unroll
    for (int base = 0; base < kItems; base += 2 * register_stride) {
#pragma unroll
      for (int j = 0; j < register_stride; ++j) {
        const float lower = values[base + j];
        const float upper = values[base + j + register_stride];
        values[base + j] = lower + upper;
        values[base + j + register_stride] = lower - upper;
      }
    }
  }

  const float kNormalization = rsqrtf(static_cast<float>(D));
#pragma unroll
  for (int item = 0; item < kItems; ++item) values[item] *= kNormalization;

  // 量化路径先舍入到 T，严格匹配显式的 FHT(T output) -> quantize 组合。
  if constexpr (kQuantize) {
#pragma unroll
    for (int item = 0; item < kItems; ++item)
      values[item] = round_to_storage<T>(values[item]);
  }

  if constexpr (!kQuantize) {
    T* row_output = output + row * D;
#pragma unroll
    for (int item = 0; item < kItems; ++item) {
      row_output[lane + item * kWarpSize] =
          store_from_float<T>(values[item]);
    }
  } else {
    // 先在 warp 内求出该 row 的 absmax，再由 lane 0 写入逐行 scale。
    float absmax = 0.0f;
#pragma unroll
    for (int item = 0; item < kItems; ++item) {
      absmax = fmaxf(absmax, fabsf(values[item]));
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      absmax = fmaxf(absmax,
                     __shfl_down_sync(0xffffffffu, absmax, offset));
    }
    absmax = __shfl_sync(0xffffffffu, absmax, 0);
    constexpr float kQmax =
        kMode == 8 ? 127.0f : (kMode == 4 ? 7.0f : 448.0f);
    const float scale = absmax > 0.0f ? absmax / kQmax : 1.0f;
    const float inverse_scale = 1.0f / scale;
    if (lane == 0) scales[row] = scale;

    if constexpr (kMode == 8) {
      // INT8 使用对称范围 [-127, 127]，每个元素占一个字节。
      auto* row_output = static_cast<std::int8_t*>(quantized) + row * D;
#pragma unroll
      for (int item = 0; item < kItems; ++item) {
        const float rounded = roundf(values[item] * inverse_scale);
        const int q = static_cast<int>(fminf(127.0f, fmaxf(-127.0f, rounded)));
        row_output[lane + item * kWarpSize] = static_cast<std::int8_t>(q);
      }
    } else if constexpr (kMode == 4) {
      // INT4 使用 [-7, 7]，相邻两个维度打包到一个 byte。
      auto* row_output = static_cast<std::uint8_t*>(quantized) + row * (D / 2);
#pragma unroll
      for (int item = 0; item < kItems; ++item) {
        const float rounded = roundf(values[item] * inverse_scale);
        const int q = static_cast<int>(fminf(7.0f, fmaxf(-7.0f, rounded)));
        const int neighbor = __shfl_down_sync(0xffffffffu, q, 1);
        if ((lane & 1) == 0) {
          const std::uint8_t low = static_cast<std::uint8_t>(q) & 0x0f;
          const std::uint8_t high =
              (static_cast<std::uint8_t>(neighbor) & 0x0f) << 4;
          row_output[item * 16 + lane / 2] = low | high;
        }
      }
    } else {
      // FP8 使用 E4M3 finite 模式，并通过 row scale 覆盖输入动态范围。
      auto* row_output = static_cast<__nv_fp8_storage_t*>(quantized) + row * D;
#pragma unroll
      for (int item = 0; item < kItems; ++item) {
        row_output[lane + item * kWarpSize] = __nv_cvt_float_to_fp8(
            values[item] * inverse_scale, __NV_SATFINITE, __NV_E4M3);
      }
    }
  }
}

template <typename T, int D>
__global__ void shared_memory_fht_kernel(const T* __restrict__ input,
                                         T* __restrict__ output,
                                         std::size_t rows,
                                         std::uint32_t sign_seed) {
  // 每个 block 只处理一行，shared memory 保存该行的全部 FP32 中间值。
  __shared__ float values[D];
  const std::size_t row = blockIdx.x;
  if (row >= rows) return;
  const int tid = threadIdx.x;

  const int first = tid;
  const int second = tid + D / 2;
  values[first] = load_as_float(input + row * D + first) *
                  random_sign(first, sign_seed);
  values[second] = load_as_float(input + row * D + second) *
                   random_sign(second, sign_seed);
  __syncthreads();

  // 每一级都需要读同步、写同步，因此它是 warp 版本的基线而不是主路径。
  for (int stride = 1; stride < D; stride <<= 1) {
    const int group = tid / stride;
    const int offset = tid - group * stride;
    const int lower_index = group * (2 * stride) + offset;
    const int upper_index = lower_index + stride;
    const float lower = values[lower_index];
    const float upper = values[upper_index];
    __syncthreads();
    values[lower_index] = lower + upper;
    values[upper_index] = lower - upper;
    __syncthreads();
  }

  const float kNormalization = rsqrtf(static_cast<float>(D));
  output[row * D + first] = store_from_float<T>(values[first] * kNormalization);
  output[row * D + second] =
      store_from_float<T>(values[second] * kNormalization);
}

template <typename T, int D, int kMode>
__global__ __launch_bounds__(kWarpsPerBlock * kWarpSize)
void row_quant_kernel(const T* __restrict__ input, void* __restrict__ output,
                      float* __restrict__ scales, std::size_t rows) {
  constexpr int kItems = D / kWarpSize;
  const int lane = threadIdx.x & (kWarpSize - 1);
  const int warp_in_block = threadIdx.x / kWarpSize;
  const std::size_t row =
      static_cast<std::size_t>(blockIdx.x) * kWarpsPerBlock + warp_in_block;
  if (row >= rows) return;

  // 该 kernel 输入已经是存储类型 T，用于单独执行量化并和融合版本比较。
  float values[kItems];
  float absmax = 0.0f;
#pragma unroll
  for (int item = 0; item < kItems; ++item) {
    values[item] = load_as_float(input + row * D + lane + item * kWarpSize);
    absmax = fmaxf(absmax, fabsf(values[item]));
  }
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    absmax = fmaxf(absmax, __shfl_down_sync(0xffffffffu, absmax, offset));
  }
  absmax = __shfl_sync(0xffffffffu, absmax, 0);
  constexpr float kQmax =
      kMode == 8 ? 127.0f : (kMode == 4 ? 7.0f : 448.0f);
  const float scale = absmax > 0.0f ? absmax / kQmax : 1.0f;
  const float inverse_scale = 1.0f / scale;
  if (lane == 0) scales[row] = scale;

  if constexpr (kMode == 8) {
    auto* row_output = static_cast<std::int8_t*>(output) + row * D;
#pragma unroll
    for (int item = 0; item < kItems; ++item) {
      const float rounded = roundf(values[item] * inverse_scale);
      const int q = static_cast<int>(fminf(127.0f, fmaxf(-127.0f, rounded)));
      row_output[lane + item * kWarpSize] = static_cast<std::int8_t>(q);
    }
  } else if constexpr (kMode == 4) {
    auto* row_output = static_cast<std::uint8_t*>(output) + row * (D / 2);
#pragma unroll
    for (int item = 0; item < kItems; ++item) {
      const float rounded = roundf(values[item] * inverse_scale);
      const int q = static_cast<int>(fminf(7.0f, fmaxf(-7.0f, rounded)));
      const int neighbor = __shfl_down_sync(0xffffffffu, q, 1);
      if ((lane & 1) == 0) {
        row_output[item * 16 + lane / 2] =
            (static_cast<std::uint8_t>(q) & 0x0f) |
            ((static_cast<std::uint8_t>(neighbor) & 0x0f) << 4);
      }
    }
  } else {
    auto* row_output = static_cast<__nv_fp8_storage_t*>(output) + row * D;
#pragma unroll
    for (int item = 0; item < kItems; ++item) {
      row_output[lane + item * kWarpSize] = __nv_cvt_float_to_fp8(
          values[item] * inverse_scale, __NV_SATFINITE, __NV_E4M3);
    }
  }
}

template <typename T, bool kBaseline>
cudaError_t dispatch_transform(const T* input, T* output, std::size_t rows,
                               int head_dim, std::uint32_t sign_seed,
                               cudaStream_t stream) {
  if (rows == 0) return cudaSuccess;
  if constexpr (kBaseline) {
    // 基线是一行一个 block，线程数为 D/2，每个线程负责两个元素。
#define LAUNCH_BASELINE(D)                                                     \
  shared_memory_fht_kernel<T, D><<<static_cast<unsigned>(rows), D / 2, 0,      \
                                    stream>>>(input, output, rows, sign_seed)
    switch (head_dim) {
      case 64: LAUNCH_BASELINE(64); break;
      case 128: LAUNCH_BASELINE(128); break;
      case 256: LAUNCH_BASELINE(256); break;
      default: return cudaErrorInvalidValue;
    }
#undef LAUNCH_BASELINE
  } else {
    // 快速路径按每 8 行一个 block 计算 grid，尾部不足 8 行的 warp 会提前返回。
    const unsigned blocks =
        static_cast<unsigned>((rows + kWarpsPerBlock - 1) / kWarpsPerBlock);
#define LAUNCH_FAST(D)                                                        \
  warp_fht_kernel<T, D, false, 0>                                             \
      <<<blocks, kWarpsPerBlock * kWarpSize, 0, stream>>>(                    \
          input, output, nullptr, nullptr, rows, sign_seed)
    switch (head_dim) {
      case 64: LAUNCH_FAST(64); break;
      case 128: LAUNCH_FAST(128); break;
      case 256: LAUNCH_FAST(256); break;
      default: return cudaErrorInvalidValue;
    }
#undef LAUNCH_FAST
  }
  return cudaGetLastError();
}

template <typename T, int kMode>
cudaError_t dispatch_quant(const T* input, void* output, float* scales,
                           std::size_t rows, int head_dim,
                           std::uint32_t sign_seed, cudaStream_t stream) {
  if (rows == 0) return cudaSuccess;
  // kMode=3 表示 FP8，kMode=8 表示 INT8，kMode=4 表示 INT4。
  const unsigned blocks =
      static_cast<unsigned>((rows + kWarpsPerBlock - 1) / kWarpsPerBlock);
#define LAUNCH_QUANT(D)                                                       \
  warp_fht_kernel<T, D, true, kMode>                                          \
      <<<blocks, kWarpsPerBlock * kWarpSize, 0, stream>>>(                    \
          input, nullptr, output, scales, rows, sign_seed)
  switch (head_dim) {
    case 64: LAUNCH_QUANT(64); break;
    case 128: LAUNCH_QUANT(128); break;
    case 256: LAUNCH_QUANT(256); break;
    default: return cudaErrorInvalidValue;
  }
#undef LAUNCH_QUANT
  return cudaGetLastError();
}

template <typename T, int kMode>
cudaError_t dispatch_standalone_quant(const T* input, void* output,
                                      float* scales, std::size_t rows,
                                      int head_dim, cudaStream_t stream) {
  if (rows == 0) return cudaSuccess;
  // 独立量化与融合量化使用相同的量化规则，但输入来自已落地的变换结果。
  const unsigned blocks =
      static_cast<unsigned>((rows + kWarpsPerBlock - 1) / kWarpsPerBlock);
#define LAUNCH_ROW_QUANT(D)                                                   \
  row_quant_kernel<T, D, kMode>                                               \
      <<<blocks, kWarpsPerBlock * kWarpSize, 0, stream>>>(                    \
          input, output, scales, rows)
  switch (head_dim) {
    case 64: LAUNCH_ROW_QUANT(64); break;
    case 128: LAUNCH_ROW_QUANT(128); break;
    case 256: LAUNCH_ROW_QUANT(256); break;
    default: return cudaErrorInvalidValue;
  }
#undef LAUNCH_ROW_QUANT
  return cudaGetLastError();
}

}  // namespace

cudaError_t launch_fast(const void* input, void* output, std::size_t rows,
                        int head_dim, DataType dtype, std::uint32_t sign_seed,
                        cudaStream_t stream) {
  // 对外 API 只负责校验指针并选择 dtype/维度对应的模板实例；kernel 异步执行。
  if (!input || !output) return cudaErrorInvalidDevicePointer;
  if (dtype == DataType::kFloat16) {
    return dispatch_transform<__half, false>(static_cast<const __half*>(input),
                                             static_cast<__half*>(output), rows,
                                             head_dim, sign_seed, stream);
  }
  return dispatch_transform<__nv_bfloat16, false>(
      static_cast<const __nv_bfloat16*>(input),
      static_cast<__nv_bfloat16*>(output), rows, head_dim, sign_seed, stream);
}

cudaError_t launch_baseline(const void* input, void* output, std::size_t rows,
                            int head_dim, DataType dtype,
                            std::uint32_t sign_seed, cudaStream_t stream) {
  if (!input || !output) return cudaErrorInvalidDevicePointer;
  if (dtype == DataType::kFloat16) {
    return dispatch_transform<__half, true>(static_cast<const __half*>(input),
                                            static_cast<__half*>(output), rows,
                                            head_dim, sign_seed, stream);
  }
  return dispatch_transform<__nv_bfloat16, true>(
      static_cast<const __nv_bfloat16*>(input),
      static_cast<__nv_bfloat16*>(output), rows, head_dim, sign_seed, stream);
}

cudaError_t launch_fused_quant(const void* input, void* quantized,
                               float* scales, std::size_t rows, int head_dim,
                               DataType dtype, QuantType quant,
                               std::uint32_t sign_seed, cudaStream_t stream) {
  // 融合路径不提供 QuantType::kNone；只要进入该 API 就必须指定量化格式。
  if (!input || !quantized || !scales) return cudaErrorInvalidDevicePointer;
  if (quant == QuantType::kNone) return cudaErrorInvalidValue;
  if (dtype == DataType::kFloat16) {
    if (quant == QuantType::kFp8E4M3) {
      return dispatch_quant<__half, 3>(static_cast<const __half*>(input),
                                       quantized, scales, rows, head_dim,
                                       sign_seed, stream);
    }
    if (quant == QuantType::kInt8) {
      return dispatch_quant<__half, 8>(static_cast<const __half*>(input),
                                       quantized, scales, rows, head_dim,
                                       sign_seed, stream);
    }
    return dispatch_quant<__half, 4>(static_cast<const __half*>(input),
                                     quantized, scales, rows, head_dim,
                                     sign_seed, stream);
  }
  if (quant == QuantType::kFp8E4M3) {
    return dispatch_quant<__nv_bfloat16, 3>(
        static_cast<const __nv_bfloat16*>(input), quantized, scales, rows,
        head_dim, sign_seed, stream);
  }
  if (quant == QuantType::kInt8) {
    return dispatch_quant<__nv_bfloat16, 8>(
        static_cast<const __nv_bfloat16*>(input), quantized, scales, rows,
        head_dim, sign_seed, stream);
  }
  return dispatch_quant<__nv_bfloat16, 4>(
      static_cast<const __nv_bfloat16*>(input), quantized, scales, rows,
      head_dim, sign_seed, stream);
}

cudaError_t launch_quantize(const void* transformed, void* quantized,
                            float* scales, std::size_t rows, int head_dim,
                            DataType dtype, QuantType quant,
                            cudaStream_t stream) {
  // 该 API 只量化已有的变换结果，不再执行 Hadamard。
  if (!transformed || !quantized || !scales)
    return cudaErrorInvalidDevicePointer;
  if (quant == QuantType::kNone) return cudaErrorInvalidValue;
  if (dtype == DataType::kFloat16) {
    if (quant == QuantType::kFp8E4M3) {
      return dispatch_standalone_quant<__half, 3>(
          static_cast<const __half*>(transformed), quantized, scales, rows,
          head_dim, stream);
    }
    if (quant == QuantType::kInt8) {
      return dispatch_standalone_quant<__half, 8>(
          static_cast<const __half*>(transformed), quantized, scales, rows,
          head_dim, stream);
    }
    return dispatch_standalone_quant<__half, 4>(
        static_cast<const __half*>(transformed), quantized, scales, rows,
        head_dim, stream);
  }
  if (quant == QuantType::kFp8E4M3) {
    return dispatch_standalone_quant<__nv_bfloat16, 3>(
        static_cast<const __nv_bfloat16*>(transformed), quantized, scales,
        rows, head_dim, stream);
  }
  if (quant == QuantType::kInt8) {
    return dispatch_standalone_quant<__nv_bfloat16, 8>(
        static_cast<const __nv_bfloat16*>(transformed), quantized, scales,
        rows, head_dim, stream);
  }
  return dispatch_standalone_quant<__nv_bfloat16, 4>(
      static_cast<const __nv_bfloat16*>(transformed), quantized, scales, rows,
      head_dim, stream);
}

const char* dtype_name(DataType dtype) {
  return dtype == DataType::kFloat16 ? "fp16" : "bf16";
}

const char* quant_name(QuantType quant) {
  switch (quant) {
    case QuantType::kFp8E4M3: return "fp8_e4m3";
    case QuantType::kInt8: return "int8";
    case QuantType::kInt4: return "int4";
    default: return "none";
  }
}

}  // namespace hadamard
