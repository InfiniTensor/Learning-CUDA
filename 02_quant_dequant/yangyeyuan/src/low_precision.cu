#include "low_precision.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cmath>
#include <cstdint>

namespace lpq {
namespace {

// 这里的 kernel 完全在普通 CUDA core 上执行，格式只通过位级编码模拟。
// 因而这些常量同时也是 MXFP8/NVFP4 的饱和值和归一化公式的一部分。
constexpr float kE4m3Max = 448.0f;
constexpr float kE2m1Max = 6.0f;
constexpr int kThreads = 256;

// 采用显式实现的 host/device 小工具，避免 host 端测试和 device 端 kernel
// 因为调用到不同的数学语义。
__host__ __device__ inline float abs_value(float value) {
  return value < 0.0f ? -value : value;
}

__host__ __device__ inline float power_of_two(int exponent) {
#ifdef __CUDA_ARCH__
  return ldexpf(1.0f, exponent);
#else
  return std::ldexp(1.0f, exponent);
#endif
}

__host__ __device__ inline bool value_is_nan(float value) {
#ifdef __CUDA_ARCH__
  return isnan(value);
#else
  return std::isnan(value);
#endif
}

__host__ __device__ inline bool value_is_finite(float value) {
#ifdef __CUDA_ARCH__
  return isfinite(value);
#else
  return std::isfinite(value);
#endif
}

__host__ __device__ inline float quiet_nan() {
  union {
    std::uint32_t bits;
    float value;
  } result{0x7fc00000U};
  return result.value;
}

// 解码 E4M3 的正数部分；符号位由调用方单独处理。0x7f 被保留给 NaN，
// 因此这里只接收 0x00 到 0x7e 的 magnitude code。
__host__ __device__ float decode_positive_e4m3(std::uint8_t code) {
  const int exponent = (code >> 3) & 0x0f;
  const int mantissa = code & 0x07;
  if (exponent == 0) {
    return static_cast<float>(mantissa) * power_of_two(-9);
  }
  return (1.0f + static_cast<float>(mantissa) * 0.125f) *
         power_of_two(exponent - 7);
}

// 为随机舍入生成与元素下标绑定的伪随机 bit。使用整数 hash 可以避免维护
// device RNG 状态，并保证相同 seed、相同线性下标得到相同结果。
__device__ inline std::uint32_t mix_bits(std::uint64_t seed,
                                         std::uint64_t index) {
  std::uint64_t z = seed + index * 0x9e3779b97f4a7c15ULL;
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  return static_cast<std::uint32_t>((z ^ (z >> 31)) >> 32);
}

__device__ inline float uniform01(std::uint64_t seed, std::uint64_t index) {
  return static_cast<float>(mix_bits(seed, index)) * (1.0f / 4294967296.0f);
}

// E4M3 最近邻量化：先在单调的正数 codebook 上二分查找，再按距离选择。
// 距离相等时选择低位为 0 的 code，实现 round-to-nearest-even。
__host__ __device__ std::uint8_t encode_e4m3_nearest(float value) {
  if (value_is_nan(value)) return 0x7f;
  const bool negative = value < 0.0f;
  float magnitude = abs_value(value);
  if (!value_is_finite(magnitude) || magnitude >= kE4m3Max) {
    return static_cast<std::uint8_t>((negative ? 0x80 : 0) | 0x7e);
  }
  if (magnitude == 0.0f) return negative ? 0x80 : 0;

  int lo = 0;
  int hi = 126;
  while (lo + 1 < hi) {
    const int mid = (lo + hi) / 2;
    if (decode_positive_e4m3(static_cast<std::uint8_t>(mid)) <= magnitude)
      lo = mid;
    else
      hi = mid;
  }
  const float low = decode_positive_e4m3(static_cast<std::uint8_t>(lo));
  const float high = decode_positive_e4m3(static_cast<std::uint8_t>(hi));
  const float low_error = magnitude - low;
  const float high_error = high - magnitude;
  int selected = lo;
  if (high_error < low_error ||
      (high_error == low_error && (lo & 1) != 0)) {
    selected = hi;
  }
  return static_cast<std::uint8_t>((negative ? 0x80 : 0) | selected);
}

// E4M3 的随机舍入仍先找出相邻 codepoint，再按线性插值概率选择上界。
// NaN/Inf 和格式边界必须走确定性的饱和路径。
__device__ std::uint8_t encode_e4m3(float value, Rounding rounding,
                                    std::uint64_t seed,
                                    std::uint64_t index) {
  if (rounding == Rounding::kNearest || !isfinite(value)) {
    return encode_e4m3_nearest(value);
  }
  const bool negative = value < 0.0f;
  const float magnitude = abs_value(value);
  if (magnitude == 0.0f) return negative ? 0x80 : 0;
  if (magnitude >= kE4m3Max) {
    return static_cast<std::uint8_t>((negative ? 0x80 : 0) | 0x7e);
  }
  int lo = 0;
  int hi = 126;
  while (lo + 1 < hi) {
    const int mid = (lo + hi) / 2;
    if (decode_positive_e4m3(static_cast<std::uint8_t>(mid)) <= magnitude)
      lo = mid;
    else
      hi = mid;
  }
  const float low = decode_positive_e4m3(static_cast<std::uint8_t>(lo));
  const float high = decode_positive_e4m3(static_cast<std::uint8_t>(hi));
  const float probability_high = (magnitude - low) / (high - low);
  const int selected = uniform01(seed, index) < probability_high ? hi : lo;
  return static_cast<std::uint8_t>((negative ? 0x80 : 0) | selected);
}

// E8M0 只有指数没有尾数，因此 scale 必须向上取整到不小于目标值的 2 的
// 整数次幂；0 表示全零分组，255 保留为 NaN 编码。
__host__ __device__ std::uint8_t encode_e8m0_ceil(float scale) {
  if (!(scale > 0.0f)) return 0;
  if (!value_is_finite(scale)) return 254;
#ifdef __CUDA_ARCH__
  int exponent = static_cast<int>(ceilf(log2f(scale)));
#else
  int exponent = static_cast<int>(std::ceil(std::log2(scale)));
#endif
  exponent = exponent < -127 ? -127 : exponent;
  exponent = exponent > 127 ? 127 : exponent;
  return static_cast<std::uint8_t>(exponent + 127);
}

// E2M1 没有连续的浮点 codebook，先查找相邻的 8 个正值，再附加 sign bit。
// NVFP4 的 packed nibble 只保留返回值的低 4 bit。
__device__ std::uint8_t encode_e2m1(float value, Rounding rounding,
                                    std::uint64_t seed,
                                    std::uint64_t index) {
  constexpr float codebook[8] = {0.0f, 0.5f, 1.0f, 1.5f,
                                 2.0f, 3.0f, 4.0f, 6.0f};
  const bool negative = value < 0.0f;
  float magnitude = abs_value(value);
  if (!isfinite(magnitude) || magnitude >= kE2m1Max) {
    return static_cast<std::uint8_t>((negative ? 8 : 0) | 7);
  }
  int hi = 1;
  while (hi < 8 && codebook[hi] < magnitude) ++hi;
  if (hi == 8) hi = 7;
  const int lo = hi == 0 ? 0 : hi - 1;
  int selected = lo;
  if (rounding == Rounding::kStochastic && codebook[hi] != codebook[lo]) {
    const float p = (magnitude - codebook[lo]) /
                    (codebook[hi] - codebook[lo]);
    selected = uniform01(seed, index) < p ? hi : lo;
  } else {
    const float low_error = magnitude - codebook[lo];
    const float high_error = codebook[hi] - magnitude;
    if (high_error < low_error ||
        (high_error == low_error && (lo & 1) != 0)) {
      selected = hi;
    }
  }
  return static_cast<std::uint8_t>((negative ? 8 : 0) | selected);
}

// 统一把 FP32/FP16 输入转换成 FP32 参与缩放和编码；FP16 需要显式调用 CUDA
// 的转换函数，避免依赖隐式的 device 类型转换。
template <typename T>
__device__ inline float load_input(const T* input, std::size_t index) {
  return static_cast<float>(input[index]);
}

template <>
__device__ inline float load_input<__half>(const __half* input,
                                           std::size_t index) {
  return __half2float(input[index]);
}

template <typename Input>
__global__ void tensor_amax_kernel(const Input* input, std::size_t count,
                                   float* output) {
  // 先做每线程局部最大值，再用 warp shuffle 和一个 global atomic 合并。
  // CLI 在进入 API 前已拒绝 NaN/Inf；这里仍保留有限性判断作为边界保护。
  float local = 0.0f;
  for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
       i += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
    const float value = abs_value(load_input(input, i));
    if (isfinite(value)) local = fmaxf(local, value);
  }
  for (int offset = 16; offset > 0; offset /= 2)
    local = fmaxf(local, __shfl_down_sync(0xffffffff, local, offset));
  if ((threadIdx.x & 31) == 0) atomicMax(reinterpret_cast<unsigned int*>(output),
                                         __float_as_uint(local));
}

__global__ void finish_global_scale_kernel(float* global_scale,
                                           const float* amax,
                                           Format format) {
  // NVFP4 需要两级 scale：全局 scale 把 E2M1 的最大值映射到 E4M3 的范围。
  // MXFP8 没有第二级 scale，统一写 1 以简化后续 kernel 的参数接口。
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  *global_scale = format == Format::kNvfp4 && *amax > 0.0f
                      ? *amax / (kE4m3Max * kE2m1Max)
                      : 1.0f;
}

__global__ void finish_mx_tensor_scale_kernel(std::uint8_t* scales,
                                              const float* amax) {
  // tensor 模式的 MXFP8 只有一个 E8M0 局部 scale。
  if (threadIdx.x == 0 && blockIdx.x == 0)
    scales[0] = encode_e8m0_ceil(*amax / kE4m3Max);
}

template <typename Input>
__global__ void mx_block_scale_kernel(const Input* input,
                                      std::uint8_t* scales,
                                      std::size_t count, int block_size) {
  // 一个 CUDA block 负责一个逻辑量化分组，尾 block 只遍历实际存在的元素。
  // 归约结果直接编码为 E8M0 byte，后面的量化 kernel 不再访问 FP32 scale。
  const std::size_t begin = static_cast<std::size_t>(blockIdx.x) * block_size;
  float local = 0.0f;
  for (int j = threadIdx.x; j < block_size && begin + j < count;
       j += blockDim.x) {
    const float value = abs_value(load_input(input, begin + j));
    if (isfinite(value)) local = fmaxf(local, value);
  }
  __shared__ float reduction[kThreads];
  reduction[threadIdx.x] = local;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride)
      reduction[threadIdx.x] =
          fmaxf(reduction[threadIdx.x], reduction[threadIdx.x + stride]);
    __syncthreads();
  }
  if (threadIdx.x == 0)
    scales[blockIdx.x] = encode_e8m0_ceil(reduction[0] / kE4m3Max);
}

template <typename Input>
__global__ void nv_block_scale_kernel(const Input* input,
                                      std::uint8_t* scales,
                                      const float* global_scale,
                                      std::size_t count, int block_size,
                                      bool tensor_mode) {
  // NVFP4 局部 scale 先按 E2M1 的最大值归一化，再量化为 E4M3。
  // tensor_mode 复用同一个 kernel，只发射一个 block 并令它覆盖全张量。
  const std::size_t begin = tensor_mode
                                ? 0
                                : static_cast<std::size_t>(blockIdx.x) *
                                      block_size;
  const std::size_t length = tensor_mode ? count : block_size;
  float local = 0.0f;
  for (std::size_t j = threadIdx.x; j < length && begin + j < count;
       j += blockDim.x) {
    const float value = abs_value(load_input(input, begin + j));
    if (isfinite(value)) local = fmaxf(local, value);
  }
  __shared__ float reduction[kThreads];
  reduction[threadIdx.x] = local;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride)
      reduction[threadIdx.x] =
          fmaxf(reduction[threadIdx.x], reduction[threadIdx.x + stride]);
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    const float global = *global_scale;
    const float wanted = global > 0.0f
                             ? (reduction[0] / kE2m1Max) / global
                             : 0.0f;
    scales[blockIdx.x] = encode_e4m3_nearest(wanted);
  }
}

template <typename Input>
__global__ void mx_quantize_kernel(const Input* input, std::uint8_t* packed,
                                   const std::uint8_t* scales,
                                   std::size_t count, int block_size,
                                   bool tensor_mode, Rounding rounding,
                                   std::uint64_t seed) {
  // 每个线程处理一个元素：读取对应 E8M0 scale，编码成一个完整的 E4M3 byte。
  for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
       i += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
    const std::size_t scale_index = tensor_mode ? 0 : i / block_size;
    const float scale = decode_e8m0(scales[scale_index]);
    const float value = load_input(input, i) / scale;
    packed[i] = encode_e4m3(value, rounding, seed, i);
  }
}

template <typename Input>
__global__ void nv_quantize_kernel(const Input* input, std::uint8_t* packed,
                                   const std::uint8_t* scales,
                                   const float* global_scale,
                                   std::size_t count, int block_size,
                                   bool tensor_mode, Rounding rounding,
                                   std::uint64_t seed) {
  // 每个线程负责一个输出 byte，同时量化两个元素，因此不会发生 nibble 写入
  // 竞争。奇数元素个数时 high 保持 0，形成文件格式规定的尾 nibble。
  const std::size_t byte_count = (count + 1) / 2;
  for (std::size_t byte = blockIdx.x * blockDim.x + threadIdx.x;
       byte < byte_count;
       byte += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
    const std::size_t first = byte * 2;
    const std::size_t first_scale = tensor_mode ? 0 : first / block_size;
    const float denominator =
        decode_e4m3(scales[first_scale]) * (*global_scale);
    const float low_value = denominator > 0.0f
                                ? load_input(input, first) / denominator
                                : 0.0f;
    const std::uint8_t low =
        encode_e2m1(low_value, rounding, seed, first) & 0x0f;
    std::uint8_t high = 0;
    if (first + 1 < count) {
      const std::size_t second_scale =
          tensor_mode ? 0 : (first + 1) / block_size;
      const float second_denominator =
          decode_e4m3(scales[second_scale]) * (*global_scale);
      const float high_value = second_denominator > 0.0f
                                   ? load_input(input, first + 1) /
                                         second_denominator
                                   : 0.0f;
      high = encode_e2m1(high_value, rounding, seed, first + 1) & 0x0f;
    }
    packed[byte] = static_cast<std::uint8_t>(low | (high << 4));
  }
}

template <typename Output>
__device__ inline void store_output(Output* output, std::size_t index,
                                    float value) {
  // 通用路径覆盖 FP32；FP16/BF16 在下面特化为 CUDA 提供的舍入转换。
  output[index] = static_cast<Output>(value);
}

template <>
__device__ inline void store_output<__half>(__half* output,
                                             std::size_t index, float value) {
  output[index] = __float2half_rn(value);
}

template <>
__device__ inline void store_output<__nv_bfloat16>(__nv_bfloat16* output,
                                                    std::size_t index,
                                                    float value) {
  output[index] = __float2bfloat16_rn(value);
}

template <typename Output>
__global__ void dequantize_kernel(const std::uint8_t* packed,
                                  const std::uint8_t* scales,
                                  const float* global_scale, Output* output,
                                  std::size_t count, Format format,
                                  int block_size, bool tensor_mode) {
  // 反量化是连续的 grid-stride 读写。NVFP4 先按元素下标提取 nibble，再乘
  // 局部 E4M3 scale 和全局 FP32 scale，最后转换成用户指定的输出类型。
  for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
       i += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
    const std::size_t scale_index = tensor_mode ? 0 : i / block_size;
    float value;
    if (format == Format::kMxfp8) {
      value = decode_e4m3(packed[i]) * decode_e8m0(scales[scale_index]);
    } else {
      const std::uint8_t byte = packed[i / 2];
      const std::uint8_t nibble =
          (i & 1) == 0 ? (byte & 0x0f) : ((byte >> 4) & 0x0f);
      value = decode_e2m1(nibble) * decode_e4m3(scales[scale_index]) *
              (*global_scale);
    }
    store_output(output, i, value);
  }
}

inline int launch_blocks(std::size_t work) {
  // 限制 grid 大小，依靠 kernel 内部的 grid-stride loop 覆盖超大张量；空工作
  // 量也至少发射一个 block，避免非法的零维 grid。
  const std::size_t blocks = (work + kThreads - 1) / kThreads;
  return static_cast<int>(blocks > 4096 ? 4096 : (blocks == 0 ? 1 : blocks));
}

template <typename Input>
cudaError_t launch_quantize_typed(const Input* input, std::uint8_t* packed,
                                  std::uint8_t* scales, float* global_scale,
                                  float* scratch_amax, std::size_t count,
                                  const QuantConfig& config,
                                  cudaStream_t stream) {
  const bool tensor_mode = config.scale_mode == ScaleMode::kTensor;
  // 量化分三段：计算 amax/scale、编码元素、把所有 kernel 排入同一 stream。
  // block scale 可以直接在各自分组内求 amax；tensor scale 则先做全张量归约。
  if (config.format == Format::kMxfp8 && !tensor_mode) {
    const auto blocks = static_cast<unsigned int>(
        (count + config.block_size - 1) / config.block_size);
    mx_block_scale_kernel<<<blocks, kThreads, 0, stream>>>(
        input, scales, count, config.block_size);
    finish_global_scale_kernel<<<1, 1, 0, stream>>>(global_scale, scratch_amax,
                                                    Format::kMxfp8);
  } else {
    cudaError_t status = cudaMemsetAsync(scratch_amax, 0, sizeof(float), stream);
    if (status != cudaSuccess) return status;
    tensor_amax_kernel<<<launch_blocks(count), kThreads, 0, stream>>>(
        input, count, scratch_amax);
    finish_global_scale_kernel<<<1, 1, 0, stream>>>(
        global_scale, scratch_amax, config.format);
    if (config.format == Format::kMxfp8) {
      finish_mx_tensor_scale_kernel<<<1, 1, 0, stream>>>(scales,
                                                         scratch_amax);
    } else {
      const auto blocks = tensor_mode
                              ? 1U
                              : static_cast<unsigned int>(
                                    (count + config.block_size - 1) /
                                    config.block_size);
      nv_block_scale_kernel<<<blocks, kThreads, 0, stream>>>(
          input, scales, global_scale, count, config.block_size, tensor_mode);
    }
  }
  if (config.format == Format::kMxfp8) {
    mx_quantize_kernel<<<launch_blocks(count), kThreads, 0, stream>>>(
        input, packed, scales, count, config.block_size, tensor_mode,
        config.rounding, config.stochastic_seed);
  } else {
    nv_quantize_kernel<<<launch_blocks((count + 1) / 2), kThreads, 0, stream>>>(
        input, packed, scales, global_scale, count, config.block_size,
        tensor_mode, config.rounding, config.stochastic_seed);
  }
  return cudaGetLastError();
}

template <typename Output>
cudaError_t launch_dequantize_typed(const std::uint8_t* packed,
                                    const std::uint8_t* scales,
                                    const float* global_scale, Output* output,
                                    std::size_t count,
                                    const QuantConfig& config,
                                    cudaStream_t stream) {
  // 反量化只有一个 kernel，调用方可以通过同一 stream 与其他 CUDA 工作重叠。
  dequantize_kernel<<<launch_blocks(count), kThreads, 0, stream>>>(
      packed, scales, global_scale, output, count, config.format,
      config.block_size, config.scale_mode == ScaleMode::kTensor);
  return cudaGetLastError();
}

}  // namespace

__host__ __device__ float decode_e4m3(std::uint8_t bits) {
  // E4M3 的最高位是 sign，0x7f/0xff 是 NaN，其余 magnitude 交给正数解码器。
  const bool negative = (bits & 0x80) != 0;
  const std::uint8_t magnitude = bits & 0x7f;
  if (magnitude == 0x7f) return quiet_nan();
  const float value = decode_positive_e4m3(magnitude);
  return negative ? -value : value;
}

__host__ __device__ float decode_e8m0(std::uint8_t bits) {
  // E8M0 的 byte 是带 bias=127 的指数；255 不参与普通 scale 运算。
  if (bits == 255) return quiet_nan();
  return power_of_two(static_cast<int>(bits) - 127);
}

__host__ __device__ float decode_e2m1(std::uint8_t bits) {
  // 低 3 bit 索引固定 codebook，bit 3 表示符号。
  constexpr float codebook[8] = {0.0f, 0.5f, 1.0f, 1.5f,
                                 2.0f, 3.0f, 4.0f, 6.0f};
  const float value = codebook[bits & 0x07];
  return (bits & 0x08) != 0 ? -value : value;
}

std::size_t packed_bytes(std::size_t count, Format format) {
  // NVFP4 两个 4-bit code 共占一个 byte，奇数 count 向上取整。
  return format == Format::kNvfp4 ? (count + 1) / 2 : count;
}

std::size_t scale_count(std::size_t count, const QuantConfig& config) {
  // tensor 模式的 scale 数量与格式无关；block 模式需要覆盖尾分组。
  if (config.scale_mode == ScaleMode::kTensor) return 1;
  return (count + static_cast<std::size_t>(config.block_size) - 1) /
         static_cast<std::size_t>(config.block_size);
}

std::size_t output_type_size(OutputType type) {
  // CUDA 的 __half 与 __nv_bfloat16 都是 16 bit 存储。
  return type == OutputType::kFp32 ? sizeof(float) : sizeof(std::uint16_t);
}

#ifdef LPQ_WITH_HW_PATH
cudaError_t launch_mx_element_quantize(const float* input,
                                       std::uint8_t* packed,
                                       const std::uint8_t* scales,
                                       std::size_t count, int block_size,
                                       bool tensor_mode, cudaStream_t stream) {
  // 与 launch_quantize 内部的 MXFP8 编码段是同一个 kernel，固定 nearest
  // 舍入；单独暴露只是为了和硬件 cvt 编码 kernel 做元素级吞吐对照。
  if (count == 0 || block_size <= 0 || input == nullptr || packed == nullptr ||
      scales == nullptr)
    return cudaErrorInvalidValue;
  mx_quantize_kernel<<<launch_blocks(count), kThreads, 0, stream>>>(
      input, packed, scales, count, block_size, tensor_mode,
      Rounding::kNearest, 0);
  return cudaGetLastError();
}
#endif

cudaError_t launch_quantize(const void* input, InputType input_type,
                            std::uint8_t* packed, std::uint8_t* scales,
                            float* global_scale, float* scratch_amax,
                            std::size_t count, const QuantConfig& config,
                            cudaStream_t stream) {
  // 公共入口只负责检查指针和分派输入类型；具体 kernel 保持模板化，避免在
  // device 端为 FP16/FP32 维护两套几乎相同的实现。
  if (count == 0 || config.block_size <= 0 || input == nullptr ||
      packed == nullptr || scales == nullptr || global_scale == nullptr ||
      scratch_amax == nullptr)
    return cudaErrorInvalidValue;
  if (input_type == InputType::kFp32)
    return launch_quantize_typed(static_cast<const float*>(input), packed,
                                 scales, global_scale, scratch_amax, count,
                                 config, stream);
  return launch_quantize_typed(static_cast<const __half*>(input), packed,
                               scales, global_scale, scratch_amax, count,
                               config, stream);
}

cudaError_t launch_dequantize(const std::uint8_t* packed,
                              const std::uint8_t* scales,
                              const float* global_scale, void* output,
                              OutputType output_type, std::size_t count,
                              const QuantConfig& config,
                              cudaStream_t stream) {
  // 输出类型只影响最终 store；量化格式和 scale 的解释由同一个 dequantize
  // kernel 处理。
  if (count == 0 || config.block_size <= 0 || packed == nullptr ||
      scales == nullptr || global_scale == nullptr || output == nullptr)
    return cudaErrorInvalidValue;
  if (output_type == OutputType::kFp32)
    return launch_dequantize_typed(packed, scales, global_scale,
                                   static_cast<float*>(output), count, config,
                                   stream);
  if (output_type == OutputType::kFp16)
    return launch_dequantize_typed(packed, scales, global_scale,
                                   static_cast<__half*>(output), count, config,
                                   stream);
  return launch_dequantize_typed(packed, scales, global_scale,
                                 static_cast<__nv_bfloat16*>(output), count,
                                 config, stream);
}

}  // namespace lpq
