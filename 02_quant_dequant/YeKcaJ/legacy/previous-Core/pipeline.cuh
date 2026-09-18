#pragma once

#include "codec.cuh"
#include "output.cuh"
#include "pipeline_io.h"
#include "tests/reference.h"

#include <chrono>
#include <random>

namespace pipeline {

// 本文件供两种格式共用：nvfp4 在编译时选择 MXFP8 或 NVFP4。
// 注意区分量化分组（MXFP8 32 个元素 / NVFP4 16 个元素）与 CUDA 线程块。
// 阅读顺序：编码辅助函数 -> 显存和计时 -> GPU kernels -> 启动流程 -> CPU 校验。

// ===== 1. 元素编码辅助函数（不是 kernel） =====

// 按 seed 和元素下标产生随机数，与线程调度无关；固定 seed 可逐字节重现。
__host__ __device__ inline double random_unit(std::uint32_t seed, std::uint64_t index) {
  std::uint32_t x = seed ^ static_cast<std::uint32_t>(index) ^
                    static_cast<std::uint32_t>(index >> 32) * 0x9e3779b9u;
  x += 0x9e3779b9u;
  x ^= x >> 16;
  x *= 0x85ebca6bu;
  x ^= x >> 13;
  x *= 0xc2b2ae35u;
  x ^= x >> 16;
  return (static_cast<double>(x) + 0.5) / 4294967296.0;
}

// E2M1 的低 3 位表示幅值，符号由调用方处理。
__host__ __device__ inline float magnitude2(unsigned code) {
  const float values[8] = {0, .5f, 1, 1.5f, 2, 3, 4, 6};
  return values[code];
}

__device__ inline float magnitude(unsigned code, bool four) {
  return four ? magnitude2(code) : decode_e4m3(code);
}

// E4M3 nearest 直接截取 FP32 指数/尾数；中点严格向较小幅值舍入，不是 nearest-even。
// 两种格式共用：MXFP8 的元素编码，以及 NVFP4 的 E4M3 block scale 编码。
__device__ inline std::uint8_t encode_e4m3_nearest(float value) {
  const float x = fminf(fabsf(value), 448.0f);
  const unsigned sign = __float_as_uint(value) >> 24 & 0x80u;
  if (x <= 0x1p-10f) return 0;
  if (x >= 448.0f) return 126u | sign;

  const unsigned bits = __float_as_uint(x);
  const unsigned exponent = bits >> 23;
  const unsigned mantissa = bits & 0x7fffffu;
  unsigned code;
  if (exponent < 121) {
    // E4M3 subnormal 间距固定为 2^-9；此处分支的 shift 范围为 21..24。
    const unsigned significand = mantissa | 0x800000u;
    const unsigned shift = 141 - exponent;
    code = (significand >> shift) +
           ((significand & ((1u << shift) - 1)) > (1u << (shift - 1)));
  } else {
    // FP32 保留最高 3 位尾数；舍入进位自然进入下一指数区间。
    code = ((exponent - 120) << 3) + (mantissa >> 20) +
           ((mantissa & 0xfffffu) > 0x80000u);
  }
  return code | sign;
}

// four=true 编码 E2M1，否则 E4M3；baseline 保留枚举，nearest 用直接编码。
// 随机舍入仍使用原来的二分路径；NVFP4 的 E4M3 scale 已在 scale_code() 中
// 改走 encode_e4m3_nearest()，故此处 four=true 的 E2M1 分支才是 NVFP4 编码路径。
__device__ inline std::uint8_t encode(float value, bool four, bool stochastic,
                                    std::uint32_t seed, std::size_t index,
                                    bool baseline = false) {
  if (baseline && !stochastic) {
#ifdef TEST_NVFP4
    if (four) return encode_e2m1(value);
#endif
    return encode_e4m3(value);
  }

  if (!nvfp4 && !four && !stochastic) return encode_e4m3_nearest(value);

  const unsigned last = four ? 7 : 126;
  const unsigned sign = signbit(value) ? (four ? 8 : 128) : 0;
  const float x = fminf(fabsf(value), four ? 6.0f : 448.0f);
  if (x == 0) return four ? sign : 0;

  // E2M1 只有 8 个幅值，固定中点比较比二分更快；严格 > 保留中点下舍入。
  if (four && !stochastic) {
    const unsigned code = (x > .25f) + (x > .75f) + (x > 1.25f) + (x > 1.75f) +
                          (x > 2.5f) + (x > 3.5f) + (x > 5.0f);
    return code | sign;
  }

  unsigned lo = 0, hi = last;
  while (lo + 1 < hi) {
    const unsigned mid = (lo + hi) / 2;
    if (magnitude(mid, four) <= x)
      lo = mid;
    else
      hi = mid;
  }

  const float a = magnitude(lo, four), b = magnitude(hi, four);
  bool upper;
  if (stochastic)
    upper = random_unit(seed, index) < (static_cast<double>(x) - a) / (b - a);
  else
    upper = fabsf(b - x) < fabsf(x - a);
  const unsigned code = upper ? hi : lo;
  // CPU E4M3 枚举先遇到 +0；E2M1 则单独保存符号，包括 -0。
  return code == 0 && !four ? 0 : code | sign;
}

// ===== 2. 显存与 CUDA event 计时 =====

// 对象销毁时释放显存；禁用复制，避免同一指针被重复释放。
template <typename T>
struct DeviceBuffer {
  T* ptr = nullptr;

  explicit DeviceBuffer(std::size_t n) {
    if (n) CUDA_CHECK(cudaMalloc(&ptr, n * sizeof(T)));
  }

  ~DeviceBuffer() {
    if (ptr) cudaFree(ptr);
  }

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

struct Events {
  cudaEvent_t start{}, stop{};

  Events() {
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
  }

  ~Events() {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
  }

  // 统计 f() 中提交的 GPU 工作；量化计时包含整个 kernel 序列。
  template <typename F>
  float measure(F&& f) {
    CUDA_CHECK(cudaEventRecord(start));
    f();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    return ms;
  }
};

// ===== 3. GPU kernels 与 scale 辅助函数 =====

// Kernel 1：全张量绝对值最大值的第一阶段，用于 NVFP4 或 tensor scale 模式。
// 每个 CUDA 线程块固定 256 线程，跨步扫描 FP32 input，再在共享内存中归约。
// 输出 partial[blockIdx.x]：该线程块负责的数据的最大绝对值。
__global__ void maximum(const float* input, float* partial, std::size_t n) {
  __shared__ float s[256];
  float v = 0;
  for (std::size_t i = blockIdx.x * 256ull + threadIdx.x; i < n; i += gridDim.x * 256ull)
    v = fmaxf(v, fabsf(input[i]));
  s[threadIdx.x] = v;
  __syncthreads();

  for (unsigned step = 128; step; step >>= 1) {
    if (threadIdx.x < step)
      s[threadIdx.x] = fmaxf(s[threadIdx.x], s[threadIdx.x + step]);
    __syncthreads();
  }
  if (threadIdx.x == 0) partial[blockIdx.x] = s[0];
}

// Kernel 2：用一个 256 线程块归约 partial，得到全张量最大绝对值 M。
// 输出 state[0]=M；NVFP4 的 state[1]=global_scale=M/(6*448)，其他格式为 1。
// 全零输入使用 global_scale=1；正数下溢时保留最小正 FP32 值，避免除零。
__global__ void finalize_max(const float* partial, float* state, unsigned n) {
  __shared__ float s[256];
  float v = 0;
  for (unsigned i = threadIdx.x; i < n; i += 256) v = fmaxf(v, partial[i]);
  s[threadIdx.x] = v;
  __syncthreads();

  for (unsigned step = 128; step; step >>= 1) {
    if (threadIdx.x < step)
      s[threadIdx.x] = fmaxf(s[threadIdx.x], s[threadIdx.x + step]);
    __syncthreads();
  }
  if (!threadIdx.x) {
    state[0] = s[0];
    state[1] = nvfp4 && s[0] > 0 ? fmaxf(s[0] / 2688.0f, __int_as_float(1)) : 1.0f;
  }
}

// 把分组最大绝对值 m 转成 scale 字节：NVFP4 用 E4M3，MXFP8 用 E8M0。
__device__ inline std::uint8_t scale_code(float m, float global) {
  if (nvfp4) return encode_e4m3_nearest(m > 0 ? m / (6.0f * global) : 0.0f);
  if (m == 0) return 127;
  const float ratio = m / 448.0f;
  if (ratio == 0) return 0;
  return max(0, min(254, static_cast<int>(ceilf(log2f(ratio))) + 127));
}

// Kernel 3：通用/内部对照版 scale 计算，每个 CUDA 线程块固定 32 线程。
// block 模式：一个线程块归约一个量化分组；tensor 模式：直接使用 state[0]。
// 输出 scales 中的编码字节；当前默认 block 路径使用下方 Kernel 4。
__global__ void build_scales(const float* input, const float* state, std::uint8_t* scales,
                            std::size_t n, bool tensor) {
  __shared__ float s[32];
  const unsigned lane = threadIdx.x;
  const auto i = blockIdx.x * static_cast<std::size_t>(kBlockSize) + lane;
  s[lane] = !tensor && lane < kBlockSize && i < n ? fabsf(input[i]) : 0;
  __syncthreads();
  for (unsigned step = 16; step; step >>= 1) {
    if (lane < step) s[lane] = fmaxf(s[lane], s[lane + step]);
    __syncthreads();
  }
  if (!lane)
    scales[blockIdx.x] = scale_code(tensor ? state[0] : s[0], nvfp4 ? state[1] : 1.0f);
}

// 恢复实际乘数：NVFP4 为 global_scale * block_scale，MXFP8 为 2^(字节-127)。
__device__ inline float effective(const std::uint8_t* scales, const float* state,
                                 std::size_t group) {
  return nvfp4 ? state[1] * decode_e4m3(scales[group])
               : scalbnf(1.0f, static_cast<int>(scales[group]) - 127);
}

// Kernel 4：默认 block 模式的 scale 计算，输入 FP32，输出 scales 编码字节。
// 每个 CUDA 线程块 256 线程，处理 8 个 MXFP8 分组或 16 个 NVFP4 分组。
// shuffle 的 width=32/16 使各组独立归约，组内首线程写出一个 scale。
// 尾部线程仍参与 shuffle，避免使用 full mask 时缺失参与者。
__global__ void build_block_scales(const float* input, const float* state,
                                 std::uint8_t* scales, std::size_t n) {
  const std::size_t i = blockIdx.x * 256ull + threadIdx.x;
  float v = i < n ? fabsf(input[i]) : 0;
  for (unsigned step = kBlockSize / 2; step; step >>= 1)
    v = fmaxf(v, __shfl_down_sync(0xffffffffu, v, step, kBlockSize));
  if (threadIdx.x % kBlockSize == 0 && i < n)
    scales[i / kBlockSize] = scale_code(v, nvfp4 ? state[1] : 1.0f);
}

// MXFP8 block+nearest 融合 kernel：一个 warp 处理 32 元素，输入只读取一次。
// 尾部补零并参与所有 shuffle；组首计算 scale 后广播，复用寄存器中的原值编码。
// scale 公式及 FP32 除法保持原样，不引入 fast-math 或倒数近似。
__global__ void mxfp8_quantize_fused_kernel(const float* input, std::uint8_t* data,
                                          std::uint8_t* scales, std::size_t n) {
  const std::size_t i = blockIdx.x * 256ull + threadIdx.x;
  const unsigned lane = threadIdx.x & 31;
  const float value = i < n ? input[i] : 0.0f;
  float maximum = fabsf(value);
  for (unsigned step = 16; step; step >>= 1)
    maximum = fmaxf(maximum, __shfl_down_sync(0xffffffffu, maximum, step));

  unsigned code = 127;
  if (lane == 0 && i < n) {
    code = scale_code(maximum, 1.0f);
    scales[i / 32] = code;
  }
  code = __shfl_sync(0xffffffffu, code, 0);
  if (i < n) {
    const float scale = scalbnf(1.0f, static_cast<int>(code) - 127);
    data[i] = encode_e4m3_nearest(value / scale);
  }
}

// NVFP4 block+nearest 融合 kernel：一个 warp 覆盖 4 个分组（每线程 2 元素）。
// 组内用 width=kBlockSize/2 的 shuffle 归约最大值，组首算 scale 后广播，
// 复用寄存器中的原值编码并打包，input 只读取一次。
// global_scale 由前面的 maximum/finalize_max 提供，此处只做 per-block 工作。
// scale 公式、FP32 除法与 s > 0 的守卫保持与原路径一致。
__global__ void nvfp4_quantize_fused_kernel(const float* input, std::uint8_t* data,
                                           std::uint8_t* scales, const float* state,
                                           std::size_t n) {
  // 每个线程负责一个输出字节，即两个连续元素。
  const std::size_t item = blockIdx.x * 256ull + threadIdx.x;
  const std::size_t i = item * 2;
  const unsigned lane = threadIdx.x & 31;
  constexpr unsigned kLanesPerGroup = kBlockSize / 2;

  const float low_value = i < n ? input[i] : 0.0f;
  const float high_value = i + 1 < n ? input[i + 1] : 0.0f;

  // 组内归约：width 限定使每个 kBlockSize/2 个 lane 独立归约。
  float maximum = fmaxf(fabsf(low_value), fabsf(high_value));
  for (unsigned step = kLanesPerGroup / 2; step; step >>= 1)
    maximum = fmaxf(maximum, __shfl_down_sync(0xffffffffu, maximum, step, kLanesPerGroup));

  // 组首线程写 scale；其余 lane 用广播值。i 是组内第一个元素，i < n 即组有效。
  unsigned code = 0;
  if (lane % kLanesPerGroup == 0 && i < n) {
    code = scale_code(maximum, state[1]);
    scales[i / kBlockSize] = code;
  }
  code = __shfl_sync(0xffffffffu, code, 0, kLanesPerGroup);

  if (i < n) {
    const float scale = state[1] * decode_e4m3(static_cast<std::uint8_t>(code));
    const unsigned low = encode(scale > 0 ? low_value / scale : 0, true, false, 0, 0, false);
    const unsigned high = i + 1 < n
                              ? encode(scale > 0 ? high_value / scale : 0, true, false, 0, 0, false)
                              : 0;
    // 偶数元素放低 4 位、奇数放高 4 位；奇数尾部的 nibble 保持为零。
    data[item] = static_cast<std::uint8_t>(low | (high << 4));
  }
}

// Kernel 5：用已计算的 scale 将 FP32 输入量化，输出低精度 data 字节。
// 固定 256 线程/块：MXFP8 每线程写一个 E4M3 元素；NVFP4 每线程写两个 E2M1 元素。
// NVFP4 偶数元素放低 4 位、奇数元素放高 4 位，避免多个线程竞争同一个字节。
// group 决定使用分组 scale 还是全张量 scale；Baseline 选择内部对照编码算法。
template <bool Baseline>
__global__ void quantize_kernel(const float* input, std::uint8_t* data, const std::uint8_t* scales,
                               const float* state, std::size_t n, std::size_t group,
                               bool stochastic, std::uint32_t seed) {
  const auto item = blockIdx.x * 256ull + threadIdx.x;
  const auto i = nvfp4 ? item * 2 : item;
  if (i >= n) return;

  const float s = effective(scales, state, group == kBlockSize ? i / kBlockSize : 0);
  const auto low = encode(s > 0 ? input[i] / s : 0, nvfp4, stochastic, seed, i, Baseline);
  if (nvfp4) {
    // 每线程独占 packed 字节；奇数尾部的高 nibble 保持为零。
    const auto high = i + 1 < n
                          ? encode(s > 0 ? input[i + 1] / s : 0, true, stochastic,
                                   seed, i + 1, Baseline)
                          : 0;
    data[item] = low | (high << 4);
  } else
    data[item] = low;
}

// Kernel 6：读取低精度 data 和 scale，恢复数值并转换成目标输出类型 T。
// 固定 256 线程/块，每线程输出一个元素；NVFP4 先从 packed 字节提取对应 4 位。
// 先以 FP32 计算 decode(data) * effective_scale，再输出 FP32、FP16 或 BF16。
template <typename T>
__global__ void dequantize_kernel(const std::uint8_t* data, const std::uint8_t* scales, const float* state,
                                 T* output, std::size_t n, std::size_t group) {
  const auto i = blockIdx.x * 256ull + threadIdx.x;
  if (i >= n) return;

  const unsigned code = nvfp4 ? ((data[i / 2] >> (4 * (i % 2))) & 15) : data[i];
  const float value = nvfp4 ? ((code & 8) ? -magnitude2(code & 7) : magnitude2(code & 7))
                            : decode_e4m3(code);
  output[i] = cuda_output::Format<T>::convert(
      value * effective(scales, state, group == kBlockSize ? i / kBlockSize : 0));
}

// ===== 4. 主机端任务管理与 kernel 启动 =====

// 显存及 event 与一次任务同寿命；resident benchmark 在重复间复用，不夹入传输。
struct Workspace {
  // n：元素数；group：每组元素数；groups：scale 个数；partials：全局归约块数。
  std::size_t n, group, groups;
  unsigned partials;
  Options options;
  // scratch 存局部最大值；state[0] 存全局最大值，state[1] 存 global_scale。
  DeviceBuffer<float> input, scratch, state;
  DeviceBuffer<std::uint8_t> data, scales;
  // 按 FP32 预留容量，FP16/BF16 输出也复用这块显存。
  DeviceBuffer<float> output;
  Events events;

  Workspace(std::size_t count, Options o)
      : n(count),
        group(o.tensor ? std::max<std::size_t>(1, n) : o.block),
        groups(ceil_div(n, group)),
        partials(static_cast<unsigned>(std::min<std::size_t>(4096, ceil_div(n, 256)))),
        options(o),
        input(n),
        scratch(partials),
        state(2),
        data(data_size(n)),
        scales(groups),
        output(n) {
    validate(o);
  }

  void upload(const std::vector<float>& x) {
    if (x.size() != n) throw std::runtime_error("workspace input length mismatch");
    if (n) CUDA_CHECK(cudaMemcpy(input.ptr, x.data(), n * 4, cudaMemcpyHostToDevice));
  }

  // 量化入口：全局最大值（按需）-> scale -> 元素编码/打包。
  // 同一默认 stream 按顺序执行，阶段之间不需要把 scale 拷回 CPU。
  void launch_quant(bool baseline = false) {
    if (!n) return;

    if (!nvfp4 && !options.tensor && !options.stochastic && !baseline) {
      mxfp8_quantize_fused_kernel<<<static_cast<unsigned>(ceil_div(n, 256)), 256>>>(
          input.ptr, data.ptr, scales.ptr, n);
      return;
    }

    // NVFP4 block+nearest：全局归约仍是必要前置（global_scale 依赖全张量最大值），
    // 之后的 scale 计算与编码合并为一个 kernel，input 由读 3 遍降为读 2 遍。
    if (nvfp4 && !options.tensor && !options.stochastic && !baseline) {
      maximum<<<partials, 256>>>(input.ptr, scratch.ptr, n);
      finalize_max<<<1, 256>>>(scratch.ptr, state.ptr, partials);
      nvfp4_quantize_fused_kernel<<<static_cast<unsigned>(ceil_div(data_size(n), 256)), 256>>>(
          input.ptr, data.ptr, scales.ptr, state.ptr, n);
      return;
    }

    // MXFP8 block 模式只需局部最大值，跳过全张量归约。
    if (nvfp4 || options.tensor) {
      maximum<<<partials, 256>>>(input.ptr, scratch.ptr, n);
      finalize_max<<<1, 256>>>(scratch.ptr, state.ptr, partials);
    }

    if (options.tensor || baseline)
      build_scales<<<static_cast<unsigned>(groups), 32>>>(
          input.ptr, state.ptr, scales.ptr, n, options.tensor);
    else
      build_block_scales<<<static_cast<unsigned>(ceil_div(n, 256)), 256>>>(
          input.ptr, state.ptr, scales.ptr, n);

    // 按输出字节分配线程，因此 NVFP4 的线程数约为元素数的一半。
    const unsigned grid = static_cast<unsigned>(ceil_div(data_size(n), 256));
    if (baseline)
      quantize_kernel<true><<<grid, 256>>>(
          input.ptr, data.ptr, scales.ptr, state.ptr, n, group,
          options.stochastic, options.seed);
    else
      quantize_kernel<false><<<grid, 256>>>(
          input.ptr, data.ptr, scales.ptr, state.ptr, n, group,
          options.stochastic, options.seed);
  }

  template <typename T>
  void launch_dequant() {
    if (n)
      dequantize_kernel<<<static_cast<unsigned>(ceil_div(n, 256)), 256>>>(
          data.ptr, scales.ptr, state.ptr, reinterpret_cast<T*>(output.ptr), n, group);
  }

  // 下载量化结果；文件写入由 pipeline_main.cu / pipeline_io.h 负责。
  Packed download(const Tensor& t) {
    Packed q{t.rows, t.cols, group, options, 1,
             std::vector<std::uint8_t>(data_size(n)),
             std::vector<std::uint8_t>(groups)};
    if (n) {
      CUDA_CHECK(cudaMemcpy(q.data.data(), data.ptr, q.data.size(), cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(q.scales.data(), scales.ptr, groups, cudaMemcpyDeviceToHost));
      if (nvfp4) CUDA_CHECK(cudaMemcpy(&q.global, state.ptr + 1, 4, cudaMemcpyDeviceToHost));
    }
    return q;
  }

  // 上传已有低精度数据，用于不经过量化、直接从文件反量化的流程。
  void upload(const Packed& q) {
    if (checked_count(q.rows, q.cols) != n || q.group != group)
      throw std::runtime_error("workspace packed mismatch");

    const float s[2] = {0, q.global};
    CUDA_CHECK(cudaMemcpy(state.ptr, s, 8, cudaMemcpyHostToDevice));
    if (n) {
      CUDA_CHECK(cudaMemcpy(data.ptr, q.data.data(), q.data.size(), cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(scales.ptr, q.scales.data(), groups, cudaMemcpyHostToDevice));
    }
  }

  template <typename T>
  std::vector<T> download_output() {
    std::vector<T> y(n);
    if (n) CUDA_CHECK(cudaMemcpy(y.data(), output.ptr, n * sizeof(T), cudaMemcpyDeviceToHost));
    return y;
  }
};

// ===== 5. CPU 对照与结果校验（不是 GPU kernel） =====

// CPU oracle 保留枚举编码；随机舍入用线性查找邻值，不复用 GPU 二分算法。
inline unsigned reference_encode(float v, bool four, const Options& o, std::size_t i) {
  if (!o.stochastic) return four ? cpu_encode2(v) : cpu_encode4(v);
  const auto mag = [&](unsigned code) { return four ? magnitude2(code) : cpu_decode4(code); };
  const unsigned last = four ? 7 : 126, sign = std::signbit(v) ? (four ? 8 : 128) : 0;
  const float x = std::min(std::fabs(v), four ? 6.0f : 448.0f);
  unsigned hi = 0;
  while (hi < last && mag(hi) < x) ++hi;
  unsigned code = hi;
  if (hi && x != mag(hi)) {
    const double p = (static_cast<double>(x) - mag(hi - 1)) / (mag(hi) - mag(hi - 1));
    if (random_unit(o.seed, i) >= p) code = hi - 1;
  }
  return code == 0 && !four ? 0 : code | sign;
}

// 按相同格式规则在 CPU 上生成 packed 数据，用于逐字节校验 GPU 结果。
inline Packed reference_quantize(const Tensor& t, const Options& o) {
  validate(o);
  const auto n = t.values.size(), group = o.tensor ? std::max<std::size_t>(1, n) : o.block;
  Packed q{t.rows, t.cols, group, o, 1,
           std::vector<std::uint8_t>(data_size(n)),
           std::vector<std::uint8_t>(ceil_div(n, group))};

  float all = 0;
  for (float v : t.values) all = std::max(all, std::fabs(v));
  if (nvfp4 && all > 0)
    q.global = std::max(all / 2688.0f, std::numeric_limits<float>::denorm_min());

  for (std::size_t b = 0; b < q.scales.size(); ++b) {
    const auto end = std::min(n, (b + 1) * group);
    float m = 0;
    for (std::size_t i = b * group; i < end; ++i)
      m = std::max(m, std::fabs(t.values[i]));

    float s;
    if (nvfp4) {
      q.scales[b] = cpu_encode4(m > 0 ? m / (6.0f * q.global) : 0);
      s = q.global * cpu_decode4(q.scales[b]);
    } else {
      const float ratio = m / 448.0f;
      const int e = m == 0 ? 127
                    : ratio == 0 ? 0
                                 : std::clamp(static_cast<int>(std::ceil(std::log2(ratio))) + 127,
                                              0, 254);
      q.scales[b] = e;
      s = std::ldexp(1.0f, e - 127);
    }

    for (std::size_t i = b * group; i < end; ++i) {
      const auto code = reference_encode(s > 0 ? t.values[i] / s : 0, nvfp4, o, i);
      if (nvfp4)
        q.data[i / 2] |= code << ((i % 2) * 4);
      else
        q.data[i] = code;
    }
  }
  return q;
}

// CPU 反量化先返回 FP32；调用方再校验 GPU 的 FP32/FP16/BF16 输出。
inline std::vector<float> reference_dequantize(const Packed& q) {
  std::vector<float> y(checked_count(q.rows, q.cols));
  for (std::size_t i = 0; i < y.size(); ++i) {
    const unsigned code = nvfp4 ? (q.data[i / 2] >> ((i % 2) * 4)) & 15 : q.data[i];
    const float v = nvfp4 ? (code & 8 ? -magnitude2(code & 7) : magnitude2(code & 7))
                          : cpu_decode4(code);
    const float s = nvfp4 ? q.global * cpu_decode4(q.scales[i / q.group])
                          : std::ldexp(1.0f, q.scales[i / q.group] - 127);
    y[i] = v * s;
  }
  return y;
}

// 同时检查形状、分组、data/scale 字节及 global_scale 的 FP32 位模式。
inline void compare_packed(const Packed& a, const Packed& b) {
  if (a.rows != b.rows || a.cols != b.cols || a.group != b.group ||
      a.data != b.data || a.scales != b.scales || std::memcmp(&a.global, &b.global, 4))
    throw std::runtime_error("pipeline CPU/GPU packed mismatch");
}

}  // namespace pipeline
