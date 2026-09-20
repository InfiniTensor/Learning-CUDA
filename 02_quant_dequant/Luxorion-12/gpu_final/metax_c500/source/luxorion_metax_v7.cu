#include "benchmark.h"
#include "benchmark_helpers.cuh"
#include "io.h"
#include "metrics.h"
#include "mxfp8.h"
#include "nvfp4.h"
#include "quant_config.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace fs = std::filesystem;
namespace {

using luxorion::OutputType;
using luxorion::QuantConfig;
using luxorion::QuantFormat;
using luxorion::RoundingMode;
using luxorion::ScaleMode;
using luxorion::TensorDType;
using luxorion::TimingStats;

constexpr unsigned kReduceThreads = 256;
constexpr unsigned kMaxReduceBlocks = 1024;
#ifndef LUX_DEQUANT_THREADS
#define LUX_DEQUANT_THREADS 128
#endif
constexpr unsigned kDequantThreads = LUX_DEQUANT_THREADS;
static_assert(kDequantThreads == 64 || kDequantThreads == 128 ||
                  kDequantThreads == 256 || kDequantThreads == 512,
              "unsupported dequant threads-per-block candidate");

void check(cudaError_t status, const char *what) {
  if (status != cudaSuccess)
    throw std::runtime_error(std::string(what) + ": " +
                             cudaGetErrorString(status));
}

template <typename T> class DeviceBuffer {
public:
  explicit DeviceBuffer(std::size_t n) {
    check(cudaMalloc(reinterpret_cast<void **>(&p_), n * sizeof(T)),
          "cudaMalloc");
  }
  ~DeviceBuffer() {
    if (p_)
      cudaFree(p_);
  }
  T *get() { return p_; }

private:
  T *p_ = nullptr;
};

struct Quantized {
  std::int64_t rows = 0, cols = 0;
  TensorDType input_dtype = TensorDType::FP32;
  QuantFormat format = QuantFormat::Mxfp8;
  ScaleMode scale_mode = ScaleMode::Block;
  RoundingMode rounding = RoundingMode::Nearest;
  OutputType output_type = OutputType::FP32;
  std::uint32_t block_size = 0;
  std::uint32_t seed = 1234;
  float global_scale = 1.0F;
  std::vector<std::uint8_t> data;
  std::vector<std::uint8_t> scales;
};

__device__ float load_value(float v) { return v; }
__device__ float load_value(__half v) { return __half2float(v); }

__device__ std::uint32_t hash32(std::uint32_t x) {
  x ^= x >> 16;
  x *= 0x7feb352dU;
  x ^= x >> 15;
  x *= 0x846ca68bU;
  return x ^ (x >> 16);
}
__device__ float random01(std::size_t i, std::uint32_t seed) {
  return static_cast<float>(
      (hash32(static_cast<std::uint32_t>(i) ^ seed) >> 8) * (1.0 / 16777216.0));
}
__device__ float e4(std::uint8_t c) {
  unsigned e = (c >> 3) & 15U, m = c & 7U;
  float v = e == 0 ? ldexpf(static_cast<float>(m), -9)
                   : ldexpf(1.0F + static_cast<float>(m) / 8.0F,
                            static_cast<int>(e) - 7);
  return copysignf(v, (c & 0x80U) ? -1.0F : 1.0F);
}
__device__ float e2(std::uint8_t c) {
  const float v[8] = {0, 0.5F, 1, 1.5F, 2, 3, 4, 6};
  return copysignf(v[c & 7U], (c & 8U) ? -1.0F : 1.0F);
}
__device__ float e8(std::uint8_t c) {
  return ldexpf(1.0F, static_cast<int>(c) - 127);
}

__device__ std::uint8_t encode_e4(float value, bool stochastic,
                                  std::size_t index, std::uint32_t seed) {
  unsigned sign = signbit(value) ? 0x80U : 0U;
  double x = fabs(static_cast<double>(value));
  if (x >= 448.0)
    return static_cast<std::uint8_t>(sign | 0x7EU);
  unsigned first = 0, last = 0x7EU;
#pragma unroll 1
  for (unsigned step = 0; step < 7; ++step) {
    const unsigned mid = (first + last) / 2;
    if (static_cast<double>(e4(static_cast<std::uint8_t>(mid))) < x)
      first = mid + 1;
    else
      last = mid;
  }
  const unsigned upper = first;
  const unsigned lower =
      static_cast<double>(e4(static_cast<std::uint8_t>(upper))) == x
          ? upper
          : upper - 1;
  unsigned chosen = lower;
  double lo = e4(static_cast<std::uint8_t>(lower)),
         hi = e4(static_cast<std::uint8_t>(upper));
  if (stochastic && upper != lower)
    chosen = random01(index, seed) < (x - lo) / (hi - lo) ? upper : lower;
  else if (!stochastic) {
    double dl = x - lo, du = hi - x;
    chosen = du < dl || (du == dl && (upper & 1U) == 0U) ? upper : lower;
  }
  return static_cast<std::uint8_t>(sign | chosen);
}
__device__ std::uint8_t encode_e2(float value, bool stochastic,
                                  std::size_t index, std::uint32_t seed) {
  unsigned sign = signbit(value) ? 8U : 0U;
  double x = fabs(static_cast<double>(value));
  if (x >= 6)
    return sign | 7U;
  unsigned first = 0, last = 7;
#pragma unroll 1
  for (unsigned step = 0; step < 3; ++step) {
    const unsigned mid = (first + last) / 2;
    if (static_cast<double>(e2(static_cast<std::uint8_t>(mid))) < x)
      first = mid + 1;
    else
      last = mid;
  }
  const unsigned upper = first;
  const unsigned lower =
      static_cast<double>(e2(static_cast<std::uint8_t>(upper))) == x
          ? upper
          : upper - 1;
  double lo = e2(lower), hi = e2(upper);
  unsigned chosen = lower;
  if (stochastic && upper != lower)
    chosen = random01(index, seed) < (x - lo) / (hi - lo) ? upper : lower;
  else if (!stochastic) {
    double dl = x - lo, du = hi - x;
    chosen = du < dl || (du == dl && (upper & 1U) == 0U) ? upper : lower;
  }
  return static_cast<std::uint8_t>(sign | chosen);
}
__device__ std::uint8_t select_e8(float amax) {
  if (amax == 0)
    return 127;
  for (unsigned c = 0; c <= 254; ++c)
    if (static_cast<double>(amax) <= ldexp(448.0, static_cast<int>(c) - 127))
      return c;
  return 254;
}

template <typename T>
__global__ void amax_partial(const T *x, std::size_t n, float *partial) {
  __shared__ float s[kReduceThreads];
  float v = 0;
  for (std::size_t i =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < n; i += static_cast<std::size_t>(gridDim.x) * blockDim.x)
    v = fmaxf(v, fabsf(load_value(x[i])));
  s[threadIdx.x] = v;
  __syncthreads();
  for (unsigned o = 128; o; o >>= 1) {
    if (threadIdx.x < o)
      s[threadIdx.x] = fmaxf(s[threadIdx.x], s[threadIdx.x + o]);
    __syncthreads();
  }
  if (threadIdx.x == 0)
    partial[blockIdx.x] = s[0];
}
__global__ void finish_amax(const float *p, unsigned n, float *out) {
  __shared__ float s[kReduceThreads];
  float v = 0;
  for (unsigned i = threadIdx.x; i < n; i += blockDim.x)
    v = fmaxf(v, p[i]);
  s[threadIdx.x] = v;
  __syncthreads();
  for (unsigned o = 128; o; o >>= 1) {
    if (threadIdx.x < o)
      s[threadIdx.x] = fmaxf(s[threadIdx.x], s[threadIdx.x + o]);
    __syncthreads();
  }
  if (threadIdx.x == 0)
    *out = s[0];
}

template <typename T>
__global__ void mx_block(const T *x, std::uint8_t *data, std::uint8_t *scales,
                         std::size_t n, bool sr, std::uint32_t seed) {
  __shared__ float s[32];
  __shared__ float scale;
  unsigned lane = threadIdx.x;
  std::size_t i = static_cast<std::size_t>(blockIdx.x) * 32 + lane;
  float v = i < n ? load_value(x[i]) : 0;
  s[lane] = fabsf(v);
  __syncthreads();
  for (unsigned o = 16; o; o >>= 1) {
    if (lane < o)
      s[lane] = fmaxf(s[lane], s[lane + o]);
    __syncthreads();
  }
  if (lane == 0) {
    auto c = select_e8(s[0]);
    scales[blockIdx.x] = c;
    scale = e8(c);
  }
  __syncthreads();
  if (i < n)
    data[i] = encode_e4(v / scale, sr, i, seed);
}
template <typename T>
__global__ void mx_tensor(const T *x, std::uint8_t *data,
                          std::uint8_t *scale_code, const float *amax,
                          std::size_t n, bool sr, std::uint32_t seed) {
  __shared__ float scale;
  if (threadIdx.x == 0) {
    auto c = select_e8(*amax);
    *scale_code = c;
    scale = e8(c);
  }
  __syncthreads();
  for (std::size_t i = threadIdx.x; i < n; i += blockDim.x)
    data[i] = encode_e4(load_value(x[i]) / scale, sr, i, seed);
}
__global__ void nv_global_scale(const float *amax, float *global) {
  if (threadIdx.x == 0)
    *global = *amax == 0 ? 1.0F : *amax / 2688.0F;
}
template <typename T>
__global__ void nv_block(const T *x, std::uint8_t *data, std::uint8_t *scales,
                         const float *global, std::size_t n, bool sr,
                         std::uint32_t seed) {
  __shared__ float s[16];
  // One 32-bit word per lane maps consecutive codes to distinct shared banks.
  __shared__ std::uint32_t c[16];
  __shared__ float scale;
  unsigned lane = threadIdx.x;
  std::size_t i = static_cast<std::size_t>(blockIdx.x) * 16 + lane;
  float v = i < n ? load_value(x[i]) : 0;
  s[lane] = fabsf(v);
  __syncthreads();
  for (unsigned o = 8; o; o >>= 1) {
    if (lane < o)
      s[lane] = fmaxf(s[lane], s[lane + o]);
    __syncthreads();
  }
  if (lane == 0) {
    float ideal = s[0] == 0 ? 1.0F : (s[0] / 6.0F) / (*global);
    auto code = encode_e4(ideal, false, blockIdx.x, seed);
    if (s[0] != 0 && code == 0)
      code = 1;
    scales[blockIdx.x] = code;
    scale = (*global) * e4(code);
  }
  __syncthreads();
  c[lane] = i < n ? encode_e2(v / scale, sr, i, seed) : 0U;
  __syncthreads();
  if ((lane & 1U) == 0)
    data[static_cast<std::size_t>(blockIdx.x) * 8 + lane / 2] =
        static_cast<std::uint8_t>((c[lane] << 4) | c[lane + 1]);
}
template <typename T>
__global__ void nv_tensor(const T *x, std::uint8_t *data, std::uint8_t *local,
                          const float *global, const float *amax, std::size_t n,
                          bool sr, std::uint32_t seed) {
  __shared__ float scale;
  if (threadIdx.x == 0) {
    float ideal = *amax == 0 ? 1.0F : (*amax / 6.0F) / (*global);
    auto c = encode_e4(ideal, false, 0, seed);
    if (*amax != 0 && c == 0)
      c = 1;
    *local = c;
    scale = (*global) * e4(c);
  }
  __syncthreads();
  for (std::size_t pair = threadIdx.x; pair < (n + 1) / 2; pair += blockDim.x) {
    std::size_t i = pair * 2;
    auto a = encode_e2(load_value(x[i]) / scale, sr, i, seed);
    auto b = i + 1 < n
                 ? encode_e2(load_value(x[i + 1]) / scale, sr, i + 1, seed)
                 : 0;
    data[pair] = static_cast<std::uint8_t>((a << 4) | b);
  }
}

__device__ void store(float *p, std::size_t i, float v) { p[i] = v; }
__device__ void store(__half *p, std::size_t i, float v) {
  p[i] = __float2half_rn(v);
}
__device__ void store(__nv_bfloat16 *p, std::size_t i, float v) {
  p[i] = __float2bfloat16_rn(v);
}

__device__ void store_pair(float *p, std::size_t i, float a, float b) {
  *reinterpret_cast<float2 *>(p + i) = make_float2(a, b);
}
__device__ void store_pair(__half *p, std::size_t i, float a, float b) {
  *reinterpret_cast<__half2 *>(p + i) = __floats2half2_rn(a, b);
}
__device__ void store_pair(__nv_bfloat16 *p, std::size_t i, float a, float b) {
  *reinterpret_cast<__nv_bfloat162 *>(p + i) = __floats2bfloat162_rn(a, b);
}

template <typename O>
__global__ void mx_deq(const std::uint8_t *data, const std::uint8_t *scales,
                        O *out, std::size_t n, bool tensor) {
  std::size_t i =
      (static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x) * 4;
  if (i >= n)
    return;

  const float scale = e8(scales[tensor ? 0 : i / 32]);
  if (i + 4 <= n) {
    const auto packed = *reinterpret_cast<const std::uint32_t *>(data + i);
    const float a = e4(static_cast<std::uint8_t>(packed)) * scale;
    const float b = e4(static_cast<std::uint8_t>(packed >> 8)) * scale;
    const float c = e4(static_cast<std::uint8_t>(packed >> 16)) * scale;
    const float d = e4(static_cast<std::uint8_t>(packed >> 24)) * scale;
    store_pair(out, i, a, b);
    store_pair(out, i + 2, c, d);
    return;
  }
  for (; i < n; ++i)
    store(out, i, e4(data[i]) * e8(scales[tensor ? 0 : i / 32]));
}
template <typename O>
__global__ void nv_deq(const std::uint8_t *data, const std::uint8_t *scales,
                        float global, O *out, std::size_t n, bool tensor) {
  std::size_t i =
      (static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x) * 2;
  if (i >= n)
    return;

  const auto packed = data[i / 2];
  const float local_scale = e4(scales[tensor ? 0 : i / 16]);
  const float a =
      e2(static_cast<std::uint8_t>(packed >> 4)) * global * local_scale;
  if (i + 1 < n) {
    const float b =
        e2(static_cast<std::uint8_t>(packed & 15U)) * global * local_scale;
    store_pair(out, i, a, b);
  } else {
    store(out, i, a);
  }
}

template <typename InputT>
Quantized quantize_impl(const void *host, std::size_t n, const QuantConfig &cfg,
                        std::int64_t rows, std::int64_t cols, TensorDType dtype,
                        unsigned warmup, unsigned repeats,
                        TimingStats &timing) {
  Quantized q;
  q.rows = rows;
  q.cols = cols;
  q.input_dtype = dtype;
  q.format = cfg.format;
  q.scale_mode = cfg.scale_mode;
  q.rounding = cfg.rounding;
  q.output_type = cfg.output_type;
  q.block_size = static_cast<std::uint32_t>(cfg.block_size);
  q.seed = cfg.seed;
  bool tensor = cfg.scale_mode == ScaleMode::Tensor,
       sr = cfg.rounding == RoundingMode::Stochastic;
  std::size_t groups =
      tensor ? 1 : (n / cfg.block_size + (n % cfg.block_size != 0));
  q.scales.resize(groups);
  q.data.resize(cfg.format == QuantFormat::Mxfp8
                    ? n
                    : (tensor ? (n + 1) / 2 : groups * 8));
  DeviceBuffer<InputT> dx(n);
  DeviceBuffer<std::uint8_t> dd(q.data.size()), ds(groups);
  check(cudaMemcpy(dx.get(), host, n * sizeof(InputT), cudaMemcpyHostToDevice),
        "input H2D");
  std::size_t rb = std::min<std::size_t>((n + 255) / 256, kMaxReduceBlocks);
  DeviceBuffer<float> partial(rb), amax(1), global(1);
  auto launch = [&] {
    if (cfg.format == QuantFormat::Mxfp8) {
      if (tensor) {
        amax_partial<<<static_cast<unsigned>(rb), 256>>>(dx.get(), n,
                                                         partial.get());
        finish_amax<<<1, 256>>>(partial.get(), static_cast<unsigned>(rb),
                                amax.get());
        mx_tensor<<<1, 256>>>(dx.get(), dd.get(), ds.get(), amax.get(), n, sr,
                              cfg.seed);
      } else
        mx_block<<<static_cast<unsigned>(groups), 32>>>(
            dx.get(), dd.get(), ds.get(), n, sr, cfg.seed);
    } else {
      amax_partial<<<static_cast<unsigned>(rb), 256>>>(dx.get(), n,
                                                       partial.get());
      finish_amax<<<1, 256>>>(partial.get(), static_cast<unsigned>(rb),
                              amax.get());
      nv_global_scale<<<1, 1>>>(amax.get(), global.get());
      if (tensor)
        nv_tensor<<<1, 256>>>(dx.get(), dd.get(), ds.get(), global.get(),
                              amax.get(), n, sr, cfg.seed);
      else
        nv_block<<<static_cast<unsigned>(groups), 16>>>(
            dx.get(), dd.get(), ds.get(), global.get(), n, sr, cfg.seed);
    }
    check(cudaGetLastError(), "quantize launch");
  };
  timing =
      luxorion::benchmark_detail::measure_cuda_events(warmup, repeats, launch);
  launch();
  check(cudaDeviceSynchronize(), "quantize sync");
  check(cudaMemcpy(q.data.data(), dd.get(), q.data.size(),
                   cudaMemcpyDeviceToHost),
        "data D2H");
  check(cudaMemcpy(q.scales.data(), ds.get(), q.scales.size(),
                   cudaMemcpyDeviceToHost),
        "scale D2H");
  if (cfg.format == QuantFormat::Nvfp4)
    check(cudaMemcpy(&q.global_scale, global.get(), 4, cudaMemcpyDeviceToHost),
          "global D2H");
  return q;
}

Quantized quantize(const luxorion::TensorData &t, const QuantConfig &c,
                   unsigned w, unsigned r, TimingStats &timing) {
  if (t.dtype == TensorDType::FP16)
    return quantize_impl<__half>(t.fp16_bits.data(), t.values.size(), c,
                                 t.num_rows, t.num_cols, t.dtype, w, r, timing);
  return quantize_impl<float>(t.values.data(), t.values.size(), c, t.num_rows,
                              t.num_cols, t.dtype, w, r, timing);
}

void validate_block_nearest(const Quantized &q,
                            const std::vector<float> &input) {
  if (q.scale_mode != ScaleMode::Block || q.rounding != RoundingMode::Nearest)
    return;
  if (q.format == QuantFormat::Mxfp8) {
    auto cpu = luxorion::mxfp8_quantize_cpu(input);
    if (q.data != cpu.data || q.scales != cpu.scales)
      throw std::runtime_error("MXFP8 CUDA result differs from CPU oracle");
  } else {
    auto cpu = luxorion::nvfp4_quantize_cpu(input);
    if (q.data != cpu.packed_data || q.scales != cpu.block_scales ||
        q.global_scale != cpu.global_scale)
      throw std::runtime_error("NVFP4 CUDA result differs from CPU oracle");
  }
}

float half_to_float(std::uint16_t h) {
  unsigned s = h >> 15, e = (h >> 10) & 31, m = h & 1023;
  if (e == 31)
    return std::numeric_limits<float>::quiet_NaN();
  double v = e == 0 ? std::ldexp(static_cast<double>(m), -24)
                    : std::ldexp(1.0 + static_cast<double>(m) / 1024,
                                 static_cast<int>(e) - 15);
  return std::copysign(static_cast<float>(v), s ? -1.0F : 1.0F);
}
float bf16_to_float(std::uint16_t b) {
  std::uint32_t u = static_cast<std::uint32_t>(b) << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

struct Dequantized {
  std::vector<std::uint8_t> bytes;
  std::vector<float> values;
  TimingStats timing;
};
template <typename O>
Dequantized dequant_impl(const Quantized &q, unsigned w, unsigned r) {
  std::size_t n = static_cast<std::size_t>(q.rows * q.cols);
  DeviceBuffer<std::uint8_t> dd(q.data.size()), ds(q.scales.size());
  DeviceBuffer<O> out(n);
  check(cudaMemcpy(dd.get(), q.data.data(), q.data.size(),
                   cudaMemcpyHostToDevice),
        "qdata H2D");
  check(cudaMemcpy(ds.get(), q.scales.data(), q.scales.size(),
                   cudaMemcpyHostToDevice),
        "qscale H2D");
  const std::size_t work_items =
      q.format == QuantFormat::Mxfp8 ? (n + 3) / 4 : (n + 1) / 2;
  unsigned blocks = static_cast<unsigned>(
      (work_items + kDequantThreads - 1) / kDequantThreads);
  bool tensor = q.scale_mode == ScaleMode::Tensor;
  auto launch = [&] {
    if (q.format == QuantFormat::Mxfp8)
      mx_deq<<<blocks, kDequantThreads>>>(dd.get(), ds.get(), out.get(), n,
                                         tensor);
    else
      nv_deq<<<blocks, kDequantThreads>>>(dd.get(), ds.get(), q.global_scale,
                                         out.get(), n, tensor);
    check(cudaGetLastError(), "dequantize launch");
  };
  Dequantized d;
  d.timing = luxorion::benchmark_detail::measure_cuda_events(w, r, launch);
  launch();
  d.bytes.resize(n * sizeof(O));
  check(cudaMemcpy(d.bytes.data(), out.get(), d.bytes.size(),
                   cudaMemcpyDeviceToHost),
        "output D2H");
  d.values.resize(n);
  if constexpr (sizeof(O) == 4) {
    std::memcpy(d.values.data(), d.bytes.data(), d.bytes.size());
  } else {
    for (std::size_t i = 0; i < n; ++i) {
      std::uint16_t bits;
      std::memcpy(&bits, d.bytes.data() + i * 2, 2);
      if constexpr (std::is_same<O, __half>::value) {
        d.values[i] = half_to_float(bits);
      } else {
        d.values[i] = bf16_to_float(bits);
      }
    }
  }
  for (float value : d.values) {
    if (!std::isfinite(value))
      throw std::overflow_error("dequantized output is not finite");
  }
  return d;
}
Dequantized dequantize(const Quantized &q, unsigned w, unsigned r) {
  if (q.output_type == OutputType::FP16)
    return dequant_impl<__half>(q, w, r);
  if (q.output_type == OutputType::BF16)
    return dequant_impl<__nv_bfloat16>(q, w, r);
  return dequant_impl<float>(q, w, r);
}

void put32(std::ostream &o, std::uint32_t v) {
  for (int i = 0; i < 4; ++i)
    o.put(static_cast<char>((v >> (8 * i)) & 255));
}
void put64(std::ostream &o, std::uint64_t v) {
  for (int i = 0; i < 8; ++i)
    o.put(static_cast<char>((v >> (8 * i)) & 255));
}
std::uint32_t get32(std::istream &i) {
  std::uint32_t v = 0;
  for (int k = 0; k < 4; ++k) {
    int c = i.get();
    if (c < 0)
      throw std::runtime_error("truncated header");
    v |= static_cast<std::uint32_t>(c) << (8 * k);
  }
  return v;
}
std::uint64_t get64(std::istream &i) {
  std::uint64_t v = 0;
  for (int k = 0; k < 8; ++k) {
    int c = i.get();
    if (c < 0)
      throw std::runtime_error("truncated header");
    v |= static_cast<std::uint64_t>(c) << (8 * k);
  }
  return v;
}
void write_q(const fs::path &p, const Quantized &q) {
  std::ofstream o(p, std::ios::binary);
  if (!o)
    throw std::runtime_error("cannot create quantized file");
  o.write("LUXQ0001", 8);
  put64(o, q.rows);
  put64(o, q.cols);
  put64(o, static_cast<std::uint64_t>(q.rows) *
               static_cast<std::uint64_t>(q.cols));
  put32(o, q.input_dtype == TensorDType::FP16 ? 2 : 1);
  put32(o, q.format == QuantFormat::Mxfp8 ? 1 : 2);
  put32(o, q.scale_mode == ScaleMode::Block ? 1 : 2);
  put32(o, q.rounding == RoundingMode::Nearest ? 1 : 2);
  put32(o, q.output_type == OutputType::FP16   ? 1
           : q.output_type == OutputType::BF16 ? 2
                                               : 3);
  put32(o, q.block_size);
  put32(o, q.seed);
  put64(o, q.data.size());
  put64(o, q.scales.size());
  std::uint32_t bits;
  std::memcpy(&bits, &q.global_scale, 4);
  put32(o, bits);
  put32(o, 0);
  o.write(reinterpret_cast<const char *>(q.data.data()), q.data.size());
  o.write(reinterpret_cast<const char *>(q.scales.data()), q.scales.size());
  if (!o)
    throw std::runtime_error("failed writing quantized file");
}
Quantized read_q(const fs::path &p) {
  constexpr std::uint64_t header = 84;
  std::ifstream i(p, std::ios::binary);
  char m[8];
  i.read(m, 8);
  if (!i || std::memcmp(m, "LUXQ0001", 8))
    throw std::invalid_argument("invalid LUXQ file");
  Quantized q;
  q.rows = static_cast<std::int64_t>(get64(i));
  q.cols = static_cast<std::int64_t>(get64(i));
  const std::uint64_t stored_n = get64(i);
  auto dt = get32(i), fmt = get32(i), sm = get32(i), rm = get32(i),
       ot = get32(i);
  if ((dt != 1 && dt != 2) || (fmt != 1 && fmt != 2) || (sm != 1 && sm != 2) ||
      (rm != 1 && rm != 2) || (ot < 1 || ot > 3))
    throw std::invalid_argument("invalid LUXQ enum field");
  q.input_dtype = dt == 2 ? TensorDType::FP16 : TensorDType::FP32;
  q.format = fmt == 1 ? QuantFormat::Mxfp8 : QuantFormat::Nvfp4;
  q.scale_mode = sm == 1 ? ScaleMode::Block : ScaleMode::Tensor;
  q.rounding = rm == 1 ? RoundingMode::Nearest : RoundingMode::Stochastic;
  q.output_type = ot == 1   ? OutputType::FP16
                  : ot == 2 ? OutputType::BF16
                            : OutputType::FP32;
  q.block_size = get32(i);
  q.seed = get32(i);
  auto db = get64(i), sb = get64(i);
  auto gb = get32(i);
  if (get32(i) != 0)
    throw std::invalid_argument("invalid LUXQ reserved field");
  std::memcpy(&q.global_scale, &gb, 4);
  if (q.rows <= 0 || q.cols <= 0 ||
      db > std::numeric_limits<std::size_t>::max() ||
      sb > std::numeric_limits<std::size_t>::max())
    throw std::invalid_argument("invalid LUXQ sizes");
  auto n =
      static_cast<std::uint64_t>(q.rows) * static_cast<std::uint64_t>(q.cols);
  if (n / static_cast<std::uint64_t>(q.rows) !=
      static_cast<std::uint64_t>(q.cols))
    throw std::invalid_argument("LUXQ element count overflow");
  if (stored_n != n)
    throw std::invalid_argument("LUXQ element count does not match shape");
  std::uint64_t groups = q.scale_mode == ScaleMode::Tensor
                             ? 1
                             : (n / q.block_size + (n % q.block_size != 0));
  std::uint64_t expected_data =
      q.format == QuantFormat::Mxfp8
          ? n
          : (q.scale_mode == ScaleMode::Tensor ? (n + 1) / 2 : groups * 8);
  if (q.block_size != (q.format == QuantFormat::Mxfp8 ? 32U : 16U) ||
      db != expected_data || sb != groups ||
      fs::file_size(p) != header + db + sb)
    throw std::invalid_argument("inconsistent LUXQ layout");
  if (!std::isfinite(q.global_scale) || q.global_scale <= 0)
    throw std::invalid_argument("invalid LUXQ global scale");
  q.data.resize(static_cast<std::size_t>(db));
  q.scales.resize(static_cast<std::size_t>(sb));
  i.read(reinterpret_cast<char *>(q.data.data()), q.data.size());
  i.read(reinterpret_cast<char *>(q.scales.data()), q.scales.size());
  if (!i || i.peek() != EOF)
    throw std::invalid_argument("invalid LUXQ payload length");
  if (q.format == QuantFormat::Mxfp8) {
    for (std::uint8_t code : q.scales)
      if (code == 0xFFU)
        throw std::invalid_argument("LUXQ contains an MXFP8 NaN scale");
    for (std::uint8_t code : q.data)
      if ((code & 0x7FU) == 0x7FU)
        throw std::invalid_argument("LUXQ contains an MXFP8 NaN element");
  } else {
    for (std::uint8_t code : q.scales)
      if (code == 0 || code > 0x7EU)
        throw std::invalid_argument("LUXQ contains an invalid NVFP4 scale");
    if ((n & 1U) && (q.data[(n - 1) / 2] & 0x0FU) != 0)
      throw std::invalid_argument("LUXQ contains nonzero NVFP4 tail padding");
    for (std::size_t byte = static_cast<std::size_t>((n + 1) / 2);
         byte < q.data.size(); ++byte)
      if (q.data[byte] != 0)
        throw std::invalid_argument(
            "LUXQ contains nonzero NVFP4 padding bytes");
  }
  return q;
}

void write_output(const fs::path &p, const Quantized &q, const Dequantized &d) {
  std::ofstream o(p, std::ios::binary);
  if (!o)
    throw std::runtime_error("cannot create output tensor");
  put64(o, q.rows);
  put64(o, q.cols);
  const char *name = q.output_type == OutputType::FP16   ? "fp16"
                     : q.output_type == OutputType::BF16 ? "bf16"
                                                         : "fp32";
  o.write(name, 4);
  o.write(reinterpret_cast<const char *>(d.bytes.data()), d.bytes.size());
  if (!o)
    throw std::runtime_error("failed writing output tensor");
}
void write_report(const fs::path &p, const Quantized &q,
                  const std::vector<float> *original, const TimingStats &qt,
                  const Dequantized &d, const std::string &target) {
  std::ofstream o(p);
  auto metrics = original ? luxorion::calculate_metrics(*original, d.values)
                          : luxorion::Metrics{};
  std::size_t n = static_cast<std::size_t>(q.rows * q.cols),
              raw = n * (q.input_dtype == TensorDType::FP16 ? 2 : 4),
              payload = q.data.size() + q.scales.size() +
                        (q.format == QuantFormat::Nvfp4 ? 4 : 0),
              outbytes = d.bytes.size();
  const bool has_amax_pass = q.format == QuantFormat::Nvfp4 ||
                             q.scale_mode == ScaleMode::Tensor;
  double qbytes = static_cast<double>(raw + payload + (has_amax_pass ? raw : 0));
  double dbytes = static_cast<double>(payload + outbytes);
  o << std::setprecision(10) << "{\n  \"format\": \""
    << (q.format == QuantFormat::Mxfp8 ? "mxfp8" : "nvfp4")
    << "\",\n  \"input_dtype\": \""
    << (q.input_dtype == TensorDType::FP16 ? "fp16" : "fp32")
    << "\",\n  \"output_type\": \""
    << (q.output_type == OutputType::FP16   ? "fp16"
        : q.output_type == OutputType::BF16 ? "bf16"
                                            : "fp32")
    << "\",\n  \"scale_mode\": \""
    << (q.scale_mode == ScaleMode::Block ? "block" : "tensor")
    << "\",\n  \"rounding\": \""
    << (q.rounding == RoundingMode::Nearest ? "nearest" : "stochastic")
    << "\",\n  \"seed\": " << q.seed << ",\n  \"target_gpu\": \"" << target
    << "\",\n  \"quantize_kernel_ms\": {\"median\": " << qt.median_ms
    << ", \"p95\": " << qt.p95_ms
    << "},\n  \"dequantize_kernel_ms\": {\"median\": " << d.timing.median_ms
    << ", \"p95\": " << d.timing.p95_ms << "},\n  \"quantize_effective_gbps\": "
    << (qt.median_ms > 0 ? qbytes / (qt.median_ms * 1e6) : 0)
    << ",\n  \"dequantize_effective_gbps\": "
    << (d.timing.median_ms > 0 ? dbytes / (d.timing.median_ms * 1e6) : 0)
    << ",\n  \"max_abs\": " << metrics.max_abs << ", \"mae\": " << metrics.mae
    << ", \"mse\": " << metrics.mse << ",\n  \"original_bytes\": " << raw
    << ", \"quantized_payload_bytes\": " << payload
    << ",\n  \"compression_ratio\": " << static_cast<double>(raw) / payload
    << "\n}\n";
}

unsigned count(const char *s) {
  auto v = std::stoul(s);
  if (v == 0)
    throw std::invalid_argument("repeats must be positive");
  return static_cast<unsigned>(v);
}
} // namespace

int main(int argc, char **argv) {
  try {
    if (argc < 2)
      throw std::invalid_argument(
          "usage: luxorion_cli run <tensor.bin> <quant.cfg> <output_dir> "
          "[warmup] [repeats] | dequantize <weights.luxq> <output.tensor.bin> "
          "[warmup] [repeats]");
    std::string op = argv[1];
    if (op == "run") {
      if (argc < 5 || argc > 7)
        throw std::invalid_argument(
            "usage: luxorion_cli run <tensor.bin> <quant.cfg> <output_dir> "
            "[warmup] [repeats]");
      unsigned w = argc > 5 ? static_cast<unsigned>(std::stoul(argv[5])) : 10,
               r = argc > 6 ? count(argv[6]) : 100;
      auto t = luxorion::read_tensor_file(argv[2]);
      auto c = luxorion::read_quant_config(argv[3]);
      fs::path out = argv[4];
      fs::create_directories(out);
      TimingStats qt;
      auto q = quantize(t, c, w, r, qt);
      validate_block_nearest(q, t.values);
      auto d = dequantize(q, w, r);
      write_q(out / "weights.luxq", q);
      write_output(out / "restored.tensor.bin", q, d);
      write_report(out / "report.json", q, &t.values, qt, d, c.target_gpu);
      std::cout << "weights: " << fs::absolute(out / "weights.luxq")
                << "\nrestored: " << fs::absolute(out / "restored.tensor.bin")
                << "\nreport: " << fs::absolute(out / "report.json") << '\n';
    } else if (op == "dequantize") {
      if (argc < 4 || argc > 6)
        throw std::invalid_argument(
            "usage: luxorion_cli dequantize <weights.luxq> <output.tensor.bin> "
            "[warmup] [repeats]");
      unsigned w = argc > 4 ? static_cast<unsigned>(std::stoul(argv[4])) : 10,
               r = argc > 5 ? count(argv[5]) : 100;
      auto q = read_q(argv[2]);
      auto d = dequantize(q, w, r);
      write_output(argv[3], q, d);
      std::cout << std::setprecision(10)
                << "restored: " << fs::absolute(argv[3]) << '\n'
                << "dequant_median_ms: " << d.timing.median_ms << '\n'
                << "dequant_p95_ms: " << d.timing.p95_ms << '\n';
    } else
      throw std::invalid_argument("operation must be run or dequantize");
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "luxorion_cli failed: " << e.what() << '\n';
    return 1;
  }
}
