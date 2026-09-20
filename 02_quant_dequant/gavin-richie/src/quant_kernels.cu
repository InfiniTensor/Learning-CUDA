#include "quant_kernels.cuh"
#include "fp_formats.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call)                                                    \
  do {                                                                      \
    cudaError_t e_ = (call);                                                \
    if (e_ != cudaSuccess) {                                                \
      fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_),   \
              __FILE__, __LINE__);                                          \
      exit(1);                                                              \
    }                                                                       \
  } while (0)

namespace {

// ------------------------------------------------------------------ helpers

__host__ __device__ inline uint32_t hash_u32(uint32_t x) {
  x ^= x >> 16; x *= 0x7feb352dU; x ^= x >> 15; x *= 0x846ca68bU; x ^= x >> 16;
  return x;
}

// uniform [0,1) from an integer index; deterministic per element+seed
__device__ inline float uniform01(uint64_t index, uint32_t seed) {
  uint32_t h = hash_u32((uint32_t)(index * 2654435761u) ^ seed ^
                        (uint32_t)(index >> 32));
  return (float)(h >> 8) * (1.0f / 16777216.0f);
}

// ULP of the FP8 grid at magnitude |y| (used by stochastic rounding).
__host__ __device__ inline float fp8_grid_step(float y, bool e4m3) {
  const int mant = e4m3 ? 3 : 2;
  const int bias = e4m3 ? 7 : 15;
  const int emin = 1 - bias;
  float a = fabsf(y);
  if (a == 0.0f) return ldexpf(1.0f, emin - 1 - mant);
  int e;
  frexpf(a, &e);
  --e;
  if (e < emin) e = emin - 1;
  return ldexpf(1.0f, e - mant);
}

// ULP of the E2M1 grid at magnitude |y|.
__host__ __device__ inline float e2m1_grid_step(float y) {
  float a = fabsf(y);
  if (a < 1.0f) return 0.5f;
  if (a < 2.0f) return 0.5f;
  if (a < 4.0f) return 1.0f;
  return 2.0f;
}

// Generic output store for the three supported dtypes.
__device__ inline void store_out(void* out, uint64_t i, float v, OutputType t) {
  switch (t) {
    case OutputType::FP32:
      static_cast<float*>(out)[i] = v; break;
    case OutputType::FP16:
      static_cast<__half*>(out)[i] = __float2half(v); break;
    case OutputType::BF16:
      static_cast<__nv_bfloat16*>(out)[i] = __float2bfloat16(v); break;
  }
}

__device__ inline float load_out(const void* out, uint64_t i, OutputType t) {
  switch (t) {
    case OutputType::FP32: return static_cast<const float*>(out)[i];
    case OutputType::FP16: return __half2float(static_cast<const __half*>(out)[i]);
    case OutputType::BF16:
      return __bfloat162float(static_cast<const __nv_bfloat16*>(out)[i]);
  }
  return 0.0f;
}

// ---------------------------------------------------------------- amax reduce

__global__ void amax_kernel(const float* __restrict__ data, uint64_t n,
                            uint32_t* d_amax_bits) {
  __shared__ float s[256];
  const int tid = threadIdx.x;
  float m = 0.0f;
  for (uint64_t i = blockIdx.x * blockDim.x + tid; i < n;
       i += (uint64_t)gridDim.x * blockDim.x)
    m = fmaxf(m, fabsf(data[i]));
  s[tid] = m;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) s[tid] = fmaxf(s[tid], s[tid + stride]);
    __syncthreads();
  }
  if (tid == 0) {
    // amax >= 0 so the IEEE bit pattern is order-preserving
    uint32_t bits;
    float v = s[0];
    memcpy(&bits, &v, 4);
    atomicMax(d_amax_bits, bits);
  }
}

// ------------------------------------------------------------- MXFP8 quantize

// Block mode: one thread per element block; the thread computes the block
// amax, encodes the E8M0 scale and writes E4M3/E5M2 payloads (1 byte each).
// Tensor mode: tensor_scale is applied instead (computed by the host from the
// tensor amax); no block scale is written.
__global__ void quant_mxfp8_kernel(const float* __restrict__ in, uint64_t n,
                                   uint32_t block_size, uint32_t mode_block,
                                   uint32_t e4m3, uint32_t stochastic,
                                   uint32_t seed, float tensor_scale,
                                   uint8_t* __restrict__ packed,
                                   uint8_t* __restrict__ scales) {
  const uint64_t start = (uint64_t)(blockIdx.x * blockDim.x + threadIdx.x) *
                         block_size;
  if (start >= n) return;
  const uint64_t end = min(start + block_size, n);

  float scale;
  uint8_t scale_code = 0;
  if (mode_block) {
    float amax = 0.0f;
    for (uint64_t i = start; i < end; ++i)
      amax = fmaxf(amax, fabsf(in[i]));
    if (amax == 0.0f) {
      scale_code = 127;  // 2^0
    } else {
      // scale >= amax/448 so scaled mantissas fit the format
      const float fmax = e4m3 ? 448.0f : 57344.0f;
      scale_code = lp::fp32_to_e8m0_ceil(amax / fmax);
    }
    scale = lp::e8m0_to_fp32(scale_code);
    scales[start / block_size] = scale_code;
  } else {
    scale = tensor_scale;
  }

  for (uint64_t i = start; i < end; ++i) {
    float y = in[i] / scale;
    if (stochastic) {
      const float u = uniform01(i, seed);
      y += (u - 0.5f) * fp8_grid_step(y, e4m3);
    }
    packed[i] = lp::fp32_to_fp8(y, e4m3);
  }
}

// ------------------------------------------------------------- NVFP4 quantize

// One thread per 16-element block. Two-level scaling:
//   global scale (fp32, host): GS = max(1, amax_tensor / 2688)  [2688 = 448*6]
//   block scale (E4M3):        Sb = e4m3(amax_block / GS / 6)
//   payload:                   e2m1(x / GS / Sb)
// Two elements are packed per byte (element 2k -> low nibble of byte k):
// one 8-byte packed store per 16-element block (ulonglong1).
__global__ void quant_nvfp4_kernel(const float* __restrict__ in, uint64_t n,
                                   uint32_t block_size, uint32_t mode_block,
                                   uint32_t stochastic, uint32_t seed,
                                   float global_scale, float tensor_scale,
                                   uint8_t* __restrict__ packed,
                                   uint8_t* __restrict__ scales) {
  const uint64_t start = (uint64_t)(blockIdx.x * blockDim.x + threadIdx.x) *
                         block_size;
  if (start >= n) return;
  const uint64_t end = min(start + block_size, n);

  float outer_scale, sb;
  if (mode_block) {
    float amax = 0.0f;
    for (uint64_t i = start; i < end; ++i)
      amax = fmaxf(amax, fabsf(in[i]));
    amax /= global_scale;
    sb = (amax == 0.0f) ? 1.0f : lp::fp8_to_fp32(
                                     lp::fp32_to_fp8(amax / 6.0f, true), true);
    scales[start / block_size] = lp::fp32_to_fp8(sb, true);
    outer_scale = global_scale * sb;
  } else {
    outer_scale = tensor_scale;
  }
  const float inv_scale = 1.0f / outer_scale;

  // Pack two nibbles per byte; flush 8 assembled bytes (one full block)
  // with a single packed store. The tail block may contain an odd element,
  // in which case the last byte's high nibble is zero-filled.
  const uint64_t nbytes_block = block_size / 2;
  uint8_t buf[8];
  const uint64_t out_base = start / 2;
  for (uint64_t k = 0; k < nbytes_block; ++k) {
    const uint64_t i0 = start + 2 * k;
    if (i0 >= end) break;
    float y0 = in[i0] * inv_scale;
    float y1 = (i0 + 1 < end) ? in[i0 + 1] * inv_scale : 0.0f;
    if (stochastic) {
      y0 += (uniform01(i0, seed) - 0.5f) * e2m1_grid_step(y0);
      if (i0 + 1 < end)
        y1 += (uniform01(i0 + 1, seed) - 0.5f) * e2m1_grid_step(y1);
    }
    const uint8_t lo = lp::fp32_to_e2m1(y0);
    const uint8_t hi = (i0 + 1 < end) ? lp::fp32_to_e2m1(y1) : 0;
    buf[k & 7] = lo | (uint8_t)(hi << 4);
    if ((k & 7) == 7 || k + 1 == nbytes_block || i0 + 2 >= end) {
      const int cnt = (int)((k & 7) + 1);
      if (cnt == 8) {
        uint64_t word;
        memcpy(&word, buf, 8);
        reinterpret_cast<unsigned long long*>(packed)[out_base / 8] = word;
      } else {
        for (int b = 0; b < cnt; ++b) packed[out_base + b] = buf[b];
      }
    }
  }
}

// ------------------------------------------------------------ dequantization

__global__ void dequant_mxfp8_kernel(const uint8_t* __restrict__ packed,
                                     const uint8_t* __restrict__ scales,
                                     uint64_t n, uint32_t block_size,
                                     uint32_t mode_block, uint32_t e4m3,
                                     float tensor_scale, void* __restrict__ out,
                                     uint32_t out_type) {
  const uint64_t i = (uint64_t)(blockIdx.x * blockDim.x) + threadIdx.x;
  if (i >= n) return;
  float scale;
  if (mode_block)
    scale = lp::e8m0_to_fp32(scales[i / block_size]);
  else
    scale = tensor_scale;
  const float v = lp::fp8_to_fp32(packed[i], e4m3) * scale;
  store_out(out, i, v, (OutputType)out_type);
}

// Packed load: one thread loads one byte (two 4-bit elements), extracts both
// nibbles and emits two outputs.
__global__ void dequant_nvfp4_kernel(const uint8_t* __restrict__ packed,
                                     const uint8_t* __restrict__ scales,
                                     uint64_t n, uint32_t block_size,
                                     uint32_t mode_block, float global_scale,
                                     float tensor_scale, void* __restrict__ out,
                                     uint32_t out_type) {
  const uint64_t b = (uint64_t)(blockIdx.x * blockDim.x) + threadIdx.x;
  if (b * 2 >= n) return;
  const uint8_t byte = packed[b];
  const uint64_t i0 = b * 2;
  float scale;
  if (mode_block)
    scale = lp::fp8_to_fp32(scales[i0 / block_size], true) * global_scale;
  else
    scale = tensor_scale;
  store_out(out, i0, lp::e2m1_to_fp32(byte & 0xF) * scale, (OutputType)out_type);
  if (i0 + 1 < n)
    store_out(out, i0 + 1, lp::e2m1_to_fp32(byte >> 4) * scale,
              (OutputType)out_type);
}

// ------------------------------------------------------------------ metrics

struct MetricSums {
  float max_abs;
  double mae;
  double mse;
};

__global__ void metrics_kernel(const float* __restrict__ ref,
                               const void* __restrict__ out, uint64_t n,
                               uint32_t out_type, float* d_max,
                               double* d_mae, double* d_mse) {
  const int tid = threadIdx.x;
  __shared__ float s_max[256];
  __shared__ double s_mae[256];
  __shared__ double s_mse[256];
  float m = 0.0f; double mae = 0.0, mse = 0.0;
  for (uint64_t i = (uint64_t)blockIdx.x * blockDim.x + tid; i < n;
       i += (uint64_t)gridDim.x * blockDim.x) {
    const float d = fabsf(ref[i] - load_out(out, i, (OutputType)out_type));
    m = fmaxf(m, d);
    mae += d;
    mse += (double)d * d;
  }
  s_max[tid] = m; s_mae[tid] = mae; s_mse[tid] = mse;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      s_max[tid] = fmaxf(s_max[tid], s_max[tid + stride]);
      s_mae[tid] += s_mae[tid + stride];
      s_mse[tid] += s_mse[tid + stride];
    }
    __syncthreads();
  }
  if (tid == 0) {
    uint32_t bits;
    float v = s_max[0];
    memcpy(&bits, &v, 4);
    atomicMax((uint32_t*)d_max, bits);
    atomicAdd(d_mae, s_mae[0]);
    atomicAdd(d_mse, s_mse[0]);
  }
}

}  // namespace

// -------------------------------------------------------------- host launches

float reduce_amax(const float* d_data, uint64_t n) {
  uint32_t* d_bits;
  CUDA_CHECK(cudaMalloc(&d_bits, 4));
  CUDA_CHECK(cudaMemset(d_bits, 0, 4));
  const int block = 256;
  const int grid = (int)std::min<uint64_t>((n + block - 1) / block, 4096);
  amax_kernel<<<grid, block>>>(d_data, n, d_bits);
  CUDA_CHECK(cudaGetLastError());
  uint32_t h_bits = 0;
  CUDA_CHECK(cudaMemcpy(&h_bits, d_bits, 4, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d_bits));
  float v;
  memcpy(&v, &h_bits, 4);
  return v;
}

void launch_quantize(const float* d_data, uint64_t n, const QuantConfig& cfg,
                     uint8_t* d_packed, uint8_t* d_scales,
                     float* d_tensor_scale, float* d_global_scale,
                     StageTiming* timing) {
  const uint32_t bs = (uint32_t)cfg.block_size;
  const uint64_t nblocks = (n + bs - 1) / bs;
  const int block = 128;
  const int grid = (int)std::min<uint64_t>((nblocks + block - 1) / block,
                                           1u << 20);
  const uint32_t seed = 0x9E3779B9u;

  float tensor_scale = 1.0f, global_scale = 1.0f;
  const float amax = reduce_amax(d_data, n);
  if (cfg.scale_mode == ScaleMode::TENSOR) {
    tensor_scale = (amax > 0.0f) ? amax / (cfg.format == QuantFormat::NVFP4
                                               ? 6.0f
                                               : 448.0f)
                                 : 1.0f;
  } else if (cfg.format == QuantFormat::NVFP4) {
    global_scale = (amax > 2688.0f) ? amax / 2688.0f : 1.0f;
  }

  cudaEvent_t ev0, ev1;
  CUDA_CHECK(cudaEventCreate(&ev0));
  CUDA_CHECK(cudaEventCreate(&ev1));
  CUDA_CHECK(cudaEventRecord(ev0));

  switch (cfg.format) {
    case QuantFormat::MXFP8_E4M3:
    case QuantFormat::MXFP8_E5M2: {
      const uint32_t e4m3 = cfg.format == QuantFormat::MXFP8_E4M3 ? 1 : 0;
      quant_mxfp8_kernel<<<grid, block>>>(
          d_data, n, bs, cfg.scale_mode == ScaleMode::BLOCK ? 1 : 0, e4m3,
          cfg.rounding == Rounding::STOCHASTIC ? 1 : 0, seed, tensor_scale,
          d_packed, d_scales);
      break;
    }
    case QuantFormat::NVFP4:
      quant_nvfp4_kernel<<<grid, block>>>(
          d_data, n, bs, cfg.scale_mode == ScaleMode::BLOCK ? 1 : 0,
          cfg.rounding == Rounding::STOCHASTIC ? 1 : 0, seed, global_scale,
          tensor_scale, d_packed, d_scales);
      break;
  }
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventRecord(ev1));
  CUDA_CHECK(cudaEventSynchronize(ev1));
  if (timing)
    CUDA_CHECK(cudaEventElapsedTime(&timing->gpu_ms, ev0, ev1));
  CUDA_CHECK(cudaEventDestroy(ev0));
  CUDA_CHECK(cudaEventDestroy(ev1));

  if (d_tensor_scale) CUDA_CHECK(cudaMemcpy(d_tensor_scale, &tensor_scale, 4, cudaMemcpyHostToDevice));
  if (d_global_scale) CUDA_CHECK(cudaMemcpy(d_global_scale, &global_scale, 4, cudaMemcpyHostToDevice));
}

void launch_dequantize(const uint8_t* d_packed, uint64_t n,
                       const uint8_t* d_scales, float tensor_scale,
                       float global_scale, const QuantConfig& cfg,
                       void* d_out, StageTiming* timing) {
  const uint32_t bs = (uint32_t)cfg.block_size;
  const int block = 256;
  int grid;
  cudaEvent_t ev0, ev1;
  CUDA_CHECK(cudaEventCreate(&ev0));
  CUDA_CHECK(cudaEventCreate(&ev1));
  CUDA_CHECK(cudaEventRecord(ev0));
  switch (cfg.format) {
    case QuantFormat::MXFP8_E4M3:
    case QuantFormat::MXFP8_E5M2:
      grid = (int)std::min<uint64_t>((n + block - 1) / block, 1u << 21);
      dequant_mxfp8_kernel<<<grid, block>>>(
          d_packed, d_scales, n, bs,
          cfg.scale_mode == ScaleMode::BLOCK ? 1 : 0,
          cfg.format == QuantFormat::MXFP8_E4M3 ? 1 : 0, tensor_scale, d_out,
          (uint32_t)cfg.output_type);
      break;
    case QuantFormat::NVFP4: {
      const uint64_t nthreads = (n + 1) / 2;
      grid = (int)std::min<uint64_t>((nthreads + block - 1) / block, 1u << 21);
      dequant_nvfp4_kernel<<<grid, block>>>(
          d_packed, d_scales, n, bs,
          cfg.scale_mode == ScaleMode::BLOCK ? 1 : 0, global_scale,
          tensor_scale, d_out, (uint32_t)cfg.output_type);
      break;
    }
  }
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventRecord(ev1));
  CUDA_CHECK(cudaEventSynchronize(ev1));
  if (timing)
    CUDA_CHECK(cudaEventElapsedTime(&timing->gpu_ms, ev0, ev1));
  CUDA_CHECK(cudaEventDestroy(ev0));
  CUDA_CHECK(cudaEventDestroy(ev1));
}

ErrorMetrics compute_metrics(const float* d_ref, const void* d_out, uint64_t n,
                             OutputType dtype) {
  float* d_max;
  double* d_sums;  // [mae, mse]
  CUDA_CHECK(cudaMalloc(&d_max, 4));
  CUDA_CHECK(cudaMalloc(&d_sums, 2 * sizeof(double)));
  CUDA_CHECK(cudaMemset(d_max, 0, 4));
  CUDA_CHECK(cudaMemset(d_sums, 0, 2 * sizeof(double)));
  const int block = 256;
  const int grid = (int)std::min<uint64_t>((n + block - 1) / block, 4096);
  metrics_kernel<<<grid, block>>>(d_ref, d_out, n, (uint32_t)dtype, d_max,
                                  d_sums, d_sums + 1);
  CUDA_CHECK(cudaGetLastError());
  uint32_t h_max = 0;
  double h_sums[2] = {0, 0};
  CUDA_CHECK(cudaMemcpy(&h_max, d_max, 4, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_sums, d_sums, 2 * sizeof(double),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d_max));
  CUDA_CHECK(cudaFree(d_sums));
  ErrorMetrics m;
  memcpy(&m.max_abs, &h_max, 4);
  m.mae = (float)(h_sums[0] / n);
  m.mse = (float)(h_sums[1] / n);
  return m;
}
