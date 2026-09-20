#pragma once
#include "vector.cuh"
#include <algorithm>

namespace lp {
__global__ void maximum(const void *x, size_t n, int dtype, float *result) {
  float a = 0;
  for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
       i += size_t(blockDim.x) * gridDim.x)
    a = fmaxf(a, fabsf(load(x, i, dtype)));
  a = warp_max(a);
  __shared__ float warp_results[8];
  if ((threadIdx.x & 31) == 0)
    warp_results[threadIdx.x / 32] = a;
  __syncthreads();
  if (threadIdx.x < 32) {
    a = threadIdx.x < 8 ? warp_results[threadIdx.x] : 0.0f;
    a = warp_max(a);
    if (threadIdx.x == 0)
      atomicMax(reinterpret_cast<unsigned *>(result), __float_as_uint(a));
  }
}
__global__ void scales_kernel(const void *x, uint8_t *scales, const float *amax,
                              Layout p) {
  size_t groups_per_row = (p.cols + p.block - 1) / p.block;
  size_t g = size_t(blockIdx.x) * 8 + threadIdx.x / 32;
  int lane = threadIdx.x & 31;
  size_t ng = p.tensor ? 1 : p.rows * groups_per_row;
  if (g >= ng)
    return;
  float a = 0;
  if (p.tensor)
    a = *amax;
  else {
    size_t row = g / groups_per_row, start = (g % groups_per_row) * p.block;
    for (int j = lane; j < p.block && start + j < p.cols; j += 32)
      a = fmaxf(a, fabsf(load(x, row * p.cols + start + j, p.dtype)));
    a = warp_max(a);
  }
  if (lane == 0)
    scales[g] =
        p.fmt == MXFP8 ? mx_scale(a) : encode8(divide_rn(divide_rn(a, global_scale(*amax)), 6.0f));
}
__device__ inline size_t group_index(size_t row, size_t col, Layout p) {
  return p.tensor ? 0 : row * ((p.cols + p.block - 1) / p.block) + col / p.block;
}
__device__ inline uint8_t quant_element(const void *x, size_t row, size_t col,
                                        const uint8_t *scales, const float *amax,
                                        Layout p) {
  size_t i = row * p.cols + col;
  float s = scale_value(scales[group_index(row, col, p)], p.fmt);
  float value = load(x, i, p.dtype);
  if (p.fmt == NVFP4)
    value = divide_rn(value, global_scale(*amax));
  float y = s == 0 ? 0 : divide_rn(value, s);
  float u = uniform(i, p.seed);
  return p.fmt == MXFP8 ? encode8(y, p.stochastic, u) : encode4(y, p.stochastic, u);
}
__global__ void quant_kernel(const void *x, uint8_t *data, const uint8_t *scales,
                             const float *amax, Layout p) {
  size_t stride = p.fmt == MXFP8 ? p.cols : (p.cols + 1) / 2;
  size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= p.rows * stride)
    return;
  size_t row = i / stride, col = i % stride;
  if (p.fmt == MXFP8)
    data[i] = quant_element(x, row, col, scales, amax, p);
  else {
    col *= 2;
    uint8_t lo = quant_element(x, row, col, scales, amax, p);
    uint8_t hi = col + 1 < p.cols ? quant_element(x, row, col + 1, scales, amax, p) : 0;
    data[i] = lo | (hi << 4);
  }
}
__global__ void dequant_kernel(const uint8_t *data, const uint8_t *scales, float global,
                               void *out, int out_type, Layout p) {
  size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= p.rows * p.cols)
    return;
  size_t row = i / p.cols, col = i % p.cols;
  uint8_t c = p.fmt == MXFP8
                  ? data[i]
                  : (data[row * ((p.cols + 1) / 2) + col / 2] >> ((col & 1) * 4)) & 15;
  float v = p.fmt == MXFP8 ? fp8_value(c) : fp4_value(c);
  float s = scale_value(scales[group_index(row, col, p)], p.fmt);
  // Same multiplication order as the independent file decoder.
  save(out, i, multiply_rn(multiply_rn(v, s), global), out_type);
}
// A row-aligned tile fuses block scaling and packing. Each lane loads once;
// NVFP4 uses two independent 16-lane reductions and one writer per packed byte.
template <int FMT, bool FLAT = false>
__global__ void quant_blocks_kernel(const void *x, uint8_t *data, uint8_t *scales,
                                    const float *amax, Layout p) {
  constexpr int B = FMT == MXFP8 ? 32 : 16;
  const size_t row = FLAT ? 0 : blockIdx.x;
  const size_t limit = FLAT ? p.rows * p.cols : p.cols;
  const size_t col = size_t(FLAT ? blockIdx.x : blockIdx.y) * 256 + threadIdx.x;
  const int lane = threadIdx.x & 31;
  float value = col < limit ? load(x, row * p.cols + col, p.dtype) : 0;
  float a = p.tensor ? 0.0f : warp_max(fabsf(value), B);
  float global = FMT == NVFP4 ? global_scale(*amax) : 1.0f;
  unsigned code_scale = 0;
  if (lane % B == 0)
    code_scale = p.tensor ? scales[0]
                          : (FMT == MXFP8 ? mx_scale(a) : encode8(divide_rn(divide_rn(a, global), 6.0f)));
  code_scale = __shfl_sync(FULL_WARP_MASK, code_scale, 0, B);
  if (!p.tensor && col < limit && lane % B == 0)
    scales[row * ((p.cols + B - 1) / B) + col / B] = uint8_t(code_scale);
  float scale = scale_value(uint8_t(code_scale), FMT);
  if (FMT == NVFP4)
    value = divide_rn(value, global);
  float z = FMT == MXFP8 ? mx_scaled(value, uint8_t(code_scale))
                         : (scale == 0 ? 0 : divide_rn(value, scale));
  float u = p.stochastic ? uniform(row * p.cols + col, p.seed) : 0.0f;
  unsigned code = FMT == MXFP8 ? encode8(z, p.stochastic, u)
                               : encode4_scaled(value, scale, p.stochastic, u);
  if (FMT == MXFP8) {
    if (col < limit)
      data[row * p.cols + col] = uint8_t(code);
  } else {
    unsigned hi = __shfl_xor_sync(FULL_WARP_MASK, code, 1);
    if (col < limit && !(lane & 1))
      data[row * ((p.cols + 1) / 2) + col / 2] =
          uint8_t(code | ((col + 1 < limit ? hi : 0) << 4));
  }
}

// Constant-format indexing removes per-element 64-bit division from the common
// 16/32-element layouts. A packed NVFP4 byte is fetched once for two outputs.
template <int FMT, bool FLAT = false>
__global__ void dequant_blocks_kernel(const uint8_t *data, const uint8_t *scales,
                                      float global, void *out, int out_type, Layout p) {
  constexpr int B = FMT == MXFP8 ? 32 : 16;
  size_t row = FLAT ? 0 : blockIdx.x;
  size_t limit = FLAT ? p.rows * p.cols : p.cols;
  size_t col = (size_t(FLAT ? blockIdx.x : blockIdx.y) * 256 + threadIdx.x) *
               (FMT == MXFP8 ? 1 : 2);
  if (col >= limit)
    return;
  size_t g = p.tensor ? 0 : row * ((p.cols + B - 1) / B) + col / B;
  float s = scale_value(scales[g], FMT);
  uint8_t code =
      data[FMT == MXFP8 ? row * p.cols + col : row * ((p.cols + 1) / 2) + col / 2];
  float v = FMT == MXFP8 ? fp8_value(code) : fp4_value(code & 15);
  save(out, row * p.cols + col, multiply_rn(multiply_rn(v, s), global), out_type);
  if (FMT == NVFP4 && col + 1 < limit)
    save(out, row * p.cols + col + 1, multiply_rn(multiply_rn(fp4_value(code >> 4), s), global), out_type);
}

template <int FMT>
inline void launch_quant_vector(const void *x, uint8_t *data, uint8_t *scales,
                                const float *amax, Layout p) {
#if defined(__MACACC__)
  // C500: scalar tiles avoid underfilling small grids; eight values per lane
  // amortize software encoding and improve large-tensor memory transactions.
  if (p.count() <= 65536) {
    quant_blocks_kernel<FMT, true><<<(p.count() + 255) / 256, 256>>>(
        x, data, scales, amax, p);
    return;
  }
  constexpr int V = 8, THREADS = 256;
#elif defined(__MUSACC__)
  // S4000: scalar small tiles avoid setup overhead. Four-value tiles favor
  // medium inputs; eight-value tiles amortize packing on streamed tensors.
  if (p.count() <= 65536) {
    quant_blocks_kernel<FMT, true><<<(p.count() + 255) / 256, 256>>>(
        x, data, scales, amax, p);
    return;
  }
  if (p.count() <= 1048576) {
    size_t grid = (p.count() + 2047) / 2048;
    if (p.dtype == FP32)
      quant_vector_kernel<FMT, FP32, 4><<<grid, 512>>>(x, data, scales, amax, p);
    else if (p.dtype == FP16)
      quant_vector_kernel<FMT, FP16, 4><<<grid, 512>>>(x, data, scales, amax, p);
    else
      quant_vector_kernel<FMT, BF16, 4><<<grid, 512>>>(x, data, scales, amax, p);
    return;
  }
  constexpr int V = 8, THREADS = 128;
#elif defined(__ILUVATAR__)
  // MR-V100: two FP32 values per lane favor MXFP8, while NVFP4's
  // 16-bit inputs benefit from eight. The scan is in results/iluvatar/.
  if (p.count() <= 65536 || (p.dtype == FP32 && FMT == MXFP8)) {
    constexpr int T = 256, VEC = 2;
    size_t grid = (p.count() + T * VEC - 1) / (T * VEC);
    if (p.dtype == FP32)
      quant_vector_kernel<FMT, FP32, VEC><<<grid, T>>>(x, data, scales, amax, p);
    else if (p.dtype == FP16)
      quant_vector_kernel<FMT, FP16, VEC><<<grid, T>>>(x, data, scales, amax, p);
    else
      quant_vector_kernel<FMT, BF16, VEC><<<grid, T>>>(x, data, scales, amax, p);
    return;
  }
  if (FMT == NVFP4 && p.dtype != FP32) {
    size_t grid = (p.count() + 1023) / 1024;
    if (p.dtype == FP16)
      quant_vector_kernel<FMT, FP16, 8><<<grid, 128>>>(x, data, scales, amax, p);
    else
      quant_vector_kernel<FMT, BF16, 8><<<grid, 128>>>(x, data, scales, amax, p);
    return;
  }
  constexpr int V = 4, THREADS = 128;
#else
  constexpr int V = 4, THREADS = 128;
#endif
  size_t blocks = (p.count() + V * THREADS - 1) / (V * THREADS);
  if (p.dtype == FP32)
    quant_vector_kernel<FMT, FP32, V><<<blocks, THREADS>>>(x, data, scales, amax, p);
  else if (p.dtype == FP16)
    quant_vector_kernel<FMT, FP16, V><<<blocks, THREADS>>>(x, data, scales, amax, p);
  else
    quant_vector_kernel<FMT, BF16, V><<<blocks, THREADS>>>(x, data, scales, amax, p);
}

inline void launch_max(const void *x, size_t n, int dtype, float *amax) {
  cudaMemsetAsync(amax, 0, sizeof(float));
#if defined(__MUSACC__)
  // The scan includes clear + reduction. Small grids reduce atomic overhead;
  // streamed FP32 inputs need more CTAs to cover the device's 64 MPs.
  bool small = n <= 1048576;
  if (dtype == FP32) {
    if (small)
      maximum_vector<FP32, 4><<<std::min(size_t(128), (n + 1023) / 1024), 256>>>(x, n, amax);
    else
      maximum_vector<FP32, 8><<<std::min(size_t(1024), (n + 4095) / 4096), 512>>>(x, n, amax);
  } else {
    int threads = small ? 128 : 256;
    size_t grid = std::min(size_t(small ? 128 : 256), (n + threads * 8 - 1) / (threads * 8));
    if (dtype == FP16)
      maximum_vector<FP16, 8><<<grid, threads>>>(x, n, amax);
    else
      maximum_vector<BF16, 8><<<grid, threads>>>(x, n, amax);
  }
#elif defined(__ILUVATAR__)
  // Scalar FP32 accesses outperform vector loads on MR-V100. Half-size
  // inputs use four values per lane; cap the grid to amortize the final atomics.
  if (dtype == FP32)
    maximum<<<std::min(size_t(256), (n + 255) / 256), 256>>>(x, n, dtype, amax);
  else if (dtype == FP16)
    maximum_vector<FP16, 4><<<std::min(size_t(128), (n + 2047) / 2048), 512>>>(x, n, amax);
  else
    maximum_vector<BF16, 4><<<std::min(size_t(128), (n + 2047) / 2048), 512>>>(x, n, amax);
#else
  size_t blocks = std::min(size_t(512), (n + 2047) / 2048);
#if defined(__MACACC__)
  constexpr int FP32_VECTOR = 4;
#else
  constexpr int FP32_VECTOR = 8;
#endif
  if (dtype == FP32)
    maximum_vector<FP32, FP32_VECTOR><<<blocks, 256>>>(x, n, amax);
  else if (dtype == FP16)
    maximum_vector<FP16, 8><<<blocks, 256>>>(x, n, amax);
  else
    maximum_vector<BF16, 8><<<blocks, 256>>>(x, n, amax);
#endif
}
// Reuse an already computed tensor maximum (e.g. a fused transform+amax).
inline void quantize_with_amax(const void *x, uint8_t *data, uint8_t *scales,
                               float *amax, Layout p) {
  if (p.rows <= 2147483647 && (p.cols + 255) / 256 <= 65535 &&
      p.block == (p.fmt == MXFP8 ? 32 : 16)) {
    if (p.tensor)
      scales_kernel<<<1, 256>>>(x, scales, amax, p);
    if (p.cols % p.block == 0) {
      if (p.fmt == MXFP8)
        launch_quant_vector<MXFP8>(x, data, scales, amax, p);
      else
        launch_quant_vector<NVFP4>(x, data, scales, amax, p);
      return;
    }
    dim3 grid(p.rows, (p.cols + 255) / 256);
    if (p.fmt == MXFP8)
      quant_blocks_kernel<MXFP8><<<grid, 256>>>(x, data, scales, amax, p);
    else
      quant_blocks_kernel<NVFP4><<<grid, 256>>>(x, data, scales, amax, p);
    return;
  }
  scales_kernel<<<(p.groups() + 7) / 8, 256>>>(x, scales, amax, p);
  quant_kernel<<<(p.bytes() + 255) / 256, 256>>>(x, data, scales, amax, p);
}
inline void quantize(const void *x, uint8_t *data, uint8_t *scales, float *amax,
                     Layout p) {
  if (p.fmt == NVFP4 || p.tensor)
    launch_max(x, p.count(), p.dtype, amax);
  quantize_with_amax(x, data, scales, amax, p);
}
template <int FMT>
inline void launch_dequant_vector(const uint8_t *data, const uint8_t *scales,
                                  float global, void *out, int out_type, Layout p) {
#if defined(__MACACC__)
  if (p.count() <= 65536) {
    dequant_blocks_kernel<FMT, true><<<(p.bytes() + 255) / 256, 256>>>(
        data, scales, global, out, out_type, p);
    return;
  }
  int half_threads = p.count() <= 1048576 ? 512 : 128;
#elif defined(__MUSACC__)
  if (p.count() <= 65536) {
    dequant_blocks_kernel<FMT, true><<<(p.bytes() + 255) / 256, 256>>>(
        data, scales, global, out, out_type, p);
    return;
  }
  constexpr int half_threads = 128;
  if (p.count() <= 1048576) {
    size_t grid = (p.count() + 2047) / 2048;
    if (out_type == FP32)
      dequant_vector_kernel<FMT, FP32, 4><<<grid, 512>>>(data, scales, global, out, p);
    else if (out_type == FP16)
      dequant_vector_kernel<FMT, FP16, 4><<<grid, 512>>>(data, scales, global, out, p);
    else
      dequant_vector_kernel<FMT, BF16, 4><<<grid, 512>>>(data, scales, global, out, p);
    return;
  }
  if (out_type == FP32 && FMT == NVFP4) {
    dequant_vector_kernel<FMT, FP32, 8><<<(p.count() + 1023) / 1024, 128>>>(
        data, scales, global, out, p);
    return;
  }
#elif defined(__ILUVATAR__)
  // Small outputs favor fewer, wider CTAs; large outputs retain the streamed
  // vector path. Keep the cutoff aligned with the tested quantization boundary.
  int half_threads = p.count() <= 65536 ? 512 : 128;
  if (out_type == FP32 && p.count() <= 65536) {
    dequant_vector_kernel<FMT, FP32, 8><<<(p.count() + 4095) / 4096, 512>>>(
        data, scales, global, out, p);
    return;
  }
#else
  constexpr int half_threads = 128;
#endif
  if (out_type == FP32)
    dequant_vector_kernel<FMT, FP32, 4>
        <<<(p.count() + 1023) / 1024, 256>>>(data, scales, global, out, p);
  else if (out_type == FP16)
    dequant_vector_kernel<FMT, FP16, 8>
        <<<(p.count() + 8 * half_threads - 1) / (8 * half_threads), half_threads>>>(
            data, scales, global, out, p);
  else
    dequant_vector_kernel<FMT, BF16, 8>
        <<<(p.count() + 8 * half_threads - 1) / (8 * half_threads), half_threads>>>(
            data, scales, global, out, p);
}
inline void dequantize(const uint8_t *data, const uint8_t *scales, float global,
                       void *out, int out_type, Layout p) {
  if (p.rows <= 2147483647 && (p.cols + 255) / 256 <= 65535 &&
      p.block == (p.fmt == MXFP8 ? 32 : 16)) {
    if (p.cols % p.block == 0) {
      if (p.fmt == MXFP8)
        launch_dequant_vector<MXFP8>(data, scales, global, out, out_type, p);
      else
        launch_dequant_vector<NVFP4>(data, scales, global, out, out_type, p);
      return;
    }
    size_t bytes_per_row = p.fmt == MXFP8 ? p.cols : (p.cols + 1) / 2;
    dim3 grid(p.rows, (bytes_per_row + 255) / 256);
    if (p.fmt == MXFP8)
      dequant_blocks_kernel<MXFP8>
          <<<grid, 256>>>(data, scales, global, out, out_type, p);
    else
      dequant_blocks_kernel<NVFP4>
          <<<grid, 256>>>(data, scales, global, out, out_type, p);
    return;
  }
  dequant_kernel<<<(p.count() + 255) / 256, 256>>>(data, scales, global, out, out_type,
                                                   p);
}
} // namespace lp
