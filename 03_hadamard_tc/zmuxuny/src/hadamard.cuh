#pragma once
#include "../include/gpu.cuh"
#include <mma.h>
#if defined(__ILUVATAR__)
#include "iluvatar_matrix.cuh"
#endif

namespace lp {
#if defined(__MACACC__)
constexpr int BUTTERFLY_LANES = 64;
#else
constexpr int BUTTERFLY_LANES = 32;
#endif
// One logical group of W lanes transforms a row. XOR butterflies stay in
// registers: log2(W) stages exchange lanes; remaining stages exchange registers.
template <int D, int MODE, int FMT, int WARPS = 4,
          int W = (D <= 128 ? 32 : BUTTERFLY_LANES)>
__global__ void hadamard_kernel(const void *x, void *y, size_t rows, int dtype,
                                bool normalize, bool random_sign, uint32_t sign_seed,
                                uint8_t *data, uint8_t *scales, float *amax, Layout p) {
  // Small dimensions favor two logical 32-lane rows per C500 wave;
  // larger rows benefit from all 64 lanes and fewer values per thread.
  constexpr int V = (D + W - 1) / W;
  int lane = threadIdx.x % W;
  size_t row = size_t(blockIdx.x) * WARPS + threadIdx.x / W;
  if (row >= rows && MODE != 1)
    return;
  float v[V];
#pragma unroll
  for (int j = 0; j < V; ++j) {
    int c = j * W + lane;
    v[j] = c < D && row < rows ? load(x, row * D + c, dtype) : 0;
    if (random_sign && uniform(c, sign_seed) < 0.5f)
      v[j] = -v[j];
  }
#pragma unroll
  for (int step = 1; step < W && step < D; step *= 2) {
#pragma unroll
    for (int j = 0; j < V; ++j) {
      float other = __shfl_xor_sync(FULL_WARP_MASK, v[j], step, W);
      v[j] = (lane & step) ? other - v[j] : v[j] + other;
    }
  }
#pragma unroll
  for (int step = 1; step < V; step *= 2) {
#pragma unroll
    for (int j = 0; j < V; ++j)
      if (!(j & step)) {
        float a = v[j], b = v[j + step];
        v[j] = a + b;
        v[j + step] = a - b;
      }
  }
#pragma unroll
  for (int j = 0; j < V; ++j)
    v[j] = cast(v[j] * (normalize ? 1.0f / sqrtf(float(D)) : 1.0f), dtype);
  if (MODE == 0) {
#pragma unroll
    for (int j = 0; j < V; ++j)
      if (j * W + lane < D)
        save(y, row * D + j * W + lane, v[j], dtype);
  } else if (MODE == 1) {
    float m = 0;
#pragma unroll
    for (int j = 0; j < V; ++j)
      if (j * W + lane < D)
        m = fmaxf(m, fabsf(v[j]));
    m = warp_max(m, W);
    __shared__ float maxima[WARPS];
    if (lane == 0)
      maxima[threadIdx.x / W] = m;
    __syncthreads();
    if (threadIdx.x < 32) {
      m = warp_max(lane < WARPS ? maxima[lane] : 0.0f);
      if (lane == 0)
        atomicMax(reinterpret_cast<unsigned *>(amax), __float_as_uint(m));
    }
  } else {
    // Fused path uses canonical 32-value MXFP8 / 16-value NVFP4 blocks.
    // Rounding to the transform output dtype before quantizing makes the
    // packed result exactly equal to the unfused, materialized pipeline.
    constexpr int b = FMT == MXFP8 ? 32 : 16;
    float global = FMT == NVFP4 ? global_scale(*amax) : 1;
#pragma unroll
    for (int j = 0; j < V; ++j) {
      int c = j * W + lane;
      float m = warp_max(c < D ? fabsf(v[j]) : 0, b);
      unsigned shared_scale = 0;
      if (lane % b == 0)
        shared_scale = FMT == MXFP8 ? mx_scale(m) : encode8(divide_rn(divide_rn(m, global), 6.0f));
      uint8_t s = uint8_t(__shfl_sync(FULL_WARP_MASK, shared_scale, 0, b));
      float scale = scale_value(s, FMT);
      float z = FMT == NVFP4 ? divide_rn(v[j], global) : v[j];
      float unscaled = z;
      z = FMT == MXFP8 ? mx_scaled(z, s) : (scale == 0 ? 0 : divide_rn(z, scale));
      size_t i = row * D + c;
      uint8_t code =
          FMT == MXFP8
              ? encode8(z, p.stochastic, (p.stochastic ? uniform(i, p.seed) : 0.0f))
              : encode4_scaled(unscaled, scale, p.stochastic,
                               (p.stochastic ? uniform(i, p.seed) : 0.0f));
      if (c < D && lane % b == 0)
        scales[row * ((D + b - 1) / b) + c / b] = s;
      if (FMT == MXFP8) {
        if (c < D)
          data[i] = code;
      } else {
        unsigned hi = __shfl_xor_sync(FULL_WARP_MASK, unsigned(code), 1);
        if (c < D && !(lane & 1))
          data[row * ((D + 1) / 2) + c / 2] = code | ((c + 1 < D ? hi : 0) << 4);
      }
    }
  }
}
template <int MODE, int FMT>
inline void launch_had_format(const void *x, void *y, size_t rows, int d, int dtype,
                              bool norm, bool signs, uint32_t seed, uint8_t *data,
                              uint8_t *scales, float *amax, Layout p) {
#if defined(__ILUVATAR__)
  // Native 64-lane rows reduce register pressure for D>=128. The amax
  // prepass uses more rows per CTA to reduce atomic contention; D=1024
  // benefits from two waves per CTA in the transform/output stages.
#define HAD_CASE(D)                                                             \
  case D: {                                                                     \
    constexpr int W = (MODE == 1 && D <= 128) || D < 128 ? 32 : 64;              \
    constexpr int R = MODE == 1 ? 16 : (D >= 512 ? 2 : 4);                      \
    hadamard_kernel<D, MODE, FMT, R, W><<<(rows + R - 1) / R, R * W>>>(          \
        x, y, rows, dtype, norm, signs, seed, data, scales, amax, p);             \
    break;                                                                      \
  }
#else
#define HAD_CASE(D)                                                             \
  case D:                                                                       \
    hadamard_kernel<D, MODE, FMT>                                                \
        <<<(rows + 3) / 4, 4 * (D <= 128 ? 32 : BUTTERFLY_LANES)>>>(             \
            x, y, rows, dtype, norm, signs, seed, data, scales, amax, p);          \
    break
#endif
  switch (d) {
    HAD_CASE(1);
    HAD_CASE(2);
    HAD_CASE(4);
    HAD_CASE(8);
    HAD_CASE(16);
    HAD_CASE(32);
    HAD_CASE(64);
    HAD_CASE(128);
    HAD_CASE(256);
    HAD_CASE(512);
    HAD_CASE(1024);
  }
#undef HAD_CASE
}
template <int MODE>
inline void launch_had(const void *x, void *y, size_t rows, int d, int dtype, bool norm,
                       bool signs, uint32_t seed, uint8_t *data, uint8_t *scales,
                       float *amax, Layout p) {
  if constexpr (MODE == 2) {
    if (p.fmt == NVFP4) {
      launch_had_format<MODE, NVFP4>(x, y, rows, d, dtype, norm, signs, seed, data,
                                     scales, amax, p);
      return;
    }
  }
  launch_had_format<MODE, MXFP8>(x, y, rows, d, dtype, norm, signs, seed, data, scales,
                                 amax, p);
}
inline void fused_had(const void *x, size_t rows, int d, int dtype, bool norm,
                      bool signs, uint32_t seed, uint8_t *data, uint8_t *scales,
                      float *amax, Layout p) {
  if (p.fmt == NVFP4) {
    cudaMemsetAsync(amax, 0, sizeof(float));
    launch_had<1>(x, nullptr, rows, d, dtype, norm, signs, seed, nullptr, nullptr, amax,
                  p);
  }
  launch_had<2>(x, nullptr, rows, d, dtype, norm, signs, seed, data, scales, amax, p);
}

// MACA and CoreX matrix fragments span 64 lanes; NVIDIA fragments span 32.
// Quantization still groups consecutive values into logical 16/32-value blocks.
#if defined(__MUSACC__)
constexpr int MATRIX_LANES = 128;
#elif defined(__MACACC__) || defined(__ILUVATAR__)
constexpr int MATRIX_LANES = 64;
#else
constexpr int MATRIX_LANES = 32;
#endif

#if defined(__MUSACC__)
// One physical wave per dense-reference CTA permits an explicit block barrier
// around shared-memory loads/stores, including dimensions with a single tile.
constexpr int DENSE_WAVES = 1;
#else
constexpr int DENSE_WAVES = 4;
#endif

#if !defined(__ILUVATAR__)
__device__ inline void dense_matrix_sync() {
#if defined(__MUSACC__)
  __syncthreads();
#else
  __syncwarp();
#endif
}
// Explicit FP16 Tensor Core comparison: dense X*H, 16x16 WMMA tiles.
// Its O(D^2) arithmetic differs from the O(D log D) butterfly algorithm.
__global__ void hadamard_tc(const __half *x, const __half *h, __half *y, size_t rows,
                            int d, bool norm, bool signs, uint32_t seed) {
#if defined(__MUSACC__)
  using namespace mtmusa;
#else
  using namespace nvcuda;
#endif
  __shared__ __align__(32) __half a[DENSE_WAVES][256];
  __shared__ __align__(32) float result[DENSE_WAVES][256];
  int warp = threadIdx.x / MATRIX_LANES, lane = threadIdx.x % MATRIX_LANES;
  int col = (blockIdx.y * DENSE_WAVES + warp) * 16;
  size_t row = blockIdx.x * 16;
  if (col >= d)
    return;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
  wmma::fill_fragment(acc, 0.0f);
  for (int k = 0; k < d; k += 16) {
    for (int i = lane; i < 256; i += MATRIX_LANES) {
      size_t r = row + i / 16;
      int c = k + i % 16;
      float v = r < rows ? __half2float(x[r * d + c]) : 0;
      if (signs && uniform(c, seed) < 0.5f)
        v = -v;
      a[warp][i] = __float2half_rn(v);
    }
    dense_matrix_sync();
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> af;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> bf;
    wmma::load_matrix_sync(af, a[warp], 16);
    wmma::load_matrix_sync(bf, h + k * d + col, d);
    wmma::mma_sync(acc, af, bf, acc);
    dense_matrix_sync();
  }
  wmma::store_matrix_sync(result[warp], acc, 16, wmma::mem_row_major);
  dense_matrix_sync();
  for (int i = lane; i < 256; i += MATRIX_LANES) {
    size_t r = row + i / 16;
    if (r < rows)
      y[r * d + col + i % 16] =
          __float2half_rn(result[warp][i] * (norm ? 1.0f / sqrtf(float(d)) : 1.0f));
  }
}
#endif
} // namespace lp
