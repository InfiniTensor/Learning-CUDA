#pragma once
#include "../include/gpu.cuh"
#include <mma.h>
#include <type_traits>

#if defined(__MUSA_ARCH__) && __MUSA_ARCH__ != 220
#error "MUSA fragment mapping is validated for S4000/mp_22 only"
#endif

namespace lp {
// S4000 (mp_22) WMMA is collective across 128 physical lanes, although its
// shuffle API uses 32-lane logical groups. The SDK's store_matrix_sync maps
// accumulator x[j] in lanes 0..31 to (lane/8 + 4*(j/2), lane%8 + 8*(j%2)).
// Use the public fragment load/MMA API, and this SDK 4.3.6 mapping only
// for the following FP32 butterflies and output. No NVIDIA PTX is reused.
template <int D, int TYPE, int MODE, int FMT, int WARPS = 4>
__global__ void hadamard_mma_kernel(const uint16_t *x, uint16_t *y, size_t n,
                                    bool normalize, bool signs, uint32_t seed,
                                    uint8_t *data, uint8_t *scales, float *amax,
                                    Layout p) {
  using namespace mtmusa;
  using Input = typename std::conditional<TYPE == FP16, __half, __mt_bfloat16>::type;
  constexpr int TILES = D > 256 ? D / 256 : 1;
  constexpr int ITEMS = TILES * 256;
  constexpr unsigned ONE = TYPE == FP16 ? 0x3c00U : 0x3f80U;
  __shared__ __align__(32) uint16_t input[WARPS][256];
  __shared__ __align__(32) uint16_t h16[256];
  int wave = threadIdx.x / 128, lane = threadIdx.x % 128;
  int group = lane / 8, col = lane % 8;
  size_t start = (size_t(blockIdx.x) * WARPS + wave) * ITEMS;
  for (int i = threadIdx.x; i < 256; i += blockDim.x)
    h16[i] = uint16_t(ONE | ((__popc(unsigned((i / 16) & (i % 16))) & 1) << 15));
  __syncthreads();
  wmma::fragment<wmma::matrix_b, 16, 16, 16, Input, wmma::row_major> b;
  wmma::load_matrix_sync(b, reinterpret_cast<const Input *>(h16), 16);
  float v[TILES][8];
#pragma unroll
  for (int t = 0; t < TILES; ++t) {
    // All 128 lanes load two values, including zero padding in the final CTA.
    // All CTAs follow the same barriers; no partial-wave early return here.
    for (int j = lane; j < 256; j += 128) {
      size_t i = start + t * 256 + j;
      uint16_t bits = i < n ? x[i] : 0;
      if (signs && uniform(i % D, seed) < 0.5f)
        bits ^= 0x8000U;
      input[wave][j] = bits;
    }
    __syncthreads();
    wmma::fragment<wmma::matrix_a, 16, 16, 16, Input, wmma::row_major> a;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
    wmma::load_matrix_sync(a, reinterpret_cast<const Input *>(input[wave]), 16);
    wmma::fill_fragment(acc, 0.0f);
    wmma::mma_sync(acc, a, b, acc);
#pragma unroll
    for (int j = 0; j < 8; ++j)
      v[t][j] = acc.x[j];
    __syncthreads(); // Do not overwrite shared input until every wave has read it.
  }
  if (lane < 32) {
    // H16 accounts for the low four column bits. The next two factor bits
    // exchange the four row groups; higher bits exchange registers/tiles.
#pragma unroll
    for (int step = 1; step < 4 && step < D / 16; step *= 2)
#pragma unroll
      for (int t = 0; t < TILES; ++t)
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          float other = __shfl_xor_sync(0xffffffffU, v[t][j], step * 8, 32);
          v[t][j] = (group & step) ? other - v[t][j] : v[t][j] + other;
        }
#pragma unroll
    for (int step = 2; step < 8 && step * 32 < D; step *= 2)
#pragma unroll
      for (int t = 0; t < TILES; ++t)
#pragma unroll
        for (int j = 0; j < 8; ++j)
          if (!(j & step)) {
            float a = v[t][j], b = v[t][j + step];
            v[t][j] = a + b;
            v[t][j + step] = a - b;
          }
#pragma unroll
    for (int step = 1; step < TILES; step *= 2)
#pragma unroll
      for (int t = 0; t < TILES; ++t)
        if (!(t & step))
#pragma unroll
          for (int j = 0; j < 8; ++j) {
            float a = v[t][j], b = v[t + step][j];
            v[t][j] = a + b;
            v[t + step][j] = a - b;
          }
    float norm = normalize ? 1.0f / sqrtf(float(D)) : 1.0f;
#pragma unroll
    for (int t = 0; t < TILES; ++t)
#pragma unroll
      for (int j = 0; j < 8; ++j)
        // Preserve materialized FP16/BF16 rounding before quantization.
        v[t][j] = cast(v[t][j] * norm, TYPE);
  }
  if constexpr (MODE == 1 || MODE == 3) {
    __shared__ float maxima[WARPS];
    if (lane < 32) {
      float m = 0;
#pragma unroll
      for (int t = 0; t < TILES; ++t)
#pragma unroll
        for (int j = 0; j < 8; ++j)
          m = fmaxf(m, fabsf(v[t][j]));
      m = warp_max(m);
      if (lane == 0)
        maxima[wave] = m;
    }
    __syncthreads();
    if (threadIdx.x < 32) {
      float m = warp_max(threadIdx.x < WARPS ? maxima[threadIdx.x] : 0.0f);
      if (threadIdx.x == 0)
        atomicMax(reinterpret_cast<unsigned *>(amax), __float_as_uint(m));
    }
  }
  if (lane >= 32)
    return; // WMMA result ownership is restricted to these 32 lanes.
  if constexpr (MODE == 0 || MODE == 3) {
#pragma unroll
    for (int t = 0; t < TILES; ++t)
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        size_t i = start + t * 256 + (group + 4 * (j / 2)) * 16 + col + 8 * (j % 2);
        if (i < n)
          save(y, i, v[t][j], TYPE);
      }
  }
  if constexpr (MODE == 2) {
    constexpr int B = FMT == MXFP8 ? 32 : 16;
    constexpr int GROUPS = (D + B - 1) / B;
    float global = FMT == NVFP4 ? global_scale(*amax) : 1.0f;
#pragma unroll
    for (int t = 0; t < TILES; ++t)
#pragma unroll
      for (int row = 0; row < 4; ++row) {
        float m = fmaxf(fabsf(v[t][row * 2]), fabsf(v[t][row * 2 + 1]));
        m = warp_max(m, 8);
        if constexpr (FMT == MXFP8 && D >= 32)
          m = fmaxf(m, __shfl_xor_sync(0xffffffffU, m, 8, 32));
        uint8_t s =
            FMT == MXFP8 ? mx_scale(m) : encode8(divide_rn(divide_rn(m, global), 6.0f));
        float scale = scale_value(s, FMT);
        size_t base = start + t * 256 + (group + row * 4) * 16;
        if (base < n && col == 0 && (FMT == NVFP4 || D == 16 || !(group & 1)))
          scales[(base / D) * GROUPS + (base % D) / B] = s;
#pragma unroll
        for (int half = 0; half < 2; ++half) {
          size_t i = base + col + half * 8;
          float value = v[t][row * 2 + half];
          float u = p.stochastic ? uniform(i, p.seed) : 0.0f;
          unsigned code = FMT == MXFP8 ? encode8(mx_scaled(value, s), p.stochastic, u)
                                       : encode4_scaled(divide_rn(value, global), scale,
                                                        p.stochastic, u);
          if constexpr (FMT == MXFP8) {
            if (i < n)
              data[i] = uint8_t(code);
          } else {
            unsigned high = __shfl_xor_sync(0xffffffffU, code, 1, 32);
            if (i < n && !(col & 1))
              data[i / 2] = uint8_t(code | (high << 4));
          }
        }
      }
  }
}
} // namespace lp
