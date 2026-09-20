#pragma once
#include <mma.h>
#include <type_traits>

namespace lp {
// MACA 3.3 m16n16k16 fragment layout, taken from the SDK's
// __clang_maca_mma_functions.h load/store_matrix_sync implementations.
// A: row=lane%16, col=4*(lane/16)+j.
// B/C: row=4*(lane/16)+j, col=lane%16. All 64 lanes participate.
// This layout is specific to MACA; CUDA uses the separate PTX implementation.
template <int D, int TYPE, int MODE, int FMT, int WARPS = 4>
__global__ void hadamard_mma_kernel(const uint16_t *x, uint16_t *y, size_t n,
                                    bool normalize, bool signs, uint32_t seed,
                                    uint8_t *data, uint8_t *scales, float *amax,
                                    Layout p) {
  using namespace mxmaca;
  using Element = typename std::conditional<TYPE == FP16, __half, maca_bfloat16>::type;
  constexpr int TILES = D > 256 ? D / 256 : 1;
  constexpr int ITEMS = D > 256 ? D : 256;
  constexpr unsigned ONE = TYPE == FP16 ? 0x3c00U : 0x3f80U;
  constexpr unsigned long long MASK = ~0ULL;
  int lane = threadIdx.x % 64, column = lane % 16, group = lane / 16;
  size_t start = (size_t(blockIdx.x) * WARPS + threadIdx.x / 64) * ITEMS;
  if (start >= n && MODE != 1 && MODE != 3)
    return;
  float v[TILES][4];
  wmma::fragment<wmma::matrix_b, 16, 16, 16, Element, wmma::row_major> b;
  static_assert(decltype(b)::num_elements == 4, "MACA fragment layout changed");
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    uint16_t bits = ONE | ((__popc(unsigned((group * 4 + j) & column)) & 1) << 15);
    b.x[j] = __builtin_bit_cast(_Float16, bits);
  }
#pragma unroll
  for (int t = 0; t < TILES; ++t) {
    wmma::fragment<wmma::matrix_a, 16, 16, 16, Element, wmma::row_major> a;
    // Every input segment contains 16 values, so an aligned group of four
    // is either wholly present or wholly padding, including tail rows.
    size_t base = start + t * 256 + column * 16 + group * 4;
    uint2 packed = base < n ? *reinterpret_cast<const uint2 *>(x + base)
                           : make_uint2(0, 0);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      size_t i = base + j;
      uint16_t bits = uint16_t((j < 2 ? packed.x : packed.y) >> ((j & 1) * 16));
      if (signs && uniform(i % D, seed) < 0.5f)
        bits ^= 0x8000U;
      a.x[j] = __builtin_bit_cast(_Float16, bits);
    }
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c;
    wmma::fill_fragment(c, 0.0f);
    wmma::mma_sync(c, a, b, c);
#pragma unroll
    for (int j = 0; j < 4; ++j)
      v[t][j] = c.x[j];
  }
  // Segment-index butterfly: low two bits are registers, next two bits lanes.
#pragma unroll
  for (int step = 1; step < 4 && step < D / 16; step *= 2)
#pragma unroll
    for (int t = 0; t < TILES; ++t)
#pragma unroll
      for (int j = 0; j < 4; ++j)
        if (!(j & step)) {
          float a = v[t][j], b = v[t][j + step];
          v[t][j] = a + b;
          v[t][j + step] = a - b;
        }
#pragma unroll
  for (int step = 1; step < 4 && step < D / 64; step *= 2)
#pragma unroll
    for (int t = 0; t < TILES; ++t)
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        float other = __shfl_xor_sync(MASK, v[t][j], step * 16, 64);
        v[t][j] = (group & step) ? other - v[t][j] : v[t][j] + other;
      }
#pragma unroll
  for (int step = 1; step < TILES; step *= 2)
#pragma unroll
    for (int t = 0; t < TILES; ++t)
      if (!(t & step))
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          float a = v[t][j], b = v[t + step][j];
          v[t][j] = a + b;
          v[t + step][j] = a - b;
        }
  float factor = normalize ? 1.0f / sqrtf(float(D)) : 1.0f;
#pragma unroll
  for (int t = 0; t < TILES; ++t)
#pragma unroll
    for (int j = 0; j < 4; ++j)
      v[t][j] = MODE == 0 ? v[t][j] * factor : cast(v[t][j] * factor, TYPE);
  if constexpr (MODE == 1 || MODE == 3) {
    float a = 0;
#pragma unroll
    for (int t = 0; t < TILES; ++t)
#pragma unroll
      for (int j = 0; j < 4; ++j)
        a = fmaxf(a, fabsf(v[t][j]));
#pragma unroll
    for (int step = 32; step; step /= 2)
      a = fmaxf(a, __shfl_xor_sync(MASK, a, step, 64));
    __shared__ float maxima[WARPS];
    if (lane == 0)
      maxima[threadIdx.x / 64] = a;
    __syncthreads();
    if (threadIdx.x < 64) {
      a = lane < WARPS ? maxima[lane] : 0;
#pragma unroll
      for (int step = 32; step; step /= 2)
        a = fmaxf(a, __shfl_xor_sync(MASK, a, step, 64));
      if (lane == 0)
        atomicMax(reinterpret_cast<unsigned *>(amax), __float_as_uint(a));
    }
  }
  if constexpr (MODE == 2) {
    constexpr int B = FMT == MXFP8 ? 32 : 16;
    constexpr int GROUPS_PER_ROW = (D + B - 1) / B;
    float global = FMT == NVFP4 ? global_scale(*amax) : 1.0f;
#pragma unroll
    for (int t = 0; t < TILES; ++t)
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        size_t i = start + t * 256 + (group * 4 + j) * 16 + column;
        float a = fabsf(v[t][j]);
        if constexpr (FMT == MXFP8 && D >= 32)
          a = fmaxf(a, fabsf(v[t][j ^ 1]));
#pragma unroll
        for (int step = 8; step; step /= 2)
          a = fmaxf(a, __shfl_xor_sync(MASK, a, step, 16));
        uint8_t s = FMT == MXFP8 ? mx_scale(a) : encode8((a / global) / 6.0f);
        if (i < n && column == 0 && (FMT == NVFP4 || D == 16 || !(j & 1)))
          scales[(i / D) * GROUPS_PER_ROW + (i % D) / B] = s;
        float u = p.stochastic ? uniform(i, p.seed) : 0;
        unsigned code = FMT == MXFP8 ? encode8(mx_scaled(v[t][j], s), p.stochastic, u)
                                     : encode4_scaled(v[t][j] / global, fp8_value(s),
                                                      p.stochastic, u);
        if constexpr (FMT == MXFP8) {
          if (i < n)
            data[i] = uint8_t(code);
        } else {
          unsigned hi = __shfl_xor_sync(MASK, code, 1, 16);
          if (i < n && !(column & 1))
            data[i / 2] = uint8_t(code | (hi << 4));
        }
      }
  } else if constexpr (MODE != 1) {
#pragma unroll
    for (int t = 0; t < TILES; ++t)
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        size_t i = start + t * 256 + (group * 4 + j) * 16 + column;
        if (i < n)
          save(y, i, v[t][j], TYPE);
      }
  }
}
} // namespace lp
