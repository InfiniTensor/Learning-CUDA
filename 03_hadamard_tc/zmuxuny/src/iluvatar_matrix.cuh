#pragma once
#include <mma.h>

#if __IVCORE_ARCH__ != 0 && __IVCORE_ARCH__ != 11
#error "This native fragment layout requires CoreX ivcore11 (tested on MR-V100)"
#endif

namespace lp {
// CoreX 4.4 ivcore11 native m16n16k16 matrix operation. Register coordinates
// follow crt/iluvatar_mma.hpp: __hmma_ld_{row,col}_b16 and __imma_st_row_b32.
// These are 64-lane fragments, incompatible with NVIDIA PTX and MACA layouts.
// Constructing registers directly avoids the SDK's swizzled memory layout.
template <int TYPE>
__device__ __forceinline__ void iluvatar_mma_h16(float *out, const uint16_t *a,
                                                 const uint16_t *b) {
  v4f32 zero = {0, 0, 0, 0};
  v4f32 result;
  if constexpr (TYPE == FP16) {
    v4f16 av, bv;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      av[j] = __builtin_bit_cast(__fp16, a[j]);
      bv[j] = __builtin_bit_cast(__fp16, b[j]);
    }
    result = __ivcorex_matrix_mad_f32x4_f16x4(av, bv, zero);
  } else {
    v4bf16 av, bv;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      av[j] = __builtin_bit_cast(__bf16, a[j]);
      bv[j] = __builtin_bit_cast(__bf16, b[j]);
    }
    result = __ivcorex_matrix_mad_f32x4_bf16x4(av, bv, zero);
  }
#pragma unroll
  for (int j = 0; j < 4; ++j)
    out[j] = result[j];
}

template <int D, int TYPE, int MODE, int FMT, int WARPS = 4>
__global__ void hadamard_mma_kernel(const uint16_t *x, uint16_t *y, size_t n,
                                    bool normalize, bool signs, uint32_t seed,
                                    uint8_t *data, uint8_t *scales, float *amax,
                                    Layout p) {
  constexpr int TILES = D > 256 ? D / 256 : 1;
  constexpr int ITEMS = D > 256 ? D : 256;
  constexpr unsigned ONE = TYPE == FP16 ? 0x3c00U : 0x3f80U;
  constexpr unsigned MASK = 0xffffffffU;
  int lane = threadIdx.x % 64, column = lane % 16, group = lane / 16;
  size_t start = (size_t(blockIdx.x) * WARPS + threadIdx.x / 64) * ITEMS;
  if (start >= n && MODE != 1 && MODE != 3)
    return;
  float v[TILES][4];
  // The FP16/BF16 MMA consumes four values per lane. A rows are
  // 2*group + (j%2) + 8*(j/2); B uses the same K indices. Its FP32
  // accumulator instead has rows group + 4*j. No shared-memory transpose.
  uint16_t b[4];
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    int k = 2 * group + (j % 2) + 8 * (j / 2);
    b[j] = ONE | ((__popc(unsigned(k & column)) & 1) << 15);
  }
#pragma unroll
  for (int t = 0; t < TILES; ++t) {
    uint16_t a[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      size_t i = start + t * 256 + (2 * group + (j % 2) + 8 * (j / 2)) * 16 + column;
      a[j] = i < n ? x[i] : 0;
      if (signs && uniform(i % D, seed) < 0.5f)
        a[j] ^= 0x8000U;
    }
    iluvatar_mma_h16<TYPE>(v[t], a, b);
  }
  // In the accumulator, segment bits 0/1 select lane groups and bits
  // 2/3 select registers. Transform only segment bits belonging to this row.
#pragma unroll
  for (int step = 1; step < 4 && step < D / 16; step *= 2)
#pragma unroll
    for (int t = 0; t < TILES; ++t)
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        float other = __shfl_xor_sync(MASK, v[t][j], step * 16, 64);
        v[t][j] = (group & step) ? other - v[t][j] : v[t][j] + other;
      }
#pragma unroll
  for (int step = 1; step < 4 && step < D / 64; step *= 2)
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
        size_t i = start + t * 256 + (group + 4 * j) * 16 + column;
        float a = fabsf(v[t][j]);
        if constexpr (FMT == MXFP8 && D >= 32)
          a = fmaxf(a, __shfl_xor_sync(MASK, a, 16, 64));
#pragma unroll
        for (int step = 8; step; step /= 2)
          a = fmaxf(a, __shfl_xor_sync(MASK, a, step, 16));
        uint8_t s = FMT == MXFP8 ? mx_scale(a) : encode8((a / global) / 6.0f);
        if (i < n && column == 0 && (FMT == NVFP4 || D == 16 || !(group & 1)))
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
        size_t i = start + t * 256 + (group + 4 * j) * 16 + column;
        if (i < n)
          save(y, i, v[t][j], TYPE);
      }
  }
}

// Dense O(D^2) comparison, using the same native H16 operation. Each wave
// owns 16 rows by 16 columns; tails are zero padded before the collective.
__global__ void hadamard_tc(const __half *x, const __half *h, __half *y, size_t rows,
                            int d, bool norm, bool signs, uint32_t seed) {
  int lane = threadIdx.x % 64, group = lane / 16, column = lane % 16;
  size_t row_base = size_t(blockIdx.x) * 16;
  int col_base = (blockIdx.y * 4 + threadIdx.x / 64) * 16;
  if (col_base >= d)
    return;
  // Compensate the dense path's long serial sum to preserve FP16 rounding
  // near midpoints. The fast factorized path uses a short FP32 butterfly.
  float sum[4] = {0, 0, 0, 0}, correction[4] = {0, 0, 0, 0};
  for (int k = 0; k < d; k += 16) {
    uint16_t a[4], b[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      int r = 2 * group + (j % 2) + 8 * (j / 2);
      size_t row = row_base + r;
      a[j] =
          row < rows ? reinterpret_cast<const uint16_t *>(x)[row * d + k + column] : 0;
      if (signs && uniform(k + column, seed) < 0.5f)
        a[j] ^= 0x8000U;
      b[j] = reinterpret_cast<const uint16_t *>(h)[(k + r) * d + col_base + column];
    }
    float tile[4];
    iluvatar_mma_h16<FP16>(tile, a, b);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      float term = tile[j] - correction[j];
      float next = sum[j] + term;
      correction[j] = (next - sum[j]) - term;
      sum[j] = next;
    }
  }
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    size_t row = row_base + group + 4 * j;
    if (row < rows)
      y[row * d + col_base + column] =
          __float2half_rn(sum[j] * (norm ? 1.0f / sqrtf(float(d)) : 1.0f));
  }
}
} // namespace lp
