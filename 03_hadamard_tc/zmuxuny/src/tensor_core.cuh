#pragma once
#include "../include/gpu.cuh"

#ifndef LP_CUDA_ARCH
#define LP_CUDA_ARCH 75
#endif

#if defined(__MACACC__)
#include "metax_matrix.cuh"
#elif defined(__ILUVATAR__)
#include "iluvatar_matrix.cuh"
#elif defined(__MUSACC__)
#include "musa_matrix.cuh"
#endif

namespace lp {
#if defined(__MACACC__) || defined(__ILUVATAR__) || defined(__MUSACC__) || LP_CUDA_ARCH >= 80
#if !defined(__MACACC__) && !defined(__ILUVATAR__) && !defined(__MUSACC__)
// Documented PTX m16n8k16 fragment layout, not an opaque WMMA layout assumption.
// Two MMA instructions transform 16 independent 16-element segments. Remaining
// Hadamard factors act on the segment index using shuffles and FP32 registers.
template <int TYPE>
__device__ __forceinline__ void mma_h16(float *c, const unsigned *a, unsigned b0,
                                        unsigned b1) {
  if constexpr (TYPE == FP16) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%10,%10,%10};"
                 : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1),
                   "f"(0.0f));
  } else {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%10,%10,%10};"
                 : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1),
                   "f"(0.0f));
  }
}

template <int TYPE> __device__ __forceinline__ unsigned h_pair(int k, int column) {
  constexpr unsigned ONE = TYPE == FP16 ? 0x3c00U : 0x3f80U;
  unsigned lo = ONE | ((__popc(unsigned(k & column)) & 1) << 15);
  unsigned hi = ONE | ((__popc(unsigned((k + 1) & column)) & 1) << 15);
  return lo | (hi << 16);
}

template <int D, int TYPE, int MODE, int FMT, int WARPS = 4>
__global__ void hadamard_mma_kernel(const uint16_t *x, uint16_t *y, size_t n,
                                    bool normalize, bool signs, uint32_t seed,
                                    uint8_t *data, uint8_t *scales, float *amax,
                                    Layout p) {
  constexpr int TILES = D > 256 ? D / 256 : 1;
  constexpr int ITEMS = D > 256 ? D : 256;
  int lane = threadIdx.x & 31, group = lane >> 2, pair = (lane & 3) * 2;
  size_t start = (size_t(blockIdx.x) * WARPS + threadIdx.x / 32) * ITEMS;
  // Every lane must participate in MMA, including zero-padded tail rows.
  if (start >= n && MODE != 1 && MODE != 3)
    return;
  float v[TILES][8];
  unsigned b00 = h_pair<TYPE>(pair, group), b01 = h_pair<TYPE>(pair + 8, group);
  unsigned b10 = h_pair<TYPE>(pair, group + 8), b11 = h_pair<TYPE>(pair + 8, group + 8);
#pragma unroll
  for (int t = 0; t < TILES; ++t) {
    unsigned a[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      size_t i = start + t * 256 + (group + (j & 1) * 8) * 16 + pair + (j / 2) * 8;
      a[j] = i < n ? *reinterpret_cast<const unsigned *>(x + i) : 0;
      if (signs) {
        int col = int(i % D);
        unsigned mask = (uniform(col, seed) < 0.5f ? 0x8000U : 0U) |
                        (uniform(col + 1, seed) < 0.5f ? 0x80000000U : 0U);
        a[j] ^= mask;
      }
    }
    mma_h16<TYPE>(v[t], a, b00, b01);
    mma_h16<TYPE>(v[t] + 4, a, b10, b11);
  }
#pragma unroll
  for (int step = 1; step < 8 && step < D / 16; step *= 2) {
#pragma unroll
    for (int t = 0; t < TILES; ++t) {
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        float other = __shfl_xor_sync(0xffffffff, v[t][j], step * 4);
        v[t][j] = (group & step) ? other - v[t][j] : v[t][j] + other;
      }
    }
  }
  if constexpr (D >= 256) {
#pragma unroll
    for (int t = 0; t < TILES; ++t) {
#pragma unroll
      for (int j = 0; j < 8; ++j)
        if (!(j & 2)) {
          float a = v[t][j], b = v[t][j + 2];
          v[t][j] = a + b;
          v[t][j + 2] = a - b;
        }
    }
  }
#pragma unroll
  for (int step = 1; step < TILES; step *= 2) {
#pragma unroll
    for (int t = 0; t < TILES; ++t)
      if (!(t & step)) {
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          float a = v[t][j], b = v[t + step][j];
          v[t][j] = a + b;
          v[t + step][j] = a - b;
        }
      }
  }
  float normalization = normalize ? 1.0f / sqrtf(float(D)) : 1.0f;
#pragma unroll
  for (int t = 0; t < TILES; ++t)
#pragma unroll
    for (int j = 0; j < 8; ++j)
      v[t][j] =
          MODE == 0 ? v[t][j] * normalization : cast(v[t][j] * normalization, TYPE);
  if constexpr (MODE == 1 || MODE == 3) {
    float a = 0;
#pragma unroll
    for (int t = 0; t < TILES; ++t)
#pragma unroll
      for (int j = 0; j < 8; ++j)
        a = fmaxf(a, fabsf(v[t][j]));
    a = warp_max(a);
    __shared__ float maxima[WARPS];
    if (lane == 0)
      maxima[threadIdx.x / 32] = a;
    __syncthreads();
    if (threadIdx.x < 32) {
      a = warp_max(lane < WARPS ? maxima[lane] : 0.0f);
      if (lane == 0)
        atomicMax(reinterpret_cast<unsigned *>(amax), __float_as_uint(a));
    }
  }
  if constexpr (MODE == 2) {
    constexpr int B = FMT == MXFP8 ? 32 : 16;
    constexpr int GROUPS_PER_ROW = (D + B - 1) / B;
    float global = FMT == NVFP4 ? global_scale(*amax) : 1.0f;
#pragma unroll
    for (int t = 0; t < TILES; ++t) {
#pragma unroll
      for (int halfrow = 0; halfrow < 2; ++halfrow) {
        int j = halfrow * 2;
        float a = fmaxf(fmaxf(fabsf(v[t][j]), fabsf(v[t][j + 1])),
                        fmaxf(fabsf(v[t][j + 4]), fabsf(v[t][j + 5])));
        a = fmaxf(a, __shfl_xor_sync(0xffffffff, a, 1));
        a = fmaxf(a, __shfl_xor_sync(0xffffffff, a, 2));
        if constexpr (FMT == MXFP8 && D >= 32)
          a = fmaxf(a, __shfl_xor_sync(0xffffffff, a, 4));
        uint8_t s = FMT == MXFP8 ? mx_scale(a) : encode8((a / global) / 6.0f);
        float scale = scale_value(s, FMT);
        size_t base = start + t * 256 + (group + halfrow * 8) * 16;
        if (base < n && pair == 0 && (FMT == NVFP4 || D == 16 || !(group & 1)))
          scales[(base / D) * GROUPS_PER_ROW + (base % D) / B] = s;
#pragma unroll
        for (int upper = 0; upper < 2; ++upper) {
          size_t i = base + pair + upper * 8;
          unsigned codes[2];
#if LP_NATIVE_FP8
          if constexpr (FMT == MXFP8) {
            uint16_t pair =
                encode8_pair(mx_scaled(v[t][j + upper * 4], s),
                             mx_scaled(v[t][j + upper * 4 + 1], s), p.stochastic,
                             p.stochastic ? uniform(i, p.seed) : 0,
                             p.stochastic ? uniform(i + 1, p.seed) : 0);
            codes[0] = pair & 255;
            codes[1] = pair >> 8;
          } else
#endif
          {
#pragma unroll
            for (int k = 0; k < 2; ++k) {
              float value = v[t][j + upper * 4 + k];
              if constexpr (FMT == NVFP4)
                value /= global;
              float z =
                  FMT == MXFP8 ? mx_scaled(value, s) : (scale == 0 ? 0 : value / scale);
              float u = p.stochastic ? uniform(i + k, p.seed) : 0;
              codes[k] = FMT == MXFP8 ? encode8(z, p.stochastic, u)
                                      : encode4_scaled(value, scale, p.stochastic, u);
            }
          }
          if (i < n) {
            if constexpr (FMT == MXFP8)
              *reinterpret_cast<uint16_t *>(data + i) =
                  uint16_t(codes[0] | (codes[1] << 8));
            else
              data[i / 2] = uint8_t(codes[0] | (codes[1] << 4));
          }
        }
      }
    }
  } else if constexpr (MODE != 1) {
#pragma unroll
    for (int t = 0; t < TILES; ++t) {
#pragma unroll
      for (int j = 0; j < 8; j += 2) {
        size_t i = start + t * 256 + (group + ((j & 2) ? 8 : 0)) * 16 + pair +
                   (j >= 4 ? 8 : 0);
        if (i < n) {
          unsigned bits;
          if constexpr (TYPE == FP16) {
            __half2 h = __floats2half2_rn(v[t][j], v[t][j + 1]);
            bits = *reinterpret_cast<unsigned *>(&h);
          } else {
            // PTX packs its first source into the high half, second into low.
            asm("cvt.rn.bf16x2.f32 %0, %1, %2;"
                : "=r"(bits)
                : "f"(v[t][j + 1]), "f"(v[t][j]));
          }
          *reinterpret_cast<unsigned *>(y + i) = bits;
        }
      }
    }
  }
}

#endif // NVIDIA-specific fragment implementation
#if defined(__MUSACC__)
constexpr int MMA_LANES = 128;
#elif defined(__MACACC__) || defined(__ILUVATAR__)
constexpr int MMA_LANES = 64;
#else
constexpr int MMA_LANES = 32;
#endif

template <int TYPE, int MODE, int FMT>
inline void launch_had_mma_type(const void *x, void *y, size_t rows, int d, bool norm,
                                bool signs, uint32_t seed, uint8_t *data,
                                uint8_t *scales, float *amax, Layout p) {
  size_t n = rows * d;
#if defined(__MUSACC__)
  // S4000 scans favor two physical waves for large standalone transforms.
  // Full fusion has higher register pressure and uses one wave at D>=512.
#define MMA_CASE(D)                                                             \
  case D: {                                                                     \
    constexpr int W = MODE == 2                                                \
                          ? ((D >= 1024 || (FMT == NVFP4 && D >= 512)) ? 1      \
                             : (FMT == NVFP4 && D >= 128) ? 2 : 4)             \
                          : (D >= 1024 ? 2 : 4);                               \
    constexpr int ITEMS = D > 256 ? D : 256;                                    \
    hadamard_mma_kernel<D, TYPE, MODE, FMT, W>                                   \
        <<<(n + W * ITEMS - 1) / (W * ITEMS), W * MMA_LANES>>>(                  \
            static_cast<const uint16_t *>(x), static_cast<uint16_t *>(y), n,    \
            norm, signs, seed, data, scales, amax, p);                           \
    break;                                                                      \
  }
#elif defined(__ILUVATAR__)
  // One wave per CTA improves large MXFP8 fusion; retain four for the
  // bandwidth-bound standalone transform and the NVFP4 reduction pipeline.
#define MMA_CASE(D)                                                             \
  case D: {                                                                     \
    constexpr int W = MODE == 2 && FMT == MXFP8 && D >= 256 ? 1 : 4;            \
    constexpr int ITEMS = D > 256 ? D : 256;                                    \
    hadamard_mma_kernel<D, TYPE, MODE, FMT, W>                                   \
        <<<(n + W * ITEMS - 1) / (W * ITEMS), W * MMA_LANES>>>(                  \
            static_cast<const uint16_t *>(x), static_cast<uint16_t *>(y), n,    \
            norm, signs, seed, data, scales, amax, p);                           \
    break;                                                                      \
  }
#else
#define MMA_CASE(D)                                                             \
  case D:                                                                       \
    hadamard_mma_kernel<D, TYPE, MODE, FMT>                                      \
        <<<(n + 4 * (D > 256 ? D : 256) - 1) / (4 * (D > 256 ? D : 256)),        \
           4 * MMA_LANES>>>(                                                    \
            static_cast<const uint16_t *>(x), static_cast<uint16_t *>(y), n,    \
            norm, signs, seed, data, scales, amax, p);                           \
    break
#endif
  switch (d) {
    MMA_CASE(16);
    MMA_CASE(32);
    MMA_CASE(64);
    MMA_CASE(128);
    MMA_CASE(256);
    MMA_CASE(512);
    MMA_CASE(1024);
  }
#undef MMA_CASE
}
template <int MODE, int FMT>
inline void launch_had_mma_mode(const void *x, void *y, size_t rows, int d, int dtype,
                                bool norm, bool signs, uint32_t seed, uint8_t *data,
                                uint8_t *scales, float *amax, Layout p) {
  if (dtype == FP16)
    launch_had_mma_type<FP16, MODE, FMT>(x, y, rows, d, norm, signs, seed, data, scales,
                                         amax, p);
  else
    launch_had_mma_type<BF16, MODE, FMT>(x, y, rows, d, norm, signs, seed, data, scales,
                                         amax, p);
}
inline void launch_had_mma(const void *x, void *y, size_t rows, int d, int dtype,
                           bool norm, bool signs, uint32_t seed) {
  launch_had_mma_mode<0, MXFP8>(x, y, rows, d, dtype, norm, signs, seed, nullptr,
                                nullptr, nullptr, Layout{});
}
inline void fused_had_mma(const void *x, size_t rows, int d, int dtype, bool norm,
                          bool signs, uint32_t seed, uint8_t *data, uint8_t *scales,
                          float *amax, Layout p) {
  if (p.fmt == NVFP4) {
    cudaMemsetAsync(amax, 0, sizeof(float));
    launch_had_mma_mode<1, NVFP4>(x, nullptr, rows, d, dtype, norm, signs, seed,
                                  nullptr, nullptr, amax, p);
    launch_had_mma_mode<2, NVFP4>(x, nullptr, rows, d, dtype, norm, signs, seed, data,
                                  scales, amax, p);
  } else
    launch_had_mma_mode<2, MXFP8>(x, nullptr, rows, d, dtype, norm, signs, seed, data,
                                  scales, amax, p);
}
// Two GPU kernels: transform+amax writes the rounded intermediate once;
// vector quantization consumes it without a second amax or transform pass.
inline void materialized_had_mma(const void *x, void *scratch, size_t rows, int d,
                                 int dtype, bool norm, bool signs, uint32_t seed,
                                 uint8_t *data, uint8_t *scales, float *amax,
                                 Layout p) {
  cudaMemsetAsync(amax, 0, sizeof(float));
  launch_had_mma_mode<3, NVFP4>(x, scratch, rows, d, dtype, norm, signs, seed, nullptr,
                                nullptr, amax, p);
  quantize_with_amax(scratch, data, scales, amax, p);
}

#endif
} // namespace lp
