#pragma once
#include "layout.cuh"

namespace lp {
// Aligned vectors preserve element order and rounding. Selected only when the
// row length is a multiple of the canonical 16/32-element quantization block.
template <int TYPE, int V>
__device__ __forceinline__ void load_vector(const void *x, size_t i, float (&v)[V]) {
  if constexpr (TYPE == FP32) {
    if constexpr (V == 2) {
      float2 a = *reinterpret_cast<const float2 *>(static_cast<const float *>(x) + i);
      v[0] = a.x;
      v[1] = a.y;
    } else {
#pragma unroll
      for (int k = 0; k < V; k += 4) {
        float4 a =
            *reinterpret_cast<const float4 *>(static_cast<const float *>(x) + i + k);
        v[k] = a.x;
        v[k + 1] = a.y;
        v[k + 2] = a.z;
        v[k + 3] = a.w;
      }
    }
  } else {
    unsigned packed[V / 2];
    const uint16_t *ptr = static_cast<const uint16_t *>(x) + i;
    if constexpr (V == 2)
      packed[0] = *reinterpret_cast<const unsigned *>(ptr);
    if constexpr (V == 4) {
      uint2 a = *reinterpret_cast<const uint2 *>(ptr);
      packed[0] = a.x;
      packed[1] = a.y;
    }
    if constexpr (V == 8) {
      uint4 a = *reinterpret_cast<const uint4 *>(ptr);
      packed[0] = a.x;
      packed[1] = a.y;
      packed[2] = a.z;
      packed[3] = a.w;
    }
#pragma unroll
    for (int k = 0; k < V; ++k) {
      uint16_t bits = uint16_t(packed[k / 2] >> ((k & 1) * 16));
      v[k] = TYPE == FP16 ? __half2float(__ushort_as_half(bits)) : unbf16(bits);
    }
  }
}

template <int FMT, int TYPE, int V>
__global__ void quant_vector_kernel(const void *x, uint8_t *data, uint8_t *scales,
                                    const float *amax, Layout p) {
  constexpr int B = FMT == MXFP8 ? 32 : 16, WIDTH = B / V;
  size_t i = (size_t(blockIdx.x) * blockDim.x + threadIdx.x) * V;
  bool valid = i < p.rows * p.cols;
  float values[V] = {};
  if (valid)
    load_vector<TYPE, V>(x, i, values);
  float a = 0;
#pragma unroll
  for (int k = 0; k < V; ++k)
    a = fmaxf(a, fabsf(values[k]));
  if (!p.tensor)
    a = warp_max(a, WIDTH);
  float global = FMT == NVFP4 ? global_scale(*amax) : 1;
  unsigned s = 0;
  if (threadIdx.x % WIDTH == 0)
    s = p.tensor ? scales[0] : (FMT == MXFP8 ? mx_scale(a) : encode8(divide_rn(divide_rn(a, global), 6.0f)));
  s = __shfl_sync(FULL_WARP_MASK, s, 0, WIDTH);
  if (valid && !p.tensor && threadIdx.x % WIDTH == 0)
    scales[i / B] = uint8_t(s);
  float scale = scale_value(uint8_t(s), FMT);
  unsigned codes[V];
#if LP_NATIVE_FP8
  if constexpr (FMT == MXFP8) {
#pragma unroll
    for (int k = 0; k < V; k += 2) {
      uint16_t pair = encode8_pair(mx_scaled(values[k], uint8_t(s)),
                                   mx_scaled(values[k + 1], uint8_t(s)), p.stochastic,
                                   p.stochastic ? uniform(i + k, p.seed) : 0,
                                   p.stochastic ? uniform(i + k + 1, p.seed) : 0);
      codes[k] = pair & 255;
      codes[k + 1] = pair >> 8;
    }
  } else
#endif
  {
#pragma unroll
    for (int k = 0; k < V; ++k) {
      float u = p.stochastic ? uniform(i + k, p.seed) : 0;
      if constexpr (FMT == MXFP8)
        codes[k] = encode8(mx_scaled(values[k], uint8_t(s)), p.stochastic, u);
      else
        codes[k] = encode4_scaled(divide_rn(values[k], global), scale, p.stochastic, u);
    }
  }
  if (valid) {
    if constexpr (FMT == NVFP4) {
      unsigned packed = 0;
#pragma unroll
      for (int k = 0; k < V; ++k)
        packed |= codes[k] << (4 * k);
      if constexpr (V == 2)
        data[i / 2] = uint8_t(packed);
      if constexpr (V == 4)
        *reinterpret_cast<uint16_t *>(data + i / 2) = uint16_t(packed);
      if constexpr (V == 8)
        *reinterpret_cast<unsigned *>(data + i / 2) = packed;
    } else {
      if constexpr (V == 2)
        *reinterpret_cast<uint16_t *>(data + i) = uint16_t(codes[0] | (codes[1] << 8));
      else {
#pragma unroll
        for (int k = 0; k < V; k += 4)
          *reinterpret_cast<unsigned *>(data + i + k) = codes[k] | (codes[k + 1] << 8) |
                                                        (codes[k + 2] << 16) |
                                                        (codes[k + 3] << 24);
      }
    }
  }
}

template <int TYPE> __device__ __forceinline__ unsigned pack_pair(float lo, float hi) {
  if constexpr (TYPE == FP16) {
    __half2 h = __floats2half2_rn(lo, hi);
    return *reinterpret_cast<unsigned *>(&h);
  } else {
#if !defined(__MACACC__) && !defined(__ILUVATAR__) && !defined(__MUSACC__) && defined(__CUDA_ARCH__) && \
    __CUDA_ARCH__ >= 800
    unsigned bits;
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(bits) : "f"(hi), "f"(lo));
    return bits;
#else
    return unsigned(bf16(lo)) | (unsigned(bf16(hi)) << 16);
#endif
  }
}

template <int FMT, int TYPE, int V>
__global__ void dequant_vector_kernel(const uint8_t *data, const uint8_t *scales,
                                      float global, void *out, Layout p) {
  constexpr int B = FMT == MXFP8 ? 32 : 16;
  size_t i = (size_t(blockIdx.x) * blockDim.x + threadIdx.x) * V;
  if (i >= p.rows * p.cols)
    return;
  float s = scale_value(scales[p.tensor ? 0 : i / B], FMT);
  unsigned packed[V / 4];
  if constexpr (FMT == MXFP8) {
#pragma unroll
    for (int k = 0; k < V; k += 4)
      packed[k / 4] = *reinterpret_cast<const unsigned *>(data + i + k);
  } else {
    if constexpr (V == 4)
      packed[0] = *reinterpret_cast<const uint16_t *>(data + i / 2);
    if constexpr (V == 8)
      packed[0] = *reinterpret_cast<const unsigned *>(data + i / 2);
  }
  float v[V];
#pragma unroll
  for (int k = 0; k < V; ++k) {
    unsigned code = FMT == MXFP8 ? (packed[k / 4] >> ((k % 4) * 8)) & 255
                                 : (packed[0] >> (k * 4)) & 15;
    v[k] = multiply_rn(multiply_rn(FMT == MXFP8 ? fp8_value(code) : fp4_value(code), s), global);
  }
  if constexpr (TYPE == FP32) {
#pragma unroll
    for (int k = 0; k < V; k += 4)
      *reinterpret_cast<float4 *>(static_cast<float *>(out) + i + k) =
          make_float4(v[k], v[k + 1], v[k + 2], v[k + 3]);
  } else {
#pragma unroll
    for (int k = 0; k < V; k += 4)
      *reinterpret_cast<uint2 *>(static_cast<uint16_t *>(out) + i + k) = make_uint2(
          pack_pair<TYPE>(v[k], v[k + 1]), pack_pair<TYPE>(v[k + 2], v[k + 3]));
  }
}
template <int TYPE, int V>
__global__ void maximum_vector(const void *x, size_t n, float *result) {
  float a = 0;
  for (size_t i = (size_t(blockIdx.x) * blockDim.x + threadIdx.x) * V; i < n;
       i += size_t(blockDim.x) * gridDim.x * V) {
    float v[V] = {};
    if (i + V <= n)
      load_vector<TYPE, V>(x, i, v);
    else {
#pragma unroll
      for (int k = 0; k < V; ++k)
        if (i + k < n)
          v[k] = load(x, i + k, TYPE);
    }
#pragma unroll
    for (int k = 0; k < V; ++k)
      a = fmaxf(a, fabsf(v[k]));
  }
  a = warp_max(a);
  __shared__ float maxima[32];
  if ((threadIdx.x & 31) == 0)
    maxima[threadIdx.x / 32] = a;
  __syncthreads();
  if (threadIdx.x < 32) {
    a = warp_max(threadIdx.x < blockDim.x / 32 ? maxima[threadIdx.x] : 0);
    if (threadIdx.x == 0)
      atomicMax(reinterpret_cast<unsigned *>(result), __float_as_uint(a));
  }
}
} // namespace lp
