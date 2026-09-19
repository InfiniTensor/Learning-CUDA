// =============================================================================
//  common.cuh -- device-side helpers shared by every FWHT kernel.
//
//  Conventions used across the project
//  -----------------------------------
//    D            : Hadamard size (== head_dim), a power of two.
//    row          : one length-D vector; the transform is applied per row.
//    kLogD        : log2(D), a compile-time constant in every kernel.
//    kThreadsPerRow (TPR) : lanes cooperating on one row, a power of two.
//    kEltsPerThread (EPT) : D / TPR, elements held by one lane, contiguous.
//
//  Bits of a row index are split between the two axes:
//    bits [0, log2(EPT))  -> "in-register" stage  (lane-local butterfly)
//    bits [log2(EPT), kLogD) -> "cross-lane" stage (warp shuffle / smem / MMA)
// =============================================================================
#pragma once

#include "fhwt/backend.cuh"
#include <cstdint>

namespace fhwt {

// Runtime codes shared between the host API and the device kernels.
enum QuantCode { kQuantNone = 0, kQuantFp8E4M3 = 1, kQuantInt4 = 2 };

// -----------------------------------------------------------------------------
//  Compile-time helpers
// -----------------------------------------------------------------------------
// log2 of a positive compile-time constant (used for template bookkeeping).
constexpr int clog2(int v) { return v > 1 ? 1 + clog2(v >> 1) : 0; }

constexpr int cdiv(int a, int b) { return (a + b - 1) / b; }

__host__ __device__ __forceinline__ constexpr bool is_pow2(int v) {
  return v > 0 && (v & (v - 1)) == 0;
}

// -----------------------------------------------------------------------------
//  Vectorised global memory access types, keyed by byte width.
// -----------------------------------------------------------------------------
template <int kBytes> struct BytesToType;
template <> struct BytesToType<16> {
  using type = uint4;
};
template <> struct BytesToType<8> {
  using type = uint2;
};
template <> struct BytesToType<4> {
  using type = uint32_t;
};
template <> struct BytesToType<2> {
  using type = uint16_t;
};

// -----------------------------------------------------------------------------
//  cuda_fp8-free FP8 (E4M3) software encoder.
//  Kept in the project so that the fused quantisation path builds on any
//  architecture (no sm_89+ hardware conversion required, see topic notes).
// -----------------------------------------------------------------------------
__device__ __forceinline__ uint8_t float_to_e4m3_rne(float x) {
  // E4M3: 1 sign, 4 exponent (bias 7), 3 mantissa bits, no inf, NaN = 0x7F/0xFF.
  uint32_t u = __float_as_uint(x);
  uint32_t sign = (u >> 24) & 0x80u;
  int32_t exp = (int32_t)((u >> 23) & 0xFFu) - 127;
  uint32_t mant = u & 0x7FFFFFu;

  if (exp > 8 || (exp == 8 && mant >= 0x700000u)) { // saturates to +-448
    return (uint8_t)(sign | 0x7Eu);
  }
  if (exp < -10)
    return (uint8_t)sign; // rounds to signed zero

  uint32_t mag;
  if (exp >= -6) {
    // Normal E4M3 range: 3 mantissa bits are kept, the remaining 20 round.
    mag = (uint32_t)((exp + 7) << 3) | (mant >> 20);
    uint32_t rest = mant & 0xFFFFFu;
    if (rest > 0x80000u || (rest == 0x80000u && (mag & 1u)))
      mag += 1u; // RNE
    if (mag > 0x7Eu)
      mag = (fabsf(x) >= 512.0f) ? 0x7Eu : 0x7Eu;
  } else {
    // Subnormal E4M3: value = m * 2^-9.  The fp32 mantissa is scaled by 2^23,
    // so shifting by (14 - exp) converts (1.mant) into units of 2^-9.
    uint32_t full = mant | 0x800000u;
    uint32_t shift = (uint32_t)(14 - exp);
    mag = (shift >= 32) ? 0u : (full >> shift);
    uint32_t rest = (shift >= 32) ? full : (full & ((1u << shift) - 1u));
    uint32_t half = (shift >= 32) ? 0u : (1u << (shift - 1));
    // Rounding may carry into the smallest normal value (mag == 8), which is
    // exactly the correct E4M3 encoding 0x08, so no clamp is applied here.
    if (rest > half || (rest == half && (mag & 1u)))
      mag += 1u;
  }
  return (uint8_t)(sign | (mag & 0x7Fu));
}
__device__ __forceinline__ float e4m3_to_float(uint8_t v) {
  uint32_t sign = (uint32_t)(v & 0x80u) << 24;
  uint32_t exp = (v >> 3) & 0x0Fu;
  uint32_t mant = v & 0x07u;
  if (exp == 0) { // subnormal
    float f = ldexpf((float)mant, -9);
    return sign ? -f : f;
  }
  uint32_t bits = sign | ((exp + 120) << 23) | (mant << 20);
  return __uint_as_float(bits);
}

// -----------------------------------------------------------------------------
//  Small numeric utilities used by the quantisation kernels.
// -----------------------------------------------------------------------------
__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1)
    v = fmaxf(v, __shfl_xor_sync(0xFFFFFFFFu, v, o));
  return v;
}

__device__ __forceinline__ float block_max(float v, float *scratch) {
  int lane = threadIdx.x & 31;
  int warp = threadIdx.x >> 5;
  v = warp_max(v);
  if (lane == 0)
    scratch[warp] = v;
  __syncthreads();
  int nwarps = (blockDim.x + 31) >> 5;
  v = (threadIdx.x < nwarps) ? scratch[threadIdx.x] : -INFINITY;
  if (warp == 0)
    v = warp_max(v);
  return v;
}

// Round-to-nearest-even integer quantisation: |x| -> [0, qmax].
__device__ __forceinline__ int quant_rne(float x, float inv_scale, int qmax) {
  float v = x * inv_scale;
  v = fminf(fmaxf(v, -(float)qmax - 1.0f), (float)qmax);
  // Round-to-nearest-even in one instruction.  `rintf` returns a float that a
  // second instruction then has to convert; for a value already clamped into
  // [-(qmax+1), qmax] the two agree bit for bit (checked by the quantiser
  // tests), and on MUSA `rintf` is documented as a slow round.
  return __float2int_rn(v);
}

} // namespace fhwt
