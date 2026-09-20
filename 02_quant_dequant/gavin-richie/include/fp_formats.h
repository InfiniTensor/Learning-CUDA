#pragma once
#include <cstdint>
#include <cmath>
#include <cstring>

// ---------------------------------------------------------------------------
// Software emulation of low-precision float formats. No hardware FP8/FP4
// instructions are used anywhere; everything below is plain integer/float
// arithmetic that runs on any CUDA GPU (and on the host for reference tests).
//
//   E4M3 : 4 exponent bits, 3 mantissa bits, no inf, NaN = 0xF7,
//          max finite value +/-448 (OCP MX spec).
//   E5M2 : 5 exponent bits, 2 mantissa bits, IEEE-style inf/NaN,
//          max finite value +/-57344.
//   E2M1 : 2 exponent bits, 1 mantissa bit;
//          magnitudes {0, .5, 1, 1.5, 2, 3, 4, 6} -> codes 0..7.
//   E8M0 : 8-bit power-of-two scale, value = 2^(code - 127).
//
// All float->low precision conversions use round-to-nearest-even, matching
// hardware RNE behaviour. Stochastic rounding is layered on top in the
// quantization kernels by pre-perturbing the input.
// ---------------------------------------------------------------------------

namespace lp {

// ---- FP8 (E4M3 / E5M2) ----------------------------------------------------

__host__ __device__ inline uint8_t fp32_to_fp8(float x, bool e4m3) {
  const int mant_bits = e4m3 ? 3 : 2;
  const int bias = e4m3 ? 7 : 15;
  const float fmax = e4m3 ? 448.0f : 57344.0f;
  const uint8_t max_code = e4m3 ? 0x7E : 0x7B;  // exponent-1 | all mantissa
  const uint8_t nan_code = e4m3 ? 0x7F : 0x7E;  // exp all ones, top mantissa

  uint32_t bits;
  memcpy(&bits, &x, 4);
  const uint8_t sign = (bits >> 24) & 0x80u;
  const float a = fabsf(x);

  if (a != a) return sign | nan_code;          // NaN
  if (a == 0.0f) return sign;                  // zero
  if (a >= fmax) return sign | max_code;       // saturate (incl. +inf)

  const int emin = 1 - bias;                   // min normal exponent
  int e;
  frexpf(a, &e);                               // a = f * 2^e, f in [0.5,1)
  --e;                                         // a = f * 2^(e+1) -> floor(log2(a)) == e

  if (e >= emin) {
    // Normal: quantize mantissa to `mant_bits` fractional bits with RNE.
    // q lands in [2^mant, 2^(mant+1)) as the integer mantissa.
    const float q = ldexpf(a, mant_bits - e);
    int r = (int)rintf(q);
    const int mant_top = 1 << mant_bits;
    if (r == mant_top * 2) { r = mant_top; ++e; }  // rounded up to next binade
    return sign | (uint8_t)((((e - emin + 1) << mant_bits) | (r - mant_top)));
  }
  // Subnormal (or zero): ulp = 2^(emin - mant_bits); r in [0, 2^(mant+1)].
  // r == 2^mant is the min-normal code (exp field 1, mantissa 0), so the
  // plain integer encoding below covers it.
  const float q = ldexpf(a, mant_bits - emin);
  const int r = (int)rintf(q);
  return sign | (uint8_t)r;
}

__host__ __device__ inline float fp8_to_fp32(uint8_t v, bool e4m3) {
  const int mant_bits = e4m3 ? 3 : 2;
  const int bias = e4m3 ? 7 : 15;
  const uint32_t sign = (v & 0x80u) ? 0x80000000u : 0u;
  const uint32_t ef = e4m3 ? (v >> 3) & 0xFu : (v >> 2) & 0x1Fu;
  const uint32_t m = v & ((1u << mant_bits) - 1u);

  float mag;
  if (e4m3) {
    if (ef == 0xF && m == 0x7) {               // NaN
      uint32_t r = 0x7fc00000u; memcpy(&mag, &r, 4); return sign ? -mag : mag;
    }
    if (ef == 0) mag = ldexpf((float)m, 1 - bias - mant_bits);
    else         mag = ldexpf((float)((1u << mant_bits) | m), (int)ef - bias - mant_bits);
  } else {
    if (ef == 0x1F && m) { uint32_t r = 0x7fc00000u; memcpy(&mag, &r, 4); return sign ? -mag : mag; }
    if (ef == 0x1F) { uint32_t r = 0x7f800000u; memcpy(&mag, &r, 4); return sign ? -mag : mag; }
    if (ef == 0) mag = ldexpf((float)m, 1 - bias - mant_bits);
    else         mag = ldexpf((float)((1u << mant_bits) | m), (int)ef - bias - mant_bits);
  }
  return sign ? -mag : mag;
}

// ---- E2M1 (NVFP4 payload) -------------------------------------------------
// code&7 : 0:0  1:.5  2:1  3:1.5  4:2  5:3  6:4  7:6 ;  code&8 = sign
// Midpoints .25,.75,1.25,1.75,2.5,3.5,5 round to the even code.

__host__ __device__ inline uint8_t fp32_to_e2m1(float x) {
  uint32_t bits;
  memcpy(&bits, &x, 4);
  const uint8_t sign = (bits >> 28) & 0x8u;
  const float a = fabsf(x);
  uint8_t code;
  if (a < 0.25f)   code = 0;
  else if (a < 0.75f)  code = 1;   // (a==0.25 -> even 0)
  else if (a < 1.25f)  code = 2;
  else if (a < 1.75f)  code = 3;
  else if (a < 2.5f)   code = 4;
  else if (a < 3.5f)   code = 5;
  else if (a < 5.0f)   code = 6;
  else                 code = 7;
  // exact midpoint corrections (round to even code)
  if (a == 0.25f) code = 0;
  if (a == 0.75f) code = 2;
  if (a == 1.25f) code = 2;
  if (a == 1.75f) code = 4;
  if (a == 2.5f)  code = 4;
  if (a == 3.5f)  code = 6;
  if (a == 5.0f)  code = 6;
  return sign | code;
}

__host__ __device__ inline float e2m1_to_fp32(uint8_t code) {
  static const float tab[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
  const float m = tab[code & 0x7u];
  return (code & 0x8u) ? -m : m;
}

// ---- E8M0 block scale -------------------------------------------------------
// value = 2^(code-127). Encoding takes the power-of-two >= |x| (ceil), so the
// scaled mantissas fall inside [-1, 1) and never overflow the format.

__host__ __device__ inline uint8_t fp32_to_e8m0_ceil(float x) {
  int e;
  const float f = frexpf(x, &e);               // x = f*2^e, f in [0.5,1)
  if (f == 0.5f) --e;                          // x is an exact power of two
  return (uint8_t)(e + 127);                   // 2^e >= x
}

__host__ __device__ inline float e8m0_to_fp32(uint8_t code) {
  return ldexpf(1.0f, (int)code - 127);
}

}  // namespace lp
