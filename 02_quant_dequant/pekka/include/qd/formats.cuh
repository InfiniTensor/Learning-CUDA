#pragma once

#include <cmath>
#include <cstdint>

#ifdef __CUDACC__
#define QD_HD __host__ __device__
#else
#define QD_HD
#endif

namespace qd::formats {

// OCP E4M3FN: sign/exponent/mantissa = 1/4/3, exponent bias 7.  Exponent 15
// remains finite for mantissas 0..6, giving the largest value 448.  The two
// patterns with exponent=15,mantissa=7 are NaNs and are never emitted here.
QD_HD inline float decode_e4m3(std::uint8_t bits) {
  const int sign = (bits & 0x80U) ? -1 : 1;
  const int exponent = (bits >> 3U) & 0x0F;
  const int mantissa = bits & 0x07U;
  if (exponent == 0) {
    return sign * std::ldexp(static_cast<float>(mantissa), -9);
  }
  if (exponent == 15 && mantissa == 7) {
    return NAN;
  }
  return sign * std::ldexp(1.0F + static_cast<float>(mantissa) / 8.0F,
                           exponent - 7);
}

// E2M1: sign/exponent/mantissa = 1/2/1, exponent bias 1.  All 16 patterns are
// finite; positive magnitudes are exactly {0,.5,1,1.5,2,3,4,6}.
QD_HD inline float decode_e2m1(std::uint8_t bits) {
  const int sign = (bits & 0x08U) ? -1 : 1;
  const int exponent = (bits >> 1U) & 0x03;
  const int mantissa = bits & 0x01U;
  if (exponent == 0) {
    return sign * (mantissa ? 0.5F : 0.0F);
  }
  return sign * std::ldexp(1.0F + static_cast<float>(mantissa) / 2.0F,
                           exponent - 1);
}

QD_HD inline bool is_nan(float value) { return value != value; }

// Search the small representable set.  On an exact midpoint, an encoding with
// an even least-significant bit wins (round-to-nearest, ties-to-even).
QD_HD inline std::uint8_t encode_e4m3(float value) {
  if (is_nan(value)) return 0;
  const bool negative = std::signbit(value);
  float magnitude = std::fabs(value);
  if (!std::isfinite(magnitude) || magnitude >= 448.0F) {
    return static_cast<std::uint8_t>((negative ? 0x80U : 0U) | 0x7EU);
  }
  std::uint8_t best = 0;
  float best_error = magnitude;
  for (int code = 1; code <= 0x7E; ++code) {
    const float candidate = decode_e4m3(static_cast<std::uint8_t>(code));
    const float error = std::fabs(magnitude - candidate);
    if (error < best_error ||
        (error == best_error && (code & 1) == 0 && (best & 1U) != 0U)) {
      best = static_cast<std::uint8_t>(code);
      best_error = error;
    }
  }
  return static_cast<std::uint8_t>(best | (negative ? 0x80U : 0U));
}

QD_HD inline std::uint8_t encode_e2m1(float value) {
  if (is_nan(value)) return 0;
  const bool negative = std::signbit(value);
  float magnitude = std::fabs(value);
  if (!std::isfinite(magnitude) || magnitude >= 6.0F) {
    return static_cast<std::uint8_t>((negative ? 0x08U : 0U) | 0x07U);
  }
  std::uint8_t best = 0;
  float best_error = magnitude;
  for (int code = 1; code <= 7; ++code) {
    const float candidate = decode_e2m1(static_cast<std::uint8_t>(code));
    const float error = std::fabs(magnitude - candidate);
    if (error < best_error ||
        (error == best_error && (code & 1) == 0 && (best & 1U) != 0U)) {
      best = static_cast<std::uint8_t>(code);
      best_error = error;
    }
  }
  return static_cast<std::uint8_t>(best | (negative ? 0x08U : 0U));
}

// E8M0 stores only an exponent.  We choose the smallest power of two not less
// than ideal_scale, which guarantees that the subsequently scaled E4M3 value
// cannot overflow solely because scale rounding went down.
QD_HD inline std::uint8_t encode_e8m0_ceil(float ideal_scale) {
  if (!(ideal_scale > 0.0F) || is_nan(ideal_scale)) return 127; // scale 1
  if (!std::isfinite(ideal_scale)) return 254;
  int exponent = 0;
  const float fraction = std::frexp(ideal_scale, &exponent);
  int unbiased = exponent - 1;
  if (fraction != 0.5F) ++unbiased;
  if (unbiased < -127) unbiased = -127;
  if (unbiased > 127) unbiased = 127;
  return static_cast<std::uint8_t>(unbiased + 127);
}

QD_HD inline float decode_e8m0(std::uint8_t bits) {
  return std::ldexp(1.0F, static_cast<int>(bits) - 127);
}

// Counter-based integer mixing makes stochastic rounding reproducible without
// maintaining per-thread RNG state.  The first version exposes the seed and
// primitive, while nearest rounding remains the default and reference path.
QD_HD inline float uniform01(std::uint64_t seed, std::uint64_t index) {
  std::uint64_t x = seed + index * 0x9E3779B97F4A7C15ULL;
  x ^= x >> 30U;
  x *= 0xBF58476D1CE4E5B9ULL;
  x ^= x >> 27U;
  x *= 0x94D049BB133111EBULL;
  x ^= x >> 31U;
  return static_cast<float>((x >> 40U) * (1.0 / 16777216.0));
}

// Stochastic variants use the same ordered representable set as nearest
// encoding, then choose the upper neighbor with probability proportional to
// the input's position between the two neighbors. The counter-based random
// value makes CPU and GPU results reproducible for a fixed seed and index.
QD_HD inline std::uint8_t encode_e4m3_stochastic(float value,
                                                  std::uint64_t seed,
                                                  std::uint64_t index) {
  if (is_nan(value)) return 0;
  const bool negative = std::signbit(value);
  const float magnitude = std::fabs(value);
  if (!std::isfinite(magnitude) || magnitude >= 448.0F) {
    return static_cast<std::uint8_t>((negative ? 0x80U : 0U) | 0x7EU);
  }
  std::uint8_t lower = 0;
  std::uint8_t upper = 0x7E;
  for (int code = 0; code <= 0x7E; ++code) {
    const float candidate = decode_e4m3(static_cast<std::uint8_t>(code));
    if (candidate <= magnitude) lower = static_cast<std::uint8_t>(code);
    if (candidate >= magnitude) {
      upper = static_cast<std::uint8_t>(code);
      break;
    }
  }
  if (lower == upper) return static_cast<std::uint8_t>(lower | (negative ? 0x80U : 0U));
  const float lo = decode_e4m3(lower);
  const float hi = decode_e4m3(upper);
  const float probability = (magnitude - lo) / (hi - lo);
  const std::uint8_t selected = uniform01(seed, index) < probability ? upper : lower;
  return static_cast<std::uint8_t>(selected | (negative ? 0x80U : 0U));
}

QD_HD inline std::uint8_t encode_e2m1_stochastic(float value,
                                                  std::uint64_t seed,
                                                  std::uint64_t index) {
  if (is_nan(value)) return 0;
  const bool negative = std::signbit(value);
  const float magnitude = std::fabs(value);
  if (!std::isfinite(magnitude) || magnitude >= 6.0F) {
    return static_cast<std::uint8_t>((negative ? 0x08U : 0U) | 0x07U);
  }
  std::uint8_t lower = 0;
  std::uint8_t upper = 7;
  for (int code = 0; code <= 7; ++code) {
    const float candidate = decode_e2m1(static_cast<std::uint8_t>(code));
    if (candidate <= magnitude) lower = static_cast<std::uint8_t>(code);
    if (candidate >= magnitude) {
      upper = static_cast<std::uint8_t>(code);
      break;
    }
  }
  if (lower == upper) return static_cast<std::uint8_t>(lower | (negative ? 0x08U : 0U));
  const float lo = decode_e2m1(lower);
  const float hi = decode_e2m1(upper);
  const float probability = (magnitude - lo) / (hi - lo);
  const std::uint8_t selected = uniform01(seed, index) < probability ? upper : lower;
  return static_cast<std::uint8_t>(selected | (negative ? 0x08U : 0U));
}

} // namespace qd::formats

#undef QD_HD
