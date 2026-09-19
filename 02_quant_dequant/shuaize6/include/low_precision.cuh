#pragma once

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace lpq {

constexpr int kMxBlockSize = 32;
constexpr int kNvBlockSize = 16;
constexpr float kE4M3Max = 448.0F;
constexpr float kE5M2Max = 57344.0F;
constexpr float kE2M1Max = 6.0F;

enum class MxFp8Kind : int {
  kE4M3 = 0,
  kE5M2 = 1,
};

#if defined(__CUDACC__)
#define LPQ_HD __host__ __device__
#else
#define LPQ_HD
#endif

namespace detail {

LPQ_HD inline int round_ties_to_even_positive(float x) {
  const float lower_f = floorf(x);
  const int lower = static_cast<int>(lower_f);
  const float fraction = x - lower_f;
  if (fraction > 0.5F || (fraction == 0.5F && (lower & 1) != 0)) {
    return lower + 1;
  }
  return lower;
}

LPQ_HD inline std::uint8_t encode_fp8(float value, int exponent_bits,
                                      int mantissa_bits, int exponent_bias,
                                      int max_finite_exponent,
                                      int max_finite_mantissa,
                                      std::uint8_t nan_code) {
  const std::uint8_t sign = signbit(value) ? 0x80U : 0U;
  const float magnitude = fabsf(value);
  const int mantissa_levels = 1 << mantissa_bits;
  const std::uint8_t max_finite_code = static_cast<std::uint8_t>(
      (max_finite_exponent << mantissa_bits) | max_finite_mantissa);

  if (isnan(value)) {
    return static_cast<std::uint8_t>(sign | nan_code);
  }
  if (isinf(value)) {
    return static_cast<std::uint8_t>(sign | max_finite_code);
  }
  if (magnitude == 0.0F) {
    return sign;
  }

  const float max_finite =
      ldexpf(1.0F + static_cast<float>(max_finite_mantissa) /
                        static_cast<float>(mantissa_levels),
             max_finite_exponent - exponent_bias);
  if (magnitude >= max_finite) {
    return static_cast<std::uint8_t>(sign | max_finite_code);
  }

  int binary_exponent = 0;
  frexpf(magnitude, &binary_exponent);
  int unbiased_exponent = binary_exponent - 1;
  const int min_normal_exponent = 1 - exponent_bias;

  if (unbiased_exponent < min_normal_exponent) {
    const float subnormal_step =
        ldexpf(1.0F, min_normal_exponent - mantissa_bits);
    const int mantissa =
        round_ties_to_even_positive(magnitude / subnormal_step);
    if (mantissa <= 0) {
      return sign;
    }
    if (mantissa >= mantissa_levels) {
      return static_cast<std::uint8_t>(sign | (1 << mantissa_bits));
    }
    return static_cast<std::uint8_t>(sign | mantissa);
  }

  const float normalized = ldexpf(magnitude, -unbiased_exponent);
  int mantissa = round_ties_to_even_positive(
      (normalized - 1.0F) * static_cast<float>(mantissa_levels));
  if (mantissa == mantissa_levels) {
    mantissa = 0;
    ++unbiased_exponent;
  }

  const int encoded_exponent = unbiased_exponent + exponent_bias;
  if (encoded_exponent > max_finite_exponent ||
      (encoded_exponent == max_finite_exponent &&
       mantissa > max_finite_mantissa)) {
    return static_cast<std::uint8_t>(sign | max_finite_code);
  }
  const int exponent_mask = (1 << exponent_bits) - 1;
  return static_cast<std::uint8_t>(
      sign | ((encoded_exponent & exponent_mask) << mantissa_bits) |
      mantissa);
}

LPQ_HD inline float decode_fp8(std::uint8_t code, int exponent_bits,
                               int mantissa_bits, int exponent_bias,
                               bool has_infinity) {
  const bool negative = (code & 0x80U) != 0;
  const int exponent_mask = (1 << exponent_bits) - 1;
  const int mantissa_mask = (1 << mantissa_bits) - 1;
  const int exponent = (code >> mantissa_bits) & exponent_mask;
  const int mantissa = code & mantissa_mask;

  if (exponent == exponent_mask) {
    if (has_infinity) {
      if (mantissa == 0) {
        return negative ? -INFINITY : INFINITY;
      }
      return NAN;
    }
    if (mantissa == mantissa_mask) {
      return NAN;
    }
  }

  float value = 0.0F;
  if (exponent == 0) {
    value = ldexpf(static_cast<float>(mantissa),
                   1 - exponent_bias - mantissa_bits);
  } else {
    value = ldexpf(1.0F +
                       static_cast<float>(mantissa) /
                           static_cast<float>(1 << mantissa_bits),
                   exponent - exponent_bias);
  }
  return negative ? -value : value;
}

}  // namespace detail

LPQ_HD inline std::uint8_t encode_e4m3(float value) {
  return detail::encode_fp8(value, 4, 3, 7, 15, 6, 0x7FU);
}

LPQ_HD inline float decode_e4m3(std::uint8_t code) {
  return detail::decode_fp8(code, 4, 3, 7, false);
}

LPQ_HD inline std::uint8_t encode_e5m2(float value) {
  return detail::encode_fp8(value, 5, 2, 15, 30, 3, 0x7FU);
}

LPQ_HD inline float decode_e5m2(std::uint8_t code) {
  return detail::decode_fp8(code, 5, 2, 15, true);
}

// Positive E4M3 scale rounded toward +infinity. This prevents the largest
// value in an NVFP4 block from overflowing after division by the scale.
LPQ_HD inline std::uint8_t encode_e4m3_round_up(float value) {
  if (!(value > 0.0F)) {
    return 0U;
  }
  if (value >= kE4M3Max || isinf(value)) {
    return 0x7EU;
  }
  std::uint8_t code = encode_e4m3(value);
  const float decoded = decode_e4m3(code);
  if (decoded < value && code < 0x7EU) {
    ++code;
  }
  return code;
}

LPQ_HD inline float decode_e2m1(std::uint8_t code) {
  const bool negative = (code & 0x8U) != 0;
  const int exponent = (code >> 1) & 0x3;
  const int mantissa = code & 0x1;
  float value = 0.0F;
  if (exponent == 0) {
    value = 0.5F * static_cast<float>(mantissa);
  } else {
    value = ldexpf(1.0F + 0.5F * static_cast<float>(mantissa), exponent - 1);
  }
  return negative ? -value : value;
}

LPQ_HD inline std::uint8_t encode_e2m1(float value) {
  const std::uint8_t sign = signbit(value) ? 0x8U : 0U;
  const float magnitude = fabsf(value);
  if (isnan(value)) {
    return 0U;
  }
  if (isinf(value) || magnitude >= kE2M1Max) {
    return static_cast<std::uint8_t>(sign | 0x7U);
  }

#if defined(__CUDA_ARCH__)
  // The three E2M1 ranges have steps 0.5, 1, and 2. Constructing the
  // multiplier from the range avoids divergent control flow within a warp.
  const int range = static_cast<int>(magnitude >= 2.0F) +
                    static_cast<int>(magnitude >= 4.0F);
  const float multiplier =
      __uint_as_float(0x40000000U - (static_cast<unsigned>(range) << 23));
  const int code = __float2int_rn(magnitude * multiplier) + 2 * range;
  return static_cast<std::uint8_t>(sign | code);
#else
  // Midpoint comparisons implement the same round-to-nearest-even rule for
  // the portable CPU reference path.
  int code = 0;
  code += magnitude > 0.25F;
  code += magnitude >= 0.75F;
  code += magnitude > 1.25F;
  code += magnitude >= 1.75F;
  code += magnitude > 2.5F;
  code += magnitude >= 3.5F;
  code += magnitude > 5.0F;
  return static_cast<std::uint8_t>(sign | code);
#endif
}

LPQ_HD inline float decode_e8m0(std::uint8_t code) {
  return code == 0xFFU ? NAN : ldexpf(1.0F, static_cast<int>(code) - 127);
}

LPQ_HD inline std::uint8_t choose_mx_scale(float block_amax,
                                           MxFp8Kind kind) {
  if (!(block_amax > 0.0F) || !isfinite(block_amax)) {
    return 127U;  // 2^0 for an all-zero/non-finite-only block.
  }
  int exponent = 0;
  frexpf(block_amax, &exponent);
  const int largest_element_power = kind == MxFp8Kind::kE4M3 ? 8 : 15;
  int scale_exponent = (exponent - 1) - largest_element_power;
  if (scale_exponent < -127) {
    scale_exponent = -127;
  }
  if (scale_exponent > 127) {
    scale_exponent = 127;
  }
  return static_cast<std::uint8_t>(scale_exponent + 127);
}

LPQ_HD constexpr std::size_t mx_scale_count(std::size_t count) {
  return (count + kMxBlockSize - 1) / kMxBlockSize;
}

LPQ_HD constexpr std::size_t nv_scale_count(std::size_t count) {
  return (count + kNvBlockSize - 1) / kNvBlockSize;
}

LPQ_HD constexpr std::size_t nv_packed_count(std::size_t count) {
  return (count + 1) / 2;
}

cudaError_t quantize_mxfp8(const float* input, std::size_t count,
                           std::uint8_t* values, std::uint8_t* scales,
                           MxFp8Kind kind, cudaStream_t stream = nullptr);

cudaError_t dequantize_mxfp8(const std::uint8_t* values,
                             const std::uint8_t* scales, std::size_t count,
                             float* output, MxFp8Kind kind,
                             cudaStream_t stream = nullptr);

cudaError_t quantize_nvfp4(const float* input, std::size_t count,
                           std::uint8_t* packed_values,
                           std::uint8_t* block_scales, float* global_scale,
                           cudaStream_t stream = nullptr);

cudaError_t dequantize_nvfp4(const std::uint8_t* packed_values,
                             const std::uint8_t* block_scales,
                             const float* global_scale, std::size_t count,
                             float* output, cudaStream_t stream = nullptr);

#undef LPQ_HD

}  // namespace lpq
