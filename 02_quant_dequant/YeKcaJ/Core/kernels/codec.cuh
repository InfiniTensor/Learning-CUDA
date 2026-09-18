#pragma once

// 设备编码/解码辅助函数；nearest、随机舍入和内部枚举对照都在这里。
#include <cuda_runtime.h>
#include "common/types.h"
#include "common/numeric.h"

namespace pipeline {

// 从旧版原样抽出的 E4M3FN 解码规则；指数 0 表示 subnormal，0x7f/0xff 为 NaN。
__device__ inline float decode_e4m3(std::uint8_t code) {
  const int sign = (code & 0x80u) ? -1 : 1;
  const int exponent = (code >> 3) & 0x0fu;
  const int mantissa = code & 0x07u;
  if (exponent == 0) return sign * scalbnf(static_cast<float>(mantissa), -9);
  if (exponent == 15 && mantissa == 7) return __int_as_float(0x7fc00000);
  return sign * scalbnf(1.0f + static_cast<float>(mantissa) / 8.0f, exponent - 7);
}

// 内部对照路径：枚举有限 E4M3 编码；只在误差严格更小时更新，保留原中点规则。
// 默认快速编码位于 本文件的 encode()，不能用这里代表默认路径的性能。
__device__ inline std::uint8_t encode_e4m3(float value) {
  if (isnan(value) || value == 0.0f) return 0;
  const float clipped = fminf(fmaxf(value, -448.0f), 448.0f);
  const float positive_infinity = __int_as_float(0x7f800000);
  float best_error = positive_infinity;
  std::uint8_t best = 0;
  for (int code = 0; code < 256; ++code) {
    const float candidate = decode_e4m3(static_cast<std::uint8_t>(code));
    if (!isfinite(candidate)) continue;
    const float error = fabsf(candidate - clipped);
    if (error < best_error) {
      best_error = error;
      best = static_cast<std::uint8_t>(code);
    }
  }
  return best;
}

// 内部 NVFP4 对照编码：枚举 8 个幅值，中点保留较小幅值，并保存负零符号。
__device__ inline std::uint8_t encode_e2m1(float value) {
  constexpr float magnitudes[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
  const bool negative = signbit(value);
  const float magnitude = fabsf(value);
  int best_index = 0;
  float best_error = __int_as_float(0x7f800000);
  for (int i = 0; i < 8; ++i) {
    const float error = fabsf(magnitudes[i] - magnitude);
    if (error < best_error) {
      best_error = error;
      best_index = i;
    }
  }
  return static_cast<std::uint8_t>(best_index | (negative ? 0x8 : 0));
}

__device__ inline float magnitude(unsigned code, bool four) {
  return four ? magnitude2(code) : decode_e4m3(code);
}

// E4M3 nearest 直接截取 FP32 指数/尾数；中点严格向较小幅值舍入，不是 nearest-even。
// 两种格式共用：MXFP8 的元素编码，以及 NVFP4 的 E4M3 block scale 编码。
__device__ inline std::uint8_t encode_e4m3_nearest(float value) {
  const float x = fminf(fabsf(value), 448.0f);
  const unsigned sign = __float_as_uint(value) >> 24 & 0x80u;
  if (x <= 0x1p-10f) return 0;
  if (x >= 448.0f) return 126u | sign;

  const unsigned bits = __float_as_uint(x);
  const unsigned exponent = bits >> 23;
  const unsigned mantissa = bits & 0x7fffffu;
  unsigned code;
  if (exponent < 121) {
    // E4M3 subnormal 间距固定为 2^-9；此处分支的 shift 范围为 21..24。
    const unsigned significand = mantissa | 0x800000u;
    const unsigned shift = 141 - exponent;
    code = (significand >> shift) +
           ((significand & ((1u << shift) - 1)) > (1u << (shift - 1)));
  } else {
    // FP32 保留最高 3 位尾数；舍入进位自然进入下一指数区间。
    code = ((exponent - 120) << 3) + (mantissa >> 20) +
           ((mantissa & 0xfffffu) > 0x80000u);
  }
  return code | sign;
}

// four=true 编码 E2M1，否则 E4M3；baseline 保留枚举，nearest 用直接编码。
// 随机舍入仍使用原来的二分路径；NVFP4 的 E4M3 scale 已在 scale_code() 中
// 改走 encode_e4m3_nearest()，故此处 four=true 的 E2M1 分支才是 NVFP4 编码路径。
__device__ inline std::uint8_t encode(float value, bool four, bool stochastic,
                                    std::uint32_t seed, std::size_t index,
                                    bool baseline = false) {
  if (baseline && !stochastic) {
#ifdef TEST_NVFP4
    if (four) return encode_e2m1(value);
#endif
    return encode_e4m3(value);
  }

  if (!nvfp4 && !four && !stochastic) return encode_e4m3_nearest(value);

  const unsigned last = four ? 7 : 126;
  const unsigned sign = signbit(value) ? (four ? 8 : 128) : 0;
  const float x = fminf(fabsf(value), four ? 6.0f : 448.0f);
  if (x == 0) return four ? sign : 0;

  // E2M1 只有 8 个幅值，固定中点比较比二分更快；严格 > 保留中点下舍入。
  if (four && !stochastic) {
    const unsigned code = (x > .25f) + (x > .75f) + (x > 1.25f) + (x > 1.75f) +
                          (x > 2.5f) + (x > 3.5f) + (x > 5.0f);
    return code | sign;
  }

  unsigned lo = 0, hi = last;
  while (lo + 1 < hi) {
    const unsigned mid = (lo + hi) / 2;
    if (magnitude(mid, four) <= x)
      lo = mid;
    else
      hi = mid;
  }

  const float a = magnitude(lo, four), b = magnitude(hi, four);
  bool upper;
  if (stochastic)
    upper = random_unit(seed, index) < (static_cast<double>(x) - a) / (b - a);
  else
    upper = fabsf(b - x) < fabsf(x - a);
  const unsigned code = upper ? hi : lo;
  // CPU E4M3 枚举先遇到 +0；E2M1 则单独保存符号，包括 -0。
  return code == 0 && !four ? 0 : code | sign;
}

}  // namespace pipeline
