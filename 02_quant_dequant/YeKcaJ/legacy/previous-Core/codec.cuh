#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace pipeline {

#ifdef TEST_NVFP4
constexpr std::size_t kBlockSize = 16;
#else
constexpr std::size_t kBlockSize = 32;
#endif

// 将 CUDA API 错误转成带源码位置的异常，供 main 统一处理。
inline void check_cuda(cudaError_t status, const char* expression, const char* file, int line) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(expression) + " failed at " + file + ":" +
                             std::to_string(line) + ": " + cudaGetErrorString(status));
  }
}

#define CUDA_CHECK(expression) ::pipeline::check_cuda((expression), #expression, __FILE__, __LINE__)

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
// 默认快速编码位于 pipeline.cuh 的 encode()，不能用这里代表默认路径的性能。
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

}  // namespace pipeline
