#pragma once

// 输出正确性比较：与 CPU 值转换后的位模式对照，不执行文件读写。
#include "common/output_type.cuh"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace cuda_output {

// 比较的是 CUDA 与 CPU 反量化结果，不是原始输入的量化损失。
// 16 位结果与转换后的 golden 逐位比较；FP32 有限值使用绝对误差阈值。
template <typename T>
void compare(const std::vector<T>& actual, const std::vector<float>& golden) {
  if (actual.size() != golden.size()) throw std::runtime_error("dequant shape mismatch");
  std::size_t mismatches = 0;
  double max_diff = 0.0;
  for (std::size_t i = 0; i < actual.size(); ++i) {
    const T expected = Format<T>::convert(golden[i]);
    const float a = static_cast<float>(actual[i]);
    const float e = static_cast<float>(expected);
    // NaN 比较类别，不要求 payload 一致；Inf 必须同号，零保留符号。
    bool equal = false;
    if (std::isnan(e)) equal = std::isnan(a);
    else if constexpr (std::is_same_v<T, float>) {
      if (std::isfinite(a) && std::isfinite(e)) {
        const double diff = std::fabs(static_cast<double>(a) - e);
        max_diff = std::max(max_diff, diff);
        equal = diff <= 1e-6 && !(a == 0 && e == 0 && std::signbit(a) != std::signbit(e));
      } else equal = a == e;
    } else {
      equal = std::memcmp(&actual[i], &expected, sizeof(T)) == 0;
    }
    if (!equal) ++mismatches;
  }
  std::cout << "output_dtype=" << Format<T>::name << " element_mismatches=" << mismatches;
  if constexpr (std::is_same_v<T, float>) std::cout << " max_abs_diff=" << max_diff;
  else std::cout << " comparison=rounded_cpu_bits";
  std::cout << '\n';
  if (mismatches) throw std::runtime_error("CUDA/CPU dequant comparison failed");
}

}  // namespace cuda_output
