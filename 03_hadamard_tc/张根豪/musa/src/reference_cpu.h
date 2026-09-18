// reference_cpu.h — CPU FP32 参考实现（正确性基准）
//
// 说明：
//   - 只对最后一维 head_dim 做快速 Walsh-Hadamard 变换（蝶形，迭代实现），
//     不显式构造完整 Hadamard 矩阵（那是 O(n^2)，仅用于小尺寸对照）；
//   - 全程 FP32 计算；调用方负责先把 FP16/BF16 输入转成 FP32；
//   - 与 GPU kernel 使用完全一致的蝶形顺序，保证"同一变换"的逐位可比性。
//
// 归一化约定：
//   - 非归一化: y = H x
//   - 归一化:   y = H x / sqrt(head_dim)      （默认，见 docs/final_report.md）
#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

// 原地快速 Walsh-Hadamard 变换。
//   data:         [total_tokens * head_dim] 的 FP32 数据
//   head_dim:     2 的幂
//   normalize:    是否除以 sqrt(head_dim)
inline void wht_fp32_inplace(float* data, std::size_t total_tokens,
                             int head_dim, bool normalize) {
  const float scale =
      normalize ? static_cast<float>(1.0 / std::sqrt(static_cast<double>(head_dim)))
                : 1.0f;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (std::size_t t = 0; t < total_tokens; ++t) {
    float* row = data + t * static_cast<std::size_t>(head_dim);
    for (int stride = 1; stride < head_dim; stride <<= 1) {
      const int step = stride << 1;
      for (int i = 0; i < head_dim; i += step) {
        for (int j = 0; j < stride; ++j) {
          const float a = row[i + j];
          const float b = row[i + j + stride];
          row[i + j] = a + b;
          row[i + j + stride] = a - b;
        }
      }
    }
    if (normalize) {
      for (int i = 0; i < head_dim; ++i) {
        row[i] *= scale;
      }
    }
  }
}

// 显式构造 HEAD_DIM x HEAD_DIM Sylvester Hadamard 矩阵并做矩阵乘的对照版本。
// 只用于小尺寸（如 head_dim <= 64、token 数很少）时验证蝶形实现本身，
// 复杂度 O(n^2)，不要在大规模上使用。
inline void hadamard_matmul_small(const float* input, float* output,
                                  std::size_t total_tokens, int head_dim,
                                  bool normalize) {
  // H_1 = [1]; H_{2n} = [[H_n, H_n], [H_n, -H_n]]
  std::vector<float> H(static_cast<std::size_t>(head_dim) * head_dim, 1.0f);
  for (int size = 2; size <= head_dim; size <<= 1) {
    const int half = size >> 1;
    for (int r = 0; r < size; ++r) {
      for (int c = 0; c < size; ++c) {
        const float base = H[(r & (half - 1)) * head_dim + (c & (half - 1))];
        const bool neg = (r & half) && (c & half);
        H[static_cast<std::size_t>(r) * head_dim + c] = neg ? -base : base;
      }
    }
  }
  const float scale =
      normalize ? static_cast<float>(1.0 / std::sqrt(static_cast<double>(head_dim)))
                : 1.0f;
  for (std::size_t t = 0; t < total_tokens; ++t) {
    const float* x = input + t * head_dim;
    float* y = output + t * head_dim;
    for (int r = 0; r < head_dim; ++r) {
      float acc = 0.0f;
      for (int c = 0; c < head_dim; ++c) {
        acc += H[static_cast<std::size_t>(r) * head_dim + c] * x[c];
      }
      y[r] = acc * scale;
    }
  }
}

// 误差指标：全部在 FP32/FP64 域计算，避免低精度比较带来的二次误差。
//
// 注意绝对误差的适用范围：
//   低精度输出的舍入误差与该元素量级成正比（FP16/BF16 是定相对精度格式），
//   因此"绝对误差 < 阈值"只在输出量级为 O(1) 时合理（例如归一化 + 标准正态输入）。
//   对含异常值（outlier）的输入，个别大输出元素的绝对舍入误差可能超过阈值，
//   这是格式本身的极限，不是实现错误——此时应看 max_rel_error：
//     rel_i = |a_i - b_i| / max(1, |b_i|)
struct ErrorMetrics {
  double max_abs_error = 0.0;
  double mean_abs_error = 0.0;
  double rmse = 0.0;
  double max_rel_error = 0.0;  // 平滑相对误差，对任意量级输入都公平
};

inline ErrorMetrics compute_error_metrics(const float* a, const float* b,
                                          std::size_t n) {
  ErrorMetrics m;
  if (n == 0) {
    return m;
  }
  double sum_abs = 0.0;
  double sum_sq = 0.0;
  double max_abs = 0.0;
  double max_rel = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    const double d = static_cast<double>(a[i]) - static_cast<double>(b[i]);
    const double ad = std::fabs(d);
    if (ad > max_abs) max_abs = ad;
    const double denom = std::fmax(1.0, std::fabs(static_cast<double>(b[i])));
    const double rel = ad / denom;
    if (rel > max_rel) max_rel = rel;
    sum_abs += ad;
    sum_sq += d * d;
  }
  m.max_abs_error = max_abs;
  m.mean_abs_error = sum_abs / static_cast<double>(n);
  m.rmse = std::sqrt(sum_sq / static_cast<double>(n));
  m.max_rel_error = max_rel;
  return m;
}
