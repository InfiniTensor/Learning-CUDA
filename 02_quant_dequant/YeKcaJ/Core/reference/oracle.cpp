// CPU 扩展参考：保留原公式与枚举舍入，不调用 GPU 编码器。
#include "reference/oracle.h"
#include "common/numeric.h"
#include <cmath>
#include <cstring>
#include <limits>

namespace pipeline {

// CPU oracle 保留枚举编码；随机舍入用线性查找邻值，不复用 GPU 二分算法。
unsigned reference_encode(float v, bool four, const Options& o, std::size_t i) {
  if (!o.stochastic) return four ? cpu_encode2(v) : cpu_encode4(v);
  const auto mag = [&](unsigned code) { return four ? magnitude2(code) : cpu_decode4(code); };
  const unsigned last = four ? 7 : 126, sign = std::signbit(v) ? (four ? 8 : 128) : 0;
  const float x = std::min(std::fabs(v), four ? 6.0f : 448.0f);
  unsigned hi = 0;
  while (hi < last && mag(hi) < x) ++hi;
  unsigned code = hi;
  if (hi && x != mag(hi)) {
    const double p = (static_cast<double>(x) - mag(hi - 1)) / (mag(hi) - mag(hi - 1));
    if (random_unit(o.seed, i) >= p) code = hi - 1;
  }
  return code == 0 && !four ? 0 : code | sign;
}

// 按相同格式规则在 CPU 上生成 packed 数据，用于逐字节校验 GPU 结果。
Packed reference_quantize(const Tensor& t, const Options& o) {
  validate(o);
  const auto n = t.values.size(), group = o.tensor ? std::max<std::size_t>(1, n) : o.block;
  Packed q{t.rows, t.cols, group, o, 1,
           std::vector<std::uint8_t>(data_size(n)),
           std::vector<std::uint8_t>(ceil_div(n, group))};

  float all = 0;
  for (float v : t.values) all = std::max(all, std::fabs(v));
  if (nvfp4 && all > 0)
    q.global = std::max(all / 2688.0f, std::numeric_limits<float>::denorm_min());

  for (std::size_t b = 0; b < q.scales.size(); ++b) {
    const auto end = std::min(n, (b + 1) * group);
    float m = 0;
    for (std::size_t i = b * group; i < end; ++i)
      m = std::max(m, std::fabs(t.values[i]));

    float s;
    if (nvfp4) {
      q.scales[b] = cpu_encode4(m > 0 ? m / (6.0f * q.global) : 0);
      s = q.global * cpu_decode4(q.scales[b]);
    } else {
      const float ratio = m / 448.0f;
      const int e = m == 0 ? 127
                    : ratio == 0 ? 0
                                 : std::clamp(static_cast<int>(std::ceil(std::log2(ratio))) + 127,
                                              0, 254);
      q.scales[b] = e;
      s = std::ldexp(1.0f, e - 127);
    }

    for (std::size_t i = b * group; i < end; ++i) {
      const auto code = reference_encode(s > 0 ? t.values[i] / s : 0, nvfp4, o, i);
      if (nvfp4)
        q.data[i / 2] |= code << ((i % 2) * 4);
      else
        q.data[i] = code;
    }
  }
  return q;
}

// CPU 反量化先返回 FP32；调用方再校验 GPU 的 FP32/FP16/BF16 输出。
std::vector<float> reference_dequantize(const Packed& q) {
  std::vector<float> y(checked_count(q.rows, q.cols));
  for (std::size_t i = 0; i < y.size(); ++i) {
    const unsigned code = nvfp4 ? (q.data[i / 2] >> ((i % 2) * 4)) & 15 : q.data[i];
    const float v = nvfp4 ? (code & 8 ? -magnitude2(code & 7) : magnitude2(code & 7))
                          : cpu_decode4(code);
    const float s = nvfp4 ? q.global * cpu_decode4(q.scales[i / q.group])
                          : std::ldexp(1.0f, q.scales[i / q.group] - 127);
    y[i] = v * s;
  }
  return y;
}

// 同时检查形状、分组、data/scale 字节及 global_scale 的 FP32 位模式。
void compare_packed(const Packed& a, const Packed& b) {
  for (std::size_t i = 0; i < std::min(a.scales.size(), b.scales.size()); ++i)
    if (a.scales[i] != b.scales[i])
      throw std::runtime_error("scale mismatch at " + std::to_string(i) + ": got=" +
          std::to_string(a.scales[i]) + " expected=" + std::to_string(b.scales[i]));
  for (std::size_t i = 0; i < std::min(a.data.size(), b.data.size()); ++i)
    if (a.data[i] != b.data[i])
      throw std::runtime_error("data mismatch at " + std::to_string(i) + ": got=" +
          std::to_string(a.data[i]) + " expected=" + std::to_string(b.data[i]));
  if (a.rows != b.rows || a.cols != b.cols || a.group != b.group ||
      a.data != b.data || a.scales != b.scales || std::memcmp(&a.global, &b.global, 4))
    throw std::runtime_error("pipeline CPU/GPU packed mismatch");
}

}  // namespace pipeline
