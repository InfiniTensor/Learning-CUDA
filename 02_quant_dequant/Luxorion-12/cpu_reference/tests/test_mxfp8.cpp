// 测试模块用途：验证MXFP8从分块、scale到编码及恢复的完整CPU数值链路。
// 输入：手工、随机、边界数据，以及刻意构造的错误量化结果。
// 输出：独立期望与实际结果的检查结论；主流程关系：守住未来CUDA对照基准。
#include "mxfp8.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
// 阅读顺序：test_manual_and_layout -> compare_oracle -> 其余边界/随机测试。
// oracle是独立计算的期望结果，不调用生产格式编解码器，以免两边同错。
// 随机数据在内存中生成；这里不写CSV，查看小样例请运行演示入口。
unsigned checks = 0;
void check(bool condition, const std::string& message) {
  ++checks;
  if (!condition) throw std::runtime_error(message);
}

template <typename Exception, typename Function>
void expect_error(Function function, const std::string& label) {
  bool rejected = false;
  try { function(); } catch (const Exception&) { rejected = true; }
  check(rejected, label);
}

// 测试 oracle：枚举 scale 和 E4M3 候选，不调用被测编解码器。
std::array<float, 127> magnitude_table() {
  std::array<float, 127> values{};
  float step = 1.0F / 512.0F;
  for (unsigned i = 0; i < 8; ++i) values[i] = i * step;
  float base = 1.0F / 64.0F;
  for (unsigned e = 1; e <= 15; ++e) {
    for (unsigned m = 0; m < 8 && e * 8 + m < values.size(); ++m) {
      values[e * 8 + m] = base + m * step;
    }
    base *= 2.0F;
    step *= 2.0F;
  }
  return values;
}

void compare_oracle(const std::vector<float>& input) {
  const auto values = magnitude_table();
  const auto q = luxorion::mxfp8_quantize_cpu(input);
  check(q.data.size() == input.size(), "data must not be padded");
  check(q.scales.size() == input.size() / 32 + (input.size() % 32 != 0),
        "scale count");
  const auto restored = luxorion::mxfp8_dequantize_cpu(q);
  check(restored.size() == input.size(), "restored length");
  for (std::size_t b = 0; b < q.scales.size(); ++b) {
    const std::size_t begin = b * 32;
    const std::size_t end = std::min(begin + 32, input.size());
    float amax = 0.0F;
    for (auto i = begin; i < end; ++i) amax = std::max(amax, std::fabs(input[i]));
    unsigned scale_code = 127;
    double scale = 1.0;
    if (amax != 0.0F) {
      scale_code = 0;
      scale = 0x1p-127;
      const double required = static_cast<double>(amax) / 448.0;
      while (scale < required && scale_code < 254) {
        scale *= 2.0;
        ++scale_code;
      }
    }
    check(q.scales[b] == scale_code, "scale block " + std::to_string(b));
    for (auto i = begin; i < end; ++i) {
      const float normalized = input[i] / static_cast<float>(scale);
      const double magnitude = std::fabs(static_cast<double>(normalized));
      unsigned best = 0;
      double distance = std::numeric_limits<double>::infinity();
      for (unsigned candidate = 0; candidate < values.size(); ++candidate) {
        const double current = std::fabs(magnitude - values[candidate]);
        if (current < distance ||
            (current == distance && candidate % 2 == 0 && best % 2 != 0)) {
          best = candidate;
          distance = current;
        }
      }
      const unsigned sign = std::signbit(normalized) ? 128 : 0;
      check(q.data[i] == (sign | best), "data element " + std::to_string(i));
      const float decoded = sign ? -values[best] : values[best];
      const float expected = decoded * static_cast<float>(scale);
      check(restored[i] == expected && std::signbit(restored[i]) == std::signbit(expected),
            "restored element " + std::to_string(i));
    }
  }
}

void test_manual_and_layout() {
  const auto q = luxorion::mxfp8_quantize_cpu({1.0F, -1.1F, 15.5F, 260.0F});
  check(q.scales == std::vector<std::uint8_t>{127}, "manual scale = 1");
  check(q.data == std::vector<std::uint8_t>{0x38, 0xB9, 0x58, 0x78}, "manual bytes");
  check(luxorion::mxfp8_dequantize_cpu(q) == std::vector<float>{1.0F, -1.125F, 16.0F, 256.0F},
        "manual restoration");
  const auto large = luxorion::mxfp8_quantize_cpu({600.0F});
  check(large.scales[0] == 128, "600: scale rounds up to 2");
  check(large.data[0] == 0x79, "600 / 2 = 300 -> 288");
  check(luxorion::mxfp8_dequantize_cpu(large)[0] == 576.0F, "600 -> 576");
  // 模拟 2x20 行主序矩阵：第二行的最大值仍影响第一个 32 元素块。
  std::vector<float> matrix(40, 1.0F);
  matrix[20] = 448.0F;
  const auto cross_row = luxorion::mxfp8_quantize_cpu(matrix);
  check(cross_row.scales == std::vector<std::uint8_t>{127, 119}, "continuous blocks cross rows");
  check(cross_row.data[0] == 0x38 && cross_row.data[32] == 0x78, "different block scales");
  compare_oracle(matrix);
}

void test_lengths_and_distributions() {
  std::mt19937 engine(1234);
  std::uniform_real_distribution<float> uniform(-2.0F, 2.0F);
  std::normal_distribution<float> normal(0.0F, 2.0F);
  for (std::size_t n : {1U, 31U, 32U, 33U, 40U, 65U, 257U, 1057U}) {
    for (unsigned distribution = 0; distribution < 4; ++distribution) {
      std::vector<float> input(n);
      for (std::size_t i = 0; i < n; ++i) {
        input[i] = distribution == 0 ? (i % 2 ? -0.0F : 0.0F)
            : distribution == 1 ? uniform(engine) : normal(engine);
      }
      if (distribution == 3) input[n / 2] = 6000.0F;
      compare_oracle(input);
    }
  }
}

void test_scale_boundaries() {
  for (int exponent : {-110, -30, -1, 0, 10, 110}) {
    const float threshold = std::ldexp(448.0F, exponent);
    const float inputs[] = {std::nextafter(threshold, 0.0F), threshold,
        std::nextafter(threshold, std::numeric_limits<float>::infinity())};
    for (unsigned i = 0; i < 3; ++i) {
      std::vector<float> block(32, inputs[i]);
      const auto q = luxorion::mxfp8_quantize_cpu(block);
      check(q.scales[0] == exponent + 127 + (i == 2), "power boundary scale");
      compare_oracle(block);
    }
  }
  const auto tiny = luxorion::mxfp8_quantize_cpu({2.0e-40F, -2.0e-40F});
  check(tiny.scales[0] == 0, "tiny block uses minimum E8M0 scale");
  compare_oracle({2.0e-40F, -2.0e-40F});
  compare_oracle({std::numeric_limits<float>::denorm_min(),
                  -std::numeric_limits<float>::denorm_min()});
}

void test_invalid_and_overflow() {
  expect_error<std::invalid_argument>([] { luxorion::mxfp8_quantize_cpu({}); }, "empty input");
  for (float bad : {std::numeric_limits<float>::quiet_NaN(),
                    std::numeric_limits<float>::infinity(),
                    -std::numeric_limits<float>::infinity()}) {
    expect_error<std::invalid_argument>([bad] { luxorion::mxfp8_quantize_cpu({1.0F, bad}); },
                                       "nonfinite input");
  }
  const luxorion::MXFP8Result malformed[] = {
      {{}, {}}, {{0x38}, {}}, {{0x38}, {127, 127}},
      {{0x38}, {255}}, {{0x7F}, {127}}, {{0xFF}, {127}}};
  for (const auto& q : malformed) {
    expect_error<std::invalid_argument>([&q] { luxorion::mxfp8_dequantize_cpu(q); },
                                       "malformed quantized input");
  }
  expect_error<std::overflow_error>([] {
    luxorion::mxfp8_dequantize_cpu({{0x7E}, {254}});
  }, "externally constructed restoration overflow");
  for (float sign : {1.0F, -1.0F}) {
    expect_error<std::overflow_error>([sign] {
      const auto q = luxorion::mxfp8_quantize_cpu({sign * std::numeric_limits<float>::max()});
      luxorion::mxfp8_dequantize_cpu(q);
    }, "finite FP32 maximum may quantize to an overflowing restoration");
  }
}
}  // namespace

int main() {
  try {
    test_manual_and_layout();
    test_lengths_and_distributions();
    test_scale_boundaries();
    test_invalid_and_overflow();
    std::cout << "MXFP8 CPU: " << checks << " checks passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "MXFP8 test failed: " << error.what() << '\n';
    return 1;
  }
}
