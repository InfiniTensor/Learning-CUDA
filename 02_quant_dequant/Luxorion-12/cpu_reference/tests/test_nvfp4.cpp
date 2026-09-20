// 测试模块用途：验证NVFP4两级scale、四位量化、字节打包与尾块填充的完整CPU链路。
// 输入：手工/随机数组及刻意构造的非法布局、scale或极值结果。
// 输出：编码、存储布局和恢复值的检查结论；主流程关系：守住未来GPU对照基准。
#include "nvfp4.h"

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
// 阅读顺序：test_manual_and_tail -> compare_oracle -> 填充/极值/非法结果检查。
// 独立候选表验证scale和元素舍入，算术拼接验证高四位在前的存储约定。
// global/local公式与生产代码遵循同一数值约定；本测试不是外部官方实现对照。
unsigned checks = 0;
void check(bool condition, const std::string& message) {
  ++checks;
  if (!condition) throw std::runtime_error(message);
}

template <typename Exception, typename Function>
void expect_error(Function function, const std::string& message) {
  bool rejected = false;
  try { function(); } catch (const Exception&) { rejected = true; }
  check(rejected, message);
}

std::array<float, 127> e4_table() {
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

// 枚举候选作为测试 oracle，不调用生产编解码器。
template <std::size_t Size>
unsigned nearest(float value, const std::array<float, Size>& table) {
  const double magnitude = std::fabs(static_cast<double>(value));
  if (magnitude >= table.back()) return Size - 1;
  unsigned best = 0;
  double distance = std::numeric_limits<double>::infinity();
  for (unsigned i = 0; i < Size; ++i) {
    const double current = std::fabs(magnitude - table[i]);
    if (current < distance || (current == distance && i % 2 == 0 && best % 2 != 0)) {
      best = i;
      distance = current;
    }
  }
  return best;
}

void compare_oracle(const std::vector<float>& input) {
  const auto e4 = e4_table();
  constexpr std::array<float, 8> e2 = {0.0F, 0.5F, 1.0F, 1.5F, 2.0F, 3.0F, 4.0F, 6.0F};
  const auto q = luxorion::nvfp4_quantize_cpu(input);
  const auto restored = luxorion::nvfp4_dequantize_cpu(q);
  const std::size_t blocks = input.size() / 16 + (input.size() % 16 != 0);
  float global_amax = 0.0F;
  for (float value : input) global_amax = std::max(global_amax, std::fabs(value));
  const float global = global_amax == 0.0F ? 1.0F
      : static_cast<float>(static_cast<double>(global_amax) / 2688.0);
  check(q.element_count == input.size(), "original element count");
  check(q.global_scale == global, "FP32 global scale");
  check(q.block_scales.size() == blocks, "block scale count");
  check(q.packed_data.size() == blocks * 8, "eight bytes per block including tail");
  check(restored.size() == input.size(), "restored length excludes padding");
  std::vector<std::uint8_t> expected_data(blocks * 8, 0);
  for (std::size_t b = 0; b < blocks; ++b) {
    const std::size_t begin = b * 16;
    const std::size_t end = std::min(begin + 16, input.size());
    float block_amax = 0.0F;
    for (auto i = begin; i < end; ++i) block_amax = std::max(block_amax, std::fabs(input[i]));
    const float ideal = block_amax == 0.0F ? 1.0F
        : static_cast<float>((static_cast<double>(block_amax) / 6.0) / global);
    unsigned scale_code = nearest(ideal, e4);
    if (block_amax != 0.0F && scale_code == 0) scale_code = 1;
    check(q.block_scales[b] == scale_code, "E4M3 block scale " + std::to_string(b));
    const float total = global * e4[scale_code];
    for (auto i = begin; i < end; ++i) {
      const float normalized = input[i] / total;
      const unsigned magnitude = nearest(normalized, e2);
      const bool negative = std::signbit(normalized);
      const unsigned code = magnitude + (negative ? 8 : 0);
      // 算术构造 packed 字节，独立检查高四位在前。
      expected_data[i / 2] += static_cast<std::uint8_t>(code * (i % 2 == 0 ? 16 : 1));
      const float decoded = negative ? -e2[magnitude] : e2[magnitude];
      const float expected = decoded * total;
      check(restored[i] == expected && std::signbit(restored[i]) == std::signbit(expected),
            "FP32 restored element " + std::to_string(i));
    }
  }
  for (std::size_t byte = 0; byte < expected_data.size(); ++byte) {
    check(q.packed_data[byte] == expected_data[byte], "packed/padded byte " + std::to_string(byte));
  }
}

void test_manual_and_tail() {
  std::vector<float> input(32, 0.0F);
  input[0] = 2688.0F; // 全局 scale 精确为 1，第一块局部 scale 精确为 448。
  input[16] = 6.0F;
  input[17] = -3.0F;
  input[18] = 2.5F;
  input[19] = 3.5F;
  const auto q = luxorion::nvfp4_quantize_cpu(input);
  check(q.global_scale == 1.0F, "manual global scale = 1");
  check(q.block_scales == std::vector<std::uint8_t>{0x7E, 0x38}, "manual block scales 448,1");
  check(q.packed_data[0] == 0x70 && q.packed_data[8] == 0x7D
        && q.packed_data[9] == 0x46, "manual packing and E2M1 ties");
  const auto restored = luxorion::nvfp4_dequantize_cpu(q);
  check(restored[0] == 2688.0F && restored[16] == 6.0F && restored[17] == -3.0F
        && restored[18] == 2.0F && restored[19] == 4.0F, "manual restoration");
  compare_oracle(input);

  std::vector<float> regression(32, 0.0F);
  regression[0] = 448.0F;
  regression[16] = 98.0F;
  regression[17] = 81.0F;
  const auto hand = luxorion::nvfp4_quantize_cpu(regression);
  check(hand.global_scale == 1.0F / 6.0F, "448/2688 global scale");
  check(hand.block_scales[1] == 0x6C, "ideal 98 -> actual scale 96");
  check(hand.packed_data[8] == 0x77, "98 and 81 both encode as E2M1 six");
  const auto hand_restored = luxorion::nvfp4_dequantize_cpu(hand);
  check(hand_restored[17] == 96.0F && std::fabs(81.0F - hand_restored[17]) == 15.0F,
        "81 -> 96, absolute error 15");
  compare_oracle(regression);

  input.resize(17);
  const auto tail = luxorion::nvfp4_quantize_cpu(input);
  check(tail.packed_data.size() == 16 && tail.packed_data[8] == 0x70,
        "17 elements: padded tail occupies eight bytes, first element in high nibble");
  for (unsigned byte = 9; byte < 16; ++byte) check(tail.packed_data[byte] == 0, "tail zero byte");
  check(luxorion::nvfp4_dequantize_cpu(tail).size() == 17, "only valid tail element restored");
  compare_oracle(input);
}

void test_lengths_distributions_and_tiny_scales() {
  std::mt19937 engine(1234);
  std::uniform_real_distribution<float> uniform(-2.0F, 2.0F);
  std::normal_distribution<float> normal(0.0F, 2.0F);
  for (std::size_t n : {1U, 15U, 16U, 17U, 19U, 31U, 32U, 33U, 65U, 257U, 1057U}) {
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
  compare_oracle({2.0e-40F, -2.0e-40F});
  std::vector<float> mixed(32, 0.0F);
  mixed[0] = 2688.0F;
  mixed[16] = 1.0e-20F;
  const auto q = luxorion::nvfp4_quantize_cpu(mixed);
  check(q.block_scales[1] == 1, "nonzero tiny block scale clamps to minimum positive E4M3");
  compare_oracle(mixed);
}

void test_all_packed_bytes() {
  constexpr std::array<float, 8> values = {0.0F, 0.5F, 1.0F, 1.5F, 2.0F, 3.0F, 4.0F, 6.0F};
  for (unsigned byte = 0; byte < 256; ++byte) {
    luxorion::NVFP4Result q;
    q.element_count = 16;
    q.packed_data.assign(8, static_cast<std::uint8_t>(byte));
    q.block_scales = {0x38};
    const auto restored = luxorion::nvfp4_dequantize_cpu(q);
    for (unsigned i = 0; i < 16; ++i) {
      const unsigned code = i % 2 == 0 ? byte / 16 : byte % 16;
      const float expected = code >= 8 ? -values[code % 8] : values[code];
      check(restored[i] == expected && std::signbit(restored[i]) == std::signbit(expected),
            "externally constructed packed byte " + std::to_string(byte));
    }
  }
}

void test_invalid_and_numeric_failures() {
  expect_error<std::invalid_argument>([] { luxorion::nvfp4_quantize_cpu({}); }, "empty input");
  for (float bad : {std::numeric_limits<float>::quiet_NaN(),
                    std::numeric_limits<float>::infinity(),
                    -std::numeric_limits<float>::infinity()}) {
    expect_error<std::invalid_argument>([bad] { luxorion::nvfp4_quantize_cpu({1.0F, bad}); },
                                       "nonfinite input");
  }
  const auto valid = luxorion::nvfp4_quantize_cpu(std::vector<float>(17, 1.0F));
  auto check_bad = [](const luxorion::NVFP4Result& q) {
    expect_error<std::invalid_argument>([&q] { luxorion::nvfp4_dequantize_cpu(q); }, "malformed result");
  };
  auto bad = valid; bad.element_count = 0; check_bad(bad);
  bad = valid; bad.block_scales.pop_back(); check_bad(bad);
  bad = valid; bad.block_scales.push_back(0x38); check_bad(bad);
  bad = valid; bad.packed_data.pop_back(); check_bad(bad);
  bad = valid; bad.packed_data.push_back(0); check_bad(bad);
  for (float scale : {0.0F, -0.0F, -1.0F, std::numeric_limits<float>::quiet_NaN(),
                      std::numeric_limits<float>::infinity()}) {
    bad = valid; bad.global_scale = scale; check_bad(bad);
  }
  for (std::uint8_t code : {0, 127, 128, 255}) {
    bad = valid; bad.block_scales[0] = code; check_bad(bad);
  }
  bad = valid; bad.packed_data[8] |= 1; check_bad(bad); // 奇数尾元素后的低四位。
  for (unsigned byte = 9; byte < 16; ++byte) {
    bad = valid; bad.packed_data[byte] = 1; check_bad(bad);
  }
  expect_error<std::underflow_error>([] {
    luxorion::nvfp4_quantize_cpu({std::numeric_limits<float>::denorm_min()});
  }, "global scale underflow");
  bad = valid;
  bad.global_scale = std::numeric_limits<float>::denorm_min();
  bad.block_scales[0] = 1;
  expect_error<std::underflow_error>([&bad] { luxorion::nvfp4_dequantize_cpu(bad); },
                                   "effective scale underflow");
  bad = valid;
  bad.global_scale = std::numeric_limits<float>::max();
  bad.block_scales[0] = 0x7E;
  expect_error<std::overflow_error>([&bad] { luxorion::nvfp4_dequantize_cpu(bad); },
                                  "effective scale overflow");
  bad = valid;
  bad.global_scale = std::numeric_limits<float>::max() / 4.0F;
  bad.block_scales[0] = 0x38;
  bad.packed_data[0] = 0x70;
  expect_error<std::overflow_error>([&bad] { luxorion::nvfp4_dequantize_cpu(bad); },
                                  "restoration overflow");
}
}  // namespace

int main() {
  try {
    test_manual_and_tail();
    test_lengths_distributions_and_tiny_scales();
    test_all_packed_bytes();
    test_invalid_and_numeric_failures();
    std::cout << "NVFP4 CPU: " << checks << " checks passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "NVFP4 test failed: " << error.what() << '\n';
    return 1;
  }
}
