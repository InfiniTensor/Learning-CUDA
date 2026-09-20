// 测试模块用途：验证底层单值格式规则，为分块量化提供可信的基础工具。
// 输入：手工值、全编码枚举、中点邻值和非法值；输出：检查结果或失败原因。
// 主流程关系：它检查encode/decode与pack/unpack，不负责完整张量量化实验。
#include "nvfp4.h"
#include "formats.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace {
// 建议先读手工样例，再读全编码/中点测试：
// E2M1覆盖可表示值、正负零、饱和与非法输入；E4M3覆盖字段和舍入；
// E8M0覆盖全部编码及非法scale；pack/unpack覆盖全部合法字节组合。
// 检查数是断言次数，不是输入数据集的数量。
unsigned checks = 0;

// 不用 assert：Release/NDEBUG 下这些检查也必须执行。
void check(bool condition, const std::string& message) {
  ++checks;
  if (!condition) throw std::runtime_error(message);
}

template <typename Function>
void expect_invalid(Function function, const std::string& message) {
  bool rejected = false;
  try {
    function();
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  check(rejected, message);
}

void test_representable_values() {
  // 独立写出的期望表，不读取编码器内部的数据。
  constexpr std::array<float, 8> values = {
      0.0F, 0.5F, 1.0F, 1.5F, 2.0F, 3.0F, 4.0F, 6.0F};
  for (unsigned sign : {0U, 8U}) {
    for (unsigned magnitude = 0; magnitude < values.size(); ++magnitude) {
      const auto code = static_cast<std::uint8_t>(sign | magnitude);
      const float expected = sign ? -values[magnitude] : values[magnitude];
      const float actual = luxorion::decode_e2m1(code);
      const std::string label = "representable code " + std::to_string(code);
      check(actual == expected, label + ": decoding");
      check(std::signbit(actual) == std::signbit(expected), label + ": sign");
      check(luxorion::encode_e2m1(expected) == code, label + ": encoding");
      check(luxorion::encode_e2m1(actual) == code, label + ": roundtrip");
    }
  }
}

void test_midpoints_and_neighbors() {
  constexpr std::array<float, 7> midpoints = {
      0.25F, 0.75F, 1.25F, 1.75F, 2.5F, 3.5F, 5.0F};
  constexpr std::array<unsigned, 7> ties = {0, 2, 2, 4, 4, 6, 6};
  for (unsigned sign : {0U, 8U}) {
    for (unsigned i = 0; i < midpoints.size(); ++i) {
      const float middle = midpoints[i];
      const float below = std::nextafter(middle, 0.0F);
      const float above = std::nextafter(middle, std::numeric_limits<float>::infinity());
      const std::string label = "midpoint " + std::to_string(middle) +
                                " sign " + std::to_string(sign);
      auto encode = [sign](float magnitude) {
        return luxorion::encode_e2m1(sign ? -magnitude : magnitude);
      };
      check(encode(below) == (sign | i), label + ": lower magnitude neighbor");
      check(encode(middle) == (sign | ties[i]), label + ": ties-to-even");
      check(encode(above) == (sign | (i + 1)), label + ": upper magnitude neighbor");
    }
  }
  check(luxorion::decode_e2m1(luxorion::encode_e2m1(2.7F)) == 3.0F,
        "ordinary rounding: 2.7 -> 3");
}

void test_range_and_invalid_inputs() {
  for (float magnitude : {6.0F, std::nextafter(6.0F, 7.0F), 100.0F,
                          1.0e20F, std::numeric_limits<float>::max()}) {
    check(luxorion::encode_e2m1(magnitude) == 7, "positive saturation");
    check(luxorion::encode_e2m1(-magnitude) == 15, "negative saturation");
  }
  const float tiny = std::numeric_limits<float>::denorm_min();
  check(luxorion::encode_e2m1(tiny) == 0, "tiny positive -> positive zero");
  check(luxorion::encode_e2m1(-tiny) == 8, "tiny negative -> negative zero");
  for (float value : {std::numeric_limits<float>::quiet_NaN(),
                      std::numeric_limits<float>::infinity(),
                      -std::numeric_limits<float>::infinity()}) {
    expect_invalid([value] { luxorion::encode_e2m1(value); },
                   "nonfinite input must be rejected");
  }
  for (unsigned code = 16; code <= 255; ++code) {
    expect_invalid([code] {
      luxorion::decode_e2m1(static_cast<std::uint8_t>(code));
    }, "invalid decoding code " + std::to_string(code));
  }
}
// 独立的测试表示值：按指数区间构造表；编码器不读取这张表。
std::array<float, 127> e4m3_expected_magnitudes() {
  std::array<float, 127> result{};
  float step = 1.0F / 512.0F;
  for (unsigned i = 0; i < 8; ++i) result[i] = i * step;
  float base = 1.0F / 64.0F;
  for (unsigned exponent = 1; exponent <= 15; ++exponent) {
    for (unsigned mantissa = 0; mantissa < 8; ++mantissa) {
      const unsigned code = exponent * 8 + mantissa;
      if (code < result.size()) result[code] = base + mantissa * step;
    }
    base *= 2.0F;
    step *= 2.0F;
  }
  return result;
}

void test_e4m3_values_and_midpoints() {
  const auto values = e4m3_expected_magnitudes();
  for (unsigned sign : {0U, 128U}) {
    for (unsigned magnitude = 0; magnitude < values.size(); ++magnitude) {
      const auto code = static_cast<std::uint8_t>(sign | magnitude);
      const float expected = sign ? -values[magnitude] : values[magnitude];
      const float actual = luxorion::decode_e4m3(code);
      const std::string label = "E4M3 code " + std::to_string(code);
      check(actual == expected, label + ": decoding");
      check(std::signbit(actual) == std::signbit(expected), label + ": sign");
      check(luxorion::encode_e4m3(expected) == code, label + ": encoding");
      check(luxorion::encode_e4m3(actual) == code, label + ": roundtrip");
    }
    for (unsigned i = 0; i + 1 < values.size(); ++i) {
      const float middle = (values[i] + values[i + 1]) / 2.0F;
      const float below = std::nextafter(middle, 0.0F);
      const float above = std::nextafter(middle, std::numeric_limits<float>::infinity());
      const unsigned even_code = (i % 2 == 0) ? i : i + 1;
      const std::string label = "E4M3 midpoint after code " + std::to_string(sign | i);
      auto encode = [sign](float magnitude) {
        return luxorion::encode_e4m3(sign ? -magnitude : magnitude);
      };
      check(encode(below) == (sign | i), label + ": lower magnitude neighbor");
      check(encode(middle) == (sign | even_code), label + ": ties-to-even");
      check(encode(above) == (sign | (i + 1)), label + ": upper magnitude neighbor");
    }
  }
}

void test_e4m3_manual_cases_and_limits() {
  struct Example { float input; std::uint8_t code; float restored; };
  const Example examples[] = {
      {1.0F, 0x38, 1.0F}, {-1.1F, 0xB9, -1.125F},
      {15.5F, 0x58, 16.0F}, {98.0F, 0x6C, 96.0F},
      {260.0F, 0x78, 256.0F}, {448.0F, 0x7E, 448.0F},
      {1.0F / 512.0F, 0x01, 1.0F / 512.0F},
      {1.0F / 64.0F, 0x08, 1.0F / 64.0F}};
  for (const auto& example : examples) {
    const std::string label = "E4M3 manual input " + std::to_string(example.input);
    check(luxorion::encode_e4m3(example.input) == example.code, label + ": code");
    check(luxorion::decode_e4m3(example.code) == example.restored, label + ": value");
  }
  for (float magnitude : {448.0F, std::nextafter(448.0F, 449.0F),
                          449.0F, 1.0e20F, std::numeric_limits<float>::max()}) {
    check(luxorion::encode_e4m3(magnitude) == 0x7E, "E4M3 positive saturation");
    check(luxorion::encode_e4m3(-magnitude) == 0xFE, "E4M3 negative saturation");
  }
  const float tiny = std::numeric_limits<float>::denorm_min();
  check(luxorion::encode_e4m3(tiny) == 0, "E4M3 tiny positive -> zero");
  check(luxorion::encode_e4m3(-tiny) == 0x80, "E4M3 tiny negative -> negative zero");
  for (float value : {std::numeric_limits<float>::quiet_NaN(),
                      std::numeric_limits<float>::infinity(),
                      -std::numeric_limits<float>::infinity()}) {
    expect_invalid([value] { luxorion::encode_e4m3(value); },
                   "E4M3 nonfinite input must be rejected");
  }
  check(std::isnan(luxorion::decode_e4m3(0x7F)), "E4M3 positive NaN encoding");
  check(std::isnan(luxorion::decode_e4m3(0xFF)), "E4M3 negative NaN encoding");
}
void test_e8m0_all_codes() {
  // 用连续翻倍生成期望值，不调用实现里的 frexp/ldexp。
  float expected = 0x1p-127F;
  for (unsigned code = 0; code < 255; ++code) {
    const auto byte = static_cast<std::uint8_t>(code);
    const float actual = luxorion::decode_e8m0(byte);
    const std::string label = "E8M0 code " + std::to_string(code);
    check(actual == expected, label + ": decoding");
    check(std::isfinite(actual) && actual > 0.0F, label + ": finite positive scale");
    check(luxorion::encode_e8m0(expected) == byte, label + ": encoding");
    check(luxorion::encode_e8m0(actual) == byte, label + ": roundtrip");
    // 2 的幂上下紧邻的 FP32 值不是精确的 2 的幂，不能被静默舍入。
    const float below = std::nextafter(expected, 0.0F);
    const float above = std::nextafter(expected, std::numeric_limits<float>::infinity());
    expect_invalid([below] { luxorion::encode_e8m0(below); }, label + ": lower neighbor");
    expect_invalid([above] { luxorion::encode_e8m0(above); }, label + ": upper neighbor");
    if (code < 254) expected *= 2.0F;
  }
  check(std::isnan(luxorion::decode_e8m0(255)), "E8M0 NaN encoding");
}

void test_e8m0_examples_and_invalid_inputs() {
  struct Example { float scale; std::uint8_t code; };
  const Example examples[] = {{0.5F, 126}, {1.0F, 127}, {2.0F, 128}, {8.0F, 130}};
  for (const auto& example : examples) {
    check(luxorion::encode_e8m0(example.scale) == example.code, "E8M0 manual encoding");
    check(luxorion::decode_e8m0(example.code) == example.scale, "E8M0 manual decoding");
  }
  for (float scale : {0.0F, -0.0F, -1.0F, -8.0F, 1.3F, 3.0F,
                      0x1p-128F, std::numeric_limits<float>::denorm_min(),
                      std::numeric_limits<float>::max(),
                      std::numeric_limits<float>::quiet_NaN(),
                      std::numeric_limits<float>::infinity(),
                      -std::numeric_limits<float>::infinity()}) {
    expect_invalid([scale] { luxorion::encode_e8m0(scale); },
                   "E8M0 invalid scale " + std::to_string(scale));
  }
}
void test_e2m1_packing() {
  for (unsigned first = 0; first < 16; ++first) {
    for (unsigned second = 0; second < 16; ++second) {
      const auto a = static_cast<std::uint8_t>(first);
      const auto b = static_cast<std::uint8_t>(second);
      // 用算术建立期望字节，独立核对位序，避免错误 pack/unpack 互相抵消。
      const auto expected = static_cast<std::uint8_t>(first * 16 + second);
      const auto actual = luxorion::pack_e2m1(a, b);
      const std::string label = "E2M1 packed byte " + std::to_string(expected);
      check(actual == expected, label + ": packing order");
      std::uint8_t restored_first = 0;
      std::uint8_t restored_second = 0;
      // 解包外部构造的字节，而不是仅解包 pack 自己生成的结果。
      luxorion::unpack_e2m1(expected, restored_first, restored_second);
      check(restored_first == a, label + ": first code");
      check(restored_second == b, label + ": second code");
      check(luxorion::pack_e2m1(restored_first, restored_second) == expected,
            label + ": roundtrip");
    }
  }
  check(luxorion::pack_e2m1(0x6, 0xB) == 0x6B, "manual packing: 6,B -> 6B");
  for (unsigned code = 16; code <= 255; ++code) {
    const auto invalid = static_cast<std::uint8_t>(code);
    expect_invalid([invalid] { luxorion::pack_e2m1(invalid, 0); },
                   "invalid first packing code " + std::to_string(code));
    expect_invalid([invalid] { luxorion::pack_e2m1(0, invalid); },
                   "invalid second packing code " + std::to_string(code));
  }
  expect_invalid([] { luxorion::pack_e2m1(16, 16); }, "both packing codes invalid");
}
}  // namespace

int main() {
  try {
    test_representable_values();
    test_midpoints_and_neighbors();
    test_range_and_invalid_inputs();
    const unsigned e2m1_checks = checks;
    test_e4m3_values_and_midpoints();
    test_e4m3_manual_cases_and_limits();
    const unsigned e4m3_checks = checks - e2m1_checks;
    test_e8m0_all_codes();
    test_e8m0_examples_and_invalid_inputs();
    const unsigned e8m0_checks = checks - e2m1_checks - e4m3_checks;
    test_e2m1_packing();
    std::cout << "E2M1: " << e2m1_checks << " checks passed\n"
              << "E4M3: " << e4m3_checks << " checks passed\n"
              << "E8M0: " << e8m0_checks << " checks passed\n"
              << "E2M1 pack/unpack: " << checks - e2m1_checks - e4m3_checks - e8m0_checks
              << " checks passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "Format test failed: " << error.what() << '\n';
    return 1;
  }
}
