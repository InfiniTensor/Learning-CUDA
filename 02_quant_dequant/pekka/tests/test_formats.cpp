#include "qd/formats.cuh"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>

namespace {
void require(bool condition, const char *message) {
  if (!condition) throw std::runtime_error(message);
}
} // namespace

int main() {
  using namespace qd::formats;
  require(decode_e4m3(0x00) == 0.0F, "E4M3 zero");
  require(decode_e4m3(0x01) == std::ldexp(1.0F, -9), "E4M3 subnormal");
  require(decode_e4m3(0x38) == 1.0F, "E4M3 one");
  require(decode_e4m3(0x7E) == 448.0F, "E4M3 maximum");
  require(decode_e4m3(0xB8) == -1.0F, "E4M3 negative one");
  require(std::isnan(decode_e4m3(0x7F)), "E4M3 NaN");
  require(encode_e4m3(1.0F) == 0x38, "encode E4M3 one");
  require(encode_e4m3(448.0F) == 0x7E, "encode E4M3 maximum");
  require(encode_e4m3(INFINITY) == 0x7E, "encode positive infinity");
  require(encode_e4m3(-INFINITY) == 0xFE, "encode negative infinity");
  require(encode_e4m3(NAN) == 0, "encode NaN");

  // Every finite bit pattern except duplicate signed zero is canonical under
  // decode followed by encode.
  for (int code = 0; code < 256; ++code) {
    if ((code & 0x7F) == 0x7F) continue;
    require(encode_e4m3(decode_e4m3(static_cast<std::uint8_t>(code))) == code,
            "E4M3 exhaustive round trip");
  }
  const float fp4_values[] = {0.0F, 0.5F, 1.0F, 1.5F, 2.0F, 3.0F, 4.0F, 6.0F};
  for (int code = 0; code < 8; ++code) {
    require(decode_e2m1(static_cast<std::uint8_t>(code)) == fp4_values[code],
            "E2M1 decode");
    require(encode_e2m1(fp4_values[code]) == code, "E2M1 encode positive");
    require(encode_e2m1(-fp4_values[code]) == (code == 0 ? 0x08 : code | 0x08),
            "E2M1 encode negative");
  }
  require(decode_e8m0(127) == 1.0F, "E8M0 one");
  require(encode_e8m0_ceil(1.0F) == 127, "E8M0 exact power");
  require(encode_e8m0_ceil(1.01F) == 128, "E8M0 ceiling");
  require(encode_e8m0_ceil(0.5F) == 126, "E8M0 half");
  const auto stochastic_a = encode_e4m3_stochastic(0.37F, 20260827, 11);
  const auto stochastic_b = encode_e4m3_stochastic(0.37F, 20260827, 11);
  require(stochastic_a == stochastic_b, "stochastic E4M3 reproducibility");
  require(encode_e4m3_stochastic(1.0F, 20260827, 11) == 0x38,
          "stochastic exact E4M3 value");
  require(encode_e2m1_stochastic(1.0F, 20260827, 11) == 2,
          "stochastic exact E2M1 value");
  std::cout << "format tests passed\n";
}
