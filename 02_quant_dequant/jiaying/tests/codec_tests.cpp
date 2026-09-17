#include "lp.hpp"
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

void require(bool condition, const char *message) {
    if (!condition)
        throw std::runtime_error(message);
}
int main() {
    try {
        const float fp4[] = {0, .5f, 1, 1.5f, 2, 3, 4, 6};
        for (int c = 0; c < 16; ++c) {
            require(lp::decode_e2m1(c) == ((c & 8) ? -fp4[c & 7] : fp4[c]), "FP4 codebook");
            require(lp::encode_e2m1(lp::decode_e2m1(c)) == c, "FP4 roundtrip/sign");
        }
        for (int c = 0; c < 256; ++c) {
            if ((c & 127) == 127)
                require(std::isnan(lp::decode_e4m3(c)), "FP8 NaN");
            else
                require(lp::encode_e4m3(lp::decode_e4m3(c)) == c, "FP8 exhaustive roundtrip");
        }
        for (int c = 0; c < 126; ++c) {
            const float a = lp::decode_e4m3(c), b = lp::decode_e4m3(c + 1), mid = (a + b) / 2;
            require(lp::encode_e4m3(mid) == c + (c & 1), "FP8 ties to even");
            require(lp::encode_e4m3(std::nextafter(mid, 0.f)) == c, "FP8 below midpoint");
            require(lp::encode_e4m3(std::nextafter(mid, 500.f)) == c + 1, "FP8 above midpoint");
        }
        for (int c = 0; c < 7; ++c) {
            const float mid = (fp4[c] + fp4[c + 1]) / 2;
            require(lp::encode_e2m1(mid) == c + (c & 1), "FP4 ties to even");
        }
        for (int h = 0; h < 65536; ++h) {
            float x = lp::from_half(h);
            if (!std::isnan(x))
                require(lp::to_half(x) == h, "FP16 exhaustive roundtrip");
            x = lp::from_bfloat(h);
            if (!std::isnan(x))
                require(lp::to_bfloat(x) == h, "BF16 exhaustive roundtrip");
        }
        require(lp::encode_e4m3(10000) == 126, "FP8 saturation");
        require(lp::encode_e2m1(-10000) == 15, "FP4 saturation");
        require(lp::decode_e4m3(126) == 448, "FP8 max");
        require(lp::mx_scale(1) == 119 && lp::mx_scale(448) == 127, "MX exponent selection");
        require(lp::mx_scale(std::numeric_limits<float>::denorm_min()) == 0, "E8M0 minimum");
        require(lp::decode_e8m0(0) == 0x1p-127f, "E8M0 code zero is not zero");
        require(lp::to_half(65520.f) == 0x7c00, "FP16 overflow rounding");
        require(lp::to_half(0x1p-25f) == 0 && lp::to_half(3 * 0x1p-25f) == 2,
                "FP16 underflow ties");
        double sum = 0;
        constexpr int n = 200000;
        for (int i = 0; i < n; ++i)
            sum += lp::decode_e2m1(lp::encode_e2m1(1.125f, true, lp::random_unit(i, 42)));
        require(std::abs(sum / n - 1.125) < .003, "Stochastic rounding expectation");
        std::cout
            << "PASS: exhaustive codecs, ties, saturation, subnormals, stochastic expectation\n";
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
