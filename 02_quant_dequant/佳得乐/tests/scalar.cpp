#include "project.hpp"
#include <cstdlib>
#include <iostream>
using namespace lp;
void require(bool x, const char *msg) {
    if (!x)
        throw std::runtime_error(msg);
}
// 独立整数权重求和解码，不调用生产 decode 的指数/尾数公式。
long double oracle(unsigned c, int f) {
    int m = f == 0   ? 3
            : f == 1 ? 2
                     : 1,
        e = f == 0   ? 4
            : f == 1 ? 5
                     : 2,
        bias = f == 0   ? 7
               : f == 1 ? 15
                        : 1;
    unsigned a = c & ((1 << (e + m)) - 1);
    if (f == 0 && a == 127)
        return NAN;
    if (f == 1 && (a >> m) == 31)
        return (a & 3) ? NAN : ((c & 128) ? -INFINITY : INFINITY);
    long double v = 0;
    int exp = int(a >> m);
    for (int k = 0; k < m; ++k)
        if (a & (1 << k))
            v += std::pow(2.L, (exp ? exp : 1) - bias - m + k);
    if (exp)
        v += std::pow(2.L, exp - bias);
    return (c & (1 << (e + m))) ? -v : v;
}
int main() {
    try {
        for (int f = 0; f < 3; ++f) {
            for (int c = 0; c < (f == 2 ? 16 : 256); ++c) {
                double a = decode(c, f);
                long double b = oracle(c, f);
                require((std::isnan(a) && std::isnan(b)) || a == b, "independent decode");
                require((std::isnan(a) && std::isnan(decode_fast(c, f))) || a == decode_fast(c, f),
                        "fast decode");
                if (std::isfinite(a))
                    require(encode(a, f) == c, "roundtrip");
            }
            for (int c = 0; c < maxcode(f); ++c) {
                double mid = double((oracle(c, f) + oracle(c + 1, f)) / 2);
                for (double x : {std::nextafter(mid, 0.0), mid, std::nextafter(mid, INFINITY)}) {
                    int best = 0;
                    long double distance = INFINITY;
                    for (int j = 0; j <= maxcode(f); ++j) {
                        auto d = fabsl((long double)x - oracle(j, f));
                        if (d < distance || (d == distance && !(j & 1))) {
                            distance = d;
                            best = j;
                        }
                    }
                    require(encode(x, f) == best, "midpoint RNE");
                    require(encode_fast(x, f) == best, "fast midpoint RNE");
                    require((encode(-x, f) & (f == 2 ? 7 : 127)) == best, "negative midpoint");
                }
            }
            require((encode(INFINITY, f) & (f == 2 ? 7 : 127)) == maxcode(f), "saturation");
            require(encode(NAN, f) == (f == 0 ? 127 : f == 1 ? 126 : 0), "nan policy");
            std::cout << "format=" << f << " exhaustive and midpoint PASS\n";
        }
        for (int s = 0; s < 255; ++s)
            require(scale_value(s, 0, 1) == std::pow(2.0, s - 127), "E8M0");
        require(mxscale(0, 0) == 127 && nvscale(0, 1) == 56, "zero scale");
        require(mxscale(448, 0) == 127 && mxscale(std::nextafter(448., INFINITY), 0) == 128,
                "scale boundary");
        for (unsigned h = 0; h < 65536; ++h) {
            float v = half_value(h);
            if (!std::isnan(v))
                require(half_bits(v) == h, "half exhaustive");
        }
        int up = 0;
        for (int i = 0; i < 100000; ++i)
            up += encode(1.125, 2, true, 42, i) == 3;
        require(std::abs(up - 25000) < 700, "stochastic probability");
        for (unsigned b = 0; b < 65536; ++b) {
            float v = frombits(b << 16);
            if (!std::isnan(v))
                require(bf_bits(v) == b, "BF16 exhaustive");
        }
        require(half_bits(1.00048828125f) == 0x3c00 && half_bits(1.00146484375f) == 0x3c02,
                "FP16 ties even");
        require(bf_bits(1.00390625f) == 0x3f80 && bf_bits(1.01171875f) == 0x3f82, "BF16 ties even");
        require(std::isnan(frombits(output_bits(NAN, 0))) &&
                    std::isinf(half_value(output_bits(INFINITY, 1))),
                "special output conversion");
        std::cout << "scales/FP16/stochastic PASS upper=" << up << "/100000\n";
    } catch (std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
