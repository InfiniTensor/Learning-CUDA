#include "reference.hpp"
#include <iostream>
#include <random>

int main() {
    using namespace hadamard;
    std::mt19937 rng(2026);
    std::uniform_real_distribution<float> dist(-1, 1);
    std::size_t cases = 0;
    for (std::size_t n : {1, 2, 4, 8, 16, 32, 64, 128, 256}) {
        std::vector<float> x(3 * n);
        for (float& v : x) v = dist(rng);
        auto oracle = dense_reference(x, n);
        std::vector<double> actual(x.begin(), x.end());
        fwht(actual.data(), 3, n);
        for (std::size_t i = 0; i < actual.size(); ++i)
            if (std::abs(actual[i] - oracle[i]) > 1e-10) return 1;
        fwht(actual.data(), 3, n, 1.0 / n);
        for (std::size_t i = 0; i < x.size(); ++i)
            if (std::abs(actual[i] - x[i]) > 1e-10) return 2;
        ++cases;
    }
    if (nearest_even(0.5f) != 0 || nearest_even(1.5f) != 2 ||
        nearest_even(-0.5f) != 0 || nearest_even(-1.5f) != -2) return 3;
    auto quant = quantize_int4({-7, 7, 0, 1}, 4);
    if (quant.scales != std::vector<float>{1} ||
        quant.packed != std::vector<std::uint8_t>{0x79, 0x10}) return 4;
    auto zero = quantize_int4({0}, 1);
    if (zero.scales[0] != 1 || zero.packed[0] != 0) return 5;
    bool rejected = false;
    try { float x[3]{}; fwht(x, 1, 3); } catch (const std::invalid_argument&) { rejected = true; }
    if (!rejected) return 6;
    std::cout << "PASS: " << cases << " matrix-oracle/involution cases, rounding, packing, zero, invalid shape\n";
}
