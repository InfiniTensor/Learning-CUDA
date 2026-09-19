#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace hadamard {

inline bool power_of_two(std::size_t n) { return n && !(n & (n - 1)); }

// Sylvester ordering. The public transform is unnormalized unless scale is set.
template <class F>
void fwht(F* values, std::size_t rows, std::size_t dim, F scale = F(1)) {
    if (!power_of_two(dim)) throw std::invalid_argument("dim must be a power of two");
    for (std::size_t row = 0; row < rows; ++row) {
        F* x = values + row * dim;
        for (std::size_t stride = 1; stride < dim; stride *= 2) {
            for (std::size_t base = 0; base < dim; base += 2 * stride) {
                for (std::size_t j = 0; j < stride; ++j) {
                    const F a = x[base + j], b = x[base + j + stride];
                    x[base + j] = a + b;
                    x[base + j + stride] = a - b;
                }
            }
        }
        for (std::size_t j = 0; j < dim; ++j) x[j] *= scale;
    }
}

// Independent O(N^2) oracle: no butterfly code is shared with the kernels.
inline std::vector<double> dense_reference(const std::vector<float>& x,
                                           std::size_t dim, double scale = 1) {
    if (!power_of_two(dim) || x.size() % dim)
        throw std::invalid_argument("invalid reference shape");
    std::vector<double> y(x.size(), 0.0);
    for (std::size_t row = 0; row < x.size() / dim; ++row) {
        for (std::size_t out = 0; out < dim; ++out) {
            double sum = 0;
            for (std::size_t in = 0; in < dim; ++in) {
                auto bits = out & in;
                bool odd = false;
                while (bits) { odd = !odd; bits &= bits - 1; }
                sum += (odd ? -1.0 : 1.0) * x[row * dim + in];
            }
            y[row * dim + out] = sum * scale;
        }
    }
    return y;
}

// Round-to-nearest, ties-to-even, independent of the host rounding mode.
inline int nearest_even(float value) {
    const float lower = std::floor(value);
    const float fraction = value - lower;
    int q = static_cast<int>(lower);
    if (fraction > 0.5f || (fraction == 0.5f && q % 2 != 0)) ++q;
    return q;
}

struct Int4Result {
    std::vector<std::uint8_t> packed;
    std::vector<float> scales;
};

// Per-row symmetric INT4, [-7,7]; even element in low nibble. Zero row: scale=1.
// Input must already be rounded to the transform's FP16/BF16 output dtype.
inline Int4Result quantize_int4(const std::vector<float>& values, std::size_t dim) {
    if (!dim || values.size() % dim) throw std::invalid_argument("invalid quantization shape");
    const std::size_t rows = values.size() / dim, bytes = (dim + 1) / 2;
    Int4Result out{std::vector<std::uint8_t>(rows * bytes, 0), std::vector<float>(rows)};
    for (std::size_t r = 0; r < rows; ++r) {
        float magnitude = 0;
        for (std::size_t j = 0; j < dim; ++j) {
            if (!std::isfinite(values[r * dim + j]))
                throw std::invalid_argument("INT4 quantization requires finite values");
            magnitude = std::max(magnitude, std::abs(values[r * dim + j]));
        }
        const float scale = magnitude == 0 ? 1.0f : magnitude / 7.0f;
        out.scales[r] = scale;
        for (std::size_t j = 0; j < dim; ++j) {
            const int q = std::clamp(nearest_even(values[r * dim + j] / scale), -7, 7);
            out.packed[r * bytes + j / 2] |= static_cast<std::uint8_t>((q & 15) << (4 * (j % 2)));
        }
    }
    return out;
}

}  // namespace hadamard
