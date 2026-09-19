#include "exact_int4.hpp"
#include "reference.hpp"
#include <cmath>
#include <cstring>
#include <cstddef>
#include <cstdint>

extern "C" std::size_t scan_quant(const float* values, const float* scales,
                                  std::size_t count, std::uint32_t* first) {
    std::size_t errors = 0;
    for (std::size_t i = 0; i < count; ++i) {
        std::uint32_t vb = 0, sb = 0;
        std::memcpy(&vb, values + i, sizeof(float));
        std::memcpy(&sb, scales + i, sizeof(float));
        const float quotient = values[i] / scales[i];
        const int expected = std::clamp(hadamard::nearest_even(quotient), -7, 7);
        const int actual = hadamard::moore::detail::exact_int4_from_bits(vb, sb);
        if (actual != expected) {
            if (!errors) {
                first[0] = static_cast<std::uint32_t>(i);
                first[1] = vb; first[2] = sb;
                first[3] = static_cast<std::uint32_t>(expected);
                first[4] = static_cast<std::uint32_t>(actual);
            }
            ++errors;
        }
    }
    return errors;
}
