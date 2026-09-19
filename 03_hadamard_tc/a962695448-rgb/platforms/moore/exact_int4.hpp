#pragma once

#include <cstdint>

#ifndef HADAMARD_QUANT_HD
#define HADAMARD_QUANT_HD
#define HADAMARD_QUANT_HD_LOCAL
#endif

namespace hadamard::moore::detail {

// Compute clamp(RNE_integer(RNE_float32(value / scale)), -7, 7).
// The inputs are IEEE-754 binary32 bit patterns, finite value and finite scale>0.
// No floating arithmetic or division is used in this function.
HADAMARD_QUANT_HD inline int exact_int4_from_bits(std::uint32_t value_bits,
                                                std::uint32_t scale_bits) {
    const std::uint32_t magnitude_bits = value_bits & 0x7fffffffu;
    if (magnitude_bits == 0) return 0;
    const int value_exponent = static_cast<int>((magnitude_bits >> 23) & 255u);
    const int scale_exponent = static_cast<int>((scale_bits >> 23) & 255u);
    const std::uint32_t value_significand = (magnitude_bits & 0x7fffffu)
        | (value_exponent ? 0x800000u : 0u);
    const std::uint32_t scale_significand = (scale_bits & 0x7fffffu)
        | (scale_exponent ? 0x800000u : 0u);
    const int shift = (value_exponent ? value_exponent - 150 : -149)
        - (scale_exponent ? scale_exponent - 150 : -149) + 25;
    int quantized = 0;
    if (shift >= 30) {
        quantized = 7;
    } else if (shift >= 0) {
        // At most 24+29 bits, so this shift cannot overflow uint64_t.
        const std::uint64_t numerator = static_cast<std::uint64_t>(value_significand) << shift;
        int low = 0, high = 7;
        // Seven increasing transition boundaries require exactly three decisions.
        for (int iteration = 0; iteration < 3; ++iteration) {
            const int middle = (low + high) / 2;
            const int half_ulp_units = middle == 0 ? 1 : (middle == 1 ? 2 : (middle <= 3 ? 4 : 8));
            // A half-integer has an even binary32 significand. Include its rounding
            // interval on the side selected by the subsequent ties-to-even integer step.
            const std::uint32_t boundary = static_cast<std::uint32_t>(
                ((2 * middle + 1) << 24) + ((middle & 1) ? -half_ulp_units : half_ulp_units));
            const std::uint64_t denominator = static_cast<std::uint64_t>(scale_significand) * boundary;
            const bool above = (middle & 1) ? numerator >= denominator : numerator > denominator;
            low = above ? middle + 1 : low;
            high = above ? high : middle;
        }
        quantized = low;
    }
    return (value_bits >> 31) ? -quantized : quantized;
}

}  // namespace hadamard::moore::detail

#ifdef HADAMARD_QUANT_HD_LOCAL
#undef HADAMARD_QUANT_HD
#undef HADAMARD_QUANT_HD_LOCAL
#endif
