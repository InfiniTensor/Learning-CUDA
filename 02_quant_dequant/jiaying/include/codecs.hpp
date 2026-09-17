#pragma once
#include <cmath>
#include <cstdint>
#include <cstring>

#ifdef __CUDACC__
#define LP_HD __host__ __device__ __forceinline__
#else
#define LP_HD inline
#endif

namespace lp {
LP_HD uint32_t float_bits(float x) {
#ifdef __CUDA_ARCH__
    return __float_as_uint(x);
#else
    uint32_t u;
    std::memcpy(&u, &x, 4);
    return u;
#endif
}
LP_HD float bits_float(uint32_t u) {
#ifdef __CUDA_ARCH__
    return __uint_as_float(u);
#else
    float x;
    std::memcpy(&x, &u, 4);
    return x;
#endif
}

// Counter-based randomness: independent of launch shape, reproducible on CPU/GPU.
LP_HD float random_unit(uint64_t index, uint64_t seed) {
    uint64_t z = index + seed + 0x9e3779b97f4a7c15ULL;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    z ^= z >> 31;
    return static_cast<float>(z >> 40) * 0x1p-24f;
}

// Finite saturation, signed zero, gradual underflow. No FP8/FP4 instruction.
template <int Mantissa, int Bias, int MaxCode, int Sign>
LP_HD uint8_t encode_small(float x, bool stochastic = false, float random = 0.0f) {
    const uint32_t bits = float_bits(x);
    const uint8_t sign = (bits >> 31) ? Sign : 0;
    const float a = bits_float(bits & 0x7fffffffU);
    const int max_exp = MaxCode >> Mantissa;
    const float max_value = ldexpf(
        1.0f + float(MaxCode & ((1 << Mantissa) - 1)) / float(1 << Mantissa), max_exp - Bias);
    if (a >= max_value)
        return sign | MaxCode;
    if (a == 0)
        return sign;
    int lower;
    float fraction;
    if (a < ldexpf(1.0f, 1 - Bias)) {
        const float y = ldexpf(a, Bias - 1 + Mantissa);
        lower = static_cast<int>(y);
        fraction = y - lower;
    } else {
        constexpr int shift = 23 - Mantissa;
        const uint32_t magnitude = bits & 0x7fffffffU;
        lower = static_cast<int>(magnitude >> shift) - ((127 - Bias) << Mantissa);
        fraction = float(magnitude & ((1U << shift) - 1)) * (1.0f / (1U << shift));
    }
    const bool up =
        stochastic ? random < fraction : fraction > 0.5f || (fraction == 0.5f && (lower & 1));
    return sign | static_cast<uint8_t>(lower + up);
}
LP_HD uint8_t encode_e4m3(float x, bool stochastic = false, float random = 0) {
    return encode_small<3, 7, 126, 128>(x, stochastic, random);
}
LP_HD uint8_t encode_e2m1(float x, bool stochastic = false, float random = 0) {
    return encode_small<1, 1, 7, 8>(x, stochastic, random);
}
LP_HD float decode_e4m3(uint8_t code) {
    const unsigned mag = code & 127;
    float value;
    if (mag == 127)
        value = bits_float(0x7fc00000U);
    else if (mag < 8)
        value = float(mag) * 0x1p-9f;
    else
        value = bits_float(((mag / 8 + 120) << 23) | ((mag & 7) << 20));
    return bits_float(float_bits(value) | (uint32_t(code & 128) << 24));
}
LP_HD float decode_e2m1(uint8_t code) {
    const unsigned mag = code & 7;
    const float value =
        mag < 2 ? float(mag) * 0.5f : bits_float(((mag / 2 + 126) << 23) | ((mag & 1) << 22));
    return bits_float(float_bits(value) | (uint32_t(code & 8) << 28));
}
LP_HD float decode_e8m0(uint8_t code) { return ldexpf(1.0f, int(code) - 127); }
LP_HD uint8_t mx_scale(float amax) {
    if (amax == 0)
        return 127; // E8M0 has no zero; use 1 for an all-zero block.
    int exponent;
    frexpf(amax, &exponent);
    int shared = exponent - 1 - 8;
    shared = shared < -127 ? -127 : (shared > 127 ? 127 : shared);
    return static_cast<uint8_t>(shared + 127);
}
LP_HD float nv_global(float amax) {
    if (amax == 0)
        return 1.0f;
    // Protect the global scale against FP32 underflow to zero.
    return fmaxf(amax / 2688.0f, bits_float(1));
}
LP_HD uint8_t nv_scale(float amax, float global) { return encode_e4m3((amax / global) / 6.0f); }

LP_HD float from_half(uint16_t h) {
    const unsigned sign = uint32_t(h & 0x8000) << 16;
    const unsigned exp = (h >> 10) & 31;
    const unsigned mant = h & 1023;
    float x = exp == 0 ? float(mant) * 0x1p-24f
                       : bits_float(exp == 31 ? 0x7f800000U | (mant << 13)
                                              : ((exp + 112) << 23) | (mant << 13));
    return bits_float(float_bits(x) | sign);
}
LP_HD uint16_t to_half(float x) {
    uint32_t u = float_bits(x);
    const uint16_t sign = (u >> 16) & 0x8000;
    const unsigned mag = u & 0x7fffffffU;
    if (mag >= 0x7f800000U)
        return sign | (mag > 0x7f800000U ? 0x7e00 : 0x7c00);
    int e = int((mag >> 23) & 255) - 127;
    if (e > 15)
        return sign | 0x7c00;
    if (e < -25)
        return sign;
    if (e < -14) {
        const unsigned mant = (mag & 0x7fffff) | 0x800000;
        const int shift = -e - 1;
        const unsigned base = mant >> shift;
        const unsigned rem = mant & ((1U << shift) - 1);
        const unsigned mid = 1U << (shift - 1);
        return sign | (base + (rem > mid || (rem == mid && (base & 1))));
    }
    const unsigned rounded = mag + 0xfff + ((mag >> 13) & 1);
    return sign | static_cast<uint16_t>((rounded >> 13) - (112 << 10));
}
LP_HD uint16_t to_bfloat(float x) {
    uint32_t u = float_bits(x);
    if ((u & 0x7fffffffU) > 0x7f800000U)
        return (u >> 16) | 0x40;
    return static_cast<uint16_t>((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}
LP_HD float from_bfloat(uint16_t h) { return bits_float(uint32_t(h) << 16); }
} // namespace lp
