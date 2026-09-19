#include "low_precision/quant.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <random>
#include <stdexcept>

namespace low_precision {
namespace {

constexpr std::size_t kMxBlock = 32;
constexpr std::size_t kNvBlock = 16;
constexpr float kMxFp8Max = 448.0f;
constexpr float kNvFp4Max = 6.0f;

std::mt19937& random_engine() {
    static thread_local std::mt19937 engine(0x4d584650u);
    return engine;
}

template <typename Code, typename Decode>
Code encode_table(float value,
                  Code max_code,
                  Code sign_bit,
                  Decode decode,
                  Rounding rounding,
                  float max_value) {
    const bool negative = std::signbit(value);
    const float magnitude = std::fabs(value);
    if (magnitude == 0.0f) {
        return negative ? sign_bit : static_cast<Code>(0);
    }

    float clipped = std::min(magnitude, max_value);
    Code lower = 0;
    Code upper = max_code;
    for (Code code = 0; code <= max_code; ++code) {
        const float candidate = decode(code);
        if (candidate <= clipped) {
            lower = code;
        }
        if (candidate >= clipped) {
            upper = code;
            break;
        }
    }

    Code selected = lower;
    if (rounding == Rounding::Stochastic && lower != upper && clipped < max_value) {
        const float lo = decode(lower);
        const float hi = decode(upper);
        const float probability = (clipped - lo) / (hi - lo);
        std::uniform_real_distribution<float> distribution(0.0f, 1.0f);
        selected = distribution(random_engine()) < probability ? upper : lower;
    } else {
        const float lo_distance = clipped - decode(lower);
        const float hi_distance = decode(upper) - clipped;
        if (hi_distance < lo_distance) {
            selected = upper;
        } else if (hi_distance == lo_distance) {
            // Mantissa LSB zero is the ties-to-even choice for both tables.
            selected = (static_cast<unsigned>(lower) & 1u) == 0u ? lower : upper;
        }
    }

    if (negative) {
        return static_cast<Code>(selected | sign_bit);
    }
    return selected;
}

float decode_e4m3_magnitude(std::uint8_t code) {
    const unsigned exponent = (code >> 3u) & 0x0fu;
    const unsigned mantissa = code & 0x07u;
    if (exponent == 0) {
        return std::ldexp(static_cast<float>(mantissa), -9);
    }
    if (exponent == 15 && mantissa == 7) {
        return std::numeric_limits<float>::quiet_NaN();
    }
    return std::ldexp(1.0f + static_cast<float>(mantissa) / 8.0f,
                      static_cast<int>(exponent) - 7);
}

std::size_t block_count(std::size_t cols, std::size_t block_size) {
    return (cols + block_size - 1) / block_size;
}

void check_dimensions(std::size_t actual_size,
                      std::size_t rows,
                      std::size_t cols) {
    if (rows == 0 || cols == 0 || actual_size != rows * cols) {
        throw std::invalid_argument("matrix dimensions do not match input size");
    }
}

std::uint32_t round_shift(std::uint32_t value, unsigned shift) {
    const std::uint32_t truncated = value >> shift;
    const std::uint32_t remainder = value & ((1u << shift) - 1u);
    const std::uint32_t halfway = 1u << (shift - 1u);
    if (remainder > halfway ||
        (remainder == halfway && (truncated & 1u) != 0u)) {
        return truncated + 1u;
    }
    return truncated;
}

}  // namespace

Fp16 fp16_from_float(float value) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const std::uint16_t sign = static_cast<std::uint16_t>((bits >> 16u) & 0x8000u);
    const unsigned exponent = (bits >> 23u) & 0xffu;
    const std::uint32_t mantissa = bits & 0x7fffffu;
    if (exponent == 0xffu) {
        const std::uint16_t payload = mantissa == 0 ? 0 : 0x0200u;
        return {static_cast<std::uint16_t>(sign | 0x7c00u | payload)};
    }

    const int half_exponent = static_cast<int>(exponent) - 127 + 15;
    if (half_exponent >= 31) {
        return {static_cast<std::uint16_t>(sign | 0x7c00u)};
    }
    if (half_exponent <= 0) {
        if (half_exponent < -10) {
            return {sign};
        }
        const std::uint32_t significand = mantissa | 0x800000u;
        const unsigned shift = static_cast<unsigned>(14 - half_exponent);
        return {static_cast<std::uint16_t>(sign | round_shift(significand, shift))};
    }

    std::uint32_t rounded_mantissa = round_shift(mantissa, 13);
    int normalized_exponent = half_exponent;
    if (rounded_mantissa == 0x400u) {
        rounded_mantissa = 0;
        ++normalized_exponent;
    }
    if (normalized_exponent >= 31) {
        return {static_cast<std::uint16_t>(sign | 0x7c00u)};
    }
    return {static_cast<std::uint16_t>(sign |
                                       (static_cast<std::uint16_t>(normalized_exponent) << 10u) |
                                       static_cast<std::uint16_t>(rounded_mantissa))};
}

float fp16_to_float(Fp16 value) {
    const std::uint16_t sign = value.bits & 0x8000u;
    const unsigned exponent = (value.bits >> 10u) & 0x1fu;
    const unsigned mantissa = value.bits & 0x03ffu;
    float result = 0.0f;
    if (exponent == 0) {
        result = std::ldexp(static_cast<float>(mantissa), -24);
    } else if (exponent == 31) {
        result = mantissa == 0 ? std::numeric_limits<float>::infinity()
                               : std::numeric_limits<float>::quiet_NaN();
    } else {
        result = std::ldexp(1.0f + static_cast<float>(mantissa) / 1024.0f,
                            static_cast<int>(exponent) - 15);
    }
    return (sign != 0) ? -result : result;
}

Bf16 bf16_from_float(float value) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const std::uint32_t rounding = 0x7fffu + ((bits >> 16u) & 1u);
    bits += rounding;
    return {static_cast<std::uint16_t>(bits >> 16u)};
}

float bf16_to_float(Bf16 value) {
    const std::uint32_t bits = static_cast<std::uint32_t>(value.bits) << 16u;
    float result = 0.0f;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

float decode_e2m1(std::uint8_t code) {
    const bool negative = (code & 0x08u) != 0;
    const unsigned magnitude_code = code & 0x07u;
    const unsigned exponent = magnitude_code >> 1u;
    const unsigned mantissa = magnitude_code & 0x01u;
    float value = 0.0f;
    if (exponent == 0) {
        value = static_cast<float>(mantissa) * 0.5f;
    } else {
        value = std::ldexp(1.0f + static_cast<float>(mantissa) * 0.5f,
                           static_cast<int>(exponent) - 1);
    }
    return negative ? -value : value;
}

std::uint8_t encode_e2m1(float value, Rounding rounding) {
    return encode_table<std::uint8_t>(value, 7, 8, decode_e2m1, rounding, kNvFp4Max);
}

float decode_e4m3(std::uint8_t code) {
    const float magnitude = decode_e4m3_magnitude(code & 0x7fu);
    return (code & 0x80u) ? -magnitude : magnitude;
}

std::uint8_t encode_e4m3(float value, Rounding rounding) {
    return encode_table<std::uint8_t>(value, 0x7eu, 0x80u, decode_e4m3_magnitude,
                                      rounding, kMxFp8Max);
}

float decode_e8m0(std::uint8_t code) {
    return std::ldexp(1.0f, static_cast<int>(code) - 127);
}

std::uint8_t encode_e8m0(float scale) {
    if (!(scale > 0.0f) || !std::isfinite(scale)) {
        return 127;
    }
    const int exponent = static_cast<int>(std::ceil(std::log2(scale))) + 127;
    return static_cast<std::uint8_t>(std::clamp(exponent, 0, 254));
}

MxFp8Tensor quantize_mxfp8(const std::vector<float>& input,
                           std::size_t rows,
                           std::size_t cols,
                           Rounding rounding) {
    check_dimensions(input.size(), rows, cols);
    const std::size_t blocks_per_row = block_count(cols, kMxBlock);
    MxFp8Tensor output{rows, cols, std::vector<std::uint8_t>(input.size()),
                       std::vector<std::uint8_t>(rows * blocks_per_row)};

    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t block = 0; block < blocks_per_row; ++block) {
            const std::size_t begin = block * kMxBlock;
            const std::size_t end = std::min(begin + kMxBlock, cols);
            float amax = 0.0f;
            for (std::size_t col = begin; col < end; ++col) {
                amax = std::max(amax, std::fabs(input[row * cols + col]));
            }
            const float requested_scale = amax == 0.0f ? 1.0f : amax / kMxFp8Max;
            const std::uint8_t scale_code = encode_e8m0(requested_scale);
            const float scale = decode_e8m0(scale_code);
            output.scales[row * blocks_per_row + block] = scale_code;
            for (std::size_t col = begin; col < end; ++col) {
                output.values[row * cols + col] = encode_e4m3(input[row * cols + col] / scale,
                                                               rounding);
            }
        }
    }
    return output;
}

MxFp8Tensor quantize_mxfp8(const std::vector<Fp16>& input,
                           std::size_t rows,
                           std::size_t cols,
                           Rounding rounding) {
    std::vector<float> converted(input.size());
    for (std::size_t index = 0; index < input.size(); ++index) {
        converted[index] = fp16_to_float(input[index]);
    }
    return quantize_mxfp8(converted, rows, cols, rounding);
}

std::vector<float> dequantize_mxfp8(const MxFp8Tensor& input) {
    check_dimensions(input.values.size(), input.rows, input.cols);
    const std::size_t blocks_per_row = block_count(input.cols, kMxBlock);
    if (input.scales.size() != input.rows * blocks_per_row ||
        input.values.size() != input.rows * input.cols) {
        throw std::invalid_argument("invalid MXFP8 payload size");
    }
    std::vector<float> output(input.values.size());
    for (std::size_t row = 0; row < input.rows; ++row) {
        for (std::size_t col = 0; col < input.cols; ++col) {
            const std::size_t block = col / kMxBlock;
            output[row * input.cols + col] =
                decode_e4m3(input.values[row * input.cols + col]) *
                decode_e8m0(input.scales[row * blocks_per_row + block]);
        }
    }
    return output;
}

std::vector<Fp16> dequantize_mxfp8_fp16(const MxFp8Tensor& input) {
    const std::vector<float> decoded = dequantize_mxfp8(input);
    std::vector<Fp16> output(decoded.size());
    for (std::size_t index = 0; index < decoded.size(); ++index) {
        output[index] = fp16_from_float(decoded[index]);
    }
    return output;
}

std::vector<Bf16> dequantize_mxfp8_bf16(const MxFp8Tensor& input) {
    const std::vector<float> decoded = dequantize_mxfp8(input);
    std::vector<Bf16> output(decoded.size());
    for (std::size_t index = 0; index < decoded.size(); ++index) {
        output[index] = bf16_from_float(decoded[index]);
    }
    return output;
}

NvFp4Tensor quantize_nvfp4(const std::vector<float>& input,
                           std::size_t rows,
                           std::size_t cols,
                           Rounding rounding) {
    check_dimensions(input.size(), rows, cols);
    const std::size_t blocks_per_row = block_count(cols, kNvBlock);
    float global_amax = 0.0f;
    for (float value : input) {
        global_amax = std::max(global_amax, std::fabs(value));
    }
    const float global_scale = global_amax == 0.0f ? 1.0f : global_amax / (kMxFp8Max * kNvFp4Max);
    NvFp4Tensor output{rows, cols,
                       std::vector<std::uint8_t>((input.size() + 1) / 2),
                       std::vector<std::uint8_t>(rows * blocks_per_row), global_scale};

    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t block = 0; block < blocks_per_row; ++block) {
            const std::size_t begin = block * kNvBlock;
            const std::size_t end = std::min(begin + kNvBlock, cols);
            float block_amax = 0.0f;
            for (std::size_t col = begin; col < end; ++col) {
                block_amax = std::max(block_amax, std::fabs(input[row * cols + col]));
            }
            const float requested_block_scale = block_amax == 0.0f
                                                    ? 1.0f
                                                    : std::max((block_amax / kNvFp4Max) /
                                                                   global_scale,
                                                               decode_e4m3(0x01));
            const std::uint8_t block_scale_code = encode_e4m3(requested_block_scale, rounding);
            const float block_scale = decode_e4m3(block_scale_code);
            output.block_scales[row * blocks_per_row + block] = block_scale_code;
            for (std::size_t col = begin; col < end; ++col) {
                const std::size_t index = row * cols + col;
                const std::uint8_t code = encode_e2m1(
                    input[index] / (global_scale * block_scale), rounding);
                if ((index & 1u) == 0) {
                    output.values[index / 2] = code;
                } else {
                    output.values[index / 2] |= static_cast<std::uint8_t>(code << 4u);
                }
            }
        }
    }
    return output;
}

NvFp4Tensor quantize_nvfp4(const std::vector<Fp16>& input,
                           std::size_t rows,
                           std::size_t cols,
                           Rounding rounding) {
    std::vector<float> converted(input.size());
    for (std::size_t index = 0; index < input.size(); ++index) {
        converted[index] = fp16_to_float(input[index]);
    }
    return quantize_nvfp4(converted, rows, cols, rounding);
}

std::vector<float> dequantize_nvfp4(const NvFp4Tensor& input) {
    check_dimensions(input.rows * input.cols, input.rows, input.cols);
    const std::size_t blocks_per_row = block_count(input.cols, kNvBlock);
    if (input.values.size() != (input.rows * input.cols + 1) / 2 ||
        input.block_scales.size() != input.rows * blocks_per_row) {
        throw std::invalid_argument("invalid NVFP4 payload size");
    }
    std::vector<float> output(input.rows * input.cols);
    for (std::size_t row = 0; row < input.rows; ++row) {
        for (std::size_t col = 0; col < input.cols; ++col) {
            const std::size_t index = row * input.cols + col;
            const std::uint8_t packed = input.values[index / 2];
            const std::uint8_t code = (index & 1u) == 0 ? packed & 0x0fu : packed >> 4u;
            const std::size_t block = col / kNvBlock;
            output[index] = decode_e2m1(code) * input.global_scale *
                            decode_e4m3(input.block_scales[row * blocks_per_row + block]);
        }
    }
    return output;
}

std::vector<Fp16> dequantize_nvfp4_fp16(const NvFp4Tensor& input) {
    const std::vector<float> decoded = dequantize_nvfp4(input);
    std::vector<Fp16> output(decoded.size());
    for (std::size_t index = 0; index < decoded.size(); ++index) {
        output[index] = fp16_from_float(decoded[index]);
    }
    return output;
}

std::vector<Bf16> dequantize_nvfp4_bf16(const NvFp4Tensor& input) {
    const std::vector<float> decoded = dequantize_nvfp4(input);
    std::vector<Bf16> output(decoded.size());
    for (std::size_t index = 0; index < decoded.size(); ++index) {
        output[index] = bf16_from_float(decoded[index]);
    }
    return output;
}

}  // namespace low_precision
