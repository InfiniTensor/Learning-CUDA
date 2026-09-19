#include "kernel_operator.h"

using namespace AscendC;

namespace {

constexpr uint64_t kMxBlock = 32;
constexpr uint64_t kNvBlock = 16;
constexpr float kMxFp8Max = 448.0f;
constexpr float kNvFp4Max = 6.0f;

union FloatBits {
    float value;
    uint32_t bits;
};

__aicore__ inline float from_bits(uint32_t bits) {
    FloatBits result;
    result.bits = bits;
    return result.value;
}

__aicore__ inline uint32_t bits_of(float value) {
    FloatBits result;
    result.value = value;
    return result.bits;
}

__aicore__ inline float abs_value(float value) {
    return from_bits(bits_of(value) & 0x7fffffffu);
}

__aicore__ inline bool is_negative(float value) {
    return (bits_of(value) & 0x80000000u) != 0u;
}

__aicore__ inline float decode_e2m1(uint32_t code) {
    const uint32_t magnitude = code & 0x07u;
    float value = 0.0f;
    switch (magnitude) {
        case 1u: value = 0.5f; break;
        case 2u: value = 1.0f; break;
        case 3u: value = 1.5f; break;
        case 4u: value = 2.0f; break;
        case 5u: value = 3.0f; break;
        case 6u: value = 4.0f; break;
        case 7u: value = 6.0f; break;
        default: break;
    }
    return (code & 0x08u) != 0u ? -value : value;
}

__aicore__ inline float decode_e4m3_magnitude(uint32_t code) {
    const uint32_t exponent = (code >> 3u) & 0x0fu;
    const uint32_t mantissa = code & 0x07u;
    if (exponent == 0u) {
        switch (mantissa) {
            case 1u: return 0.001953125f;
            case 2u: return 0.00390625f;
            case 3u: return 0.005859375f;
            case 4u: return 0.0078125f;
            case 5u: return 0.009765625f;
            case 6u: return 0.01171875f;
            case 7u: return 0.013671875f;
            default: return 0.0f;
        }
    }
    if (exponent == 15u && mantissa == 7u) {
        return from_bits(0x7fc00000u);
    }
    float scale = 32.0f;
    switch (exponent) {
        case 1u: scale = 0.001953125f; break;
        case 2u: scale = 0.00390625f; break;
        case 3u: scale = 0.0078125f; break;
        case 4u: scale = 0.015625f; break;
        case 5u: scale = 0.03125f; break;
        case 6u: scale = 0.0625f; break;
        case 7u: scale = 0.125f; break;
        case 8u: scale = 0.25f; break;
        case 9u: scale = 0.5f; break;
        case 10u: scale = 1.0f; break;
        case 11u: scale = 2.0f; break;
        case 12u: scale = 4.0f; break;
        case 13u: scale = 8.0f; break;
        case 14u: scale = 16.0f; break;
        default: break;
    }
    float significand = 1.0f;
    switch (mantissa) {
        case 1u: significand = 1.125f; break;
        case 2u: significand = 1.25f; break;
        case 3u: significand = 1.375f; break;
        case 4u: significand = 1.5f; break;
        case 5u: significand = 1.625f; break;
        case 6u: significand = 1.75f; break;
        case 7u: significand = 1.875f; break;
        default: break;
    }
    return significand * (scale * 8.0f);
}

__aicore__ inline float decode_e4m3(uint32_t code) {
    const float value = decode_e4m3_magnitude(code & 0x7fu);
    return (code & 0x80u) != 0u ? -value : value;
}

__aicore__ inline float decode_e8m0(uint32_t code) {
    return code == 0u ? from_bits(0x00400000u) : from_bits(code << 23u);
}

__aicore__ inline float random_uniform(uint64_t seed) {
    uint64_t value = seed + 0x9e3779b97f4a7c15ull;
    value = (value ^ (value >> 30u)) * 0xbf58476d1ce4e5b9ull;
    value = (value ^ (value >> 27u)) * 0x94d049bb133111ebull;
    value ^= value >> 31u;
    const uint32_t mantissa = value >> 41u;
    return from_bits(0x3f800000u | mantissa) - 1.0f;
}

__aicore__ inline uint32_t encode_e2m1(float value, uint32_t stochastic, uint64_t seed) {
    const bool sign = is_negative(value);
    float magnitude = abs_value(value);
    if (magnitude > kNvFp4Max) {
        magnitude = kNvFp4Max;
    }
    uint32_t lower = 0u;
    uint32_t upper = 7u;
    for (uint32_t code = 0u; code <= 7u; ++code) {
        const float candidate = decode_e2m1(code);
        if (candidate <= magnitude) lower = code;
        if (candidate >= magnitude) {
            upper = code;
            break;
        }
    }
    uint32_t selected = lower;
    const float lower_value = decode_e2m1(lower);
    const float upper_value = decode_e2m1(upper);
    if (stochastic != 0u && lower != upper && magnitude < kNvFp4Max) {
        selected = random_uniform(seed) < (magnitude - lower_value) / (upper_value - lower_value)
                       ? upper
                       : lower;
    } else {
        const float low_distance = magnitude - lower_value;
        const float high_distance = upper_value - magnitude;
        if (high_distance < low_distance ||
            (high_distance == low_distance && (lower & 1u) != 0u)) selected = upper;
    }
    return sign ? selected | 0x08u : selected;
}

__aicore__ inline uint32_t encode_e4m3(float value, uint32_t stochastic, uint64_t seed) {
    const bool sign = is_negative(value);
    float magnitude = abs_value(value);
    if (magnitude > kMxFp8Max) {
        magnitude = kMxFp8Max;
    }
    uint32_t lower = 0u;
    uint32_t upper = 0x7eu;
    for (uint32_t code = 0u; code <= 0x7eu; ++code) {
        const float candidate = decode_e4m3_magnitude(code);
        if (candidate <= magnitude) lower = code;
        if (candidate >= magnitude) {
            upper = code;
            break;
        }
    }
    uint32_t selected = lower;
    const float lower_value = decode_e4m3_magnitude(lower);
    const float upper_value = decode_e4m3_magnitude(upper);
    if (stochastic != 0u && lower != upper && magnitude < kMxFp8Max) {
        selected = random_uniform(seed) < (magnitude - lower_value) / (upper_value - lower_value)
                       ? upper
                       : lower;
    } else {
        const float low_distance = magnitude - lower_value;
        const float high_distance = upper_value - magnitude;
        if (high_distance < low_distance ||
            (high_distance == low_distance && (lower & 1u) != 0u)) selected = upper;
    }
    return sign ? selected | 0x80u : selected;
}

__aicore__ inline uint32_t encode_e8m0(float scale) {
    const uint32_t bits = bits_of(scale);
    const uint32_t exponent = (bits >> 23u) & 0xffu;
    if (!(scale > 0.0f) || exponent == 0xffu) return 127u;
    const uint32_t result = exponent + ((bits & 0x7fffffu) == 0u ? 0u : 1u);
    return result > 254u ? 254u : result;
}

__aicore__ inline uint32_t round_shift(uint32_t value, uint32_t shift) {
    const uint32_t truncated = value >> shift;
    const uint32_t remainder = value & ((1u << shift) - 1u);
    const uint32_t halfway = 1u << (shift - 1u);
    return (remainder > halfway || (remainder == halfway && (truncated & 1u) != 0u))
               ? truncated + 1u
               : truncated;
}

__aicore__ inline float fp16_to_float(uint16_t value) {
    const uint32_t sign = static_cast<uint32_t>(value >> 15u);
    const uint32_t exponent = (static_cast<uint32_t>(value) >> 10u) & 0x1fu;
    const uint32_t mantissa = static_cast<uint32_t>(value) & 0x03ffu;
    const uint32_t sign_bits = sign << 31u;
    if (exponent == 0u) {
        if (mantissa == 0u) return from_bits(sign_bits);
        uint32_t highest = 9u;
        while ((mantissa & (1u << highest)) == 0u) --highest;
        return from_bits(sign_bits | ((highest + 103u) << 23u) |
                         ((mantissa - (1u << highest)) << (23u - highest)));
    }
    if (exponent == 31u) return from_bits(sign_bits | (0xffu << 23u) |
                                         (mantissa == 0u ? 0u : 0x400000u));
    return from_bits(sign_bits | ((exponent + 112u) << 23u) | (mantissa << 13u));
}

__aicore__ inline uint16_t fp16_from_float(float value) {
    const uint32_t bits = bits_of(value);
    const uint16_t sign = static_cast<uint16_t>((bits >> 16u) & 0x8000u);
    const uint32_t exponent = (bits >> 23u) & 0xffu;
    const uint32_t mantissa = bits & 0x7fffffu;
    if (exponent == 0xffu) return static_cast<uint16_t>(sign | 0x7c00u |
                                                        (mantissa == 0u ? 0u : 0x0200u));
    const int32_t half_exponent = static_cast<int32_t>(exponent) - 112;
    if (half_exponent >= 31) return static_cast<uint16_t>(sign | 0x7c00u);
    if (half_exponent <= 0) {
        if (half_exponent < -10) return sign;
        return static_cast<uint16_t>(sign | round_shift(mantissa | 0x800000u,
                                                         static_cast<uint32_t>(14 - half_exponent)));
    }
    uint32_t rounded = round_shift(mantissa, 13u);
    int32_t normalized = half_exponent;
    if (rounded == 0x400u) {
        rounded = 0u;
        ++normalized;
    }
    if (normalized >= 31) return static_cast<uint16_t>(sign | 0x7c00u);
    return static_cast<uint16_t>(sign | (static_cast<uint16_t>(normalized) << 10u) |
                                 static_cast<uint16_t>(rounded));
}

__aicore__ inline uint16_t bf16_from_float(float value) {
    uint32_t bits = bits_of(value);
    bits += 0x7fffu + ((bits >> 16u) & 1u);
    return static_cast<uint16_t>(bits >> 16u);
}

__aicore__ inline float load(GlobalTensor<float>& fp32, GlobalTensor<uint16_t>& fp16,
                             uint64_t index, uint32_t input_type) {
    return input_type == 0u ? fp32.GetValue(index) : fp16_to_float(fp16.GetValue(index));
}

__aicore__ inline void store(float value, GlobalTensor<float>& fp32,
                              GlobalTensor<uint16_t>& fp16_or_bf16, uint64_t index,
                              uint32_t output_type) {
    if (output_type == 0u) fp32.SetValue(index, value);
    else if (output_type == 1u) fp16_or_bf16.SetValue(index, fp16_from_float(value));
    else fp16_or_bf16.SetValue(index, bf16_from_float(value));
}

}  // namespace

extern "C" __global__ __aicore__ void mxfp8_quant(GM_ADDR input, GM_ADDR values, GM_ADDR scales,
                                                    uint64_t rows, uint64_t cols,
                                                    uint32_t input_type, uint32_t stochastic,
                                                    uint64_t seed) {
    GlobalTensor<float> fp32;
    GlobalTensor<uint16_t> fp16;
    GlobalTensor<uint8_t> output;
    GlobalTensor<uint8_t> scale_output;
    fp32.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(input));
    fp16.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(input));
    output.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(values));
    scale_output.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(scales));
    const uint64_t blocks_per_row = (cols + kMxBlock - 1u) / kMxBlock;
    const uint64_t total_blocks = rows * blocks_per_row;
    for (uint64_t block_index = GetBlockIdx(); block_index < total_blocks;
         block_index += GetBlockNum()) {
        const uint64_t row = block_index / blocks_per_row;
        const uint64_t begin = (block_index % blocks_per_row) * kMxBlock;
        uint64_t end = begin + kMxBlock;
        if (end > cols) end = cols;
        float amax = 0.0f;
        for (uint64_t col = begin; col < end; ++col) {
            const float magnitude = abs_value(load(fp32, fp16, row * cols + col, input_type));
            if (magnitude > amax) amax = magnitude;
        }
        const uint32_t scale_code = encode_e8m0(amax == 0.0f ? 1.0f : amax / kMxFp8Max);
        const float scale = decode_e8m0(scale_code);
        scale_output.SetValue(block_index, static_cast<uint8_t>(scale_code));
        for (uint64_t col = begin; col < end; ++col) {
            const uint64_t index = row * cols + col;
            output.SetValue(index, static_cast<uint8_t>(encode_e4m3(
                load(fp32, fp16, index, input_type) / scale, stochastic,
                seed ^ (index * 0x9e3779b97f4a7c15ull))));
        }
    }
}

extern "C" __global__ __aicore__ void mxfp8_dequant(GM_ADDR values, GM_ADDR scales,
                                                      GM_ADDR output, uint64_t rows,
                                                      uint64_t cols, uint32_t output_type) {
    GlobalTensor<uint8_t> input;
    GlobalTensor<uint8_t> scale_input;
    GlobalTensor<float> fp32;
    GlobalTensor<uint16_t> fp16_or_bf16;
    input.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(values));
    scale_input.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(scales));
    fp32.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(output));
    fp16_or_bf16.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(output));
    const uint64_t blocks_per_row = (cols + kMxBlock - 1u) / kMxBlock;
    const uint64_t elements = rows * cols;
    for (uint64_t index = GetBlockIdx(); index < elements; index += GetBlockNum()) {
        const uint64_t row = index / cols;
        const uint64_t block = (index % cols) / kMxBlock;
        store(decode_e4m3(input.GetValue(index)) *
                  decode_e8m0(scale_input.GetValue(row * blocks_per_row + block)),
              fp32, fp16_or_bf16, index, output_type);
    }
}

extern "C" __global__ __aicore__ void nvfp4_global_scale(GM_ADDR input, GM_ADDR global_scale,
                                                            uint64_t elements, uint32_t input_type) {
    GlobalTensor<float> fp32;
    GlobalTensor<uint16_t> fp16;
    GlobalTensor<float> output;
    fp32.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(input));
    fp16.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(input));
    output.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(global_scale));
    float amax = 0.0f;
    for (uint64_t index = 0u; index < elements; ++index) {
        const float magnitude = abs_value(load(fp32, fp16, index, input_type));
        if (magnitude > amax) amax = magnitude;
    }
    output.SetValue(0, amax == 0.0f ? 1.0f : amax / (kMxFp8Max * kNvFp4Max));
}

extern "C" __global__ __aicore__ void nvfp4_block_scales(GM_ADDR input, GM_ADDR global_scale,
                                                           GM_ADDR scales, uint64_t rows,
                                                           uint64_t cols, uint32_t input_type,
                                                           uint32_t stochastic, uint64_t seed) {
    GlobalTensor<float> fp32;
    GlobalTensor<uint16_t> fp16;
    GlobalTensor<float> global_input;
    GlobalTensor<uint8_t> output;
    fp32.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(input));
    fp16.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(input));
    global_input.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(global_scale));
    output.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(scales));
    const uint64_t blocks_per_row = (cols + kNvBlock - 1u) / kNvBlock;
    const uint64_t total_blocks = rows * blocks_per_row;
    const float selected_global_scale = global_input.GetValue(0);
    for (uint64_t block_index = GetBlockIdx(); block_index < total_blocks;
         block_index += GetBlockNum()) {
        const uint64_t row = block_index / blocks_per_row;
        const uint64_t begin = (block_index % blocks_per_row) * kNvBlock;
        uint64_t end = begin + kNvBlock;
        if (end > cols) end = cols;
        float amax = 0.0f;
        for (uint64_t col = begin; col < end; ++col) {
            const float magnitude = abs_value(load(fp32, fp16, row * cols + col, input_type));
            if (magnitude > amax) amax = magnitude;
        }
        float requested = amax == 0.0f ? 1.0f : (amax / kNvFp4Max) / selected_global_scale;
        const float minimum = decode_e4m3_magnitude(1u);
        if (requested < minimum) requested = minimum;
        output.SetValue(block_index, static_cast<uint8_t>(encode_e4m3(
            requested, stochastic, seed ^ (block_index * 0x9e3779b97f4a7c15ull))));
    }
}

extern "C" __global__ __aicore__ void nvfp4_quant(GM_ADDR input, GM_ADDR global_scale,
                                                    GM_ADDR block_scales, GM_ADDR values,
                                                    uint64_t rows, uint64_t cols,
                                                    uint32_t input_type, uint32_t stochastic,
                                                    uint64_t seed) {
    GlobalTensor<float> fp32;
    GlobalTensor<uint16_t> fp16;
    GlobalTensor<float> global_input;
    GlobalTensor<uint8_t> scales;
    GlobalTensor<uint8_t> output;
    fp32.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(input));
    fp16.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(input));
    global_input.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(global_scale));
    scales.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(block_scales));
    output.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(values));
    const uint64_t elements = rows * cols;
    const uint64_t pairs = (elements + 1u) / 2u;
    const uint64_t blocks_per_row = (cols + kNvBlock - 1u) / kNvBlock;
    const float selected_global_scale = global_input.GetValue(0);
    for (uint64_t pair = GetBlockIdx(); pair < pairs; pair += GetBlockNum()) {
        const uint64_t first = pair * 2u;
        const uint64_t first_row = first / cols;
        const uint64_t first_block = (first % cols) / kNvBlock;
        const float first_scale = selected_global_scale *
            decode_e4m3(scales.GetValue(first_row * blocks_per_row + first_block));
        uint32_t packed = encode_e2m1(load(fp32, fp16, first, input_type) / first_scale,
                                       stochastic, seed ^ (first * 0x9e3779b97f4a7c15ull));
        const uint64_t second = first + 1u;
        if (second < elements) {
            const uint64_t second_row = second / cols;
            const uint64_t second_block = (second % cols) / kNvBlock;
            const float second_scale = selected_global_scale *
                decode_e4m3(scales.GetValue(second_row * blocks_per_row + second_block));
            packed |= encode_e2m1(load(fp32, fp16, second, input_type) / second_scale,
                                  stochastic, seed ^ (second * 0x9e3779b97f4a7c15ull)) << 4u;
        }
        output.SetValue(pair, static_cast<uint8_t>(packed));
    }
}

extern "C" __global__ __aicore__ void nvfp4_dequant(GM_ADDR values, GM_ADDR block_scales,
                                                      GM_ADDR output, float global_scale,
                                                      uint64_t rows, uint64_t cols,
                                                      uint32_t output_type) {
    GlobalTensor<uint8_t> input;
    GlobalTensor<uint8_t> scales;
    GlobalTensor<float> fp32;
    GlobalTensor<uint16_t> fp16_or_bf16;
    input.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(values));
    scales.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(block_scales));
    fp32.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(output));
    fp16_or_bf16.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(output));
    const uint64_t elements = rows * cols;
    const uint64_t blocks_per_row = (cols + kNvBlock - 1u) / kNvBlock;
    for (uint64_t index = GetBlockIdx(); index < elements; index += GetBlockNum()) {
        const uint64_t row = index / cols;
        const uint64_t block = (index % cols) / kNvBlock;
        const uint32_t packed = input.GetValue(index / 2u);
        const uint32_t code = (index & 1u) == 0u ? packed & 0x0fu : packed >> 4u;
        store(decode_e2m1(code) * global_scale *
                  decode_e4m3(scales.GetValue(row * blocks_per_row + block)),
              fp32, fp16_or_bf16, index, output_type);
    }
}
