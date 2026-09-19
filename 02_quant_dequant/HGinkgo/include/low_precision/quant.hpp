#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace low_precision {

enum class Format { MxFp8, NvFp4 };
enum class Rounding { NearestEven, Stochastic };

struct Fp16 {
    std::uint16_t bits = 0;

    friend constexpr bool operator==(Fp16 lhs, Fp16 rhs) { return lhs.bits == rhs.bits; }
};

struct Bf16 {
    std::uint16_t bits = 0;

    friend constexpr bool operator==(Bf16 lhs, Bf16 rhs) { return lhs.bits == rhs.bits; }
};

struct MxFp8Tensor {
    std::size_t rows = 0;
    std::size_t cols = 0;
    std::vector<std::uint8_t> values;
    std::vector<std::uint8_t> scales;
};

struct NvFp4Tensor {
    std::size_t rows = 0;
    std::size_t cols = 0;
    std::vector<std::uint8_t> values;
    std::vector<std::uint8_t> block_scales;
    float global_scale = 1.0f;
};

Fp16 fp16_from_float(float value);
float fp16_to_float(Fp16 value);
Bf16 bf16_from_float(float value);
float bf16_to_float(Bf16 value);

float decode_e2m1(std::uint8_t code);
std::uint8_t encode_e2m1(float value, Rounding rounding = Rounding::NearestEven);

float decode_e4m3(std::uint8_t code);
std::uint8_t encode_e4m3(float value, Rounding rounding = Rounding::NearestEven);

float decode_e8m0(std::uint8_t code);
std::uint8_t encode_e8m0(float scale);

MxFp8Tensor quantize_mxfp8(const std::vector<float>& input,
                           std::size_t rows,
                           std::size_t cols,
                           Rounding rounding = Rounding::NearestEven);
MxFp8Tensor quantize_mxfp8(const std::vector<Fp16>& input,
                           std::size_t rows,
                           std::size_t cols,
                           Rounding rounding = Rounding::NearestEven);
std::vector<float> dequantize_mxfp8(const MxFp8Tensor& input);
std::vector<Fp16> dequantize_mxfp8_fp16(const MxFp8Tensor& input);
std::vector<Bf16> dequantize_mxfp8_bf16(const MxFp8Tensor& input);

NvFp4Tensor quantize_nvfp4(const std::vector<float>& input,
                           std::size_t rows,
                           std::size_t cols,
                           Rounding rounding = Rounding::NearestEven);
NvFp4Tensor quantize_nvfp4(const std::vector<Fp16>& input,
                           std::size_t rows,
                           std::size_t cols,
                           Rounding rounding = Rounding::NearestEven);
std::vector<float> dequantize_nvfp4(const NvFp4Tensor& input);
std::vector<Fp16> dequantize_nvfp4_fp16(const NvFp4Tensor& input);
std::vector<Bf16> dequantize_nvfp4_bf16(const NvFp4Tensor& input);

}  // namespace low_precision
