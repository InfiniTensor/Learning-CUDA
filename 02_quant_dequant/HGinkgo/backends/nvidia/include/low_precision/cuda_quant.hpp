#pragma once

#include "low_precision/quant.hpp"

#include <cstddef>
#include <cstdint>

namespace low_precision {

void cuda_quantize_mxfp8(const float* d_input,
                         std::uint8_t* d_values,
                         std::uint8_t* d_scales,
                         std::size_t rows,
                         std::size_t cols,
                         Rounding rounding = Rounding::NearestEven,
                         std::uint64_t seed = 0);

void cuda_quantize_mxfp8(const Fp16* d_input,
                         std::uint8_t* d_values,
                         std::uint8_t* d_scales,
                         std::size_t rows,
                         std::size_t cols,
                         Rounding rounding = Rounding::NearestEven,
                         std::uint64_t seed = 0);

void cuda_dequantize_mxfp8(const std::uint8_t* d_values,
                           const std::uint8_t* d_scales,
                           float* d_output,
                           std::size_t rows,
                           std::size_t cols);

void cuda_dequantize_mxfp8(const std::uint8_t* d_values,
                           const std::uint8_t* d_scales,
                           Fp16* d_output,
                           std::size_t rows,
                           std::size_t cols);

void cuda_dequantize_mxfp8(const std::uint8_t* d_values,
                           const std::uint8_t* d_scales,
                           Bf16* d_output,
                           std::size_t rows,
                           std::size_t cols);

void cuda_quantize_nvfp4(const float* d_input,
                         std::uint8_t* d_values,
                         std::uint8_t* d_block_scales,
                         std::size_t rows,
                         std::size_t cols,
                         float* global_scale,
                         Rounding rounding = Rounding::NearestEven,
                         std::uint64_t seed = 0);

void cuda_quantize_nvfp4(const Fp16* d_input,
                         std::uint8_t* d_values,
                         std::uint8_t* d_block_scales,
                         std::size_t rows,
                         std::size_t cols,
                         float* global_scale,
                         Rounding rounding = Rounding::NearestEven,
                         std::uint64_t seed = 0);

void cuda_dequantize_nvfp4(const std::uint8_t* d_values,
                           const std::uint8_t* d_block_scales,
                           float global_scale,
                           float* d_output,
                           std::size_t rows,
                           std::size_t cols);

void cuda_dequantize_nvfp4(const std::uint8_t* d_values,
                           const std::uint8_t* d_block_scales,
                           float global_scale,
                           Fp16* d_output,
                           std::size_t rows,
                           std::size_t cols);

void cuda_dequantize_nvfp4(const std::uint8_t* d_values,
                           const std::uint8_t* d_block_scales,
                           float global_scale,
                           Bf16* d_output,
                           std::size_t rows,
                           std::size_t cols);

}  // namespace low_precision
