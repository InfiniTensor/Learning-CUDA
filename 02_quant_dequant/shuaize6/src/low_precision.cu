#include "low_precision.cuh"

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace lpq {
namespace {

constexpr int kThreads = 256;

__device__ inline float finite_magnitude(float value) {
  return isfinite(value) ? fabsf(value) : 0.0F;
}

__device__ inline std::uint8_t quantize_e2m1_scaled(
    float value, std::uint32_t value_bits, float combined_scale) {
  if (!(combined_scale > 0.0F)) {
    return 0U;
  }
  if (value == 0.0F) {
    return (value_bits >> 31) != 0 ? 0x8U : 0U;
  }
  return encode_e2m1(value / combined_scale);
}

__device__ inline std::uint8_t encode_e4m3_round_up_fast(float value) {
  if (!(value > 0.0F)) {
    return 0U;
  }
  const std::uint32_t bits = __float_as_uint(value);
  if (bits >= 0x43E00000U) {
    return 0x7EU;
  }

  const std::uint32_t source_exponent = bits >> 23;
  const std::uint32_t source_mantissa = bits & 0x7FFFFFU;
  const int unbiased_exponent = static_cast<int>(source_exponent) - 127;
  if (unbiased_exponent < -6) {
    return static_cast<std::uint8_t>(__float2int_ru(value * 512.0F));
  }

  int encoded_exponent = unbiased_exponent + 7;
  int mantissa = static_cast<int>(source_mantissa >> 20);
  mantissa += (source_mantissa & 0xFFFFFU) != 0U;
  if (mantissa == 8) {
    mantissa = 0;
    ++encoded_exponent;
  }
  if (encoded_exponent > 15 ||
      (encoded_exponent == 15 && mantissa > 6)) {
    return 0x7EU;
  }
  return static_cast<std::uint8_t>((encoded_exponent << 3) | mantissa);
}

__device__ inline float decode_e4m3_positive_fast(std::uint8_t code) {
  const std::uint32_t exponent = code >> 3;
  const std::uint32_t mantissa = code & 0x7U;
  if (exponent == 0U) {
    return static_cast<float>(mantissa) * 0.001953125F;
  }
  return __uint_as_float(((exponent + 120U) << 23) | (mantissa << 20));
}

__device__ inline float decode_e2m1_fast(std::uint8_t code) {
  const std::uint32_t sign = static_cast<std::uint32_t>(code & 0x8U) << 28;
  const std::uint32_t magnitude = code & 0x7U;
  const std::uint32_t exponent = magnitude >> 1;
  const std::uint32_t mantissa = magnitude & 1U;
  const std::uint32_t normal =
      ((126U + exponent) << 23) | (mantissa << 22);
  const std::uint32_t nonzero_mask =
      0U - static_cast<std::uint32_t>(magnitude != 0U);
  const std::uint32_t one_mask =
      0U - static_cast<std::uint32_t>(magnitude == 1U);
  const std::uint32_t bits =
      ((normal & ~one_mask) | (0x3F000000U & one_mask)) & nonzero_mask;
  return __uint_as_float(sign | bits);
}

__device__ inline float warp_max(float value, unsigned mask,
                                 int width = warpSize) {
  for (int offset = width / 2; offset > 0; offset /= 2) {
    value = fmaxf(value, __shfl_down_sync(mask, value, offset, width));
  }
  return value;
}

template <MxFp8Kind Kind>
__device__ inline std::uint8_t encode_mxfp8_fast(float value) {
  constexpr int kMantissaBits =
      Kind == MxFp8Kind::kE4M3 ? 3 : 2;
  constexpr int kExponentBias =
      Kind == MxFp8Kind::kE4M3 ? 7 : 15;
  constexpr int kMinNormalExponent = 1 - kExponentBias;
  constexpr int kMaxFiniteExponent =
      Kind == MxFp8Kind::kE4M3 ? 15 : 30;
  constexpr int kMaxFiniteMantissa =
      Kind == MxFp8Kind::kE4M3 ? 6 : 3;
  constexpr std::uint32_t kMaxFiniteBits =
      Kind == MxFp8Kind::kE4M3 ? 0x43E00000U : 0x47600000U;
  constexpr float kSubnormalMultiplier =
      Kind == MxFp8Kind::kE4M3 ? 512.0F : 65536.0F;

  const std::uint32_t bits = __float_as_uint(value);
  const std::uint8_t sign =
      static_cast<std::uint8_t>((bits >> 24) & 0x80U);
  const std::uint32_t magnitude_bits = bits & 0x7FFFFFFFU;
  const std::uint32_t source_exponent = magnitude_bits >> 23;
  const std::uint32_t source_mantissa = magnitude_bits & 0x7FFFFFU;
  const std::uint8_t max_code = static_cast<std::uint8_t>(
      (kMaxFiniteExponent << kMantissaBits) | kMaxFiniteMantissa);

  if (source_exponent == 0xFFU) {
    return source_mantissa != 0U
               ? static_cast<std::uint8_t>(sign | 0x7FU)
               : static_cast<std::uint8_t>(sign | max_code);
  }
  if (magnitude_bits == 0U) {
    return sign;
  }
  if (magnitude_bits >= kMaxFiniteBits) {
    return static_cast<std::uint8_t>(sign | max_code);
  }
  if (source_exponent == 0U) {
    return sign;
  }

  int unbiased_exponent = static_cast<int>(source_exponent) - 127;
  if (unbiased_exponent < kMinNormalExponent) {
    const float magnitude = __uint_as_float(magnitude_bits);
    const int mantissa =
        __float2int_rn(magnitude * kSubnormalMultiplier);
    return static_cast<std::uint8_t>(sign | mantissa);
  }

  constexpr int kDiscardedBits = 23 - kMantissaBits;
  constexpr std::uint32_t kRemainderMask =
      (1U << kDiscardedBits) - 1U;
  constexpr std::uint32_t kHalfway = 1U << (kDiscardedBits - 1);
  int mantissa = static_cast<int>(source_mantissa >> kDiscardedBits);
  const std::uint32_t remainder = source_mantissa & kRemainderMask;
  mantissa += remainder > kHalfway ||
              (remainder == kHalfway && (mantissa & 1) != 0);
  if (mantissa == (1 << kMantissaBits)) {
    mantissa = 0;
    ++unbiased_exponent;
  }

  const int encoded_exponent = unbiased_exponent + kExponentBias;
  if (encoded_exponent > kMaxFiniteExponent ||
      (encoded_exponent == kMaxFiniteExponent &&
       mantissa > kMaxFiniteMantissa)) {
    return static_cast<std::uint8_t>(sign | max_code);
  }
  return static_cast<std::uint8_t>(
      sign | (encoded_exponent << kMantissaBits) | mantissa);
}

template <MxFp8Kind Kind>
__global__ void quantize_mxfp8_kernel(const float* input, std::size_t count,
                                      std::uint8_t* values,
                                      std::uint8_t* scales) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const std::size_t blocks_per_cta = blockDim.x / warpSize;
  const std::size_t quant_block =
      static_cast<std::size_t>(blockIdx.x) * blocks_per_cta + warp;
  const std::size_t index = quant_block * kMxBlockSize + lane;
  if (quant_block >= mx_scale_count(count)) {
    return;
  }

  const float value = index < count ? input[index] : 0.0F;
  const float amax = warp_max(finite_magnitude(value), 0xFFFFFFFFU);
  int scale_code = 0;
  if (lane == 0) {
    scale_code = choose_mx_scale(amax, Kind);
    scales[quant_block] = static_cast<std::uint8_t>(scale_code);
  }
  scale_code = __shfl_sync(0xFFFFFFFFU, scale_code, 0);

  if (index < count) {
    const float scaled = value / decode_e8m0(scale_code);
    if constexpr (Kind == MxFp8Kind::kE4M3) {
      values[index] = encode_mxfp8_fast<Kind>(scaled);
    } else {
      values[index] = encode_e5m2(scaled);
    }
  }
}

template <MxFp8Kind Kind>
__device__ inline float decode_mxfp8(std::uint8_t code) {
  if constexpr (Kind == MxFp8Kind::kE4M3) {
    return decode_e4m3(code);
  } else {
    return decode_e5m2(code);
  }
}

template <MxFp8Kind Kind>
__global__ void dequantize_mxfp8_kernel(const std::uint8_t* values,
                                        const std::uint8_t* scales,
                                        std::size_t count, float* output) {
  const std::size_t packed_index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t begin = packed_index * 4;
  if (begin >= count) {
    return;
  }

  const float scale = decode_e8m0(scales[begin / kMxBlockSize]);
  if (begin + 3 < count) {
    const std::uint32_t packed =
        reinterpret_cast<const std::uint32_t*>(values)[packed_index];
    const float4 decoded = make_float4(
        decode_mxfp8<Kind>(static_cast<std::uint8_t>(packed)) * scale,
        decode_mxfp8<Kind>(static_cast<std::uint8_t>(packed >> 8)) * scale,
        decode_mxfp8<Kind>(static_cast<std::uint8_t>(packed >> 16)) * scale,
        decode_mxfp8<Kind>(static_cast<std::uint8_t>(packed >> 24)) * scale);
    reinterpret_cast<float4*>(output)[packed_index] = decoded;
    return;
  }

  for (std::size_t index = begin; index < count; ++index) {
    output[index] = decode_mxfp8<Kind>(values[index]) * scale;
  }
}

__global__ void global_amax_kernel(const float* input, std::size_t count,
                                   float* global_amax) {
  __shared__ float warp_maxima[kThreads / 32];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  float local_max = 0.0F;
  for (std::size_t index =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < count;
       index += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    local_max = fmaxf(local_max, finite_magnitude(input[index]));
  }
  local_max = warp_max(local_max, 0xFFFFFFFFU);
  if (lane == 0) {
    warp_maxima[warp] = local_max;
  }
  __syncthreads();

  if (warp == 0) {
    float block_max = lane < blockDim.x / warpSize ? warp_maxima[lane] : 0.0F;
    block_max = warp_max(block_max, 0xFFFFFFFFU);
    if (lane == 0) {
      atomicMax(reinterpret_cast<unsigned int*>(global_amax),
                __float_as_uint(block_max));
    }
  }
}

__global__ void finalize_global_scale_kernel(float* global_scale) {
  if (threadIdx.x == 0) {
    const float amax = *global_scale;
    *global_scale = amax > 0.0F ? amax / (kE4M3Max * kE2M1Max) : 1.0F;
  }
}

__global__ void quantize_nvfp4_kernel(const float* input, std::size_t count,
                                      std::uint8_t* packed_values,
                                      std::uint8_t* block_scales,
                                      const float* global_scale) {
  const int lane_in_half_warp = threadIdx.x & 15;
  const int half_warp = threadIdx.x >> 4;
  const std::size_t blocks_per_cta = blockDim.x / kNvBlockSize;
  const std::size_t quant_block =
      static_cast<std::size_t>(blockIdx.x) * blocks_per_cta + half_warp;
  if (quant_block >= nv_scale_count(count)) {
    return;
  }
  const std::size_t index = quant_block * kNvBlockSize + lane_in_half_warp;
  const std::uint32_t value_bits =
      index < count
          ? reinterpret_cast<const std::uint32_t*>(input)[index]
          : 0U;
  const float value = __uint_as_float(value_bits);
  const int half_in_warp = (threadIdx.x & 31) >> 4;
  const unsigned mask = 0xFFFFU << (half_in_warp * 16);
  const float block_amax = warp_max(finite_magnitude(value), mask, 16);

  float tensor_scale = lane_in_half_warp == 0 ? *global_scale : 0.0F;
  tensor_scale = __shfl_sync(mask, tensor_scale, 0, 16);
  float combined_scale = 1.0F;
  if (lane_in_half_warp == 0) {
    // A positive unit scale keeps the E2M1 NaN/Inf policy meaningful even
    // when a block has no finite non-zero values.
    const float raw_scale = block_amax > 0.0F
                                ? (block_amax / kE2M1Max) / tensor_scale
                                : 1.0F;
    const std::uint8_t scale_code = encode_e4m3_round_up_fast(raw_scale);
    block_scales[quant_block] = scale_code;
    combined_scale =
        tensor_scale * decode_e4m3_positive_fast(scale_code);
  }
  combined_scale = __shfl_sync(mask, combined_scale, 0, 16);
  const std::uint8_t quantized =
      quantize_e2m1_scaled(value, value_bits, combined_scale);
  const int adjacent =
      __shfl_down_sync(mask, static_cast<int>(quantized), 1, 16);
  if ((lane_in_half_warp & 1) == 0 && index < count) {
    packed_values[index / 2] = static_cast<std::uint8_t>(
        (quantized & 0xFU) | ((adjacent & 0xFU) << 4));
  }
}

__global__ void dequantize_nvfp4_kernel(const std::uint8_t* packed_values,
                                        const std::uint8_t* block_scales,
                                        const float* global_scale,
                                        std::size_t count, float* output) {
  const std::size_t word_index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t packed_begin = word_index * 4;
  const std::size_t output_begin = packed_begin * 2;
  if (output_begin >= count) {
    return;
  }

  const float scale =
      decode_e4m3_positive_fast(block_scales[output_begin / kNvBlockSize]) *
      *global_scale;
  if (output_begin + 7 < count) {
    const std::uint32_t packed =
        reinterpret_cast<const std::uint32_t*>(packed_values)[word_index];
    const float4 first = make_float4(
        decode_e2m1_fast(static_cast<std::uint8_t>(packed) & 0xFU) * scale,
        decode_e2m1_fast(static_cast<std::uint8_t>(packed >> 4) & 0xFU) *
            scale,
        decode_e2m1_fast(static_cast<std::uint8_t>(packed >> 8) & 0xFU) *
            scale,
        decode_e2m1_fast(static_cast<std::uint8_t>(packed >> 12) & 0xFU) *
            scale);
    const float4 second = make_float4(
        decode_e2m1_fast(static_cast<std::uint8_t>(packed >> 16) & 0xFU) *
            scale,
        decode_e2m1_fast(static_cast<std::uint8_t>(packed >> 20) & 0xFU) *
            scale,
        decode_e2m1_fast(static_cast<std::uint8_t>(packed >> 24) & 0xFU) *
            scale,
        decode_e2m1_fast(static_cast<std::uint8_t>(packed >> 28) & 0xFU) *
            scale);
    float4* vector_output = reinterpret_cast<float4*>(output);
    vector_output[output_begin / 4] = first;
    vector_output[output_begin / 4 + 1] = second;
    return;
  }

  for (std::size_t index = output_begin; index < count; ++index) {
    const std::uint8_t packed = packed_values[index / 2];
    const std::uint8_t code =
        (index & 1U) == 0 ? packed & 0xFU : (packed >> 4) & 0xFU;
    output[index] = decode_e2m1_fast(code) * scale;
  }
}

int grid_for_elements(std::size_t count) {
  return static_cast<int>((count + kThreads - 1) / kThreads);
}

}  // namespace

cudaError_t quantize_mxfp8(const float* input, std::size_t count,
                           std::uint8_t* values, std::uint8_t* scales,
                           MxFp8Kind kind, cudaStream_t stream) {
  if (count == 0) {
    return cudaSuccess;
  }
  constexpr int kBlocksPerCta = kThreads / kMxBlockSize;
  const int grid = static_cast<int>(
      (mx_scale_count(count) + kBlocksPerCta - 1) / kBlocksPerCta);
  if (kind == MxFp8Kind::kE4M3) {
    quantize_mxfp8_kernel<MxFp8Kind::kE4M3>
        <<<grid, kThreads, 0, stream>>>(input, count, values, scales);
  } else {
    quantize_mxfp8_kernel<MxFp8Kind::kE5M2>
        <<<grid, kThreads, 0, stream>>>(input, count, values, scales);
  }
  return cudaGetLastError();
}

cudaError_t dequantize_mxfp8(const std::uint8_t* values,
                             const std::uint8_t* scales, std::size_t count,
                             float* output, MxFp8Kind kind,
                             cudaStream_t stream) {
  if (count == 0) {
    return cudaSuccess;
  }
  if (kind == MxFp8Kind::kE4M3) {
    dequantize_mxfp8_kernel<MxFp8Kind::kE4M3>
        <<<grid_for_elements((count + 3) / 4), kThreads, 0, stream>>>(
            values, scales, count, output);
  } else {
    dequantize_mxfp8_kernel<MxFp8Kind::kE5M2>
        <<<grid_for_elements((count + 3) / 4), kThreads, 0, stream>>>(
            values, scales, count, output);
  }
  return cudaGetLastError();
}

cudaError_t quantize_nvfp4(const float* input, std::size_t count,
                           std::uint8_t* packed_values,
                           std::uint8_t* block_scales, float* global_scale,
                           cudaStream_t stream) {
  cudaError_t status = cudaMemsetAsync(global_scale, 0, sizeof(float), stream);
  if (status != cudaSuccess) {
    return status;
  }
  if (count > 0) {
    constexpr int kMaxReductionBlocks = 180;
    const int reduction_grid =
        std::min(grid_for_elements(count), kMaxReductionBlocks);
    global_amax_kernel<<<reduction_grid, kThreads, 0, stream>>>(
        input, count, global_scale);
    status = cudaGetLastError();
    if (status != cudaSuccess) {
      return status;
    }
  }
  finalize_global_scale_kernel<<<1, 1, 0, stream>>>(global_scale);
  status = cudaGetLastError();
  if (status != cudaSuccess || count == 0) {
    return status;
  }

  constexpr int kBlocksPerCta = kThreads / kNvBlockSize;
  const int grid = static_cast<int>(
      (nv_scale_count(count) + kBlocksPerCta - 1) / kBlocksPerCta);
  quantize_nvfp4_kernel<<<grid, kThreads, 0, stream>>>(
      input, count, packed_values, block_scales, global_scale);
  return cudaGetLastError();
}

cudaError_t dequantize_nvfp4(const std::uint8_t* packed_values,
                             const std::uint8_t* block_scales,
                             const float* global_scale, std::size_t count,
                             float* output, cudaStream_t stream) {
  if (count == 0) {
    return cudaSuccess;
  }
  dequantize_nvfp4_kernel<<<grid_for_elements((nv_packed_count(count) + 3) / 4),
                             kThreads, 0, stream>>>(
      packed_values, block_scales, global_scale, count, output);
  return cudaGetLastError();
}

}  // namespace lpq
