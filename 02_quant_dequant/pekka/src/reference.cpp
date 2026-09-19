#include "qd/reference.hpp"

#include "qd/formats.cuh"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace qd {
namespace {

std::size_t group_count(std::size_t count, const Config &config) {
  return config.scale_mode == ScaleMode::kTensor
             ? 1U
             : (count + config.block_size - 1U) / config.block_size;
}

std::pair<std::size_t, std::size_t> group_range(std::size_t group,
                                                std::size_t count,
                                                const Config &config) {
  if (config.scale_mode == ScaleMode::kTensor) return {0, count};
  const std::size_t begin = group * config.block_size;
  return {begin, std::min(count, begin + config.block_size)};
}

float finite_amax(const std::vector<float> &values, std::size_t begin,
                  std::size_t end) {
  float maximum = 0.0F;
  for (std::size_t index = begin; index < end; ++index) {
    if (std::isfinite(values[index])) maximum = std::max(maximum, std::fabs(values[index]));
  }
  return maximum;
}

std::size_t scale_index(std::size_t element, const QuantizedData &data) {
  return data.scale_mode == ScaleMode::kTensor ? 0U : element / data.block_size;
}

} // namespace

QuantizedData quantize_reference(const Matrix &matrix, const Config &config) {
  const std::size_t count = matrix.values.size();
  if (count != element_count(matrix.rows, matrix.cols)) {
    throw std::runtime_error("matrix shape and value count differ");
  }
  QuantizedData output{config.format, matrix.original_type, config.scale_mode,
                       config.rounding, matrix.rows, matrix.cols,
                       config.block_size, {}, {}, 1.0F};
  const std::size_t groups = group_count(count, config);
  output.local_scales.resize(groups);

  if (config.format == Format::kMxfp8) {
    output.packed_values.resize(count);
    for (std::size_t group = 0; group < groups; ++group) {
      const auto [begin, end] = group_range(group, count, config);
      const float amax = finite_amax(matrix.values, begin, end);
      output.local_scales[group] =
          formats::encode_e8m0_ceil(amax == 0.0F ? 1.0F : amax / 448.0F);
    }
    for (std::size_t index = 0; index < count; ++index) {
      const float scale = formats::decode_e8m0(output.local_scales[
          config.scale_mode == ScaleMode::kTensor ? 0U : index / config.block_size]);
      const float normalized = matrix.values[index] / scale;
      output.packed_values[index] =
          config.rounding == Rounding::kStochastic
              ? formats::encode_e4m3_stochastic(normalized, config.seed, index)
              : formats::encode_e4m3(normalized);
    }
    return output;
  }

  // NVFP4 uses one FP32 tensor scale followed by one E4M3 local scale per
  // block.  Keeping global scale unencoded preserves range for large tensors.
  const float tensor_amax = finite_amax(matrix.values, 0, count);
  output.global_scale = tensor_amax == 0.0F ? 1.0F : tensor_amax / (6.0F * 448.0F);
  for (std::size_t group = 0; group < groups; ++group) {
    const auto [begin, end] = group_range(group, count, config);
    const float amax = finite_amax(matrix.values, begin, end);
    const float ideal = amax == 0.0F ? 1.0F : amax / (6.0F * output.global_scale);
    output.local_scales[group] = formats::encode_e4m3(ideal);
  }
  output.packed_values.assign((count + 1U) / 2U, 0);
  for (std::size_t index = 0; index < count; ++index) {
    const float local = formats::decode_e4m3(output.local_scales[
        config.scale_mode == ScaleMode::kTensor ? 0U : index / config.block_size]);
    const float normalized = matrix.values[index] / (output.global_scale * local);
    const std::uint8_t code =
        config.rounding == Rounding::kStochastic
            ? formats::encode_e2m1_stochastic(normalized, config.seed, index)
            : formats::encode_e2m1(normalized);
    if ((index & 1U) == 0) output.packed_values[index / 2U] = code;
    else output.packed_values[index / 2U] |= static_cast<std::uint8_t>(code << 4U);
  }
  return output;
}

std::vector<float> dequantize_reference(const QuantizedData &data) {
  const std::size_t count = element_count(data.rows, data.cols);
  std::vector<float> output(count);
  for (std::size_t index = 0; index < count; ++index) {
    const std::uint8_t scale_bits = data.local_scales[scale_index(index, data)];
    if (data.format == Format::kMxfp8) {
      output[index] = formats::decode_e4m3(data.packed_values[index]) *
                      formats::decode_e8m0(scale_bits);
    } else {
      const std::uint8_t packed = data.packed_values[index / 2U];
      const std::uint8_t code = (index & 1U) == 0 ? packed & 0x0FU : packed >> 4U;
      output[index] = formats::decode_e2m1(code) *
                      formats::decode_e4m3(scale_bits) * data.global_scale;
    }
  }
  return output;
}

} // namespace qd
