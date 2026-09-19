#pragma once

#include "qd/common.hpp"

#include <vector>

namespace qd {

struct GpuQuantizeResult {
  QuantizedData data;
  float milliseconds = 0.0F;
};

struct GpuDequantizeResult {
  std::vector<float> fp32_values;
  std::vector<std::uint8_t> output_payload;
  float milliseconds = 0.0F;
};

GpuQuantizeResult quantize_gpu(const Matrix &matrix, const Config &config,
                               int iterations);
GpuDequantizeResult dequantize_gpu(const QuantizedData &data,
                                   DataType output_type, int iterations);

} // namespace qd
