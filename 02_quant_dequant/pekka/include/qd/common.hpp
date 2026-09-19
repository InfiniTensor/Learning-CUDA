#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace qd {

enum class DataType : std::uint8_t { kFp16 = 1, kFp32 = 2, kBf16 = 3 };
enum class Format : std::uint8_t { kMxfp8 = 1, kNvfp4 = 2 };
enum class ScaleMode : std::uint8_t { kTensor = 1, kBlock = 2 };
enum class Rounding : std::uint8_t { kNearest = 1, kStochastic = 2 };

struct Config {
  Format format = Format::kMxfp8;
  std::uint32_t block_size = 32;
  ScaleMode scale_mode = ScaleMode::kBlock;
  DataType output_type = DataType::kFp16;
  Rounding rounding = Rounding::kNearest;
  std::string target_gpu = "unspecified";
  std::uint64_t seed = 20260827;
};

// Matrix values are converted to float on input. original_type is retained so
// QLOW can describe the original payload and compression ratio accurately.
struct Matrix {
  std::uint64_t rows = 0;
  std::uint64_t cols = 0;
  DataType original_type = DataType::kFp32;
  std::vector<float> values;
};

struct QuantizedData {
  Format format = Format::kMxfp8;
  DataType original_type = DataType::kFp32;
  ScaleMode scale_mode = ScaleMode::kBlock;
  Rounding rounding = Rounding::kNearest;
  std::uint64_t rows = 0;
  std::uint64_t cols = 0;
  std::uint32_t block_size = 0;
  std::vector<std::uint8_t> packed_values;
  std::vector<std::uint8_t> local_scales;
  float global_scale = 1.0F;
};

struct Metrics {
  double max_abs_error = 0.0;
  double mean_abs_error = 0.0;
  double mean_squared_error = 0.0;
  std::uint64_t finite_count = 0;
  std::uint64_t non_finite_input_count = 0;
};

struct Timings {
  float quantize_ms = 0.0F;
  float dequantize_ms = 0.0F;
  int iterations = 1;
};

inline std::size_t element_count(std::uint64_t rows, std::uint64_t cols) {
  return static_cast<std::size_t>(rows * cols);
}

inline std::size_t data_type_size(DataType type) {
  return type == DataType::kFp32 ? 4U : 2U;
}

const char *to_string(DataType value);
const char *to_string(Format value);
const char *to_string(ScaleMode value);
const char *to_string(Rounding value);

} // namespace qd
