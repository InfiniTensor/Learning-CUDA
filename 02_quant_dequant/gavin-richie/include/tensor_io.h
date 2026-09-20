#pragma once
#include "quant_types.h"
#include "quant_kernels.cuh"

#include <string>
#include <vector>

// Tensor file format (little endian):
//   [header] int64 num_rows; int64 num_cols; char dtype[4] ("fp32"/"fp16")
//   [data]   values row-major
struct TensorData {
  TensorHeader header;
  std::vector<float> values;  // always materialized as fp32 on the host
};

TensorData read_tensor(const std::string& path);

// Writes a dequantized tensor: dtype string is the output_type name
// ("fp16"/"bf16"/"fp32") and raw_row_major holds n elements already stored
// in that dtype.
void write_dequant_tensor(const std::string& path, uint64_t rows,
                          uint64_t cols, OutputType type,
                          const void* raw_row_major);

const char* output_type_name(OutputType t);
size_t output_type_size(OutputType t);

// Low-precision weight file (see WeightFileHeader in quant_types.h).
void save_weights(const std::string& path, const QuantizedBuffer& q,
                  uint64_t rows, uint64_t cols);
QuantizedBuffer load_weights(const std::string& path, uint64_t* rows,
                             uint64_t* cols);
