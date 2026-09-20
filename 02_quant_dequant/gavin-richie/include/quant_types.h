#pragma once
#include <cstdint>
#include <string>

// Low-precision format tags
enum class QuantFormat { MXFP8_E4M3, MXFP8_E5M2, NVFP4 };

// Dequantized output dtype
enum class OutputType { FP16, BF16, FP32 };

// Scaling strategy
enum class ScaleMode { TENSOR, BLOCK };

// Rounding mode
enum class Rounding { NEAREST, STOCHASTIC };

struct QuantConfig {
  QuantFormat format = QuantFormat::MXFP8_E4M3;
  int block_size = 32;  // mxfp8 default 32, nvfp4 default 16
  ScaleMode scale_mode = ScaleMode::BLOCK;
  OutputType output_type = OutputType::FP16;
  Rounding rounding = Rounding::NEAREST;
  std::string target_gpu = "T4";  // reporting only
};

// Input tensor dtype
enum class InputDtype { FP32, FP16 };

struct TensorHeader {
  int64_t num_rows = 0;
  int64_t num_cols = 0;
  InputDtype dtype = InputDtype::FP32;
};

// Binary header of the low-precision weight file (little endian, fixed layout)
struct __attribute__((packed)) WeightFileHeader {
  char magic[4];        // "QWHT"
  uint32_t version = 1;
  uint32_t format;      // QuantFormat as uint32
  uint32_t scale_mode;  // ScaleMode as uint32
  uint32_t block_size;
  uint64_t num_rows;
  uint64_t num_cols;
  uint64_t num_packed_bytes;  // packed data size in bytes
  uint64_t num_scales;        // scale element count (see layout below)
  uint32_t scale_dtype;       // 0: fp32 (per-tensor & nvfp4 global), 1: uint8 (E8M0 block scale for mxfp8)
  uint32_t has_global_scale;  // 1: an extra fp32 global scale follows the scale array (nvfp4)
};

// Parse helpers (file_io), implemented in src/file_io.cpp
QuantConfig parse_config(const std::string& path);
const char* format_name(QuantFormat f);
int bits_per_element(QuantFormat f);
