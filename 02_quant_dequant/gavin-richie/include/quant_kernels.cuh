#pragma once
#include "quant_types.h"
#include <cstdint>
#include <vector>

// ---------------------------------------------------------------------------
// Quantize / dequantize CUDA kernels (software simulation, no FP8/FP4 HW).
//
// Quantized buffer layouts
//   MXFP8 : one uint8 per element (E4M3 or E5M2 payload).
//           block scale: one uint8 E8M0 per block (block mode),
//           or one fp32 per tensor (tensor mode).
//   NVFP4 : two 4-bit elements per byte; element 2k in the low nibble,
//           element 2k+1 in the high nibble of byte k. One thread performs
//           packed stores of 8 bytes (16 elements = one block) per block.
//           block scale: one uint8 E4M3 payload per block(16),
//           plus one fp32 global scale for the whole tensor.
//
// A "block" of elements is contiguous in the row-major flattened tensor.
// The tail block may be shorter than block_size.
// ---------------------------------------------------------------------------

// Everything the dequantizer needs, mirrored from the weight file header.
struct QuantizedBuffer {
  QuantFormat format;
  ScaleMode scale_mode;
  uint32_t block_size;
  uint64_t num_elems;        // rows*cols
  std::vector<uint8_t> data;        // packed payloads
  std::vector<uint8_t> block_scales;  // E8M0 (mxfp8) or E4M3 (nvfp4), per block
  float tensor_scale = 1.0f;        // fp32 per-tensor scale (tensor mode)
  float global_scale = 1.0f;        // nvfp4 extra global scale (block mode)
};

// Result of one pipeline stage for the perf log.
struct StageTiming {
  float gpu_ms = 0.0f;
};

// Compute tensor-wide amax (used for per-tensor scale and nvfp4 global scale).
float reduce_amax(const float* d_data, uint64_t n);

// Quantize fp32 device buffer -> device packed data + device scales.
// d_data: n floats. Writes into d_packed (n bytes for mxfp8, ceil(n/2) for
// nvfp4), d_scales (one uint8 per block; unused=nullptr in tensor mode),
// and optionally *d_tensor_scale / *d_global_scale.
void launch_quantize(const float* d_data, uint64_t n, const QuantConfig& cfg,
                     uint8_t* d_packed, uint8_t* d_scales,
                     float* d_tensor_scale, float* d_global_scale,
                     StageTiming* timing);

// Dequantize device packed data -> d_out with the requested output dtype
// (fp16: __half, bf16: uint16 bit pattern, fp32: float), always n elements.
void launch_dequantize(const uint8_t* d_packed, uint64_t n,
                       const uint8_t* d_scales, float tensor_scale,
                       float global_scale, const QuantConfig& cfg,
                       void* d_out, StageTiming* timing);

// Error metrics between fp32 reference and dequantized buffer (any dtype).
struct ErrorMetrics {
  float max_abs;
  float mae;
  float mse;
};
ErrorMetrics compute_metrics(const float* d_ref, const void* d_out, uint64_t n,
                             OutputType dtype);
