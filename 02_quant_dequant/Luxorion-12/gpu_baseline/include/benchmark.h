#pragma once

// GPU baseline性能结果。kernel统计使用CUDA event；end_to_end使用Host稳态时钟，
// 包含H2D、量化、反量化和D2H，但不包含显存分配、文件IO与CPU校验。

#include "mxfp8.h"
#include "nvfp4.h"

#include <cstdint>
#include <vector>

namespace luxorion {

struct TimingStats {
  double median_ms = 0.0;
  double p95_ms = 0.0;
};

struct GpuBaselineTimings {
  TimingStats quantize;
  TimingStats dequantize;
  TimingStats device_pipeline;
  TimingStats end_to_end;
  double quantize_effective_gbps = 0.0;
  double dequantize_effective_gbps = 0.0;
  unsigned warmup = 0;
  unsigned repeats = 0;
};

struct MXFP8BenchmarkResult {
  GpuBaselineTimings timings;
  MXFP8Result quantized;
  std::vector<float> restored;
};

struct NVFP4BenchmarkResult {
  GpuBaselineTimings timings;
  NVFP4Result quantized;
  std::vector<float> restored;
};

MXFP8BenchmarkResult benchmark_mxfp8_cuda(
    const std::vector<float>& input, unsigned warmup, unsigned repeats);
MXFP8BenchmarkResult benchmark_mxfp8_cuda_fp16(
    const std::vector<std::uint16_t>& input_bits,
    const std::vector<float>& decoded_input,
    unsigned warmup, unsigned repeats);
NVFP4BenchmarkResult benchmark_nvfp4_cuda(
    const std::vector<float>& input, unsigned warmup, unsigned repeats);
NVFP4BenchmarkResult benchmark_nvfp4_cuda_fp16(
    const std::vector<std::uint16_t>& input_bits,
    const std::vector<float>& decoded_input,
    unsigned warmup, unsigned repeats);

}  // namespace luxorion
