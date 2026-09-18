// A/B benchmark for the optimized FHT and fused per-token symmetric INT4.

#include "cuda_check.cuh"
#include "hadamard.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <random>
#include <string>
#include <vector>

namespace fs = std::filesystem;

struct Config {
  int batch = 4;
  int seq = 1024;
  int heads = 32;
  int head_dim = 128;
  std::string dtype = "fp16";
  bool normalize = true;
  int warmup = 20;
  int iters = 100;
  uint64_t seed = 42;
  std::string csv;
};

static bool parse_bool(const char* text) {
  return std::strcmp(text, "true") == 0 || std::strcmp(text, "1") == 0 ||
         std::strcmp(text, "yes") == 0;
}

static bool parse_args(int argc, char** argv, Config& cfg) {
  for (int i = 1; i < argc; ++i) {
    const std::string key = argv[i];
    auto next = [&]() -> const char* {
      if (++i >= argc) {
        std::fprintf(stderr, "[error] missing value for %s\n", key.c_str());
        std::exit(2);
      }
      return argv[i];
    };
    if (key == "--batch") cfg.batch = std::atoi(next());
    else if (key == "--seq") cfg.seq = std::atoi(next());
    else if (key == "--heads") cfg.heads = std::atoi(next());
    else if (key == "--head_dim") cfg.head_dim = std::atoi(next());
    else if (key == "--dtype") cfg.dtype = next();
    else if (key == "--normalize") cfg.normalize = parse_bool(next());
    else if (key == "--warmup") cfg.warmup = std::atoi(next());
    else if (key == "--iters") cfg.iters = std::atoi(next());
    else if (key == "--seed") cfg.seed = std::strtoull(next(), nullptr, 10);
    else if (key == "--csv") cfg.csv = next();
    else if (key == "--help" || key == "-h") {
      std::printf(
          "Usage: %s [--batch N --seq N --heads N --head_dim N] "
          "[--dtype fp16|bf16 --normalize true|false] "
          "[--warmup N --iters N --seed N --csv PATH]\n",
          argv[0]);
      std::exit(0);
    } else {
      std::fprintf(stderr, "[error] unknown option: %s\n", key.c_str());
      return false;
    }
  }
  const bool supported_dim = cfg.head_dim == 32 || cfg.head_dim == 64 ||
                             cfg.head_dim == 128 || cfg.head_dim == 256 ||
                             cfg.head_dim == 512 || cfg.head_dim == 1024;
  if (cfg.batch < 1 || cfg.seq < 1 || cfg.heads < 1 || !supported_dim ||
      (cfg.dtype != "fp16" && cfg.dtype != "bf16") || cfg.warmup < 0 ||
      cfg.iters < 1) {
    std::fprintf(stderr, "[error] invalid configuration\n");
    return false;
  }
  return true;
}

static std::vector<uint16_t> make_input(size_t elements,
                                        const std::string& dtype,
                                        uint64_t seed) {
  std::mt19937 generator(seed);
  std::normal_distribution<float> normal(0.0f, 1.0f);
  std::vector<uint16_t> output(elements);
  for (size_t i = 0; i < elements; ++i) {
    const float value = normal(generator);
    if (dtype == "fp16") {
      const __half converted = __float2half_rn(value);
      std::memcpy(&output[i], &converted, sizeof(uint16_t));
    } else {
      const __nv_bfloat16 converted = __float2bfloat16_rn(value);
      std::memcpy(&output[i], &converted, sizeof(uint16_t));
    }
  }
  return output;
}

static float native_to_float(uint16_t bits, const std::string& dtype) {
  if (dtype == "fp16") {
    __half value;
    std::memcpy(&value, &bits, sizeof(value));
    return __half2float(value);
  }
  __nv_bfloat16 value;
  std::memcpy(&value, &bits, sizeof(value));
  return __bfloat162float(value);
}

static int quantize_int4_cpu(float value, float inv_scale) {
  int quantized = static_cast<int>(std::nearbyint(value * inv_scale));
  return std::max(-7, std::min(7, quantized));
}

struct QuantReference {
  std::vector<unsigned char> packed;
  std::vector<float> scales;
  double max_abs_error = 0.0;
  double mae = 0.0;
  double rmse = 0.0;
};

static QuantReference quantize_reference(const std::vector<uint16_t>& input,
                                         int total_tokens, int head_dim,
                                         const std::string& dtype) {
  QuantReference result;
  result.packed.resize(static_cast<size_t>(total_tokens) * head_dim / 2);
  result.scales.resize(total_tokens);
  double abs_sum = 0.0;
  double squared_sum = 0.0;
  for (int token = 0; token < total_tokens; ++token) {
    const size_t base = static_cast<size_t>(token) * head_dim;
    float max_abs = 0.0f;
    for (int i = 0; i < head_dim; ++i) {
      max_abs = std::max(max_abs, std::fabs(native_to_float(input[base + i], dtype)));
    }
    const float scale = max_abs > 0.0f ? max_abs / 7.0f : 1.0f;
    const float inv_scale = 1.0f / scale;
    result.scales[token] = scale;
    for (int pair = 0; pair < head_dim / 2; ++pair) {
      const float low_value = native_to_float(input[base + pair * 2], dtype);
      const float high_value = native_to_float(input[base + pair * 2 + 1], dtype);
      const int low = quantize_int4_cpu(low_value, inv_scale);
      const int high = quantize_int4_cpu(high_value, inv_scale);
      result.packed[static_cast<size_t>(token) * head_dim / 2 + pair] =
          static_cast<unsigned char>((low & 0x0f) | ((high & 0x0f) << 4));
      const double low_error = static_cast<double>(low) * scale - low_value;
      const double high_error = static_cast<double>(high) * scale - high_value;
      result.max_abs_error = std::max(
          result.max_abs_error, std::max(std::fabs(low_error), std::fabs(high_error)));
      abs_sum += std::fabs(low_error) + std::fabs(high_error);
      squared_sum += low_error * low_error + high_error * high_error;
    }
  }
  const double count = static_cast<double>(input.size());
  result.mae = abs_sum / count;
  result.rmse = std::sqrt(squared_sum / count);
  return result;
}

struct Timing {
  double avg_ms = 0.0;
  double min_ms = 0.0;
  double max_ms = 0.0;
};

template <typename Launch>
static Timing time_launch(Launch launch, int warmup, int iters,
                          cudaStream_t stream) {
  for (int i = 0; i < warmup; ++i) launch();
  CUDA_CHECK(cudaStreamSynchronize(stream));
  cudaEvent_t start;
  cudaEvent_t stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  double sum = 0.0;
  double minimum = std::numeric_limits<double>::max();
  double maximum = 0.0;
  for (int i = 0; i < iters; ++i) {
    CUDA_CHECK(cudaEventRecord(start, stream));
    launch();
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float elapsed = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed, start, stop));
    sum += elapsed;
    minimum = std::min(minimum, static_cast<double>(elapsed));
    maximum = std::max(maximum, static_cast<double>(elapsed));
  }
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return {sum / iters, minimum, maximum};
}

static void append_csv(const Config& cfg, int total_tokens,
                       const char* implementation, const Timing& timing,
                       double speedup_vs_baseline,
                       double fused_speedup_vs_unfused, size_t output_bytes,
                       double compression_ratio, bool transform_exact,
                       bool fused_exact, bool cpu_quant_exact,
                       const QuantReference& quant) {
  if (cfg.csv.empty()) return;
  const fs::path path(cfg.csv);
  if (!path.parent_path().empty()) fs::create_directories(path.parent_path());
  const bool header = !fs::exists(path) || fs::file_size(path) == 0;
  std::ofstream file(path, std::ios::app);
  if (header) {
    file << "dtype,batch_size,seq_len,num_heads,head_dim,total_tokens,implementation,"
            "warmup_runs,bench_runs,avg_ms,min_ms,max_ms,speedup_vs_baseline,"
            "fused_speedup_vs_unfused,output_bytes,compression_ratio,"
            "optimized_vs_baseline_bit_exact,fused_vs_unfused_bit_exact,"
            "gpu_vs_cpu_quant_bit_exact,quant_max_abs_error,quant_mae,quant_rmse\n";
  }
  file << cfg.dtype << ',' << cfg.batch << ',' << cfg.seq << ',' << cfg.heads
       << ',' << cfg.head_dim << ',' << total_tokens << ',' << implementation
       << ',' << cfg.warmup << ',' << cfg.iters << ',' << timing.avg_ms << ','
       << timing.min_ms << ',' << timing.max_ms << ',' << speedup_vs_baseline
       << ',';
  if (fused_speedup_vs_unfused > 0.0) file << fused_speedup_vs_unfused;
  file << ',' << output_bytes << ',' << compression_ratio << ','
       << (transform_exact ? "true" : "false") << ','
       << (fused_exact ? "true" : "false") << ','
       << (cpu_quant_exact ? "true" : "false") << ',';
  if (std::strstr(implementation, "int4")) {
    file << quant.max_abs_error << ',' << quant.mae << ',' << quant.rmse;
  } else {
    file << ",,";
  }
  file << '\n';
}

int main(int argc, char** argv) {
  Config cfg;
  if (!parse_args(argc, argv, cfg)) return 2;
  const DataType dtype = cfg.dtype == "fp16" ? DataType::FP16 : DataType::BF16;
  const int total_tokens = cfg.batch * cfg.seq * cfg.heads;
  const size_t elements = static_cast<size_t>(total_tokens) * cfg.head_dim;
  const size_t input_bytes = elements * sizeof(uint16_t);
  const size_t packed_bytes = elements / 2;
  const size_t scale_bytes = static_cast<size_t>(total_tokens) * sizeof(float);
  const size_t compressed_bytes = packed_bytes + scale_bytes;

  int device = 0;
  CUDA_CHECK(cudaGetDevice(&device));
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  std::printf("[advanced] GPU=%s dtype=%s head_dim=%d tokens=%d\n", prop.name,
              cfg.dtype.c_str(), cfg.head_dim, total_tokens);

  const std::vector<uint16_t> host_input = make_input(elements, cfg.dtype, cfg.seed);
  void* input = nullptr;
  void* baseline_output = nullptr;
  void* optimized_output = nullptr;
  unsigned char* unfused_packed = nullptr;
  unsigned char* fused_packed = nullptr;
  float* unfused_scales = nullptr;
  float* fused_scales = nullptr;
  CUDA_CHECK(cudaMalloc(&input, input_bytes));
  CUDA_CHECK(cudaMalloc(&baseline_output, input_bytes));
  CUDA_CHECK(cudaMalloc(&optimized_output, input_bytes));
  CUDA_CHECK(cudaMalloc(&unfused_packed, packed_bytes));
  CUDA_CHECK(cudaMalloc(&fused_packed, packed_bytes));
  CUDA_CHECK(cudaMalloc(&unfused_scales, scale_bytes));
  CUDA_CHECK(cudaMalloc(&fused_scales, scale_bytes));
  CUDA_CHECK(cudaMemcpy(input, host_input.data(), input_bytes, cudaMemcpyHostToDevice));
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  auto baseline = [&]() {
    launch_hadamard(input, baseline_output, cfg.batch, cfg.seq, cfg.heads,
                    cfg.head_dim, dtype, cfg.normalize, stream);
  };
  auto optimized = [&]() {
    launch_hadamard_optimized(input, optimized_output, cfg.batch, cfg.seq,
                              cfg.heads, cfg.head_dim, dtype, cfg.normalize,
                              stream);
  };
  auto unfused = [&]() {
    launch_hadamard_optimized(input, optimized_output, cfg.batch, cfg.seq,
                              cfg.heads, cfg.head_dim, dtype, cfg.normalize,
                              stream);
    launch_quantize_int4(optimized_output, unfused_packed, unfused_scales,
                         total_tokens, cfg.head_dim, dtype, stream);
  };
  auto fused = [&]() {
    launch_hadamard_fused_quant_int4(
        input, fused_packed, fused_scales, cfg.batch, cfg.seq, cfg.heads,
        cfg.head_dim, dtype, cfg.normalize, stream);
  };

  baseline();
  optimized();
  unfused();
  fused();
  CUDA_CHECK(cudaStreamSynchronize(stream));

  std::vector<uint16_t> host_baseline(elements);
  std::vector<uint16_t> host_optimized(elements);
  std::vector<unsigned char> host_unfused(packed_bytes);
  std::vector<unsigned char> host_fused(packed_bytes);
  std::vector<float> host_unfused_scales(total_tokens);
  std::vector<float> host_fused_scales(total_tokens);
  CUDA_CHECK(cudaMemcpy(host_baseline.data(), baseline_output, input_bytes,
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(host_optimized.data(), optimized_output, input_bytes,
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(host_unfused.data(), unfused_packed, packed_bytes,
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(host_fused.data(), fused_packed, packed_bytes,
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(host_unfused_scales.data(), unfused_scales, scale_bytes,
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(host_fused_scales.data(), fused_scales, scale_bytes,
                        cudaMemcpyDeviceToHost));

  const bool transform_exact = host_baseline == host_optimized;
  const bool fused_exact = host_unfused == host_fused &&
      std::memcmp(host_unfused_scales.data(), host_fused_scales.data(),
                  scale_bytes) == 0;
  const QuantReference reference = quantize_reference(
      host_baseline, total_tokens, cfg.head_dim, cfg.dtype);
  const bool cpu_quant_exact = reference.packed == host_unfused &&
      std::memcmp(reference.scales.data(), host_unfused_scales.data(),
                  scale_bytes) == 0;
  std::printf("[advanced] checks: optimized_vs_baseline=%s "
              "fused_vs_unfused=%s gpu_vs_cpu_quant=%s\n",
              transform_exact ? "BIT-EXACT" : "FAIL",
              fused_exact ? "BIT-EXACT" : "FAIL",
              cpu_quant_exact ? "BIT-EXACT" : "FAIL");

  const Timing baseline_timing = time_launch(baseline, cfg.warmup, cfg.iters, stream);
  const Timing optimized_timing = time_launch(optimized, cfg.warmup, cfg.iters, stream);
  const Timing unfused_timing = time_launch(unfused, cfg.warmup, cfg.iters, stream);
  const Timing fused_timing = time_launch(fused, cfg.warmup, cfg.iters, stream);
  const double optimized_speedup = baseline_timing.avg_ms / optimized_timing.avg_ms;
  const double fused_speedup = unfused_timing.avg_ms / fused_timing.avg_ms;
  const double compression = static_cast<double>(input_bytes) / compressed_bytes;
  std::printf(
      "[advanced] baseline=%.6f ms optimized=%.6f ms (%.2fx) "
      "unfused_int4=%.6f ms fused_int4=%.6f ms (%.2fx) compression=%.3fx\n",
      baseline_timing.avg_ms, optimized_timing.avg_ms, optimized_speedup,
      unfused_timing.avg_ms, fused_timing.avg_ms, fused_speedup, compression);
  std::printf("[advanced] INT4 error: max=%.6e mae=%.6e rmse=%.6e\n",
              reference.max_abs_error, reference.mae, reference.rmse);

  append_csv(cfg, total_tokens, "baseline", baseline_timing, 1.0, 0.0,
             input_bytes, 1.0, transform_exact, fused_exact, cpu_quant_exact,
             reference);
  append_csv(cfg, total_tokens, "optimized", optimized_timing,
             optimized_speedup, 0.0, input_bytes, 1.0, transform_exact,
             fused_exact, cpu_quant_exact, reference);
  append_csv(cfg, total_tokens, "unfused_int4", unfused_timing,
             baseline_timing.avg_ms / unfused_timing.avg_ms, 0.0,
             compressed_bytes, compression, transform_exact, fused_exact,
             cpu_quant_exact, reference);
  append_csv(cfg, total_tokens, "fused_int4", fused_timing,
             baseline_timing.avg_ms / fused_timing.avg_ms, fused_speedup,
             compressed_bytes, compression, transform_exact, fused_exact,
             cpu_quant_exact, reference);

  CUDA_CHECK(cudaStreamDestroy(stream));
  CUDA_CHECK(cudaFree(input));
  CUDA_CHECK(cudaFree(baseline_output));
  CUDA_CHECK(cudaFree(optimized_output));
  CUDA_CHECK(cudaFree(unfused_packed));
  CUDA_CHECK(cudaFree(fused_packed));
  CUDA_CHECK(cudaFree(unfused_scales));
  CUDA_CHECK(cudaFree(fused_scales));
  return transform_exact && fused_exact && cpu_quant_exact ? 0 : 1;
}
