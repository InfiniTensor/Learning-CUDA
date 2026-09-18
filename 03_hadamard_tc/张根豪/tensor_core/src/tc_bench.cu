// tc_bench.cu — Tensor Core Hadamard 与非 Tensor Core 实现的对照基准
//
// 对同一份输入、同一个 stream、同一套 CUDA Event 计时边界，依次评测：
//
//   变换路径：
//     baseline      shared-memory 教学版（src/hadamard.cu）
//     optimized     9.1 half2 + warp shuffle 版（src/hadamard.cu）
//     warp          warp-per-token 全寄存器版（src/hadamard_warp.cu）
//     tc_fast       Tensor Core，2 次 mma
//     tc_split      Tensor Core，3 次 mma（中间结果 hi/lo 拆分）
//
//   融合量化路径（per-token symmetric INT4）：
//     unfused_int4      warp FHT -> 低精度中间张量 -> 独立 quantize kernel
//     fused_opt_int4    已验收的 9.2 融合 kernel（block-per-token + shared 归约）
//     fused_warp_int4   warp FHT + INT4 单 kernel（纯 warp 归约）
//     fused_tc_int4     Tensor Core FHT + INT4 单 kernel（fast / split）
//
// 正确性口径（三条互补，不能相互替代）：
//   1. 全张量与 --reference_bin 低精度参考的绝对误差 —— PDF 验收；
//      未提供外部参考时对 optimized 做回归，不宣称完成参考库验收；
//      FP64 CPU 参考仅为额外数值分析，不改变 PDF 阈值；
//   2. 与 optimized FHT 输出的逐位一致率 —— 说明是否严格等价于已验收路径；
//   3. fused 与 unfused 的 packed bytes / scales 是否逐位一致 —— 对应题目
//      “融合量化结果需与先变换后量化一致”。

#include "cuda_check.cuh"
#include "hadamard_tc.cuh"

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
  int ref_tokens = 4096;  // FP64 参考只算前若干 token，控制 CPU 时间
  uint64_t seed = 42;
  std::string csv;
  std::string input_bin;
  std::string reference_bin;
  std::string dump_dir;
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
    else if (key == "--ref_tokens") cfg.ref_tokens = std::atoi(next());
    else if (key == "--seed") cfg.seed = std::strtoull(next(), nullptr, 10);
    else if (key == "--csv") cfg.csv = next();
    else if (key == "--input_bin") cfg.input_bin = next();
    else if (key == "--reference_bin") cfg.reference_bin = next();
    else if (key == "--dump_dir") cfg.dump_dir = next();
    else if (key == "--help" || key == "-h") {
      std::printf(
          "Usage: %s [--batch N --seq N --heads N --head_dim 64|128|256|512|1024]\n"
          "          [--dtype fp16|bf16 --normalize true|false]\n"
          "          [--warmup N --iters N --ref_tokens N --seed N --csv PATH]\n"
          "          [--input_bin PATH --reference_bin PATH --dump_dir PATH]\n",
          argv[0]);
      std::exit(0);
    } else {
      std::fprintf(stderr, "[error] unknown option: %s\n", key.c_str());
      return false;
    }
  }
  const bool supported_dim = cfg.head_dim == 64 || cfg.head_dim == 128 ||
                             cfg.head_dim == 256 || cfg.head_dim == 512 ||
                             cfg.head_dim == 1024;
  if (cfg.batch < 1 || cfg.seq < 1 || cfg.heads < 1 || !supported_dim ||
      (cfg.dtype != "fp16" && cfg.dtype != "bf16") || cfg.warmup < 0 ||
      cfg.iters < 1 || cfg.ref_tokens < 1 ||
      (!cfg.reference_bin.empty() && cfg.input_bin.empty()) ||
      (cfg.batch > 0 && cfg.seq > 0 && cfg.heads > 0 &&
       (static_cast<long long>(cfg.batch) * cfg.seq >
        std::numeric_limits<int>::max() / cfg.heads))) {
    std::fprintf(stderr, "[error] invalid configuration\n");
    return false;
  }
  return true;
}

static std::vector<uint16_t> load_native(const std::string& path, size_t elements) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  const size_t bytes = elements * sizeof(uint16_t);
  if (!file || file.tellg() != static_cast<std::streamoff>(bytes)) {
    std::fprintf(stderr, "[error] %s: expected exactly %zu bytes\n", path.c_str(), bytes);
    std::exit(2);
  }
  file.seekg(0);
  std::vector<uint16_t> data(elements);
  if (!file.read(reinterpret_cast<char*>(data.data()), bytes)) std::exit(2);
  return data;
}

// ---------------------------------------------------------------------------
// host 侧 dtype helper
// ---------------------------------------------------------------------------

static uint16_t float_to_native(float value, const std::string& dtype) {
  uint16_t bits = 0;
  if (dtype == "fp16") {
    const __half converted = __float2half_rn(value);
    std::memcpy(&bits, &converted, sizeof(bits));
  } else {
    const __nv_bfloat16 converted = __float2bfloat16_rn(value);
    std::memcpy(&bits, &converted, sizeof(bits));
  }
  return bits;
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

// 输出 dtype 在给定量级处的 ULP（相邻两个可表示数之间的距离）。
//
// Legacy diagnostic uses the token peak ULP, NOT each element's own ULP.
// It cannot prove correct rounding, bit equivalence, or PDF compliance.
// FP64-reference rounding error must not be confused with library-output error.
static double dtype_ulp(double reference, const std::string& dtype) {
  const float magnitude = std::fabs(static_cast<float>(reference));
  double spacing = 0.0;
  if (dtype == "fp16") {
    const __half value = __float2half_rn(magnitude);
    uint16_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    if (bits >= 0x7bffu) return std::numeric_limits<double>::infinity();
    const uint16_t next_bits = static_cast<uint16_t>(bits + 1);
    __half next;
    std::memcpy(&next, &next_bits, sizeof(next));
    spacing = static_cast<double>(__half2float(next)) -
              static_cast<double>(__half2float(value));
  } else {
    const __nv_bfloat16 value = __float2bfloat16_rn(magnitude);
    uint16_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    if (bits >= 0x7f7fu) return std::numeric_limits<double>::infinity();
    const uint16_t next_bits = static_cast<uint16_t>(bits + 1);
    __nv_bfloat16 next;
    std::memcpy(&next, &next_bits, sizeof(next));
    spacing = static_cast<double>(__bfloat162float(next)) -
              static_cast<double>(__bfloat162float(value));
  }
  // 零/次正规区间用最小正次正规数兜底，避免除零。
  return spacing > 0.0 ? spacing : (dtype == "fp16" ? 5.96e-8 : 9.18e-41);
}

static std::vector<uint16_t> make_input(size_t elements, const std::string& dtype,
                                        uint64_t seed) {
  std::mt19937 generator(seed);
  std::normal_distribution<float> normal(0.0f, 1.0f);
  std::vector<uint16_t> output(elements);
  for (size_t i = 0; i < elements; ++i) {
    output[i] = float_to_native(normal(generator), dtype);
  }
  return output;
}

// ---------------------------------------------------------------------------
// FP64 CPU 参考：与 GPU 使用同样的 stage 顺序，但全程 double
// ---------------------------------------------------------------------------

static std::vector<double> reference_fht(const std::vector<uint16_t>& input,
                                         int tokens, int head_dim,
                                         const std::string& dtype, bool normalize) {
  std::vector<double> output(static_cast<size_t>(tokens) * head_dim);
  const double scale = normalize ? 1.0 / std::sqrt(static_cast<double>(head_dim)) : 1.0;
  std::vector<double> buffer(head_dim);
  for (int token = 0; token < tokens; ++token) {
    const size_t base = static_cast<size_t>(token) * head_dim;
    for (int i = 0; i < head_dim; ++i) {
      buffer[i] = native_to_float(input[base + i], dtype);
    }
    for (int stride = 1; stride < head_dim; stride <<= 1) {
      for (int i = 0; i < head_dim; ++i) {
        if ((i & stride) == 0) {
          const double a = buffer[i];
          const double b = buffer[i | stride];
          buffer[i] = a + b;
          buffer[i | stride] = a - b;
        }
      }
    }
    for (int i = 0; i < head_dim; ++i) output[base + i] = buffer[i] * scale;
  }
  return output;
}

struct Accuracy {
  double reference_max_abs_error = 0.0;  // FULL tensor, low-precision reference
  double max_abs_error = 0.0;    // 题目直接给出的口径
  double rmse = 0.0;
  double max_rel_error = 0.0;    // |err| / max(1, 该 token 的 |y| 峰值)
  double max_ulp_error = 0.0;    // |err| / 该 token 峰值处的输出 dtype ULP
  double bit_exact_ratio = 0.0;  // 相对 optimized FHT 输出
};

static Accuracy evaluate(const std::vector<uint16_t>& candidate,
                         const std::vector<uint16_t>& optimized,
                         const std::vector<double>& reference, int ref_tokens,
                         int head_dim, const std::string& dtype,
                         const std::vector<uint16_t>& validation_reference) {
  Accuracy result;
  double squared = 0.0;
  const size_t ref_elements = static_cast<size_t>(ref_tokens) * head_dim;
  for (int token = 0; token < ref_tokens; ++token) {
    const size_t base = static_cast<size_t>(token) * head_dim;
    double token_max = 0.0;
    for (int i = 0; i < head_dim; ++i) {
      token_max = std::max(token_max, std::fabs(reference[base + i]));
    }
    const double token_ulp = dtype_ulp(token_max, dtype);
    // 相对口径的分母同样取 token 峰值，并对小量级 token 用 1 兜底，使该指标在
    // 输出量级本来就在 1 附近时退化成原来的绝对误差。
    const double token_norm = std::max(1.0, token_max);
    for (int i = 0; i < head_dim; ++i) {
      const double error = native_to_float(candidate[base + i], dtype) - reference[base + i];
      result.max_abs_error = std::max(result.max_abs_error, std::fabs(error));
      result.max_rel_error = std::max(result.max_rel_error, std::fabs(error) / token_norm);
      result.max_ulp_error = std::max(result.max_ulp_error, std::fabs(error) / token_ulp);
      squared += error * error;
    }
  }
  result.rmse = std::sqrt(squared / static_cast<double>(ref_elements));
  size_t equal = 0;
  for (size_t i = 0; i < candidate.size(); ++i) {
    if (candidate[i] == optimized[i]) ++equal;
    const double actual = native_to_float(candidate[i], dtype);
    const double expected = native_to_float(validation_reference[i], dtype);
    const double error = std::isfinite(actual) && std::isfinite(expected)
                             ? std::fabs(actual - expected)
                             : std::numeric_limits<double>::infinity();
    result.reference_max_abs_error = std::max(result.reference_max_abs_error, error);
  }
  result.bit_exact_ratio = static_cast<double>(equal) / static_cast<double>(candidate.size());
  return result;
}

// ---------------------------------------------------------------------------
// 计时
// ---------------------------------------------------------------------------

struct Timing {
  double avg_ms = 0.0;
  double min_ms = 0.0;
  double max_ms = 0.0;
};

template <typename Launch>
static Timing time_launch(Launch launch, int warmup, int iters, cudaStream_t stream) {
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

// ---------------------------------------------------------------------------
// CSV
// ---------------------------------------------------------------------------

struct Row {
  double reference_max_abs_error = 0.0;
  std::string implementation;
  std::string kind;  // transform / fused_int4
  Timing timing;
  double speedup_vs_baseline = 0.0;
  double speedup_vs_best_non_tc = 0.0;
  double max_abs_error = 0.0;
  double rmse = 0.0;
  double max_rel_error = 0.0;
  double max_ulp_error = 0.0;
  double bit_exact_ratio = 0.0;
  std::string quant_match;  // n/a / bit-exact / MISMATCH
};

static void write_csv(const Config& cfg, int total_tokens, const std::string& gpu,
                      const std::vector<Row>& rows) {
  if (cfg.csv.empty()) return;
  const fs::path path(cfg.csv);
  if (!path.parent_path().empty()) fs::create_directories(path.parent_path());
  const bool header = !fs::exists(path) || fs::file_size(path) == 0;
  std::ofstream file(path, std::ios::app);
  if (header) {
    file << "gpu,dtype,batch_size,seq_len,num_heads,head_dim,total_tokens,normalize,"
            "warmup_runs,bench_runs,implementation,kind,avg_ms,min_ms,max_ms,"
            "speedup_vs_baseline,speedup_vs_best_non_tc,max_abs_error,rmse,"
            "max_rel_error,max_ulp_error,bit_exact_vs_optimized,quant_vs_unfused,"
            "reference_source,reference_max_abs_error,absolute_pass,pdf_verified,"
            "validated_elements,fp64_ref_tokens\n";
  }
  for (const Row& row : rows) {
    file << gpu << ',' << cfg.dtype << ',' << cfg.batch << ',' << cfg.seq << ','
         << cfg.heads << ',' << cfg.head_dim << ',' << total_tokens << ','
         << (cfg.normalize ? "true" : "false") << ',' << cfg.warmup << ','
         << cfg.iters << ',' << row.implementation << ',' << row.kind << ','
         << row.timing.avg_ms << ',' << row.timing.min_ms << ',' << row.timing.max_ms
         << ',' << row.speedup_vs_baseline << ',' << row.speedup_vs_best_non_tc << ',';
    if (row.kind == "transform") {
      file << row.max_abs_error << ',' << row.rmse << ',' << row.max_rel_error << ','
           << row.max_ulp_error << ',' << row.bit_exact_ratio;
    } else {
      file << ",,,,";
    }
    file << ',' << row.quant_match << ','
         << (cfg.reference_bin.empty() ? "optimized_regression" : "external_native");
    if (row.kind == "transform") {
      const bool pass = row.reference_max_abs_error < (cfg.dtype == "fp16" ? 1e-2 : 5e-2);
      file << ',' << row.reference_max_abs_error << ',' << pass << ','
           << (pass && !cfg.reference_bin.empty()) << ','
           << static_cast<size_t>(total_tokens) * cfg.head_dim;
    } else file << ",,,,";
    file << ',' << std::min(cfg.ref_tokens, total_tokens) << '\n';
  }
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
  Config cfg;
  if (!parse_args(argc, argv, cfg)) return 2;
  const DataType dtype = cfg.dtype == "fp16" ? DataType::FP16 : DataType::BF16;
  const int total_tokens = cfg.batch * cfg.seq * cfg.heads;
  const size_t elements = static_cast<size_t>(total_tokens) * cfg.head_dim;
  const size_t tensor_bytes = elements * sizeof(uint16_t);
  const size_t packed_bytes = elements / 2;
  const size_t scale_bytes = static_cast<size_t>(total_tokens) * sizeof(float);
  const int ref_tokens = std::min(cfg.ref_tokens, total_tokens);
  // d<=256: complete tiles use TC, remaining tokens use the same warp fallback
  // as the transform API. The mixed path is validated, not skipped.
  const bool fused_tc_supported = cfg.head_dim <= 256;

  int device = 0;
  CUDA_CHECK(cudaGetDevice(&device));
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  std::printf("[tc] GPU=%s sm_%d%d dtype=%s head_dim=%d tokens=%d normalize=%s\n",
              prop.name, prop.major, prop.minor, cfg.dtype.c_str(), cfg.head_dim,
              total_tokens, cfg.normalize ? "true" : "false");
  if (prop.major < 8) {
    std::fprintf(stderr,
                 "[error] BF16 WMMA requires SM >= 80; this build targets SM 80+\n");
    return 3;
  }

  const std::vector<uint16_t> host_input = cfg.input_bin.empty()
      ? make_input(elements, cfg.dtype, cfg.seed) : load_native(cfg.input_bin, elements);

  void* input = nullptr;
  void* out_baseline = nullptr;
  void* out_optimized = nullptr;
  void* out_warp = nullptr;
  void* out_tc_fast = nullptr;
  void* out_tc_split = nullptr;
  unsigned char* packed_unfused = nullptr;
  unsigned char* packed_ref = nullptr;   // 逐算法的 "先变换后量化" 参考
  unsigned char* packed_opt = nullptr;
  unsigned char* packed_warp = nullptr;
  unsigned char* packed_tc_fast = nullptr;
  unsigned char* packed_tc_split = nullptr;
  float* scales_unfused = nullptr;
  float* scales_ref = nullptr;
  float* scales_opt = nullptr;
  float* scales_warp = nullptr;
  float* scales_tc_fast = nullptr;
  float* scales_tc_split = nullptr;
  CUDA_CHECK(cudaMalloc(&input, tensor_bytes));
  CUDA_CHECK(cudaMalloc(&out_baseline, tensor_bytes));
  CUDA_CHECK(cudaMalloc(&out_optimized, tensor_bytes));
  CUDA_CHECK(cudaMalloc(&out_warp, tensor_bytes));
  CUDA_CHECK(cudaMalloc(&out_tc_fast, tensor_bytes));
  CUDA_CHECK(cudaMalloc(&out_tc_split, tensor_bytes));
  CUDA_CHECK(cudaMalloc(&packed_unfused, packed_bytes));
  CUDA_CHECK(cudaMalloc(&packed_ref, packed_bytes));
  CUDA_CHECK(cudaMalloc(&packed_opt, packed_bytes));
  CUDA_CHECK(cudaMalloc(&packed_warp, packed_bytes));
  CUDA_CHECK(cudaMalloc(&packed_tc_fast, packed_bytes));
  CUDA_CHECK(cudaMalloc(&packed_tc_split, packed_bytes));
  CUDA_CHECK(cudaMalloc(&scales_unfused, scale_bytes));
  CUDA_CHECK(cudaMalloc(&scales_ref, scale_bytes));
  CUDA_CHECK(cudaMalloc(&scales_opt, scale_bytes));
  CUDA_CHECK(cudaMalloc(&scales_warp, scale_bytes));
  CUDA_CHECK(cudaMalloc(&scales_tc_fast, scale_bytes));
  CUDA_CHECK(cudaMalloc(&scales_tc_split, scale_bytes));
  CUDA_CHECK(cudaMemcpy(input, host_input.data(), tensor_bytes, cudaMemcpyHostToDevice));

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  auto check = [](int code, const char* what) {
    if (code != 0) {
      std::fprintf(stderr, "[error] %s returned %d\n", what, code);
      std::exit(4);
    }
  };

  auto run_baseline = [&]() {
    check(launch_hadamard(input, out_baseline, cfg.batch, cfg.seq, cfg.heads,
                          cfg.head_dim, dtype, cfg.normalize, stream), "baseline");
  };
  auto run_optimized = [&]() {
    check(launch_hadamard_optimized(input, out_optimized, cfg.batch, cfg.seq,
                                    cfg.heads, cfg.head_dim, dtype, cfg.normalize,
                                    stream), "optimized");
  };
  auto run_warp = [&]() {
    check(launch_hadamard_warp(input, out_warp, cfg.batch, cfg.seq, cfg.heads,
                               cfg.head_dim, dtype, cfg.normalize, stream), "warp");
  };
  auto run_tc_fast = [&]() {
    check(launch_hadamard_tc(input, out_tc_fast, cfg.batch, cfg.seq, cfg.heads,
                             cfg.head_dim, dtype, cfg.normalize, TcMode::Fast,
                             stream), "tc_fast");
  };
  auto run_tc_split = [&]() {
    check(launch_hadamard_tc(input, out_tc_split, cfg.batch, cfg.seq, cfg.heads,
                             cfg.head_dim, dtype, cfg.normalize, TcMode::Split,
                             stream), "tc_split");
  };
  auto run_unfused = [&]() {
    check(launch_hadamard_warp(input, out_warp, cfg.batch, cfg.seq, cfg.heads,
                               cfg.head_dim, dtype, cfg.normalize, stream), "warp");
    check(launch_quantize_int4(out_warp, packed_unfused, scales_unfused,
                               total_tokens, cfg.head_dim, dtype, stream),
          "quantize_int4");
  };
  auto run_fused_opt = [&]() {
    check(launch_hadamard_fused_quant_int4(input, packed_opt, scales_opt, cfg.batch,
                                           cfg.seq, cfg.heads, cfg.head_dim, dtype,
                                           cfg.normalize, stream), "fused_opt_int4");
  };
  auto run_fused_warp = [&]() {
    check(launch_hadamard_fused_quant_int4_warp(input, packed_warp, scales_warp,
                                                cfg.batch, cfg.seq, cfg.heads,
                                                cfg.head_dim, dtype, cfg.normalize,
                                                stream), "fused_warp_int4");
  };
  auto run_fused_tc_fast = [&]() {
    check(launch_hadamard_tc_fused_quant_int4(input, packed_tc_fast, scales_tc_fast,
                                              cfg.batch, cfg.seq, cfg.heads,
                                              cfg.head_dim, dtype, cfg.normalize,
                                              TcMode::Fast, stream),
          "fused_tc_fast_int4");
  };
  auto run_fused_tc_split = [&]() {
    check(launch_hadamard_tc_fused_quant_int4(input, packed_tc_split, scales_tc_split,
                                              cfg.batch, cfg.seq, cfg.heads,
                                              cfg.head_dim, dtype, cfg.normalize,
                                              TcMode::Split, stream),
          "fused_tc_split_int4");
  };

  // ---- 正确性 ----
  run_baseline();
  run_optimized();
  run_warp();
  run_tc_fast();
  run_tc_split();
  run_unfused();
  run_fused_opt();
  run_fused_warp();
  if (fused_tc_supported) {
    run_fused_tc_fast();
    run_fused_tc_split();
  }
  CUDA_CHECK(cudaStreamSynchronize(stream));

  std::vector<uint16_t> host_baseline(elements);
  std::vector<uint16_t> host_optimized(elements);
  std::vector<uint16_t> host_warp(elements);
  std::vector<uint16_t> host_tc_fast(elements);
  std::vector<uint16_t> host_tc_split(elements);
  CUDA_CHECK(cudaMemcpy(host_baseline.data(), out_baseline, tensor_bytes, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(host_optimized.data(), out_optimized, tensor_bytes, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(host_warp.data(), out_warp, tensor_bytes, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(host_tc_fast.data(), out_tc_fast, tensor_bytes, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(host_tc_split.data(), out_tc_split, tensor_bytes, cudaMemcpyDeviceToHost));

  std::vector<unsigned char> host_packed_unfused(packed_bytes);
  std::vector<unsigned char> host_packed_opt(packed_bytes);
  std::vector<unsigned char> host_packed_warp(packed_bytes);
  std::vector<unsigned char> host_packed_tc_fast(packed_bytes);
  std::vector<unsigned char> host_packed_tc_split(packed_bytes);
  std::vector<float> host_scales_unfused(total_tokens);
  std::vector<float> host_scales_opt(total_tokens);
  std::vector<float> host_scales_warp(total_tokens);
  std::vector<float> host_scales_tc_fast(total_tokens);
  std::vector<float> host_scales_tc_split(total_tokens);
  CUDA_CHECK(cudaMemcpy(host_packed_unfused.data(), packed_unfused, packed_bytes, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(host_packed_opt.data(), packed_opt, packed_bytes, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(host_packed_warp.data(), packed_warp, packed_bytes, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(host_scales_unfused.data(), scales_unfused, scale_bytes, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(host_scales_opt.data(), scales_opt, scale_bytes, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(host_scales_warp.data(), scales_warp, scale_bytes, cudaMemcpyDeviceToHost));
  if (fused_tc_supported) {
    CUDA_CHECK(cudaMemcpy(host_packed_tc_fast.data(), packed_tc_fast, packed_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(host_packed_tc_split.data(), packed_tc_split, packed_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(host_scales_tc_fast.data(), scales_tc_fast, scale_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(host_scales_tc_split.data(), scales_tc_split, scale_bytes, cudaMemcpyDeviceToHost));
  }

  const std::vector<double> reference =
      reference_fht(host_input, ref_tokens, cfg.head_dim, cfg.dtype, cfg.normalize);
  const std::vector<uint16_t> validation_reference = cfg.reference_bin.empty()
      ? host_optimized : load_native(cfg.reference_bin, elements);
  const Accuracy acc_baseline = evaluate(host_baseline, host_optimized, reference, ref_tokens, cfg.head_dim, cfg.dtype, validation_reference);
  const Accuracy acc_optimized = evaluate(host_optimized, host_optimized, reference, ref_tokens, cfg.head_dim, cfg.dtype, validation_reference);
  const Accuracy acc_warp = evaluate(host_warp, host_optimized, reference, ref_tokens, cfg.head_dim, cfg.dtype, validation_reference);
  const Accuracy acc_tc_fast = evaluate(host_tc_fast, host_optimized, reference, ref_tokens, cfg.head_dim, cfg.dtype, validation_reference);
  const Accuracy acc_tc_split = evaluate(host_tc_split, host_optimized, reference, ref_tokens, cfg.head_dim, cfg.dtype, validation_reference);
  if (!cfg.dump_dir.empty()) {
    fs::create_directories(cfg.dump_dir);
    auto dump = [&](const char* name, const std::vector<uint16_t>& data) {
      std::ofstream file(fs::path(cfg.dump_dir) / name, std::ios::binary);
      file.write(reinterpret_cast<const char*>(data.data()), tensor_bytes);
      if (!file) { std::fprintf(stderr, "[error] dump failed: %s\n", name); std::exit(2); }
    };
    dump("input.bin", host_input);
    dump("reference.bin", validation_reference);
    dump("baseline.bin", host_baseline);
    dump("optimized.bin", host_optimized);
    dump("warp.bin", host_warp);
    dump("tc_fast.bin", host_tc_fast);
    dump("tc_split.bin", host_tc_split);
    std::ofstream meta(fs::path(cfg.dump_dir) / "meta.json");
    meta << "{\"shape\":[" << cfg.batch << ',' << cfg.seq << ',' << cfg.heads << ','
         << cfg.head_dim << "],\"dtype\":\"" << cfg.dtype << "\",\"normalize\":"
         << (cfg.normalize ? "true" : "false") << ",\"reference_source\":\""
         << (cfg.reference_bin.empty() ? "optimized_regression" : "external_native") << "\"}\n";
  }

  // 题目要求 “融合量化的结果与先变换后量化的结果一致”。这个检查必须按算法配对：
  // 每条融合路径都与 “同一算法的变换输出 -> 独立 quantize kernel” 比较，
  // 否则比较的是两种算法的数值差异，而不是融合本身是否正确。
  std::vector<unsigned char> host_ref_packed(packed_bytes);
  std::vector<float> host_ref_scales(total_tokens);
  auto quantize_reference_of = [&](void* transform_output) {
    check(launch_quantize_int4(transform_output, packed_ref, scales_ref,
                               total_tokens, cfg.head_dim, dtype, stream),
          "quantize_int4");
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaMemcpy(host_ref_packed.data(), packed_ref, packed_bytes,
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(host_ref_scales.data(), scales_ref, scale_bytes,
                          cudaMemcpyDeviceToHost));
  };
  auto compare_with_reference = [&](const std::vector<unsigned char>& packed,
                                    const std::vector<float>& scales) {
    const bool same = packed == host_ref_packed &&
                      std::memcmp(scales.data(), host_ref_scales.data(),
                                  scale_bytes) == 0;
    return std::string(same ? "bit-exact" : "MISMATCH");
  };
  quantize_reference_of(out_optimized);
  const std::string match_opt = compare_with_reference(host_packed_opt, host_scales_opt);
  quantize_reference_of(out_warp);
  const std::string match_warp = compare_with_reference(host_packed_warp, host_scales_warp);
  std::string match_tc_fast = "n/a";
  std::string match_tc_split = "n/a";
  if (fused_tc_supported) {
    quantize_reference_of(out_tc_fast);
    match_tc_fast = compare_with_reference(host_packed_tc_fast, host_scales_tc_fast);
    quantize_reference_of(out_tc_split);
    match_tc_split = compare_with_reference(host_packed_tc_split, host_scales_tc_split);
  }
  // 另外记录：TC 融合结果与 FP32 FHT 融合结果之间是否也逐位一致（更强的口径）。
  const std::string cross_tc_split =
      fused_tc_supported && host_packed_tc_split == host_packed_warp &&
              std::memcmp(host_scales_tc_split.data(), host_scales_warp.data(),
                          scale_bytes) == 0
          ? "bit-exact"
          : (fused_tc_supported ? "differs" : "n/a");

  const double threshold = cfg.dtype == "fp16" ? 1e-2 : 5e-2;
  // Strict absolute error over every element; FP64 and relative errors are
  // diagnostics only. External reference must use the same input/dtype/scale.
  auto ok = [&](const Accuracy& acc) {
    return acc.reference_max_abs_error < threshold;
  };
  auto verdict = [&](const Accuracy& acc) {
    if (!ok(acc)) return "FAIL(abs)";
    return cfg.reference_bin.empty() ? "REGRESSION_PASS(not PDF verified)" : "PASS(abs)";
  };
  std::printf("[tc] full-tensor absolute check: reference=%s elements=%zu\n",
      cfg.reference_bin.empty() ? "optimized_regression" : "external_native", elements);
  std::printf("[tc] accuracy vs FP64 reference (first %d tokens, threshold %.0e)\n",
              ref_tokens, threshold);
  auto report = [&](const char* name, const Accuracy& acc) {
    std::printf("       %-10s max_abs=%.6e max_rel=%.6e max_ulp=%.3f rmse=%.6e "
                "bit_exact_vs_optimized=%.4f reference_max_abs=%.6e %s\n",
                name, acc.max_abs_error, acc.max_rel_error, acc.max_ulp_error,
                acc.rmse, acc.bit_exact_ratio, acc.reference_max_abs_error, verdict(acc));
  };
  report("baseline", acc_baseline);
  report("warp", acc_warp);
  report("tc_fast", acc_tc_fast);
  report("tc_split", acc_tc_split);
  std::printf("[tc] fused vs same-algorithm unfused: opt(9.2)=%s warp=%s "
              "tc_fast=%s tc_split=%s\n",
              match_opt.c_str(), match_warp.c_str(), match_tc_fast.c_str(),
              match_tc_split.c_str());
  std::printf("[tc] fused_tc_split vs fused_warp (FP32 FHT) INT4 bytes: %s\n",
              cross_tc_split.c_str());

  // ---- 计时 ----
  const Timing t_baseline = time_launch(run_baseline, cfg.warmup, cfg.iters, stream);
  const Timing t_optimized = time_launch(run_optimized, cfg.warmup, cfg.iters, stream);
  const Timing t_warp = time_launch(run_warp, cfg.warmup, cfg.iters, stream);
  const Timing t_tc_fast = time_launch(run_tc_fast, cfg.warmup, cfg.iters, stream);
  const Timing t_tc_split = time_launch(run_tc_split, cfg.warmup, cfg.iters, stream);
  const Timing t_unfused = time_launch(run_unfused, cfg.warmup, cfg.iters, stream);
  const Timing t_fused_opt = time_launch(run_fused_opt, cfg.warmup, cfg.iters, stream);
  const Timing t_fused_warp = time_launch(run_fused_warp, cfg.warmup, cfg.iters, stream);
  Timing t_fused_tc_fast;
  Timing t_fused_tc_split;
  if (fused_tc_supported) {
    t_fused_tc_fast = time_launch(run_fused_tc_fast, cfg.warmup, cfg.iters, stream);
    t_fused_tc_split = time_launch(run_fused_tc_split, cfg.warmup, cfg.iters, stream);
  }

  // 非 TC 的最好成绩作为 Tensor Core 的公平参照（题目要求的“进阶对比”）。
  const double best_non_tc = std::min(t_optimized.avg_ms, t_warp.avg_ms);
  const double best_non_tc_fused =
      std::min(std::min(t_unfused.avg_ms, t_fused_opt.avg_ms), t_fused_warp.avg_ms);

  std::vector<Row> rows;
  auto add_transform = [&](const char* name, const Timing& timing, const Accuracy& acc) {
    Row row;
    row.implementation = name;
    row.kind = "transform";
    row.timing = timing;
    row.speedup_vs_baseline = t_baseline.avg_ms / timing.avg_ms;
    row.speedup_vs_best_non_tc = best_non_tc / timing.avg_ms;
    row.max_abs_error = acc.max_abs_error;
    row.reference_max_abs_error = acc.reference_max_abs_error;
    row.rmse = acc.rmse;
    row.max_rel_error = acc.max_rel_error;
    row.max_ulp_error = acc.max_ulp_error;
    row.bit_exact_ratio = acc.bit_exact_ratio;
    row.quant_match = "n/a";
    rows.push_back(row);
  };
  auto add_fused = [&](const char* name, const Timing& timing, const std::string& match) {
    Row row;
    row.implementation = name;
    row.kind = "fused_int4";
    row.timing = timing;
    row.speedup_vs_baseline = t_unfused.avg_ms / timing.avg_ms;
    row.speedup_vs_best_non_tc = best_non_tc_fused / timing.avg_ms;
    row.quant_match = match;
    rows.push_back(row);
  };
  add_transform("baseline", t_baseline, acc_baseline);
  add_transform("optimized", t_optimized, acc_optimized);
  add_transform("warp", t_warp, acc_warp);
  add_transform("tc_fast", t_tc_fast, acc_tc_fast);
  add_transform("tc_split", t_tc_split, acc_tc_split);
  add_fused("unfused_int4", t_unfused, "reference");
  add_fused("fused_opt_int4", t_fused_opt, match_opt);
  add_fused("fused_warp_int4", t_fused_warp, match_warp);
  if (fused_tc_supported) {
    add_fused("fused_tc_fast_int4", t_fused_tc_fast, match_tc_fast);
    add_fused("fused_tc_split_int4", t_fused_tc_split, match_tc_split);
  }

  std::printf("[tc] transform timings (avg ms, speedup vs baseline / vs best non-TC)\n");
  for (const Row& row : rows) {
    if (row.kind != "transform") continue;
    std::printf("       %-10s %.6f  %.2fx  %.2fx\n", row.implementation.c_str(),
                row.timing.avg_ms, row.speedup_vs_baseline, row.speedup_vs_best_non_tc);
  }
  std::printf("[tc] fused INT4 timings (avg ms, speedup vs unfused / vs best non-TC fused)\n");
  for (const Row& row : rows) {
    if (row.kind != "fused_int4") continue;
    std::printf("       %-20s %.6f  %.2fx  %.2fx  quant=%s\n",
                row.implementation.c_str(), row.timing.avg_ms,
                row.speedup_vs_baseline, row.speedup_vs_best_non_tc,
                row.quant_match.c_str());
  }

  write_csv(cfg, total_tokens, prop.name, rows);

  CUDA_CHECK(cudaStreamDestroy(stream));
  for (void* ptr : {input, out_baseline, out_optimized, out_warp, out_tc_fast, out_tc_split}) {
    CUDA_CHECK(cudaFree(ptr));
  }
  for (void* ptr : {static_cast<void*>(packed_unfused), static_cast<void*>(packed_ref),
                    static_cast<void*>(scales_ref), static_cast<void*>(packed_opt),
                    static_cast<void*>(scales_opt), static_cast<void*>(packed_warp),
                    static_cast<void*>(packed_tc_fast), static_cast<void*>(packed_tc_split),
                    static_cast<void*>(scales_unfused), static_cast<void*>(scales_warp),
                    static_cast<void*>(scales_tc_fast), static_cast<void*>(scales_tc_split)}) {
    CUDA_CHECK(cudaFree(ptr));
  }

  const bool accuracy_ok = ok(acc_baseline) && ok(acc_optimized) && ok(acc_warp) &&
                           ok(acc_tc_fast) && ok(acc_tc_split);
  const bool quant_ok = match_opt == "bit-exact" && match_warp == "bit-exact" &&
                        (!fused_tc_supported ||
                         (match_tc_fast == "bit-exact" && match_tc_split == "bit-exact"));
  return accuracy_ok && quant_ok ? 0 : 1;
}
