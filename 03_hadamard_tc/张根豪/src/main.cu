// main.cu — hadamard_bench 可执行程序（Week1/Week2 交付物）
//
// 功能：
//   1. 生成可复现的测试输入（normal / uniform / outlier 三种分布，固定种子）；
//   2. 运行 CUDA Hadamard kernel 并用 CUDA Event 计时（支持 warmup + 多次迭代，
//      输出 avg / min / max 毫秒）；
//   3. --check true 时，用 CPU FP32 参考实现计算标准答案并比较误差，
//      输出 max_abs_error / mean_abs_error / rmse / pass-fail；
//   4. 可选把输入/输出 dump 成二进制文件，交给 tests/ 下的 Python 脚本做独立交叉验证；
//   5. 可选把结果追加到 CSV，便于 scripts/ 汇总报告。
//
// 正确性验证流程（对应 skill.md Week2 第 7 条）：
//   生成随机输入 -> H2D 拷贝 -> 运行 kernel -> D2H 拷贝
//   -> CPU FP32 参考实现 -> 误差比较 -> PASS/FAIL
//   比较统一在 FP32 域进行，且双方使用相同归一化策略。
//
// 用法示例：
//   ./hadamard_bench --batch 4 --seq 1024 --heads 32 --head_dim 128 \
//                    --dtype fp16 --normalize true --warmup 5 --iters 20 --check true

#include "cuda_check.cuh"
#include "hadamard.cuh"
#include "reference_cpu.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cctype>
#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

namespace fs = std::filesystem;

// ---------------------------------------------------------------------------
// 命令行配置
// ---------------------------------------------------------------------------

struct Config {
  int batch_size = 4;
  int seq_len = 1024;
  int num_heads = 32;
  int head_dim = 128;
  std::string dtype = "fp16";       // fp16 | bf16
  bool normalize = true;            // 默认归一化，理由见 docs/final_report.md
  std::string dist = "normal";      // normal | uniform | outlier
  uint64_t seed = 42;               // 固定种子保证可复现
  double outlier_ratio = 0.05;      // outlier 分布：被放大的元素比例
  double outlier_scale = 20.0;      // outlier 分布：放大倍数
  int warmup = 5;
  int iters = 20;
  bool check = true;
  std::string dump_dir = "";        // 非空时 dump 二进制
  std::string csv_path = "";        // 非空时追加 CSV
  std::string device_json = "";     // 非空时写出 CUDA 设备属性（Roofline 用）
  std::string input_bin = "";       // 非空时从文件加载低精度输入（跳过随机生成）
};

static void print_usage(const char* prog) {
  std::printf(
      "Usage: %s [options]\n"
      "  --batch <int>          batch size            (default 4)\n"
      "  --seq <int>            sequence length       (default 1024)\n"
      "  --heads <int>          number of heads       (default 32)\n"
      "  --head_dim <int>       2 的幂, 支持 32~1024  (default 128)\n"
      "  --dtype <str>          fp16 | bf16           (default fp16)\n"
      "  --normalize <bool>     true: y=Hx/sqrt(d)    (default true)\n"
      "  --dist <str>           normal | uniform | outlier (default normal)\n"
      "  --seed <int>           RNG 种子              (default 42)\n"
      "  --outlier_ratio <f>    outlier 比例          (default 0.05)\n"
      "  --outlier_scale <f>    outlier 放大倍数      (default 20)\n"
      "  --warmup <int>         warmup 次数           (default 5)\n"
      "  --iters <int>          计时迭代次数          (default 20)\n"
      "  --check <bool>         是否做正确性检查      (default true)\n"
      "  --dump_dir <path>      dump 二进制的目录     (default 不 dump)\n"
      "  --csv <path>           追加 CSV 日志路径     (default 不写)\n"
      "  --device_json <path>   写出 CUDA 设备属性 JSON (default 不写)\n"
      "  --input_bin <path>     从文件加载输入(低精度原始字节) (default 随机生成)\n",
      prog);
}

static bool parse_bool(const char* s) {
  std::string v(s);
  std::transform(v.begin(), v.end(), v.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return v == "true" || v == "1" || v == "yes" || v == "on";
}

// 极简参数解析：--key value 成对出现
static bool parse_args(int argc, char** argv, Config& cfg) {
  for (int i = 1; i < argc; ++i) {
    const std::string key = argv[i];
    auto next = [&](const char* what) -> const char* {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "[error] %s 需要一个参数: %s\n", what, key.c_str());
        std::exit(2);
      }
      return argv[++i];
    };
    if (key == "--batch") cfg.batch_size = std::atoi(next("int"));
    else if (key == "--seq") cfg.seq_len = std::atoi(next("int"));
    else if (key == "--heads") cfg.num_heads = std::atoi(next("int"));
    else if (key == "--head_dim") cfg.head_dim = std::atoi(next("int"));
    else if (key == "--dtype") cfg.dtype = next("str");
    else if (key == "--normalize") cfg.normalize = parse_bool(next("bool"));
    else if (key == "--dist") cfg.dist = next("str");
    else if (key == "--seed") cfg.seed = std::strtoull(next("int"), nullptr, 10);
    else if (key == "--outlier_ratio") cfg.outlier_ratio = std::atof(next("float"));
    else if (key == "--outlier_scale") cfg.outlier_scale = std::atof(next("float"));
    else if (key == "--warmup") cfg.warmup = std::atoi(next("int"));
    else if (key == "--iters") cfg.iters = std::atoi(next("int"));
    else if (key == "--check") cfg.check = parse_bool(next("bool"));
    else if (key == "--dump_dir") cfg.dump_dir = next("path");
    else if (key == "--csv") cfg.csv_path = next("path");
    else if (key == "--device_json") cfg.device_json = next("path");
    else if (key == "--input_bin") cfg.input_bin = next("path");
    else if (key == "--help" || key == "-h") { print_usage(argv[0]); std::exit(0); }
    else {
      std::fprintf(stderr, "[error] 未知参数: %s\n", key.c_str());
      print_usage(argv[0]);
      return false;
    }
  }
  // 合法性检查
  bool ok = true;
  auto positive = [&](int v, const char* name) {
    if (v < 1) { std::fprintf(stderr, "[error] %s 必须 >= 1 (当前 %d)\n", name, v); ok = false; }
  };
  positive(cfg.batch_size, "batch");
  positive(cfg.seq_len, "seq");
  positive(cfg.num_heads, "heads");
  if (cfg.head_dim < 2 || (cfg.head_dim & (cfg.head_dim - 1)) != 0 ||
      cfg.head_dim > 1024) {
    std::fprintf(stderr, "[error] head_dim 必须是 2 的幂且 <= 1024 (当前 %d)\n",
                 cfg.head_dim);
    ok = false;
  }
  if (cfg.dtype != "fp16" && cfg.dtype != "bf16") {
    std::fprintf(stderr, "[error] dtype 仅支持 fp16 / bf16 (当前 %s)\n",
                 cfg.dtype.c_str());
    ok = false;
  }
  if (cfg.dist != "normal" && cfg.dist != "uniform" && cfg.dist != "outlier") {
    std::fprintf(stderr, "[error] dist 仅支持 normal / uniform / outlier (当前 %s)\n",
                 cfg.dist.c_str());
    ok = false;
  }
  if (cfg.warmup < 0 || cfg.iters < 1) {
    std::fprintf(stderr, "[error] warmup 必须 >= 0, iters 必须 >= 1\n");
    ok = false;
  }
  return ok;
}

// ---------------------------------------------------------------------------
// 数据生成（host 侧，FP32 生成后转为目标低精度，保证输入可复现）
// ---------------------------------------------------------------------------

// 生成 FP32 原始数据，随后按目标 dtype 截断。
// 返回 native 位宽数据（uint16）以及"低精度输入反量化回 FP32"的副本——
// 参考实现必须消费与 GPU 完全相同的低精度值，因此需要这个 round-trip。
struct HostTensors {
  std::vector<uint16_t> native_input;   // 低精度输入（喂给 GPU）
  std::vector<float> input_fp32;        // 低精度输入反量化后的 FP32（喂给参考实现）
};

static HostTensors generate_input(const Config& cfg, long long total_tokens) {
  const long long n = total_tokens * cfg.head_dim;
  std::vector<float> fp32(static_cast<size_t>(n));

  std::mt19937 gen(cfg.seed);
  std::normal_distribution<float> normal(0.0f, 1.0f);
  std::uniform_real_distribution<float> uniform(-1.0f, 1.0f);
  std::uniform_real_distribution<float> unit(0.0f, 1.0f);

  for (long long i = 0; i < n; ++i) {
    float v;
    if (cfg.dist == "uniform") {
      v = uniform(gen);
    } else {
      v = normal(gen);
      if (cfg.dist == "outlier" && unit(gen) < cfg.outlier_ratio) {
        v *= static_cast<float>(cfg.outlier_scale);  // 模拟激活异常值
      }
    }
    fp32[static_cast<size_t>(i)] = v;
  }

  HostTensors t;
  t.native_input.resize(static_cast<size_t>(n));
  t.input_fp32.resize(static_cast<size_t>(n));

  if (cfg.dtype == "fp16") {
    for (long long i = 0; i < n; ++i) {
      const __half h = __float2half(fp32[static_cast<size_t>(i)]);
      uint16_t bits;
      std::memcpy(&bits, &h, sizeof(bits));
      t.native_input[static_cast<size_t>(i)] = bits;
      t.input_fp32[static_cast<size_t>(i)] = __half2float(h);
    }
  } else {  // bf16
    for (long long i = 0; i < n; ++i) {
      const __nv_bfloat16 b = __float2bfloat16(fp32[static_cast<size_t>(i)]);
      uint16_t bits;
      std::memcpy(&bits, &b, sizeof(bits));
      t.native_input[static_cast<size_t>(i)] = bits;
      t.input_fp32[static_cast<size_t>(i)] = __bfloat162float(b);
    }
  }
  return t;
}

// 把 GPU 输出的低精度结果转回 FP32，用于误差比较（比较统一在 FP32 域）
static std::vector<float> native_to_fp32(const std::vector<uint16_t>& native,
                                         const std::string& dtype) {
  std::vector<float> out(native.size());
  if (dtype == "fp16") {
    for (size_t i = 0; i < native.size(); ++i) {
      __half h;
      std::memcpy(&h, &native[i], sizeof(h));
      out[i] = __half2float(h);
    }
  } else {
    for (size_t i = 0; i < native.size(); ++i) {
      __nv_bfloat16 b;
      std::memcpy(&b, &native[i], sizeof(b));
      out[i] = __bfloat162float(b);
    }
  }
  return out;
}

// 从二进制文件加载低精度输入（小端 uint16 原始字节），并反量化出参考用 FP32。
// 便于复现特定输入（例如某个失败 token），也支持"预生成数据文件"的工作流。
static HostTensors load_input_bin(const Config& cfg, long long total_tokens) {
  const long long n = total_tokens * cfg.head_dim;
  std::ifstream f(cfg.input_bin, std::ios::binary);
  if (!f) {
    std::fprintf(stderr, "[error] 无法读取输入文件 %s\n", cfg.input_bin.c_str());
    std::exit(2);
  }
  HostTensors t;
  t.native_input.resize(static_cast<size_t>(n));
  f.read(reinterpret_cast<char*>(t.native_input.data()),
         static_cast<std::streamsize>(n * sizeof(uint16_t)));
  if (f.gcount() != static_cast<std::streamsize>(n * sizeof(uint16_t)) ||
      f.peek() != std::char_traits<char>::eof()) {
    std::fprintf(stderr,
                 "[error] %s 大小与形状不符：期望 %lld 字节 (total_tokens=%lld, "
                 "head_dim=%d, dtype=%s)\n",
                 cfg.input_bin.c_str(), n * 2LL, total_tokens, cfg.head_dim,
                 cfg.dtype.c_str());
    std::exit(2);
  }
  t.input_fp32 = native_to_fp32(t.native_input, cfg.dtype);
  std::printf("[hadamard_bench] 从 %s 加载输入 (%lld 元素)\n",
              cfg.input_bin.c_str(), n);
  return t;
}

// ---------------------------------------------------------------------------
// dump 二进制 + meta（供 Python 交叉验证脚本使用）
// ---------------------------------------------------------------------------

static void write_raw(const fs::path& p, const void* data, size_t bytes) {
  std::ofstream f(p, std::ios::binary);
  if (!f) {
    std::fprintf(stderr, "[error] 无法写入 %s\n", p.c_str());
    std::exit(EXIT_FAILURE);
  }
  f.write(static_cast<const char*>(data), static_cast<std::streamsize>(bytes));
}

static void dump_binaries(const Config& cfg, long long total_tokens,
                          const std::vector<uint16_t>& native_input,
                          const std::vector<uint16_t>& native_output,
                          const std::vector<float>& ref_fp32) {
  fs::create_directories(cfg.dump_dir);
  const size_t n = static_cast<size_t>(total_tokens * cfg.head_dim);
  write_raw(fs::path(cfg.dump_dir) / "input.bin", native_input.data(),
            n * sizeof(uint16_t));
  write_raw(fs::path(cfg.dump_dir) / "gpu_output.bin", native_output.data(),
            n * sizeof(uint16_t));
  write_raw(fs::path(cfg.dump_dir) / "ref_output.bin", ref_fp32.data(),
            n * sizeof(float));
  std::ofstream meta(fs::path(cfg.dump_dir) / "meta.json");
  meta << "{\n"
       << "  \"dtype\": \"" << cfg.dtype << "\",\n"
       << "  \"batch_size\": " << cfg.batch_size << ",\n"
       << "  \"seq_len\": " << cfg.seq_len << ",\n"
       << "  \"num_heads\": " << cfg.num_heads << ",\n"
       << "  \"head_dim\": " << cfg.head_dim << ",\n"
       << "  \"total_tokens\": " << total_tokens << ",\n"
       << "  \"normalize\": " << (cfg.normalize ? "true" : "false") << ",\n"
       << "  \"dist\": \"" << cfg.dist << "\",\n"
       << "  \"seed\": " << cfg.seed << ",\n"
       << "  \"native_elem_bytes\": 2\n"
       << "}\n";
  std::printf("[hadamard_bench] dumped binaries to %s\n", cfg.dump_dir.c_str());
}

// ---------------------------------------------------------------------------
// CSV 日志
// ---------------------------------------------------------------------------

static void append_csv(const Config& cfg, long long total_tokens, int warmup,
                       int iters, double avg_ms, double min_ms, double max_ms,
                       bool checked, const ErrorMetrics& m, double threshold,
                       bool pass) {
  const bool need_header =
      !fs::exists(cfg.csv_path) || fs::file_size(cfg.csv_path) == 0;
  if (need_header && !fs::exists(cfg.csv_path)) {
    fs::create_directories(fs::path(cfg.csv_path).parent_path().empty()
                               ? fs::path(".")
                               : fs::path(cfg.csv_path).parent_path());
  }
  std::ofstream f(cfg.csv_path, std::ios::app);
  if (!f) {
    std::fprintf(stderr, "[error] 无法写入 CSV %s\n", cfg.csv_path.c_str());
    return;
  }
  if (need_header) {
    f << "dtype,batch_size,seq_len,num_heads,head_dim,total_tokens,normalize,"
         "dist,seed,warmup_runs,bench_runs,avg_kernel_ms,min_kernel_ms,"
         "max_kernel_ms,check_enabled,max_abs_error,mean_abs_error,rmse,"
         "max_rel_error,threshold,pass\n";
  }
  char err_buf[160] = "";
  if (checked) {
    std::snprintf(err_buf, sizeof(err_buf), "%.6e,%.6e,%.6e,%.6e,%.1e,%s",
                  m.max_abs_error, m.mean_abs_error, m.rmse, m.max_rel_error,
                  threshold, pass ? "true" : "false");
  } else {
    std::snprintf(err_buf, sizeof(err_buf), ",,,,," );
  }
  f << cfg.dtype << "," << cfg.batch_size << "," << cfg.seq_len << ","
    << cfg.num_heads << "," << cfg.head_dim << "," << total_tokens << ","
    << (cfg.normalize ? "true" : "false") << "," << cfg.dist << "," << cfg.seed
    << "," << warmup << "," << iters << "," << avg_ms << "," << min_ms << ","
    << max_ms << "," << (cfg.check ? "true" : "false") << "," << err_buf
    << "\n";
}

// 保存计算 Roofline 所需的设备上限。clockRate/memoryClockRate 来自 CUDA runtime
// 的设备属性；绘图脚本会明确把由这些属性推导的 roof 标为 theoretical。
static void write_device_json(const std::string& path, int device,
                              const cudaDeviceProp& prop) {
  const fs::path output(path);
  if (!output.parent_path().empty()) {
    fs::create_directories(output.parent_path());
  }
  std::ofstream f(output);
  if (!f) {
    std::fprintf(stderr, "[error] 无法写入设备信息 %s\n", path.c_str());
    std::exit(EXIT_FAILURE);
  }
  int clock_rate_khz = 0;
  int memory_clock_rate_khz = 0;
#if CUDART_VERSION >= 13000
  // CUDA 13 removed these two deprecated cudaDeviceProp members. Their
  // cudaDeviceAttr ABI identifiers remain 13 and 36, respectively.
  CUDA_CHECK(cudaDeviceGetAttribute(
      &clock_rate_khz, static_cast<cudaDeviceAttr>(13), device));
  CUDA_CHECK(cudaDeviceGetAttribute(
      &memory_clock_rate_khz, static_cast<cudaDeviceAttr>(36), device));
#else
  clock_rate_khz = prop.clockRate;
  memory_clock_rate_khz = prop.memoryClockRate;
#endif
  f << "{\n"
    << "  \"device_index\": " << device << ",\n"
    << "  \"name\": \"" << prop.name << "\",\n"
    << "  \"compute_capability_major\": " << prop.major << ",\n"
    << "  \"compute_capability_minor\": " << prop.minor << ",\n"
    << "  \"sm_count\": " << prop.multiProcessorCount << ",\n"
    << "  \"clock_rate_khz\": " << clock_rate_khz << ",\n"
    << "  \"memory_clock_rate_khz\": " << memory_clock_rate_khz << ",\n"
    << "  \"memory_bus_width_bits\": " << prop.memoryBusWidth << ",\n"
    << "  \"total_global_memory_bytes\": " << prop.totalGlobalMem << ",\n"
    << "  \"l2_cache_bytes\": " << prop.l2CacheSize << ",\n"
    << "  \"shared_memory_per_block_bytes\": " << prop.sharedMemPerBlock
    << ",\n"
    << "  \"registers_per_block\": " << prop.regsPerBlock << ",\n"
    << "  \"warp_size\": " << prop.warpSize << "\n"
    << "}\n";
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
  Config cfg;
  if (!parse_args(argc, argv, cfg)) {
    return 2;
  }

  // 环境信息（便于日志定位硬件）
  int device = 0;
  CUDA_CHECK(cudaGetDevice(&device));
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  if (!cfg.device_json.empty()) {
    write_device_json(cfg.device_json, device, prop);
  }

  const DataType dtype = (cfg.dtype == "fp16") ? DataType::FP16 : DataType::BF16;
  const long long total_tokens =
      static_cast<long long>(cfg.batch_size) * cfg.seq_len * cfg.num_heads;
  const long long n = total_tokens * cfg.head_dim;
  if (total_tokens > 2147483647LL) {
    std::fprintf(stderr, "[error] total_tokens 超出 baseline grid 支持范围\n");
    return 2;
  }

  std::printf(
      "[hadamard_bench] GPU: %s | config: dtype=%s batch=%d seq=%d heads=%d "
      "head_dim=%d normalize=%s dist=%s seed=%llu\n",
      prop.name, cfg.dtype.c_str(), cfg.batch_size, cfg.seq_len, cfg.num_heads,
      cfg.head_dim, cfg.normalize ? "true" : "false", cfg.dist.c_str(),
      static_cast<unsigned long long>(cfg.seed));
  std::printf("[hadamard_bench] total_tokens=%lld elements=%lld bytes_per_buffer=%lld\n",
              total_tokens, n, n * 2LL);

  // ---- 1. 生成输入（host），或从文件加载 ----
  HostTensors host = cfg.input_bin.empty() ? generate_input(cfg, total_tokens)
                                           : load_input_bin(cfg, total_tokens);

  // ---- 2. 设备内存 + H2D ----
  const size_t bytes = static_cast<size_t>(n) * sizeof(uint16_t);
  void* d_in = nullptr;
  void* d_out = nullptr;
  CUDA_CHECK(cudaMalloc(&d_in, bytes));
  CUDA_CHECK(cudaMalloc(&d_out, bytes));
  CUDA_CHECK(cudaMemcpy(d_in, host.native_input.data(), bytes, cudaMemcpyHostToDevice));

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  // 校验分发（dtype x head_dim）是否支持
  {
    int rc = launch_hadamard(d_in, d_out, cfg.batch_size, cfg.seq_len,
                             cfg.num_heads, cfg.head_dim, dtype, cfg.normalize,
                             stream);
    if (rc != 0) {
      std::fprintf(stderr, "[error] launch_hadamard 不支持的配置 (rc=%d)\n", rc);
      return 2;
    }
    CUDA_SYNC_CHECK();
  }

  // ---- 3. warmup ----
  for (int i = 0; i < cfg.warmup; ++i) {
    launch_hadamard(d_in, d_out, cfg.batch_size, cfg.seq_len, cfg.num_heads,
                    cfg.head_dim, dtype, cfg.normalize, stream);
  }
  CUDA_SYNC_CHECK();

  // ---- 4. CUDA Event 计时 ----
  cudaEvent_t ev_start, ev_stop;
  CUDA_CHECK(cudaEventCreate(&ev_start));
  CUDA_CHECK(cudaEventCreate(&ev_stop));
  double sum_ms = 0.0, min_ms = 1e300, max_ms = 0.0;
  for (int i = 0; i < cfg.iters; ++i) {
    CUDA_CHECK(cudaEventRecord(ev_start, stream));
    launch_hadamard(d_in, d_out, cfg.batch_size, cfg.seq_len, cfg.num_heads,
                    cfg.head_dim, dtype, cfg.normalize, stream);
    CUDA_CHECK(cudaEventRecord(ev_stop, stream));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
    sum_ms += ms;
    min_ms = std::min(min_ms, static_cast<double>(ms));
    max_ms = std::max(max_ms, static_cast<double>(ms));
  }
  const double avg_ms = sum_ms / cfg.iters;
  std::printf(
      "[hadamard_bench] kernel time: warmup=%d iters=%d avg=%.4f ms min=%.4f "
      "ms max=%.4f ms\n",
      cfg.warmup, cfg.iters, avg_ms, min_ms, max_ms);

  // ---- 5. 正确性检查 ----
  ErrorMetrics metrics;
  double threshold = (dtype == DataType::FP16) ? 1e-2 : 5e-2;
#if defined(TIANSHU_COREX)
  // 未归一化 Hadamard 输出的舍入误差随 sqrt(head_dim) 增长；保持绝对误差
  // 判定，但按该输出尺度调整 CoreX 的验收阈值。
  if (!cfg.normalize) {
    threshold *= std::sqrt(static_cast<double>(cfg.head_dim));
  }
#endif
  bool pass = true;
  std::vector<uint16_t> native_output;
  if (cfg.check) {
    native_output.resize(static_cast<size_t>(n));
    CUDA_CHECK(cudaMemcpy(native_output.data(), d_out, bytes,
                          cudaMemcpyDeviceToHost));

    // CPU 参考：对"低精度输入反量化后的 FP32"做同参数的快速 WHT
    std::vector<float> ref = host.input_fp32;
    wht_fp32_inplace(ref.data(), static_cast<size_t>(total_tokens),
                     cfg.head_dim, cfg.normalize);

    // GPU 输出转 FP32 后比较
    std::vector<float> gpu_fp32 = native_to_fp32(native_output, cfg.dtype);
    metrics = compute_error_metrics(gpu_fp32.data(), ref.data(),
                                    static_cast<size_t>(n));
    pass = metrics.max_abs_error < threshold;
    std::printf(
        "[hadamard_bench] check: max_abs_error=%.6e mean_abs_error=%.6e "
        "rmse=%.6e max_rel_error=%.6e threshold=%.1e -> %s\n",
        metrics.max_abs_error, metrics.mean_abs_error, metrics.rmse,
        metrics.max_rel_error, threshold, pass ? "PASS" : "FAIL");
  }

  // ---- 6. dump / CSV ----
  if (!cfg.dump_dir.empty()) {
    if (native_output.empty()) {  // --check false 时也要能 dump
      native_output.resize(static_cast<size_t>(n));
      CUDA_CHECK(cudaMemcpy(native_output.data(), d_out, bytes,
                            cudaMemcpyDeviceToHost));
    }
    std::vector<float> ref = host.input_fp32;
    wht_fp32_inplace(ref.data(), static_cast<size_t>(total_tokens),
                     cfg.head_dim, cfg.normalize);
    dump_binaries(cfg, total_tokens, host.native_input, native_output, ref);
  }
  if (!cfg.csv_path.empty()) {
    append_csv(cfg, total_tokens, cfg.warmup, cfg.iters, avg_ms, min_ms,
               max_ms, cfg.check, metrics, threshold, pass);
    std::printf("[hadamard_bench] csv appended to %s\n", cfg.csv_path.c_str());
  }

  // ---- 清理 ----
  CUDA_CHECK(cudaEventDestroy(ev_start));
  CUDA_CHECK(cudaEventDestroy(ev_stop));
  CUDA_CHECK(cudaStreamDestroy(stream));
  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_out));

  if (cfg.check && !pass) {
    return 1;
  }
  return 0;
}
