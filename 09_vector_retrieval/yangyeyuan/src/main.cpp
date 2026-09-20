#include "vector_search.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace {

// 命令行参数表: "--key value" 或裸开关 "--flag"(值为 "true")。
using Options = std::unordered_map<std::string, std::string>;

Options parse_options(int argc, char** argv, int begin) {
  Options options;
  for (int i = begin; i < argc; ++i) {
    const std::string key = argv[i];
    if (key.rfind("--", 0) != 0) throw std::runtime_error("expected option, got: " + key);
    if (i + 1 >= argc || std::string(argv[i + 1]).rfind("--", 0) == 0) {
      // 下一个参数是另一个选项或没有值 → 当作布尔开关
      options[key] = "true";
    } else {
      options[key] = argv[++i];
    }
  }
  return options;
}

// 读取必填选项,缺失即报错。
const std::string& required(const Options& options, const std::string& key) {
  const auto found = options.find(key);
  if (found == options.end()) throw std::runtime_error("missing required option: " + key);
  return found->second;
}

// 读取可选选项,缺失时用 fallback。
std::string optional(const Options& options, const std::string& key,
                     const std::string& fallback = {}) {
  const auto found = options.find(key);
  return found == options.end() ? fallback : found->second;
}

// 读取无符号整数选项,严格拒绝空串/负数/带尾字符的输入。
std::uint64_t unsigned_option(const Options& options, const std::string& key,
                              std::uint64_t fallback = 0, bool is_required = false) {
  const auto found = options.find(key);
  if (found == options.end()) {
    if (is_required) throw std::runtime_error("missing required option: " + key);
    return fallback;
  }
  if (found->second.empty() || found->second.front() == '-') {
    throw std::runtime_error("invalid unsigned integer: " + key);
  }
  std::size_t consumed = 0;
  const auto value = std::stoull(found->second, &consumed);
  if (consumed != found->second.size()) throw std::runtime_error("invalid integer: " + key);
  return value;
}

vs::DType parse_dtype(const std::string& value) {
  if (value == "fp32") return vs::DType::kFp32;
  if (value == "fp16") return vs::DType::kFp16;
  throw std::runtime_error("dtype must be fp32 or fp16");
}

// 生成有聚类结构的测试数据,让 IVF 的 k-means 有据可依。
// 流程:先撒 clusters 个高斯"簇中心",每个向量由"随机簇中心 + 噪声"生成。
// database 参数非空时表示生成查询:从库里随机抽向量再加更小噪声,使真值有区分度。
vs::VectorDataset make_clustered(std::uint64_t count, std::uint32_t dim,
                                 std::uint32_t clusters, std::uint64_t seed,
                                 const vs::VectorDataset* database = nullptr) {
  if (count == 0 || dim == 0 || clusters == 0) throw std::runtime_error("invalid generated shape");
  std::mt19937_64 random(seed);
  std::normal_distribution<float> center_distribution(0.0f, 3.0f);
  std::normal_distribution<float> noise(0.0f, database ? 0.02f : 0.25f);
  vs::VectorDataset data;
  data.count = count;
  data.dim = dim;
  data.metric = vs::Metric::kL2;
  data.values.resize(static_cast<std::size_t>(count) * dim);
  if (database) {
    if (database->dim != dim || database->count == 0) throw std::runtime_error("invalid query source");
    std::uniform_int_distribution<std::uint64_t> pick(0, database->count - 1);
    for (std::uint64_t i = 0; i < count; ++i) {
      const std::uint64_t source = pick(random);
      for (std::uint32_t d = 0; d < dim; ++d) {
        data.values[static_cast<std::size_t>(i) * dim + d] =
            database->values[static_cast<std::size_t>(source) * dim + d] + noise(random);
      }
    }
    return data;
  }
  // 生成向量库:随机簇中心 + 噪声
  std::vector<float> centers(static_cast<std::size_t>(clusters) * dim);
  for (float& value : centers) value = center_distribution(random);
  std::uniform_int_distribution<std::uint32_t> choose_cluster(0, clusters - 1);
  for (std::uint64_t i = 0; i < count; ++i) {
    const std::uint32_t cluster = choose_cluster(random);
    for (std::uint32_t d = 0; d < dim; ++d) {
      data.values[static_cast<std::size_t>(i) * dim + d] =
          centers[static_cast<std::size_t>(cluster) * dim + d] + noise(random);
    }
  }
  return data;
}

// 打印一行性能摘要。
void print_metrics(const vs::SearchMetrics& metrics) {
  std::cout << std::fixed << std::setprecision(3)
            << "total_ms=" << metrics.total_ms << " gpu_ms=" << metrics.gpu_ms
            << " qps=" << metrics.qps << " p50_batch_ms=" << metrics.p50_ms
            << " p99_batch_ms=" << metrics.p99_ms
            << " peak_device_bytes=" << metrics.peak_device_bytes;
  if (metrics.cpu_baseline_ms > 0.0) {
    std::cout << " cpu_ms=" << metrics.cpu_baseline_ms
              << " speedup=" << metrics.speedup_vs_cpu;
  }
  std::cout << '\n';
}

// 正确性核对:GPU 结果必须与 CPU 参考一致。
// ID 要求完全一致;score 允许浮点相对容差。
void verify_same(const vs::SearchResult& actual, const vs::SearchResult& expected,
                 float tolerance, const std::string& label) {
  if (actual.num_queries != expected.num_queries || actual.top_k != expected.top_k) {
    throw std::runtime_error(label + ": result shape mismatch");
  }
  for (std::size_t i = 0; i < actual.ids.size(); ++i) {
    if (actual.ids[i] != expected.ids[i]) {
      throw std::runtime_error(label + ": id mismatch at " + std::to_string(i));
    }
    const float scale = std::max(1.0f, std::abs(expected.scores[i]));
    if (std::abs(actual.scores[i] - expected.scores[i]) > tolerance * scale) {
      throw std::runtime_error(label + ": score mismatch at " + std::to_string(i));
    }
  }
}

// 冒烟测试:跑一遍核心路径,失败即抛异常。覆盖:
//  - CPU Exact vs GPU Exact(K=1/10/50/100)
//  - FP16 文件保存/加载往返
//  - IVF 构建/保存/加载,以及 nprobe=nlist 与 Exact 等价
void self_test() {
  std::string reason;
  if (!vs::cuda_available(&reason)) throw std::runtime_error("CUDA unavailable: " + reason);
  const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
  const std::filesystem::path temporary =
      std::filesystem::temp_directory_path() / ("vector_search_self_test_" + std::to_string(stamp));
  std::filesystem::create_directories(temporary);
  try {
    // 生成 513 个向量(非整 block 的 N)和 17 个查询,验证尾部处理。
    auto database = make_clustered(513, 32, 8, 11);
    auto queries = make_clustered(17, 32, 8, 12, &database);
    // 把前几个数据库向量原样拷进查询,保证查询有"标准答案"。
    std::copy_n(database.values.data(), database.dim,
                database.values.data() + database.dim);
    std::copy_n(database.values.data(), database.dim, queries.values.data());
    vs::FlatGpuIndex exact_index(database);
    vs::SearchResult exact;
    for (const std::uint32_t k : {1u, 10u, 50u, 100u}) {
      const auto cpu = vs::cpu_exact_search(database, queries, k);
      const auto gpu = exact_index.search(queries, k, 7);
      verify_same(gpu, cpu, 2e-5f, "GPU exact vs CPU K=" + std::to_string(k));
      if (k == 10) exact = gpu;
    }

    // FP16 向量库 + FP32 查询文件的往返验证。
    const auto vectors_path = (temporary / "vectors.fp16.bin").string();
    const auto queries_path = (temporary / "queries.fp32.bin").string();
    vs::save_vectors(vectors_path, database, false, vs::DType::kFp16);
    vs::save_vectors(queries_path, queries, true, vs::DType::kFp32);
    const auto loaded_vectors = vs::load_vectors(vectors_path, false);
    const auto loaded_queries = vs::load_vectors(queries_path, true);
    if (loaded_vectors.count != database.count || loaded_vectors.dim != database.dim ||
        loaded_queries.count != queries.count) {
      throw std::runtime_error("vector file round-trip failed");
    }

    // FP16 转换的边界值:subnormal、Inf、normal、最大有限值。
    vs::VectorDataset half_edges;
    half_edges.count = 5;
    half_edges.dim = 1;
    half_edges.values = {std::ldexp(1.0f, -24), -std::ldexp(1.0f, -24),
                         std::ldexp(1.0f, -14), 1.0f, 65504.0f};
    const auto half_edges_path = (temporary / "half_edges.bin").string();
    vs::save_vectors(half_edges_path, half_edges, false, vs::DType::kFp16);
    const auto loaded_half_edges = vs::load_vectors(half_edges_path, false);
    if (loaded_half_edges.values != half_edges.values) {
      throw std::runtime_error("FP16 edge-value round-trip failed");
    }

    // 用 FP16 读入的库做一次 GPU Exact 对照。
    vs::FlatGpuIndex fp16_index(loaded_vectors);
    verify_same(fp16_index.search(loaded_queries, 10, 7),
                vs::cpu_exact_search(loaded_vectors, loaded_queries, 10),
                2e-5f, "FP16 file GPU exact vs CPU");

    // IVF 构建/保存/加载,再验证 nprobe=nlist 时与 Exact 逐 ID 等价。
    vs::SearchConfig config;
    config.nlist = 8;
    config.nprobe = 8;
    config.batch_size = 7;
    config.kmeans_iters = 4;
    config.train_size = 513;
    const auto built = vs::build_ivf_flat(database, config);
    const auto index_path = (temporary / "test.ivf").string();
    vs::save_index(index_path, built);
    const auto loaded = vs::load_index(index_path);
    vs::IvfGpuIndex ivf(loaded);
    const auto all_lists = ivf.search(queries, 10, config.nlist, config.batch_size);
    verify_same(all_lists, exact, 1e-6f, "IVF nprobe=nlist vs exact");
    // 缩小 nprobe 做近似搜索,并检查 recall 是否在合理范围。
    const auto approximate = ivf.search(queries, 10, 4, config.batch_size);
    const auto quality = vs::measure_quality(approximate, exact);
    if (quality.recall_at_k < 0.5 || !std::isfinite(quality.mean_distance_error)) {
      throw std::runtime_error("unexpected IVF quality");
    }
    std::filesystem::remove_all(temporary);
    std::cout << "self-test passed: CPU exact, GPU exact, fp16 I/O, IVF save/load, recall="
              << quality.recall_at_k << '\n';
  } catch (...) {
    std::filesystem::remove_all(temporary);
    throw;
  }
}

// CLI 使用说明。
void usage() {
  std::cout <<
      "Usage:\n"
      "  vector_search_cli generate --vectors DB --queries Q --num-vectors N --num-queries QN --dim D [--dtype fp32|fp16] [--seed N]\n"
      "  vector_search_cli build --vectors DB --index INDEX --config CONFIG [--performance JSON]\n"
      "  vector_search_cli search --queries Q --config CONFIG --output RESULT (--vectors DB | --index INDEX) [--performance JSON] [--ground-truth RESULT --quality JSON] [--cpu-baseline]\n"
      "  vector_search_cli cpu --vectors DB --queries Q --config CONFIG --output RESULT [--performance JSON]\n"
      "  vector_search_cli self-test\n";
}

}  // namespace

// 命令分发入口:第一个参数是子命令,其余是 --key value 选项。
int main(int argc, char** argv) {
  try {
    if (argc < 2 || std::string(argv[1]) == "--help" || std::string(argv[1]) == "help") {
      usage();
      return argc < 2 ? 1 : 0;
    }
    const std::string command = argv[1];
    const Options options = parse_options(argc, argv, 2);

    if (command == "self-test") {
      self_test();
      return 0;
    }

    if (command == "generate") {
      // 生成有聚类结构的向量库和查询集,各自存为二进制文件。
      const auto count = unsigned_option(options, "--num-vectors", 0, true);
      const auto query_count = unsigned_option(options, "--num-queries", 0, true);
      const auto dim64 = unsigned_option(options, "--dim", 0, true);
      if (dim64 > std::numeric_limits<std::uint32_t>::max()) throw std::runtime_error("dim too large");
      const auto dim = static_cast<std::uint32_t>(dim64);
      const auto seed = unsigned_option(options, "--seed", 2026);
      const auto dtype = parse_dtype(optional(options, "--dtype", "fp32"));
      const auto clusters = static_cast<std::uint32_t>(std::min<std::uint64_t>(64, count));
      auto database = make_clustered(count, dim, clusters, seed);
      auto queries = make_clustered(query_count, dim, clusters, seed + 1, &database);
      vs::save_vectors(required(options, "--vectors"), database, false, dtype);
      vs::save_vectors(required(options, "--queries"), queries, true, dtype);
      std::cout << "generated vectors=" << count << " queries=" << query_count
                << " dim=" << dim << '\n';
      return 0;
    }

    const auto config = vs::load_config(required(options, "--config"));

    if (command == "build") {
      // 离线建 IVF 索引:加载向量库 → k-means 聚类 → 存成 .ivf。
      const auto database = vs::load_vectors(required(options, "--vectors"), false);
      vs::SearchMetrics metrics;
      const auto index = vs::build_ivf_flat(database, config, &metrics);
      vs::save_index(required(options, "--index"), index);
      const auto performance = optional(options, "--performance");
      if (!performance.empty()) vs::save_performance_log(performance, config, metrics);
      std::cout << "built IVF-Flat: vectors=" << index.count << " dim=" << index.dim
                << " nlist=" << index.nlist << " build_ms=" << metrics.build_ms << '\n';
      return 0;
    }

    if (command == "cpu") {
      // 纯 CPU 参考搜索,通常用于生成 ground truth。
      const auto database = vs::load_vectors(required(options, "--vectors"), false);
      const auto queries = vs::load_vectors(required(options, "--queries"), true);
      vs::SearchMetrics metrics;
      const auto result = vs::cpu_exact_search(database, queries, config.top_k, &metrics);
      vs::save_results(required(options, "--output"), result);
      const auto performance = optional(options, "--performance");
      if (!performance.empty()) vs::save_performance_log(performance, config, metrics);
      print_metrics(metrics);
      return 0;
    }

    if (command == "search") {
      // 在线检索:按 search_mode 分流到 Exact 或 IVF。
      const auto queries = vs::load_vectors(required(options, "--queries"), true);
      vs::SearchMetrics metrics;
      vs::SearchResult result;
      if (config.search_mode == "exact") {
        const auto database = vs::load_vectors(required(options, "--vectors"), false);
        vs::FlatGpuIndex index(database);
        result = index.search(queries, config.top_k, config.batch_size, &metrics);
        // --cpu-baseline:同数据跑一遍 CPU,逐 ID 核对,并统计加速比。
        if (options.count("--cpu-baseline")) {
          vs::SearchMetrics cpu_metrics;
          const auto cpu = vs::cpu_exact_search(database, queries, config.top_k, &cpu_metrics);
          verify_same(result, cpu, 2e-5f, "GPU exact vs CPU");
          metrics.cpu_baseline_ms = cpu_metrics.total_ms;
          metrics.speedup_vs_cpu = cpu_metrics.total_ms / metrics.total_ms;
        }
      } else {
        // IVF:加载建好的索引,传入 nprobe/batch_size 做近似搜索。
        const auto host_index = vs::load_index(required(options, "--index"));
        vs::IvfGpuIndex index(host_index);
        result = index.search(queries, config.top_k, config.nprobe, config.batch_size, &metrics);
      }
      vs::save_results(required(options, "--output"), result);
      const auto performance = optional(options, "--performance");
      if (!performance.empty()) vs::save_performance_log(performance, config, metrics);
      // --ground-truth:与 Exact 结果对比,输出 recall 等质量指标。
      const auto ground_truth_path = optional(options, "--ground-truth");
      if (!ground_truth_path.empty()) {
        const auto quality = vs::measure_quality(result, vs::load_results(ground_truth_path));
        const auto quality_path = required(options, "--quality");
        vs::save_quality_log(quality_path, config, quality);
        std::cout << "recall@" << config.top_k << '=' << quality.recall_at_k
                  << " mean_distance_error=" << quality.mean_distance_error << '\n';
      }
      print_metrics(metrics);
      return 0;
    }
    throw std::runtime_error("unknown command: " + command);
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
}
