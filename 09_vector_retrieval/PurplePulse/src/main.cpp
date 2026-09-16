#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "file_io.h"
#include "search.h"

namespace {

void printUsage(const char* program) {
  std::cerr << "用法:\n  " << program
            << " --database <文件> --queries <文件> --params <文件>"
               " --backend <cpu|gpu> --output <文件>"
               " [--warmup 1] [--repeat 5]"
               " [--profile-breakdown 0|1] [--profile-repeat 5]\n";
}

std::unordered_map<std::string, std::string> parseArguments(int argc,
                                                            char** argv) {
  std::unordered_map<std::string, std::string> arguments;
  for (int index = 1; index < argc; index += 2) {
    if (index + 1 >= argc || std::string(argv[index]).rfind("--", 0) != 0) {
      throw std::runtime_error("命令行参数必须是 --名称 值 的形式");
    }
    arguments[argv[index]] = argv[index + 1];
  }
  for (const char* required :
       {"--database", "--queries", "--params", "--backend", "--output"}) {
    if (arguments.count(required) == 0) {
      throw std::runtime_error(std::string("缺少参数: ") + required);
    }
  }
  return arguments;
}

double percentile(std::vector<double> values, double fraction) {
  if (values.empty()) {
    return 0.0;
  }
  std::sort(values.begin(), values.end());
  const std::size_t index =
      static_cast<std::size_t>(
          std::ceil(fraction * static_cast<double>(values.size()))) -
      1;
  return values[index];
}

std::uint32_t readCount(
    const std::unordered_map<std::string, std::string>& arguments,
    const std::string& name, std::uint32_t default_value, bool allow_zero) {
  const auto found = arguments.find(name);
  if (found == arguments.end()) {
    return default_value;
  }
  std::size_t parsed_chars = 0;
  const unsigned long parsed = std::stoul(found->second, &parsed_chars);
  if (parsed_chars != found->second.size() || (!allow_zero && parsed == 0) ||
      parsed > UINT32_MAX) {
    throw std::runtime_error(name + " 的值不合法");
  }
  return static_cast<std::uint32_t>(parsed);
}

void addStats(SearchStats* total, const SearchStats& current) {
  total->host_selection_ms += current.host_selection_ms;
  total->center_selection_ms += current.center_selection_ms;
  total->database_h2d_ms += current.database_h2d_ms;
  total->query_h2d_ms += current.query_h2d_ms;
  total->distance_kernel_ms += current.distance_kernel_ms;
  total->topk_kernel_ms += current.topk_kernel_ms;
  total->result_d2h_ms += current.result_d2h_ms;
  total->batch_latency_ms.insert(total->batch_latency_ms.end(),
                                 current.batch_latency_ms.begin(),
                                 current.batch_latency_ms.end());
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc == 2 && std::string(argv[1]) == "--help") {
      printUsage(argv[0]);
      return EXIT_SUCCESS;
    }
    if (argc == 1) {
      printUsage(argv[0]);
      return 1;
    }
    const auto arguments = parseArguments(argc, argv);
    const VectorDatabase database =
        readVectorDatabase(arguments.at("--database"));
    const QuerySet queries = readQuerySet(arguments.at("--queries"));
    const SearchParams params = readSearchParams(arguments.at("--params"));
    validateInputs(database, queries, params);
    const std::uint32_t warmup = readCount(arguments, "--warmup", 0, true);
    const std::uint32_t repeat = readCount(arguments, "--repeat", 1, false);
    const bool profile_breakdown =
        readCount(arguments, "--profile-breakdown", 0, true) != 0;
    const std::uint32_t profile_repeat =
        readCount(arguments, "--profile-repeat", repeat, false);

    std::cout << "向量库: " << database.num_vectors << " x " << database.dim
              << "\n查询数: " << queries.num_queries
              << "\ndtype: " << dataTypeName(database.dtype)
              << "\nmetric: " << metricName(database.metric)
              << "\ntop_k: " << params.top_k
              << "\nbatch_size: " << params.batch_size
              << "\ndistance_mode: " << params.distance_mode
              << "\ntopk_mode: " << params.topk_mode
              << "\nexact_query_tile: " << params.exact_query_tile
              << "\nexact_fused_layout: " << params.exact_fused_layout
              << "\nexact_fused_blocks: " << params.exact_fused_blocks
              << "\nwarmup/repeat: " << warmup << '/' << repeat << '\n';

    SearchResults results;
    SearchStats stats;
    const std::string backend = arguments.at("--backend");
    if (backend != "cpu" && backend != "gpu") {
      throw std::runtime_error("backend 必须是 cpu 或 gpu");
    }
    if (profile_breakdown &&
        (backend != "gpu" || params.topk_mode != "fused")) {
      throw std::runtime_error(
          "--profile-breakdown 只支持 backend=gpu、topk_mode=fused");
    }

    std::unique_ptr<GpuExactSearchEngine> gpu_engine;
    double gpu_initialization_ms = 0.0;
    if (backend == "gpu") {
      const auto initialization_start = std::chrono::steady_clock::now();
      gpu_engine =
          std::make_unique<GpuExactSearchEngine>(database, queries, params);
      const auto initialization_end = std::chrono::steady_clock::now();
      gpu_initialization_ms = std::chrono::duration<double, std::milli>(
                                  initialization_end - initialization_start)
                                  .count();
    }

    const auto run_search = [&](SearchStats* run_stats) {
      if (backend == "cpu") {
        return cpuExactSearch(database, queries, params);
      }
      return gpu_engine->search(queries, run_stats);
    };

    for (std::uint32_t run = 0; run < warmup; ++run) {
      SearchStats ignored_stats;
      results = run_search(&ignored_stats);
    }

    std::vector<double> run_times_ms;
    for (std::uint32_t run = 0; run < repeat; ++run) {
      SearchStats run_stats;
      const auto start = std::chrono::steady_clock::now();
      results = run_search(&run_stats);
      const auto end = std::chrono::steady_clock::now();
      run_times_ms.push_back(
          std::chrono::duration<double, std::milli>(end - start).count());
      addStats(&stats, run_stats);
    }

    const std::filesystem::path output_path(arguments.at("--output"));
    if (output_path.has_parent_path()) {
      std::filesystem::create_directories(output_path.parent_path());
    }
    writeSearchResults(output_path.string(), results);

    double total_ms = 0.0;
    for (double time_ms : run_times_ms) {
      total_ms += time_ms;
    }
    const double average_ms = total_ms / repeat;
    const double qps =
        static_cast<double>(queries.num_queries) * repeat / (total_ms / 1000.0);
    std::cout << "backend: " << backend;
    if (backend == "gpu") {
      std::cout << "\nGPU 一次性初始化总时间: " << gpu_initialization_ms
                << " ms\n数据库一次性 H2D: "
                << gpu_engine->databaseH2DMilliseconds() << " ms\nGPU 缓冲区: "
                << gpu_engine->deviceBytes() / (1024.0 * 1024.0) << " MiB"
                << "\n估算冷启动端到端: " << gpu_initialization_ms + average_ms
                << " ms\n平均常驻查询时间: " << average_ms;
    } else {
      std::cout << "\n平均单次总时间: " << average_ms;
    }
    std::cout << " ms\nrun P50: " << percentile(run_times_ms, 0.50)
              << " ms\nrun P99: " << percentile(run_times_ms, 0.99) << " ms\n"
              << "run samples: " << run_times_ms.size() << '\n'
              << (backend == "gpu" ? "常驻 QPS: " : "QPS: ") << qps
              << "\n结果: " << output_path << '\n';
    if (backend == "gpu") {
      double distance_only_ms = 0.0;
      if (profile_breakdown) {
        distance_only_ms = gpu_engine->profileDistanceOnlyMilliseconds(
            queries, 1, profile_repeat);
      }
      std::cout
          << "平均查询 H2D: " << stats.query_h2d_ms / repeat << " ms\n"
          << (params.topk_mode == "fused" ? "平均融合距离+局部 Top-K kernel: "
                                          : "平均距离 kernel: ")
          << stats.distance_kernel_ms / repeat
          << " ms\n平均最终 Top-K kernel: " << stats.topk_kernel_ms / repeat
          << " ms\n平均结果 D2H: " << stats.result_d2h_ms / repeat
          << " ms\nbatch P50: " << percentile(stats.batch_latency_ms, 0.50)
          << " ms\nbatch P99: " << percentile(stats.batch_latency_ms, 0.99)
          << " ms\nbatch samples: " << stats.batch_latency_ms.size() << '\n';
      if (profile_breakdown) {
        const double fused_ms = stats.distance_kernel_ms / repeat;
        const double final_merge_ms = stats.topk_kernel_ms / repeat;
        const double local_topk_estimated_ms =
            std::max(0.0, fused_ms - distance_only_ms);
        std::cout
            << "剖析方法: distance-only differential\n"
            << "剖析重复次数: " << profile_repeat << '\n'
            << "剖析距离-only kernel: " << distance_only_ms << " ms\n"
            << "剖析局部 Top-K 估算: " << local_topk_estimated_ms << " ms\n"
            << "剖析融合 kernel: " << fused_ms << " ms\n"
            << "剖析最终归并 kernel: " << final_merge_ms << " ms\n";
      }
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "错误: " << error.what() << '\n';
    printUsage(argv[0]);
    return 1;
  }
}
