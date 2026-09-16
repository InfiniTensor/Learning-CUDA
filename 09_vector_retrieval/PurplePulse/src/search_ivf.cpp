#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "file_io.h"
#include "ivf_flat.h"

namespace {

void printUsage(const char* program) {
  std::cerr << "用法:\n  " << program
            << " --index <文件> --queries <文件> --params <文件>"
               " --output <文件> [--nlist N] [--nprobe N]"
               " [--nprobe-policy fixed|score_mass]"
               " [--adaptive-nprobe-min N] [--adaptive-nprobe-step N]"
               " [--adaptive-target-mass F] [--adaptive-temperature F]"
               " [--memory-metadata <文件>]"
               " [--memory-mode disabled|fused|rerank]"
               " [--filter-session-id N|any] [--filter-source-type N|any]"
               " [--backend cpu|gpu] [--warmup 1] [--repeat 5]\n";
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
       {"--index", "--queries", "--params", "--output"}) {
    if (arguments.count(required) == 0) {
      throw std::runtime_error(std::string("缺少参数: ") + required);
    }
  }
  return arguments;
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

float readPositiveFloat(
    const std::unordered_map<std::string, std::string>& arguments,
    const std::string& name, float default_value) {
  const auto found = arguments.find(name);
  if (found == arguments.end()) {
    return default_value;
  }
  std::size_t parsed_chars = 0;
  const float parsed = std::stof(found->second, &parsed_chars);
  if (parsed_chars != found->second.size() || !std::isfinite(parsed) ||
      parsed <= 0.0F) {
    throw std::runtime_error(name + " 的值必须是有限正数");
  }
  return parsed;
}

float readNonnegativeFloat(
    const std::unordered_map<std::string, std::string>& arguments,
    const std::string& name, float default_value) {
  const auto found = arguments.find(name);
  if (found == arguments.end()) {
    return default_value;
  }
  std::size_t parsed_chars = 0;
  const float parsed = std::stof(found->second, &parsed_chars);
  if (parsed_chars != found->second.size() || !std::isfinite(parsed) ||
      parsed < 0.0F) {
    throw std::runtime_error(name + " 的值必须是有限非负数");
  }
  return parsed;
}

std::uint64_t readUint64(
    const std::unordered_map<std::string, std::string>& arguments,
    const std::string& name, std::uint64_t default_value) {
  const auto found = arguments.find(name);
  if (found == arguments.end()) {
    return default_value;
  }
  if (found->second.empty() || found->second.front() == '-') {
    throw std::runtime_error(name + " 的值必须是非负整数");
  }
  std::size_t parsed_chars = 0;
  const unsigned long long parsed = std::stoull(found->second, &parsed_chars);
  if (parsed_chars != found->second.size()) {
    throw std::runtime_error(name + " 的值必须是非负整数");
  }
  return static_cast<std::uint64_t>(parsed);
}

std::uint32_t readOptionalId(
    const std::unordered_map<std::string, std::string>& arguments,
    const std::string& name, std::uint32_t default_value) {
  const auto found = arguments.find(name);
  if (found == arguments.end()) {
    return default_value;
  }
  if (found->second == "any") {
    return UINT32_MAX;
  }
  const std::uint64_t parsed = readUint64(arguments, name, default_value);
  if (parsed >= UINT32_MAX) {
    throw std::runtime_error(name + " 的值必须小于 UINT32_MAX");
  }
  return static_cast<std::uint32_t>(parsed);
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

void addStats(SearchStats* total, const SearchStats& current) {
  total->host_selection_ms += current.host_selection_ms;
  total->center_selection_ms += current.center_selection_ms;
  total->adaptive_policy_ms += current.adaptive_policy_ms;
  total->database_h2d_ms += current.database_h2d_ms;
  total->query_h2d_ms += current.query_h2d_ms;
  total->distance_kernel_ms += current.distance_kernel_ms;
  total->topk_kernel_ms += current.topk_kernel_ms;
  total->memory_rerank_ms += current.memory_rerank_ms;
  total->result_d2h_ms += current.result_d2h_ms;
  total->batch_latency_ms.insert(total->batch_latency_ms.end(),
                                 current.batch_latency_ms.begin(),
                                 current.batch_latency_ms.end());
  if (current.selected_probe_queries != 0) {
    if (total->selected_probe_queries == 0) {
      total->selected_probe_min = current.selected_probe_min;
    } else {
      total->selected_probe_min =
          std::min(total->selected_probe_min, current.selected_probe_min);
    }
    total->selected_probe_max =
        std::max(total->selected_probe_max, current.selected_probe_max);
    total->selected_probe_sum += current.selected_probe_sum;
    total->selected_probe_queries += current.selected_probe_queries;
    if (total->selected_probe_counts.empty()) {
      total->selected_probe_counts = current.selected_probe_counts;
    }
  }
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
      return EXIT_FAILURE;
    }
    const auto arguments = parseArguments(argc, argv);
    const auto load_start = std::chrono::steady_clock::now();
    const IvfFlatIndex index = readIvfFlatIndex(arguments.at("--index"));
    const auto load_end = std::chrono::steady_clock::now();
    const QuerySet queries = readQuerySet(arguments.at("--queries"));
    SearchParams params = readSearchParams(arguments.at("--params"));
    if (params.search_mode != "ivf_flat") {
      throw std::runtime_error("IVF 查询要求 search_mode=ivf_flat");
    }
    if (arguments.count("--nlist") != 0) {
      params.nlist = readCount(arguments, "--nlist", params.nlist, false);
    }
    if (arguments.count("--nprobe") != 0) {
      params.nprobe = readCount(arguments, "--nprobe", params.nprobe, false);
    }
    if (arguments.count("--nprobe-policy") != 0) {
      params.nprobe_policy = arguments.at("--nprobe-policy");
    }
    if (arguments.count("--adaptive-execution") != 0) {
      params.adaptive_execution = arguments.at("--adaptive-execution");
    }
    params.adaptive_nprobe_min = readCount(arguments, "--adaptive-nprobe-min",
                                           params.adaptive_nprobe_min, false);
    params.adaptive_nprobe_step = readCount(arguments, "--adaptive-nprobe-step",
                                            params.adaptive_nprobe_step, false);
    params.adaptive_target_mass = readPositiveFloat(
        arguments, "--adaptive-target-mass", params.adaptive_target_mass);
    params.adaptive_temperature = readPositiveFloat(
        arguments, "--adaptive-temperature", params.adaptive_temperature);
    if (arguments.count("--memory-mode") != 0) {
      params.memory_mode = arguments.at("--memory-mode");
    }
    params.memory_semantic_weight = readPositiveFloat(
        arguments, "--memory-semantic-weight", params.memory_semantic_weight);
    params.memory_importance_weight =
        readNonnegativeFloat(arguments, "--memory-importance-weight",
                             params.memory_importance_weight);
    params.memory_recency_weight = readNonnegativeFloat(
        arguments, "--memory-recency-weight", params.memory_recency_weight);
    params.memory_time_scale = readPositiveFloat(
        arguments, "--memory-time-scale", params.memory_time_scale);
    params.memory_now =
        readUint64(arguments, "--memory-now", params.memory_now);
    params.filter_min_timestamp = readUint64(
        arguments, "--filter-min-timestamp", params.filter_min_timestamp);
    params.filter_session_id = readOptionalId(arguments, "--filter-session-id",
                                              params.filter_session_id);
    params.filter_source_type = readOptionalId(
        arguments, "--filter-source-type", params.filter_source_type);
    params.memory_rerank_factor = readCount(arguments, "--memory-rerank-factor",
                                            params.memory_rerank_factor, false);
    if (params.nlist != index.nlist) {
      throw std::runtime_error("参数 nlist 与索引 nlist 不一致");
    }
    const std::uint32_t warmup = readCount(arguments, "--warmup", 0, true);
    const std::uint32_t repeat = readCount(arguments, "--repeat", 1, false);
    const std::string backend =
        arguments.count("--backend") == 0 ? "cpu" : arguments.at("--backend");
    if (backend != "cpu" && backend != "gpu") {
      throw std::runtime_error("backend 必须是 cpu 或 gpu");
    }
    if (backend == "cpu" && params.nprobe_policy != "fixed") {
      throw std::runtime_error("自适应 nprobe 当前只支持 GPU backend");
    }
    MemoryMetadata memory_metadata;
    const MemoryMetadata* metadata = nullptr;
    const auto metadata_argument = arguments.find("--memory-metadata");
    if (metadata_argument != arguments.end()) {
      memory_metadata = readMemoryMetadata(metadata_argument->second);
      validateMemoryMetadata(memory_metadata, index.num_vectors);
      metadata = &memory_metadata;
    }
    if (params.memory_mode != "disabled" && metadata == nullptr) {
      throw std::runtime_error("memory_mode 启用时必须传入 --memory-metadata");
    }

    std::unique_ptr<GpuIvfFlatSearchEngine> gpu_engine;
    double gpu_initialization_ms = 0.0;
    if (backend == "gpu") {
      const auto initialization_start = std::chrono::steady_clock::now();
      gpu_engine = std::make_unique<GpuIvfFlatSearchEngine>(index, queries,
                                                            params, metadata);
      const auto initialization_end = std::chrono::steady_clock::now();
      gpu_initialization_ms = std::chrono::duration<double, std::milli>(
                                  initialization_end - initialization_start)
                                  .count();
    }

    SearchResults results;
    const auto run_search = [&](SearchStats* stats) {
      return backend == "gpu"
                 ? gpu_engine->search(queries, stats)
                 : cpuIvfFlatSearch(index, queries, params, metadata);
    };
    for (std::uint32_t run = 0; run < warmup; ++run) {
      SearchStats ignored;
      results = run_search(&ignored);
    }
    std::vector<double> run_times;
    SearchStats stats;
    for (std::uint32_t run = 0; run < repeat; ++run) {
      SearchStats run_stats;
      const auto start = std::chrono::steady_clock::now();
      results = run_search(&run_stats);
      const auto end = std::chrono::steady_clock::now();
      run_times.push_back(
          std::chrono::duration<double, std::milli>(end - start).count());
      addStats(&stats, run_stats);
    }
    const std::filesystem::path output_path(arguments.at("--output"));
    if (output_path.has_parent_path()) {
      std::filesystem::create_directories(output_path.parent_path());
    }
    writeSearchResults(output_path.string(), results);
    double total_ms = 0.0;
    for (const double time : run_times) {
      total_ms += time;
    }
    const double average_ms = total_ms / repeat;
    const double load_ms =
        std::chrono::duration<double, std::milli>(load_end - load_start)
            .count();
    std::cout << "IVF-Flat " << backend << " 查询完成"
              << "\n索引: " << index.num_vectors << " x " << index.dim
              << "\ndtype/metric: " << dataTypeName(index.dtype) << '/'
              << metricName(index.metric)
              << "\nnlist/nprobe/top_k: " << index.nlist << '/' << params.nprobe
              << '/' << params.top_k
              << "\nnprobe policy: " << params.nprobe_policy
              << "\nmemory mode: " << params.memory_mode
              << "\n索引加载: " << load_ms << " ms"
              << (backend == "gpu" ? "\nGPU 初始化: " : "")
              << (backend == "gpu" ? std::to_string(gpu_initialization_ms)
                                   : std::string())
              << (backend == "gpu" ? " ms\n索引 H2D: " : "")
              << (backend == "gpu"
                      ? std::to_string(gpu_engine->indexH2DMilliseconds())
                      : std::string())
              << (backend == "gpu" ? " ms" : "")
              << (backend == "gpu" ? "\nGPU 缓冲区: " : "")
              << (backend == "gpu" ? std::to_string(gpu_engine->deviceBytes() /
                                                    (1024.0 * 1024.0))
                                   : std::string())
              << (backend == "gpu" ? " MiB" : "")
              << "\n平均查询: " << average_ms << " ms"
              << "\nrun P50/P99: " << percentile(run_times, 0.50) << '/'
              << percentile(run_times, 0.99) << " ms"
              << "\nrun samples: " << run_times.size()
              << "\nQPS: " << queries.num_queries * repeat / (total_ms / 1000.0)
              << "\n输出: " << output_path << '\n';
    if (params.nprobe_policy == "score_mass") {
      std::cout << "自适应 min/step/target/temperature: "
                << params.adaptive_nprobe_min << '/'
                << params.adaptive_nprobe_step << '/'
                << params.adaptive_target_mass << '/'
                << params.adaptive_temperature << '\n';
      std::cout << "自适应执行方式: " << params.adaptive_execution << '\n';
    }
    if (backend == "gpu") {
      std::cout
          << "平均 CPU 中心/桶选择: " << stats.host_selection_ms / repeat
          << " ms\n平均 query H2D: " << stats.query_h2d_ms / repeat
          << " ms\n平均 GPU 中心选择: " << stats.center_selection_ms / repeat
          << " ms\n平均自适应分层开销: " << stats.adaptive_policy_ms / repeat
          << " ms\n平均 GPU 桶扫描+局部 Top-K: "
          << stats.distance_kernel_ms / repeat
          << " ms\n平均 GPU 最终 Top-K 归并: " << stats.topk_kernel_ms / repeat
          << " ms\n平均记忆独立重排: " << stats.memory_rerank_ms / repeat
          << " ms\n平均结果 D2H: " << stats.result_d2h_ms / repeat
          << " ms\nbatch P50/P99: " << percentile(stats.batch_latency_ms, 0.50)
          << '/' << percentile(stats.batch_latency_ms, 0.99)
          << " ms\nbatch samples: " << stats.batch_latency_ms.size() << '\n';
      if (stats.selected_probe_queries != 0) {
        std::cout << "实际 nprobe min/avg/max: " << stats.selected_probe_min
                  << '/'
                  << static_cast<double>(stats.selected_probe_sum) /
                         stats.selected_probe_queries
                  << '/' << stats.selected_probe_max << '\n';
      }
      const auto probe_counts_argument =
          arguments.find("--probe-counts-output");
      if (probe_counts_argument != arguments.end()) {
        const std::filesystem::path counts_path(probe_counts_argument->second);
        if (counts_path.has_parent_path()) {
          std::filesystem::create_directories(counts_path.parent_path());
        }
        std::ofstream counts_output(counts_path);
        if (!counts_output) {
          throw std::runtime_error("无法写入实际 nprobe 文件: " +
                                   counts_path.string());
        }
        for (std::size_t query_id = 0;
             query_id < stats.selected_probe_counts.size(); ++query_id) {
          counts_output << query_id << ' '
                        << stats.selected_probe_counts[query_id] << '\n';
        }
      }
    }
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    std::cerr << "错误: " << error.what() << '\n';
    printUsage(argv[0]);
    return EXIT_FAILURE;
  }
}
