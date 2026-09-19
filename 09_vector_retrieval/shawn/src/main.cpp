#include "vector_search/search.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Arguments {
  std::string command;
  std::string database_path;
  std::string query_path;
  std::string config_path;
  std::string index_path;
  std::string output_path = "results.txt";
  std::string performance_path = "performance.log";
  std::string quality_path = "quality.log";
  std::string backend = "auto";
};

void usage(const char* program) {
  std::cout
      << "Usage:\n  " << program
      << " build --database DB --config CFG --index INDEX"
         " [--backend auto|cpu|cuda] [--performance FILE]\n  "
      << program
      << " search --database DB --queries Q --config CFG [--index INDEX]"
         " [--backend auto|cpu|cuda] [--output FILE] [--performance FILE]"
         " [--quality FILE]\n";
}

Arguments parse_arguments(int argc, char** argv) {
  if (argc < 2) throw std::runtime_error("missing build/search command");
  Arguments arguments;
  arguments.command = argv[1];
  if (arguments.command != "build" && arguments.command != "search") {
    if (arguments.command == "--help" || arguments.command == "-h") {
      usage(argv[0]);
      std::exit(0);
    }
    throw std::runtime_error("first argument must be build or search");
  }
  for (int index = 2; index < argc; ++index) {
    const std::string flag = argv[index];
    if (flag == "--help" || flag == "-h") {
      usage(argv[0]);
      std::exit(0);
    }
    if (index + 1 >= argc) throw std::runtime_error("missing value after " + flag);
    const std::string value = argv[++index];
    if (flag == "--database") {
      arguments.database_path = value;
    } else if (flag == "--queries") {
      arguments.query_path = value;
    } else if (flag == "--config") {
      arguments.config_path = value;
    } else if (flag == "--index") {
      arguments.index_path = value;
    } else if (flag == "--output") {
      arguments.output_path = value;
    } else if (flag == "--performance") {
      arguments.performance_path = value;
    } else if (flag == "--quality") {
      arguments.quality_path = value;
    } else if (flag == "--backend") {
      arguments.backend = value;
    } else {
      throw std::runtime_error("unknown argument: " + flag);
    }
  }
  if (arguments.database_path.empty() || arguments.config_path.empty()) {
    throw std::runtime_error("--database and --config are required");
  }
  if (arguments.command == "search" && arguments.query_path.empty()) {
    throw std::runtime_error("search requires --queries");
  }
  if (arguments.backend != "auto" && arguments.backend != "cpu" &&
      arguments.backend != "cuda") {
    throw std::runtime_error("--backend must be auto, cpu or cuda");
  }
  return arguments;
}

void ensure_parent(const std::string& path) {
  const auto parent = std::filesystem::path(path).parent_path();
  if (!parent.empty()) std::filesystem::create_directories(parent);
}

double percentile(std::vector<double> values, double quantile) {
  if (values.empty()) return 0.0;
  std::sort(values.begin(), values.end());
  const std::size_t index = static_cast<std::size_t>(
      std::ceil(quantile * static_cast<double>(values.size())) - 1.0);
  return values[std::min(index, values.size() - 1)];
}

vector_search::SearchResults slice_results(
    const vector_search::SearchResults& results, std::int64_t count) {
  if (count <= 0 || count > results.num_queries) {
    throw std::invalid_argument("invalid result slice");
  }
  vector_search::SearchResults sliced;
  sliced.num_queries = count;
  sliced.top_k = results.top_k;
  const auto elements = static_cast<std::size_t>(count) * results.top_k;
  sliced.ids.assign(results.ids.begin(), results.ids.begin() + elements);
  sliced.scores.assign(results.scores.begin(), results.scores.begin() + elements);
  sliced.query_latency_ms.assign(results.query_latency_ms.begin(),
                                 results.query_latency_ms.begin() + count);
  sliced.peak_candidates = results.peak_candidates;
  sliced.working_set_bytes = results.working_set_bytes;
  sliced.backend = results.backend;
  return sliced;
}

bool choose_cuda(const Arguments& arguments) {
  const bool available = vector_search::cuda_backend_available();
  if (arguments.backend == "cuda" && !available) {
    throw std::runtime_error("CUDA backend requested but unavailable");
  }
  return arguments.backend == "cuda" ||
         (arguments.backend == "auto" && available);
}

void write_build_performance(const Arguments& arguments,
                             const vector_search::VectorDatabase& database,
                             const vector_search::SearchConfig& config,
                             bool use_cuda, double elapsed_ms,
                             std::uintmax_t index_bytes) {
  // 记录建库耗时、索引规模和训练参数。
  ensure_parent(arguments.performance_path);
  std::ofstream output(arguments.performance_path);
  if (!output) throw std::runtime_error("cannot write build performance log");
  output << std::setprecision(12)
         << "operation=build_index\n"
         << "search_mode=ivf_flat\n"
         << "backend=" << (use_cuda ? "cuda" : "cpu") << '\n'
         << "num_vectors=" << database.count << '\n'
         << "dim=" << database.dim << '\n'
         << "metric=" << vector_search::to_string(database.metric) << '\n'
         << "nlist=" << config.nlist << '\n'
         << "training_samples="
         << std::min(database.count, config.training_samples) << '\n'
         << "kmeans_iterations=" << config.kmeans_iterations << '\n'
         << "build_time_ms=" << elapsed_ms << '\n'
         << "index_bytes=" << index_bytes << '\n';
}

void write_search_logs(
    const Arguments& arguments,
    const vector_search::VectorDatabase& database,
    const vector_search::SearchConfig& config,
    const vector_search::SearchResults& results, double total_ms,
    std::uint64_t resident_bytes, double recall, double rank_agreement,
    double score_error,
    std::int64_t quality_queries, double cpu_reference_ms,
    double sampled_backend_ms) {
  // 性能指标与质量指标分别写入日志。
  ensure_parent(arguments.performance_path);
  std::ofstream performance(arguments.performance_path);
  if (!performance) throw std::runtime_error("cannot write performance log");
  const double qps = total_ms > 0.0
                         ? static_cast<double>(results.num_queries) /
                               (total_ms / 1000.0)
                         : 0.0;
  const double mean_latency =
      std::accumulate(results.query_latency_ms.begin(),
                      results.query_latency_ms.end(), 0.0) /
      static_cast<double>(results.query_latency_ms.size());
  const double cpu_reference_qps =
      cpu_reference_ms > 0.0
          ? static_cast<double>(quality_queries) /
                (cpu_reference_ms / 1000.0)
          : 0.0;
  const double sampled_backend_qps =
      sampled_backend_ms > 0.0
          ? static_cast<double>(quality_queries) /
                (sampled_backend_ms / 1000.0)
          : 0.0;
  performance << std::setprecision(12)
              << "operation=search\n"
              << "search_mode=" << vector_search::to_string(config.mode) << '\n'
              << "backend=" << results.backend << '\n'
              << "num_vectors=" << database.count << '\n'
              << "dim=" << database.dim << '\n'
              << "metric=" << vector_search::to_string(database.metric) << '\n'
              << "num_queries=" << results.num_queries << '\n'
              << "top_k=" << results.top_k << '\n'
              << "batch_size=" << config.batch_size << '\n'
              << "nlist=" << config.nlist << '\n'
              << "nprobe=" << config.nprobe << '\n'
              << "query_total_ms=" << total_ms << '\n'
              << "qps=" << qps << '\n'
              << "mean_latency_ms=" << mean_latency << '\n'
              << "p50_latency_ms="
              << percentile(results.query_latency_ms, 0.50) << '\n'
              << "p99_latency_ms="
              << percentile(results.query_latency_ms, 0.99) << '\n'
              << "evaluated_candidates=" << results.evaluated_candidates << '\n'
              << "mean_candidates_per_query="
              << static_cast<double>(results.evaluated_candidates) /
                     results.num_queries
              << '\n'
              << "peak_candidates_per_query=" << results.peak_candidates
              << '\n'
              << "resident_data_and_index_bytes=" << resident_bytes << '\n'
              << "backend_working_set_bytes=" << results.working_set_bytes
              << '\n'
              << "estimated_device_working_set_bytes="
              << (results.backend.rfind("cuda", 0) == 0
                      ? results.working_set_bytes
                      : 0)
              << '\n'
              << "timing_scope=coarse_selection_plus_distance_topk_and_transfers\n"
              << "cpu_reference_queries=" << quality_queries << '\n'
              << "cpu_reference_total_ms=" << cpu_reference_ms << '\n'
              << "cpu_reference_qps=" << cpu_reference_qps << '\n'
              << "sampled_backend_total_ms=" << sampled_backend_ms << '\n'
              << "sampled_backend_qps=" << sampled_backend_qps << '\n'
              << "speedup_vs_cpu_exact="
              << (cpu_reference_qps > 0.0
                      ? sampled_backend_qps / cpu_reference_qps
                      : 0.0)
              << '\n';

  ensure_parent(arguments.quality_path);
  std::ofstream quality(arguments.quality_path);
  if (!quality) throw std::runtime_error("cannot write quality log");
  quality << std::setprecision(12)
          << "quality_queries=" << quality_queries << '\n'
          << "recall_at_k=" << recall << '\n'
          << "rank_id_agreement=" << rank_agreement << '\n'
          << "mean_rank_score_absolute_error=" << score_error << '\n'
          << "ground_truth=cpu_exact\n"
          << "top_k=" << results.top_k << '\n'
          << "nprobe=" << config.nprobe << '\n'
          << "batch_size=" << config.batch_size << '\n';
}

}  // namespace

int main(int argc, char** argv) {
  try {
    // 文件读取和结果写出不计入 query_total_ms。
    const Arguments arguments = parse_arguments(argc, argv);
    const auto database =
        vector_search::load_vector_database(arguments.database_path);
    const auto config = vector_search::load_search_config(arguments.config_path);
    const bool use_cuda = choose_cuda(arguments);

    if (arguments.command == "build") {
      if (config.mode != vector_search::SearchMode::kIvfFlat) {
        throw std::runtime_error("build command requires search_mode=ivf_flat");
      }
      if (arguments.index_path.empty()) {
        throw std::runtime_error("build command requires --index");
      }
      const auto start = std::chrono::steady_clock::now();
      const auto index =
          vector_search::build_ivf_index(database, config, use_cuda);
      vector_search::save_ivf_index(index, arguments.index_path);
      const auto end = std::chrono::steady_clock::now();
      const double elapsed_ms =
          std::chrono::duration<double, std::milli>(end - start).count();
      write_build_performance(
          arguments, database, config, use_cuda, elapsed_ms,
          std::filesystem::file_size(arguments.index_path));
      std::cout << "built " << config.nlist << " IVF lists for "
                << database.count << " vectors in " << elapsed_ms << " ms\n";
      return 0;
    }

    const auto queries = vector_search::load_queries(arguments.query_path);
    if (queries.dim != database.dim) {
      throw std::runtime_error("query dimension does not match database");
    }
    const auto search_start = std::chrono::steady_clock::now();
    vector_search::SearchResults results;
    std::uint64_t resident_bytes = database.values.size() * sizeof(float);
    if (config.mode == vector_search::SearchMode::kExact) {
      results = use_cuda
                    ? vector_search::search_exact_cuda(
                          database, queries, config.top_k, config.batch_size)
                    : vector_search::search_exact_cpu(
                          database, queries, config.top_k, config.batch_size);
    } else {
      if (arguments.index_path.empty()) {
        throw std::runtime_error("ivf_flat search requires --index");
      }
      const auto index = vector_search::load_ivf_index(arguments.index_path);
      resident_bytes += index.centers.size() * sizeof(float) +
                        index.offsets.size() * sizeof(std::uint64_t) +
                        index.ids.size() * sizeof(std::int64_t);
      results = vector_search::search_ivf(database, queries, index, config,
                                          use_cuda);
    }
    const auto search_end = std::chrono::steady_clock::now();
    const double total_ms =
        std::chrono::duration<double, std::milli>(search_end - search_start)
            .count();
    vector_search::write_search_results(results, arguments.output_path);

    const std::int64_t quality_query_count =
        std::min<std::int64_t>(queries.count, config.quality_queries);
    const auto quality_queries =
        vector_search::slice_queries(queries, quality_query_count);
    const auto ground_truth = vector_search::search_exact_cpu(
        database, quality_queries, config.top_k, config.batch_size);
    const auto sampled_results = slice_results(results, quality_query_count);
    const double recall =
        vector_search::recall_at_k(sampled_results, ground_truth);
    const double rank_agreement =
        vector_search::rank_id_agreement(sampled_results, ground_truth);
    const double score_error =
        vector_search::mean_rank_score_error(sampled_results, ground_truth);
    const double cpu_reference_ms =
        std::accumulate(ground_truth.query_latency_ms.begin(),
                        ground_truth.query_latency_ms.end(), 0.0);
    const double sampled_backend_ms =
        std::accumulate(sampled_results.query_latency_ms.begin(),
                        sampled_results.query_latency_ms.end(), 0.0);
    write_search_logs(arguments, database, config, results, total_ms,
                      resident_bytes, recall, rank_agreement, score_error,
                      quality_query_count, cpu_reference_ms, sampled_backend_ms);

    std::cout << std::setprecision(6) << "backend=" << results.backend
              << " qps="
              << static_cast<double>(results.num_queries) /
                     (total_ms / 1000.0)
              << " p50_ms=" << percentile(results.query_latency_ms, 0.50)
              << " p99_ms=" << percentile(results.query_latency_ms, 0.99)
              << " recall@" << results.top_k << '=' << recall << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
}
