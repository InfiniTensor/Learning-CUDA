#include "vector_search/search.hpp"

#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool condition, const std::string &message) {
  if (!condition)
    throw std::runtime_error(message);
}

template <typename T> void write_pod(std::ostream &output, T value) {
  output.write(reinterpret_cast<const char *>(&value), sizeof(value));
}

std::filesystem::path unique_temp(const std::string &suffix) {
  const auto tick = std::chrono::steady_clock::now().time_since_epoch().count();
  return std::filesystem::temp_directory_path() /
         ("vector_search_test_" + std::to_string(tick) + suffix);
}

vector_search::VectorDatabase small_database() {
  vector_search::VectorDatabase database;
  database.count = 6;
  database.dim = 2;
  database.metric = vector_search::Metric::kL2;
  database.values = {0.0f,  0.0f,  1.0f, 0.0f,  0.0f,  1.0f,
                     10.0f, 10.0f, 9.0f, 10.0f, 10.0f, 9.0f};
  return database;
}

vector_search::QuerySet small_queries() {
  vector_search::QuerySet queries;
  queries.count = 2;
  queries.dim = 2;
  queries.values = {0.1f, 0.1f, 9.9f, 9.9f};
  return queries;
}

void test_exact_search() {
  // 手算数据：最近邻、同分 ID 顺序及最终排序。
  const auto results =
      vector_search::search_exact_cpu(small_database(), small_queries(), 3, 2);
  require(results.ids[0] == 0 && results.ids[3] == 3,
          "exact L2 nearest neighbor is wrong");
  require(results.ids[1] == 1 && results.ids[2] == 2,
          "equal L2 scores must use vector id as the CPU tie-breaker");
  for (std::int64_t query = 0; query < results.num_queries; ++query) {
    for (int rank = 1; rank < results.top_k; ++rank) {
      const auto offset =
          static_cast<std::size_t>(query) * results.top_k + rank;
      require(results.scores[offset - 1] <= results.scores[offset],
              "exact results are not sorted");
    }
  }
  require(vector_search::rank_id_agreement(results, results) == 1.0,
          "identical exact results must have full rank ID agreement");
}

void test_metrics_and_topk_ranges() {
  vector_search::VectorDatabase database;
  database.count = 128;
  database.dim = 4;
  database.metric = vector_search::Metric::kInnerProduct;
  database.values.resize(static_cast<std::size_t>(database.count) *
                         database.dim);
  for (std::int64_t id = 0; id < database.count; ++id) {
    database.values[static_cast<std::size_t>(id) * database.dim] =
        static_cast<float>(id);
    database.values[static_cast<std::size_t>(id) * database.dim + 1] = 1.0f;
  }
  vector_search::QuerySet queries;
  queries.count = 1;
  queries.dim = 4;
  queries.values = {1.0f, 0.0f, 0.0f, 0.0f};
  for (const int top_k : {1, 10, 50, 100}) {
    const auto results =
        vector_search::search_exact_cpu(database, queries, top_k, 16);
    require(results.ids.front() == 127,
            "inner-product results must be descending");
    require(results.ids.size() == static_cast<std::size_t>(top_k),
            "Top-K result size is wrong");
  }

  database.metric = vector_search::Metric::kCosine;
  database.values.assign(database.values.size(), 0.0f);
  database.values[4] = 1.0f;
  const auto cosine = vector_search::search_exact_cpu(database, queries, 1, 1);
  require(cosine.ids[0] == 1 && std::isfinite(cosine.scores[0]),
          "cosine search must handle zero vectors without NaN");
}

void test_ivf_round_trip_and_recall() {
  // nprobe=nlist 时 IVF 候选集合应覆盖全库，因此必须恢复 exact search。
  const auto database = small_database();
  const auto queries = small_queries();
  vector_search::SearchConfig config;
  config.mode = vector_search::SearchMode::kIvfFlat;
  config.top_k = 3;
  config.batch_size = 2;
  config.nlist = 2;
  config.nprobe = 2;
  config.training_samples = 6;
  config.kmeans_iterations = 3;
  const auto index = vector_search::build_ivf_index(database, config, false);
  const auto path = unique_temp(".ivf");
  vector_search::save_ivf_index(index, path.string());
  const auto loaded = vector_search::load_ivf_index(path.string());
  std::filesystem::remove(path);
  require(loaded.num_vectors == index.num_vectors && loaded.dim == index.dim &&
              loaded.nlist == index.nlist && loaded.metric == index.metric &&
              loaded.centers == index.centers &&
              loaded.offsets == index.offsets && loaded.ids == index.ids,
          "IVF index round-trip changed index contents");
  const auto approximate =
      vector_search::search_ivf(database, queries, loaded, config, false);
  const auto exact = vector_search::search_exact_cpu(database, queries, 3, 2);
  require(std::abs(vector_search::recall_at_k(approximate, exact) - 1.0) <
              1e-12,
          "nprobe=nlist must reproduce exact recall");

  config.nprobe = 1;
  const auto narrow =
      vector_search::search_ivf(database, queries, loaded, config, false);
  const double narrow_recall = vector_search::recall_at_k(narrow, exact);
  config.nprobe = 2;
  const auto wide =
      vector_search::search_ivf(database, queries, loaded, config, false);
  const double wide_recall = vector_search::recall_at_k(wide, exact);
  require(wide_recall + 1e-12 >= narrow_recall &&
              std::abs(wide_recall - 1.0) < 1e-12,
          "IVF recall must not decrease when nprobe expands to all lists");

  // 打乱倒排表中的同分 ID，检查结果仍按全局 ID 排序。
  vector_search::VectorDatabase tie_database;
  tie_database.count = 4;
  tie_database.dim = 1;
  tie_database.metric = vector_search::Metric::kL2;
  tie_database.values = {0.0f, 2.0f, 0.0f, 2.0f};
  vector_search::QuerySet tie_queries;
  tie_queries.count = 1;
  tie_queries.dim = 1;
  tie_queries.values = {1.0f};
  vector_search::IvfIndex tie_index;
  tie_index.num_vectors = 4;
  tie_index.dim = 1;
  tie_index.nlist = 2;
  tie_index.metric = vector_search::Metric::kL2;
  tie_index.centers = {0.0f, 2.0f};
  tie_index.offsets = {0, 2, 4};
  tie_index.ids = {3, 1, 2, 0};
  config.top_k = 4;
  config.batch_size = 1;
  config.nlist = 2;
  config.nprobe = 2;
  const auto tied = vector_search::search_ivf(
      tie_database, tie_queries, tie_index, config, false);
  require(tied.ids == std::vector<std::int64_t>({0, 1, 2, 3}),
          "IVF equal scores must use the global vector-id tie-breaker");
}

void test_binary_readers() {
  // 覆盖 FP32 数据库、FP16 查询与 FP16 数据库。
  const auto database_path = unique_temp(".database.bin");
  {
    std::ofstream output(database_path, std::ios::binary);
    const std::array<char, 8> magic = {'V', 'E', 'C', 'D', 'B', '0', '1', '\0'};
    output.write(magic.data(), magic.size());
    write_pod<std::uint32_t>(output, 1);
    write_pod<std::int64_t>(output, 2);
    write_pod<std::int32_t>(output, 2);
    write_pod<std::uint8_t>(output, 1);
    write_pod<std::uint8_t>(output, 1);
    write_pod<std::uint16_t>(output, 0);
    const std::array<float, 4> values = {1.0f, 2.0f, 3.0f, 4.0f};
    output.write(reinterpret_cast<const char *>(values.data()),
                 values.size() * sizeof(float));
  }
  const auto database =
      vector_search::load_vector_database(database_path.string());
  std::filesystem::remove(database_path);
  require(database.count == 2 && database.dim == 2 &&
              database.values[3] == 4.0f,
          "FP32 database reader failed");

  const auto query_path = unique_temp(".queries.bin");
  {
    std::ofstream output(query_path, std::ios::binary);
    const std::array<char, 8> magic = {'V', 'E', 'C', 'Q', 'R', 'Y', '1', '\0'};
    output.write(magic.data(), magic.size());
    write_pod<std::uint32_t>(output, 1);
    write_pod<std::int64_t>(output, 1);
    write_pod<std::int32_t>(output, 2);
    write_pod<std::uint8_t>(output, 2);
    const std::array<char, 3> reserved{};
    output.write(reserved.data(), reserved.size());
    const std::array<std::uint16_t, 2> values = {0x3c00U, 0xc000U};
    output.write(reinterpret_cast<const char *>(values.data()),
                 values.size() * sizeof(std::uint16_t));
  }
  const auto queries = vector_search::load_queries(query_path.string());
  std::filesystem::remove(query_path);
  require(queries.values[0] == 1.0f && queries.values[1] == -2.0f,
          "FP16 query conversion failed");

  const auto half_database_path = unique_temp(".fp16.database.bin");
  {
    std::ofstream output(half_database_path, std::ios::binary);
    const std::array<char, 8> magic = {'V', 'E', 'C', 'D', 'B', '0', '1', '\0'};
    output.write(magic.data(), magic.size());
    write_pod<std::uint32_t>(output, 1);
    write_pod<std::int64_t>(output, 3);
    write_pod<std::int32_t>(output, 2);
    write_pod<std::uint8_t>(output, 2);
    write_pod<std::uint8_t>(output, 1);
    write_pod<std::uint16_t>(output, 0);
    const std::array<std::uint16_t, 6> values = {0x0000U, 0x0000U, 0x3c00U,
                                                 0x0000U, 0x0000U, 0x3c00U};
    output.write(reinterpret_cast<const char *>(values.data()),
                 values.size() * sizeof(std::uint16_t));
  }
  const auto half_database =
      vector_search::load_vector_database(half_database_path.string());
  std::filesystem::remove(half_database_path);
  const auto half_results =
      vector_search::search_exact_cpu(half_database, queries, 2, 1);
  require(half_results.ids[0] == 1 && half_results.ids[1] == 0,
          "FP16 database/query end-to-end search failed");
}

void test_invalid_inputs_and_corrupt_files() {
  // 非法输入应抛出异常。
  bool rejected_top_k = false;
  try {
    static_cast<void>(vector_search::search_exact_cpu(small_database(),
                                                      small_queries(), 0, 1));
  } catch (const std::invalid_argument &) {
    rejected_top_k = true;
  }
  require(rejected_top_k, "Top-K=0 was not rejected");

  auto mismatched_queries = small_queries();
  mismatched_queries.dim = 3;
  bool rejected_dimensions = false;
  try {
    static_cast<void>(vector_search::search_exact_cpu(
        small_database(), mismatched_queries, 1, 1));
  } catch (const std::invalid_argument &) {
    rejected_dimensions = true;
  }
  require(rejected_dimensions, "mismatched query dimensions were not rejected");

  const auto corrupt_path = unique_temp(".corrupt.bin");
  {
    std::ofstream output(corrupt_path, std::ios::binary);
    const std::array<char, 8> bad_magic{};
    output.write(bad_magic.data(), bad_magic.size());
  }
  bool rejected_corrupt_file = false;
  try {
    static_cast<void>(
        vector_search::load_vector_database(corrupt_path.string()));
  } catch (const std::runtime_error &) {
    rejected_corrupt_file = true;
  }
  std::filesystem::remove(corrupt_path);
  require(rejected_corrupt_file, "corrupt database file was not rejected");
}

void test_config() {
  const std::string root = VECTOR_SEARCH_SOURCE_DIR;
  const auto config =
      vector_search::load_search_config(root + "/configs/ivf_smoke.txt");
  require(config.mode == vector_search::SearchMode::kIvfFlat &&
              config.nlist == 64 && config.nprobe == 8,
          "search config parser failed");
}

#ifdef VECTOR_SEARCH_HAS_CUDA
vector_search::VectorDatabase cuda_database(vector_search::Metric metric) {
  vector_search::VectorDatabase database;
  database.count = 257;
  database.dim = 8;
  database.metric = metric;
  database.values.resize(static_cast<std::size_t>(database.count) *
                         database.dim);
  for (std::int64_t id = 0; id < database.count; ++id) {
    for (int component = 0; component < database.dim; ++component) {
      // ID 相关的线性项打破近邻分数 tie，三角项提供非轴对齐测试数据。
      database.values[static_cast<std::size_t>(id) * database.dim + component] =
          std::sin(static_cast<float>((id + 1) * (component + 3)) * 0.013f) +
          static_cast<float>(id) * 0.0017f + component * 0.011f;
    }
  }
  return database;
}

vector_search::QuerySet
cuda_queries(const vector_search::VectorDatabase &database) {
  vector_search::QuerySet queries;
  queries.count = 5;
  queries.dim = database.dim;
  queries.values.resize(static_cast<std::size_t>(queries.count) * queries.dim);
  const std::array<std::int64_t, 5> sources = {3, 41, 103, 188, 251};
  for (std::int64_t query = 0; query < queries.count; ++query) {
    for (int component = 0; component < queries.dim; ++component) {
      queries
          .values[static_cast<std::size_t>(query) * queries.dim + component] =
          database
              .values[static_cast<std::size_t>(sources[query]) * database.dim +
                      component] +
          static_cast<float>(component + 1) * 1.0e-4f;
    }
  }
  return queries;
}

void require_cuda_matches_cpu(const vector_search::SearchResults &gpu,
                              const vector_search::SearchResults &cpu,
                              const std::string &context) {
  require(std::abs(vector_search::recall_at_k(gpu, cpu) - 1.0) < 1e-12,
          context + ": CUDA Recall@K mismatch");
  require(std::abs(vector_search::rank_id_agreement(gpu, cpu) - 1.0) < 1e-12,
          context + ": CUDA rank ordering mismatch");
  require(vector_search::mean_rank_score_error(gpu, cpu) < 2.0e-5,
          context + ": CUDA score mismatch");
}

void test_cuda_gate() {
  if (!vector_search::cuda_backend_available())
    return;

  // 三种度量与不满 batch 的查询，均对照 CPU 精确结果。
  for (const auto metric :
       {vector_search::Metric::kL2, vector_search::Metric::kInnerProduct,
        vector_search::Metric::kCosine}) {
    const auto database = cuda_database(metric);
    const auto queries = cuda_queries(database);
    const auto cpu = vector_search::search_exact_cpu(database, queries, 10, 3);
    const auto gpu = vector_search::search_exact_cuda(database, queries, 10, 3);
    require_cuda_matches_cpu(gpu, cpu, "exact metric gate");
  }

  const auto tie_cpu =
      vector_search::search_exact_cpu(small_database(), small_queries(), 3, 2);
  const auto tie_gpu =
      vector_search::search_exact_cuda(small_database(), small_queries(), 3, 2);
  require_cuda_matches_cpu(tie_gpu, tie_cpu, "exact tie-breaking gate");

  vector_search::VectorDatabase zero_cosine;
  zero_cosine.count = 3;
  zero_cosine.dim = 2;
  zero_cosine.metric = vector_search::Metric::kCosine;
  zero_cosine.values = {0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f};
  vector_search::QuerySet cosine_query;
  cosine_query.count = 1;
  cosine_query.dim = 2;
  cosine_query.values = {1.0f, 0.0f};
  const auto cosine_cpu =
      vector_search::search_exact_cpu(zero_cosine, cosine_query, 1, 1);
  const auto cosine_gpu =
      vector_search::search_exact_cuda(zero_cosine, cosine_query, 1, 1);
  require_cuda_matches_cpu(cosine_gpu, cosine_cpu, "zero-cosine gate");
  require(std::isfinite(cosine_gpu.scores[0]),
          "CUDA cosine search produced NaN for a zero vector");

  // GPU assignment 构建索引，再分别用 CPU/CUDA rerank 同一候选集合。
  const auto database = cuda_database(vector_search::Metric::kL2);
  const auto queries = cuda_queries(database);
  vector_search::SearchConfig config;
  config.mode = vector_search::SearchMode::kIvfFlat;
  config.top_k = 10;
  config.batch_size = 3;
  config.nlist = 8;
  config.nprobe = 8;
  config.training_samples = database.count;
  config.kmeans_iterations = 3;
  const auto gpu_built_index =
      vector_search::build_ivf_index(database, config, true);
  const auto cpu_ivf = vector_search::search_ivf(
      database, queries, gpu_built_index, config, false);
  const auto gpu_ivf = vector_search::search_ivf(database, queries,
                                                 gpu_built_index, config, true);
  require_cuda_matches_cpu(gpu_ivf, cpu_ivf, "IVF candidate gate");
}
#endif

} // namespace

int main() {
  try {
    test_exact_search();
    test_metrics_and_topk_ranges();
    test_ivf_round_trip_and_recall();
    test_binary_readers();
    test_invalid_inputs_and_corrupt_files();
    test_config();
#ifdef VECTOR_SEARCH_HAS_CUDA
    test_cuda_gate();
#endif
    std::cout << "all vector-search tests passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "test failure: " << error.what() << '\n';
    return 1;
  }
}
