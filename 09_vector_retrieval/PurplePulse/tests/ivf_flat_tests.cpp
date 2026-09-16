#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>

#include "ivf_flat.h"
#include "search.h"

namespace {

void require(bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

VectorDatabase makeDatabase(Metric metric, DataType dtype) {
  VectorDatabase database;
  database.num_vectors = 12;
  database.dim = 2;
  database.metric = metric;
  database.dtype = dtype;
  const std::vector<float> values = {
      1.0F,   0.0F, 1.0F, 0.25F, 0.75F,  0.0F,  0.0F,  1.0F,
      0.25F,  1.0F, 0.0F, 0.75F, -1.0F,  0.0F,  -1.0F, -0.25F,
      -0.75F, 0.0F, 0.0F, -1.0F, -0.25F, -1.0F, 0.0F,  -0.75F,
  };
  if (dtype == DataType::kFloat32) {
    database.values = values;
  } else {
    // 测试数据只使用 0、±1、±0.75、±0.25，均可被 binary16 精确表示。
    const auto half = [](float value) -> std::uint16_t {
      if (value == 1.0F) return 0x3c00;
      if (value == -1.0F) return 0xbc00;
      if (value == 0.75F) return 0x3a00;
      if (value == -0.75F) return 0xba00;
      if (value == 0.25F) return 0x3400;
      if (value == -0.25F) return 0xb400;
      return 0x0000;
    };
    for (const float value : values) {
      database.half_values.push_back(half(value));
    }
  }
  return database;
}

QuerySet makeQueries(DataType dtype) {
  QuerySet queries;
  queries.num_queries = 4;
  queries.dim = 2;
  queries.dtype = dtype;
  if (dtype == DataType::kFloat32) {
    queries.values = {1.0F, 0.0F, 0.0F, 1.0F, -1.0F, 0.0F, 0.0F, -1.0F};
  } else {
    queries.half_values = {0x3c00, 0x0000, 0x0000, 0x3c00,
                           0xbc00, 0x0000, 0x0000, 0xbc00};
  }
  return queries;
}

MemoryMetadata makeMemoryMetadata(std::uint64_t num_vectors) {
  MemoryMetadata metadata;
  metadata.num_vectors = num_vectors;
  for (std::uint64_t id = 0; id < num_vectors; ++id) {
    metadata.timestamps.push_back(900 + id * 10);
    metadata.importance.push_back(static_cast<float>(id) /
                                  static_cast<float>(num_vectors));
    metadata.session_ids.push_back(static_cast<std::uint32_t>(id % 2));
    metadata.source_types.push_back(static_cast<std::uint32_t>(id % 3));
  }
  return metadata;
}

std::uint16_t exactHalf(float value) {
  if (value == 1.0F) return 0x3c00;
  if (value == -1.0F) return 0xbc00;
  if (value == 0.75F) return 0x3a00;
  if (value == -0.75F) return 0xba00;
  if (value == 0.25F) return 0x3400;
  if (value == -0.25F) return 0xb400;
  return 0x0000;
}

VectorDatabase makeLargeKDatabase(Metric metric, DataType dtype) {
  constexpr float levels[] = {-1.0F, -0.75F, -0.25F, 0.0F, 0.25F, 0.75F, 1.0F};
  VectorDatabase database;
  database.num_vectors = 128;
  database.dim = 4;
  database.metric = metric;
  database.dtype = dtype;
  std::vector<float> values;
  values.reserve(database.num_vectors * database.dim);
  for (std::uint64_t vector_id = 0; vector_id < database.num_vectors;
       ++vector_id) {
    std::uint64_t code = vector_id;
    for (std::uint32_t d = 0; d < database.dim; ++d) {
      values.push_back(levels[code % 7]);
      code /= 7;
    }
  }
  if (dtype == DataType::kFloat32) {
    database.values = std::move(values);
  } else {
    database.half_values.reserve(values.size());
    for (const float value : values) {
      database.half_values.push_back(exactHalf(value));
    }
  }
  return database;
}

QuerySet makeLargeKQueries(DataType dtype) {
  const std::vector<float> values = {
      1.0F,  0.25F, -0.75F, 0.0F,   -0.25F, 1.0F,
      0.75F, -1.0F, 0.0F,   -0.75F, 1.0F,   0.25F,
  };
  QuerySet queries;
  queries.num_queries = 3;
  queries.dim = 4;
  queries.dtype = dtype;
  if (dtype == DataType::kFloat32) {
    queries.values = values;
  } else {
    for (const float value : values) {
      queries.half_values.push_back(exactHalf(value));
    }
  }
  return queries;
}

void testLargeK(Metric metric, DataType dtype) {
  const VectorDatabase database = makeLargeKDatabase(metric, dtype);
  const QuerySet queries = makeLargeKQueries(dtype);
  const IvfFlatIndex index = buildIvfFlatIndex(database, 8, 5, 128);
  for (const std::uint32_t top_k : {50U, 100U}) {
    SearchParams params;
    params.top_k = top_k;
    params.search_mode = "ivf_flat";
    params.nlist = 8;
    params.nprobe = 8;
    params.batch_size = 4;
    params.distance_mode = "warp";
    SearchParams exact_params = params;
    exact_params.search_mode = "exact";
    exact_params.topk_mode = "block";
    const SearchResults exact = cpuExactSearch(database, queries, exact_params);
    const SearchResults cpu_ivf = cpuIvfFlatSearch(index, queries, params);
    std::string error;
    require(resultsMatch(exact, cpu_ivf, 0.0F, &error),
            "大 K CPU IVF 与 exact 不一致: " + error);
    SearchResults warp8_results;
    for (const std::string& distance_mode :
         {"warp8", "warp4", "warp2", "warp1", "warp_compact",
          "bucket_major"}) {
      params.distance_mode = distance_mode;
      GpuIvfFlatSearchEngine engine(index, queries, params);
      const SearchResults gpu_ivf = engine.search(queries);
      require(resultsMatch(cpu_ivf, gpu_ivf, 1e-5F, &error),
              "大 K " + distance_mode + " GPU IVF 与 CPU IVF 不一致: " + error);
      if (distance_mode == "warp8") {
        warp8_results = gpu_ivf;
      } else {
        require(resultsMatch(warp8_results, gpu_ivf, 0.0F, &error),
                "大 K compact warp 与 warp8 结果不一致: " + error);
      }
    }
  }
}

void testRoundTrip(Metric metric, DataType dtype,
                   const std::filesystem::path& directory) {
  const VectorDatabase database = makeDatabase(metric, dtype);
  const QuerySet queries = makeQueries(dtype);
  const IvfFlatIndex built = buildIvfFlatIndex(database, 4, 8, 12);
  const IvfFlatIndex parallel_built =
      buildIvfFlatIndex(database, 4, 8, 12, 4);
  const IvfFlatIndex parallel_repeated =
      buildIvfFlatIndex(database, 4, 8, 12, 4);
  require(parallel_built.centers == parallel_repeated.centers &&
              parallel_built.offsets == parallel_repeated.offsets &&
              parallel_built.ids == parallel_repeated.ids &&
              parallel_built.values == parallel_repeated.values &&
              parallel_built.half_values == parallel_repeated.half_values,
          "固定线程数的并行 IVF 建库结果应可重复");
  const std::filesystem::path path =
      directory /
      (std::string("index-") + std::to_string(static_cast<int>(metric)) + "-" +
       std::to_string(static_cast<int>(dtype)) + ".bin");
  writeIvfFlatIndex(path.string(), built);
  const IvfFlatIndex loaded = readIvfFlatIndex(path.string());
  require(loaded.centers == built.centers && loaded.offsets == built.offsets &&
              loaded.ids == built.ids && loaded.values == built.values &&
              loaded.half_values == built.half_values,
          "IVF 索引保存/加载后内容变化");

  SearchParams exact_params;
  exact_params.top_k = 3;
  const SearchResults exact = cpuExactSearch(database, queries, exact_params);
  SearchParams ivf_params = exact_params;
  ivf_params.search_mode = "ivf_flat";
  ivf_params.nlist = 4;
  ivf_params.nprobe = 4;
  const SearchResults exhaustive_ivf =
      cpuIvfFlatSearch(loaded, queries, ivf_params);
  std::string error;
  require(resultsMatch(exact, exhaustive_ivf, 0.0F, &error),
          "nprobe=nlist 应与 exact 完全一致: " + error);
  require(recallAtK(exact, exhaustive_ivf) == 1.0,
          "nprobe=nlist 的 recall@K 应为 1");

  ivf_params.nprobe = 2;
  const SearchResults approximate =
      cpuIvfFlatSearch(loaded, queries, ivf_params);
  const double recall = recallAtK(exact, approximate);
  require(recall >= 0.0 && recall <= 1.0, "IVF recall@K 必须位于 [0, 1]");

  ivf_params.distance_mode = "simple";
  GpuIvfFlatSearchEngine scalar_gpu_engine(loaded, queries, ivf_params);
  const SearchResults scalar_gpu = scalar_gpu_engine.search(queries);
  require(resultsMatch(approximate, scalar_gpu, 1e-5F, &error),
          "scalar GPU IVF 与 CPU IVF 不一致: " + error);

  ivf_params.distance_mode = "warp";
  GpuIvfFlatSearchEngine gpu_engine(loaded, queries, ivf_params);
  SearchStats first_stats;
  const SearchResults gpu_first = gpu_engine.search(queries, &first_stats);
  const SearchResults gpu_second = gpu_engine.search(queries);
  require(resultsMatch(approximate, gpu_first, 1e-5F, &error),
          "GPU IVF 与 CPU IVF 不一致: " + error);
  require(resultsMatch(gpu_first, gpu_second, 0.0F, &error),
          "GPU IVF 常驻引擎重复查询不一致: " + error);
  require(first_stats.host_selection_ms >= 0.0 &&
              first_stats.distance_kernel_ms >= 0.0 &&
              gpu_engine.indexH2DMilliseconds() >= 0.0 &&
              gpu_engine.deviceBytes() > 0,
          "GPU IVF 计时不应为负数且设备缓冲区应非空");

  SearchParams adaptive_params = ivf_params;
  adaptive_params.nprobe = 4;
  adaptive_params.nprobe_policy = "score_mass";
  adaptive_params.adaptive_nprobe_min = 2;
  adaptive_params.adaptive_nprobe_step = 1;
  adaptive_params.adaptive_target_mass = 0.5F;
  adaptive_params.adaptive_temperature = 0.2F;
  GpuIvfFlatSearchEngine adaptive_engine(loaded, queries, adaptive_params);
  SearchStats adaptive_stats;
  const SearchResults adaptive_results =
      adaptive_engine.search(queries, &adaptive_stats);
  require(adaptive_results.size() == queries.num_queries &&
              adaptive_stats.selected_probe_queries == queries.num_queries &&
              adaptive_stats.selected_probe_min >= 2 &&
              adaptive_stats.selected_probe_max <= 4 &&
              adaptive_stats.selected_probe_sum >= 2 * queries.num_queries &&
              adaptive_stats.selected_probe_sum <= 4 * queries.num_queries,
          "自适应 nprobe 统计或边界错误");

  adaptive_params.adaptive_nprobe_min = 4;
  adaptive_params.adaptive_target_mass = 0.1F;
  adaptive_params.adaptive_execution = "grouped";
  GpuIvfFlatSearchEngine capped_adaptive_engine(loaded, queries,
                                                adaptive_params);
  SearchStats capped_stats;
  const SearchResults capped_adaptive =
      capped_adaptive_engine.search(queries, &capped_stats);
  require(resultsMatch(exhaustive_ivf, capped_adaptive, 1e-5F, &error),
          "min_nprobe=max_nprobe 时自适应结果应与固定全桶一致: " + error);
  require(capped_stats.selected_probe_min == 4 &&
              capped_stats.selected_probe_max == 4,
          "自适应 nprobe 上下限相同时统计错误");

  const MemoryMetadata metadata = makeMemoryMetadata(database.num_vectors);
  SearchParams memory_params = ivf_params;
  memory_params.nprobe = 4;
  memory_params.top_k = 3;
  memory_params.memory_mode = "fused";
  memory_params.memory_importance_weight = 0.2F;
  memory_params.memory_recency_weight = 0.1F;
  memory_params.memory_time_scale = 50.0F;
  memory_params.memory_now = 1000;
  memory_params.filter_session_id = 0;
  const SearchResults cpu_fused =
      cpuIvfFlatSearch(loaded, queries, memory_params, &metadata);
  GpuIvfFlatSearchEngine fused_engine(loaded, queries, memory_params,
                                      &metadata);
  const SearchResults gpu_fused = fused_engine.search(queries);
  require(resultsMatch(cpu_fused, gpu_fused, 1e-5F, &error),
          "融合记忆评分的 GPU IVF 与 CPU baseline 不一致: " + error);
  for (const auto& query_results : gpu_fused) {
    require(query_results.size() == memory_params.top_k,
            "过滤后候选充足时应返回完整 Top-K");
    for (const Neighbor& neighbor : query_results) {
      require(neighbor.id % 2 == 0, "GPU session filter 返回了不匹配候选");
    }
  }

  memory_params.filter_source_type = 99;
  const SearchResults cpu_empty =
      cpuIvfFlatSearch(loaded, queries, memory_params, &metadata);
  GpuIvfFlatSearchEngine empty_filter_engine(loaded, queries, memory_params,
                                             &metadata);
  const SearchResults gpu_empty = empty_filter_engine.search(queries);
  require(resultsMatch(cpu_empty, gpu_empty, 0.0F, &error),
          "过滤后不足 K 个候选时 CPU/GPU 结果不一致: " + error);
  for (const auto& query_results : gpu_empty) {
    require(query_results.empty(), "完全不匹配的过滤条件应返回空结果");
  }

  memory_params.memory_mode = "rerank";
  memory_params.filter_source_type = UINT32_MAX;
  memory_params.memory_rerank_factor = 4;
  const SearchResults cpu_rerank =
      cpuIvfFlatSearch(loaded, queries, memory_params, &metadata);
  GpuIvfFlatSearchEngine rerank_engine(loaded, queries, memory_params,
                                       &metadata);
  SearchStats rerank_stats;
  const SearchResults gpu_rerank = rerank_engine.search(queries, &rerank_stats);
  require(resultsMatch(cpu_rerank, gpu_rerank, 1e-5F, &error),
          "独立记忆重排的 GPU IVF 与 CPU baseline 不一致: " + error);
  require(rerank_stats.memory_rerank_ms >= 0.0, "独立记忆重排计时不应为负数");
}

}  // namespace

int main() {
  const auto unique =
      std::chrono::steady_clock::now().time_since_epoch().count();
  const std::filesystem::path directory =
      std::filesystem::temp_directory_path() /
      ("purplepulse-ivf-test-" + std::to_string(unique));
  try {
    std::filesystem::create_directories(directory);
    for (const DataType dtype : {DataType::kFloat32, DataType::kFloat16}) {
      for (const Metric metric :
           {Metric::kL2, Metric::kInnerProduct, Metric::kCosine}) {
        testRoundTrip(metric, dtype, directory);
        testLargeK(metric, dtype);
      }
    }
    std::filesystem::remove_all(directory);
    std::cout << "IVF-Flat 建库、持久化、查询与 recall 测试通过\n";
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    std::filesystem::remove_all(directory);
    std::cerr << "测试失败: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
