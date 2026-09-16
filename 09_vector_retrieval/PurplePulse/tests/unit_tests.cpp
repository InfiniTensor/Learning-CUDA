#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "file_io.h"
#include "search.h"

namespace {

void require(bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

template <typename Operation>
void requireThrows(Operation operation, const std::string& message) {
  try {
    operation();
  } catch (const std::runtime_error&) {
    return;
  }
  throw std::runtime_error(message);
}

VectorDatabase makeDatabase(Metric metric, DataType dtype) {
  VectorDatabase database;
  database.num_vectors = 5;
  database.dim = 2;
  database.metric = metric;
  database.dtype = dtype;
  if (dtype == DataType::kFloat32) {
    database.values = {
        1.0F,  0.0F,  // id 0
        0.0F,  1.0F,  // id 1
        1.0F,  1.0F,  // id 2
        2.0F,  0.0F,  // id 3
        -1.0F, 0.0F   // id 4
    };
  } else {
    // IEEE 754 binary16: 0=0x0000, 1=0x3c00, 2=0x4000, -1=0xbc00。
    database.half_values = {0x3c00, 0x0000, 0x0000, 0x3c00, 0x3c00,
                            0x3c00, 0x4000, 0x0000, 0xbc00, 0x0000};
  }
  return database;
}

QuerySet makeQueries(DataType dtype) {
  QuerySet queries;
  queries.num_queries = 2;
  queries.dim = 2;
  queries.dtype = dtype;
  if (dtype == DataType::kFloat32) {
    queries.values = {1.0F, 0.0F, 0.0F, 1.0F};
  } else {
    queries.half_values = {0x3c00, 0x0000, 0x0000, 0x3c00};
  }
  return queries;
}

void testMetric(Metric metric, DataType dtype) {
  const VectorDatabase database = makeDatabase(metric, dtype);
  const QuerySet queries = makeQueries(dtype);
  SearchParams params;
  params.top_k = 3;
  params.batch_size = 2;

  const SearchResults cpu = cpuExactSearch(database, queries, params);
  params.topk_mode = "simple";
  const SearchResults gpu_simple = gpuExactSearch(database, queries, params);
  params.topk_mode = "block";
  const SearchResults gpu_block = gpuExactSearch(database, queries, params);
  params.topk_mode = "two_stage";
  const SearchResults gpu_two_stage = gpuExactSearch(database, queries, params);
  params.topk_mode = "fused";
  const SearchResults gpu_fused = gpuExactSearch(database, queries, params);
  params.exact_query_tile = 2;
  const SearchResults gpu_fused_tile2 =
      gpuExactSearch(database, queries, params);
  params.exact_query_tile = 4;
  const SearchResults gpu_fused_tile4 =
      gpuExactSearch(database, queries, params);
  params.exact_query_tile = 8;
  const SearchResults gpu_fused_tile8 =
      gpuExactSearch(database, queries, params);
  params.exact_fused_layout = "block_tile";
  const SearchResults gpu_fused_block_tile =
      gpuExactSearch(database, queries, params);
  params.exact_fused_layout = "warp_query";
  params.exact_query_tile = 1;
  std::string error;
  require(resultsMatch(cpu, gpu_simple, 1e-5F, &error),
          "simple Top-K: " + error);
  require(resultsMatch(cpu, gpu_block, 1e-5F, &error), "block Top-K: " + error);
  require(resultsMatch(cpu, gpu_two_stage, 1e-5F, &error),
          "two-stage Top-K: " + error);
  require(resultsMatch(cpu, gpu_fused, 1e-5F, &error), "fused Top-K: " + error);
  require(resultsMatch(cpu, gpu_fused_tile2, 1e-5F, &error),
          "fused query-tile=2 Top-K: " + error);
  require(resultsMatch(cpu, gpu_fused_tile4, 1e-5F, &error),
          "fused query-tile=4 Top-K: " + error);
  require(resultsMatch(cpu, gpu_fused_tile8, 1e-5F, &error),
          "fused query-tile=8 Top-K: " + error);
  require(resultsMatch(cpu, gpu_fused_block_tile, 1e-5F, &error),
          "fused block-tile Top-K: " + error);

  QuerySet initial_queries = queries;
  initial_queries.num_queries = 1;
  if (dtype == DataType::kFloat32) {
    initial_queries.values.resize(initial_queries.dim);
  } else {
    initial_queries.half_values.resize(initial_queries.dim);
  }
  // 首次校验只传一个 query，引擎仍应按 params.batch_size 分配，
  // 并在后续查询中处理更多 query。
  GpuExactSearchEngine resident_engine(database, initial_queries, params);
  SearchStats first_stats;
  SearchStats second_stats;
  const SearchResults resident_first =
      resident_engine.search(queries, &first_stats);
  const SearchResults resident_second =
      resident_engine.search(queries, &second_stats);
  require(resultsMatch(cpu, resident_first, 1e-5F, &error),
          "常驻引擎首次查询: " + error);
  require(resultsMatch(resident_first, resident_second, 0.0F, &error),
          "常驻引擎重复查询: " + error);
  require(resident_engine.databaseH2DMilliseconds() >= 0.0 &&
              resident_engine.deviceBytes() > 0,
          "数据库 H2D 计时不应为负数且设备缓冲区应非空");
  require(
      first_stats.database_h2d_ms == 0.0 && second_stats.database_h2d_ms == 0.0,
      "常驻查询不应重复计入数据库 H2D");
  const double distance_only_ms =
      resident_engine.profileDistanceOnlyMilliseconds(queries, 0, 1);
  require(distance_only_ms >= 0.0,
          "fused distance-only 差分剖析时间不应为负数");

  require(cpu.size() == 2 && cpu[0].size() == 3, "结果形状不正确");
  if (metric == Metric::kL2 || metric == Metric::kCosine) {
    require(cpu[0][0].id == 0, "L2 第一名应当是 id 0");
  } else {
    require(cpu[0][0].id == 3, "内积第一名应当是 id 3");
  }
}

void testBoundaryCases() {
  VectorDatabase database;
  database.num_vectors = 5;
  database.dim = 2;
  database.metric = Metric::kCosine;
  database.dtype = DataType::kFloat32;
  database.values = {
      0.0F,  0.0F,  // id 0: zero vector
      1.0F,  0.0F,  // id 1
      1.0F,  0.0F,  // id 2: tie with id 1
      0.0F,  1.0F,  // id 3
      -1.0F, 0.0F   // id 4
  };

  QuerySet queries;
  queries.num_queries = 5;  // 故意不能被 batch_size=2 整除。
  queries.dim = 2;
  queries.dtype = DataType::kFloat32;
  queries.values = {
      0.0F, 0.0F,  // zero query: every score is zero
      1.0F, 0.0F, 0.0F, 1.0F, -1.0F, 0.0F, 1.0F, 1.0F,
  };

  SearchParams params;
  params.top_k = static_cast<std::uint32_t>(database.num_vectors);
  params.batch_size = 2;
  const SearchResults cpu = cpuExactSearch(database, queries, params);
  require(cpu.size() == queries.num_queries &&
              cpu.front().size() == database.num_vectors,
          "top_k=num_vectors 的结果形状不正确");
  for (std::uint64_t id = 0; id < database.num_vectors; ++id) {
    require(cpu[0][id].id == id && cpu[0][id].score == 0.0F,
            "零向量 cosine 应按较小 ID 打破平局");
  }
  require(cpu[1][0].id == 1 && cpu[1][1].id == 2,
          "相同 cosine 分数应按较小 ID 打破平局");

  std::string error;
  for (const std::string& distance_mode : {"simple", "warp"}) {
    params.distance_mode = distance_mode;
    for (const std::string& topk_mode :
         {"simple", "block", "two_stage", "fused"}) {
      if (topk_mode == "fused" && distance_mode != "warp") {
        continue;
      }
      params.topk_mode = topk_mode;
      const SearchResults gpu = gpuExactSearch(database, queries, params);
      require(resultsMatch(cpu, gpu, 1e-5F, &error),
              distance_mode + "/" + topk_mode + " 边界测试: " + error);
    }
  }
}

VectorDatabase makeLargeKDatabase(DataType dtype) {
  VectorDatabase database;
  database.num_vectors = 128;
  database.dim = 4;
  database.metric = Metric::kL2;
  database.dtype = dtype;
  const std::uint16_t positive_one = 0x3c00;
  const std::uint16_t negative_one = 0xbc00;
  for (std::uint64_t vector_id = 0; vector_id < database.num_vectors;
       ++vector_id) {
    for (std::uint32_t d = 0; d < database.dim; ++d) {
      const bool positive = ((vector_id >> d) & 1U) != 0;
      if (dtype == DataType::kFloat32) {
        database.values.push_back(positive ? 1.0F : -1.0F);
      } else {
        database.half_values.push_back(positive ? positive_one : negative_one);
      }
    }
  }
  return database;
}

QuerySet makeLargeKQueries(DataType dtype) {
  QuerySet queries;
  queries.num_queries = 3;
  queries.dim = 4;
  queries.dtype = dtype;
  const std::vector<float> values = {1.0F,  1.0F, 1.0F, 1.0F,  -1.0F, 1.0F,
                                     -1.0F, 1.0F, 1.0F, -1.0F, 1.0F,  -1.0F};
  if (dtype == DataType::kFloat32) {
    queries.values = values;
  } else {
    for (const float value : values) {
      queries.half_values.push_back(value > 0.0F ? 0x3c00 : 0xbc00);
    }
  }
  return queries;
}

void testFusedLargeK() {
  for (const DataType dtype : {DataType::kFloat32, DataType::kFloat16}) {
    const VectorDatabase database = makeLargeKDatabase(dtype);
    const QuerySet queries = makeLargeKQueries(dtype);
    for (const std::uint32_t top_k : {50U, 100U}) {
      SearchParams params;
      params.top_k = top_k;
      params.batch_size = 2;
      params.distance_mode = "warp";
      params.topk_mode = "fused";
      const SearchResults cpu = cpuExactSearch(database, queries, params);
      GpuExactSearchEngine fused_engine(database, queries, params);
      const SearchResults fused = fused_engine.search(queries);
      std::string error;
      require(resultsMatch(cpu, fused, 1e-5F, &error),
              "fused large-K: " + error);
    }
  }
}

void testFusedDeviceMemory() {
  VectorDatabase database;
  database.num_vectors = 10000;
  database.dim = 1;
  database.metric = Metric::kL2;
  database.dtype = DataType::kFloat32;
  database.values.resize(database.num_vectors, 0.0F);

  QuerySet queries;
  queries.num_queries = 1;
  queries.dim = 1;
  queries.dtype = DataType::kFloat32;
  queries.values = {0.0F};

  SearchParams params;
  params.top_k = 100;
  params.batch_size = 64;
  params.distance_mode = "warp";
  params.topk_mode = "block";
  GpuExactSearchEngine matrix_engine(database, queries, params);
  params.topk_mode = "fused";
  GpuExactSearchEngine fused_engine(database, queries, params);
  require(fused_engine.deviceBytes() < matrix_engine.deviceBytes(),
          "规模化 fused exact 应减少设备缓冲区占用");
}

void testInvalidInputs() {
  VectorDatabase database = makeDatabase(Metric::kL2, DataType::kFloat32);
  QuerySet queries = makeQueries(DataType::kFloat32);
  SearchParams params;

  params.top_k = 0;
  requireThrows([&] { validateInputs(database, queries, params); },
                "top_k=0 应被拒绝");
  params.top_k = 1;
  params.batch_size = 0;
  requireThrows([&] { validateInputs(database, queries, params); },
                "batch_size=0 应被拒绝");
  params.batch_size = 1;

  params.distance_mode = "simple";
  params.topk_mode = "fused";
  requireThrows([&] { validateInputs(database, queries, params); },
                "fused Top-K 与 simple distance 组合应被拒绝");
  params.distance_mode = "warp";
  params.topk_mode = "two_stage";

  params.exact_query_tile = 3;
  requireThrows([&] { validateInputs(database, queries, params); },
                "不支持的 exact_query_tile 应被拒绝");
  params.exact_query_tile = 2;
  requireThrows([&] { validateInputs(database, queries, params); },
                "非 fused Top-K 不应接受 query tile");
  params.exact_query_tile = 1;
  params.exact_fused_blocks = 32;
  requireThrows([&] { validateInputs(database, queries, params); },
                "非 fused Top-K 不应接受 exact_fused_blocks");
  params.exact_fused_blocks = 0;
  params.topk_mode = "fused";
  params.exact_fused_layout = "invalid";
  requireThrows([&] { validateInputs(database, queries, params); },
                "未知 exact_fused_layout 应被拒绝");
  params.exact_fused_layout = "block_tile";
  requireThrows([&] { validateInputs(database, queries, params); },
                "block_tile 不应接受 query tile 1");
  params.exact_fused_layout = "warp_query";
  params.topk_mode = "two_stage";

  VectorDatabase empty_database = database;
  empty_database.num_vectors = 0;
  empty_database.values.clear();
  requireThrows([&] { validateInputs(empty_database, queries, params); },
                "空向量库应被拒绝");

  QuerySet empty_queries = queries;
  empty_queries.num_queries = 0;
  empty_queries.values.clear();
  requireThrows([&] { validateInputs(database, empty_queries, params); },
                "空查询集应被拒绝");

  QuerySet wrong_length = queries;
  wrong_length.values.pop_back();
  requireThrows([&] { validateInputs(database, wrong_length, params); },
                "数据长度与元数据不一致时应被拒绝");
}

}  // namespace

int main() {
  try {
    for (const DataType dtype : {DataType::kFloat32, DataType::kFloat16}) {
      testMetric(Metric::kL2, dtype);
      testMetric(Metric::kInnerProduct, dtype);
      testMetric(Metric::kCosine, dtype);
    }
    testBoundaryCases();
    testFusedLargeK();
    testFusedDeviceMemory();
    testInvalidInputs();
    std::cout << "FP32/FP16 的 CPU/GPU 正确性与边界测试通过\n";
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    std::cerr << "测试失败: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
