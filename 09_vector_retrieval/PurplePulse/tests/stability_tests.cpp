#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "ivf_flat.h"
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

constexpr std::array<float, 9> kLevels = {
    -1.0F, -0.75F, -0.5F, -0.25F, 0.0F,
    0.25F, 0.5F, 0.75F, 1.0F,
};

std::uint16_t exactHalf(float value) {
  if (value == -1.0F) return 0xbc00;
  if (value == -0.75F) return 0xba00;
  if (value == -0.5F) return 0xb800;
  if (value == -0.25F) return 0xb400;
  if (value == 0.25F) return 0x3400;
  if (value == 0.5F) return 0x3800;
  if (value == 0.75F) return 0x3a00;
  if (value == 1.0F) return 0x3c00;
  return 0x0000;
}

float deterministicValue(std::uint64_t row, std::uint32_t dimension) {
  const std::uint64_t mixed = row * 37U + dimension * 17U +
                              (row / 3U) * 5U + dimension / 7U;
  return kLevels[mixed % kLevels.size()];
}

void appendValue(std::vector<float>* fp32, std::vector<std::uint16_t>* fp16,
                 DataType dtype, float value) {
  if (dtype == DataType::kFloat32) {
    fp32->push_back(value);
  } else {
    fp16->push_back(exactHalf(value));
  }
}

VectorDatabase makeCoverageDatabase(std::uint32_t dim, Metric metric,
                                    DataType dtype,
                                    std::uint64_t num_vectors = 107) {
  VectorDatabase database;
  database.num_vectors = num_vectors;
  database.dim = dim;
  database.metric = metric;
  database.dtype = dtype;
  database.values.reserve(dtype == DataType::kFloat32 ? num_vectors * dim : 0);
  database.half_values.reserve(dtype == DataType::kFloat16 ? num_vectors * dim
                                                            : 0);
  for (std::uint64_t row = 0; row < num_vectors; ++row) {
    // Rows 0 and 1 are deliberately identical. This validates stable ID
    // tie-breaking for every dimension, dtype, metric, batch, and K.
    const std::uint64_t source_row = row == 1 ? 0 : row;
    for (std::uint32_t d = 0; d < dim; ++d) {
      appendValue(&database.values, &database.half_values, dtype,
                  deterministicValue(source_row, d));
    }
  }
  return database;
}

QuerySet makeCoverageQueries(std::uint32_t dim, DataType dtype,
                             std::uint64_t num_queries = 9) {
  QuerySet queries;
  queries.num_queries = num_queries;
  queries.dim = dim;
  queries.dtype = dtype;
  queries.values.reserve(dtype == DataType::kFloat32 ? num_queries * dim : 0);
  queries.half_values.reserve(dtype == DataType::kFloat16 ? num_queries * dim
                                                           : 0);
  for (std::uint64_t row = 0; row < num_queries; ++row) {
    for (std::uint32_t d = 0; d < dim; ++d) {
      // The first query is all zero, making every IP/cosine score tie. The
      // second query exactly matches duplicate database rows 0 and 1.
      const float value = row == 0 ? 0.0F : deterministicValue(row - 1, d);
      appendValue(&queries.values, &queries.half_values, dtype, value);
    }
  }
  return queries;
}

void compareExact(const VectorDatabase& database, const QuerySet& queries,
                  const SearchParams& params, const std::string& label) {
  const SearchResults cpu = cpuExactSearch(database, queries, params);
  GpuExactSearchEngine engine(database, queries, params);
  const SearchResults gpu = engine.search(queries);
  std::string error;
  require(resultsMatch(cpu, gpu, 5e-5F, &error), label + ": " + error);
  if ((database.metric == Metric::kInnerProduct ||
       database.metric == Metric::kCosine) &&
      !gpu.empty()) {
    const std::uint32_t expected = std::min<std::uint32_t>(
        params.top_k, static_cast<std::uint32_t>(database.num_vectors));
    for (std::uint32_t rank = 0; rank < expected; ++rank) {
      require(gpu[0][rank].id == rank,
              label + ": equal-score candidates must use ascending IDs");
    }
  }
}

void testExactCoverageMatrix() {
  constexpr std::array<std::uint32_t, 4> dimensions = {64, 128, 384, 768};
  constexpr std::array<std::uint32_t, 6> batches = {1, 8, 32, 40, 64, 128};
  constexpr std::array<std::uint32_t, 4> top_ks = {1, 10, 50, 100};
  std::uint32_t cases = 0;
  for (const DataType dtype : {DataType::kFloat32, DataType::kFloat16}) {
    for (const Metric metric :
         {Metric::kL2, Metric::kInnerProduct, Metric::kCosine}) {
      for (const std::uint32_t dim : dimensions) {
        const VectorDatabase database = makeCoverageDatabase(dim, metric, dtype);
        const QuerySet queries = makeCoverageQueries(dim, dtype);
        for (const std::uint32_t top_k : top_ks) {
          for (const std::uint32_t batch_size : batches) {
            SearchParams params;
            params.top_k = top_k;
            params.batch_size = batch_size;
            params.distance_mode = "warp";
            params.topk_mode = "fused";
            compareExact(database, queries, params,
                         "Exact coverage dim=" + std::to_string(dim) +
                             " batch=" + std::to_string(batch_size) +
                             " K=" + std::to_string(top_k));
            ++cases;
          }
        }
      }
    }
  }
  require(cases == 576, "Exact coverage matrix case count changed");
  std::cout << "Exact coverage matrix: " << cases << " cases\n";
}

void testBatchCapacityAndNonWarpDimension() {
  constexpr std::array<std::uint32_t, 6> batches = {1, 8, 32, 40, 64, 128};
  for (const std::uint32_t batch_size : batches) {
    const VectorDatabase database =
        makeCoverageDatabase(65, Metric::kL2, DataType::kFloat32, 101);
    const QuerySet queries =
        makeCoverageQueries(65, DataType::kFloat32, batch_size + 1U);
    SearchParams params;
    params.top_k = 10;
    params.batch_size = batch_size;
    params.distance_mode = "warp";
    params.topk_mode = "fused";
    compareExact(database, queries, params,
                 "non-warp dim=65 full batch=" + std::to_string(batch_size));
  }

  for (const DataType dtype : {DataType::kFloat32, DataType::kFloat16}) {
    for (const Metric metric :
         {Metric::kL2, Metric::kInnerProduct, Metric::kCosine}) {
      const VectorDatabase database = makeCoverageDatabase(65, metric, dtype);
      const QuerySet queries = makeCoverageQueries(65, dtype);
      SearchParams params;
      params.top_k = 100;
      params.batch_size = 8;
      params.distance_mode = "warp";
      params.topk_mode = "fused";
      compareExact(database, queries, params,
                   "non-warp dim=65 dtype/metric coverage");
    }
  }
}

void testExactRecommendedAndFallbackPaths() {
  struct RecommendedCase {
    std::uint32_t dim;
    std::uint32_t top_k;
    std::uint32_t query_tile;
    std::uint32_t blocks;
  };
  constexpr std::array<RecommendedCase, 3> cases = {
      RecommendedCase{128, 10, 4, 64},
      RecommendedCase{384, 10, 8, 160},
      RecommendedCase{384, 50, 4, 64},
  };
  for (const RecommendedCase& item : cases) {
    const VectorDatabase database = makeCoverageDatabase(
        item.dim, Metric::kCosine, DataType::kFloat32);
    const QuerySet queries = makeCoverageQueries(item.dim, DataType::kFloat32);
    SearchParams params;
    params.top_k = item.top_k;
    params.batch_size = 64;
    params.distance_mode = "warp";
    params.topk_mode = "fused";
    params.exact_query_tile = item.query_tile;
    params.exact_fused_blocks = item.blocks;
    compareExact(database, queries, params,
                 "recommended query-tile dim=" + std::to_string(item.dim) +
                     " K=" + std::to_string(item.top_k));
  }

  VectorDatabase database = makeCoverageDatabase(
      768, Metric::kCosine, DataType::kFloat32);
  QuerySet queries = makeCoverageQueries(768, DataType::kFloat32);
  SearchParams params;
  params.top_k = 10;
  params.batch_size = 64;
  params.distance_mode = "warp";
  params.topk_mode = "fused";
  params.exact_query_tile = 8;
  params.exact_fused_layout = "block_tile";
  requireThrows([&] { (void)GpuExactSearchEngine(database, queries, params); },
                "FP32 dim=768 block-tile must fall back instead of launching");

  params.exact_fused_layout = "warp_query";
  params.exact_query_tile = 1;
  compareExact(database, queries, params,
               "FP32 dim=768 safe warp-query fallback");

  params.top_k = 50;
  params.exact_query_tile = 8;
  requireThrows([&] { (void)GpuExactSearchEngine(database, queries, params); },
                "K=50 query-tile=8 must be rejected");
  params.exact_query_tile = 1;
  compareExact(database, queries, params,
               "K=50 query-tile=1 safe fallback");
}

IvfFlatIndex makePathologicalIndex(Metric metric, DataType dtype) {
  constexpr std::uint64_t num_vectors = 128;
  constexpr std::uint32_t dim = 65;
  constexpr std::uint32_t nlist = 8;
  const VectorDatabase database =
      makeCoverageDatabase(dim, metric, dtype, num_vectors);
  IvfFlatIndex index;
  index.num_vectors = num_vectors;
  index.dim = dim;
  index.nlist = nlist;
  index.dtype = dtype;
  index.metric = metric;
  index.offsets = {0, 0, 1, 1, 3, 3, 3, 3, num_vectors};
  index.ids.resize(num_vectors);
  for (std::uint64_t id = 0; id < num_vectors; ++id) {
    index.ids[id] = id;
  }
  index.values = database.values;
  index.half_values = database.half_values;
  index.centers.resize(static_cast<std::size_t>(nlist) * dim);
  for (std::uint32_t center = 0; center < nlist; ++center) {
    for (std::uint32_t d = 0; d < dim; ++d) {
      index.centers[static_cast<std::size_t>(center) * dim + d] =
          deterministicValue(center + 11U, d);
    }
  }
  validateIvfFlatIndex(index);
  return index;
}

void testEmptyAndSkewedIvfBuckets() {
  for (const DataType dtype : {DataType::kFloat32, DataType::kFloat16}) {
    for (const Metric metric :
         {Metric::kL2, Metric::kInnerProduct, Metric::kCosine}) {
      const IvfFlatIndex index = makePathologicalIndex(metric, dtype);
      const QuerySet queries = makeCoverageQueries(65, dtype);
      for (const std::uint32_t top_k : {1U, 10U, 50U, 100U}) {
        SearchParams params;
        params.search_mode = "ivf_flat";
        params.top_k = top_k;
        params.batch_size = 8;
        params.nlist = index.nlist;
        params.nprobe = index.nlist;
        params.distance_mode = "warp";
        const SearchResults cpu = cpuIvfFlatSearch(index, queries, params);
        std::string error;
        for (const std::string& mode : {"warp", "bucket_major"}) {
          params.distance_mode = mode;
          GpuIvfFlatSearchEngine engine(index, queries, params);
          const SearchResults gpu = engine.search(queries);
          require(resultsMatch(cpu, gpu, 5e-5F, &error),
                  "empty/skewed IVF " + mode + ": " + error);
        }
      }
    }
  }
  std::cout << "IVF empty/skewed buckets: 48 cases\n";
}

}  // namespace

int main() {
  try {
    testExactCoverageMatrix();
    testBatchCapacityAndNonWarpDimension();
    testExactRecommendedAndFallbackPaths();
    testEmptyAndSkewedIvfBuckets();
    std::cout << "Stage 8 stability coverage passed\n";
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    std::cerr << "Stage 8 stability failure: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
