#pragma once

#include <cstdint>
#include <string>
#include <vector>

enum class DataType : std::uint32_t {
  kFloat32 = 1,
  kFloat16 = 2,
};

enum class Metric : std::uint32_t {
  kL2 = 1,
  kInnerProduct = 2,
  kCosine = 3,
};

struct VectorDatabase {
  std::uint64_t num_vectors = 0;
  std::uint32_t dim = 0;
  DataType dtype = DataType::kFloat32;
  Metric metric = Metric::kL2;
  std::vector<float> values;
  std::vector<std::uint16_t> half_values;
};

struct QuerySet {
  std::uint64_t num_queries = 0;
  std::uint32_t dim = 0;
  DataType dtype = DataType::kFloat32;
  std::vector<float> values;
  std::vector<std::uint16_t> half_values;
};

// Agent memory metadata is kept in original vector-ID order.  The IVF index
// may reorder vectors by bucket, so GPU kernels use index.ids[position] to
// address these columns without duplicating or rewriting the metadata file.
struct MemoryMetadata {
  std::uint64_t num_vectors = 0;
  std::vector<std::uint64_t> timestamps;
  std::vector<float> importance;
  std::vector<std::uint32_t> session_ids;
  std::vector<std::uint32_t> source_types;
};

// IVF-Flat 在桶内保留原始向量，只改变排列顺序。聚类中心始终使用 FP32，
// 桶内向量仍保持输入 dtype。
struct IvfFlatIndex {
  std::uint64_t num_vectors = 0;
  std::uint32_t dim = 0;
  std::uint32_t nlist = 0;
  DataType dtype = DataType::kFloat32;
  Metric metric = Metric::kL2;
  std::vector<float> centers;
  std::vector<std::uint64_t> offsets;
  std::vector<std::uint64_t> ids;
  std::vector<float> values;
  std::vector<std::uint16_t> half_values;
};

struct SearchParams {
  std::uint32_t top_k = 10;
  std::string search_mode = "exact";
  std::uint32_t batch_size = 8;
  std::string distance_mode = "warp";
  std::string topk_mode = "two_stage";
  // Exact fused experimental query tiling. A block reuses each database value
  // across this many queries; 1 preserves the established kernel.
  std::uint32_t exact_query_tile = 1;
  // warp_query is the established/query-tiled warp layout; block_tile assigns
  // one query per warp and reuses a shared database-vector tile.
  std::string exact_fused_layout = "warp_query";
  // Zero selects the occupancy heuristic; a positive value pins grid.x for
  // controlled Exact fused chunk-count sweeps.
  std::uint32_t exact_fused_blocks = 0;
  std::uint32_t nlist = 4096;
  std::uint32_t nprobe = 16;
  std::string nprobe_policy = "fixed";
  std::string adaptive_execution = "masked";
  std::uint32_t adaptive_nprobe_min = 1;
  std::uint32_t adaptive_nprobe_step = 1;
  float adaptive_target_mass = 0.9F;
  float adaptive_temperature = 0.2F;
  std::string memory_mode = "disabled";
  float memory_semantic_weight = 1.0F;
  float memory_importance_weight = 0.0F;
  float memory_recency_weight = 0.0F;
  float memory_time_scale = 1.0F;
  std::uint64_t memory_now = 0;
  std::uint64_t filter_min_timestamp = 0;
  std::uint32_t filter_session_id = UINT32_MAX;
  std::uint32_t filter_source_type = UINT32_MAX;
  std::uint32_t memory_rerank_factor = 4;
  std::uint32_t pq_m = 16;
};

struct Neighbor {
  std::uint64_t id = 0;
  float score = 0.0F;
};

using SearchResults = std::vector<std::vector<Neighbor>>;

struct SearchStats {
  double host_selection_ms = 0.0;
  double center_selection_ms = 0.0;
  double adaptive_policy_ms = 0.0;
  double database_h2d_ms = 0.0;
  double query_h2d_ms = 0.0;
  double distance_kernel_ms = 0.0;
  double topk_kernel_ms = 0.0;
  double memory_rerank_ms = 0.0;
  double result_d2h_ms = 0.0;
  std::vector<double> batch_latency_ms;
  std::uint64_t selected_probe_sum = 0;
  std::uint64_t selected_probe_queries = 0;
  std::uint32_t selected_probe_min = 0;
  std::uint32_t selected_probe_max = 0;
  std::vector<std::uint32_t> selected_probe_counts;
};

const char* metricName(Metric metric);
const char* dataTypeName(DataType dtype);
Metric parseMetric(const std::string& text);
