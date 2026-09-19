#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace vector_search {

enum class DType : std::uint8_t {
  kFloat32 = 1,
  kFloat16 = 2,
};

enum class Metric : std::uint8_t {
  kL2 = 1,
  kInnerProduct = 2,
  kCosine = 3,
};

enum class SearchMode {
  kExact,
  kIvfFlat,
};

struct VectorDatabase {
  std::int64_t count = 0;
  std::int32_t dim = 0;
  DType source_dtype = DType::kFloat32;
  Metric metric = Metric::kL2;
  std::vector<float> values;
};

struct QuerySet {
  std::int64_t count = 0;
  std::int32_t dim = 0;
  DType source_dtype = DType::kFloat32;
  std::vector<float> values;
};

struct SearchConfig {
  int top_k = 10;
  SearchMode mode = SearchMode::kExact;
  int batch_size = 128;
  int nlist = 4096;
  int nprobe = 16;
  int pq_m = 16;
  int kmeans_iterations = 2;
  std::int64_t training_samples = 65536;
  int quality_queries = 10;
  std::uint64_t seed = 1234;
};

struct SearchResults {
  std::int64_t num_queries = 0;
  int top_k = 0;
  std::vector<std::int64_t> ids;
  std::vector<float> scores;
  std::vector<double> query_latency_ms;
  std::uint64_t evaluated_candidates = 0;
  std::uint64_t peak_candidates = 0;
  std::uint64_t working_set_bytes = 0;
  std::string backend;
};

struct IvfIndex {
  std::int64_t num_vectors = 0;
  std::int32_t dim = 0;
  int nlist = 0;
  Metric metric = Metric::kL2;
  std::vector<float> centers;
  std::vector<std::uint64_t> offsets;
  std::vector<std::int64_t> ids;
};

VectorDatabase load_vector_database(const std::string& path);
QuerySet load_queries(const std::string& path);
SearchConfig load_search_config(const std::string& path);

std::string to_string(DType dtype);
std::string to_string(Metric metric);
std::string to_string(SearchMode mode);

void save_ivf_index(const IvfIndex& index, const std::string& path);
IvfIndex load_ivf_index(const std::string& path);

SearchResults search_exact_cpu(const VectorDatabase& database,
                               const QuerySet& queries, int top_k,
                               int batch_size);
SearchResults search_candidates_cpu(
    const VectorDatabase& database, const QuerySet& queries,
    const std::vector<std::vector<std::int64_t>>& candidate_ids, int top_k,
    int batch_size);

bool cuda_backend_available();
SearchResults search_exact_cuda(const VectorDatabase& database,
                                const QuerySet& queries, int top_k,
                                int batch_size);
SearchResults search_candidates_cuda(
    const VectorDatabase& database, const QuerySet& queries,
    const std::vector<std::vector<std::int64_t>>& candidate_ids, int top_k,
    int batch_size);
std::vector<int> assign_nearest_cuda(const std::vector<float>& vectors,
                                     std::int64_t count, int dim,
                                     const std::vector<float>& centers,
                                     int nlist, Metric metric);

IvfIndex build_ivf_index(const VectorDatabase& database,
                         const SearchConfig& config, bool use_cuda);
SearchResults search_ivf(const VectorDatabase& database,
                         const QuerySet& queries, const IvfIndex& index,
                         const SearchConfig& config, bool use_cuda);

QuerySet slice_queries(const QuerySet& queries, std::int64_t count);
double recall_at_k(const SearchResults& approximate,
                   const SearchResults& ground_truth);
double rank_id_agreement(const SearchResults& candidate,
                         const SearchResults& ground_truth);
double mean_rank_score_error(const SearchResults& approximate,
                             const SearchResults& ground_truth);

void write_search_results(const SearchResults& results,
                          const std::string& path);

}  // namespace vector_search
