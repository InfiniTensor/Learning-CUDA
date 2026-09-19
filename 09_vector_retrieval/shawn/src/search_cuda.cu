#include "vector_search/search.hpp"

#include <cub/device/device_radix_sort.cuh>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace vector_search {
namespace {

constexpr int kThreads = 256;

void check_cuda(cudaError_t status, const char* expression) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(expression) + ": " +
                             cudaGetErrorString(status));
  }
}

#define VECTOR_CUDA_CHECK(expression) check_cuda((expression), #expression)

template <typename T>
class DeviceBuffer {
 public:
  DeviceBuffer() = default;
  explicit DeviceBuffer(std::size_t count) { allocate(count); }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  ~DeviceBuffer() {
    if (pointer_ != nullptr) cudaFree(pointer_);
  }

  void allocate(std::size_t count) {
    if (pointer_ != nullptr) cudaFree(pointer_);
    pointer_ = nullptr;
    if (count > 0) {
      VECTOR_CUDA_CHECK(
          cudaMalloc(reinterpret_cast<void**>(&pointer_), count * sizeof(T)));
    }
  }
  T* get() { return pointer_; }
  const T* get() const { return pointer_; }

 private:
  T* pointer_ = nullptr;
};

__device__ float vector_score(const float* query, const float* vector, int dim,
                              int metric) {
  // 与 CPU 保持相同度量定义；零范数向量的 cosine 记为 0。
  if (metric == static_cast<int>(Metric::kL2)) {
    float distance = 0.0f;
    for (int component = 0; component < dim; ++component) {
      const float difference = query[component] - vector[component];
      distance += difference * difference;
    }
    return distance;
  }
  float dot = 0.0f;
  float query_norm = 0.0f;
  float vector_norm = 0.0f;
  for (int component = 0; component < dim; ++component) {
    dot += query[component] * vector[component];
    if (metric == static_cast<int>(Metric::kCosine)) {
      query_norm += query[component] * query[component];
      vector_norm += vector[component] * vector[component];
    }
  }
  if (metric == static_cast<int>(Metric::kCosine)) {
    const float denominator = sqrtf(query_norm * vector_norm);
    return denominator > 0.0f ? dot / denominator : 0.0f;
  }
  return dot;
}

__global__ void score_candidates_kernel(
    const float* database, std::int64_t num_vectors, int dim,
    const float* query, const std::int64_t* candidate_ids,
    std::int64_t candidate_count, int metric, float* scores,
    std::int64_t* ids) {
  // 逐候选计算 score/id，随后由 CUB 全排序取 Top-K。
  const std::int64_t thread_id =
      static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::int64_t stride =
      static_cast<std::int64_t>(gridDim.x) * blockDim.x;
  for (std::int64_t offset = thread_id; offset < candidate_count;
       offset += stride) {
    const std::int64_t vector_id =
        candidate_ids == nullptr ? offset : candidate_ids[offset];
    if (vector_id < 0 || vector_id >= num_vectors) continue;
    scores[offset] = vector_score(query, database + vector_id * dim, dim, metric);
    ids[offset] = vector_id;
  }
}

__global__ void assign_nearest_kernel(const float* vectors,
                                      std::int64_t count, int dim,
                                      const float* centers, int nlist,
                                      int metric, int* assignments) {
  // 扫描全部粗聚类中心，确定每个向量所属的倒排表。
  const std::int64_t thread_id =
      static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::int64_t stride =
      static_cast<std::int64_t>(gridDim.x) * blockDim.x;
  for (std::int64_t vector_id = thread_id; vector_id < count;
       vector_id += stride) {
    const float* vector = vectors + vector_id * dim;
    int best_list = 0;
    float best_score = vector_score(vector, centers, dim, metric);
    for (int list = 1; list < nlist; ++list) {
      const float score =
          vector_score(vector, centers + static_cast<std::int64_t>(list) * dim,
                       dim, metric);
      const bool is_better = metric == static_cast<int>(Metric::kL2)
                                 ? score < best_score
                                 : score > best_score;
      if (is_better) {
        best_score = score;
        best_list = list;
      }
    }
    assignments[vector_id] = best_list;
  }
}

void validate_search(const VectorDatabase& database, const QuerySet& queries,
                     int top_k, int batch_size) {
  if (database.count <= 0 || database.dim <= 0 || queries.count <= 0 ||
      queries.dim != database.dim || top_k <= 0 || top_k > 100 ||
      top_k > database.count || batch_size <= 0 ||
      database.count > std::numeric_limits<int>::max()) {
    throw std::invalid_argument("invalid CUDA search dimensions");
  }
}

SearchResults search_cuda_impl(
    const VectorDatabase& database, const QuerySet& queries,
    const std::vector<std::vector<std::int64_t>>* candidate_ids, int top_k,
    int batch_size) {
  // 数据库在本次调用内常驻显存，查询逐条执行；batch_size 仅分组，不并发。
  validate_search(database, queries, top_k, batch_size);
  if (candidate_ids != nullptr &&
      candidate_ids->size() != static_cast<std::size_t>(queries.count)) {
    throw std::invalid_argument("candidate-list count must match queries");
  }

  std::size_t max_candidates = static_cast<std::size_t>(database.count);
  if (candidate_ids != nullptr) {
    max_candidates = 0;
    for (const auto& candidates : *candidate_ids) {
      max_candidates = std::max(max_candidates, candidates.size());
    }
    if (max_candidates == 0) {
      throw std::invalid_argument("every CUDA candidate search is empty");
    }
  }

  DeviceBuffer<float> device_database(database.values.size());
  DeviceBuffer<float> device_query(static_cast<std::size_t>(database.dim));
  DeviceBuffer<std::int64_t> device_candidates(
      candidate_ids == nullptr ? 0 : max_candidates);
  DeviceBuffer<float> scores_in(max_candidates);
  DeviceBuffer<float> scores_out(max_candidates);
  DeviceBuffer<std::int64_t> ids_in(max_candidates);
  DeviceBuffer<std::int64_t> ids_out(max_candidates);
  VECTOR_CUDA_CHECK(cudaMemcpy(device_database.get(), database.values.data(),
                               database.values.size() * sizeof(float),
                               cudaMemcpyHostToDevice));

  std::size_t temp_bytes = 0;
  const int maximum_items = static_cast<int>(max_candidates);
  if (database.metric == Metric::kL2) {
    VECTOR_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
        nullptr, temp_bytes, scores_in.get(), scores_out.get(), ids_in.get(),
        ids_out.get(), maximum_items));
  } else {
    VECTOR_CUDA_CHECK(cub::DeviceRadixSort::SortPairsDescending(
        nullptr, temp_bytes, scores_in.get(), scores_out.get(), ids_in.get(),
        ids_out.get(), maximum_items));
  }
  DeviceBuffer<std::uint8_t> temporary(temp_bytes);

  SearchResults results;
  results.num_queries = queries.count;
  results.top_k = top_k;
  results.ids.assign(static_cast<std::size_t>(queries.count) * top_k, -1);
  const float missing_score = database.metric == Metric::kL2
                                  ? std::numeric_limits<float>::infinity()
                                  : -std::numeric_limits<float>::infinity();
  results.scores.assign(results.ids.size(), missing_score);
  results.query_latency_ms.resize(static_cast<std::size_t>(queries.count));
  results.backend = candidate_ids == nullptr ? "cuda-exact" : "cuda-candidates";
  results.peak_candidates = static_cast<std::uint64_t>(max_candidates);
  results.working_set_bytes =
      static_cast<std::uint64_t>(database.values.size() * sizeof(float)) +
      static_cast<std::uint64_t>(database.dim * sizeof(float)) +
      static_cast<std::uint64_t>(2 * max_candidates * sizeof(float)) +
      static_cast<std::uint64_t>(2 * max_candidates * sizeof(std::int64_t)) +
      static_cast<std::uint64_t>(temp_bytes) +
      (candidate_ids == nullptr
           ? 0
           : static_cast<std::uint64_t>(max_candidates *
                                        sizeof(std::int64_t)));

  for (std::int64_t batch_begin = 0; batch_begin < queries.count;
       batch_begin += batch_size) {
    const std::int64_t batch_end =
        std::min<std::int64_t>(queries.count, batch_begin + batch_size);
    for (std::int64_t query_index = batch_begin; query_index < batch_end;
         ++query_index) {
      const auto start = std::chrono::steady_clock::now();
      const float* query =
          queries.values.data() + query_index * queries.dim;
      VECTOR_CUDA_CHECK(cudaMemcpy(device_query.get(), query,
                                   database.dim * sizeof(float),
                                   cudaMemcpyHostToDevice));

      const std::int64_t candidate_count =
          candidate_ids == nullptr
              ? database.count
              : static_cast<std::int64_t>(
                    (*candidate_ids)[static_cast<std::size_t>(query_index)]
                        .size());
      if (candidate_count <= 0 || candidate_count > maximum_items) {
        throw std::runtime_error("invalid per-query candidate count");
      }
      if (candidate_ids != nullptr) {
        const auto& host_candidates =
            (*candidate_ids)[static_cast<std::size_t>(query_index)];
        VECTOR_CUDA_CHECK(cudaMemcpy(
            device_candidates.get(), host_candidates.data(),
            host_candidates.size() * sizeof(std::int64_t),
            cudaMemcpyHostToDevice));
      }
      const int blocks = std::min<int>(
          static_cast<int>((candidate_count + kThreads - 1) / kThreads), 65535);
      score_candidates_kernel<<<blocks, kThreads>>>(
          device_database.get(), database.count, database.dim,
          device_query.get(),
          candidate_ids == nullptr ? nullptr : device_candidates.get(),
          candidate_count, static_cast<int>(database.metric), scores_in.get(),
          ids_in.get());
      VECTOR_CUDA_CHECK(cudaGetLastError());

      if (database.metric == Metric::kL2) {
        // L2 越小越好；IP/Cosine 越大越好，因此 CUB 排序方向不同。
        VECTOR_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
            temporary.get(), temp_bytes, scores_in.get(), scores_out.get(),
            ids_in.get(), ids_out.get(), static_cast<int>(candidate_count)));
      } else {
        VECTOR_CUDA_CHECK(cub::DeviceRadixSort::SortPairsDescending(
            temporary.get(), temp_bytes, scores_in.get(), scores_out.get(),
            ids_in.get(), ids_out.get(), static_cast<int>(candidate_count)));
      }
      const int returned =
          std::min<int>(top_k, static_cast<int>(candidate_count));
      const std::size_t output =
          static_cast<std::size_t>(query_index) * top_k;
      VECTOR_CUDA_CHECK(cudaMemcpy(results.ids.data() + output, ids_out.get(),
                                   returned * sizeof(std::int64_t),
                                   cudaMemcpyDeviceToHost));
      VECTOR_CUDA_CHECK(cudaMemcpy(results.scores.data() + output,
                                   scores_out.get(), returned * sizeof(float),
                                   cudaMemcpyDeviceToHost));
      const auto end = std::chrono::steady_clock::now();
      results.query_latency_ms[static_cast<std::size_t>(query_index)] =
          std::chrono::duration<double, std::milli>(end - start).count();
      results.evaluated_candidates +=
          static_cast<std::uint64_t>(candidate_count);
    }
  }
  return results;
}

}  // namespace

bool cuda_backend_available() {
  int count = 0;
  const cudaError_t status = cudaGetDeviceCount(&count);
  if (status != cudaSuccess) {
    cudaGetLastError();
    return false;
  }
  return count > 0;
}

SearchResults search_exact_cuda(const VectorDatabase& database,
                                const QuerySet& queries, int top_k,
                                int batch_size) {
  return search_cuda_impl(database, queries, nullptr, top_k, batch_size);
}

SearchResults search_candidates_cuda(
    const VectorDatabase& database, const QuerySet& queries,
    const std::vector<std::vector<std::int64_t>>& candidate_ids, int top_k,
    int batch_size) {
  return search_cuda_impl(database, queries, &candidate_ids, top_k, batch_size);
}

std::vector<int> assign_nearest_cuda(const std::vector<float>& vectors,
                                     std::int64_t count, int dim,
                                     const std::vector<float>& centers,
                                     int nlist, Metric metric) {
  // 上传向量和中心，完成分配后取回倒排表编号。
  if (count <= 0 || dim <= 0 || nlist <= 0 ||
      vectors.size() != static_cast<std::size_t>(count) * dim ||
      centers.size() != static_cast<std::size_t>(nlist) * dim) {
    throw std::invalid_argument("invalid CUDA IVF assignment dimensions");
  }
  DeviceBuffer<float> device_vectors(vectors.size());
  DeviceBuffer<float> device_centers(centers.size());
  DeviceBuffer<int> device_assignments(static_cast<std::size_t>(count));
  VECTOR_CUDA_CHECK(cudaMemcpy(device_vectors.get(), vectors.data(),
                               vectors.size() * sizeof(float),
                               cudaMemcpyHostToDevice));
  VECTOR_CUDA_CHECK(cudaMemcpy(device_centers.get(), centers.data(),
                               centers.size() * sizeof(float),
                               cudaMemcpyHostToDevice));
  const int blocks = std::min<int>(
      static_cast<int>((count + kThreads - 1) / kThreads), 65535);
  assign_nearest_kernel<<<blocks, kThreads>>>(
      device_vectors.get(), count, dim, device_centers.get(), nlist,
      static_cast<int>(metric), device_assignments.get());
  VECTOR_CUDA_CHECK(cudaGetLastError());
  std::vector<int> assignments(static_cast<std::size_t>(count));
  VECTOR_CUDA_CHECK(cudaMemcpy(assignments.data(), device_assignments.get(),
                               assignments.size() * sizeof(int),
                               cudaMemcpyDeviceToHost));
  return assignments;
}

}  // namespace vector_search
