#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <functional>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "file_io.h"
#include "ivf_flat.h"

namespace {

void checkCuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(status));
  }
}

class DeviceBuffer {
 public:
  explicit DeviceBuffer(std::size_t bytes) {
    checkCuda(cudaMalloc(&data_, bytes), "cudaMalloc IVF");
  }
  ~DeviceBuffer() {
    if (data_ != nullptr) {
      cudaFree(data_);
    }
  }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  void* data() { return data_; }

 private:
  void* data_ = nullptr;
};

class CudaEvent {
 public:
  CudaEvent() { checkCuda(cudaEventCreate(&event_), "cudaEventCreate IVF"); }
  ~CudaEvent() { cudaEventDestroy(event_); }
  CudaEvent(const CudaEvent&) = delete;
  CudaEvent& operator=(const CudaEvent&) = delete;
  void record() { checkCuda(cudaEventRecord(event_), "cudaEventRecord IVF"); }
  void synchronize() {
    checkCuda(cudaEventSynchronize(event_), "cudaEventSynchronize IVF");
  }
  cudaEvent_t get() const { return event_; }

 private:
  cudaEvent_t event_{};
};

double elapsedMilliseconds(const CudaEvent& start, const CudaEvent& end) {
  float milliseconds = 0.0F;
  checkCuda(cudaEventElapsedTime(&milliseconds, start.get(), end.get()),
            "cudaEventElapsedTime IVF");
  return milliseconds;
}

__device__ __forceinline__ float loadInput(const float* values,
                                           std::uint64_t index) {
  return values[index];
}

__device__ __forceinline__ float loadInput(const __half* values,
                                           std::uint64_t index) {
  return __half2float(values[index]);
}

__device__ __forceinline__ float warpReduceSum(float value) {
  for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xFFFFFFFFU, value, offset);
  }
  return value;
}

__device__ bool isBetter(float candidate_score, std::uint64_t candidate_id,
                         float current_score, std::uint64_t current_id,
                         Metric metric) {
  if (candidate_score == current_score) {
    return candidate_id < current_id;
  }
  return metric == Metric::kL2 ? candidate_score < current_score
                               : candidate_score > current_score;
}

struct DeviceMemoryView {
  const std::uint64_t* timestamps = nullptr;
  const float* importance = nullptr;
  const std::uint32_t* session_ids = nullptr;
  const std::uint32_t* source_types = nullptr;
};

struct DeviceMemoryOptions {
  float semantic_weight = 1.0F;
  float importance_weight = 0.0F;
  float recency_weight = 0.0F;
  float time_scale = 1.0F;
  std::uint64_t now = 0;
  std::uint64_t min_timestamp = 0;
  std::uint32_t session_id = UINT32_MAX;
  std::uint32_t source_type = UINT32_MAX;
};

__device__ __forceinline__ bool memoryCandidateMatches(
    DeviceMemoryView metadata, std::uint64_t vector_id,
    DeviceMemoryOptions options) {
  return metadata.timestamps[vector_id] >= options.min_timestamp &&
         (options.session_id == UINT32_MAX ||
          metadata.session_ids[vector_id] == options.session_id) &&
         (options.source_type == UINT32_MAX ||
          metadata.source_types[vector_id] == options.source_type);
}

template <Metric MetricValue>
__device__ __forceinline__ float applyMemoryScore(float semantic_score,
                                                  std::uint64_t vector_id,
                                                  DeviceMemoryView metadata,
                                                  DeviceMemoryOptions options) {
  float recency = 0.0F;
  if (options.recency_weight != 0.0F && options.now != 0) {
    const std::uint64_t timestamp = metadata.timestamps[vector_id];
    const double age = timestamp >= options.now
                           ? 0.0
                           : static_cast<double>(options.now - timestamp);
    recency = expf(-static_cast<float>(age / options.time_scale));
  }
  const float boost =
      options.importance_weight * metadata.importance[vector_id] +
      options.recency_weight * recency;
  const float weighted_semantic = options.semantic_weight * semantic_score;
  return MetricValue == Metric::kL2 ? weighted_semantic - boost
                                    : weighted_semantic + boost;
}

template <typename InputType>
__global__ void ivfCenterScoresKernel(const float* centers,
                                      const InputType* queries,
                                      float* center_scores, std::uint32_t nlist,
                                      std::uint32_t dim,
                                      std::uint32_t batch_count,
                                      Metric metric) {
  const std::uint32_t center_id = blockIdx.x * blockDim.x + threadIdx.x;
  const std::uint32_t query_id = blockIdx.y;
  if (center_id >= nlist || query_id >= batch_count) {
    return;
  }
  const std::uint64_t center_offset =
      static_cast<std::uint64_t>(center_id) * dim;
  const std::uint64_t query_offset = static_cast<std::uint64_t>(query_id) * dim;
  float dot = 0.0F;
  float center_norm = 0.0F;
  float query_norm = 0.0F;
  float squared_l2 = 0.0F;
  for (std::uint32_t d = 0; d < dim; ++d) {
    const float query = loadInput(queries, query_offset + d);
    const float center = centers[center_offset + d];
    if (metric == Metric::kL2) {
      const float difference = query - center;
      squared_l2 = fmaf(difference, difference, squared_l2);
    } else {
      dot = fmaf(query, center, dot);
      if (metric == Metric::kCosine) {
        query_norm = fmaf(query, query, query_norm);
        center_norm = fmaf(center, center, center_norm);
      }
    }
  }
  float score = dot;
  if (metric == Metric::kL2) {
    score = squared_l2;
  } else if (metric == Metric::kCosine) {
    score = query_norm == 0.0F || center_norm == 0.0F
                ? 0.0F
                : dot / sqrtf(query_norm * center_norm);
  }
  center_scores[static_cast<std::uint64_t>(query_id) * nlist + center_id] =
      score;
}

// 并行选择 nprobe：每轮让 256 个线程共同从剩余中心中归约出最佳中心，
// 选中后把该分数改为哨兵，再进行下一轮。
__global__ void ivfSelectProbesBlockKernel(float* center_scores,
                                           std::uint32_t* selected_centers,
                                           float* selected_scores,
                                           std::uint32_t nlist,
                                           std::uint32_t nprobe,
                                           Metric metric) {
  constexpr std::uint32_t kThreads = 256;
  constexpr float kFloatMaximum = 3.402823466e+38F;
  __shared__ float shared_scores[kThreads];
  __shared__ std::uint32_t shared_ids[kThreads];
  const std::uint32_t thread_id = threadIdx.x;
  const std::uint32_t query_id = blockIdx.x;
  const float worst_score =
      metric == Metric::kL2 ? kFloatMaximum : -kFloatMaximum;
  float* query_scores =
      center_scores + static_cast<std::uint64_t>(query_id) * nlist;

  for (std::uint32_t probe = 0; probe < nprobe; ++probe) {
    float local_best_score = worst_score;
    std::uint32_t local_best_id = UINT32_MAX;
    for (std::uint32_t center_id = thread_id; center_id < nlist;
         center_id += blockDim.x) {
      const float score = query_scores[center_id];
      if (isBetter(score, center_id, local_best_score, local_best_id, metric)) {
        local_best_score = score;
        local_best_id = center_id;
      }
    }
    shared_scores[thread_id] = local_best_score;
    shared_ids[thread_id] = local_best_id;
    __syncthreads();
    for (std::uint32_t stride = kThreads / 2; stride > 0; stride >>= 1) {
      if (thread_id < stride &&
          isBetter(shared_scores[thread_id + stride],
                   shared_ids[thread_id + stride], shared_scores[thread_id],
                   shared_ids[thread_id], metric)) {
        shared_scores[thread_id] = shared_scores[thread_id + stride];
        shared_ids[thread_id] = shared_ids[thread_id + stride];
      }
      __syncthreads();
    }
    if (thread_id == 0) {
      const std::uint32_t selected_id = shared_ids[0];
      selected_centers[static_cast<std::uint64_t>(query_id) * nprobe + probe] =
          selected_id;
      if (selected_scores != nullptr) {
        selected_scores[static_cast<std::uint64_t>(query_id) * nprobe + probe] =
            shared_scores[0];
      }
      query_scores[selected_id] = worst_score;
    }
    __syncthreads();
  }
}

// 将已按优劣排序的中心分数转换为概率质量。分数差先除以当前 query 的
// top-1 到第 max_nprobe 个中心的跨度，使 temperature 对查询模长和 metric
// 尺度不敏感。集中分布会较早达到 target_mass，使用较少 probe；平坦分布
// 会自动接近 nprobe 上限。
__global__ void chooseAdaptiveNprobeKernel(
    const float* selected_scores, std::uint32_t* query_nprobes,
    std::uint32_t batch_count, std::uint32_t max_nprobe,
    std::uint32_t min_nprobe, std::uint32_t step, float target_mass,
    float temperature, Metric metric) {
  const std::uint32_t query_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (query_id >= batch_count) {
    return;
  }
  const float* scores =
      selected_scores + static_cast<std::uint64_t>(query_id) * max_nprobe;
  const float direction = metric == Metric::kL2 ? -1.0F : 1.0F;
  const float best_utility = direction * scores[0];
  const float worst_utility = direction * scores[max_nprobe - 1];
  const float utility_span = fmaxf(best_utility - worst_utility, 1.0e-6F);
  const float inverse_scale = 1.0F / (temperature * utility_span);
  float total_mass = 0.0F;
  for (std::uint32_t probe = 0; probe < max_nprobe; ++probe) {
    total_mass +=
        expf((direction * scores[probe] - best_utility) * inverse_scale);
  }

  std::uint32_t selected_nprobe = max_nprobe;
  if (target_mass < 1.0F) {
    float prefix_mass = 0.0F;
    for (std::uint32_t probe = 0; probe < max_nprobe; ++probe) {
      prefix_mass +=
          expf((direction * scores[probe] - best_utility) * inverse_scale);
      const std::uint32_t count = probe + 1;
      const bool selectable =
          count >= min_nprobe &&
          (((count - min_nprobe) % step) == 0 || count == max_nprobe);
      if (selectable && prefix_mass >= target_mass * total_mass) {
        selected_nprobe = count;
        break;
      }
    }
  }
  query_nprobes[query_id] = selected_nprobe;
}

// 正确性 baseline：每个 query 使用一个 CUDA 线程，直接遍历选中桶的候选，
// 距离计算和有序 Top-K 在同一个 kernel 中完成。索引常驻显存，避免复制向量。
// 下一版会让一个 block/多个 warp 共同扫描一个 query 的候选。
template <typename InputType>
__global__ void fusedIvfScanTopKKernel(
    const InputType* vectors, const std::uint64_t* original_ids,
    const InputType* queries, const std::uint64_t* candidate_positions,
    const std::uint64_t* candidate_counts, float* top_scores,
    std::uint64_t* top_ids, std::uint64_t candidate_stride, std::uint32_t dim,
    std::uint32_t top_k, Metric metric) {
  if (threadIdx.x != 0) {
    return;
  }
  constexpr std::uint32_t kMaximumK = 100;
  constexpr float kFloatMaximum = 3.402823466e+38F;
  const float worst_score =
      metric == Metric::kL2 ? kFloatMaximum : -kFloatMaximum;
  float best_scores[kMaximumK];
  std::uint64_t best_ids[kMaximumK];
  for (std::uint32_t rank = 0; rank < top_k; ++rank) {
    best_scores[rank] = worst_score;
    best_ids[rank] = UINT64_MAX;
  }

  const std::uint32_t query_id = blockIdx.x;
  const std::uint64_t query_offset = static_cast<std::uint64_t>(query_id) * dim;
  float query_norm = 0.0F;
  if (metric == Metric::kCosine) {
    for (std::uint32_t d = 0; d < dim; ++d) {
      const float query = loadInput(queries, query_offset + d);
      query_norm = fmaf(query, query, query_norm);
    }
  }

  const std::uint64_t positions_offset =
      static_cast<std::uint64_t>(query_id) * candidate_stride;
  for (std::uint64_t candidate = 0; candidate < candidate_counts[query_id];
       ++candidate) {
    const std::uint64_t position =
        candidate_positions[positions_offset + candidate];
    const std::uint64_t vector_offset = position * dim;
    float dot = 0.0F;
    float vector_norm = 0.0F;
    float squared_l2 = 0.0F;
    for (std::uint32_t d = 0; d < dim; ++d) {
      const float query = loadInput(queries, query_offset + d);
      const float vector = loadInput(vectors, vector_offset + d);
      if (metric == Metric::kL2) {
        const float difference = query - vector;
        squared_l2 = fmaf(difference, difference, squared_l2);
      } else {
        dot = fmaf(query, vector, dot);
        if (metric == Metric::kCosine) {
          vector_norm = fmaf(vector, vector, vector_norm);
        }
      }
    }
    float score = dot;
    if (metric == Metric::kL2) {
      score = squared_l2;
    } else if (metric == Metric::kCosine) {
      score = query_norm == 0.0F || vector_norm == 0.0F
                  ? 0.0F
                  : dot / sqrtf(query_norm * vector_norm);
    }
    const std::uint64_t vector_id = original_ids[position];
    for (std::uint32_t rank = 0; rank < top_k; ++rank) {
      if (isBetter(score, vector_id, best_scores[rank], best_ids[rank],
                   metric)) {
        for (std::uint32_t shift = top_k - 1; shift > rank; --shift) {
          best_scores[shift] = best_scores[shift - 1];
          best_ids[shift] = best_ids[shift - 1];
        }
        best_scores[rank] = score;
        best_ids[rank] = vector_id;
        break;
      }
    }
  }

  const std::uint64_t output_offset =
      static_cast<std::uint64_t>(query_id) * top_k;
  for (std::uint32_t rank = 0; rank < top_k; ++rank) {
    top_scores[output_offset + rank] = best_scores[rank];
    top_ids[output_offset + rank] = best_ids[rank];
  }
}

// 并行版：一个 block 处理一个 query。每个线程扫描一部分候选并维护无序
// 局部 Top-K，随后 block 用共享内存归约出最终 K 个结果。
template <typename InputType>
__global__ void fusedIvfBlockScanTopKKernel(
    const InputType* vectors, const std::uint64_t* original_ids,
    const InputType* queries, const std::uint64_t* candidate_positions,
    const std::uint64_t* candidate_counts, float* top_scores,
    std::uint64_t* top_ids, std::uint64_t candidate_stride, std::uint32_t dim,
    std::uint32_t top_k, Metric metric) {
  constexpr std::uint32_t kThreads = 256;
  constexpr std::uint32_t kMaximumK = 10;
  constexpr float kFloatMaximum = 3.402823466e+38F;
  __shared__ float shared_scores[kThreads];
  __shared__ std::uint64_t shared_ids[kThreads];

  const std::uint32_t thread_id = threadIdx.x;
  const std::uint32_t query_id = blockIdx.x;
  const float worst_score =
      metric == Metric::kL2 ? kFloatMaximum : -kFloatMaximum;
  float local_scores[kMaximumK];
  std::uint64_t local_ids[kMaximumK];
  for (std::uint32_t rank = 0; rank < top_k; ++rank) {
    local_scores[rank] = worst_score;
    local_ids[rank] = UINT64_MAX;
  }
  std::uint32_t local_worst_rank = 0;

  const std::uint64_t query_offset = static_cast<std::uint64_t>(query_id) * dim;
  float query_norm = 0.0F;
  if (metric == Metric::kCosine) {
    for (std::uint32_t d = 0; d < dim; ++d) {
      const float query = loadInput(queries, query_offset + d);
      query_norm = fmaf(query, query, query_norm);
    }
  }
  const std::uint64_t positions_offset =
      static_cast<std::uint64_t>(query_id) * candidate_stride;
  for (std::uint64_t candidate = thread_id;
       candidate < candidate_counts[query_id]; candidate += blockDim.x) {
    const std::uint64_t position =
        candidate_positions[positions_offset + candidate];
    const std::uint64_t vector_offset = position * dim;
    float dot = 0.0F;
    float vector_norm = 0.0F;
    float squared_l2 = 0.0F;
    for (std::uint32_t d = 0; d < dim; ++d) {
      const float query = loadInput(queries, query_offset + d);
      const float vector = loadInput(vectors, vector_offset + d);
      if (metric == Metric::kL2) {
        const float difference = query - vector;
        squared_l2 = fmaf(difference, difference, squared_l2);
      } else {
        dot = fmaf(query, vector, dot);
        if (metric == Metric::kCosine) {
          vector_norm = fmaf(vector, vector, vector_norm);
        }
      }
    }
    float score = dot;
    if (metric == Metric::kL2) {
      score = squared_l2;
    } else if (metric == Metric::kCosine) {
      score = query_norm == 0.0F || vector_norm == 0.0F
                  ? 0.0F
                  : dot / sqrtf(query_norm * vector_norm);
    }
    const std::uint64_t vector_id = original_ids[position];
    if (isBetter(score, vector_id, local_scores[local_worst_rank],
                 local_ids[local_worst_rank], metric)) {
      local_scores[local_worst_rank] = score;
      local_ids[local_worst_rank] = vector_id;
      local_worst_rank = 0;
      for (std::uint32_t rank = 1; rank < top_k; ++rank) {
        if (isBetter(local_scores[local_worst_rank],
                     local_ids[local_worst_rank], local_scores[rank],
                     local_ids[rank], metric)) {
          local_worst_rank = rank;
        }
      }
    }
  }

  for (std::uint32_t output_rank = 0; output_rank < top_k; ++output_rank) {
    std::uint32_t local_best_rank = 0;
    for (std::uint32_t rank = 1; rank < top_k; ++rank) {
      if (isBetter(local_scores[rank], local_ids[rank],
                   local_scores[local_best_rank], local_ids[local_best_rank],
                   metric)) {
        local_best_rank = rank;
      }
    }
    shared_scores[thread_id] = local_scores[local_best_rank];
    shared_ids[thread_id] = local_ids[local_best_rank];
    __syncthreads();
    for (std::uint32_t stride = kThreads / 2; stride > 0; stride >>= 1) {
      if (thread_id < stride &&
          isBetter(shared_scores[thread_id + stride],
                   shared_ids[thread_id + stride], shared_scores[thread_id],
                   shared_ids[thread_id], metric)) {
        shared_scores[thread_id] = shared_scores[thread_id + stride];
        shared_ids[thread_id] = shared_ids[thread_id + stride];
      }
      __syncthreads();
    }
    if (thread_id == 0) {
      const std::uint64_t output_offset =
          static_cast<std::uint64_t>(query_id) * top_k + output_rank;
      top_scores[output_offset] = shared_scores[0];
      top_ids[output_offset] = shared_ids[0];
    }
    const std::uint64_t selected_id = shared_ids[0];
    if (local_ids[local_best_rank] == selected_id) {
      local_scores[local_best_rank] = worst_score;
      local_ids[local_best_rank] = UINT64_MAX;
    }
    __syncthreads();
  }
}

template <typename InputType, std::uint32_t MaximumK>
__global__ void ivfScanBucketsScalarKernel(
    const InputType* vectors, const std::uint64_t* original_ids,
    const std::uint64_t* bucket_offsets, const std::uint32_t* selected_centers,
    const InputType* queries, float* probe_scores, std::uint64_t* probe_ids,
    const std::uint32_t* active_query_ids, const std::uint32_t* query_nprobes,
    std::uint32_t nprobe, std::uint32_t dim, std::uint32_t top_k, Metric metric,
    bool apply_memory, DeviceMemoryView metadata,
    DeviceMemoryOptions memory_options) {
  constexpr std::uint32_t kThreads = 256;
  constexpr float kFloatMaximum = 3.402823466e+38F;
  __shared__ float shared_scores[kThreads];
  __shared__ std::uint64_t shared_ids[kThreads];

  const std::uint32_t thread_id = threadIdx.x;
  const std::uint32_t probe = blockIdx.x;
  const std::uint32_t query_id =
      active_query_ids == nullptr ? blockIdx.y : active_query_ids[blockIdx.y];
  if (query_nprobes != nullptr && probe >= query_nprobes[query_id]) {
    return;
  }
  const std::uint32_t center_id =
      selected_centers[static_cast<std::uint64_t>(query_id) * nprobe + probe];
  const std::uint64_t bucket_start = bucket_offsets[center_id];
  const std::uint64_t bucket_end = bucket_offsets[center_id + 1];
  const float worst_score =
      metric == Metric::kL2 ? kFloatMaximum : -kFloatMaximum;
  float local_scores[MaximumK];
  std::uint64_t local_ids[MaximumK];
  for (std::uint32_t rank = 0; rank < top_k; ++rank) {
    local_scores[rank] = worst_score;
    local_ids[rank] = UINT64_MAX;
  }
  std::uint32_t local_worst_rank = 0;

  const std::uint64_t query_offset = static_cast<std::uint64_t>(query_id) * dim;
  float query_norm = 0.0F;
  if (metric == Metric::kCosine) {
    for (std::uint32_t d = 0; d < dim; ++d) {
      const float query = loadInput(queries, query_offset + d);
      query_norm = fmaf(query, query, query_norm);
    }
  }
  for (std::uint64_t position = bucket_start + thread_id; position < bucket_end;
       position += blockDim.x) {
    const std::uint64_t vector_id = original_ids[position];
    if (apply_memory &&
        !memoryCandidateMatches(metadata, vector_id, memory_options)) {
      continue;
    }
    const std::uint64_t vector_offset = position * dim;
    float dot = 0.0F;
    float vector_norm = 0.0F;
    float squared_l2 = 0.0F;
    for (std::uint32_t d = 0; d < dim; ++d) {
      const float query = loadInput(queries, query_offset + d);
      const float vector = loadInput(vectors, vector_offset + d);
      if (metric == Metric::kL2) {
        const float difference = query - vector;
        squared_l2 = fmaf(difference, difference, squared_l2);
      } else {
        dot = fmaf(query, vector, dot);
        if (metric == Metric::kCosine) {
          vector_norm = fmaf(vector, vector, vector_norm);
        }
      }
    }
    float score = dot;
    if (metric == Metric::kL2) {
      score = squared_l2;
    } else if (metric == Metric::kCosine) {
      score = query_norm == 0.0F || vector_norm == 0.0F
                  ? 0.0F
                  : dot / sqrtf(query_norm * vector_norm);
    }
    if (apply_memory) {
      const float boost =
          memory_options.importance_weight * metadata.importance[vector_id] +
          (memory_options.recency_weight == 0.0F || memory_options.now == 0
               ? 0.0F
               : memory_options.recency_weight *
                     expf(-static_cast<float>(
                         (metadata.timestamps[vector_id] >= memory_options.now
                              ? 0.0
                              : static_cast<double>(
                                    memory_options.now -
                                    metadata.timestamps[vector_id])) /
                         memory_options.time_scale)));
      score = memory_options.semantic_weight * score +
              (metric == Metric::kL2 ? -boost : boost);
    }
    if (isBetter(score, vector_id, local_scores[local_worst_rank],
                 local_ids[local_worst_rank], metric)) {
      local_scores[local_worst_rank] = score;
      local_ids[local_worst_rank] = vector_id;
      local_worst_rank = 0;
      for (std::uint32_t rank = 1; rank < top_k; ++rank) {
        if (isBetter(local_scores[local_worst_rank],
                     local_ids[local_worst_rank], local_scores[rank],
                     local_ids[rank], metric)) {
          local_worst_rank = rank;
        }
      }
    }
  }

  for (std::uint32_t output_rank = 0; output_rank < top_k; ++output_rank) {
    std::uint32_t local_best_rank = 0;
    for (std::uint32_t rank = 1; rank < top_k; ++rank) {
      if (isBetter(local_scores[rank], local_ids[rank],
                   local_scores[local_best_rank], local_ids[local_best_rank],
                   metric)) {
        local_best_rank = rank;
      }
    }
    shared_scores[thread_id] = local_scores[local_best_rank];
    shared_ids[thread_id] = local_ids[local_best_rank];
    __syncthreads();
    for (std::uint32_t stride = kThreads / 2; stride > 0; stride >>= 1) {
      if (thread_id < stride &&
          isBetter(shared_scores[thread_id + stride],
                   shared_ids[thread_id + stride], shared_scores[thread_id],
                   shared_ids[thread_id], metric)) {
        shared_scores[thread_id] = shared_scores[thread_id + stride];
        shared_ids[thread_id] = shared_ids[thread_id + stride];
      }
      __syncthreads();
    }
    if (thread_id == 0) {
      const std::uint64_t output_offset =
          (static_cast<std::uint64_t>(query_id) * nprobe + probe) * top_k +
          output_rank;
      probe_scores[output_offset] = shared_scores[0];
      probe_ids[output_offset] = shared_ids[0];
    }
    const std::uint64_t selected_id = shared_ids[0];
    if (local_ids[local_best_rank] == selected_id) {
      local_scores[local_best_rank] = worst_score;
      local_ids[local_best_rank] = UINT64_MAX;
    }
    __syncthreads();
  }
}

__global__ void countSelectedCenterTasksKernel(
    const std::uint32_t* selected_centers, std::uint32_t* center_counts,
    std::uint32_t task_count) {
  const std::uint32_t task = blockIdx.x * blockDim.x + threadIdx.x;
  if (task < task_count) {
    atomicAdd(&center_counts[selected_centers[task]], 1U);
  }
}

__global__ void prepareCenterTaskCursorsKernel(
    const std::uint32_t* center_counts, std::uint32_t* center_cursors,
    std::uint32_t nlist) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  std::uint32_t offset = 0;
  for (std::uint32_t center = 0; center < nlist; ++center) {
    center_cursors[center] = offset;
    offset += center_counts[center];
  }
}

__global__ void scatterSelectedCenterTasksKernel(
    const std::uint32_t* selected_centers, std::uint32_t* center_cursors,
    std::uint64_t* bucket_tasks, std::uint32_t nprobe,
    std::uint32_t task_count) {
  const std::uint32_t task = blockIdx.x * blockDim.x + threadIdx.x;
  if (task >= task_count) {
    return;
  }
  const std::uint32_t center = selected_centers[task];
  const std::uint32_t destination = atomicAdd(&center_cursors[center], 1U);
  const std::uint32_t query_id = task / nprobe;
  const std::uint32_t probe = task - query_id * nprobe;
  bucket_tasks[destination] =
      (static_cast<std::uint64_t>(query_id) << 32U) | probe;
}

// 一个 block 包含若干 warp，每个 warp 协作计算一个候选向量。相邻 lane
// 读取相邻维度，适合行主序向量；lane 0 为该 warp 维护局部 Top-K，最后在
// WarpsPerBlock 可取 1/2/4/8。大 K 使用更少的独立 heap，并让更多 probe block
// 并发以维持活跃 warp；旧 8-warp 路径和 scalar 路径都保留用于 A/B。
template <typename InputType, std::uint32_t MaximumK,
          std::uint32_t WarpsPerBlock, Metric MetricValue,
          bool BucketMajor>
__global__ void ivfScanBucketsWarpKernel(
    const InputType* vectors, const std::uint64_t* original_ids,
    const std::uint64_t* bucket_offsets, const std::uint32_t* selected_centers,
    const std::uint64_t* bucket_tasks, const InputType* queries,
    float* probe_scores, std::uint64_t* probe_ids,
    const std::uint32_t* active_query_ids, const std::uint32_t* query_nprobes,
    std::uint32_t nprobe, std::uint32_t dim, std::uint32_t top_k,
    std::uint32_t heap_k,
    bool cache_query, bool apply_memory, DeviceMemoryView metadata,
    DeviceMemoryOptions memory_options) {
  constexpr std::uint32_t kWarpSize = 32;
  constexpr std::uint32_t kWarps = WarpsPerBlock;
  static_assert(WarpsPerBlock == 1 || WarpsPerBlock == 2 ||
                    WarpsPerBlock == 4 || WarpsPerBlock == 8,
                "IVF warp block 只支持 1/2/4/8 个 warp");
  constexpr float kFloatMaximum = 3.402823466e+38F;
  __shared__ float warp_scores[kWarps][MaximumK];
  __shared__ std::uint64_t warp_ids[kWarps][MaximumK];
  __shared__ float best_scores[kWarps];
  __shared__ std::uint64_t best_ids[kWarps];
  __shared__ std::uint32_t warp_ranks[kWarps];
  extern __shared__ __align__(16) unsigned char dynamic_shared[];
  auto* shared_query = reinterpret_cast<InputType*>(dynamic_shared);

  const std::uint32_t thread_id = threadIdx.x;
  const std::uint32_t lane_id = thread_id % kWarpSize;
  const std::uint32_t warp_id = thread_id / kWarpSize;
  std::uint32_t query_slot = blockIdx.y;
  std::uint32_t probe = blockIdx.x;
  if constexpr (BucketMajor) {
    const std::uint64_t task = bucket_tasks[blockIdx.x];
    query_slot = static_cast<std::uint32_t>(task >> 32U);
    probe = static_cast<std::uint32_t>(task);
  }
  const std::uint32_t query_id = active_query_ids == nullptr
                                     ? query_slot
                                     : active_query_ids[query_slot];
  if (query_nprobes != nullptr && probe >= query_nprobes[query_id]) {
    return;
  }
  const std::uint32_t center_id =
      selected_centers[static_cast<std::uint64_t>(query_id) * nprobe + probe];
  const std::uint64_t bucket_start = bucket_offsets[center_id];
  const std::uint64_t bucket_end = bucket_offsets[center_id + 1];
  const float worst_score =
      MetricValue == Metric::kL2 ? kFloatMaximum : -kFloatMaximum;
  float heap_root_score = worst_score;
  std::uint64_t heap_root_id = UINT64_MAX;

  if (lane_id == 0) {
    for (std::uint32_t rank = 0; rank < heap_k; ++rank) {
      warp_scores[warp_id][rank] = worst_score;
      warp_ids[warp_id][rank] = UINT64_MAX;
    }
  }
  const std::uint64_t query_offset = static_cast<std::uint64_t>(query_id) * dim;
  const InputType* query_values = queries + query_offset;
  if (cache_query) {
    for (std::uint32_t d = thread_id; d < dim; d += blockDim.x) {
      shared_query[d] = query_values[d];
    }
  }
  // 同时等待 heap 初始化和可选 query 缓存完成，不增加额外 block barrier。
  __syncthreads();
  if (cache_query) {
    query_values = shared_query;
  }

  float query_norm = 0.0F;
  if constexpr (MetricValue == Metric::kCosine) {
    for (std::uint32_t d = lane_id; d < dim; d += kWarpSize) {
      const float query = loadInput(query_values, d);
      query_norm = fmaf(query, query, query_norm);
    }
    query_norm = warpReduceSum(query_norm);
  }

  for (std::uint64_t position = bucket_start + warp_id; position < bucket_end;
       position += kWarps) {
    const std::uint64_t vector_id = original_ids[position];
    const bool accepted =
        !apply_memory ||
        (lane_id == 0 &&
         memoryCandidateMatches(metadata, vector_id, memory_options));
    if (!__shfl_sync(0xFFFFFFFFU, accepted, 0)) {
      continue;
    }
    const std::uint64_t vector_offset = position * dim;
    float dot = 0.0F;
    float vector_norm = 0.0F;
    float squared_l2 = 0.0F;
    for (std::uint32_t d = lane_id; d < dim; d += kWarpSize) {
      const float query = loadInput(query_values, d);
      const float vector = loadInput(vectors, vector_offset + d);
      if constexpr (MetricValue == Metric::kL2) {
        const float difference = query - vector;
        squared_l2 = fmaf(difference, difference, squared_l2);
      } else {
        dot = fmaf(query, vector, dot);
        if constexpr (MetricValue == Metric::kCosine) {
          vector_norm = fmaf(vector, vector, vector_norm);
        }
      }
    }
    // 每种 metric 只归约实际使用的累加器。旧代码在 inner-product 下还会
    // 对恒为 0 的 squared_l2 做一次完整 shuffle 归约，L2 下也会归约恒为
    // 0 的 dot；这在每个候选上浪费 5 次 warp shuffle。
    if constexpr (MetricValue == Metric::kL2) {
      squared_l2 = warpReduceSum(squared_l2);
    } else {
      dot = warpReduceSum(dot);
      if constexpr (MetricValue == Metric::kCosine) {
        vector_norm = warpReduceSum(vector_norm);
      }
    }
    if (lane_id == 0) {
      float score = dot;
      if constexpr (MetricValue == Metric::kL2) {
        score = squared_l2;
      } else if constexpr (MetricValue == Metric::kCosine) {
        score = query_norm == 0.0F || vector_norm == 0.0F
                    ? 0.0F
                    : dot / sqrtf(query_norm * vector_norm);
      }
      if (apply_memory) {
        score = applyMemoryScore<MetricValue>(score, vector_id, metadata,
                                              memory_options);
      }
      if (isBetter(score, vector_id, heap_root_score, heap_root_id,
                   MetricValue)) {
        // 以空穴下沉代替逐层交换：候选保存在寄存器中，每层只把更差的
        // child 上移一次，最后写入候选。根节点也缓存在寄存器中，避免每个
        // 被扫描向量都读取 shared memory。
        std::uint32_t parent = 0;
        while (true) {
          const std::uint32_t left = parent * 2 + 1;
          if (left >= heap_k) {
            break;
          }
          const std::uint32_t right = left + 1;
          std::uint32_t worse_child = left;
          if (right < heap_k &&
              isBetter(warp_scores[warp_id][left], warp_ids[warp_id][left],
                       warp_scores[warp_id][right], warp_ids[warp_id][right],
                       MetricValue)) {
            worse_child = right;
          }
          if (!isBetter(score, vector_id, warp_scores[warp_id][worse_child],
                        warp_ids[warp_id][worse_child], MetricValue)) {
            break;
          }
          warp_scores[warp_id][parent] = warp_scores[warp_id][worse_child];
          warp_ids[warp_id][parent] = warp_ids[warp_id][worse_child];
          parent = worse_child;
        }
        warp_scores[warp_id][parent] = score;
        warp_ids[warp_id][parent] = vector_id;
        heap_root_score = warp_scores[warp_id][0];
        heap_root_id = warp_ids[warp_id][0];
      }
    }
  }
  __syncthreads();

  // 根节点始终是当前最差候选。原地 heap sort 后，每个 warp 的数组按
  // best -> worst 排列，随后只需合并少量 warp 有序列表。
  if (lane_id == 0) {
    for (std::uint32_t heap_size = heap_k; heap_size > 1; --heap_size) {
      const std::uint32_t last = heap_size - 1;
      const float sorted_score = warp_scores[warp_id][0];
      const std::uint64_t sorted_id = warp_ids[warp_id][0];
      const float candidate_score = warp_scores[warp_id][last];
      const std::uint64_t candidate_id = warp_ids[warp_id][last];
      warp_scores[warp_id][last] = sorted_score;
      warp_ids[warp_id][last] = sorted_id;
      std::uint32_t parent = 0;
      while (true) {
        const std::uint32_t left = parent * 2 + 1;
        if (left >= last) {
          break;
        }
        const std::uint32_t right = left + 1;
        std::uint32_t worse_child = left;
        if (right < last &&
            isBetter(warp_scores[warp_id][left], warp_ids[warp_id][left],
                     warp_scores[warp_id][right], warp_ids[warp_id][right],
                     MetricValue)) {
          worse_child = right;
        }
        if (!isBetter(candidate_score, candidate_id,
                      warp_scores[warp_id][worse_child],
                      warp_ids[warp_id][worse_child], MetricValue)) {
          break;
        }
        warp_scores[warp_id][parent] = warp_scores[warp_id][worse_child];
        warp_ids[warp_id][parent] = warp_ids[warp_id][worse_child];
        parent = worse_child;
      }
      warp_scores[warp_id][parent] = candidate_score;
      warp_ids[warp_id][parent] = candidate_id;
    }
    warp_ranks[warp_id] = 0;
  }
  __syncthreads();

  if constexpr (kWarps == 1) {
    if (lane_id == 0) {
      const std::uint64_t output_offset =
          (static_cast<std::uint64_t>(query_id) * nprobe + probe) * top_k;
      for (std::uint32_t rank = 0; rank < top_k; ++rank) {
        probe_scores[output_offset + rank] = warp_scores[0][rank];
        probe_ids[output_offset + rank] = warp_ids[0][rank];
      }
    }
  } else {
    for (std::uint32_t output_rank = 0; output_rank < top_k; ++output_rank) {
      if (lane_id == 0) {
        const std::uint32_t rank = warp_ranks[warp_id];
        best_scores[warp_id] =
            rank < heap_k ? warp_scores[warp_id][rank] : worst_score;
        best_ids[warp_id] =
            rank < heap_k ? warp_ids[warp_id][rank] : UINT64_MAX;
      }
      __syncthreads();
      if (thread_id == 0) {
        std::uint32_t best_warp = 0;
        for (std::uint32_t candidate_warp = 1; candidate_warp < kWarps;
             ++candidate_warp) {
          if (isBetter(best_scores[candidate_warp], best_ids[candidate_warp],
                       best_scores[best_warp], best_ids[best_warp],
                       MetricValue)) {
            best_warp = candidate_warp;
          }
        }
        const std::uint64_t output_offset =
            (static_cast<std::uint64_t>(query_id) * nprobe + probe) * top_k +
            output_rank;
        probe_scores[output_offset] = best_scores[best_warp];
        probe_ids[output_offset] = best_ids[best_warp];
        ++warp_ranks[best_warp];
      }
      __syncthreads();
    }
  }
}

template <typename InputType, std::uint32_t MaximumK,
          std::uint32_t WarpsPerBlock, bool BucketMajor>
void launchIvfBucketScanWarp(
    dim3 grid, const InputType* vectors, const std::uint64_t* original_ids,
    const std::uint64_t* bucket_offsets, const std::uint32_t* selected_centers,
    const std::uint64_t* bucket_tasks, const InputType* queries,
    float* probe_scores, std::uint64_t* probe_ids,
    const std::uint32_t* active_query_ids, const std::uint32_t* query_nprobes,
    std::uint32_t nprobe, std::uint32_t dim, std::uint32_t top_k,
    std::uint32_t heap_k, Metric metric, bool apply_memory,
    DeviceMemoryView metadata,
    DeviceMemoryOptions memory_options) {
  constexpr std::uint32_t kThreads = WarpsPerBlock * 32;
  constexpr std::size_t kMaximumCachedQueryBytes = 8192;
  const std::size_t query_bytes =
      static_cast<std::size_t>(dim) * sizeof(InputType);
  const bool cache_query = query_bytes <= kMaximumCachedQueryBytes;
  const std::size_t shared_bytes = cache_query ? query_bytes : 0;
  if (metric == Metric::kL2) {
    ivfScanBucketsWarpKernel<InputType, MaximumK, WarpsPerBlock, Metric::kL2,
                             BucketMajor>
        <<<grid, kThreads, shared_bytes>>>(
            vectors, original_ids, bucket_offsets, selected_centers,
            bucket_tasks, queries, probe_scores, probe_ids, active_query_ids,
            query_nprobes, nprobe, dim, top_k, heap_k, cache_query, apply_memory,
            metadata, memory_options);
  } else if (metric == Metric::kInnerProduct) {
    ivfScanBucketsWarpKernel<InputType, MaximumK, WarpsPerBlock,
                             Metric::kInnerProduct, BucketMajor>
        <<<grid, kThreads, shared_bytes>>>(
            vectors, original_ids, bucket_offsets, selected_centers,
            bucket_tasks, queries, probe_scores, probe_ids, active_query_ids,
            query_nprobes, nprobe, dim, top_k, heap_k, cache_query, apply_memory,
            metadata, memory_options);
  } else {
    ivfScanBucketsWarpKernel<InputType, MaximumK, WarpsPerBlock,
                             Metric::kCosine, BucketMajor>
        <<<grid, kThreads, shared_bytes>>>(
        vectors, original_ids, bucket_offsets, selected_centers, bucket_tasks,
        queries, probe_scores, probe_ids, active_query_ids, query_nprobes,
        nprobe, dim, top_k, heap_k, cache_query, apply_memory, metadata,
        memory_options);
  }
}

template <typename InputType, std::uint32_t MaximumK>
void launchIvfBucketScan(
    dim3 grid, const InputType* vectors, const std::uint64_t* original_ids,
    const std::uint64_t* bucket_offsets, const std::uint32_t* selected_centers,
    const std::uint64_t* bucket_tasks, const InputType* queries,
    float* probe_scores, std::uint64_t* probe_ids,
    const std::uint32_t* active_query_ids, const std::uint32_t* query_nprobes,
    std::uint32_t nprobe, std::uint32_t dim, std::uint32_t top_k, Metric metric,
    std::uint32_t warps_per_block, bool bucket_major,
    std::uint32_t partitioned_topk_warps,
    std::uint32_t partitioned_topk_heap_k,
    bool apply_memory, DeviceMemoryView metadata,
    DeviceMemoryOptions memory_options) {
  constexpr std::uint32_t kThreads = 256;
  if (partitioned_topk_warps == 2) {
    constexpr std::uint32_t kPartitionHeapK = 32;
    if (bucket_major) {
      launchIvfBucketScanWarp<InputType, kPartitionHeapK, 2, true>(
          grid, vectors, original_ids, bucket_offsets, selected_centers,
          bucket_tasks, queries, probe_scores, probe_ids, active_query_ids,
          query_nprobes, nprobe, dim, top_k, kPartitionHeapK, metric,
          apply_memory, metadata, memory_options);
    } else {
      launchIvfBucketScanWarp<InputType, kPartitionHeapK, 2, false>(
          grid, vectors, original_ids, bucket_offsets, selected_centers,
          bucket_tasks, queries, probe_scores, probe_ids, active_query_ids,
          query_nprobes, nprobe, dim, top_k, kPartitionHeapK, metric,
          apply_memory, metadata, memory_options);
    }
    return;
  }
  if (partitioned_topk_warps == 4) {
    if (partitioned_topk_heap_k == 16) {
      constexpr std::uint32_t kPartitionHeapK = 16;
      if (bucket_major) {
        launchIvfBucketScanWarp<InputType, kPartitionHeapK, 4, true>(
            grid, vectors, original_ids, bucket_offsets, selected_centers,
            bucket_tasks, queries, probe_scores, probe_ids, active_query_ids,
            query_nprobes, nprobe, dim, top_k, kPartitionHeapK, metric,
            apply_memory, metadata, memory_options);
      } else {
        launchIvfBucketScanWarp<InputType, kPartitionHeapK, 4, false>(
            grid, vectors, original_ids, bucket_offsets, selected_centers,
            bucket_tasks, queries, probe_scores, probe_ids, active_query_ids,
            query_nprobes, nprobe, dim, top_k, kPartitionHeapK, metric,
            apply_memory, metadata, memory_options);
      }
    } else if (top_k <= 50) {
      constexpr std::uint32_t kPartitionHeapK = 20;
      if (bucket_major) {
        launchIvfBucketScanWarp<InputType, kPartitionHeapK, 4, true>(
            grid, vectors, original_ids, bucket_offsets, selected_centers,
            bucket_tasks, queries, probe_scores, probe_ids, active_query_ids,
            query_nprobes, nprobe, dim, top_k, kPartitionHeapK, metric,
            apply_memory, metadata, memory_options);
      } else {
        launchIvfBucketScanWarp<InputType, kPartitionHeapK, 4, false>(
            grid, vectors, original_ids, bucket_offsets, selected_centers,
            bucket_tasks, queries, probe_scores, probe_ids, active_query_ids,
            query_nprobes, nprobe, dim, top_k, kPartitionHeapK, metric,
            apply_memory, metadata, memory_options);
      }
    } else {
      constexpr std::uint32_t kPartitionHeapK = 40;
      if (bucket_major) {
        launchIvfBucketScanWarp<InputType, kPartitionHeapK, 4, true>(
            grid, vectors, original_ids, bucket_offsets, selected_centers,
            bucket_tasks, queries, probe_scores, probe_ids, active_query_ids,
            query_nprobes, nprobe, dim, top_k, kPartitionHeapK, metric,
            apply_memory, metadata, memory_options);
      } else {
        launchIvfBucketScanWarp<InputType, kPartitionHeapK, 4, false>(
            grid, vectors, original_ids, bucket_offsets, selected_centers,
            bucket_tasks, queries, probe_scores, probe_ids, active_query_ids,
            query_nprobes, nprobe, dim, top_k, kPartitionHeapK, metric,
            apply_memory, metadata, memory_options);
      }
    }
    return;
  }
  if (partitioned_topk_warps == 8) {
    constexpr std::uint32_t kPartitionHeapK = 8;
    if (bucket_major) {
      launchIvfBucketScanWarp<InputType, kPartitionHeapK, 8, true>(
          grid, vectors, original_ids, bucket_offsets, selected_centers,
          bucket_tasks, queries, probe_scores, probe_ids, active_query_ids,
          query_nprobes, nprobe, dim, top_k, kPartitionHeapK, metric,
          apply_memory, metadata, memory_options);
    } else {
      launchIvfBucketScanWarp<InputType, kPartitionHeapK, 8, false>(
          grid, vectors, original_ids, bucket_offsets, selected_centers,
          bucket_tasks, queries, probe_scores, probe_ids, active_query_ids,
          query_nprobes, nprobe, dim, top_k, kPartitionHeapK, metric,
          apply_memory, metadata, memory_options);
    }
    return;
  }
  if (warps_per_block == 1) {
    if (bucket_major) {
      launchIvfBucketScanWarp<InputType, MaximumK, 1, true>(
          grid, vectors, original_ids, bucket_offsets, selected_centers,
          bucket_tasks, queries, probe_scores, probe_ids, active_query_ids,
          query_nprobes, nprobe, dim, top_k, top_k, metric, apply_memory, metadata,
          memory_options);
    } else {
      launchIvfBucketScanWarp<InputType, MaximumK, 1, false>(
          grid, vectors, original_ids, bucket_offsets, selected_centers,
          bucket_tasks, queries, probe_scores, probe_ids, active_query_ids,
          query_nprobes, nprobe, dim, top_k, top_k, metric, apply_memory, metadata,
          memory_options);
    }
  } else if (warps_per_block == 2) {
    if (bucket_major) {
      launchIvfBucketScanWarp<InputType, MaximumK, 2, true>(
          grid, vectors, original_ids, bucket_offsets, selected_centers,
          bucket_tasks, queries, probe_scores, probe_ids, active_query_ids,
          query_nprobes, nprobe, dim, top_k, top_k, metric, apply_memory, metadata,
          memory_options);
    } else {
      launchIvfBucketScanWarp<InputType, MaximumK, 2, false>(
          grid, vectors, original_ids, bucket_offsets, selected_centers,
          bucket_tasks, queries, probe_scores, probe_ids, active_query_ids,
          query_nprobes, nprobe, dim, top_k, top_k, metric, apply_memory, metadata,
          memory_options);
    }
  } else if (warps_per_block == 4) {
    if (bucket_major) {
      launchIvfBucketScanWarp<InputType, MaximumK, 4, true>(
          grid, vectors, original_ids, bucket_offsets, selected_centers,
          bucket_tasks, queries, probe_scores, probe_ids, active_query_ids,
          query_nprobes, nprobe, dim, top_k, top_k, metric, apply_memory, metadata,
          memory_options);
    } else {
      launchIvfBucketScanWarp<InputType, MaximumK, 4, false>(
          grid, vectors, original_ids, bucket_offsets, selected_centers,
          bucket_tasks, queries, probe_scores, probe_ids, active_query_ids,
          query_nprobes, nprobe, dim, top_k, top_k, metric, apply_memory, metadata,
          memory_options);
    }
  } else if (warps_per_block == 8) {
    if (bucket_major) {
      launchIvfBucketScanWarp<InputType, MaximumK, 8, true>(
          grid, vectors, original_ids, bucket_offsets, selected_centers,
          bucket_tasks, queries, probe_scores, probe_ids, active_query_ids,
          query_nprobes, nprobe, dim, top_k, top_k, metric, apply_memory, metadata,
          memory_options);
    } else {
      launchIvfBucketScanWarp<InputType, MaximumK, 8, false>(
          grid, vectors, original_ids, bucket_offsets, selected_centers,
          bucket_tasks, queries, probe_scores, probe_ids, active_query_ids,
          query_nprobes, nprobe, dim, top_k, top_k, metric, apply_memory, metadata,
          memory_options);
    }
  } else {
    ivfScanBucketsScalarKernel<InputType, MaximumK><<<grid, kThreads>>>(
        vectors, original_ids, bucket_offsets, selected_centers, queries,
        probe_scores, probe_ids, active_query_ids, query_nprobes, nprobe, dim,
        top_k, metric, apply_memory, metadata, memory_options);
  }
}

// 每个 probe 已按优劣输出有序 Top-K。最终阶段只需维护每个 probe 的当前
// 游标，每轮在所有列表头之间归约出一个全局结果，避免为 K=50/100 给每个
// CUDA 线程分配大型局部数组。
__global__ void ivfMergeProbeTopKKernel(const float* probe_scores,
                                        const std::uint64_t* probe_ids,
                                        float* top_scores,
                                        std::uint64_t* top_ids,
                                        const std::uint32_t* query_nprobes,
                                        std::uint32_t nprobe,
                                        std::uint32_t top_k, Metric metric) {
  constexpr std::uint32_t kThreads = 256;
  constexpr float kFloatMaximum = 3.402823466e+38F;
  __shared__ float shared_scores[kThreads];
  __shared__ std::uint64_t shared_ids[kThreads];
  __shared__ std::uint32_t shared_probes[kThreads];
  extern __shared__ std::uint32_t probe_ranks[];
  const std::uint32_t thread_id = threadIdx.x;
  const std::uint32_t query_id = blockIdx.x;
  const std::uint32_t active_nprobe =
      query_nprobes == nullptr ? nprobe : query_nprobes[query_id];
  const float worst_score =
      metric == Metric::kL2 ? kFloatMaximum : -kFloatMaximum;
  for (std::uint32_t probe = thread_id; probe < active_nprobe;
       probe += blockDim.x) {
    probe_ranks[probe] = 0;
  }
  __syncthreads();

  for (std::uint32_t output_rank = 0; output_rank < top_k; ++output_rank) {
    float local_best_score = worst_score;
    std::uint64_t local_best_id = UINT64_MAX;
    std::uint32_t local_best_probe = UINT32_MAX;
    for (std::uint32_t probe = thread_id; probe < active_nprobe;
         probe += blockDim.x) {
      const std::uint32_t rank = probe_ranks[probe];
      if (rank < top_k) {
        const std::uint64_t input_offset =
            (static_cast<std::uint64_t>(query_id) * nprobe + probe) * top_k +
            rank;
        const float score = probe_scores[input_offset];
        const std::uint64_t id = probe_ids[input_offset];
        if (isBetter(score, id, local_best_score, local_best_id, metric)) {
          local_best_score = score;
          local_best_id = id;
          local_best_probe = probe;
        }
      }
    }
    shared_scores[thread_id] = local_best_score;
    shared_ids[thread_id] = local_best_id;
    shared_probes[thread_id] = local_best_probe;
    __syncthreads();
    for (std::uint32_t stride = kThreads / 2; stride > 0; stride >>= 1) {
      if (thread_id < stride &&
          isBetter(shared_scores[thread_id + stride],
                   shared_ids[thread_id + stride], shared_scores[thread_id],
                   shared_ids[thread_id], metric)) {
        shared_scores[thread_id] = shared_scores[thread_id + stride];
        shared_ids[thread_id] = shared_ids[thread_id + stride];
        shared_probes[thread_id] = shared_probes[thread_id + stride];
      }
      __syncthreads();
    }
    if (thread_id == 0) {
      const std::uint64_t output_offset =
          static_cast<std::uint64_t>(query_id) * top_k + output_rank;
      top_scores[output_offset] = shared_scores[0];
      top_ids[output_offset] = shared_ids[0];
      if (shared_probes[0] != UINT32_MAX && shared_ids[0] != UINT64_MAX) {
        ++probe_ranks[shared_probes[0]];
      }
    }
    __syncthreads();
  }
}

// Deliberately separate baseline: first retrieve semantic Top-(K * factor),
// then launch a second kernel to filter and rerank that bounded candidate set.
// This is kept beside the fused path so their latency and result quality can be
// compared without involving a CPU reranker.
__global__ void rerankMemoryTopKKernel(
    const float* semantic_scores, const std::uint64_t* semantic_ids,
    float* output_scores, std::uint64_t* output_ids, std::uint32_t semantic_k,
    std::uint32_t top_k, Metric metric, DeviceMemoryView metadata,
    DeviceMemoryOptions options) {
  if (threadIdx.x != 0) {
    return;
  }
  constexpr std::uint32_t kMaximumK = 100;
  constexpr float kFloatMaximum = 3.402823466e+38F;
  const std::uint32_t query_id = blockIdx.x;
  const float worst_score =
      metric == Metric::kL2 ? kFloatMaximum : -kFloatMaximum;
  float best_scores[kMaximumK];
  std::uint64_t best_ids[kMaximumK];
  for (std::uint32_t rank = 0; rank < top_k; ++rank) {
    best_scores[rank] = worst_score;
    best_ids[rank] = UINT64_MAX;
  }
  std::uint32_t worst_rank = 0;
  const std::uint64_t input_offset =
      static_cast<std::uint64_t>(query_id) * semantic_k;
  for (std::uint32_t candidate = 0; candidate < semantic_k; ++candidate) {
    const std::uint64_t vector_id = semantic_ids[input_offset + candidate];
    if (vector_id == UINT64_MAX ||
        !memoryCandidateMatches(metadata, vector_id, options)) {
      continue;
    }
    const float semantic_score = semantic_scores[input_offset + candidate];
    float recency = 0.0F;
    if (options.recency_weight != 0.0F && options.now != 0) {
      const std::uint64_t timestamp = metadata.timestamps[vector_id];
      const double age = timestamp >= options.now
                             ? 0.0
                             : static_cast<double>(options.now - timestamp);
      recency = expf(-static_cast<float>(age / options.time_scale));
    }
    const float boost =
        options.importance_weight * metadata.importance[vector_id] +
        options.recency_weight * recency;
    const float score = options.semantic_weight * semantic_score +
                        (metric == Metric::kL2 ? -boost : boost);
    if (isBetter(score, vector_id, best_scores[worst_rank],
                 best_ids[worst_rank], metric)) {
      best_scores[worst_rank] = score;
      best_ids[worst_rank] = vector_id;
      worst_rank = 0;
      for (std::uint32_t rank = 1; rank < top_k; ++rank) {
        if (isBetter(best_scores[worst_rank], best_ids[worst_rank],
                     best_scores[rank], best_ids[rank], metric)) {
          worst_rank = rank;
        }
      }
    }
  }
  const std::uint64_t output_offset =
      static_cast<std::uint64_t>(query_id) * top_k;
  for (std::uint32_t output_rank = 0; output_rank < top_k; ++output_rank) {
    std::uint32_t best_rank = 0;
    for (std::uint32_t rank = 1; rank < top_k; ++rank) {
      if (isBetter(best_scores[rank], best_ids[rank], best_scores[best_rank],
                   best_ids[best_rank], metric)) {
        best_rank = rank;
      }
    }
    output_scores[output_offset + output_rank] = best_scores[best_rank];
    output_ids[output_offset + output_rank] = best_ids[best_rank];
    best_scores[best_rank] = worst_score;
    best_ids[best_rank] = UINT64_MAX;
  }
}

void validateSearchConfiguration(const IvfFlatIndex& index,
                                 const QuerySet& queries,
                                 const SearchParams& params,
                                 const MemoryMetadata* metadata) {
  validateIvfFlatIndex(index);
  if (queries.num_queries == 0 || queries.dim != index.dim ||
      queries.dtype != index.dtype) {
    throw std::runtime_error("查询集与 GPU IVF 索引的维度或 dtype 不匹配");
  }
  const std::uint64_t expected_values = queries.num_queries * queries.dim;
  if ((index.dtype == DataType::kFloat32 &&
       queries.values.size() != expected_values) ||
      (index.dtype == DataType::kFloat16 &&
       queries.half_values.size() != expected_values)) {
    throw std::runtime_error("GPU IVF 查询数据长度与元数据不一致");
  }
  constexpr std::uint32_t kMaximumTopK = 100;
  constexpr std::uint32_t kMaximumNprobe = 8192;
  if (params.top_k == 0 || params.top_k > kMaximumTopK ||
      params.top_k > index.num_vectors || params.nprobe == 0 ||
      params.nprobe > index.nlist || params.nprobe > kMaximumNprobe ||
      params.batch_size == 0) {
    throw std::runtime_error("GPU IVF 的 top_k、nprobe 或 batch_size 不合法");
  }
  if (params.distance_mode != "simple" && params.distance_mode != "warp" &&
      params.distance_mode != "warp_compact" &&
      params.distance_mode != "bucket_major" &&
      params.distance_mode != "bucket_major_partitioned" &&
      params.distance_mode != "bucket_major_partitioned2" &&
      params.distance_mode != "bucket_major_partitioned4x16" &&
      params.distance_mode != "bucket_major_partitioned8x8" &&
      params.distance_mode != "warp1" && params.distance_mode != "warp2" &&
      params.distance_mode != "warp4" && params.distance_mode != "warp8") {
    throw std::runtime_error(
        "GPU IVF 的 distance_mode 必须是 simple、warp、warp_compact、"
        "bucket_major、bucket_major_partitioned、"
        "bucket_major_partitioned2、bucket_major_partitioned4x16、"
        "bucket_major_partitioned8x8、warp1、warp2、warp4 或 warp8");
  }
  const bool bucket_major = params.distance_mode == "bucket_major" ||
                            params.distance_mode ==
                                "bucket_major_partitioned" ||
                            params.distance_mode ==
                                "bucket_major_partitioned2" ||
                            params.distance_mode ==
                                "bucket_major_partitioned4x16" ||
                            params.distance_mode ==
                                "bucket_major_partitioned8x8";
  if (bucket_major &&
      (params.nprobe_policy != "fixed" ||
       static_cast<std::uint64_t>(params.batch_size) * params.nprobe >
           UINT32_MAX)) {
    throw std::runtime_error(
        "GPU IVF 的 bucket_major 当前只支持 fixed nprobe 和 32 位任务数");
  }
  const bool partitioned =
      params.distance_mode == "bucket_major_partitioned" ||
      params.distance_mode == "bucket_major_partitioned2" ||
      params.distance_mode == "bucket_major_partitioned4x16" ||
      params.distance_mode == "bucket_major_partitioned8x8";
  if (partitioned && params.top_k != 50 && params.top_k != 100) {
    throw std::runtime_error(
        "bucket_major_partitioned 仅支持 top_k=50 或 top_k=100");
  }
  if (params.distance_mode == "bucket_major_partitioned2" &&
      params.top_k != 50) {
    throw std::runtime_error(
        "bucket_major_partitioned2 仅用于 top_k=50 的 2x32 实验");
  }
  if ((params.distance_mode == "bucket_major_partitioned4x16" ||
       params.distance_mode == "bucket_major_partitioned8x8") &&
      params.top_k != 50) {
    throw std::runtime_error(
        "4x16 与 8x8 分区模式仅用于 top_k=50 实验");
  }
  if (partitioned && params.memory_mode != "disabled") {
    throw std::runtime_error(
        "bucket_major_partitioned 要求 memory_mode=disabled");
  }
  if (params.nprobe_policy != "fixed" && params.nprobe_policy != "score_mass") {
    throw std::runtime_error(
        "GPU IVF 的 nprobe_policy 必须是 fixed 或 score_mass");
  }
  if (params.nprobe_policy == "score_mass" &&
      (params.adaptive_nprobe_min == 0 ||
       params.adaptive_nprobe_min > params.nprobe ||
       params.adaptive_nprobe_step == 0 ||
       !std::isfinite(params.adaptive_target_mass) ||
       params.adaptive_target_mass <= 0.0F ||
       params.adaptive_target_mass > 1.0F ||
       !std::isfinite(params.adaptive_temperature) ||
       params.adaptive_temperature <= 0.0F)) {
    throw std::runtime_error("GPU IVF 的自适应 nprobe 参数不合法");
  }
  if (params.adaptive_execution != "masked" &&
      params.adaptive_execution != "grouped") {
    throw std::runtime_error(
        "GPU IVF 的 adaptive_execution 必须是 masked 或 grouped");
  }
  if (params.memory_mode != "disabled" && params.memory_mode != "fused" &&
      params.memory_mode != "rerank") {
    throw std::runtime_error(
        "GPU IVF 的 memory_mode 必须是 disabled、fused 或 rerank");
  }
  if (params.memory_mode != "disabled") {
    if (metadata == nullptr) {
      throw std::runtime_error("GPU 记忆检索必须提供元数据");
    }
    validateMemoryMetadata(*metadata, index.num_vectors);
    if (!std::isfinite(params.memory_semantic_weight) ||
        params.memory_semantic_weight <= 0.0F ||
        !std::isfinite(params.memory_importance_weight) ||
        params.memory_importance_weight < 0.0F ||
        !std::isfinite(params.memory_recency_weight) ||
        params.memory_recency_weight < 0.0F ||
        !std::isfinite(params.memory_time_scale) ||
        params.memory_time_scale <= 0.0F || params.memory_rerank_factor == 0) {
      throw std::runtime_error("GPU 记忆评分参数不合法");
    }
  }
}

}  // namespace

struct GpuIvfFlatSearchEngine::Impl {
  Impl(const IvfFlatIndex& index, const QuerySet& initial_queries,
       const SearchParams& search_params, const MemoryMetadata* metadata)
      : num_vectors(index.num_vectors),
        dim(index.dim),
        nlist(index.nlist),
        dtype(index.dtype),
        metric(index.metric),
        params(search_params),
        adaptive_nprobe(search_params.nprobe_policy == "score_mass"),
        group_adaptive_queries(search_params.nprobe_policy == "score_mass" &&
                               search_params.adaptive_execution == "grouped"),
        bucket_major(search_params.distance_mode == "bucket_major" ||
                     search_params.distance_mode ==
                         "bucket_major_partitioned" ||
                     search_params.distance_mode ==
                         "bucket_major_partitioned2" ||
                     search_params.distance_mode ==
                         "bucket_major_partitioned4x16" ||
                     search_params.distance_mode ==
                         "bucket_major_partitioned8x8"),
        partitioned_topk_warps(
            search_params.distance_mode == "bucket_major_partitioned2"
                ? 2U
                : search_params.distance_mode ==
                          "bucket_major_partitioned8x8"
                      ? 8U
                : search_params.distance_mode == "bucket_major_partitioned"
                          || search_params.distance_mode ==
                                 "bucket_major_partitioned4x16"
                      ? 4U
                      : 0U),
        partitioned_topk_heap_k(
            search_params.distance_mode == "bucket_major_partitioned2"
                ? 32U
                : search_params.distance_mode ==
                          "bucket_major_partitioned4x16"
                      ? 16U
                      : search_params.distance_mode ==
                                "bucket_major_partitioned8x8"
                            ? 8U
                            : search_params.top_k <= 50 ? 20U : 40U),
        memory_enabled(search_params.memory_mode != "disabled"),
        fused_memory(search_params.memory_mode == "fused"),
        separate_rerank(search_params.memory_mode == "rerank"),
        use_fp16(index.dtype == DataType::kFloat16),
        input_element_bytes(use_fp16 ? sizeof(std::uint16_t) : sizeof(float)),
        batch_capacity(search_params.batch_size),
        host_index(&index),
        host_metadata(metadata) {
    validateSearchConfiguration(index, initial_queries, params, metadata);
    scan_top_k = params.top_k;
    if (separate_rerank) {
      const std::uint64_t requested = static_cast<std::uint64_t>(params.top_k) *
                                      params.memory_rerank_factor;
      scan_top_k = static_cast<std::uint32_t>(
          std::min<std::uint64_t>({requested, 100, num_vectors}));
    }
    std::vector<std::uint64_t> bucket_sizes(index.nlist);
    for (std::uint32_t center_id = 0; center_id < index.nlist; ++center_id) {
      bucket_sizes[center_id] =
          index.offsets[center_id + 1] - index.offsets[center_id];
    }
    std::sort(bucket_sizes.begin(), bucket_sizes.end());
    const std::uint32_t minimum_probe_count =
        adaptive_nprobe ? params.adaptive_nprobe_min : params.nprobe;
    const std::uint64_t minimum_candidate_count = std::accumulate(
        bucket_sizes.begin(), bucket_sizes.begin() + minimum_probe_count,
        std::uint64_t{0});
    if (minimum_candidate_count < scan_top_k) {
      throw std::runtime_error(
          "GPU IVF 的部分 nprobe 桶组合候选数可能少于 top_k");
    }
    const std::size_t vector_bytes =
        static_cast<std::size_t>(num_vectors) * dim * input_element_bytes;
    const std::size_t query_bytes =
        static_cast<std::size_t>(batch_capacity) * dim * input_element_bytes;
    const std::size_t center_score_slots =
        static_cast<std::size_t>(batch_capacity) * nlist;
    const std::size_t selected_center_slots =
        static_cast<std::size_t>(batch_capacity) * params.nprobe;
    const std::size_t probe_result_slots = selected_center_slots * scan_top_k;
    const std::size_t result_slots =
        static_cast<std::size_t>(batch_capacity) * scan_top_k;
    device_bytes =
        vector_bytes +
        static_cast<std::size_t>(num_vectors) * sizeof(std::uint64_t) +
        (memory_enabled ? static_cast<std::size_t>(num_vectors) *
                              (sizeof(std::uint64_t) + sizeof(float) +
                               2 * sizeof(std::uint32_t))
                        : 0) +
        static_cast<std::size_t>(nlist) * dim * sizeof(float) +
        (static_cast<std::size_t>(nlist) + 1) * sizeof(std::uint64_t) +
        query_bytes + center_score_slots * sizeof(float) +
        selected_center_slots * sizeof(std::uint32_t) +
        (adaptive_nprobe ? selected_center_slots * sizeof(float) +
                               static_cast<std::size_t>(batch_capacity) *
                                   sizeof(std::uint32_t) +
                               (group_adaptive_queries
                                    ? static_cast<std::size_t>(batch_capacity) *
                                          sizeof(std::uint32_t)
                                    : 0)
                         : 0) +
        (bucket_major
             ? 2 * static_cast<std::size_t>(nlist) * sizeof(std::uint32_t) +
                   selected_center_slots * sizeof(std::uint64_t)
             : 0) +
        probe_result_slots * (sizeof(float) + sizeof(std::uint64_t)) +
        result_slots * (sizeof(float) + sizeof(std::uint64_t));
    device_vectors = std::make_unique<DeviceBuffer>(vector_bytes);
    device_ids = std::make_unique<DeviceBuffer>(
        static_cast<std::size_t>(num_vectors) * sizeof(std::uint64_t));
    device_centers = std::make_unique<DeviceBuffer>(
        static_cast<std::size_t>(nlist) * dim * sizeof(float));
    device_offsets = std::make_unique<DeviceBuffer>(
        (static_cast<std::size_t>(nlist) + 1) * sizeof(std::uint64_t));
    if (memory_enabled) {
      device_timestamps = std::make_unique<DeviceBuffer>(
          static_cast<std::size_t>(num_vectors) * sizeof(std::uint64_t));
      device_importance = std::make_unique<DeviceBuffer>(
          static_cast<std::size_t>(num_vectors) * sizeof(float));
      device_session_ids = std::make_unique<DeviceBuffer>(
          static_cast<std::size_t>(num_vectors) * sizeof(std::uint32_t));
      device_source_types = std::make_unique<DeviceBuffer>(
          static_cast<std::size_t>(num_vectors) * sizeof(std::uint32_t));
    }
    device_queries = std::make_unique<DeviceBuffer>(query_bytes);
    device_center_scores =
        std::make_unique<DeviceBuffer>(center_score_slots * sizeof(float));
    device_selected_centers = std::make_unique<DeviceBuffer>(
        selected_center_slots * sizeof(std::uint32_t));
    if (adaptive_nprobe) {
      device_selected_scores =
          std::make_unique<DeviceBuffer>(selected_center_slots * sizeof(float));
      device_query_nprobes = std::make_unique<DeviceBuffer>(
          static_cast<std::size_t>(batch_capacity) * sizeof(std::uint32_t));
      if (group_adaptive_queries) {
        device_active_query_ids = std::make_unique<DeviceBuffer>(
            static_cast<std::size_t>(batch_capacity) * sizeof(std::uint32_t));
      }
    }
    if (bucket_major) {
      device_bucket_task_counts = std::make_unique<DeviceBuffer>(
          static_cast<std::size_t>(nlist) * sizeof(std::uint32_t));
      device_bucket_task_cursors = std::make_unique<DeviceBuffer>(
          static_cast<std::size_t>(nlist) * sizeof(std::uint32_t));
      device_bucket_tasks = std::make_unique<DeviceBuffer>(
          selected_center_slots * sizeof(std::uint64_t));
    }
    device_probe_scores =
        std::make_unique<DeviceBuffer>(probe_result_slots * sizeof(float));
    device_probe_ids = std::make_unique<DeviceBuffer>(probe_result_slots *
                                                      sizeof(std::uint64_t));
    device_top_scores =
        std::make_unique<DeviceBuffer>(result_slots * sizeof(float));
    device_top_ids =
        std::make_unique<DeviceBuffer>(result_slots * sizeof(std::uint64_t));

    CudaEvent copy_start;
    CudaEvent copy_end;
    copy_start.record();
    const void* vectors =
        use_fp16 ? static_cast<const void*>(index.half_values.data())
                 : static_cast<const void*>(index.values.data());
    checkCuda(cudaMemcpy(device_vectors->data(), vectors, vector_bytes,
                         cudaMemcpyHostToDevice),
              "复制 IVF 向量到 GPU");
    checkCuda(cudaMemcpy(
                  device_ids->data(), index.ids.data(),
                  static_cast<std::size_t>(num_vectors) * sizeof(std::uint64_t),
                  cudaMemcpyHostToDevice),
              "复制 IVF ID 到 GPU");
    checkCuda(cudaMemcpy(device_centers->data(), index.centers.data(),
                         static_cast<std::size_t>(nlist) * dim * sizeof(float),
                         cudaMemcpyHostToDevice),
              "复制 IVF centers 到 GPU");
    checkCuda(cudaMemcpy(
                  device_offsets->data(), index.offsets.data(),
                  (static_cast<std::size_t>(nlist) + 1) * sizeof(std::uint64_t),
                  cudaMemcpyHostToDevice),
              "复制 IVF offsets 到 GPU");
    if (memory_enabled) {
      checkCuda(
          cudaMemcpy(
              device_timestamps->data(), metadata->timestamps.data(),
              static_cast<std::size_t>(num_vectors) * sizeof(std::uint64_t),
              cudaMemcpyHostToDevice),
          "复制记忆 timestamps 到 GPU");
      checkCuda(
          cudaMemcpy(device_importance->data(), metadata->importance.data(),
                     static_cast<std::size_t>(num_vectors) * sizeof(float),
                     cudaMemcpyHostToDevice),
          "复制记忆 importance 到 GPU");
      checkCuda(
          cudaMemcpy(
              device_session_ids->data(), metadata->session_ids.data(),
              static_cast<std::size_t>(num_vectors) * sizeof(std::uint32_t),
              cudaMemcpyHostToDevice),
          "复制记忆 session IDs 到 GPU");
      checkCuda(
          cudaMemcpy(
              device_source_types->data(), metadata->source_types.data(),
              static_cast<std::size_t>(num_vectors) * sizeof(std::uint32_t),
              cudaMemcpyHostToDevice),
          "复制记忆 source types 到 GPU");
    }
    copy_end.record();
    copy_end.synchronize();
    index_h2d_ms = elapsedMilliseconds(copy_start, copy_end);
  }

  std::uint64_t num_vectors;
  std::uint32_t dim;
  std::uint32_t nlist;
  DataType dtype;
  Metric metric;
  SearchParams params;
  bool adaptive_nprobe;
  bool group_adaptive_queries;
  bool bucket_major;
  std::uint32_t partitioned_topk_warps;
  std::uint32_t partitioned_topk_heap_k;
  bool memory_enabled;
  bool fused_memory;
  bool separate_rerank;
  bool use_fp16;
  std::size_t input_element_bytes;
  std::uint32_t batch_capacity;
  std::uint32_t scan_top_k = 0;
  const IvfFlatIndex* host_index;
  const MemoryMetadata* host_metadata;
  double index_h2d_ms = 0.0;
  std::size_t device_bytes = 0;
  std::unique_ptr<DeviceBuffer> device_vectors;
  std::unique_ptr<DeviceBuffer> device_ids;
  std::unique_ptr<DeviceBuffer> device_centers;
  std::unique_ptr<DeviceBuffer> device_offsets;
  std::unique_ptr<DeviceBuffer> device_timestamps;
  std::unique_ptr<DeviceBuffer> device_importance;
  std::unique_ptr<DeviceBuffer> device_session_ids;
  std::unique_ptr<DeviceBuffer> device_source_types;
  std::unique_ptr<DeviceBuffer> device_queries;
  std::unique_ptr<DeviceBuffer> device_center_scores;
  std::unique_ptr<DeviceBuffer> device_selected_centers;
  std::unique_ptr<DeviceBuffer> device_selected_scores;
  std::unique_ptr<DeviceBuffer> device_query_nprobes;
  std::unique_ptr<DeviceBuffer> device_active_query_ids;
  std::unique_ptr<DeviceBuffer> device_bucket_task_counts;
  std::unique_ptr<DeviceBuffer> device_bucket_task_cursors;
  std::unique_ptr<DeviceBuffer> device_bucket_tasks;
  std::unique_ptr<DeviceBuffer> device_probe_scores;
  std::unique_ptr<DeviceBuffer> device_probe_ids;
  std::unique_ptr<DeviceBuffer> device_top_scores;
  std::unique_ptr<DeviceBuffer> device_top_ids;
};

GpuIvfFlatSearchEngine::GpuIvfFlatSearchEngine(const IvfFlatIndex& index,
                                               const QuerySet& initial_queries,
                                               const SearchParams& params,
                                               const MemoryMetadata* metadata)
    : impl_(std::make_unique<Impl>(index, initial_queries, params, metadata)) {}

GpuIvfFlatSearchEngine::~GpuIvfFlatSearchEngine() = default;
GpuIvfFlatSearchEngine::GpuIvfFlatSearchEngine(
    GpuIvfFlatSearchEngine&&) noexcept = default;
GpuIvfFlatSearchEngine& GpuIvfFlatSearchEngine::operator=(
    GpuIvfFlatSearchEngine&&) noexcept = default;

double GpuIvfFlatSearchEngine::indexH2DMilliseconds() const {
  return impl_->index_h2d_ms;
}

std::size_t GpuIvfFlatSearchEngine::deviceBytes() const {
  return impl_->device_bytes;
}

SearchResults GpuIvfFlatSearchEngine::search(const QuerySet& queries,
                                             SearchStats* stats) {
  validateSearchConfiguration(*impl_->host_index, queries, impl_->params,
                              impl_->host_metadata);
  SearchStats measured;
  SearchResults results(queries.num_queries);
  std::vector<float> host_top_scores(
      static_cast<std::size_t>(impl_->batch_capacity) * impl_->params.top_k);
  std::vector<std::uint64_t> host_top_ids(
      static_cast<std::size_t>(impl_->batch_capacity) * impl_->params.top_k);
  std::vector<std::uint32_t> host_query_nprobes(impl_->batch_capacity,
                                                impl_->params.nprobe);

  CudaEvent copy_start;
  CudaEvent copy_end;
  CudaEvent center_end;
  CudaEvent scan_end;
  CudaEvent merge_end;
  CudaEvent rerank_end;
  CudaEvent result_end;
  for (std::uint64_t batch_start = 0; batch_start < queries.num_queries;
       batch_start += impl_->batch_capacity) {
    const std::uint32_t batch_count =
        static_cast<std::uint32_t>(std::min<std::uint64_t>(
            impl_->batch_capacity, queries.num_queries - batch_start));
    const std::size_t query_value_offset =
        static_cast<std::size_t>(batch_start) * impl_->dim;
    const void* query_data =
        impl_->use_fp16 ? static_cast<const void*>(queries.half_values.data() +
                                                   query_value_offset)
                        : static_cast<const void*>(queries.values.data() +
                                                   query_value_offset);
    copy_start.record();
    checkCuda(cudaMemcpy(impl_->device_queries->data(), query_data,
                         static_cast<std::size_t>(batch_count) * impl_->dim *
                             impl_->input_element_bytes,
                         cudaMemcpyHostToDevice),
              "复制 IVF queries 到 GPU");
    copy_end.record();

    constexpr std::uint32_t kThreads = 256;
    const dim3 center_grid((impl_->nlist + kThreads - 1) / kThreads,
                           batch_count);
    if (impl_->use_fp16) {
      ivfCenterScoresKernel<<<center_grid, kThreads>>>(
          static_cast<const float*>(impl_->device_centers->data()),
          static_cast<const __half*>(impl_->device_queries->data()),
          static_cast<float*>(impl_->device_center_scores->data()),
          impl_->nlist, impl_->dim, batch_count, impl_->metric);
    } else {
      ivfCenterScoresKernel<<<center_grid, kThreads>>>(
          static_cast<const float*>(impl_->device_centers->data()),
          static_cast<const float*>(impl_->device_queries->data()),
          static_cast<float*>(impl_->device_center_scores->data()),
          impl_->nlist, impl_->dim, batch_count, impl_->metric);
    }
    checkCuda(cudaGetLastError(), "启动 ivfCenterScoresKernel");
    ivfSelectProbesBlockKernel<<<batch_count, kThreads>>>(
        static_cast<float*>(impl_->device_center_scores->data()),
        static_cast<std::uint32_t*>(impl_->device_selected_centers->data()),
        impl_->adaptive_nprobe
            ? static_cast<float*>(impl_->device_selected_scores->data())
            : nullptr,
        impl_->nlist, impl_->params.nprobe, impl_->metric);
    checkCuda(cudaGetLastError(), "启动 ivfSelectProbesBlockKernel");
    if (impl_->adaptive_nprobe) {
      constexpr std::uint32_t kPolicyThreads = 128;
      const std::uint32_t policy_blocks =
          (batch_count + kPolicyThreads - 1) / kPolicyThreads;
      chooseAdaptiveNprobeKernel<<<policy_blocks, kPolicyThreads>>>(
          static_cast<const float*>(impl_->device_selected_scores->data()),
          static_cast<std::uint32_t*>(impl_->device_query_nprobes->data()),
          batch_count, impl_->params.nprobe, impl_->params.adaptive_nprobe_min,
          impl_->params.adaptive_nprobe_step,
          impl_->params.adaptive_target_mass,
          impl_->params.adaptive_temperature, impl_->metric);
      checkCuda(cudaGetLastError(), "启动 chooseAdaptiveNprobeKernel");
    }
    center_end.record();

    struct QueryTier {
      std::uint32_t nprobe;
      std::uint32_t offset;
      std::uint32_t count;
    };
    std::vector<QueryTier> query_tiers;
    std::vector<std::uint32_t> active_query_ids;
    if (impl_->group_adaptive_queries) {
      // 自适应策略会产生多个 nprobe tier。先取回很小的计数数组，在 CPU
      // 上稳定分组，再一次性上传 query ID；桶扫描因此不再启动空 block，
      // 同一个 kernel launch 内也不会混入工作量悬殊的 query。
      center_end.synchronize();
      const auto policy_start = std::chrono::steady_clock::now();
      checkCuda(
          cudaMemcpy(
              host_query_nprobes.data(), impl_->device_query_nprobes->data(),
              static_cast<std::size_t>(batch_count) * sizeof(std::uint32_t),
              cudaMemcpyDeviceToHost),
          "复制自适应 nprobe 到 CPU");
      std::vector<std::uint32_t> unique_nprobes(
          host_query_nprobes.begin(), host_query_nprobes.begin() + batch_count);
      std::sort(unique_nprobes.begin(), unique_nprobes.end());
      unique_nprobes.erase(
          std::unique(unique_nprobes.begin(), unique_nprobes.end()),
          unique_nprobes.end());
      active_query_ids.reserve(batch_count);
      for (const std::uint32_t tier_nprobe : unique_nprobes) {
        const std::uint32_t offset =
            static_cast<std::uint32_t>(active_query_ids.size());
        for (std::uint32_t query_id = 0; query_id < batch_count; ++query_id) {
          if (host_query_nprobes[query_id] == tier_nprobe) {
            active_query_ids.push_back(query_id);
          }
        }
        query_tiers.push_back(
            {tier_nprobe, offset,
             static_cast<std::uint32_t>(active_query_ids.size()) - offset});
      }
      checkCuda(
          cudaMemcpy(
              impl_->device_active_query_ids->data(), active_query_ids.data(),
              static_cast<std::size_t>(batch_count) * sizeof(std::uint32_t),
              cudaMemcpyHostToDevice),
          "复制自适应 query 分层到 GPU");
      const auto policy_end = std::chrono::steady_clock::now();
      measured.adaptive_policy_ms +=
          std::chrono::duration<double, std::milli>(policy_end - policy_start)
              .count();
    }

    const auto* device_ids =
        static_cast<const std::uint64_t*>(impl_->device_ids->data());
    const auto* device_offsets =
        static_cast<const std::uint64_t*>(impl_->device_offsets->data());
    const auto* selected_centers = static_cast<const std::uint32_t*>(
        impl_->device_selected_centers->data());
    auto* probe_scores =
        static_cast<float*>(impl_->device_probe_scores->data());
    auto* probe_ids =
        static_cast<std::uint64_t*>(impl_->device_probe_ids->data());
    DeviceMemoryView memory_view;
    DeviceMemoryOptions memory_options;
    if (impl_->memory_enabled) {
      memory_view = {
          static_cast<const std::uint64_t*>(impl_->device_timestamps->data()),
          static_cast<const float*>(impl_->device_importance->data()),
          static_cast<const std::uint32_t*>(impl_->device_session_ids->data()),
          static_cast<const std::uint32_t*>(
              impl_->device_source_types->data())};
      memory_options = {impl_->params.memory_semantic_weight,
                        impl_->params.memory_importance_weight,
                        impl_->params.memory_recency_weight,
                        impl_->params.memory_time_scale,
                        impl_->params.memory_now,
                        impl_->params.filter_min_timestamp,
                        impl_->params.filter_session_id,
                        impl_->params.filter_source_type};
    }
    const auto* query_nprobes = impl_->adaptive_nprobe
                                    ? static_cast<const std::uint32_t*>(
                                          impl_->device_query_nprobes->data())
                                    : nullptr;
    std::uint32_t warps_per_block = 0;
    const bool bucket_major = impl_->bucket_major;
    const std::uint32_t partitioned_topk_warps =
        impl_->partitioned_topk_warps;
    const std::uint64_t* bucket_tasks = nullptr;
    if (bucket_major) {
      const std::uint32_t task_count = batch_count * impl_->params.nprobe;
      auto* center_counts = static_cast<std::uint32_t*>(
          impl_->device_bucket_task_counts->data());
      auto* center_cursors = static_cast<std::uint32_t*>(
          impl_->device_bucket_task_cursors->data());
      auto* mutable_bucket_tasks = static_cast<std::uint64_t*>(
          impl_->device_bucket_tasks->data());
      checkCuda(cudaMemsetAsync(center_counts, 0,
                                static_cast<std::size_t>(impl_->nlist) *
                                    sizeof(std::uint32_t)),
                "清零 bucket-major 任务计数");
      const std::uint32_t task_blocks =
          (task_count + kThreads - 1) / kThreads;
      countSelectedCenterTasksKernel<<<task_blocks, kThreads>>>(
          selected_centers, center_counts, task_count);
      prepareCenterTaskCursorsKernel<<<1, 1>>>(
          center_counts, center_cursors, impl_->nlist);
      scatterSelectedCenterTasksKernel<<<task_blocks, kThreads>>>(
          selected_centers, center_cursors, mutable_bucket_tasks,
          impl_->params.nprobe, task_count);
      bucket_tasks = mutable_bucket_tasks;
    }
    if (impl_->params.distance_mode == "warp" ||
        impl_->params.distance_mode == "warp8") {
      warps_per_block = 8;
    } else if (impl_->params.distance_mode == "warp1") {
      warps_per_block = 1;
    } else if (impl_->params.distance_mode == "warp4") {
      warps_per_block = 4;
    } else if (impl_->params.distance_mode == "warp2") {
      warps_per_block = 2;
    } else if (impl_->params.distance_mode == "warp_compact" ||
               bucket_major) {
      // K10 保持高距离计算并行度；K50/K100 逐步减少每个 probe 重复维护的
      // heap 数。K100 的单 warp 版本在 Ada 上减少了候选插入与排序开销。
      warps_per_block = impl_->scan_top_k <= 10   ? 8
                        : impl_->scan_top_k <= 50 ? 2
                                                  : 1;
    }
    const auto launch_scan = [&](dim3 scan_grid,
                                 const std::uint32_t* active_ids) {
      if (impl_->use_fp16) {
        const auto* vectors =
            static_cast<const __half*>(impl_->device_vectors->data());
        const auto* device_queries =
            static_cast<const __half*>(impl_->device_queries->data());
        if (impl_->scan_top_k <= 10) {
          launchIvfBucketScan<__half, 10>(
              scan_grid, vectors, device_ids, device_offsets, selected_centers,
              bucket_tasks, device_queries, probe_scores, probe_ids, active_ids,
              query_nprobes, impl_->params.nprobe, impl_->dim,
              impl_->scan_top_k, impl_->metric, warps_per_block,
              bucket_major, partitioned_topk_warps,
              impl_->partitioned_topk_heap_k, impl_->fused_memory, memory_view,
              memory_options);
        } else if (impl_->scan_top_k <= 50) {
          launchIvfBucketScan<__half, 50>(
              scan_grid, vectors, device_ids, device_offsets, selected_centers,
              bucket_tasks, device_queries, probe_scores, probe_ids, active_ids,
              query_nprobes, impl_->params.nprobe, impl_->dim,
              impl_->scan_top_k, impl_->metric, warps_per_block,
              bucket_major, partitioned_topk_warps,
              impl_->partitioned_topk_heap_k, impl_->fused_memory, memory_view,
              memory_options);
        } else {
          launchIvfBucketScan<__half, 100>(
              scan_grid, vectors, device_ids, device_offsets, selected_centers,
              bucket_tasks, device_queries, probe_scores, probe_ids, active_ids,
              query_nprobes, impl_->params.nprobe, impl_->dim,
              impl_->scan_top_k, impl_->metric, warps_per_block,
              bucket_major, partitioned_topk_warps,
              impl_->partitioned_topk_heap_k, impl_->fused_memory, memory_view,
              memory_options);
        }
      } else {
        const auto* vectors =
            static_cast<const float*>(impl_->device_vectors->data());
        const auto* device_queries =
            static_cast<const float*>(impl_->device_queries->data());
        if (impl_->scan_top_k <= 10) {
          launchIvfBucketScan<float, 10>(
              scan_grid, vectors, device_ids, device_offsets, selected_centers,
              bucket_tasks, device_queries, probe_scores, probe_ids, active_ids,
              query_nprobes, impl_->params.nprobe, impl_->dim,
              impl_->scan_top_k, impl_->metric, warps_per_block,
              bucket_major, partitioned_topk_warps,
              impl_->partitioned_topk_heap_k, impl_->fused_memory, memory_view,
              memory_options);
        } else if (impl_->scan_top_k <= 50) {
          launchIvfBucketScan<float, 50>(
              scan_grid, vectors, device_ids, device_offsets, selected_centers,
              bucket_tasks, device_queries, probe_scores, probe_ids, active_ids,
              query_nprobes, impl_->params.nprobe, impl_->dim,
              impl_->scan_top_k, impl_->metric, warps_per_block,
              bucket_major, partitioned_topk_warps,
              impl_->partitioned_topk_heap_k, impl_->fused_memory, memory_view,
              memory_options);
        } else {
          launchIvfBucketScan<float, 100>(
              scan_grid, vectors, device_ids, device_offsets, selected_centers,
              bucket_tasks, device_queries, probe_scores, probe_ids, active_ids,
              query_nprobes, impl_->params.nprobe, impl_->dim,
              impl_->scan_top_k, impl_->metric, warps_per_block,
              bucket_major, partitioned_topk_warps,
              impl_->partitioned_topk_heap_k, impl_->fused_memory, memory_view,
              memory_options);
        }
      }
    };
    if (impl_->group_adaptive_queries) {
      const auto* grouped_query_ids = static_cast<const std::uint32_t*>(
          impl_->device_active_query_ids->data());
      for (const QueryTier& tier : query_tiers) {
        launch_scan(bucket_major
                        ? dim3(tier.count, impl_->host_index->nlist)
                        : dim3(tier.nprobe, tier.count),
                    grouped_query_ids + tier.offset);
      }
    } else {
      launch_scan(bucket_major
                      ? dim3(batch_count * impl_->params.nprobe)
                      : dim3(impl_->params.nprobe, batch_count),
                  nullptr);
    }
    checkCuda(cudaGetLastError(), "启动 IVF bucket scan kernel");
    scan_end.record();

    const std::size_t merge_shared_bytes =
        static_cast<std::size_t>(impl_->params.nprobe) * sizeof(std::uint32_t);
    ivfMergeProbeTopKKernel<<<batch_count, kThreads, merge_shared_bytes>>>(
        static_cast<const float*>(impl_->device_probe_scores->data()),
        static_cast<const std::uint64_t*>(impl_->device_probe_ids->data()),
        static_cast<float*>(impl_->device_top_scores->data()),
        static_cast<std::uint64_t*>(impl_->device_top_ids->data()),
        query_nprobes, impl_->params.nprobe, impl_->scan_top_k, impl_->metric);
    checkCuda(cudaGetLastError(), "启动 ivfMergeProbeTopKKernel");
    merge_end.record();

    const float* result_scores =
        static_cast<const float*>(impl_->device_top_scores->data());
    const std::uint64_t* result_ids =
        static_cast<const std::uint64_t*>(impl_->device_top_ids->data());
    if (impl_->separate_rerank) {
      rerankMemoryTopKKernel<<<batch_count, 1>>>(
          result_scores, result_ids,
          static_cast<float*>(impl_->device_probe_scores->data()),
          static_cast<std::uint64_t*>(impl_->device_probe_ids->data()),
          impl_->scan_top_k, impl_->params.top_k, impl_->metric, memory_view,
          memory_options);
      checkCuda(cudaGetLastError(), "启动 rerankMemoryTopKKernel");
      result_scores =
          static_cast<const float*>(impl_->device_probe_scores->data());
      result_ids =
          static_cast<const std::uint64_t*>(impl_->device_probe_ids->data());
    }
    rerank_end.record();

    const std::size_t result_count =
        static_cast<std::size_t>(batch_count) * impl_->params.top_k;
    checkCuda(cudaMemcpy(host_top_scores.data(), result_scores,
                         result_count * sizeof(float), cudaMemcpyDeviceToHost),
              "复制 IVF Top-K 分数到 CPU");
    checkCuda(cudaMemcpy(host_top_ids.data(), result_ids,
                         result_count * sizeof(std::uint64_t),
                         cudaMemcpyDeviceToHost),
              "复制 IVF Top-K ID 到 CPU");
    if (stats != nullptr && impl_->adaptive_nprobe &&
        !impl_->group_adaptive_queries) {
      checkCuda(
          cudaMemcpy(
              host_query_nprobes.data(), impl_->device_query_nprobes->data(),
              static_cast<std::size_t>(batch_count) * sizeof(std::uint32_t),
              cudaMemcpyDeviceToHost),
          "复制自适应 nprobe 到 CPU");
    }
    result_end.record();
    result_end.synchronize();

    measured.query_h2d_ms += elapsedMilliseconds(copy_start, copy_end);
    measured.center_selection_ms += elapsedMilliseconds(copy_end, center_end);
    measured.distance_kernel_ms += elapsedMilliseconds(center_end, scan_end);
    measured.topk_kernel_ms += elapsedMilliseconds(scan_end, merge_end);
    if (impl_->separate_rerank) {
      measured.memory_rerank_ms += elapsedMilliseconds(merge_end, rerank_end);
    }
    measured.result_d2h_ms += elapsedMilliseconds(rerank_end, result_end);
    measured.batch_latency_ms.push_back(
        elapsedMilliseconds(copy_start, result_end));
    if (stats != nullptr) {
      for (std::uint32_t query_in_batch = 0; query_in_batch < batch_count;
           ++query_in_batch) {
        const std::uint32_t selected = impl_->adaptive_nprobe
                                           ? host_query_nprobes[query_in_batch]
                                           : impl_->params.nprobe;
        if (measured.selected_probe_queries == 0) {
          measured.selected_probe_min = selected;
        } else {
          measured.selected_probe_min =
              std::min(measured.selected_probe_min, selected);
        }
        measured.selected_probe_max =
            std::max(measured.selected_probe_max, selected);
        measured.selected_probe_sum += selected;
        ++measured.selected_probe_queries;
        measured.selected_probe_counts.push_back(selected);
      }
    }
    for (std::uint32_t query_in_batch = 0; query_in_batch < batch_count;
         ++query_in_batch) {
      std::vector<Neighbor>& query_results =
          results[batch_start + query_in_batch];
      query_results.reserve(impl_->params.top_k);
      const std::size_t offset =
          static_cast<std::size_t>(query_in_batch) * impl_->params.top_k;
      for (std::uint32_t rank = 0; rank < impl_->params.top_k; ++rank) {
        if (host_top_ids[offset + rank] == UINT64_MAX) {
          break;
        }
        query_results.push_back(
            {host_top_ids[offset + rank], host_top_scores[offset + rank]});
      }
    }
  }
  if (stats != nullptr) {
    *stats = measured;
  }
  return results;
}

SearchResults gpuIvfFlatSearch(const IvfFlatIndex& index,
                               const QuerySet& queries,
                               const SearchParams& params, SearchStats* stats,
                               const MemoryMetadata* metadata) {
  GpuIvfFlatSearchEngine engine(index, queries, params, metadata);
  SearchResults results = engine.search(queries, stats);
  if (stats != nullptr) {
    stats->database_h2d_ms = engine.indexH2DMilliseconds();
  }
  return results;
}
