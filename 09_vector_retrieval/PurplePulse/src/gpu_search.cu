#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "file_io.h"
#include "search.h"

// 这是第一版 baseline：一个 CUDA 线程计算一对 query/vector 的完整距离。
// 优点是代码直观，缺点是暂时没有在线程间共同处理一个向量。
__device__ __forceinline__ float loadInput(const float* values,
                                           std::uint64_t index) {
  return values[index];
}

__device__ __forceinline__ float loadInput(const __half* values,
                                           std::uint64_t index) {
  return __half2float(values[index]);
}

template <typename InputType>
__global__ void exactDistanceKernel(const InputType* database,
                                    const InputType* queries, float* scores,
                                    std::uint64_t num_vectors,
                                    std::uint32_t dim,
                                    std::uint32_t batch_count, Metric metric) {
  const std::uint64_t vector_id =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::uint32_t query_in_batch = blockIdx.y;
  if (vector_id >= num_vectors || query_in_batch >= batch_count) {
    return;
  }

  const std::uint64_t vector_offset = vector_id * dim;
  const std::uint64_t query_offset =
      static_cast<std::uint64_t>(query_in_batch) * dim;

  float dot = 0.0F;
  float query_norm = 0.0F;
  float vector_norm = 0.0F;
  float squared_l2 = 0.0F;
  for (std::uint32_t d = 0; d < dim; ++d) {
    const float q = loadInput(queries, query_offset + d);
    const float x = loadInput(database, vector_offset + d);
    if (metric == Metric::kL2) {
      const float difference = q - x;
      squared_l2 = fmaf(difference, difference, squared_l2);
    } else {
      dot = fmaf(q, x, dot);
      if (metric == Metric::kCosine) {
        query_norm = fmaf(q, q, query_norm);
        vector_norm = fmaf(x, x, vector_norm);
      }
    }
  }

  float score = dot;
  if (metric == Metric::kL2) {
    score = squared_l2;
  } else if (metric == Metric::kCosine) {
    score = (query_norm == 0.0F || vector_norm == 0.0F)
                ? 0.0F
                : dot * rsqrtf(query_norm * vector_norm);
  }
  scores[static_cast<std::uint64_t>(query_in_batch) * num_vectors + vector_id] =
      score;
}

// 优化版距离计算：一个 warp（32 个线程）共同处理一个向量。
// lane 0 计算维度 0、32、64...，lane 1 计算 1、33、65...，因此访问是连续的。
template <typename InputType>
__global__ void exactDistanceWarpKernel(const InputType* database,
                                        const InputType* queries, float* scores,
                                        std::uint64_t num_vectors,
                                        std::uint32_t dim,
                                        std::uint32_t batch_count,
                                        Metric metric) {
  constexpr std::uint32_t kWarpSize = 32;
  const std::uint32_t lane = threadIdx.x % kWarpSize;
  const std::uint32_t warp_in_block = threadIdx.x / kWarpSize;
  const std::uint32_t warps_per_block = blockDim.x / kWarpSize;
  const std::uint64_t first_vector_id =
      static_cast<std::uint64_t>(blockIdx.x) * warps_per_block + warp_in_block;
  const std::uint32_t query_in_batch = blockIdx.y;
  if (query_in_batch >= batch_count) {
    return;
  }
  const std::uint64_t query_offset =
      static_cast<std::uint64_t>(query_in_batch) * dim;

  // grid-stride 循环让同一个 warp 连续处理多个向量，避免启动过多 block。
  const std::uint64_t vector_stride =
      static_cast<std::uint64_t>(gridDim.x) * warps_per_block;
  for (std::uint64_t vector_id = first_vector_id; vector_id < num_vectors;
       vector_id += vector_stride) {
    const std::uint64_t vector_offset = vector_id * dim;
    float dot = 0.0F;
    float query_norm = 0.0F;
    float vector_norm = 0.0F;
    float squared_l2 = 0.0F;

    for (std::uint32_t d = lane; d < dim; d += kWarpSize) {
      const float q = loadInput(queries, query_offset + d);
      const float x = loadInput(database, vector_offset + d);
      if (metric == Metric::kL2) {
        const float difference = q - x;
        squared_l2 = fmaf(difference, difference, squared_l2);
      } else {
        dot = fmaf(q, x, dot);
        if (metric == Metric::kCosine) {
          query_norm = fmaf(q, q, query_norm);
          vector_norm = fmaf(x, x, vector_norm);
        }
      }
    }

    // 每次让后半部分 lane 的结果加到前半部分，最终总和位于 lane 0。
    for (std::uint32_t offset = kWarpSize / 2; offset > 0; offset >>= 1) {
      squared_l2 += __shfl_down_sync(0xffffffff, squared_l2, offset);
      dot += __shfl_down_sync(0xffffffff, dot, offset);
      query_norm += __shfl_down_sync(0xffffffff, query_norm, offset);
      vector_norm += __shfl_down_sync(0xffffffff, vector_norm, offset);
    }

    if (lane == 0) {
      float score = dot;
      if (metric == Metric::kL2) {
        score = squared_l2;
      } else if (metric == Metric::kCosine) {
        score = (query_norm == 0.0F || vector_norm == 0.0F)
                    ? 0.0F
                    : dot * rsqrtf(query_norm * vector_norm);
      }
      scores[static_cast<std::uint64_t>(query_in_batch) * num_vectors +
             vector_id] = score;
    }
  }
}

__device__ bool isBetter(float candidate_score, std::uint64_t candidate_id,
                         float current_score, std::uint64_t current_id,
                         Metric metric) {
  if (candidate_score == current_score) {
    return candidate_id < current_id;
  }
  if (metric == Metric::kL2) {
    return candidate_score < current_score;
  }
  return candidate_score > current_score;
}

// 教学版 GPU Top-K：每个 query 使用一个线程，按顺序维护有序的前 K 个结果。
// 它的并行度不高，但逻辑清楚，并且避免把完整距离矩阵复制回 CPU。
// 下一版会用一个 block 共同处理一个 query，作为明确的性能优化步骤。
__global__ void simpleTopKKernel(const float* scores, float* top_scores,
                                 std::uint64_t* top_ids,
                                 std::uint64_t num_vectors, std::uint32_t top_k,
                                 Metric metric) {
  const std::uint32_t query_in_batch = blockIdx.x;
  if (threadIdx.x != 0) {
    return;
  }

  constexpr std::uint32_t kMaximumK = 100;
  float best_scores[kMaximumK];
  std::uint64_t best_ids[kMaximumK];
  // 使用 float 最大值作为哨兵，避免依赖额外的 CUDA 数学常量头文件。
  constexpr float kFloatMaximum = 3.402823466e+38F;
  const float worst_score =
      (metric == Metric::kL2) ? kFloatMaximum : -kFloatMaximum;
  for (std::uint32_t rank = 0; rank < top_k; ++rank) {
    best_scores[rank] = worst_score;
    best_ids[rank] = UINT64_MAX;
  }

  const float* query_scores =
      scores + static_cast<std::uint64_t>(query_in_batch) * num_vectors;
  for (std::uint64_t vector_id = 0; vector_id < num_vectors; ++vector_id) {
    const float score = query_scores[vector_id];
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
      static_cast<std::uint64_t>(query_in_batch) * top_k;
  for (std::uint32_t rank = 0; rank < top_k; ++rank) {
    top_scores[output_offset + rank] = best_scores[rank];
    top_ids[output_offset + rank] = best_ids[rank];
  }
}

// 并行版 GPU Top-K：一个 block 共同处理一个 query。
// 每一轮中，各线程先在自己负责的候选中找最好结果，再用共享内存归约出全局最好结果。
// 找到一个结果后将其标记为“最差”，重复 K 轮。这不是最终最优算法，但比单线程版本
// 更容易理解，也能直接展示线程协作带来的收益。
__global__ void blockTopKKernel(float* scores,
                                const std::uint64_t* candidate_ids,
                                float* top_scores, std::uint64_t* top_ids,
                                std::uint64_t num_vectors, std::uint32_t top_k,
                                Metric metric) {
  constexpr std::uint32_t kThreads = 256;
  __shared__ float shared_scores[kThreads];
  __shared__ std::uint64_t shared_ids[kThreads];
  __shared__ std::uint64_t shared_positions[kThreads];

  const std::uint32_t query_in_batch = blockIdx.x;
  const std::uint32_t thread_id = threadIdx.x;
  float* query_scores =
      scores + static_cast<std::uint64_t>(query_in_batch) * num_vectors;
  constexpr float kFloatMaximum = 3.402823466e+38F;
  const float worst_score =
      (metric == Metric::kL2) ? kFloatMaximum : -kFloatMaximum;

  for (std::uint32_t rank = 0; rank < top_k; ++rank) {
    float local_best_score = worst_score;
    std::uint64_t local_best_id = UINT64_MAX;
    std::uint64_t local_best_position = UINT64_MAX;

    for (std::uint64_t position = thread_id; position < num_vectors;
         position += blockDim.x) {
      const float score = query_scores[position];
      const std::uint64_t vector_id =
          candidate_ids == nullptr
              ? position
              : candidate_ids[static_cast<std::uint64_t>(query_in_batch) *
                                  num_vectors +
                              position];
      if (isBetter(score, vector_id, local_best_score, local_best_id, metric)) {
        local_best_score = score;
        local_best_id = vector_id;
        local_best_position = position;
      }
    }

    shared_scores[thread_id] = local_best_score;
    shared_ids[thread_id] = local_best_id;
    shared_positions[thread_id] = local_best_position;
    __syncthreads();

    for (std::uint32_t stride = kThreads / 2; stride > 0; stride >>= 1) {
      if (thread_id < stride &&
          isBetter(shared_scores[thread_id + stride],
                   shared_ids[thread_id + stride], shared_scores[thread_id],
                   shared_ids[thread_id], metric)) {
        shared_scores[thread_id] = shared_scores[thread_id + stride];
        shared_ids[thread_id] = shared_ids[thread_id + stride];
        shared_positions[thread_id] = shared_positions[thread_id + stride];
      }
      __syncthreads();
    }

    if (thread_id == 0) {
      const std::uint64_t output_offset =
          static_cast<std::uint64_t>(query_in_batch) * top_k + rank;
      top_scores[output_offset] = shared_scores[0];
      top_ids[output_offset] = shared_ids[0];
      query_scores[shared_positions[0]] = worst_score;
    }
    __syncthreads();
  }
}

// 两阶段 Top-K 的第一阶段：每个线程只扫描自己负责的候选，并保留本线程的 Top-K。
// 局部数组不需要有序：绝大多数候选只与当前最差项比较一次；只有候选真正
// 进入 Top-K 时，才扫描 K 个槽位重新找最差项。256 个线程产生 256*K 个候选后，
// 再交给 blockTopKKernel 做最终有序归并。
__global__ void localTopKCandidatesKernel(const float* scores,
                                          float* local_scores,
                                          std::uint64_t* local_ids,
                                          std::uint64_t num_vectors,
                                          std::uint32_t top_k, Metric metric) {
  constexpr std::uint32_t kMaximumLocalK = 10;
  const std::uint32_t query_in_batch = blockIdx.x;
  const std::uint32_t thread_id = threadIdx.x;
  constexpr float kFloatMaximum = 3.402823466e+38F;
  const float worst_score =
      (metric == Metric::kL2) ? kFloatMaximum : -kFloatMaximum;

  float best_scores[kMaximumLocalK];
  std::uint64_t best_ids[kMaximumLocalK];
  for (std::uint32_t rank = 0; rank < top_k; ++rank) {
    best_scores[rank] = worst_score;
    best_ids[rank] = UINT64_MAX;
  }
  std::uint32_t worst_rank = 0;

  const float* query_scores =
      scores + static_cast<std::uint64_t>(query_in_batch) * num_vectors;
  for (std::uint64_t vector_id = thread_id; vector_id < num_vectors;
       vector_id += blockDim.x) {
    const float score = query_scores[vector_id];
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
      (static_cast<std::uint64_t>(query_in_batch) * blockDim.x + thread_id) *
      top_k;
  for (std::uint32_t rank = 0; rank < top_k; ++rank) {
    local_scores[output_offset + rank] = best_scores[rank];
    local_ids[output_offset + rank] = best_ids[rank];
  }
}

// K=1/10 是项目的主要评测档。将 K 变成编译期常量后，编译器可以展开
// 局部槽位扫描，并尽量避免通用 kernel 因动态数组索引产生的 thread stack。
template <std::uint32_t TopK>
__global__ void localTopKCandidatesFixedKernel(const float* scores,
                                               float* local_scores,
                                               std::uint64_t* local_ids,
                                               std::uint64_t num_vectors,
                                               Metric metric) {
  static_assert(TopK >= 1 && TopK <= 10, "fixed Top-K 必须位于 [1, 10]");
  const std::uint32_t query_in_batch = blockIdx.x;
  const std::uint32_t thread_id = threadIdx.x;
  constexpr float kFloatMaximum = 3.402823466e+38F;
  const float worst_score =
      (metric == Metric::kL2) ? kFloatMaximum : -kFloatMaximum;

  float best_scores[TopK];
  std::uint64_t best_ids[TopK];
#pragma unroll
  for (std::uint32_t rank = 0; rank < TopK; ++rank) {
    best_scores[rank] = worst_score;
    best_ids[rank] = UINT64_MAX;
  }
  std::uint32_t worst_rank = 0;

  const float* query_scores =
      scores + static_cast<std::uint64_t>(query_in_batch) * num_vectors;
  for (std::uint64_t vector_id = thread_id; vector_id < num_vectors;
       vector_id += blockDim.x) {
    const float score = query_scores[vector_id];
    if (isBetter(score, vector_id, best_scores[worst_rank],
                 best_ids[worst_rank], metric)) {
      best_scores[worst_rank] = score;
      best_ids[worst_rank] = vector_id;
      worst_rank = 0;
#pragma unroll
      for (std::uint32_t rank = 1; rank < TopK; ++rank) {
        if (isBetter(best_scores[worst_rank], best_ids[worst_rank],
                     best_scores[rank], best_ids[rank], metric)) {
          worst_rank = rank;
        }
      }
    }
  }

  const std::uint64_t output_offset =
      (static_cast<std::uint64_t>(query_in_batch) * blockDim.x + thread_id) *
      TopK;
#pragma unroll
  for (std::uint32_t rank = 0; rank < TopK; ++rank) {
    local_scores[output_offset + rank] = best_scores[rank];
    local_ids[output_offset + rank] = best_ids[rank];
  }
}

// Fused exact scan: each block owns one query/chunk pair and contains eight
// cooperative warps. A warp computes one vector score and maintains a shared
// Top-K heap. The block emits only K candidates, so the full batch*N score
// matrix is never materialized. A small final merge combines all chunks.
//
// The distance-only kernel immediately below is a diagnostic differential
// probe. It intentionally mirrors the fused traversal and arithmetic but
// replaces heap maintenance with one checksum per warp. Comparing its event
// time with the full fused kernel estimates local-selection cost without
// changing the production search path or materializing the score matrix.
template <typename InputType, Metric MetricValue>
__global__ void exactFusedDistanceOnlyKernel(
    const InputType* database, const InputType* queries, float* warp_checksums,
    std::uint64_t num_vectors, std::uint32_t dim, bool cache_query) {
  constexpr std::uint32_t kWarpSize = 32;
  constexpr std::uint32_t kWarps = 8;
  extern __shared__ __align__(16) unsigned char dynamic_shared[];
  auto* shared_query = reinterpret_cast<InputType*>(dynamic_shared);

  const std::uint32_t thread_id = threadIdx.x;
  const std::uint32_t lane = thread_id % kWarpSize;
  const std::uint32_t warp = thread_id / kWarpSize;
  const std::uint32_t query_id = blockIdx.y;
  const InputType* query_values =
      queries + static_cast<std::uint64_t>(query_id) * dim;
  if (cache_query) {
    for (std::uint32_t d = thread_id; d < dim; d += blockDim.x) {
      shared_query[d] = query_values[d];
    }
  }
  __syncthreads();
  if (cache_query) {
    query_values = shared_query;
  }

  float query_norm = 0.0F;
  if constexpr (MetricValue == Metric::kCosine) {
    for (std::uint32_t d = lane; d < dim; d += kWarpSize) {
      const float query = loadInput(query_values, d);
      query_norm = fmaf(query, query, query_norm);
    }
    for (std::uint32_t offset = kWarpSize / 2; offset > 0; offset >>= 1) {
      query_norm += __shfl_down_sync(0xffffffffU, query_norm, offset);
    }
  }

  float checksum = 0.0F;
  const std::uint64_t first_vector =
      static_cast<std::uint64_t>(blockIdx.x) * kWarps + warp;
  const std::uint64_t vector_stride =
      static_cast<std::uint64_t>(gridDim.x) * kWarps;
  for (std::uint64_t vector_id = first_vector; vector_id < num_vectors;
       vector_id += vector_stride) {
    const std::uint64_t vector_offset = vector_id * dim;
    float dot = 0.0F;
    float vector_norm = 0.0F;
    float squared_l2 = 0.0F;
    for (std::uint32_t d = lane; d < dim; d += kWarpSize) {
      const float query = loadInput(query_values, d);
      const float vector = loadInput(database, vector_offset + d);
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
    for (std::uint32_t offset = kWarpSize / 2; offset > 0; offset >>= 1) {
      if constexpr (MetricValue == Metric::kL2) {
        squared_l2 += __shfl_down_sync(0xffffffffU, squared_l2, offset);
      } else {
        dot += __shfl_down_sync(0xffffffffU, dot, offset);
        if constexpr (MetricValue == Metric::kCosine) {
          vector_norm += __shfl_down_sync(0xffffffffU, vector_norm, offset);
        }
      }
    }
    if (lane == 0) {
      float score = dot;
      if constexpr (MetricValue == Metric::kL2) {
        score = squared_l2;
      } else if constexpr (MetricValue == Metric::kCosine) {
        score = query_norm == 0.0F || vector_norm == 0.0F
                    ? 0.0F
                    : dot * rsqrtf(query_norm * vector_norm);
      }
      checksum += score;
    }
  }
  if (lane == 0) {
    const std::uint64_t output =
        (static_cast<std::uint64_t>(query_id) * gridDim.x + blockIdx.x) *
            kWarps +
        warp;
    warp_checksums[output] = checksum;
  }
}

template <typename InputType, std::uint32_t MaximumK, Metric MetricValue>
__global__ void exactFusedWarpTopKKernel(
    const InputType* database, const InputType* queries, float* local_scores,
    std::uint64_t* local_ids, std::uint64_t num_vectors, std::uint32_t dim,
    std::uint32_t top_k, bool cache_query) {
  constexpr std::uint32_t kWarpSize = 32;
  constexpr std::uint32_t kWarps = 8;
  constexpr float kFloatMaximum = 3.402823466e+38F;
  __shared__ float warp_scores[kWarps][MaximumK];
  __shared__ std::uint64_t warp_ids[kWarps][MaximumK];
  __shared__ float merge_scores[kWarps];
  __shared__ std::uint64_t merge_ids[kWarps];
  __shared__ std::uint32_t warp_ranks[kWarps];
  extern __shared__ __align__(16) unsigned char dynamic_shared[];
  auto* shared_query = reinterpret_cast<InputType*>(dynamic_shared);

  const std::uint32_t thread_id = threadIdx.x;
  const std::uint32_t lane = thread_id % kWarpSize;
  const std::uint32_t warp = thread_id / kWarpSize;
  const std::uint32_t query_id = blockIdx.y;
  const float worst_score =
      MetricValue == Metric::kL2 ? kFloatMaximum : -kFloatMaximum;
  if (lane == 0) {
    for (std::uint32_t rank = 0; rank < top_k; ++rank) {
      warp_scores[warp][rank] = worst_score;
      warp_ids[warp][rank] = UINT64_MAX;
    }
  }

  const InputType* query_values =
      queries + static_cast<std::uint64_t>(query_id) * dim;
  if (cache_query) {
    for (std::uint32_t d = thread_id; d < dim; d += blockDim.x) {
      shared_query[d] = query_values[d];
    }
  }
  __syncthreads();
  if (cache_query) {
    query_values = shared_query;
  }

  float query_norm = 0.0F;
  if constexpr (MetricValue == Metric::kCosine) {
    for (std::uint32_t d = lane; d < dim; d += kWarpSize) {
      const float query = loadInput(query_values, d);
      query_norm = fmaf(query, query, query_norm);
    }
    for (std::uint32_t offset = kWarpSize / 2; offset > 0; offset >>= 1) {
      query_norm += __shfl_down_sync(0xffffffffU, query_norm, offset);
    }
  }

  const std::uint64_t first_vector =
      static_cast<std::uint64_t>(blockIdx.x) * kWarps + warp;
  const std::uint64_t vector_stride =
      static_cast<std::uint64_t>(gridDim.x) * kWarps;
  for (std::uint64_t vector_id = first_vector; vector_id < num_vectors;
       vector_id += vector_stride) {
    const std::uint64_t vector_offset = vector_id * dim;
    float dot = 0.0F;
    float vector_norm = 0.0F;
    float squared_l2 = 0.0F;
    for (std::uint32_t d = lane; d < dim; d += kWarpSize) {
      const float query = loadInput(query_values, d);
      const float vector = loadInput(database, vector_offset + d);
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
    for (std::uint32_t offset = kWarpSize / 2; offset > 0; offset >>= 1) {
      if constexpr (MetricValue == Metric::kL2) {
        squared_l2 += __shfl_down_sync(0xffffffffU, squared_l2, offset);
      } else {
        dot += __shfl_down_sync(0xffffffffU, dot, offset);
        if constexpr (MetricValue == Metric::kCosine) {
          vector_norm += __shfl_down_sync(0xffffffffU, vector_norm, offset);
        }
      }
    }
    if (lane == 0) {
      float score = dot;
      if constexpr (MetricValue == Metric::kL2) {
        score = squared_l2;
      } else if constexpr (MetricValue == Metric::kCosine) {
        score = query_norm == 0.0F || vector_norm == 0.0F
                    ? 0.0F
                    : dot * rsqrtf(query_norm * vector_norm);
      }
      if (isBetter(score, vector_id, warp_scores[warp][0], warp_ids[warp][0],
                   MetricValue)) {
        warp_scores[warp][0] = score;
        warp_ids[warp][0] = vector_id;
        std::uint32_t parent = 0;
        while (true) {
          const std::uint32_t left = parent * 2 + 1;
          if (left >= top_k) {
            break;
          }
          const std::uint32_t right = left + 1;
          std::uint32_t worse_child = left;
          if (right < top_k &&
              isBetter(warp_scores[warp][left], warp_ids[warp][left],
                       warp_scores[warp][right], warp_ids[warp][right],
                       MetricValue)) {
            worse_child = right;
          }
          if (!isBetter(warp_scores[warp][parent], warp_ids[warp][parent],
                        warp_scores[warp][worse_child],
                        warp_ids[warp][worse_child], MetricValue)) {
            break;
          }
          const float swap_score = warp_scores[warp][parent];
          const std::uint64_t swap_id = warp_ids[warp][parent];
          warp_scores[warp][parent] = warp_scores[warp][worse_child];
          warp_ids[warp][parent] = warp_ids[warp][worse_child];
          warp_scores[warp][worse_child] = swap_score;
          warp_ids[warp][worse_child] = swap_id;
          parent = worse_child;
        }
      }
    }
  }
  __syncthreads();

  // Heap sort each warp into best-to-worst order.
  if (lane == 0) {
    for (std::uint32_t heap_size = top_k; heap_size > 1; --heap_size) {
      const std::uint32_t last = heap_size - 1;
      const float swap_score = warp_scores[warp][0];
      const std::uint64_t swap_id = warp_ids[warp][0];
      warp_scores[warp][0] = warp_scores[warp][last];
      warp_ids[warp][0] = warp_ids[warp][last];
      warp_scores[warp][last] = swap_score;
      warp_ids[warp][last] = swap_id;
      std::uint32_t parent = 0;
      while (true) {
        const std::uint32_t left = parent * 2 + 1;
        if (left >= last) {
          break;
        }
        const std::uint32_t right = left + 1;
        std::uint32_t worse_child = left;
        if (right < last &&
            isBetter(warp_scores[warp][left], warp_ids[warp][left],
                     warp_scores[warp][right], warp_ids[warp][right],
                     MetricValue)) {
          worse_child = right;
        }
        if (!isBetter(warp_scores[warp][parent], warp_ids[warp][parent],
                      warp_scores[warp][worse_child],
                      warp_ids[warp][worse_child], MetricValue)) {
          break;
        }
        const float parent_score = warp_scores[warp][parent];
        const std::uint64_t parent_id = warp_ids[warp][parent];
        warp_scores[warp][parent] = warp_scores[warp][worse_child];
        warp_ids[warp][parent] = warp_ids[warp][worse_child];
        warp_scores[warp][worse_child] = parent_score;
        warp_ids[warp][worse_child] = parent_id;
        parent = worse_child;
      }
    }
    warp_ranks[warp] = 0;
  }
  __syncthreads();

  for (std::uint32_t output_rank = 0; output_rank < top_k; ++output_rank) {
    if (lane == 0) {
      const std::uint32_t rank = warp_ranks[warp];
      merge_scores[warp] = warp_scores[warp][rank];
      merge_ids[warp] = warp_ids[warp][rank];
    }
    __syncthreads();
    if (thread_id == 0) {
      std::uint32_t best_warp = 0;
      for (std::uint32_t candidate_warp = 1; candidate_warp < kWarps;
           ++candidate_warp) {
        if (isBetter(merge_scores[candidate_warp], merge_ids[candidate_warp],
                     merge_scores[best_warp], merge_ids[best_warp],
                     MetricValue)) {
          best_warp = candidate_warp;
        }
      }
      const std::uint64_t output_offset =
          (static_cast<std::uint64_t>(query_id) * gridDim.x + blockIdx.x) *
              top_k +
          output_rank;
      local_scores[output_offset] = merge_scores[best_warp];
      local_ids[output_offset] = merge_ids[best_warp];
      ++warp_ranks[best_warp];
    }
    __syncthreads();
  }
}

// Experimental cross-query tile. A block owns QueryTile queries for the same
// vector chunk, so every database element loaded by a lane is reused for all
// queries in the tile. QueryTile=1 continues to use the established kernel
// above; keeping the paths separate makes the A/B reversible.
template <typename InputType, std::uint32_t MaximumK, Metric MetricValue,
          std::uint32_t QueryTile>
__global__ void exactFusedQueryTiledTopKKernel(
    const InputType* database, const InputType* queries, float* local_scores,
    std::uint64_t* local_ids, std::uint64_t num_vectors, std::uint32_t dim,
    std::uint32_t batch_count, std::uint32_t top_k, bool cache_queries) {
  static_assert(QueryTile == 2 || QueryTile == 4 || QueryTile == 8,
                "query tile must be 2, 4, or 8");
  constexpr std::uint32_t kWarpSize = 32;
  constexpr std::uint32_t kWarps = 8;
  constexpr float kFloatMaximum = 3.402823466e+38F;
  __shared__ float warp_scores[QueryTile][kWarps][MaximumK];
  __shared__ std::uint64_t warp_ids[QueryTile][kWarps][MaximumK];
  __shared__ float merge_scores[QueryTile][kWarps];
  __shared__ std::uint64_t merge_ids[QueryTile][kWarps];
  __shared__ std::uint32_t warp_ranks[QueryTile][kWarps];
  extern __shared__ __align__(16) unsigned char dynamic_shared[];
  auto* shared_queries = reinterpret_cast<InputType*>(dynamic_shared);

  const std::uint32_t thread_id = threadIdx.x;
  const std::uint32_t lane = thread_id % kWarpSize;
  const std::uint32_t warp = thread_id / kWarpSize;
  const std::uint32_t query_base = blockIdx.y * QueryTile;
  const std::uint32_t remaining_queries =
      query_base < batch_count ? batch_count - query_base : 0;
  const std::uint32_t query_count =
      remaining_queries < QueryTile ? remaining_queries : QueryTile;
  const float worst_score =
      MetricValue == Metric::kL2 ? kFloatMaximum : -kFloatMaximum;

  if (lane == 0) {
    for (std::uint32_t query = 0; query < query_count; ++query) {
      for (std::uint32_t rank = 0; rank < top_k; ++rank) {
        warp_scores[query][warp][rank] = worst_score;
        warp_ids[query][warp][rank] = UINT64_MAX;
      }
    }
  }

  const InputType* query_values[QueryTile];
#pragma unroll
  for (std::uint32_t query = 0; query < QueryTile; ++query) {
    query_values[query] =
        query < query_count
            ? queries + static_cast<std::uint64_t>(query_base + query) * dim
            : queries;
  }
  if (cache_queries) {
    const std::uint32_t tile_values = query_count * dim;
    for (std::uint32_t index = thread_id; index < tile_values;
         index += blockDim.x) {
      const std::uint32_t query = index / dim;
      const std::uint32_t d = index - query * dim;
      shared_queries[index] = query_values[query][d];
    }
  }
  __syncthreads();
  if (cache_queries) {
#pragma unroll
    for (std::uint32_t query = 0; query < QueryTile; ++query) {
      query_values[query] = shared_queries + query * dim;
    }
  }

  float query_norm[QueryTile] = {};
  if constexpr (MetricValue == Metric::kCosine) {
#pragma unroll
    for (std::uint32_t query = 0; query < QueryTile; ++query) {
      if (query < query_count) {
        for (std::uint32_t d = lane; d < dim; d += kWarpSize) {
          const float value = loadInput(query_values[query], d);
          query_norm[query] = fmaf(value, value, query_norm[query]);
        }
        for (std::uint32_t offset = kWarpSize / 2; offset > 0; offset >>= 1) {
          query_norm[query] +=
              __shfl_down_sync(0xffffffffU, query_norm[query], offset);
        }
      }
    }
  }

  const std::uint64_t first_vector =
      static_cast<std::uint64_t>(blockIdx.x) * kWarps + warp;
  const std::uint64_t vector_stride =
      static_cast<std::uint64_t>(gridDim.x) * kWarps;
  for (std::uint64_t vector_id = first_vector; vector_id < num_vectors;
       vector_id += vector_stride) {
    const std::uint64_t vector_offset = vector_id * dim;
    float dot[QueryTile] = {};
    float squared_l2[QueryTile] = {};
    float vector_norm = 0.0F;
    for (std::uint32_t d = lane; d < dim; d += kWarpSize) {
      const float vector = loadInput(database, vector_offset + d);
      if constexpr (MetricValue == Metric::kCosine) {
        vector_norm = fmaf(vector, vector, vector_norm);
      }
#pragma unroll
      for (std::uint32_t query = 0; query < QueryTile; ++query) {
        if (query < query_count) {
          const float query_value = loadInput(query_values[query], d);
          if constexpr (MetricValue == Metric::kL2) {
            const float difference = query_value - vector;
            squared_l2[query] =
                fmaf(difference, difference, squared_l2[query]);
          } else {
            dot[query] = fmaf(query_value, vector, dot[query]);
          }
        }
      }
    }
    for (std::uint32_t offset = kWarpSize / 2; offset > 0; offset >>= 1) {
      if constexpr (MetricValue == Metric::kCosine) {
        vector_norm += __shfl_down_sync(0xffffffffU, vector_norm, offset);
      }
#pragma unroll
      for (std::uint32_t query = 0; query < QueryTile; ++query) {
        if (query < query_count) {
          if constexpr (MetricValue == Metric::kL2) {
            squared_l2[query] += __shfl_down_sync(
                0xffffffffU, squared_l2[query], offset);
          } else {
            dot[query] +=
                __shfl_down_sync(0xffffffffU, dot[query], offset);
          }
        }
      }
    }
    if (lane == 0) {
#pragma unroll
      for (std::uint32_t query = 0; query < QueryTile; ++query) {
        if (query >= query_count) {
          continue;
        }
        float score = dot[query];
        if constexpr (MetricValue == Metric::kL2) {
          score = squared_l2[query];
        } else if constexpr (MetricValue == Metric::kCosine) {
          score = query_norm[query] == 0.0F || vector_norm == 0.0F
                      ? 0.0F
                      : dot[query] *
                            rsqrtf(query_norm[query] * vector_norm);
        }
        if (isBetter(score, vector_id, warp_scores[query][warp][0],
                     warp_ids[query][warp][0], MetricValue)) {
          warp_scores[query][warp][0] = score;
          warp_ids[query][warp][0] = vector_id;
          std::uint32_t parent = 0;
          while (true) {
            const std::uint32_t left = parent * 2 + 1;
            if (left >= top_k) {
              break;
            }
            const std::uint32_t right = left + 1;
            std::uint32_t worse_child = left;
            if (right < top_k &&
                isBetter(warp_scores[query][warp][left],
                         warp_ids[query][warp][left],
                         warp_scores[query][warp][right],
                         warp_ids[query][warp][right], MetricValue)) {
              worse_child = right;
            }
            if (!isBetter(warp_scores[query][warp][parent],
                          warp_ids[query][warp][parent],
                          warp_scores[query][warp][worse_child],
                          warp_ids[query][warp][worse_child], MetricValue)) {
              break;
            }
            const float swap_score = warp_scores[query][warp][parent];
            const std::uint64_t swap_id = warp_ids[query][warp][parent];
            warp_scores[query][warp][parent] =
                warp_scores[query][warp][worse_child];
            warp_ids[query][warp][parent] =
                warp_ids[query][warp][worse_child];
            warp_scores[query][warp][worse_child] = swap_score;
            warp_ids[query][warp][worse_child] = swap_id;
            parent = worse_child;
          }
        }
      }
    }
  }
  __syncthreads();

  if (lane == 0) {
    for (std::uint32_t query = 0; query < query_count; ++query) {
      for (std::uint32_t heap_size = top_k; heap_size > 1; --heap_size) {
        const std::uint32_t last = heap_size - 1;
        const float swap_score = warp_scores[query][warp][0];
        const std::uint64_t swap_id = warp_ids[query][warp][0];
        warp_scores[query][warp][0] = warp_scores[query][warp][last];
        warp_ids[query][warp][0] = warp_ids[query][warp][last];
        warp_scores[query][warp][last] = swap_score;
        warp_ids[query][warp][last] = swap_id;
        std::uint32_t parent = 0;
        while (true) {
          const std::uint32_t left = parent * 2 + 1;
          if (left >= last) {
            break;
          }
          const std::uint32_t right = left + 1;
          std::uint32_t worse_child = left;
          if (right < last &&
              isBetter(warp_scores[query][warp][left],
                       warp_ids[query][warp][left],
                       warp_scores[query][warp][right],
                       warp_ids[query][warp][right], MetricValue)) {
            worse_child = right;
          }
          if (!isBetter(warp_scores[query][warp][parent],
                        warp_ids[query][warp][parent],
                        warp_scores[query][warp][worse_child],
                        warp_ids[query][warp][worse_child], MetricValue)) {
            break;
          }
          const float parent_score = warp_scores[query][warp][parent];
          const std::uint64_t parent_id = warp_ids[query][warp][parent];
          warp_scores[query][warp][parent] =
              warp_scores[query][warp][worse_child];
          warp_ids[query][warp][parent] = warp_ids[query][warp][worse_child];
          warp_scores[query][warp][worse_child] = parent_score;
          warp_ids[query][warp][worse_child] = parent_id;
          parent = worse_child;
        }
      }
      warp_ranks[query][warp] = 0;
    }
  }
  __syncthreads();

  for (std::uint32_t query = 0; query < query_count; ++query) {
    const std::uint32_t query_id = query_base + query;
    for (std::uint32_t output_rank = 0; output_rank < top_k; ++output_rank) {
      if (lane == 0) {
        const std::uint32_t rank = warp_ranks[query][warp];
        merge_scores[query][warp] = warp_scores[query][warp][rank];
        merge_ids[query][warp] = warp_ids[query][warp][rank];
      }
      __syncthreads();
      if (thread_id == 0) {
        std::uint32_t best_warp = 0;
        for (std::uint32_t candidate_warp = 1; candidate_warp < kWarps;
             ++candidate_warp) {
          if (isBetter(merge_scores[query][candidate_warp],
                       merge_ids[query][candidate_warp],
                       merge_scores[query][best_warp],
                       merge_ids[query][best_warp], MetricValue)) {
            best_warp = candidate_warp;
          }
        }
        const std::uint64_t output_offset =
            (static_cast<std::uint64_t>(query_id) * gridDim.x + blockIdx.x) *
                top_k +
            output_rank;
        local_scores[output_offset] = merge_scores[query][best_warp];
        local_ids[output_offset] = merge_ids[query][best_warp];
        ++warp_ranks[query][best_warp];
      }
      __syncthreads();
    }
  }
}

// Block-tiled variant: one warp owns one query while all eight query warps
// reuse a shared tile of eight database vectors. This preserves query-level
// parallelism, unlike the warp-query tile above where one warp serially
// accumulates several queries.
template <typename InputType, std::uint32_t MaximumK, Metric MetricValue>
__global__ void exactFusedBlockTiledTopKKernel(
    const InputType* database, const InputType* queries, float* local_scores,
    std::uint64_t* local_ids, std::uint64_t num_vectors, std::uint32_t dim,
    std::uint32_t batch_count, std::uint32_t top_k) {
  constexpr std::uint32_t kWarpSize = 32;
  constexpr std::uint32_t kWarps = 8;
  constexpr std::uint32_t kQueryTile = 8;
  constexpr std::uint32_t kVectorTile = 8;
  constexpr float kFloatMaximum = 3.402823466e+38F;
  __shared__ float warp_scores[kWarps][MaximumK];
  __shared__ std::uint64_t warp_ids[kWarps][MaximumK];
  __shared__ float shared_vector_norms[kVectorTile];
  extern __shared__ __align__(16) unsigned char dynamic_shared[];
  auto* shared_queries = reinterpret_cast<InputType*>(dynamic_shared);
  InputType* shared_database = shared_queries + kQueryTile * dim;

  const std::uint32_t thread_id = threadIdx.x;
  const std::uint32_t lane = thread_id % kWarpSize;
  const std::uint32_t warp = thread_id / kWarpSize;
  const std::uint32_t query_base = blockIdx.y * kQueryTile;
  const std::uint32_t remaining_queries =
      query_base < batch_count ? batch_count - query_base : 0;
  const std::uint32_t query_count =
      remaining_queries < kQueryTile ? remaining_queries : kQueryTile;
  const bool valid_query = warp < query_count;
  const float worst_score =
      MetricValue == Metric::kL2 ? kFloatMaximum : -kFloatMaximum;

  if (valid_query && lane == 0) {
    for (std::uint32_t rank = 0; rank < top_k; ++rank) {
      warp_scores[warp][rank] = worst_score;
      warp_ids[warp][rank] = UINT64_MAX;
    }
  }

  const std::uint32_t query_values = query_count * dim;
  for (std::uint32_t index = thread_id; index < query_values;
       index += blockDim.x) {
    const std::uint32_t query = index / dim;
    const std::uint32_t d = index - query * dim;
    shared_queries[index] =
        queries[static_cast<std::uint64_t>(query_base + query) * dim + d];
  }
  __syncthreads();

  float query_norm = 0.0F;
  if constexpr (MetricValue == Metric::kCosine) {
    if (valid_query) {
      for (std::uint32_t d = lane; d < dim; d += kWarpSize) {
        const float query =
            loadInput(shared_queries + static_cast<std::uint64_t>(warp) * dim,
                      d);
        query_norm = fmaf(query, query, query_norm);
      }
      for (std::uint32_t offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        query_norm += __shfl_down_sync(0xffffffffU, query_norm, offset);
      }
    }
  }

  const std::uint64_t first_tile =
      static_cast<std::uint64_t>(blockIdx.x) * kVectorTile;
  const std::uint64_t tile_stride =
      static_cast<std::uint64_t>(gridDim.x) * kVectorTile;
  for (std::uint64_t tile_base = first_tile; tile_base < num_vectors;
       tile_base += tile_stride) {
    const std::uint32_t tile_count = static_cast<std::uint32_t>(
        num_vectors - tile_base < kVectorTile ? num_vectors - tile_base
                                              : kVectorTile);
    const std::uint32_t database_values = tile_count * dim;
    for (std::uint32_t index = thread_id; index < database_values;
         index += blockDim.x) {
      const std::uint32_t vector = index / dim;
      const std::uint32_t d = index - vector * dim;
      shared_database[index] =
          database[(tile_base + vector) * static_cast<std::uint64_t>(dim) + d];
    }
    __syncthreads();

    if constexpr (MetricValue == Metric::kCosine) {
      if (warp < tile_count) {
        float vector_norm = 0.0F;
        for (std::uint32_t d = lane; d < dim; d += kWarpSize) {
          const float vector = loadInput(
              shared_database + static_cast<std::uint64_t>(warp) * dim, d);
          vector_norm = fmaf(vector, vector, vector_norm);
        }
        for (std::uint32_t offset = kWarpSize / 2; offset > 0; offset >>= 1) {
          vector_norm +=
              __shfl_down_sync(0xffffffffU, vector_norm, offset);
        }
        if (lane == 0) {
          shared_vector_norms[warp] = vector_norm;
        }
      }
      __syncthreads();
    }

    float dot[kVectorTile] = {};
    float squared_l2[kVectorTile] = {};
    if (valid_query) {
      const InputType* query_values_ptr =
          shared_queries + static_cast<std::uint64_t>(warp) * dim;
      for (std::uint32_t d = lane; d < dim; d += kWarpSize) {
        const float query = loadInput(query_values_ptr, d);
#pragma unroll
        for (std::uint32_t vector = 0; vector < kVectorTile; ++vector) {
          if (vector < tile_count) {
            const float value = loadInput(
                shared_database + static_cast<std::uint64_t>(vector) * dim, d);
            if constexpr (MetricValue == Metric::kL2) {
              const float difference = query - value;
              squared_l2[vector] =
                  fmaf(difference, difference, squared_l2[vector]);
            } else {
              dot[vector] = fmaf(query, value, dot[vector]);
            }
          }
        }
      }
      for (std::uint32_t offset = kWarpSize / 2; offset > 0; offset >>= 1) {
#pragma unroll
        for (std::uint32_t vector = 0; vector < kVectorTile; ++vector) {
          if (vector < tile_count) {
            if constexpr (MetricValue == Metric::kL2) {
              squared_l2[vector] += __shfl_down_sync(
                  0xffffffffU, squared_l2[vector], offset);
            } else {
              dot[vector] +=
                  __shfl_down_sync(0xffffffffU, dot[vector], offset);
            }
          }
        }
      }
      if (lane == 0) {
        for (std::uint32_t vector = 0; vector < tile_count; ++vector) {
          const std::uint64_t vector_id = tile_base + vector;
          float score = dot[vector];
          if constexpr (MetricValue == Metric::kL2) {
            score = squared_l2[vector];
          } else if constexpr (MetricValue == Metric::kCosine) {
            const float vector_norm = shared_vector_norms[vector];
            score = query_norm == 0.0F || vector_norm == 0.0F
                        ? 0.0F
                        : dot[vector] * rsqrtf(query_norm * vector_norm);
          }
          if (isBetter(score, vector_id, warp_scores[warp][0],
                       warp_ids[warp][0], MetricValue)) {
            warp_scores[warp][0] = score;
            warp_ids[warp][0] = vector_id;
            std::uint32_t parent = 0;
            while (true) {
              const std::uint32_t left = parent * 2 + 1;
              if (left >= top_k) {
                break;
              }
              const std::uint32_t right = left + 1;
              std::uint32_t worse_child = left;
              if (right < top_k &&
                  isBetter(warp_scores[warp][left], warp_ids[warp][left],
                           warp_scores[warp][right], warp_ids[warp][right],
                           MetricValue)) {
                worse_child = right;
              }
              if (!isBetter(warp_scores[warp][parent], warp_ids[warp][parent],
                            warp_scores[warp][worse_child],
                            warp_ids[warp][worse_child], MetricValue)) {
                break;
              }
              const float swap_score = warp_scores[warp][parent];
              const std::uint64_t swap_id = warp_ids[warp][parent];
              warp_scores[warp][parent] = warp_scores[warp][worse_child];
              warp_ids[warp][parent] = warp_ids[warp][worse_child];
              warp_scores[warp][worse_child] = swap_score;
              warp_ids[warp][worse_child] = swap_id;
              parent = worse_child;
            }
          }
        }
      }
    }
    __syncthreads();
  }

  if (valid_query && lane == 0) {
    for (std::uint32_t heap_size = top_k; heap_size > 1; --heap_size) {
      const std::uint32_t last = heap_size - 1;
      const float swap_score = warp_scores[warp][0];
      const std::uint64_t swap_id = warp_ids[warp][0];
      warp_scores[warp][0] = warp_scores[warp][last];
      warp_ids[warp][0] = warp_ids[warp][last];
      warp_scores[warp][last] = swap_score;
      warp_ids[warp][last] = swap_id;
      std::uint32_t parent = 0;
      while (true) {
        const std::uint32_t left = parent * 2 + 1;
        if (left >= last) {
          break;
        }
        const std::uint32_t right = left + 1;
        std::uint32_t worse_child = left;
        if (right < last &&
            isBetter(warp_scores[warp][left], warp_ids[warp][left],
                     warp_scores[warp][right], warp_ids[warp][right],
                     MetricValue)) {
          worse_child = right;
        }
        if (!isBetter(warp_scores[warp][parent], warp_ids[warp][parent],
                      warp_scores[warp][worse_child],
                      warp_ids[warp][worse_child], MetricValue)) {
          break;
        }
        const float parent_score = warp_scores[warp][parent];
        const std::uint64_t parent_id = warp_ids[warp][parent];
        warp_scores[warp][parent] = warp_scores[warp][worse_child];
        warp_ids[warp][parent] = warp_ids[warp][worse_child];
        warp_scores[warp][worse_child] = parent_score;
        warp_ids[warp][worse_child] = parent_id;
        parent = worse_child;
      }
    }
    const std::uint32_t query_id = query_base + warp;
    const std::uint64_t output_offset =
        (static_cast<std::uint64_t>(query_id) * gridDim.x + blockIdx.x) *
        top_k;
    for (std::uint32_t rank = 0; rank < top_k; ++rank) {
      local_scores[output_offset + rank] = warp_scores[warp][rank];
      local_ids[output_offset + rank] = warp_ids[warp][rank];
    }
  }
}

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
    checkCuda(cudaMalloc(&data_, bytes), "cudaMalloc");
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
  CudaEvent() { checkCuda(cudaEventCreate(&event_), "cudaEventCreate"); }
  ~CudaEvent() { cudaEventDestroy(event_); }

  CudaEvent(const CudaEvent&) = delete;
  CudaEvent& operator=(const CudaEvent&) = delete;

  void record() { checkCuda(cudaEventRecord(event_), "cudaEventRecord"); }
  void synchronize() {
    checkCuda(cudaEventSynchronize(event_), "cudaEventSynchronize");
  }
  cudaEvent_t get() const { return event_; }

 private:
  cudaEvent_t event_{};
};

double elapsedMilliseconds(const CudaEvent& start, const CudaEvent& end) {
  float milliseconds = 0.0F;
  checkCuda(cudaEventElapsedTime(&milliseconds, start.get(), end.get()),
            "cudaEventElapsedTime");
  return milliseconds;
}

template <typename InputType, std::uint32_t MaximumK>
void launchExactFusedWarpTopK(dim3 grid, const InputType* database,
                              const InputType* queries, float* local_scores,
                              std::uint64_t* local_ids,
                              std::uint64_t num_vectors, std::uint32_t dim,
                              std::uint32_t top_k, Metric metric) {
  constexpr std::uint32_t kThreads = 256;
  constexpr std::size_t kMaximumCachedQueryBytes = 8192;
  const std::size_t query_bytes =
      static_cast<std::size_t>(dim) * sizeof(InputType);
  const bool cache_query = query_bytes <= kMaximumCachedQueryBytes;
  const std::size_t shared_bytes = cache_query ? query_bytes : 0;
  if (metric == Metric::kL2) {
    exactFusedWarpTopKKernel<InputType, MaximumK, Metric::kL2>
        <<<grid, kThreads, shared_bytes>>>(database, queries, local_scores,
                                           local_ids, num_vectors, dim, top_k,
                                           cache_query);
  } else if (metric == Metric::kInnerProduct) {
    exactFusedWarpTopKKernel<InputType, MaximumK, Metric::kInnerProduct>
        <<<grid, kThreads, shared_bytes>>>(database, queries, local_scores,
                                           local_ids, num_vectors, dim, top_k,
                                           cache_query);
  } else {
    exactFusedWarpTopKKernel<InputType, MaximumK, Metric::kCosine>
        <<<grid, kThreads, shared_bytes>>>(database, queries, local_scores,
                                           local_ids, num_vectors, dim, top_k,
                                           cache_query);
  }
}

template <typename InputType, std::uint32_t MaximumK,
          std::uint32_t QueryTile>
void launchExactFusedQueryTiledTopK(
    dim3 grid, const InputType* database, const InputType* queries,
    float* local_scores, std::uint64_t* local_ids, std::uint64_t num_vectors,
    std::uint32_t dim, std::uint32_t batch_count, std::uint32_t top_k,
    Metric metric) {
  constexpr std::uint32_t kThreads = 256;
  constexpr std::size_t kMaximumCachedTileBytes = 8192;
  const std::size_t tile_bytes = static_cast<std::size_t>(QueryTile) * dim *
                                 sizeof(InputType);
  const bool cache_queries = tile_bytes <= kMaximumCachedTileBytes;
  const std::size_t shared_bytes = cache_queries ? tile_bytes : 0;
  if (metric == Metric::kL2) {
    exactFusedQueryTiledTopKKernel<InputType, MaximumK, Metric::kL2,
                                   QueryTile>
        <<<grid, kThreads, shared_bytes>>>(
            database, queries, local_scores, local_ids, num_vectors, dim,
            batch_count, top_k, cache_queries);
  } else if (metric == Metric::kInnerProduct) {
    exactFusedQueryTiledTopKKernel<InputType, MaximumK,
                                   Metric::kInnerProduct, QueryTile>
        <<<grid, kThreads, shared_bytes>>>(
            database, queries, local_scores, local_ids, num_vectors, dim,
            batch_count, top_k, cache_queries);
  } else {
    exactFusedQueryTiledTopKKernel<InputType, MaximumK, Metric::kCosine,
                                   QueryTile>
        <<<grid, kThreads, shared_bytes>>>(
            database, queries, local_scores, local_ids, num_vectors, dim,
            batch_count, top_k, cache_queries);
  }
}

template <typename InputType, std::uint32_t MaximumK>
void launchExactFusedBlockTiledTopK(
    dim3 grid, const InputType* database, const InputType* queries,
    float* local_scores, std::uint64_t* local_ids, std::uint64_t num_vectors,
    std::uint32_t dim, std::uint32_t batch_count, std::uint32_t top_k,
    Metric metric) {
  constexpr std::uint32_t kThreads = 256;
  constexpr std::uint32_t kQueryTile = 8;
  constexpr std::uint32_t kVectorTile = 8;
  const std::size_t shared_bytes =
      static_cast<std::size_t>(kQueryTile + kVectorTile) * dim *
      sizeof(InputType);
  if (metric == Metric::kL2) {
    exactFusedBlockTiledTopKKernel<InputType, MaximumK, Metric::kL2>
        <<<grid, kThreads, shared_bytes>>>(
            database, queries, local_scores, local_ids, num_vectors, dim,
            batch_count, top_k);
  } else if (metric == Metric::kInnerProduct) {
    exactFusedBlockTiledTopKKernel<InputType, MaximumK,
                                   Metric::kInnerProduct>
        <<<grid, kThreads, shared_bytes>>>(
            database, queries, local_scores, local_ids, num_vectors, dim,
            batch_count, top_k);
  } else {
    exactFusedBlockTiledTopKKernel<InputType, MaximumK, Metric::kCosine>
        <<<grid, kThreads, shared_bytes>>>(
            database, queries, local_scores, local_ids, num_vectors, dim,
            batch_count, top_k);
  }
}

template <typename InputType, std::uint32_t MaximumK>
void launchExactFusedWarpTopKWithQueryTile(
    dim3 grid, const InputType* database, const InputType* queries,
    float* local_scores, std::uint64_t* local_ids, std::uint64_t num_vectors,
    std::uint32_t dim, std::uint32_t batch_count, std::uint32_t top_k,
    std::uint32_t query_tile, bool block_tiled, Metric metric) {
  if (block_tiled) {
    if constexpr (MaximumK <= 10) {
      launchExactFusedBlockTiledTopK<InputType, MaximumK>(
          grid, database, queries, local_scores, local_ids, num_vectors, dim,
          batch_count, top_k, metric);
    } else {
      throw std::runtime_error(
          "block_tile cannot be instantiated for MaximumK > 10");
    }
    return;
  }
  if (query_tile == 1) {
    launchExactFusedWarpTopK<InputType, MaximumK>(
        grid, database, queries, local_scores, local_ids, num_vectors, dim,
        top_k, metric);
  } else if (query_tile == 2) {
    launchExactFusedQueryTiledTopK<InputType, MaximumK, 2>(
        grid, database, queries, local_scores, local_ids, num_vectors, dim,
        batch_count, top_k, metric);
  } else if (query_tile == 4) {
    launchExactFusedQueryTiledTopK<InputType, MaximumK, 4>(
        grid, database, queries, local_scores, local_ids, num_vectors, dim,
        batch_count, top_k, metric);
  } else {
    if constexpr (MaximumK <= 10) {
      launchExactFusedQueryTiledTopK<InputType, MaximumK, 8>(
          grid, database, queries, local_scores, local_ids, num_vectors, dim,
          batch_count, top_k, metric);
    } else {
      throw std::runtime_error(
          "exact_query_tile=8 cannot be instantiated for MaximumK > 10");
    }
  }
}

template <typename InputType>
void launchExactFusedDistanceOnly(dim3 grid, const InputType* database,
                                  const InputType* queries,
                                  float* warp_checksums,
                                  std::uint64_t num_vectors,
                                  std::uint32_t dim, Metric metric) {
  constexpr std::uint32_t kThreads = 256;
  constexpr std::size_t kMaximumCachedQueryBytes = 8192;
  const std::size_t query_bytes =
      static_cast<std::size_t>(dim) * sizeof(InputType);
  const bool cache_query = query_bytes <= kMaximumCachedQueryBytes;
  const std::size_t shared_bytes = cache_query ? query_bytes : 0;
  if (metric == Metric::kL2) {
    exactFusedDistanceOnlyKernel<InputType, Metric::kL2>
        <<<grid, kThreads, shared_bytes>>>(database, queries, warp_checksums,
                                           num_vectors, dim, cache_query);
  } else if (metric == Metric::kInnerProduct) {
    exactFusedDistanceOnlyKernel<InputType, Metric::kInnerProduct>
        <<<grid, kThreads, shared_bytes>>>(database, queries, warp_checksums,
                                           num_vectors, dim, cache_query);
  } else {
    exactFusedDistanceOnlyKernel<InputType, Metric::kCosine>
        <<<grid, kThreads, shared_bytes>>>(database, queries, warp_checksums,
                                           num_vectors, dim, cache_query);
  }
}

}  // namespace

struct GpuExactSearchEngine::Impl {
  explicit Impl(const VectorDatabase& database, const QuerySet& queries,
                const SearchParams& search_params)
      : num_vectors(database.num_vectors),
        dim(database.dim),
        dtype(database.dtype),
        metric(database.metric),
        params(search_params),
        fused_topk(search_params.topk_mode == "fused"),
        use_fp16(database.dtype == DataType::kFloat16),
        input_element_bytes(use_fp16 ? sizeof(std::uint16_t) : sizeof(float)),
        batch_capacity(search_params.batch_size) {
    validateInputs(database, queries, params);
    constexpr std::uint32_t kThreads = 256;
    const std::size_t database_value_count =
        static_cast<std::size_t>(num_vectors) * dim;
    const std::size_t database_bytes =
        database_value_count * input_element_bytes;
    const std::size_t query_bytes =
        static_cast<std::size_t>(batch_capacity) * dim * input_element_bytes;
    const std::size_t score_bytes =
        fused_topk ? 0
                   : static_cast<std::size_t>(batch_capacity) * num_vectors *
                         sizeof(float);
    const std::size_t result_count =
        static_cast<std::size_t>(batch_capacity) * params.top_k;
    const std::uint32_t required_fused_blocks =
        static_cast<std::uint32_t>((num_vectors + 8 - 1) / 8);
    // Keep enough query/chunk blocks in flight to occupy the GPU, without
    // producing 256*K candidates per query for large batches.
    // Query tiling reduces grid.y. Scale grid.x by the same factor so the
    // experiment keeps roughly the established 1024 resident/work blocks and
    // does not confuse data reuse with a simple loss of parallelism.
    const std::uint32_t target_blocks = 1024U * params.exact_query_tile;
    const std::uint32_t occupancy_blocks_per_query = std::max<std::uint32_t>(
        1, (target_blocks + batch_capacity - 1) / batch_capacity);
    const std::uint32_t selected_fused_blocks =
        params.exact_fused_blocks == 0 ? occupancy_blocks_per_query
                                       : params.exact_fused_blocks;
    fused_blocks_x =
        std::min(required_fused_blocks, selected_fused_blocks);
    const std::uint32_t local_blocks = fused_topk ? fused_blocks_x : kThreads;
    const std::size_t local_candidate_count =
        uses_local_candidates ? result_count * local_blocks : 0;
    device_bytes =
        database_bytes + query_bytes + score_bytes +
        result_count * (sizeof(float) + sizeof(std::uint64_t)) +
        local_candidate_count * (sizeof(float) + sizeof(std::uint64_t));

    device_database = std::make_unique<DeviceBuffer>(database_bytes);
    device_queries = std::make_unique<DeviceBuffer>(query_bytes);
    if (!fused_topk) {
      device_scores = std::make_unique<DeviceBuffer>(score_bytes);
    }
    device_top_scores =
        std::make_unique<DeviceBuffer>(result_count * sizeof(float));
    device_top_ids =
        std::make_unique<DeviceBuffer>(result_count * sizeof(std::uint64_t));
    if (uses_local_candidates) {
      device_local_scores =
          std::make_unique<DeviceBuffer>(local_candidate_count * sizeof(float));
      device_local_ids = std::make_unique<DeviceBuffer>(local_candidate_count *
                                                        sizeof(std::uint64_t));
    }

    CudaEvent database_copy_start;
    CudaEvent database_copy_end;
    database_copy_start.record();
    const void* database_data =
        use_fp16 ? static_cast<const void*>(database.half_values.data())
                 : static_cast<const void*>(database.values.data());
    checkCuda(cudaMemcpy(device_database->data(), database_data, database_bytes,
                         cudaMemcpyHostToDevice),
              "复制向量库到 GPU");
    database_copy_end.record();
    database_copy_end.synchronize();
    database_h2d_ms =
        elapsedMilliseconds(database_copy_start, database_copy_end);
  }

  std::uint64_t num_vectors;
  std::uint32_t dim;
  DataType dtype;
  Metric metric;
  SearchParams params;
  bool fused_topk;
  bool uses_local_candidates =
      params.topk_mode == "two_stage" || params.topk_mode == "fused";
  bool use_fp16;
  std::size_t input_element_bytes;
  std::uint32_t batch_capacity;
  std::uint32_t fused_blocks_x = 0;
  double database_h2d_ms = 0.0;
  std::size_t device_bytes = 0;
  std::unique_ptr<DeviceBuffer> device_database;
  std::unique_ptr<DeviceBuffer> device_queries;
  std::unique_ptr<DeviceBuffer> device_scores;
  std::unique_ptr<DeviceBuffer> device_top_scores;
  std::unique_ptr<DeviceBuffer> device_top_ids;
  std::unique_ptr<DeviceBuffer> device_local_scores;
  std::unique_ptr<DeviceBuffer> device_local_ids;
};

GpuExactSearchEngine::GpuExactSearchEngine(const VectorDatabase& database,
                                           const QuerySet& initial_queries,
                                           const SearchParams& params)
    : impl_(std::make_unique<Impl>(database, initial_queries, params)) {}

GpuExactSearchEngine::~GpuExactSearchEngine() = default;
GpuExactSearchEngine::GpuExactSearchEngine(GpuExactSearchEngine&&) noexcept =
    default;
GpuExactSearchEngine& GpuExactSearchEngine::operator=(
    GpuExactSearchEngine&&) noexcept = default;

double GpuExactSearchEngine::databaseH2DMilliseconds() const {
  return impl_->database_h2d_ms;
}

std::size_t GpuExactSearchEngine::deviceBytes() const {
  return impl_->device_bytes;
}

double GpuExactSearchEngine::profileDistanceOnlyMilliseconds(
    const QuerySet& queries, std::uint32_t warmup, std::uint32_t repeat) {
  if (!impl_->fused_topk) {
    throw std::runtime_error("distance-only 差分剖析只支持 fused Exact 路径");
  }
  if (repeat == 0) {
    throw std::runtime_error("distance-only 剖析 repeat 必须大于 0");
  }
  if (impl_->params.exact_query_tile != 1) {
    throw std::runtime_error(
        "distance-only 差分剖析当前只支持 exact_query_tile=1");
  }
  if (impl_->params.exact_fused_layout != "warp_query") {
    throw std::runtime_error(
        "distance-only 差分剖析当前只支持 warp_query layout");
  }
  if (queries.num_queries == 0 || queries.dim != impl_->dim ||
      queries.dtype != impl_->dtype) {
    throw std::runtime_error("剖析查询集与 GPU 常驻引擎不匹配");
  }
  const std::uint64_t expected_query_values = queries.num_queries * queries.dim;
  if ((impl_->use_fp16 &&
       queries.half_values.size() != expected_query_values) ||
      (!impl_->use_fp16 && queries.values.size() != expected_query_values)) {
    throw std::runtime_error("剖析查询数据长度与元数据不一致");
  }

  constexpr std::uint32_t kWarps = 8;
  const std::size_t checksum_count =
      static_cast<std::size_t>(impl_->batch_capacity) * impl_->fused_blocks_x *
      kWarps;
  DeviceBuffer device_checksums(checksum_count * sizeof(float));
  CudaEvent kernel_start;
  CudaEvent kernel_end;

  const auto run_once = [&](bool measure) {
    double milliseconds = 0.0;
    for (std::uint64_t batch_start = 0; batch_start < queries.num_queries;
         batch_start += impl_->batch_capacity) {
      const std::uint32_t batch_count =
          static_cast<std::uint32_t>(std::min<std::uint64_t>(
              impl_->batch_capacity, queries.num_queries - batch_start));
      const std::size_t query_value_offset =
          static_cast<std::size_t>(batch_start) * queries.dim;
      const std::size_t current_query_bytes =
          static_cast<std::size_t>(batch_count) * queries.dim *
          impl_->input_element_bytes;
      const void* query_data =
          impl_->use_fp16
              ? static_cast<const void*>(queries.half_values.data() +
                                         query_value_offset)
              : static_cast<const void*>(queries.values.data() +
                                         query_value_offset);
      checkCuda(cudaMemcpy(impl_->device_queries->data(), query_data,
                           current_query_bytes, cudaMemcpyHostToDevice),
                "复制剖析查询到 GPU");

      const dim3 grid(impl_->fused_blocks_x, batch_count);
      kernel_start.record();
      if (impl_->use_fp16) {
        launchExactFusedDistanceOnly(
            grid, static_cast<const __half*>(impl_->device_database->data()),
            static_cast<const __half*>(impl_->device_queries->data()),
            static_cast<float*>(device_checksums.data()), impl_->num_vectors,
            impl_->dim, impl_->metric);
      } else {
        launchExactFusedDistanceOnly(
            grid, static_cast<const float*>(impl_->device_database->data()),
            static_cast<const float*>(impl_->device_queries->data()),
            static_cast<float*>(device_checksums.data()), impl_->num_vectors,
            impl_->dim, impl_->metric);
      }
      checkCuda(cudaGetLastError(), "启动 exactFusedDistanceOnlyKernel");
      kernel_end.record();
      kernel_end.synchronize();
      if (measure) {
        milliseconds += elapsedMilliseconds(kernel_start, kernel_end);
      }
    }
    return milliseconds;
  };

  for (std::uint32_t run = 0; run < warmup; ++run) {
    (void)run_once(false);
  }
  double total_ms = 0.0;
  for (std::uint32_t run = 0; run < repeat; ++run) {
    total_ms += run_once(true);
  }
  return total_ms / repeat;
}

SearchResults GpuExactSearchEngine::search(const QuerySet& queries,
                                           SearchStats* stats) {
  if (queries.num_queries == 0 || queries.dim != impl_->dim ||
      queries.dtype != impl_->dtype) {
    throw std::runtime_error("查询集与 GPU 常驻引擎的维度或 dtype 不匹配");
  }
  const std::uint64_t expected_query_values = queries.num_queries * queries.dim;
  if ((impl_->use_fp16 &&
       queries.half_values.size() != expected_query_values) ||
      (!impl_->use_fp16 && queries.values.size() != expected_query_values)) {
    throw std::runtime_error("查询数据长度与元数据不一致");
  }

  SearchStats measured_stats;
  const SearchParams& params = impl_->params;
  const bool use_fp16 = impl_->use_fp16;
  const std::size_t input_element_bytes = impl_->input_element_bytes;
  const std::uint32_t batch_capacity = impl_->batch_capacity;
  const std::uint64_t num_vectors = impl_->num_vectors;
  const std::uint32_t dim = impl_->dim;
  const Metric metric = impl_->metric;
  DeviceBuffer& device_database = *impl_->device_database;
  DeviceBuffer& device_queries = *impl_->device_queries;
  DeviceBuffer* device_scores = impl_->device_scores.get();
  DeviceBuffer& device_top_scores = *impl_->device_top_scores;
  DeviceBuffer& device_top_ids = *impl_->device_top_ids;
  DeviceBuffer* device_local_scores = impl_->device_local_scores.get();
  DeviceBuffer* device_local_ids = impl_->device_local_ids.get();
  constexpr std::uint32_t kThreads = 256;

  SearchResults results(queries.num_queries);
  std::vector<float> host_top_scores(static_cast<std::size_t>(batch_capacity) *
                                     params.top_k);
  std::vector<std::uint64_t> host_top_ids(
      static_cast<std::size_t>(batch_capacity) * params.top_k);
  const std::uint32_t simple_blocks_x =
      static_cast<std::uint32_t>((num_vectors + kThreads - 1) / kThreads);
  constexpr std::uint32_t kWarpsPerBlock = kThreads / 32;
  const std::uint32_t required_warp_blocks = static_cast<std::uint32_t>(
      (num_vectors + kWarpsPerBlock - 1) / kWarpsPerBlock);
  const std::uint32_t warp_blocks_x =
      std::min<std::uint32_t>(required_warp_blocks, 4096);
  const std::uint32_t fused_blocks_x = impl_->fused_blocks_x;
  const std::uint32_t exact_query_tile = params.exact_query_tile;
  const bool block_tiled = params.exact_fused_layout == "block_tile";

  CudaEvent query_copy_start;
  CudaEvent query_copy_end;
  CudaEvent distance_end;
  CudaEvent topk_end;
  CudaEvent result_copy_end;

  for (std::uint64_t batch_start = 0; batch_start < queries.num_queries;
       batch_start += batch_capacity) {
    const std::uint32_t batch_count =
        static_cast<std::uint32_t>(std::min<std::uint64_t>(
            batch_capacity, queries.num_queries - batch_start));
    const std::size_t current_query_bytes =
        static_cast<std::size_t>(batch_count) * queries.dim *
        input_element_bytes;
    const std::size_t query_value_offset =
        static_cast<std::size_t>(batch_start) * queries.dim;
    const void* query_data =
        use_fp16 ? static_cast<const void*>(queries.half_values.data() +
                                            query_value_offset)
                 : static_cast<const void*>(queries.values.data() +
                                            query_value_offset);
    query_copy_start.record();
    checkCuda(cudaMemcpy(device_queries.data(), query_data, current_query_bytes,
                         cudaMemcpyHostToDevice),
              "复制查询到 GPU");
    query_copy_end.record();

    if (params.topk_mode == "fused") {
      const dim3 grid(fused_blocks_x,
                      (batch_count + exact_query_tile - 1) /
                          exact_query_tile);
      if (use_fp16) {
        const auto* database_values =
            static_cast<const __half*>(device_database.data());
        const auto* query_values =
            static_cast<const __half*>(device_queries.data());
        if (params.top_k <= 10) {
          launchExactFusedWarpTopKWithQueryTile<__half, 10>(
              grid, database_values, query_values,
              static_cast<float*>(device_local_scores->data()),
              static_cast<std::uint64_t*>(device_local_ids->data()),
              num_vectors, dim, batch_count, params.top_k, exact_query_tile,
              block_tiled, metric);
        } else if (params.top_k <= 50) {
          launchExactFusedWarpTopKWithQueryTile<__half, 50>(
              grid, database_values, query_values,
              static_cast<float*>(device_local_scores->data()),
              static_cast<std::uint64_t*>(device_local_ids->data()),
              num_vectors, dim, batch_count, params.top_k, exact_query_tile,
              block_tiled, metric);
        } else {
          launchExactFusedWarpTopKWithQueryTile<__half, 100>(
              grid, database_values, query_values,
              static_cast<float*>(device_local_scores->data()),
              static_cast<std::uint64_t*>(device_local_ids->data()),
              num_vectors, dim, batch_count, params.top_k, exact_query_tile,
              block_tiled, metric);
        }
      } else {
        const auto* database_values =
            static_cast<const float*>(device_database.data());
        const auto* query_values =
            static_cast<const float*>(device_queries.data());
        if (params.top_k <= 10) {
          launchExactFusedWarpTopKWithQueryTile<float, 10>(
              grid, database_values, query_values,
              static_cast<float*>(device_local_scores->data()),
              static_cast<std::uint64_t*>(device_local_ids->data()),
              num_vectors, dim, batch_count, params.top_k, exact_query_tile,
              block_tiled, metric);
        } else if (params.top_k <= 50) {
          launchExactFusedWarpTopKWithQueryTile<float, 50>(
              grid, database_values, query_values,
              static_cast<float*>(device_local_scores->data()),
              static_cast<std::uint64_t*>(device_local_ids->data()),
              num_vectors, dim, batch_count, params.top_k, exact_query_tile,
              block_tiled, metric);
        } else {
          launchExactFusedWarpTopKWithQueryTile<float, 100>(
              grid, database_values, query_values,
              static_cast<float*>(device_local_scores->data()),
              static_cast<std::uint64_t*>(device_local_ids->data()),
              num_vectors, dim, batch_count, params.top_k, exact_query_tile,
              block_tiled, metric);
        }
      }
      checkCuda(cudaGetLastError(), "启动 exactFusedWarpTopKKernel");
      distance_end.record();
      const std::uint64_t candidates_per_query =
          static_cast<std::uint64_t>(fused_blocks_x) * params.top_k;
      blockTopKKernel<<<batch_count, kThreads>>>(
          static_cast<float*>(device_local_scores->data()),
          static_cast<const std::uint64_t*>(device_local_ids->data()),
          static_cast<float*>(device_top_scores.data()),
          static_cast<std::uint64_t*>(device_top_ids.data()),
          candidates_per_query, params.top_k, metric);
      checkCuda(cudaGetLastError(), "启动 fused exact 最终归并");
    } else {
      if (params.distance_mode == "simple") {
        const dim3 grid(simple_blocks_x, batch_count);
        if (use_fp16) {
          exactDistanceKernel<<<grid, kThreads>>>(
              static_cast<const __half*>(device_database.data()),
              static_cast<const __half*>(device_queries.data()),
              static_cast<float*>(device_scores->data()), num_vectors, dim,
              batch_count, metric);
        } else {
          exactDistanceKernel<<<grid, kThreads>>>(
              static_cast<const float*>(device_database.data()),
              static_cast<const float*>(device_queries.data()),
              static_cast<float*>(device_scores->data()), num_vectors, dim,
              batch_count, metric);
        }
        checkCuda(cudaGetLastError(), "启动 exactDistanceKernel");
      } else {
        const dim3 grid(warp_blocks_x, batch_count);
        if (use_fp16) {
          exactDistanceWarpKernel<<<grid, kThreads>>>(
              static_cast<const __half*>(device_database.data()),
              static_cast<const __half*>(device_queries.data()),
              static_cast<float*>(device_scores->data()), num_vectors, dim,
              batch_count, metric);
        } else {
          exactDistanceWarpKernel<<<grid, kThreads>>>(
              static_cast<const float*>(device_database.data()),
              static_cast<const float*>(device_queries.data()),
              static_cast<float*>(device_scores->data()), num_vectors, dim,
              batch_count, metric);
        }
        checkCuda(cudaGetLastError(), "启动 exactDistanceWarpKernel");
      }
      distance_end.record();

      if (params.topk_mode == "simple") {
        simpleTopKKernel<<<batch_count, 1>>>(
            static_cast<const float*>(device_scores->data()),
            static_cast<float*>(device_top_scores.data()),
            static_cast<std::uint64_t*>(device_top_ids.data()), num_vectors,
            params.top_k, metric);
        checkCuda(cudaGetLastError(), "启动 simpleTopKKernel");
      } else if (params.topk_mode == "block") {
        blockTopKKernel<<<batch_count, kThreads>>>(
            static_cast<float*>(device_scores->data()), nullptr,
            static_cast<float*>(device_top_scores.data()),
            static_cast<std::uint64_t*>(device_top_ids.data()), num_vectors,
            params.top_k, metric);
        checkCuda(cudaGetLastError(), "启动 blockTopKKernel");
      } else {
        if (params.top_k == 1) {
          localTopKCandidatesFixedKernel<1><<<batch_count, kThreads>>>(
              static_cast<const float*>(device_scores->data()),
              static_cast<float*>(device_local_scores->data()),
              static_cast<std::uint64_t*>(device_local_ids->data()),
              num_vectors, metric);
        } else if (params.top_k == 10) {
          localTopKCandidatesFixedKernel<10><<<batch_count, kThreads>>>(
              static_cast<const float*>(device_scores->data()),
              static_cast<float*>(device_local_scores->data()),
              static_cast<std::uint64_t*>(device_local_ids->data()),
              num_vectors, metric);
        } else {
          localTopKCandidatesKernel<<<batch_count, kThreads>>>(
              static_cast<const float*>(device_scores->data()),
              static_cast<float*>(device_local_scores->data()),
              static_cast<std::uint64_t*>(device_local_ids->data()),
              num_vectors, params.top_k, metric);
        }
        checkCuda(cudaGetLastError(), "启动 localTopKCandidatesKernel");

        const std::uint64_t candidates_per_query =
            static_cast<std::uint64_t>(kThreads) * params.top_k;
        blockTopKKernel<<<batch_count, kThreads>>>(
            static_cast<float*>(device_local_scores->data()),
            static_cast<const std::uint64_t*>(device_local_ids->data()),
            static_cast<float*>(device_top_scores.data()),
            static_cast<std::uint64_t*>(device_top_ids.data()),
            candidates_per_query, params.top_k, metric);
        checkCuda(cudaGetLastError(), "启动两阶段归并 blockTopKKernel");
      }
    }
    topk_end.record();

    const std::size_t result_count =
        static_cast<std::size_t>(batch_count) * params.top_k;
    checkCuda(cudaMemcpy(host_top_scores.data(), device_top_scores.data(),
                         result_count * sizeof(float), cudaMemcpyDeviceToHost),
              "复制 Top-K 分数到 CPU");
    checkCuda(cudaMemcpy(host_top_ids.data(), device_top_ids.data(),
                         result_count * sizeof(std::uint64_t),
                         cudaMemcpyDeviceToHost),
              "复制 Top-K ID 到 CPU");
    result_copy_end.record();
    result_copy_end.synchronize();

    measured_stats.query_h2d_ms +=
        elapsedMilliseconds(query_copy_start, query_copy_end);
    measured_stats.distance_kernel_ms +=
        elapsedMilliseconds(query_copy_end, distance_end);
    measured_stats.topk_kernel_ms +=
        elapsedMilliseconds(distance_end, topk_end);
    measured_stats.result_d2h_ms +=
        elapsedMilliseconds(topk_end, result_copy_end);
    measured_stats.batch_latency_ms.push_back(
        elapsedMilliseconds(query_copy_start, result_copy_end));

    for (std::uint32_t query_in_batch = 0; query_in_batch < batch_count;
         ++query_in_batch) {
      std::vector<Neighbor>& query_results =
          results[batch_start + query_in_batch];
      query_results.resize(params.top_k);
      const std::size_t offset =
          static_cast<std::size_t>(query_in_batch) * params.top_k;
      for (std::uint32_t rank = 0; rank < params.top_k; ++rank) {
        query_results[rank] = {host_top_ids[offset + rank],
                               host_top_scores[offset + rank]};
      }
    }
  }
  if (stats != nullptr) {
    *stats = measured_stats;
  }
  return results;
}

SearchResults gpuExactSearch(const VectorDatabase& database,
                             const QuerySet& queries,
                             const SearchParams& params, SearchStats* stats) {
  GpuExactSearchEngine engine(database, queries, params);
  SearchResults results = engine.search(queries, stats);
  if (stats != nullptr) {
    stats->database_h2d_ms = engine.databaseH2DMilliseconds();
  }
  return results;
}
