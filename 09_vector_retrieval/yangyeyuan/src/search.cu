#include "vector_search.h"

#include <cuda_runtime.h>
#include <cub/block/block_radix_sort.cuh>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace vs {
namespace {

// 每个 block 的线程数;每个线程处理 4 个向量;故每 block 覆盖 1024 个候选。
constexpr int kBlockThreads = 256;
constexpr int kItemsPerThread = 4;
constexpr int kBlockItems = kBlockThreads * kItemsPerThread;

// 包装 CUDA 调用错误:失败时抛出带错误信息的异常。
void cuda_check(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
  }
}

// 分配显存(元素个数),并做溢出检查。
template <typename T>
T* device_alloc(std::size_t count) {
  if (count == 0) return nullptr;
  if (count > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
    throw std::runtime_error("device allocation size overflow");
  }
  T* pointer = nullptr;
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&pointer), count * sizeof(T)), "cudaMalloc");
  return pointer;
}

template <typename T>
void device_free(T* pointer) {
  if (pointer) cudaFree(pointer);
}

// 把磁盘的 row-major 数据转成维度主序 [dim, count] 的 SoA 布局。
// 转置后,同一 warp 在固定维度读相邻候选,global load 可合并。
__global__ void transpose_to_soa_kernel(const float* row_major, float* soa,
                                        std::uint64_t count, std::uint32_t dim) {
  const std::uint64_t i = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::uint32_t d = blockIdx.y;
  if (i < count) soa[static_cast<std::uint64_t>(d) * count + i] = row_major[i * dim + d];
}

// 把 (距离, ID) 打包成 64 位排序键: 高32位=距离位模式, 低32位=原始ID。
// 依赖 IEEE754: 非负 float 的位模式与数值同序,故按整数比大小即按距离比大小;
// 距离相同时低位 ID 自动完成 tie-break。非有限距离替换为 +Inf 的位模式,
// 使其排到末尾(对应"NaN/Inf 排末尾"规则)。
__device__ __forceinline__ std::uint64_t distance_key(float distance, std::uint32_t id) {
  if (!(distance >= 0.0f) || !isfinite(distance)) distance = __int_as_float(0x7f800000);
  return static_cast<std::uint64_t>(__float_as_uint(distance)) << 32 | id;
}

// 精确检索第一阶段:一个 block 处理 1024 个候选向量,block 内 radix sort 保留局部 Top-K。
// 网格: (候选块数, 查询数)。查询向量缓存在 shared memory。
template <int K>
__global__ void distance_partial_topk_kernel(const float* vectors_soa,
                                             std::uint64_t vector_count,
                                             std::uint32_t dim,
                                             const float* queries,
                                             std::uint32_t query_count,
                                             std::uint64_t* output) {
  using Sort = cub::BlockRadixSort<std::uint64_t, kBlockThreads, kItemsPerThread>;
  __shared__ typename Sort::TempStorage sort_storage;
  extern __shared__ float query_shared[];  // 动态共享内存,容量=dim*sizeof(float)

  const std::uint32_t q = blockIdx.y;
  if (q >= query_count) return;
  // ① 把查询向量拷进共享内存(所有线程都要读它,放共享区避免重复访存)。
  for (std::uint32_t d = threadIdx.x; d < dim; d += blockDim.x) {
    query_shared[d] = queries[static_cast<std::uint64_t>(q) * dim + d];
  }
  __syncthreads();

  // ② 每个线程算 4 个候选的距离,打包成排序键。
  std::uint64_t keys[kItemsPerThread];
#pragma unroll
  for (int item = 0; item < kItemsPerThread; ++item) {
    const std::uint64_t candidate = static_cast<std::uint64_t>(blockIdx.x) * kBlockItems +
                                    item * kBlockThreads + threadIdx.x;
    if (candidate < vector_count) {
      // FMA 累加: delta*delta + distance 一条指令完成,减少舍入。
      float distance = 0.0f;
      for (std::uint32_t d = 0; d < dim; ++d) {
        const float delta = query_shared[d] - vectors_soa[static_cast<std::uint64_t>(d) * vector_count + candidate];
        distance = fmaf(delta, delta, distance);
      }
      keys[item] = distance_key(distance, static_cast<std::uint32_t>(candidate));
    } else {
      keys[item] = 0xffffffffffffffffull;  // 越界候选填最大键,必然排到末尾
    }
  }
  // ③ block 内基数排序,把 1024 个键排成升序。
  Sort(sort_storage).Sort(keys);
  // ④ 取前 K 个写出去(该 block 的"局部 Top-K")。
  const std::uint64_t blocks = (vector_count + kBlockItems - 1) / kBlockItems;
#pragma unroll
  for (int item = 0; item < kItemsPerThread; ++item) {
    const int rank = threadIdx.x * kItemsPerThread + item;
    if (rank < K) {
      output[(static_cast<std::uint64_t>(q) * blocks + blockIdx.x) * K + rank] = keys[item];
    }
  }
}

// 归并内核:把上层的"每块 Top-K"再排一次,取前 K。
// 网格与 distance 内核相同;多个 block 的输出逐级减少,循环直到剩 K 个。
template <int K>
__global__ void merge_topk_kernel(const std::uint64_t* input,
                                  std::uint64_t input_count,
                                  std::uint32_t query_count,
                                  std::uint64_t* output) {
  using Sort = cub::BlockRadixSort<std::uint64_t, kBlockThreads, kItemsPerThread>;
  __shared__ typename Sort::TempStorage sort_storage;
  const std::uint32_t q = blockIdx.y;
  if (q >= query_count) return;
  std::uint64_t keys[kItemsPerThread];
#pragma unroll
  for (int item = 0; item < kItemsPerThread; ++item) {
    const std::uint64_t i = static_cast<std::uint64_t>(blockIdx.x) * kBlockItems +
                            item * kBlockThreads + threadIdx.x;
    keys[item] = i < input_count
        ? input[static_cast<std::uint64_t>(q) * input_count + i]
        : 0xffffffffffffffffull;
  }
  Sort(sort_storage).Sort(keys);
  const std::uint64_t blocks = (input_count + kBlockItems - 1) / kBlockItems;
#pragma unroll
  for (int item = 0; item < kItemsPerThread; ++item) {
    const int rank = threadIdx.x * kItemsPerThread + item;
    if (rank < K) {
      output[(static_cast<std::uint64_t>(q) * blocks + blockIdx.x) * K + rank] = keys[item];
    }
  }
}

// IVF 桶内搜索:一个 (query, probe) 负责一个桶,顺序扫桶内向量。
// 用 shared memory 的 best[] 保存当前 Top-K,与逐批新候选反复归并。
template <int K>
__global__ void ivf_list_topk_kernel(const float* vectors_soa,
                                     std::uint64_t vector_count,
                                     std::uint32_t dim,
                                     const std::uint64_t* offsets,
                                     const std::uint32_t* original_ids,
                                     const float* queries,
                                     const std::uint32_t* probes,
                                     std::uint32_t query_count,
                                     std::uint32_t nprobe,
                                     std::uint64_t* output) {
  using Sort = cub::BlockRadixSort<std::uint64_t, kBlockThreads, kItemsPerThread>;
  __shared__ typename Sort::TempStorage sort_storage;
  extern __shared__ unsigned char dynamic[];
  float* query_shared = reinterpret_cast<float*>(dynamic);
  const std::size_t query_bytes = (static_cast<std::size_t>(dim) * sizeof(float) + 7) & ~std::size_t{7};
  std::uint64_t* best = reinterpret_cast<std::uint64_t*>(dynamic + query_bytes);

  const std::uint32_t probe_rank = blockIdx.x;
  const std::uint32_t q = blockIdx.y;
  if (q >= query_count || probe_rank >= nprobe) return;
  // 查询向量进共享内存;best 初始化为最大键(表示"空")。
  for (std::uint32_t d = threadIdx.x; d < dim; d += blockDim.x) {
    query_shared[d] = queries[static_cast<std::uint64_t>(q) * dim + d];
  }
  for (int rank = threadIdx.x; rank < K; rank += blockDim.x) {
    best[rank] = 0xffffffffffffffffull;
  }
  __syncthreads();

  // 定位本 block 负责的桶 [begin, end)。
  const std::uint32_t list = probes[static_cast<std::uint64_t>(q) * nprobe + probe_rank];
  const std::uint64_t begin = offsets[list];
  const std::uint64_t end = offsets[list + 1];
  // 每轮只放 K 个已有 best + (1024-K) 个新候选,与 sort 能力对齐。
  constexpr int kNewItems = kBlockItems - K;
  for (std::uint64_t cursor = begin; cursor < end; cursor += kNewItems) {
    std::uint64_t keys[kItemsPerThread];
#pragma unroll
    for (int item = 0; item < kItemsPerThread; ++item) {
      const int rank = threadIdx.x * kItemsPerThread + item;
      if (rank < K) {
        keys[item] = best[rank];  // 前 K 个放已有的局部最佳
      } else {
        // 其余放新候选:算距离并打包(注意用 original_ids 还原原始 ID)。
        const std::uint64_t position = cursor + static_cast<std::uint64_t>(rank - K);
        if (position < end) {
          float distance = 0.0f;
          for (std::uint32_t d = 0; d < dim; ++d) {
            const float delta = query_shared[d] - vectors_soa[static_cast<std::uint64_t>(d) * vector_count + position];
            distance = fmaf(delta, delta, distance);
          }
          keys[item] = distance_key(distance, original_ids[position]);
        } else {
          keys[item] = 0xffffffffffffffffull;
        }
      }
    }
    // 混合排序,前 K 个成为新的局部最佳。
    Sort(sort_storage).Sort(keys);
#pragma unroll
    for (int item = 0; item < kItemsPerThread; ++item) {
      const int rank = threadIdx.x * kItemsPerThread + item;
      if (rank < K) best[rank] = keys[item];
    }
    __syncthreads();
  }
  // 桶扫完,输出该 (query, probe) 的局部 Top-K。
  for (int rank = threadIdx.x; rank < K; rank += blockDim.x) {
    output[(static_cast<std::uint64_t>(q) * nprobe + probe_rank) * K + rank] = best[rank];
  }
}

// 下面三个函数分别 launch 上面三个内核(带 K 的模板实例化)。
template <int K>
void launch_distance(const float* vectors, std::uint64_t count, std::uint32_t dim,
                     const float* queries, std::uint32_t query_count,
                     std::uint64_t* output, cudaStream_t stream) {
  const auto blocks = static_cast<unsigned int>((count + kBlockItems - 1) / kBlockItems);
  // 第三个参数是动态共享内存大小 = dim*sizeof(float),存查询向量。
  distance_partial_topk_kernel<K><<<dim3(blocks, query_count), kBlockThreads,
                                      dim * sizeof(float), stream>>>(
      vectors, count, dim, queries, query_count, output);
}

template <int K>
void launch_merge(const std::uint64_t* input, std::uint64_t count,
                  std::uint32_t query_count, std::uint64_t* output,
                  cudaStream_t stream) {
  const auto blocks = static_cast<unsigned int>((count + kBlockItems - 1) / kBlockItems);
  merge_topk_kernel<K><<<dim3(blocks, query_count), kBlockThreads, 0, stream>>>(
      input, count, query_count, output);
}

template <int K>
void launch_ivf(const float* vectors, std::uint64_t count, std::uint32_t dim,
                const std::uint64_t* offsets, const std::uint32_t* ids,
                const float* queries, const std::uint32_t* probes,
                std::uint32_t query_count, std::uint32_t nprobe,
                std::uint64_t* output, cudaStream_t stream) {
  // 动态共享内存 = 查询向量 + best 数组。
  const std::size_t query_bytes = (static_cast<std::size_t>(dim) * sizeof(float) + 7) & ~std::size_t{7};
  const std::size_t shared = query_bytes + K * sizeof(std::uint64_t);
  ivf_list_topk_kernel<K><<<dim3(nprobe, query_count), kBlockThreads, shared, stream>>>(
      vectors, count, dim, offsets, ids, queries, probes, query_count, nprobe, output);
}

// 把运行期的 K 值 dispatch 到编译期的模板实例。
// CUDA kernel 用模板参数 K 才能把 sort 规模做成编译期常量。
#define VS_DISPATCH_K(K_VALUE, CALL) \
  switch (K_VALUE) {                  \
    case 1: CALL(1); break;           \
    case 4: CALL(4); break;           \
    case 8: CALL(8); break;           \
    case 10: CALL(10); break;         \
    case 16: CALL(16); break;         \
    case 32: CALL(32); break;         \
    case 50: CALL(50); break;         \
    case 64: CALL(64); break;         \
    case 100: CALL(100); break;       \
    default: throw std::runtime_error("supported K values: 1,4,8,10,16,32,50,64,100"); \
  }

void dispatch_distance(std::uint32_t k, const float* vectors, std::uint64_t count,
                       std::uint32_t dim, const float* queries,
                       std::uint32_t query_count, std::uint64_t* output,
                       cudaStream_t stream) {
#define CALL(K) launch_distance<K>(vectors, count, dim, queries, query_count, output, stream)
  VS_DISPATCH_K(k, CALL);
#undef CALL
}

void dispatch_merge(std::uint32_t k, const std::uint64_t* input,
                    std::uint64_t count, std::uint32_t query_count,
                    std::uint64_t* output, cudaStream_t stream) {
#define CALL(K) launch_merge<K>(input, count, query_count, output, stream)
  VS_DISPATCH_K(k, CALL);
#undef CALL
}

void dispatch_ivf(std::uint32_t k, const float* vectors, std::uint64_t count,
                  std::uint32_t dim, const std::uint64_t* offsets,
                  const std::uint32_t* ids, const float* queries,
                  const std::uint32_t* probes, std::uint32_t query_count,
                  std::uint32_t nprobe, std::uint64_t* output,
                  cudaStream_t stream) {
#define CALL(K) launch_ivf<K>(vectors, count, dim, offsets, ids, queries, probes, query_count, nprobe, output, stream)
  VS_DISPATCH_K(k, CALL);
#undef CALL
}

// 求一组延迟样本的百分位。
double percentile(std::vector<double> values, double fraction) {
  if (values.empty()) return 0.0;
  std::sort(values.begin(), values.end());
  const auto rank = static_cast<std::size_t>(std::ceil(fraction * values.size()));
  return values[std::min(values.size() - 1, rank == 0 ? 0 : rank - 1)];
}

// 把 GPU 返回的打包键拆回 (ID, 距离),写进结果。
void decode_keys(const std::vector<std::uint64_t>& keys, std::uint64_t query_offset,
                 std::uint32_t k, SearchResult* result) {
  for (std::size_t i = 0; i < keys.size(); ++i) {
    const std::uint64_t key = keys[i];
    const std::size_t destination = static_cast<std::size_t>(query_offset) * k + i;
    result->ids[destination] = static_cast<std::uint32_t>(key);      // 低 32 位 = ID
    const std::uint32_t score_bits = static_cast<std::uint32_t>(key >> 32);  // 高 32 位 = 距离位模式
    std::memcpy(&result->scores[destination], &score_bits, sizeof(score_bits));
  }
}

// 数据库在显存中的字节数(count*dim 个 float)。
std::size_t index_bytes(std::uint64_t count, std::uint32_t dim) {
  return static_cast<std::size_t>(count) * dim * sizeof(float);
}

}  // namespace

// 探测是否有可用 CUDA 设备。
bool cuda_available(std::string* reason) {
  int count = 0;
  const cudaError_t status = cudaGetDeviceCount(&count);
  if (status != cudaSuccess || count == 0) {
    if (reason) *reason = status == cudaSuccess ? "no CUDA device" : cudaGetErrorString(status);
    cudaGetLastError();
    return false;
  }
  return true;
}

// FlatGpuIndex 构造:把 row-major 数据拷进显存,并转置为维度主序 [dim, count]。
FlatGpuIndex::FlatGpuIndex(const VectorDataset& database)
    : count_(database.count), dim_(database.dim) {
  if (count_ == 0 || dim_ == 0 || database.values.size() != count_ * dim_) {
    throw std::runtime_error("invalid database");
  }
  if (count_ > std::numeric_limits<std::uint32_t>::max()) {
    throw std::runtime_error("GPU index supports at most UINT32_MAX vectors");
  }
  float* row_major = nullptr;
  try {
    row_major = device_alloc<float>(static_cast<std::size_t>(count_) * dim_);
    device_vectors_soa_ = device_alloc<float>(static_cast<std::size_t>(count_) * dim_);
    cuda_check(cudaMemcpy(row_major, database.values.data(), index_bytes(count_, dim_),
                          cudaMemcpyHostToDevice), "copy database to GPU");
    // 转置 grid: (ceil(count/256), dim), 线程 y 轴是维度。
    const dim3 grid(static_cast<unsigned int>((count_ + 255) / 256), dim_);
    transpose_to_soa_kernel<<<grid, 256>>>(row_major, device_vectors_soa_, count_, dim_);
    cuda_check(cudaGetLastError(), "transpose database launch");
    cuda_check(cudaDeviceSynchronize(), "transpose database");
    device_free(row_major);
  } catch (...) {
    device_free(row_major);
    device_free(device_vectors_soa_);
    device_vectors_soa_ = nullptr;
    throw;
  }
}

FlatGpuIndex::~FlatGpuIndex() { device_free(device_vectors_soa_); }

// 精确检索主流程:按 batch 分块拷贝查询 → 分层 Top-K(首级+递归归并)→ 拷回结果。
SearchResult FlatGpuIndex::search(const VectorDataset& queries, std::uint32_t top_k,
                                  std::uint32_t batch_size,
                                  SearchMetrics* metrics) const {
  if (queries.count == 0 || queries.dim != dim_) throw std::runtime_error("invalid queries or dimension mismatch");
  if (top_k == 0 || top_k > count_ || batch_size == 0) throw std::runtime_error("invalid search parameters");
  if (queries.values.size() != queries.count * dim_) throw std::runtime_error("invalid query payload");
  const auto wall_begin = std::chrono::steady_clock::now();

  // batch 上限 65535,防止 block 网格超限。
  const std::uint32_t capacity = static_cast<std::uint32_t>(
      std::min<std::uint64_t>({batch_size, queries.count, 65535}));
  const std::uint64_t blocks = (count_ + kBlockItems - 1) / kBlockItems;
  // 第一级输出:每个 block 的局部 Top-K → blocks*K。
  const std::uint64_t first_count = blocks * top_k;
  // 第二级(以及后续各级)输出:仍是 blocks*K 但 blocks 变小。
  const std::uint64_t second_count = first_count > top_k
      ? ((first_count + kBlockItems - 1) / kBlockItems) * top_k : top_k;
  float* device_queries = nullptr;
  std::uint64_t* keys_a = nullptr;
  std::uint64_t* keys_b = nullptr;
  cudaEvent_t event_begin = nullptr;
  cudaEvent_t event_end = nullptr;
  SearchResult result;
  result.num_queries = queries.count;
  result.top_k = top_k;
  result.metric = Metric::kL2;
  result.ids.resize(static_cast<std::size_t>(queries.count) * top_k);
  result.scores.resize(static_cast<std::size_t>(queries.count) * top_k);
  std::vector<double> latencies;
  double gpu_total = 0.0;

  try {
    device_queries = device_alloc<float>(static_cast<std::size_t>(capacity) * dim_);
    keys_a = device_alloc<std::uint64_t>(static_cast<std::size_t>(capacity) * first_count);
    keys_b = device_alloc<std::uint64_t>(static_cast<std::size_t>(capacity) * second_count);
    cuda_check(cudaEventCreate(&event_begin), "create CUDA event");
    cuda_check(cudaEventCreate(&event_end), "create CUDA event");
    // 外层:按 capacity 分批处理查询,避免一次性占满显存。
    for (std::uint64_t offset = 0; offset < queries.count; offset += capacity) {
      const auto current = static_cast<std::uint32_t>(std::min<std::uint64_t>(capacity, queries.count - offset));
      const auto batch_begin = std::chrono::steady_clock::now();
      cuda_check(cudaEventRecord(event_begin), "record batch begin");
      // 查询拷到 GPU。
      cuda_check(cudaMemcpyAsync(device_queries,
                                 queries.values.data() + static_cast<std::size_t>(offset) * dim_,
                                 static_cast<std::size_t>(current) * dim_ * sizeof(float),
                                 cudaMemcpyHostToDevice), "copy queries to GPU");
      // 第一级:每个候选块出局部 Top-K。
      dispatch_distance(top_k, device_vectors_soa_, count_, dim_, device_queries,
                        current, keys_a, nullptr);
      cuda_check(cudaGetLastError(), "exact distance launch");
      // 归并循环:输入 count 逐级缩小,直到只剩 top_k。
      std::uint64_t count = first_count;
      std::uint64_t* input = keys_a;
      std::uint64_t* output = keys_b;
      while (count > top_k) {
        dispatch_merge(top_k, input, count, current, output, nullptr);
        cuda_check(cudaGetLastError(), "top-k merge launch");
        count = ((count + kBlockItems - 1) / kBlockItems) * top_k;
        std::swap(input, output);
      }
      // 最终 Top-K 拷回 host 并拆包。
      std::vector<std::uint64_t> host_keys(static_cast<std::size_t>(current) * top_k);
      cuda_check(cudaMemcpyAsync(host_keys.data(), input, host_keys.size() * sizeof(std::uint64_t),
                                 cudaMemcpyDeviceToHost), "copy exact results");
      cuda_check(cudaEventRecord(event_end), "record batch end");
      cuda_check(cudaEventSynchronize(event_end), "wait for exact search");
      float batch_gpu_ms = 0.0f;
      cuda_check(cudaEventElapsedTime(&batch_gpu_ms, event_begin, event_end), "measure exact search");
      gpu_total += batch_gpu_ms;
      decode_keys(host_keys, offset, top_k, &result);
      const auto batch_end = std::chrono::steady_clock::now();
      latencies.push_back(std::chrono::duration<double, std::milli>(batch_end - batch_begin).count());
    }
  } catch (...) {
    if (event_begin) cudaEventDestroy(event_begin);
    if (event_end) cudaEventDestroy(event_end);
    device_free(device_queries);
    device_free(keys_a);
    device_free(keys_b);
    throw;
  }
  cudaEventDestroy(event_begin);
  cudaEventDestroy(event_end);
  device_free(device_queries);
  device_free(keys_a);
  device_free(keys_b);

  if (metrics) {
    metrics->total_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - wall_begin).count();
    metrics->gpu_ms = gpu_total;
    metrics->qps = queries.count * 1000.0 / metrics->total_ms;
    metrics->p50_ms = percentile(latencies, 0.50);
    metrics->p99_ms = percentile(latencies, 0.99);
    // 显存估算: 向量库 + 查询缓冲 + 两级 keys 缓冲。
    metrics->peak_device_bytes = index_bytes(count_, dim_) +
        static_cast<std::size_t>(capacity) *
            (dim_ * sizeof(float) + (first_count + second_count) * sizeof(std::uint64_t));
  }
  return result;
}

// IvfGpuIndex 构造:把索引(向量、偏移、原始 ID、质心)整块拷进显存。
// 质心交给 FlatGpuIndex 管理,复用它做"找最近质心"的 Top-K 搜索。
IvfGpuIndex::IvfGpuIndex(const IvfIndexData& index)
    : count_(index.count), dim_(index.dim), nlist_(index.nlist), host_offsets_(index.offsets) {
  if (count_ == 0 || dim_ == 0 || nlist_ == 0 || nlist_ > count_ ||
      index.vectors_soa.size() != count_ * dim_ || index.original_ids.size() != count_ ||
      index.offsets.size() != static_cast<std::size_t>(nlist_) + 1 ||
      index.centroids.size() != static_cast<std::size_t>(nlist_) * dim_) {
    throw std::runtime_error("invalid IVF index");
  }
  VectorDataset centroids;
  centroids.count = nlist_;
  centroids.dim = dim_;
  centroids.values = index.centroids;
  try {
    device_vectors_soa_ = device_alloc<float>(static_cast<std::size_t>(count_) * dim_);
    device_offsets_ = device_alloc<std::uint64_t>(static_cast<std::size_t>(nlist_) + 1);
    device_ids_ = device_alloc<std::uint32_t>(count_);
    cuda_check(cudaMemcpy(device_vectors_soa_, index.vectors_soa.data(), index_bytes(count_, dim_),
                          cudaMemcpyHostToDevice), "copy IVF vectors");
    cuda_check(cudaMemcpy(device_offsets_, index.offsets.data(),
                          (static_cast<std::size_t>(nlist_) + 1) * sizeof(std::uint64_t),
                          cudaMemcpyHostToDevice), "copy IVF offsets");
    cuda_check(cudaMemcpy(device_ids_, index.original_ids.data(), count_ * sizeof(std::uint32_t),
                          cudaMemcpyHostToDevice), "copy IVF ids");
    centroid_index_ = new FlatGpuIndex(centroids);
  } catch (...) {
    delete centroid_index_;
    centroid_index_ = nullptr;
    device_free(device_vectors_soa_);
    device_free(device_offsets_);
    device_free(device_ids_);
    device_vectors_soa_ = nullptr;
    device_offsets_ = nullptr;
    device_ids_ = nullptr;
    throw;
  }
}

IvfGpuIndex::~IvfGpuIndex() {
  delete centroid_index_;
  device_free(device_vectors_soa_);
  device_free(device_offsets_);
  device_free(device_ids_);
}

// IVF 查询主流程:
//  1. 用 FlatGpuIndex 对质心做 Top-nprobe,得到每个查询最近的 nprobe 个桶;
//  2. 每个 (query, 桶) 一个 block 扫桶内向量,输出局部 Top-K;
//  3. 跨桶递归归并到全局 Top-K。
SearchResult IvfGpuIndex::search(const VectorDataset& queries, std::uint32_t top_k,
                                 std::uint32_t nprobe, std::uint32_t batch_size,
                                 SearchMetrics* metrics) const {
  if (queries.count == 0 || queries.dim != dim_ || top_k == 0 || top_k > count_ || batch_size == 0 ||
      nprobe == 0 || nprobe > nlist_) {
    throw std::runtime_error("invalid IVF search parameters");
  }
  const auto wall_begin = std::chrono::steady_clock::now();
  // ① 第一级:找最近 nprobe 个质心(复用 FlatGpuIndex 的精确 Top-K 引擎)。
  SearchMetrics probe_metrics;
  const SearchResult nearest_lists = centroid_index_->search(queries, nprobe, batch_size, &probe_metrics);
  const std::uint32_t capacity = static_cast<std::uint32_t>(
      std::min<std::uint64_t>({batch_size, queries.count, 65535}));
  float* device_queries = nullptr;
  std::uint32_t* device_probes = nullptr;
  std::uint64_t* keys_a = nullptr;
  std::uint64_t* keys_b = nullptr;
  cudaEvent_t event_begin = nullptr;
  cudaEvent_t event_end = nullptr;
  SearchResult result;
  result.num_queries = queries.count;
  result.top_k = top_k;
  result.metric = Metric::kL2;
  result.ids.resize(static_cast<std::size_t>(queries.count) * top_k);
  result.scores.resize(static_cast<std::size_t>(queries.count) * top_k);
  std::vector<double> latencies;
  double gpu_total = probe_metrics.gpu_ms;
  // 第二级输出:每个 (query, 桶) 的局部 Top-K → nprobe*K。
  const std::uint64_t first_count = static_cast<std::uint64_t>(nprobe) * top_k;
  const std::uint64_t second_count = first_count > top_k
      ? ((first_count + kBlockItems - 1) / kBlockItems) * top_k : top_k;
  try {
    device_queries = device_alloc<float>(static_cast<std::size_t>(capacity) * dim_);
    device_probes = device_alloc<std::uint32_t>(static_cast<std::size_t>(capacity) * nprobe);
    keys_a = device_alloc<std::uint64_t>(static_cast<std::size_t>(capacity) * first_count);
    keys_b = device_alloc<std::uint64_t>(static_cast<std::size_t>(capacity) * second_count);
    cuda_check(cudaEventCreate(&event_begin), "create IVF event");
    cuda_check(cudaEventCreate(&event_end), "create IVF event");
    for (std::uint64_t offset = 0; offset < queries.count; offset += capacity) {
      const auto current = static_cast<std::uint32_t>(std::min<std::uint64_t>(capacity, queries.count - offset));
      // 把每个查询的 nprobe 个桶编号(来自质心搜索)拷到 host 侧,顺便校验候选数。
      std::vector<std::uint32_t> host_probes(static_cast<std::size_t>(current) * nprobe);
      for (std::uint32_t q = 0; q < current; ++q) {
        std::uint64_t candidate_count = 0;
        for (std::uint32_t p = 0; p < nprobe; ++p) {
          const auto list = nearest_lists.ids[(static_cast<std::size_t>(offset) + q) * nprobe + p];
          if (list >= nlist_) throw std::runtime_error("centroid search returned invalid list");
          host_probes[static_cast<std::size_t>(q) * nprobe + p] = static_cast<std::uint32_t>(list);
          candidate_count += host_offsets_[list + 1] - host_offsets_[list];
        }
        // 桶太小可能凑不满 top_k 个候选,此时必须增大 nprobe。
        if (candidate_count < top_k) {
          throw std::runtime_error("selected IVF lists contain fewer than top_k candidates; increase nprobe");
        }
      }
      const auto batch_begin = std::chrono::steady_clock::now();
      cuda_check(cudaEventRecord(event_begin), "record IVF begin");
      cuda_check(cudaMemcpyAsync(device_queries,
                                 queries.values.data() + static_cast<std::size_t>(offset) * dim_,
                                 static_cast<std::size_t>(current) * dim_ * sizeof(float),
                                 cudaMemcpyHostToDevice), "copy IVF queries");
      cuda_check(cudaMemcpyAsync(device_probes, host_probes.data(),
                                 host_probes.size() * sizeof(std::uint32_t),
                                 cudaMemcpyHostToDevice), "copy IVF probes");
      // 第二级:每个 (query, 桶) 的 block 扫桶,出局部 Top-K。
      dispatch_ivf(top_k, device_vectors_soa_, count_, dim_, device_offsets_, device_ids_,
                   device_queries, device_probes, current, nprobe, keys_a, nullptr);
      cuda_check(cudaGetLastError(), "IVF list search launch");
      // 跨桶归并(与 Exact 的跨候选块归并共用 merge kernel)。
      std::uint64_t count = first_count;
      std::uint64_t* input = keys_a;
      std::uint64_t* output = keys_b;
      while (count > top_k) {
        dispatch_merge(top_k, input, count, current, output, nullptr);
        cuda_check(cudaGetLastError(), "IVF merge launch");
        count = ((count + kBlockItems - 1) / kBlockItems) * top_k;
        std::swap(input, output);
      }
      std::vector<std::uint64_t> host_keys(static_cast<std::size_t>(current) * top_k);
      cuda_check(cudaMemcpyAsync(host_keys.data(), input, host_keys.size() * sizeof(std::uint64_t),
                                 cudaMemcpyDeviceToHost), "copy IVF results");
      cuda_check(cudaEventRecord(event_end), "record IVF end");
      cuda_check(cudaEventSynchronize(event_end), "wait for IVF search");
      float batch_gpu_ms = 0.0f;
      cuda_check(cudaEventElapsedTime(&batch_gpu_ms, event_begin, event_end), "measure IVF search");
      gpu_total += batch_gpu_ms;
      decode_keys(host_keys, offset, top_k, &result);
      latencies.push_back(std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - batch_begin).count());
    }
  } catch (...) {
    if (event_begin) cudaEventDestroy(event_begin);
    if (event_end) cudaEventDestroy(event_end);
    device_free(device_queries);
    device_free(device_probes);
    device_free(keys_a);
    device_free(keys_b);
    throw;
  }
  cudaEventDestroy(event_begin);
  cudaEventDestroy(event_end);
  device_free(device_queries);
  device_free(device_probes);
  device_free(keys_a);
  device_free(keys_b);

  if (metrics) {
    metrics->total_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - wall_begin).count();
    metrics->gpu_ms = gpu_total;
    metrics->qps = queries.count * 1000.0 / metrics->total_ms;
    metrics->p50_ms = percentile(latencies, 0.50);
    metrics->p99_ms = percentile(latencies, 0.99);
    // 显存估算: 桶内向量 + 原始ID + 偏移 + 质心 + 工作缓冲。
    metrics->peak_device_bytes = index_bytes(count_, dim_) +
        count_ * sizeof(std::uint32_t) +
        (static_cast<std::size_t>(nlist_) + 1) * sizeof(std::uint64_t) +
        static_cast<std::size_t>(nlist_) * dim_ * sizeof(float) +
        static_cast<std::size_t>(capacity) *
            (dim_ * sizeof(float) + nprobe * sizeof(std::uint32_t) +
             (first_count + second_count) * sizeof(std::uint64_t));
  }
  return result;
}

}  // namespace vs
