#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#include <stdexcept>
#include <string>
#include <vector>
#if defined(_MSC_VER)
#include <malloc.h>
#endif

namespace vs {

using i64 = std::int64_t;
using i32 = std::int32_t;

enum class DataType {
  Fp32,
  Fp16,
};

enum class Metric {
  None,          // only legal for query files
  L2,
  InnerProduct,  // larger is better
  Cosine,        // engine reports cosine distance = 1 - cosine(a,b)
};

enum class SearchMode {
  Exact,
  IvfFlat,
  IvfPq,
};

enum class IndexKind : i32 {
  IvfFlat = 0,
  IvfPq = 1,
};

inline bool metricIsSmallerBetter(Metric m) {
  return m == Metric::L2 || m == Metric::Cosine;
}

inline const char* metricName(Metric m) {
  switch (m) {
    case Metric::None: return "none";
    case Metric::L2: return "l2";
    case Metric::InnerProduct: return "inner_product";
    case Metric::Cosine: return "cosine";
  }
  return "none";
}

inline const char* modeName(SearchMode m) {
  switch (m) {
    case SearchMode::Exact: return "exact";
    case SearchMode::IvfFlat: return "ivf_flat";
    case SearchMode::IvfPq: return "ivf_pq";
  }
  return "exact";
}

[[noreturn]] inline void throwRuntime(const std::string& msg) {
  throw std::runtime_error(msg);
}

inline i64 nowNanos() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

// 主机侧单调计时器（steady_clock），用于各阶段墙钟耗时统计。
class Timer {
 public:
  Timer() : start_(nowNanos()) {}
  void reset() { start_ = nowNanos(); }
  double ms() const { return static_cast<double>(nowNanos() - start_) / 1e6; }
  double sec() const { return static_cast<double>(nowNanos() - start_) / 1e9; }

 private:
  i64 start_;
};

// A cache-friendly aligned float allocator. std::vector<float, AlignedAllocator<float>>
// keeps CUDA memcpy fast and enables float4 reads when alignment is available.
template <typename T>
class AlignedAllocator {
 public:
  using value_type = T;
  AlignedAllocator() noexcept {}
  template <typename U>
  AlignedAllocator(const AlignedAllocator<U>&) noexcept {}

  T* allocate(std::size_t n) {
    constexpr std::size_t kAlign = 64;
    if (n == 0) return nullptr;
    void* p = nullptr;
#if defined(_MSC_VER)
    p = _aligned_malloc(n * sizeof(T), kAlign);
    if (!p) throw std::bad_alloc();
#else
    if (posix_memalign(&p, kAlign, n * sizeof(T)) != 0) throw std::bad_alloc();
#endif
    return static_cast<T*>(p);
  }

  void deallocate(T* p, std::size_t) noexcept {
    if (!p) return;
#if defined(_MSC_VER)
    _aligned_free(p);
#else
    free(p);
#endif
  }

  template <typename U>
  bool operator==(const AlignedAllocator<U>&) const noexcept {
    return true;
  }
  template <typename U>
  bool operator!=(const AlignedAllocator<U>&) const noexcept {
    return false;
  }
};

using FloatVec = std::vector<float, AlignedAllocator<float>>;

// Vector collection shared by database and query files.
//
// 向量集合：数据库与查询文件共用同一结构。data 始终以 fp32 行主序保存
// （fp16 输入在读取时转换），便于所有 kernel 使用统一的输入类型。
struct Dataset {
  i64 n = 0;          // number of rows
  i32 dim = 0;        // vector dimension
  DataType dtype = DataType::Fp32;
  Metric metric = Metric::None;
  FloatVec data;      // row-major, always converted to fp32

  std::size_t bytes() const {
    return static_cast<std::size_t>(n) * dim * sizeof(float);
  }
};

// Parameters parsed from the text retrieval-parameter file.
//
// 检索参数：来自文本参数文件，也可被命令行覆盖。前几项对应题目要求的
// top_k / search_mode / batch_size / nlist / nprobe / pq_m；其余为实验扩展项
// （k-means 采样与迭代、PQ 精排宽度、日志路径、CPU 参考 query 数等）。
struct SearchConfig {
  int top_k = 10;
  std::string search_mode = "exact";   // exact / ivf_flat / ivf_pq
  int batch_size = 128;
  int nlist = 4096;
  int nprobe = 16;
  int pq_m = 16;
  int pq_ks = 256;
  int pq_rerank = 256;          // top-N candidates rescored exactly after ADC

  // ---- 构建 k-means 的扩展参数 ----
  int kmeans_iters = 12;
  i64 kmeans_sample = 131072;          // mini-batch used while refining centers
  i64 kmeans_chunk_rows = 16384;       // rows per GEMM chunk during assignment
  int kmeans_seed = 2026;
  double kmeans_assign_frac = 0.0;     // optional early stop
  bool cosine_normalize_centers = true;

  // ---- CLI / 驱动使用的路径与开关 ----
  int nthreads = 0;                    // 0 => hardware concurrency
  std::string index_path;              // optional when used by driver
  std::string result_path;
  std::string perf_log_path;
  std::string quality_log_path;
  bool force_rebuild = false;
  i64 ref_query_limit = 200;           // queries used by the CPU reference
  i64 exact_memory_mb = 1536;          // approximate upper bound per exact chunk
};

}  // namespace vs
