#pragma once

#include <memory>
#include <string>
#include <vector>

#include "vsearch/common.hpp"

namespace vs {

struct IndexBuildStats {
  double buildMs = 0.0;         // host wall time of the full build
  double gpuWorkMs = 0.0;       // kernel/GEMM wall time measured with events
  std::size_t deviceBytes = 0;  // device-side index bytes
  std::size_t fileBytes = 0;    // serialized index bytes
  double trainMs = 0.0;
  double assignMs = 0.0;
  double pqEncodeMs = 0.0;
};

struct SearchStats {
  double wallMs = 0.0;          // host wall time of searchAll
  double gpuMs = 0.0;           // summed GPU event time over batches
  double qps = 0.0;
  double p50BatchMs = 0.0;
  double p99BatchMs = 0.0;
  double meanBatchMs = 0.0;
  std::vector<double> batchMs;  // one entry per internal batch
  std::size_t deviceUsedBytes = 0;  // free-vs-total delta snapshot
};

// GPU engine implementing exact, IVF-Flat and IVF-PQ searches. The dataset is
// uploaded once and retained; approximate index state is either built or
// loaded from a file.
class GpuEngine {
 public:
  GpuEngine(const Dataset& ds, const SearchConfig& cfg);
  ~GpuEngine();
  GpuEngine(const GpuEngine&) = delete;
  GpuEngine& operator=(const GpuEngine&) = delete;
  GpuEngine(GpuEngine&&) noexcept;
  GpuEngine& operator=(GpuEngine&&) noexcept;

  // mode must be IvfFlat or IvfPq. Requires nlist*dim*4 bytes etc. to fit.
  IndexBuildStats buildIndex(SearchMode mode);
  bool hasIndex() const;
  void saveIndex(const std::string& path) const;
  void loadIndex(const std::string& path);

  // Run one mode on the whole query set (internal batching uses cfg.batch_size,
  // or batchOverride when > 0). IDs/scores are nq*topK row-major and best-first.
  void searchAll(SearchMode mode, const float* queries, i64 nq, int topK,
                 std::vector<i32>* ids, FloatVec* scores,
                 SearchStats* stats, int batchOverride = 0);

  static std::size_t deviceTotalBytes();
  static std::size_t deviceFreeBytes();

 public:
  // Exposed for the implementation translation unit; callers should treat the
  // type as opaque.
  struct Impl;

 private:
  std::unique_ptr<Impl> impl_;
};

}  // namespace vs
