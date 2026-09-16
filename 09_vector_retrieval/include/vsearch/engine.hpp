#pragma once

#include <memory>
#include <string>
#include <vector>

#include "vsearch/common.hpp"

namespace vs {

// 索引构建统计：buildMs 为整体墙钟时间，其余为各阶段细分（训练中心、全库
// 分配、PQ 编码）与设备/序列化占用，用于性能日志与报告分析。
struct IndexBuildStats {
  double buildMs = 0.0;         // host wall time of the full build
  double gpuWorkMs = 0.0;       // kernel/GEMM wall time measured with events
  std::size_t deviceBytes = 0;  // device-side index bytes
  std::size_t fileBytes = 0;    // serialized index bytes
  double trainMs = 0.0;
  double assignMs = 0.0;
  double pqEncodeMs = 0.0;
};

// 单次批量检索统计：wallMs 是主机侧总耗时，gpuMs 是各 batch CUDA event 之和，
// batchMs 保存每个内部 batch 的耗时以计算 P50/P99；qps 基于 wallMs 计算。
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
//
// GPU 检索引擎门面类：构造时上传向量库并常驻显存；近似索引可现场构建，
// 也可从文件加载。对外只暴露 build/搜索/持久化接口，内部实现（kernel、
// 缓冲布局）全部封装在 Impl 中。
class GpuEngine {
 public:
  GpuEngine(const Dataset& ds, const SearchConfig& cfg);
  ~GpuEngine();
  GpuEngine(const GpuEngine&) = delete;
  GpuEngine& operator=(const GpuEngine&) = delete;
  GpuEngine(GpuEngine&&) noexcept;
  GpuEngine& operator=(GpuEngine&&) noexcept;

  // mode must be IvfFlat or IvfPq. Requires nlist*dim*4 bytes etc. to fit.
  // 构建近似索引（IVF-Flat / IVF-PQ），返回各阶段耗时与显存占用。
  IndexBuildStats buildIndex(SearchMode mode);
  // 索引是否已构建或成功加载。
  bool hasIndex() const;
  // 索引落盘 / 从文件加载（跨平台的小端二进制格式，见 engine.cu）。
  void saveIndex(const std::string& path) const;
  void loadIndex(const std::string& path);

  // Run one mode on the whole query set (internal batching uses cfg.batch_size,
  // or batchOverride when > 0). IDs/scores are nq*topK row-major and best-first.
  // 对整批 query 运行指定模式：内部按 batch_size 分批，输出 nq×topK 行主序、
  // 按相似度从优到劣排列的 id 与分数。
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
