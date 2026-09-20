#pragma once

#include "vsearch/common.hpp"

namespace vs {

// 一次检索的结果集合：nq×topK 行主序，每个 query 的 K 条按优到劣排列。
// metric/mode 仅用于结果文件的可读性标注。
struct SearchResult {
  i64 nq = 0;
  int topK = 0;
  Metric metric = Metric::None;
  std::string mode;
  std::vector<i32> ids;      // nq*topK
  FloatVec scores;           // nq*topK
};

void writeResultText(const std::string& path, const SearchResult& r);

// 性能日志的一行记录：覆盖题目要求的 build/search 时间、QPS、P50/P99、
// 显存占用，以及相对 CPU 精确基线的加速比。
struct PerfRecord {
  std::string mode;
  i64 nq = 0;
  i64 n = 0;
  int dim = 0;
  int topK = 0;
  double buildMs = 0.0;
  double searchMs = 0.0;
  double qps = 0.0;
  double p50Ms = 0.0;
  double p99Ms = 0.0;
  double meanBatchMs = 0.0;
  std::size_t gpuUsedBytes = 0;
  double cpuExactMs = 0.0;
  double speedupVsCpu = 0.0;
};

void writePerfLog(const std::string& path,
                  const std::vector<PerfRecord>& rows);

// Quality comparison against a reference (typically the GPU exact baseline).
// 质量日志：以 gold（通常是 GPU exact）为基准，输出 recall@K、平均距离误差
// 与逐 rank 不一致数；近似检索的召回/误差都以此衡量。
void writeQualityLog(const std::string& path, i64 nq, int topK,
                     Metric metric, const SearchResult& pred,
                     const SearchResult& gold, double avgDistanceError,
                     int mismatchedRankCount);

}  // namespace vs
