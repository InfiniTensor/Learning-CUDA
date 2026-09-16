#pragma once

#include "vsearch/common.hpp"

namespace vs {

struct SearchResult {
  i64 nq = 0;
  int topK = 0;
  Metric metric = Metric::None;
  std::string mode;
  std::vector<i32> ids;      // nq*topK
  FloatVec scores;           // nq*topK
};

void writeResultText(const std::string& path, const SearchResult& r);

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
void writeQualityLog(const std::string& path, i64 nq, int topK,
                     Metric metric, const SearchResult& pred,
                     const SearchResult& gold, double avgDistanceError,
                     int mismatchedRankCount);

}  // namespace vs
