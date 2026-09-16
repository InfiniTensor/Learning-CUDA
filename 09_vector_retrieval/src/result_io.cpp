#include "vsearch/result_io.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>

namespace vs {
namespace {

double percentile(std::vector<double> v, double p) {
  if (v.empty()) return 0.0;
  std::sort(v.begin(), v.end());
  const double idx = p * static_cast<double>(v.size() - 1);
  const std::size_t lo = static_cast<std::size_t>(idx);
  const std::size_t hi = std::min(lo + 1, v.size() - 1);
  const double frac = idx - static_cast<double>(lo);
  return v[lo] + (v[hi] - v[lo]) * frac;
}

std::string gpuBytes(std::size_t b) {
  char buf[64];
  std::snprintf(buf, sizeof(buf), "%.2f", static_cast<double>(b) / (1024.0 * 1024.0 * 1024.0));
  return std::string(buf) + " GB";
}

}  // namespace

void writeResultText(const std::string& path, const SearchResult& r) {
  FILE* f = std::fopen(path.c_str(), "w");
  if (!f) throwRuntime("cannot open result file " + path);
  std::fprintf(f, "# vector search result\n");
  std::fprintf(f, "# mode=%s metric=%s num_queries=%lld top_k=%d\n",
               r.mode.c_str(), metricName(r.metric),
               static_cast<long long>(r.nq), r.topK);
  for (i64 qi = 0; qi < r.nq; ++qi) {
    std::fprintf(f, "Q %lld:", static_cast<long long>(qi));
    for (int k = 0; k < r.topK; ++k) {
      const std::size_t idx = static_cast<std::size_t>(qi) * r.topK + k;
      std::fprintf(f, " %d %.8g", r.ids[idx], r.scores[idx]);
    }
    std::fprintf(f, "\n");
  }
  std::fclose(f);
}

void writePerfLog(const std::string& path, const std::vector<PerfRecord>& rows) {
  FILE* f = std::fopen(path.c_str(), "w");
  if (!f) throwRuntime("cannot open perf log " + path);
  std::fprintf(f, "# performance log (vsearch)\n");
  std::fprintf(f,
               "%-10s %6s %9s %6s %6s %12s %12s %12s %10s %10s %10s %14s %12s %14s\n",
               "mode", "nq", "n", "dim", "topk", "build_ms", "search_ms",
               "qps", "p50_ms", "p99_ms", "mean_batch_ms", "gpu_used",
               "cpu_ms", "speedup");
  for (const auto& r : rows) {
    std::fprintf(f,
                 "%-10s %6lld %9lld %6d %6d %12.3f %12.3f %12.1f %10.4f %10.4f "
                 "%10.4f %14s %12.3f %14.1f\n",
                 r.mode.c_str(), static_cast<long long>(r.nq),
                 static_cast<long long>(r.n), r.dim, r.topK, r.buildMs,
                 r.searchMs, r.qps, r.p50Ms, r.p99Ms, r.meanBatchMs,
                 gpuBytes(r.gpuUsedBytes).c_str(), r.cpuExactMs,
                 r.speedupVsCpu);
  }
  std::fclose(f);
}

void writeQualityLog(const std::string& path, i64 nq, int topK, Metric metric,
                     const SearchResult& pred, const SearchResult& gold,
                     double avgDistanceError, int mismatchedRankCount) {
  FILE* f = std::fopen(path.c_str(), "w");
  if (!f) throwRuntime("cannot open quality log " + path);
  const std::size_t per = static_cast<std::size_t>(topK);
  i64 matched = 0;
  i64 totalPairs = static_cast<i64>(nq) * topK;
  double sumAbsRel = 0.0;
  for (i64 qi = 0; qi < nq; ++qi) {
    const i32* pa = pred.ids.data() + qi * per;
    const i32* ga = gold.ids.data() + qi * per;
    const float* ps = pred.scores.data() + qi * per;
    const float* gs = gold.scores.data() + qi * per;
    for (int k = 0; k < topK; ++k) {
      if (pa[k] == ga[k]) {
        ++matched;
        const double base =
            std::max(std::fabs(static_cast<double>(gs[k])), 1e-9);
        sumAbsRel += std::fabs(static_cast<double>(ps[k] - gs[k])) / base;
      }
    }
  }
  const double recall = totalPairs ? static_cast<double>(matched) / totalPairs : 0.0;
  std::fprintf(f, "# quality log (vsearch)\n");
  std::fprintf(f, "num_queries=%lld top_k=%d metric=%s\n",
               static_cast<long long>(nq), topK, metricName(metric));
  std::fprintf(f, "mode=%s\n", pred.mode.c_str());
  std::fprintf(f, "recall_at_k=%.6f\n", recall);
  std::fprintf(f, "id_match_ratio=%.6f\n", recall);
  std::fprintf(f, "avg_relative_score_error=%.8g\n",
               matched ? sumAbsRel / matched : 0.0);
  std::fprintf(f, "avg_distance_error=%.8g\n", avgDistanceError);
  std::fprintf(f, "mismatched_rank_count=%d\n", mismatchedRankCount);
  std::fclose(f);
}

}  // namespace vs
