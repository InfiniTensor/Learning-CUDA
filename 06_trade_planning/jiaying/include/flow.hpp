#pragma once

#include <chrono>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace trade {
using Clock = std::chrono::steady_clock;
inline double elapsed_ms(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start)
      .count();
}
// Each input edge owns a distinct residual pair, including antiparallel edges.
struct Graph {
  int n = 0, m = 0;
  std::vector<int> row, col, cap;
};
struct ResidualGraph {
  int n = 0;
  std::vector<int> row, col, rev, cap;
  explicit ResidualGraph(const Graph &g);
};
Graph read_graph(const std::string &path);
std::vector<std::pair<int, int>> read_queries(const std::string &path, int n);
struct QueryResult {
  int64_t flow = 0;
  double ms = 0;
  uint64_t rounds = 0;
  int global_relabels = 0;
};
class CpuSolver {
public:
  explicit CpuSolver(const ResidualGraph &g);
  QueryResult solve(int source, int target, bool push_relabel = false);

private:
  const ResidualGraph &g_;
  std::vector<int> residual_, level_, current_, queue_, path_;
  std::vector<int64_t> bottleneck_, excess_;
  std::vector<unsigned char> queued_;
  int64_t dinic(int source, int target);
  int64_t preflow(int source, int target);
};
struct GpuOptions {
  int batch_rounds = 8;
  int global_interval = 64;
  int max_rounds = 1000000;
  bool warp = false;
  bool graphs = true;
};
class GpuSolver {
public:
  GpuSolver(const ResidualGraph &g, GpuOptions options);
  ~GpuSolver();
  GpuSolver(const GpuSolver &) = delete;
  GpuSolver &operator=(const GpuSolver &) = delete;
  QueryResult solve(int source, int target);
  void verify(int source, int target, int64_t expected) const;
  std::string device_name() const;
  size_t device_bytes() const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
} // namespace trade
