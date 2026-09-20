#include "flow.hpp"
#include <algorithm>
#include <deque>
#include <limits>

namespace trade {
CpuSolver::CpuSolver(const ResidualGraph &g)
    : g_(g), residual_(g.cap.size()), level_(g.n), current_(g.n), queue_(g.n),
      bottleneck_(g.n), excess_(g.n), queued_(g.n) {
  path_.reserve(g.n);
}

QueryResult CpuSolver::solve(int s, int t, bool push_relabel) {
  const auto start = Clock::now();
  std::copy(g_.cap.begin(), g_.cap.end(), residual_.begin());
  QueryResult result;
  result.flow = push_relabel ? preflow(s, t) : dinic(s, t);
  result.ms = elapsed_ms(start);
  return result;
}

int64_t CpuSolver::dinic(int s, int t) {
  int64_t flow = 0;
  for (;;) {
    std::fill(level_.begin(), level_.end(), -1);
    int head = 0, tail = 1;
    queue_[0] = s;
    level_[s] = 0;
    while (head < tail) {
      const int u = queue_[head++];
      for (int e = g_.row[u]; e < g_.row[u + 1]; ++e)
        if (residual_[e] > 0 && level_[g_.col[e]] < 0) {
          level_[g_.col[e]] = level_[u] + 1;
          queue_[tail++] = g_.col[e];
        }
    }
    if (level_[t] < 0)
      return flow;
    std::copy(g_.row.begin(), g_.row.end() - 1, current_.begin());
    // Explicit DFS stack avoids stack overflow on long trade corridors.
    path_.clear();
    int u = s;
    bottleneck_[s] = std::numeric_limits<int64_t>::max();
    for (;;) {
      if (u == t) {
        const int amount = int(bottleneck_[t]);
        flow += amount;
        for (int e : path_) {
          residual_[e] -= amount;
          residual_[g_.rev[e]] += amount;
        }
        path_.clear();
        u = s;
        continue;
      }
      int &e = current_[u];
      while (e < g_.row[u + 1] &&
             !(residual_[e] > 0 && level_[g_.col[e]] == level_[u] + 1))
        ++e;
      if (e == g_.row[u + 1]) {
        level_[u] = -1;
        if (path_.empty())
          break;
        u = g_.col[g_.rev[path_.back()]];
        path_.pop_back();
        ++current_[u];
      } else {
        const int v = g_.col[e];
        bottleneck_[v] = std::min<int64_t>(bottleneck_[u], residual_[e]);
        path_.push_back(e);
        u = v;
      }
    }
  }
}

int64_t CpuSolver::preflow(int s, int t) {
  std::fill(excess_.begin(), excess_.end(), 0);
  for (int e = g_.row[s]; e < g_.row[s + 1]; ++e) {
    if (g_.col[e] == s)
      continue;
    const int amount = residual_[e];
    residual_[e] = 0;
    residual_[g_.rev[e]] += amount;
    excess_[g_.col[e]] += amount;
  }
  std::deque<int> active;
  auto global_relabel = [&]() {
    std::fill(level_.begin(), level_.end(), g_.n);
    std::fill(queued_.begin(), queued_.end(), 0);
    std::copy(g_.row.begin(), g_.row.end() - 1, current_.begin());
    active.clear();
    int head = 0, tail = 1;
    queue_[0] = t;
    level_[t] = 0;
    while (head < tail) {
      const int u = queue_[head++];
      for (int e = g_.row[u]; e < g_.row[u + 1]; ++e) {
        const int v = g_.col[e];
        if (v != s && level_[v] == g_.n && residual_[g_.rev[e]] > 0) {
          level_[v] = level_[u] + 1;
          queue_[tail++] = v;
        }
      }
    }
    for (int u = 0; u < g_.n; ++u)
      if (u != s && u != t && excess_[u] > 0 && level_[u] < g_.n) {
        queued_[u] = 1;
        active.push_back(u);
      }
  };
  global_relabel();
  uint64_t work = 0;
  while (!active.empty()) {
    const int u = active.front();
    active.pop_front();
    queued_[u] = 0;
    while (excess_[u] > 0 && level_[u] < g_.n) {
      int &e = current_[u];
      if (e == g_.row[u + 1]) {
        int best = g_.n;
        for (int a = g_.row[u]; a < g_.row[u + 1]; ++a) {
          if (residual_[a] > 0)
            best = std::min(best, level_[g_.col[a]] + 1);
          ++work;
        }
        level_[u] = best;
        e = g_.row[u];
      } else {
        const int v = g_.col[e];
        if (residual_[e] > 0 && level_[u] == level_[v] + 1) {
          const int amount = int(std::min<int64_t>(excess_[u], residual_[e]));
          residual_[e] -= amount;
          residual_[g_.rev[e]] += amount;
          excess_[u] -= amount;
          excess_[v] += amount;
          if (v != s && v != t && !queued_[v] && level_[v] < g_.n) {
            queued_[v] = 1;
            active.push_back(v);
          }
        } else
          ++e;
        ++work;
      }
    }
    if (work > 4ULL * g_.cap.size() + g_.n) {
      global_relabel();
      work = 0;
    }
  }
  return excess_[t];
}
} // namespace trade
