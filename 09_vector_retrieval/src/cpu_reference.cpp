#include "vsearch/cpu_reference.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>

namespace vs {
namespace {

// Raw monotone score used by the CPU oracle. For L2 the squared distance is
// used for ranking and the caller receives the Euclidean distance after the
// top-K selection.
double scoreRaw(const Dataset& ds, const float* q, const float* x) {
  const i32 dim = ds.dim;
  double acc = 0.0;
  if (ds.metric == Metric::L2) {
    double sq = 0.0;
    for (i32 d = 0; d < dim; ++d) {
      const double diff = static_cast<double>(q[d]) - x[d];
      acc += static_cast<double>(q[d]) * x[d];
      sq += diff * diff;
    }
    return sq;
  }
  for (i32 d = 0; d < dim; ++d)
    acc += static_cast<double>(q[d]) * x[d];
  if (ds.metric == Metric::Cosine) return 1.0 - acc;  // data pre-normalized
  return acc;
}

double emittedScore(Metric m, double raw) {
  return (m == Metric::L2) ? std::sqrt(std::max(raw, 0.0)) : raw;
}

}  // namespace

double cpuExactSearch(const Dataset& ds, i64 nq, const float* queries,
                      int topK, std::vector<i32>* ids, FloatVec* scores) {
  if (ds.metric == Metric::None) throwRuntime("dataset metric is none");
  if (static_cast<i64>(topK) > ds.n)
    throwRuntime("cpu reference: top_k exceeds database size");
  ids->assign(static_cast<std::size_t>(nq) * topK, -1);
  scores->assign(static_cast<std::size_t>(nq) * topK, 0.0f);
  const i64 n = ds.n;
  const i32 dim = ds.dim;
  const bool smallerBetter = metricIsSmallerBetter(ds.metric);

  std::vector<double> dist(static_cast<std::size_t>(n));
  std::vector<i64> order(static_cast<std::size_t>(n));
  Timer t;
  for (i64 qi = 0; qi < nq; ++qi) {
    const float* q = queries + static_cast<std::size_t>(qi) * dim;
    for (i64 i = 0; i < n; ++i) {
      const float* x = ds.data.data() + static_cast<std::size_t>(i) * dim;
      dist[static_cast<std::size_t>(i)] = scoreRaw(ds, q, x);
    }
    std::iota(order.begin(), order.end(), 0);
    std::partial_sort(
        order.begin(), order.begin() + topK, order.end(),
        [&](i64 a, i64 b) {
          const double sa = dist[static_cast<std::size_t>(a)];
          const double sb = dist[static_cast<std::size_t>(b)];
          if (sa != sb) return smallerBetter ? (sa < sb) : (sa > sb);
          return a < b;
        });
    i32* idOut = ids->data() + static_cast<std::size_t>(qi) * topK;
    float* sOut = scores->data() + static_cast<std::size_t>(qi) * topK;
    for (int k = 0; k < topK; ++k) {
      const i64 id = order[static_cast<std::size_t>(k)];
      idOut[k] = static_cast<i32>(id);
      sOut[k] = static_cast<float>(emittedScore(ds.metric, dist[static_cast<std::size_t>(id)]));
    }
  }
  return t.ms();
}

}  // namespace vs
