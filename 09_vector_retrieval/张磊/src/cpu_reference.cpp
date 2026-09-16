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
//
// CPU 参考打分（精确基线）：与 GPU 端保持同一套度量语义，用 double 累加以
// 获得最高精度。注意：
//   * L2 返回平方距离（仅用于排序），调用方最终输出欧氏距离；
//   * inner product / cosine 返回点积，cosine 再取 1 - dot；
//   * L2 分支里对 acc 的点积累加是历史遗留，不参与返回，保留不影响结果。
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

// 把“内部用于排序的原始分值”转换为对外输出的距离/相似度：
// L2 对平方距离开根，其余度量原样返回。
double emittedScore(Metric m, double raw) {
  return (m == Metric::L2) ? std::sqrt(std::max(raw, 0.0)) : raw;
}

}  // namespace

// 单线程暴力精确检索：对每个 query 计算全库距离，取 Top-K。
// 这是 GPU exact 与近似检索的正确性/召回基准，排序规则与 GPU 完全一致：
//   * 度量决定“越小越好”（L2/cosine）还是“越大越好”（inner product）；
//   * 分值相同时按向量 id 升序，保证 tie-break 确定性、可与 GPU 逐位对比。
// 返回值为该子集的墙钟耗时（毫秒），用于计算相对 CPU 的加速比。
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
    // partial_sort 只排出前 topK 个，避免全量排序的额外开销。
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
