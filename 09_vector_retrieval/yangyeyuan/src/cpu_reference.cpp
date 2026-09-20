#include "vector_search.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <unordered_set>
#include <utility>
#include <vector>

namespace vs {

// CPU 精确检索:朴素三重循环,是正确性基准(ground truth)。
// 不做任何 GPU 优化,只保证"绝对正确"。
SearchResult cpu_exact_search(const VectorDataset& database,
                              const VectorDataset& queries,
                              std::uint32_t top_k,
                              SearchMetrics* metrics) {
  if (database.dim != queries.dim) throw std::runtime_error("database/query dimension mismatch");
  if (database.metric != Metric::kL2) throw std::runtime_error("only l2 is supported");
  if (top_k == 0 || top_k > database.count) throw std::runtime_error("invalid top_k");
  const auto begin = std::chrono::steady_clock::now();

  SearchResult result;
  result.num_queries = queries.count;
  result.top_k = top_k;
  result.metric = Metric::kL2;
  result.ids.resize(static_cast<std::size_t>(queries.count) * top_k);
  result.scores.resize(static_cast<std::size_t>(queries.count) * top_k);
  // 每个查询临时保存 (距离, ID) 候选,然后 partial_sort 取前 K。
  std::vector<std::pair<float, std::uint64_t>> candidates(database.count);

  for (std::uint64_t q = 0; q < queries.count; ++q) {
    const float* query = queries.values.data() + static_cast<std::size_t>(q) * queries.dim;
    for (std::uint64_t i = 0; i < database.count; ++i) {
      const float* vector = database.values.data() + static_cast<std::size_t>(i) * database.dim;
      float distance = 0.0f;
      // squared-L2: 逐维 (q_j - x_j)^2 累加。
      for (std::uint32_t d = 0; d < database.dim; ++d) {
        const float delta = query[d] - vector[d];
        distance += delta * delta;
      }
      // 非有限距离统一记为 +Inf,使其排到末尾(与 GPU 端规则一致)。
      if (!std::isfinite(distance)) distance = std::numeric_limits<float>::infinity();
      candidates[i] = {distance, i};
    }
    // 只排序前 top_k 名:部分排序,比全排序快。
    std::partial_sort(candidates.begin(), candidates.begin() + top_k, candidates.end(),
                      [](const auto& a, const auto& b) {
                        // 距离升序;距离相同时 ID 小的优先(与 GPU 的 tie 规则一致)。
                        return a.first < b.first || (a.first == b.first && a.second < b.second);
                      });
    for (std::uint32_t k = 0; k < top_k; ++k) {
      const auto output = static_cast<std::size_t>(q) * top_k + k;
      result.scores[output] = candidates[k].first;
      result.ids[output] = candidates[k].second;
    }
  }

  const auto end = std::chrono::steady_clock::now();
  if (metrics) {
    metrics->total_ms = std::chrono::duration<double, std::milli>(end - begin).count();
    metrics->gpu_ms = 0.0;
    metrics->qps = queries.count * 1000.0 / metrics->total_ms;
  }
  return result;
}

// 评估近似检索质量:与 ground truth(Exact 结果)对比。
QualityMetrics measure_quality(const SearchResult& approximate,
                               const SearchResult& ground_truth) {
  if (approximate.num_queries != ground_truth.num_queries ||
      approximate.top_k != ground_truth.top_k || approximate.metric != ground_truth.metric) {
    throw std::runtime_error("quality inputs have different shapes or metrics");
  }
  const std::size_t total = static_cast<std::size_t>(approximate.num_queries) * approximate.top_k;
  if (approximate.ids.size() != total || approximate.scores.size() != total ||
      ground_truth.ids.size() != total || ground_truth.scores.size() != total) {
    throw std::runtime_error("malformed quality input");
  }

  std::uint64_t hits = 0;
  double distance_error = 0.0;
  // 把 ground truth 的 Top-K ID 放进集合,方便查"近似结果是否命中"。
  std::unordered_set<std::uint64_t> exact_ids;
  exact_ids.reserve(ground_truth.top_k * 2);
  for (std::uint64_t q = 0; q < approximate.num_queries; ++q) {
    exact_ids.clear();
    const std::size_t base = static_cast<std::size_t>(q) * approximate.top_k;
    for (std::uint32_t k = 0; k < approximate.top_k; ++k) {
      exact_ids.insert(ground_truth.ids[base + k]);
    }
    for (std::uint32_t k = 0; k < approximate.top_k; ++k) {
      // recall: 近似结果的每个 ID 若在 Exact 集合里则命中一次。
      hits += exact_ids.count(approximate.ids[base + k]);
      // 距离误差:近似结果距离相比 Exact 同 rank 的"超额",取非负。
      distance_error += std::max(0.0,
          static_cast<double>(approximate.scores[base + k]) - ground_truth.scores[base + k]);
    }
  }
  QualityMetrics metrics;
  metrics.recall_at_k = total == 0 ? 0.0 : static_cast<double>(hits) / total;
  metrics.mean_distance_error = total == 0 ? 0.0 : distance_error / total;
  return metrics;
}

}  // namespace vs
