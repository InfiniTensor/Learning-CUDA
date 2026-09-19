#include "vector_search/search.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <unordered_set>
#include <utility>
#include <vector>

namespace vector_search {
namespace {

struct Candidate {
  float score = 0.0f;
  std::int64_t id = -1;
};

struct BetterCandidate {
  Metric metric;

  bool operator()(const Candidate& left, const Candidate& right) const {
    if (left.score != right.score) {
      return metric == Metric::kL2 ? left.score < right.score
                                   : left.score > right.score;
    }
    return left.id < right.id;
  }
};

float score_vector(const float* query, const float* vector, int dim,
                   Metric metric) {
  // 三种度量的定义与 CUDA 后端一致。
  float dot = 0.0f;
  float query_norm = 0.0f;
  float vector_norm = 0.0f;
  if (metric == Metric::kL2) {
    float distance = 0.0f;
    for (int component = 0; component < dim; ++component) {
      const float difference = query[component] - vector[component];
      distance += difference * difference;
    }
    return distance;
  }
  for (int component = 0; component < dim; ++component) {
    dot += query[component] * vector[component];
    if (metric == Metric::kCosine) {
      query_norm += query[component] * query[component];
      vector_norm += vector[component] * vector[component];
    }
  }
  if (metric == Metric::kCosine) {
    const float denominator = std::sqrt(query_norm * vector_norm);
    return denominator > 0.0f ? dot / denominator : 0.0f;
  }
  return dot;
}

void validate_inputs(const VectorDatabase& database, const QuerySet& queries,
                     int top_k, int batch_size) {
  // 搜索前检查维度、Top-K 和缓冲区长度。
  if (database.count <= 0 || database.dim <= 0 || queries.count <= 0 ||
      queries.dim != database.dim || top_k <= 0 || top_k > 100 ||
      top_k > database.count || batch_size <= 0 ||
      database.values.size() !=
          static_cast<std::size_t>(database.count) * database.dim ||
      queries.values.size() !=
          static_cast<std::size_t>(queries.count) * queries.dim) {
    throw std::invalid_argument("invalid database, query, Top-K or batch dimensions");
  }
}

SearchResults search_cpu_impl(
    const VectorDatabase& database, const QuerySet& queries,
    const std::vector<std::vector<std::int64_t>>* candidate_ids, int top_k,
    int batch_size) {
  // candidate_ids 为空时扫描全库，否则仅扫描候选；大小为 K 的堆维护当前最优结果。
  validate_inputs(database, queries, top_k, batch_size);
  if (candidate_ids != nullptr &&
      candidate_ids->size() != static_cast<std::size_t>(queries.count)) {
    throw std::invalid_argument("candidate-list count must match query count");
  }

  SearchResults results;
  results.num_queries = queries.count;
  results.top_k = top_k;
  results.ids.assign(static_cast<std::size_t>(queries.count) * top_k, -1);
  const float missing_score = database.metric == Metric::kL2
                                  ? std::numeric_limits<float>::infinity()
                                  : -std::numeric_limits<float>::infinity();
  results.scores.assign(results.ids.size(), missing_score);
  results.query_latency_ms.resize(static_cast<std::size_t>(queries.count));
  results.backend = "cpu";
  results.working_set_bytes =
      database.values.size() * sizeof(float) +
      queries.values.size() * sizeof(float) +
      static_cast<std::uint64_t>(top_k) * sizeof(Candidate);
  const BetterCandidate better{database.metric};

  for (std::int64_t batch_begin = 0; batch_begin < queries.count;
       batch_begin += batch_size) {
    const std::int64_t batch_end =
        std::min<std::int64_t>(queries.count, batch_begin + batch_size);
    for (std::int64_t query_index = batch_begin; query_index < batch_end;
         ++query_index) {
      const auto start = std::chrono::steady_clock::now();
      const float* query =
          queries.values.data() + query_index * queries.dim;
      std::vector<Candidate> heap;
      heap.reserve(static_cast<std::size_t>(top_k));

      const auto consider = [&](std::int64_t vector_id) {
        if (vector_id < 0 || vector_id >= database.count) {
          throw std::runtime_error("candidate vector id is out of range");
        }
        const float* vector =
            database.values.data() + vector_id * database.dim;
        const Candidate candidate{
            score_vector(query, vector, database.dim, database.metric),
            vector_id};
        if (heap.size() < static_cast<std::size_t>(top_k)) {
          heap.push_back(candidate);
          std::push_heap(heap.begin(), heap.end(), better);
        } else if (better(candidate, heap.front())) {
          std::pop_heap(heap.begin(), heap.end(), better);
          heap.back() = candidate;
          std::push_heap(heap.begin(), heap.end(), better);
        }
      };

      if (candidate_ids == nullptr) {
        for (std::int64_t vector_id = 0; vector_id < database.count;
             ++vector_id) {
          consider(vector_id);
        }
        results.evaluated_candidates +=
            static_cast<std::uint64_t>(database.count);
        results.peak_candidates = std::max<std::uint64_t>(
            results.peak_candidates, static_cast<std::uint64_t>(database.count));
      } else {
        const auto& candidates =
            (*candidate_ids)[static_cast<std::size_t>(query_index)];
        for (const auto vector_id : candidates) consider(vector_id);
        results.evaluated_candidates += candidates.size();
        results.peak_candidates = std::max<std::uint64_t>(
            results.peak_candidates,
            static_cast<std::uint64_t>(candidates.size()));
      }

      std::sort(heap.begin(), heap.end(), better);
      // 同分时按较小 ID 排序。
      for (std::size_t rank = 0; rank < heap.size(); ++rank) {
        const auto output =
            static_cast<std::size_t>(query_index) * top_k + rank;
        results.ids[output] = heap[rank].id;
        results.scores[output] = heap[rank].score;
      }
      const auto end = std::chrono::steady_clock::now();
      results.query_latency_ms[static_cast<std::size_t>(query_index)] =
          std::chrono::duration<double, std::milli>(end - start).count();
    }
  }
  return results;
}

}  // namespace

SearchResults search_exact_cpu(const VectorDatabase& database,
                               const QuerySet& queries, int top_k,
                               int batch_size) {
  // 全库精确搜索，作为正确性与召回率参考。
  return search_cpu_impl(database, queries, nullptr, top_k, batch_size);
}

SearchResults search_candidates_cpu(
    const VectorDatabase& database, const QuerySet& queries,
    const std::vector<std::vector<std::int64_t>>& candidate_ids, int top_k,
    int batch_size) {
  // 对 IVF 候选做精确重排。
  return search_cpu_impl(database, queries, &candidate_ids, top_k, batch_size);
}

QuerySet slice_queries(const QuerySet& queries, std::int64_t count) {
  // 仅用前 quality_queries 条查询评估质量。
  if (count <= 0 || count > queries.count || queries.dim <= 0) {
    throw std::invalid_argument("invalid query slice");
  }
  QuerySet sliced;
  sliced.count = count;
  sliced.dim = queries.dim;
  sliced.source_dtype = queries.source_dtype;
  sliced.values.assign(queries.values.begin(),
                       queries.values.begin() + count * queries.dim);
  return sliced;
}

double recall_at_k(const SearchResults& approximate,
                   const SearchResults& ground_truth) {
  // Recall@K 比较 ID 集合，不比较位次。
  if (approximate.num_queries != ground_truth.num_queries ||
      approximate.top_k != ground_truth.top_k || approximate.top_k <= 0) {
    throw std::invalid_argument("incompatible results for recall@K");
  }
  std::uint64_t matches = 0;
  for (std::int64_t query = 0; query < ground_truth.num_queries; ++query) {
    std::unordered_set<std::int64_t> expected;
    for (int rank = 0; rank < ground_truth.top_k; ++rank) {
      const auto id = ground_truth.ids[static_cast<std::size_t>(query) *
                                           ground_truth.top_k +
                                       rank];
      if (id >= 0) expected.insert(id);
    }
    for (int rank = 0; rank < approximate.top_k; ++rank) {
      const auto id = approximate.ids[static_cast<std::size_t>(query) *
                                          approximate.top_k +
                                      rank];
      if (id >= 0 && expected.count(id) != 0) ++matches;
    }
  }
  const double denominator = static_cast<double>(ground_truth.num_queries) *
                             ground_truth.top_k;
  return denominator > 0.0 ? static_cast<double>(matches) / denominator : 0.0;
}

double rank_id_agreement(const SearchResults& candidate,
                         const SearchResults& ground_truth) {
  // 逐位比较 ID；近同分结果交换位置也会降低此指标。
  if (candidate.num_queries != ground_truth.num_queries ||
      candidate.top_k != ground_truth.top_k || candidate.top_k <= 0 ||
      candidate.ids.size() != ground_truth.ids.size()) {
    throw std::invalid_argument("incompatible results for rank ID agreement");
  }
  std::uint64_t matches = 0;
  for (std::size_t index = 0; index < candidate.ids.size(); ++index) {
    if (candidate.ids[index] == ground_truth.ids[index]) ++matches;
  }
  return candidate.ids.empty()
             ? 0.0
             : static_cast<double>(matches) / candidate.ids.size();
}

double mean_rank_score_error(const SearchResults& approximate,
                             const SearchResults& ground_truth) {
  // 按位次比较分数，不按 ID 对齐。
  if (approximate.num_queries != ground_truth.num_queries ||
      approximate.top_k != ground_truth.top_k || approximate.top_k <= 0) {
    throw std::invalid_argument("incompatible results for score error");
  }
  double total = 0.0;
  std::uint64_t count = 0;
  for (std::size_t index = 0; index < approximate.scores.size(); ++index) {
    if (approximate.ids[index] >= 0 && ground_truth.ids[index] >= 0 &&
        std::isfinite(approximate.scores[index]) &&
        std::isfinite(ground_truth.scores[index])) {
      total += std::abs(static_cast<double>(approximate.scores[index]) -
                        ground_truth.scores[index]);
      ++count;
    }
  }
  return count > 0 ? total / static_cast<double>(count) : 0.0;
}

}  // namespace vector_search
