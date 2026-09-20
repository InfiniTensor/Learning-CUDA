#include "gvs/search.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <unordered_set>
#include <utility>

namespace gvs {
namespace {

float distance(const float* a, const float* b, std::size_t dim, Metric metric) {
    float value = 0.0f;
    if (metric == Metric::L2) {
        for (std::size_t d = 0; d < dim; ++d) {
            const float diff = a[d] - b[d];
            value += diff * diff;
        }
        return value;
    }
    for (std::size_t d = 0; d < dim; ++d) value += a[d] * b[d];
    return -value;  // Smaller is better internally for both metrics.
}

}  // namespace

SearchResult cpu_exact_search(const std::vector<float>& database,
                              const std::vector<float>& queries,
                              std::size_t n, std::size_t nq,
                              std::size_t dim, Metric metric,
                              std::size_t k) {
    if (database.size() != n * dim || queries.size() != nq * dim)
        throw std::invalid_argument("input shape does not match n/nq/dim");
    k = std::min(k, n);
    SearchResult output{nq, k, std::vector<int>(nq * k),
                        std::vector<float>(nq * k)};
    std::vector<std::pair<float, int>> candidates(n);
    for (std::size_t q = 0; q < nq; ++q) {
        const float* query = queries.data() + q * dim;
        for (std::size_t i = 0; i < n; ++i) {
            candidates[i] = {distance(query, database.data() + i * dim, dim, metric),
                             static_cast<int>(i)};
        }
        std::partial_sort(candidates.begin(), candidates.begin() + k,
                          candidates.end());
        for (std::size_t j = 0; j < k; ++j) {
            output.ids[q * k + j] = candidates[j].second;
            output.scores[q * k + j] =
                metric == Metric::InnerProduct ? -candidates[j].first
                                               : candidates[j].first;
        }
    }
    return output;
}

double recall_at_k(const SearchResult& truth, const SearchResult& candidate) {
    if (truth.queries != candidate.queries || truth.k != candidate.k)
        throw std::invalid_argument("recall inputs have different shapes");
    std::size_t hits = 0;
    for (std::size_t q = 0; q < truth.queries; ++q) {
        std::unordered_set<int> expected;
        for (std::size_t j = 0; j < truth.k; ++j)
            expected.insert(truth.ids[q * truth.k + j]);
        for (std::size_t j = 0; j < candidate.k; ++j)
            hits += expected.count(candidate.ids[q * candidate.k + j]);
    }
    const auto total = truth.queries * truth.k;
    return total == 0 ? 1.0 : static_cast<double>(hits) / total;
}

}  // namespace gvs

