// cpu_ref.cpp - CPU 精确检索参考实现（用于正确性对照）与 k-means 参考
//
// 说明：精确检索的时间复杂度为 O(nq * n * dim)。在 10^6 x 128 x 1000 的规模下
// 单线程耗时数十秒，因此这里用 OpenMP 并行（按 query 划分，结果与线程数无关，
// 具有确定性）。
#include "search.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <random>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace vs {

namespace {

// 距离/相似度语义：
//   L2            -> 平方欧氏距离，越小越好
//   INNER_PRODUCT -> 内积，越大越好
//   COSINE        -> 余弦相似度，越大越好
inline bool is_smaller_better(Metric m) {
    return m == Metric::L2;
}

inline float compute_score(const float* q, const float* b, int32_t dim, Metric metric) {
    float dot = 0.0f;
    if (metric == Metric::L2) {
        for (int32_t d = 0; d < dim; d++) {
            float t = q[d] - b[d];
            dot += t * t;
        }
        return dot;
    }
    for (int32_t d = 0; d < dim; d++)
        dot += q[d] * b[d];
    return dot;
}

// 有序插入：best 始终按“优在前”排列
inline void insert_candidate(std::vector<std::pair<float, int32_t>>& best, float score, int32_t id,
                             int k, bool smaller_better) {
    auto better = [smaller_better](float a, float b) { return smaller_better ? (a < b) : (a > b); };
    // 已满且不优于末位 -> 直接丢弃
    if ((int)best.size() >= k && !better(score, best.back().first))
        return;
    size_t pos = 0;
    while (pos < best.size() && !better(score, best[pos].first))
        pos++;
    best.insert(best.begin() + (long)pos, {score, id});
    if ((int)best.size() > k)
        best.pop_back();
}

} // namespace

SearchResult cpu_exact_search(const VectorSet& base, const VectorSet& queries, int top_k,
                              Metric metric) {
    SearchResult r;
    r.nq = queries.n;
    r.k = top_k;
    if (queries.n == 0 || base.n == 0 || top_k <= 0)
        return r;
    r.ids.assign((size_t)(queries.n * top_k), -1);
    r.dists.assign((size_t)(queries.n * top_k), 0.0f);
    r.latency_ms.assign((size_t)queries.n, 0.0);

    const int32_t dim = base.dim;
    const int64_t n = base.n;
    const bool smaller_better = is_smaller_better(metric);

    // cosine 需要归一化：预算库向量与查询向量的模长
    std::vector<float> base_norm, q_norm;
    if (metric == Metric::COSINE) {
        base_norm.resize((size_t)n);
        for (int64_t i = 0; i < n; i++) {
            const float* v = base.data.data() + i * dim;
            float s = 0.0f;
            for (int32_t d = 0; d < dim; d++)
                s += v[d] * v[d];
            base_norm[(size_t)i] = std::sqrt(s) + 1e-12f;
        }
        q_norm.resize((size_t)queries.n);
        for (int64_t i = 0; i < queries.n; i++) {
            const float* v = queries.data.data() + i * dim;
            float s = 0.0f;
            for (int32_t d = 0; d < dim; d++)
                s += v[d] * v[d];
            q_norm[(size_t)i] = std::sqrt(s) + 1e-12f;
        }
    }

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 4)
#endif
    for (int64_t qi = 0; qi < queries.n; qi++) {
        const float* q = queries.data.data() + qi * dim;
        std::vector<std::pair<float, int32_t>> best;
        best.reserve((size_t)top_k + 1);
        for (int64_t j = 0; j < n; j++) {
            const float* b = base.data.data() + j * dim;
            float score = compute_score(q, b, dim, metric);
            if (metric == Metric::COSINE)
                score = score / (q_norm[(size_t)qi] * base_norm[(size_t)j]);
            insert_candidate(best, score, (int32_t)j, top_k, smaller_better);
        }
        for (int j = 0; j < top_k; j++) {
            size_t idx = (size_t)(qi * top_k + j);
            if (j < (int)best.size()) {
                r.dists[idx] = best[(size_t)j].first;
                r.ids[idx] = best[(size_t)j].second;
            }
        }
    }
    return r;
}

void cpu_kmeans(const float* data, int64_t n, int dim, int k, int iters, uint32_t seed,
                std::vector<float>& centroids) {
    centroids.assign((size_t)((int64_t)k * dim), 0.0f);
    if (n <= 0 || k <= 0 || dim <= 0)
        return;
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int64_t> pick(0, n - 1);
    for (int c = 0; c < k; c++) {
        int64_t src = pick(rng);
        std::copy(data + src * dim, data + src * dim + dim,
                  centroids.begin() + (long)((int64_t)c * dim));
    }

    std::vector<int32_t> label((size_t)n, 0);
    std::vector<double> sum((size_t)((int64_t)k * dim), 0.0);
    std::vector<int64_t> cnt((size_t)k, 0);

    for (int it = 0; it < iters; it++) {
        // 分配
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int64_t i = 0; i < n; i++) {
            const float* x = data + i * dim;
            float best_d = std::numeric_limits<float>::max();
            int best_c = 0;
            for (int c = 0; c < k; c++) {
                const float* cen = centroids.data() + (int64_t)c * dim;
                float d = 0.0f;
                for (int t = 0; t < dim; t++) {
                    float df = x[t] - cen[t];
                    d += df * df;
                }
                if (d < best_d) {
                    best_d = d;
                    best_c = c;
                }
            }
            label[(size_t)i] = best_c;
        }
        // 更新
        std::fill(sum.begin(), sum.end(), 0.0);
        std::fill(cnt.begin(), cnt.end(), 0);
        for (int64_t i = 0; i < n; i++) {
            int c = label[(size_t)i];
            cnt[(size_t)c]++;
            const float* x = data + i * dim;
            double* s = sum.data() + (int64_t)c * dim;
            for (int t = 0; t < dim; t++)
                s[t] += x[t];
        }
        for (int c = 0; c < k; c++) {
            float* cen = centroids.data() + (int64_t)c * dim;
            if (cnt[(size_t)c] == 0)
                continue; // 空簇保持原位置
            double inv = 1.0 / (double)cnt[(size_t)c];
            const double* s = sum.data() + (int64_t)c * dim;
            for (int t = 0; t < dim; t++)
                cen[t] = (float)(s[t] * inv);
        }
    }
}

} // namespace vs
