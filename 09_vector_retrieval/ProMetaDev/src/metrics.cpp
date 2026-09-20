// metrics.cpp - 召回率与延迟指标实现
#include "metrics.h"

#include <algorithm>
#include <cmath>
#include <unordered_set>

namespace vs {

double recall_at_k(const SearchResult& approx, const SearchResult& gt) {
    if (approx.nq == 0 || approx.k == 0)
        return 0.0;
    const int k = std::min(approx.k, gt.k);
    double sum = 0.0;
    for (int64_t i = 0; i < approx.nq; i++) {
        std::unordered_set<int32_t> truth;
        truth.reserve((size_t)k * 2);
        for (int j = 0; j < k; j++)
            truth.insert(gt.ids[(size_t)(i * gt.k + j)]);
        int hit = 0;
        for (int j = 0; j < k; j++) {
            int32_t id = approx.ids[(size_t)(i * approx.k + j)];
            if (id >= 0 && truth.count(id))
                hit++;
        }
        sum += (double)hit / (double)k;
    }
    return sum / (double)approx.nq;
}

double mean_dist_error(const SearchResult& approx, const SearchResult& gt) {
    double sum = 0.0;
    int64_t cnt = 0;
    for (int64_t i = 0; i < approx.nq; i++) {
        for (int j = 0; j < approx.k; j++) {
            int32_t id = approx.ids[(size_t)(i * approx.k + j)];
            if (id < 0)
                continue;
            // 在 gt 中查找同一个 id 的距离
            for (int t = 0; t < gt.k; t++) {
                if (gt.ids[(size_t)(i * gt.k + t)] == id) {
                    sum += std::fabs((double)approx.dists[(size_t)(i * approx.k + j)] -
                                     (double)gt.dists[(size_t)(i * gt.k + t)]);
                    cnt++;
                    break;
                }
            }
        }
    }
    return cnt > 0 ? sum / (double)cnt : 0.0;
}

double percentile(std::vector<double> v, double p) {
    if (v.empty())
        return 0.0;
    std::sort(v.begin(), v.end());
    if (p <= 0.0)
        return v.front();
    if (p >= 1.0)
        return v.back();
    double idx = p * (double)(v.size() - 1);
    size_t lo = (size_t)std::floor(idx);
    size_t hi = (size_t)std::ceil(idx);
    if (lo == hi)
        return v[lo];
    double frac = idx - (double)lo;
    return v[lo] * (1.0 - frac) + v[hi] * frac;
}

void fill_latency_stats(const std::vector<double>& lat, double& p50, double& p99) {
    p50 = percentile(lat, 0.50);
    p99 = percentile(lat, 0.99);
}

CompareResult compare_results(const SearchResult& a, const SearchResult& b) {
    CompareResult r;
    const int64_t nq = std::min(a.nq, b.nq);
    const int k = std::min(a.k, b.k);
    for (int64_t i = 0; i < nq; i++) {
        std::unordered_set<int32_t> sa, sb;
        sa.reserve((size_t)k * 2);
        sb.reserve((size_t)k * 2);
        for (int j = 0; j < k; j++) {
            int32_t ia = a.ids[(size_t)(i * a.k + j)];
            int32_t ib = b.ids[(size_t)(i * b.k + j)];
            sa.insert(ia);
            sb.insert(ib);
            r.total_pairs++;
            if (ia != ib)
                r.rank_mismatch++;
            double da = a.dists[(size_t)(i * a.k + j)];
            double db = b.dists[(size_t)(i * b.k + j)];
            double denom = std::max(std::fabs(db), 1e-12);
            r.max_rel_dist_diff = std::max(r.max_rel_dist_diff, std::fabs(da - db) / denom);
        }
        if (sa != sb)
            r.set_mismatch++;
    }
    return r;
}

} // namespace vs
