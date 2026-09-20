// metrics.h - 召回率、延迟分位与质量指标
#pragma once
#include "types.h"
#include <vector>

namespace vs {

// recall@K：近似检索返回的 K 个 id 中，命中 ground truth 前 K 名的比例，
// 对所有查询求平均（结果 ∈ [0,1]）
double recall_at_k(const SearchResult& approx, const SearchResult& gt);

// 在两侧共同命中的 id 上，计算距离的平均绝对差（衡量近似检索的精度损失）
double mean_dist_error(const SearchResult& approx, const SearchResult& gt);

// 升序排序后取分位数，p ∈ [0,1]
double percentile(std::vector<double> v, double p);

// 由每条查询的延迟计算 P50 / P99
void fill_latency_stats(const std::vector<double>& lat, double& p50, double& p99);

// ======================= 结果一致性比对 =======================
// 用于验证“GPU 精确检索结果 == CPU 参考结果”。
// 由于浮点累加顺序不同，距离可能差最后几位，因此：
//   * 逐名次 id 不一致 -> rank_mismatch
//   * 每个 query 的 id 集合不一致 -> set_mismatch
//   * 距离差异按相对误差统计
struct CompareResult {
    int64_t rank_mismatch = 0;    // 逐名次 id 不同的条目数
    int64_t set_mismatch = 0;     // id 集合不一致的 query 数
    int64_t total_pairs = 0;      // 比较的条目总数
    double max_rel_dist_diff = 0; // 最大相对距离差
};

CompareResult compare_results(const SearchResult& a, const SearchResult& b);

} // namespace vs
