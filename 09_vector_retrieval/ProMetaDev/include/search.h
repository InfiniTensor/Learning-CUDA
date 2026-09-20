// search.h - CPU 参考实现与 GPU 检索接口
#pragma once
#include "types.h"
#include <string>
#include <vector>

namespace vs {

// ======================= CPU 参考 =======================
// 精确检索（全量扫描）+ Top-K。结果按“越相似越靠前”排序：
//   L2 / cosine：距离升序；inner_product：相似度降序
SearchResult cpu_exact_search(const VectorSet& base, const VectorSet& queries, int top_k,
                              Metric metric);

// CPU k-means（Lloyd 迭代）。data: n*dim，k 个聚类中心写入 centroids (k*dim)
void cpu_kmeans(const float* data, int64_t n, int dim, int k, int iters, uint32_t seed,
                std::vector<float>& centroids);

// ======================= GPU 接口 =======================
// 精确检索（全量扫描）
bool gpu_exact_search(const VectorSet& base, const VectorSet& queries, const Config& cfg,
                      SearchResult& out, PerfStats& perf, double* cpu_ms_for_speedup,
                      std::string& err);

// 建立 IVF 索引（k-means 聚类 + 倒排表分配）
bool gpu_build_ivf(const VectorSet& base, const Config& cfg, IvfIndex& idx, PerfStats& perf,
                   std::string& err);

// IVF 查询（Flat 或 PQ）
bool gpu_ivf_search(const VectorSet& base, const VectorSet& queries, const IvfIndex& idx,
                    const Config& cfg, SearchResult& out, PerfStats& perf, std::string& err);

} // namespace vs
