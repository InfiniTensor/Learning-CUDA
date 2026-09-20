// types.h - 向量检索引擎的基础类型定义
#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace vs {

// ======================= 枚举 =======================
enum class Metric { L2 = 0, INNER_PRODUCT = 1, COSINE = 2 };
enum class SearchMode { EXACT = 0, IVF_FLAT = 1, IVF_PQ = 2 };
enum class Dtype { FP32 = 0, FP16 = 1 };

const char* metric_name(Metric m);
const char* mode_name(SearchMode m);
const char* dtype_name(Dtype d);

// ======================= 检索配置 =======================
struct Config {
    int top_k = 10;
    SearchMode mode = SearchMode::EXACT;
    int batch_size = 128;  // 查询批大小
    int nlist = 4096;      // IVF 聚类中心数
    int nprobe = 16;       // 查询时访问的倒排桶数
    int pq_m = 16;         // PQ 子空间数（IVF-PQ 使用）
    int kmeans_iters = 20; // k-means 迭代次数
    uint32_t seed = 1234;
    int exact_block = 256;    // 精确检索的线程块大小
    int iters_per_thread = 8; // 每线程每轮处理向量数
    bool no_cpu = false;      // 跳过 CPU 参考（节省时间）
};

// ======================= 向量集合 =======================
// 统一以 fp32 在内存中存储，加载时按原 dtype 转换
struct VectorSet {
    int64_t n = 0;   // 向量条数
    int32_t dim = 0; // 维度
    Dtype dtype = Dtype::FP32;
    Metric metric = Metric::L2;
    std::vector<float> data; // n * dim，行主序
    int64_t bytes() const {
        return n * dim * (int64_t)sizeof(float);
    }
};

// ======================= 检索结果 =======================
struct SearchResult {
    int64_t nq = 0;
    int k = 0;
    std::vector<int32_t> ids;       // nq * k，按相似度从优到劣
    std::vector<float> dists;       // nq * k，与 ids 对应
    std::vector<double> latency_ms; // 每个 query 的耗时（用于 P50/P99）
};

// ======================= IVF 索引 =======================
struct IvfIndex {
    int32_t nlist = 0;
    int32_t dim = 0;
    Metric metric = Metric::L2;
    int32_t pq_m = 0;                // 0 表示 IVF-Flat（无 PQ）
    std::vector<float> centroids;    // nlist * dim
    std::vector<int32_t> list_start; // nlist + 1，CSR 风格
    std::vector<int32_t> list_ids;   // n，各倒排桶中的向量 id
    // PQ 附加数据（仅 IVF-PQ 使用）
    std::vector<float> pq_codebook; // pq_m * 256 * (dim / pq_m)
    std::vector<uint8_t> pq_codes;  // n * pq_m

    bool empty() const {
        return nlist == 0 || centroids.empty();
    }
};

// ======================= 性能统计 =======================
struct PerfStats {
    double build_ms = 0.0;      // 建索引耗时
    double search_ms = 0.0;     // 查询总耗时（kernel 时间）
    double qps = 0.0;           // 每秒查询数
    double p50_ms = 0.0;        // 单查询延迟中位数
    double p99_ms = 0.0;        // 单查询延迟 P99
    double gpu_mem_mb = 0.0;    // 显存占用
    double speedup = 0.0;       // 相对 CPU 精确检索的加速比
    double scanned_ratio = 0.0; // 实际扫描向量占比（近似检索）
};

} // namespace vs
