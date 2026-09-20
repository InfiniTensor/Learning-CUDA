#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

// 命名空间 vs = "vector search"。
// 整个引擎的公共接口都在这里声明,由 src/ 下各文件实现。
namespace vs {

// 输入文件的数值精度。FP16 文件读入后统一转成 FP32 计算。
enum class DType : std::uint32_t { kFp32 = 1, kFp16 = 2 };

// 距离度量。当前只实现 squared-L2(省略 sqrt 不改变最近邻排序)。
enum class Metric : std::uint32_t { kL2 = 1 };

// 一组向量(向量库或查询集)。
struct VectorDataset {
  std::uint64_t count = 0;   // 向量个数
  std::uint32_t dim = 0;     // 向量维度
  DType source_dtype = DType::kFp32;  // 磁盘上原始精度
  Metric metric = Metric::kL2;
  // Host 侧统一用 FP32 行主序(row-major)保存;FP16 文件在 load 时转换。
  std::vector<float> values;
};

// 检索配置,来自 configs/ 下的 .toml 文件(实际是极简 key=value 解析)。
struct SearchConfig {
  std::uint32_t top_k = 10;   // 返回前 K 个
  std::string search_mode = "exact";  // "exact" 或 "ivf_flat"
  std::uint32_t batch_size = 128;  // 一次拷到 GPU 的查询数(分块传输)
  std::uint32_t nlist = 4096;   // IVF: 桶(倒排列表)的个数
  std::uint32_t nprobe = 16;    // IVF: 每个查询探查的桶数(精度-速度旋钮)
  std::uint32_t kmeans_iters = 12;  // IVF 建索引时的 k-means 迭代轮数
  std::uint32_t train_size = 65536;  // IVF 建索引时的训练样本数
  std::uint64_t seed = 2026;    // 随机种子,保证结果可复现
};

// 一次检索的结果:每个查询的 Top-K 的 ID 与距离。
struct SearchResult {
  std::uint64_t num_queries = 0;  // 查询个数
  std::uint32_t top_k = 0;        // K
  Metric metric = Metric::kL2;
  std::vector<std::uint64_t> ids;    // 大小 num_queries*top_k,原始向量 ID
  std::vector<float> scores;         // 大小 num_queries*top_k,与 ids 一一对应
};

// 性能指标,最终会写成 JSON 日志。
struct SearchMetrics {
  double build_ms = 0.0;   // 建索引耗时(仅 build 命令填写)
  double total_ms = 0.0;   // search API 全程墙钟时间(含传输,不含文件 IO)
  double gpu_ms = 0.0;     // CUDA event 测得的 GPU 时间(H2D+kernel+D2H)
  double qps = 0.0;        // 每秒查询数 = 查询数 / total_ms
  double p50_ms = 0.0;     // 各 batch 延迟的中位数
  double p99_ms = 0.0;     // 各 batch 延迟的 99 分位(正式统计至少 100 个样本)
  double cpu_baseline_ms = 0.0;  // CPU 参考实现耗时(对比加速比用)
  double speedup_vs_cpu = 0.0;   // CPU 耗时 / GPU 耗时
  std::size_t peak_device_bytes = 0;  // 显存占用估算(非 profiler 实测)
};

// 近似检索质量,与 ground truth(Exact 结果)对比得到。
struct QualityMetrics {
  double recall_at_k = 0.0;          // Top-K 里与 Exact 重合的比例
  double mean_distance_error = 0.0;  // 各 rank 平均超额距离(非负)
};

// IVF 索引的宿主(CPU)表示。查询前会整块拷到 GPU。
struct IvfIndexData {
  std::uint64_t count = 0;  // 向量总数
  std::uint32_t dim = 0;
  std::uint32_t nlist = 0;  // 桶数
  Metric metric = Metric::kL2;
  // centroids 行主序;重排后的向量按"维度主序"存储。
  std::vector<float> centroids;   // nlist*dim,每个桶的质心
  std::vector<std::uint64_t> offsets;  // nlist+1 前缀和,桶 [offsets[i], offsets[i+1])
  std::vector<std::uint32_t> original_ids;  // count,重排后的每个位置对应的原始 ID
  std::vector<float> vectors_soa;  // dim*count,维度主序,桶内连续
};

// ---- IO 层 ----
VectorDataset load_vectors(const std::string& path, bool is_query);
void save_vectors(const std::string& path, const VectorDataset& data,
                  bool is_query, DType dtype);
SearchConfig load_config(const std::string& path);
void save_results(const std::string& path, const SearchResult& result);
SearchResult load_results(const std::string& path);
void save_index(const std::string& path, const IvfIndexData& index);
IvfIndexData load_index(const std::string& path);
void save_performance_log(const std::string& path,
                          const SearchConfig& config,
                          const SearchMetrics& metrics);
void save_quality_log(const std::string& path,
                      const SearchConfig& config,
                      const QualityMetrics& metrics);

// ---- CPU 参考(正确性基准)----
SearchResult cpu_exact_search(const VectorDataset& database,
                              const VectorDataset& queries,
                              std::uint32_t top_k,
                              SearchMetrics* metrics = nullptr);
QualityMetrics measure_quality(const SearchResult& approximate,
                               const SearchResult& ground_truth);

// ---- GPU 精确索引 ----
class FlatGpuIndex {
 public:
  explicit FlatGpuIndex(const VectorDataset& database);  // 构造时转置为维度主序
  ~FlatGpuIndex();
  FlatGpuIndex(const FlatGpuIndex&) = delete;
  FlatGpuIndex& operator=(const FlatGpuIndex&) = delete;

  SearchResult search(const VectorDataset& queries, std::uint32_t top_k,
                      std::uint32_t batch_size,
                      SearchMetrics* metrics = nullptr) const;

 private:
  std::uint64_t count_ = 0;
  std::uint32_t dim_ = 0;
  float* device_vectors_soa_ = nullptr;  // 显存中的 [dim, count] 布局
};

// ---- IVF 索引构建(离线一次)----
IvfIndexData build_ivf_flat(const VectorDataset& database,
                            const SearchConfig& config,
                            SearchMetrics* metrics = nullptr);

// ---- GPU IVF 索引(在线查询)----
class IvfGpuIndex {
 public:
  explicit IvfGpuIndex(const IvfIndexData& index);  // 整块拷入显存
  ~IvfGpuIndex();
  IvfGpuIndex(const IvfGpuIndex&) = delete;
  IvfGpuIndex& operator=(const IvfGpuIndex&) = delete;

  SearchResult search(const VectorDataset& queries, std::uint32_t top_k,
                      std::uint32_t nprobe, std::uint32_t batch_size,
                      SearchMetrics* metrics = nullptr) const;

 private:
  std::uint64_t count_ = 0;
  std::uint32_t dim_ = 0;
  std::uint32_t nlist_ = 0;
  std::vector<std::uint64_t> host_offsets_;  // host 侧副本,校验候选数用
  float* device_vectors_soa_ = nullptr;      // 显存中的桶内重排向量
  std::uint64_t* device_offsets_ = nullptr;  // 显存中的桶偏移
  std::uint32_t* device_ids_ = nullptr;      // 显存中的原始 ID
  FlatGpuIndex* centroid_index_ = nullptr;   // 复用它做"找最近质心"的 Top-K 搜索
};

bool cuda_available(std::string* reason = nullptr);

}  // namespace vs
