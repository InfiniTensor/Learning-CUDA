#include "vector_search.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

namespace vs {

// 构建 IVF-Flat 索引(离线,一次):训练 → k-means 聚类 → 重排 + 前缀和偏移。
// 核心思想:把 nlist 个质心当"数据库",用 FlatGpuIndex 的精确 Top-1
// 给每个向量做"分到最近质心"的分配;CPU 只负责求平均更新质心。
IvfIndexData build_ivf_flat(const VectorDataset& database,
                            const SearchConfig& config,
                            SearchMetrics* metrics) {
  if (database.count == 0 || database.dim == 0 ||
      database.values.size() != database.count * database.dim) {
    throw std::runtime_error("invalid database");
  }
  if (database.count > std::numeric_limits<std::uint32_t>::max()) {
    throw std::runtime_error("IVF index supports at most UINT32_MAX vectors");
  }
  if (config.nlist == 0 || config.nlist > database.count || config.kmeans_iters == 0 ||
      config.batch_size == 0) {
    throw std::runtime_error("invalid IVF build parameters");
  }
  const auto begin = std::chrono::steady_clock::now();

  // ① 抽样训练集: 至少 nlist 个,最多 train_size 个(若库小则全量)。
  const std::uint64_t train_count = std::min<std::uint64_t>(
      database.count, std::max<std::uint64_t>(config.nlist, config.train_size));

  // 打乱顺序,取前 train_count 个作为训练样本(保证可复现)。
  std::vector<std::uint32_t> shuffled(database.count);
  std::iota(shuffled.begin(), shuffled.end(), 0u);
  std::mt19937_64 random(config.seed);
  std::shuffle(shuffled.begin(), shuffled.end(), random);

  VectorDataset training;
  training.count = train_count;
  training.dim = database.dim;
  training.metric = Metric::kL2;
  training.values.resize(static_cast<std::size_t>(train_count) * database.dim);
  for (std::uint64_t i = 0; i < train_count; ++i) {
    const std::size_t source = static_cast<std::size_t>(shuffled[i]) * database.dim;
    const std::size_t destination = static_cast<std::size_t>(i) * database.dim;
    std::copy_n(database.values.data() + source, database.dim,
                training.values.data() + destination);
  }
  shuffled.clear();
  shuffled.shrink_to_fit();

  // ② 初始化质心: 从训练集均匀间隔抽 nlist 个向量。
  // (比纯随机更均匀,聚类更容易收敛,且确定可复现。)
  VectorDataset centroids;
  centroids.count = config.nlist;
  centroids.dim = database.dim;
  centroids.metric = Metric::kL2;
  centroids.values.resize(static_cast<std::size_t>(config.nlist) * database.dim);
  for (std::uint32_t c = 0; c < config.nlist; ++c) {
    const std::uint64_t sample = static_cast<std::uint64_t>(c) * train_count / config.nlist;
    std::copy_n(training.values.data() + static_cast<std::size_t>(sample) * database.dim,
                database.dim,
                centroids.values.data() + static_cast<std::size_t>(c) * database.dim);
  }

  // ③ Lloyd 迭代: 每轮"GPU 分配 + CPU 求平均更新质心"。
  std::vector<double> sums(static_cast<std::size_t>(config.nlist) * database.dim);
  std::vector<std::uint64_t> counts(config.nlist);
  for (std::uint32_t iteration = 0; iteration < config.kmeans_iters; ++iteration) {
    FlatGpuIndex centroid_gpu(centroids);
    // 分配: 对每个训练向量,用 GPU 找最近的质心(Top-1)。
    const SearchResult assignments = centroid_gpu.search(training, 1, config.batch_size);
    // 更新: 各质心对成员的逐维累加(用 double 减少浮点误差)。
    std::fill(sums.begin(), sums.end(), 0.0);
    std::fill(counts.begin(), counts.end(), 0);
    for (std::uint64_t i = 0; i < train_count; ++i) {
      const auto cluster = static_cast<std::uint32_t>(assignments.ids[i]);
      ++counts[cluster];
      const float* value = training.values.data() + static_cast<std::size_t>(i) * database.dim;
      double* sum = sums.data() + static_cast<std::size_t>(cluster) * database.dim;
      for (std::uint32_t d = 0; d < database.dim; ++d) sum[d] += value[d];
    }
    // 新质心 = 桶内平均值;空桶保留上一轮的值(不更新)。
    for (std::uint32_t c = 0; c < config.nlist; ++c) {
      if (counts[c] == 0) continue;
      float* centroid = centroids.values.data() + static_cast<std::size_t>(c) * database.dim;
      const double* sum = sums.data() + static_cast<std::size_t>(c) * database.dim;
      const double inverse = 1.0 / static_cast<double>(counts[c]);
      for (std::uint32_t d = 0; d < database.dim; ++d) {
        centroid[d] = static_cast<float>(sum[d] * inverse);
      }
    }
  }

  // ④ 用最终质心对全库做一次分配,得到每个向量的桶编号。
  FlatGpuIndex final_centroids(centroids);
  const SearchResult assignments = final_centroids.search(database, 1, config.batch_size);
  counts.assign(config.nlist, 0);
  for (const auto id : assignments.ids) ++counts.at(id);

  // ⑤ 组装索引: 桶偏移(前缀和) + 桶内重排(维度主序) + 原始 ID。
  IvfIndexData index;
  index.count = database.count;
  index.dim = database.dim;
  index.nlist = config.nlist;
  index.metric = Metric::kL2;
  index.centroids = std::move(centroids.values);
  // offsets[c] 是第 c 个桶的起点,offsets[nlist]=count,形成前缀和。
  index.offsets.resize(static_cast<std::size_t>(config.nlist) + 1, 0);
  for (std::uint32_t c = 0; c < config.nlist; ++c) {
    index.offsets[c + 1] = index.offsets[c] + counts[c];
  }
  // cursors 记录每个桶"下一个写入位置"的游标。
  std::vector<std::uint64_t> cursors(index.offsets.begin(), index.offsets.end() - 1);
  index.original_ids.resize(database.count);
  index.vectors_soa.resize(static_cast<std::size_t>(database.count) * database.dim);
  for (std::uint64_t original = 0; original < database.count; ++original) {
    const auto cluster = static_cast<std::uint32_t>(assignments.ids[original]);
    const std::uint64_t position = cursors[cluster]++;
    index.original_ids[position] = static_cast<std::uint32_t>(original);
    // 按桶重排: 第 d 维的第 position 个位置放"原始向量 original 的第 d 维"。
    for (std::uint32_t d = 0; d < database.dim; ++d) {
      index.vectors_soa[static_cast<std::uint64_t>(d) * database.count + position] =
          database.values[static_cast<std::size_t>(original) * database.dim + d];
    }
  }

  if (metrics) {
    metrics->build_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - begin).count();
    metrics->peak_device_bytes = static_cast<std::size_t>(database.count) * database.dim * sizeof(float);
  }
  return index;
}

}  // namespace vs
