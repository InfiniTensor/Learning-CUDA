#include "ivf_flat.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <thread>
#include <unordered_set>
#include <utility>
#include <vector>

#include "file_io.h"

namespace {

constexpr std::array<char, 8> kIvfMagic = {'P', 'P', 'I', 'V',
                                           'F', '0', '0', '1'};
constexpr std::uint64_t kHeaderBytes = 8 + 8 + 4 + 4 + 4 + 4;

template <typename T>
void writeValue(std::ofstream& output, const T& value) {
  output.write(reinterpret_cast<const char*>(&value), sizeof(T));
}

template <typename T>
T readValue(std::ifstream& input, const char* field_name) {
  T value{};
  input.read(reinterpret_cast<char*>(&value), sizeof(T));
  if (!input) {
    throw std::runtime_error(std::string("读取 IVF 字段失败: ") + field_name);
  }
  return value;
}

float halfToFloat(std::uint16_t bits) {
  const bool negative = (bits & 0x8000U) != 0;
  const std::uint32_t exponent = (bits >> 10U) & 0x1fU;
  const std::uint32_t mantissa = bits & 0x03ffU;
  float value = 0.0F;
  if (exponent == 0) {
    value = std::ldexp(static_cast<float>(mantissa), -24);
  } else if (exponent == 31) {
    value = mantissa == 0 ? std::numeric_limits<float>::infinity()
                          : std::numeric_limits<float>::quiet_NaN();
  } else {
    value = std::ldexp(static_cast<float>(1024U + mantissa),
                       static_cast<int>(exponent) - 25);
  }
  return negative ? -value : value;
}

float databaseValue(const VectorDatabase& database, std::uint64_t index) {
  return database.dtype == DataType::kFloat32
             ? database.values[index]
             : halfToFloat(database.half_values[index]);
}

float indexValue(const IvfFlatIndex& index, std::uint64_t index_position) {
  return index.dtype == DataType::kFloat32
             ? index.values[index_position]
             : halfToFloat(index.half_values[index_position]);
}

float queryValue(const QuerySet& queries, std::uint64_t index) {
  return queries.dtype == DataType::kFloat32
             ? queries.values[index]
             : halfToFloat(queries.half_values[index]);
}

bool isBetter(float left_score, std::uint64_t left_id, float right_score,
              std::uint64_t right_id, Metric metric) {
  if (left_score == right_score) {
    return left_id < right_id;
  }
  return metric == Metric::kL2 ? left_score < right_score
                               : left_score > right_score;
}

float centerScore(const float* vector, const float* center, std::uint32_t dim,
                  Metric metric) {
  float dot = 0.0F;
  float vector_norm = 0.0F;
  float center_norm = 0.0F;
  float squared_l2 = 0.0F;
  for (std::uint32_t d = 0; d < dim; ++d) {
    if (metric == Metric::kL2) {
      const float difference = vector[d] - center[d];
      squared_l2 = std::fma(difference, difference, squared_l2);
    } else {
      dot = std::fma(vector[d], center[d], dot);
      if (metric == Metric::kCosine) {
        vector_norm = std::fma(vector[d], vector[d], vector_norm);
        center_norm = std::fma(center[d], center[d], center_norm);
      }
    }
  }
  if (metric == Metric::kL2) {
    return squared_l2;
  }
  if (metric == Metric::kInnerProduct) {
    return dot;
  }
  return vector_norm == 0.0F || center_norm == 0.0F
             ? 0.0F
             : dot / std::sqrt(vector_norm * center_norm);
}

std::uint32_t nearestCenter(const float* vector,
                            const std::vector<float>& centers,
                            std::uint32_t nlist, std::uint32_t dim,
                            Metric metric) {
  std::uint32_t best = 0;
  float best_score = centerScore(vector, centers.data(), dim, metric);
  for (std::uint32_t center_id = 1; center_id < nlist; ++center_id) {
    const float score = centerScore(
        vector, centers.data() + static_cast<std::uint64_t>(center_id) * dim,
        dim, metric);
    if (isBetter(score, center_id, best_score, best, metric)) {
      best = center_id;
      best_score = score;
    }
  }
  return best;
}

void normalize(float* vector, std::uint32_t dim) {
  float squared_norm = 0.0F;
  for (std::uint32_t d = 0; d < dim; ++d) {
    squared_norm = std::fma(vector[d], vector[d], squared_norm);
  }
  if (squared_norm == 0.0F) {
    return;
  }
  const float inverse_norm = 1.0F / std::sqrt(squared_norm);
  for (std::uint32_t d = 0; d < dim; ++d) {
    vector[d] *= inverse_norm;
  }
}

std::uint64_t checkedMultiply(std::uint64_t left, std::uint64_t right,
                              const char* description) {
  if (right != 0 && left > UINT64_MAX / right) {
    throw std::runtime_error(std::string(description) + " 大小溢出");
  }
  return left * right;
}

template <typename Function>
void parallelFor(std::uint64_t count, std::uint32_t requested_threads,
                 Function function) {
  const std::uint32_t threads = static_cast<std::uint32_t>(
      std::min<std::uint64_t>(count, requested_threads));
  if (threads <= 1) {
    function(0, 0, count);
    return;
  }
  std::vector<std::thread> workers;
  workers.reserve(threads - 1);
  for (std::uint32_t worker = 1; worker < threads; ++worker) {
    const std::uint64_t begin = count * worker / threads;
    const std::uint64_t end = count * (worker + 1) / threads;
    workers.emplace_back(function, worker, begin, end);
  }
  function(0, 0, count / threads);
  for (std::thread& worker : workers) {
    worker.join();
  }
}

float candidateScore(const IvfFlatIndex& index, const QuerySet& queries,
                     std::uint64_t query_id, std::uint64_t position) {
  float dot = 0.0F;
  float query_norm = 0.0F;
  float vector_norm = 0.0F;
  float squared_l2 = 0.0F;
  for (std::uint32_t d = 0; d < index.dim; ++d) {
    const float query = queryValue(queries, query_id * index.dim + d);
    const float vector = indexValue(index, position * index.dim + d);
    if (index.metric == Metric::kL2) {
      const float difference = query - vector;
      squared_l2 = std::fma(difference, difference, squared_l2);
    } else {
      dot = std::fma(query, vector, dot);
      if (index.metric == Metric::kCosine) {
        query_norm = std::fma(query, query, query_norm);
        vector_norm = std::fma(vector, vector, vector_norm);
      }
    }
  }
  if (index.metric == Metric::kL2) {
    return squared_l2;
  }
  if (index.metric == Metric::kInnerProduct) {
    return dot;
  }
  return query_norm == 0.0F || vector_norm == 0.0F
             ? 0.0F
             : dot / std::sqrt(query_norm * vector_norm);
}

bool memoryCandidateMatches(const MemoryMetadata& metadata,
                            std::uint64_t vector_id,
                            const SearchParams& params) {
  return metadata.timestamps[vector_id] >= params.filter_min_timestamp &&
         (params.filter_session_id == UINT32_MAX ||
          metadata.session_ids[vector_id] == params.filter_session_id) &&
         (params.filter_source_type == UINT32_MAX ||
          metadata.source_types[vector_id] == params.filter_source_type);
}

float applyMemoryScore(float semantic_score, std::uint64_t vector_id,
                       Metric metric, const MemoryMetadata& metadata,
                       const SearchParams& params) {
  float recency = 0.0F;
  if (params.memory_recency_weight != 0.0F && params.memory_now != 0) {
    const std::uint64_t timestamp = metadata.timestamps[vector_id];
    const double age = timestamp >= params.memory_now
                           ? 0.0
                           : static_cast<double>(params.memory_now - timestamp);
    recency = static_cast<float>(
        std::exp(-age / static_cast<double>(params.memory_time_scale)));
  }
  const float boost =
      params.memory_importance_weight * metadata.importance[vector_id] +
      params.memory_recency_weight * recency;
  const float weighted_semantic =
      params.memory_semantic_weight * semantic_score;
  return metric == Metric::kL2 ? weighted_semantic - boost
                               : weighted_semantic + boost;
}

void validateMemorySearch(const IvfFlatIndex& index, const SearchParams& params,
                          const MemoryMetadata* metadata) {
  if (params.memory_mode != "disabled" && params.memory_mode != "fused" &&
      params.memory_mode != "rerank") {
    throw std::runtime_error("memory_mode 必须是 disabled、fused 或 rerank");
  }
  if (params.memory_mode == "disabled") {
    return;
  }
  if (metadata == nullptr) {
    throw std::runtime_error("启用记忆检索时必须提供元数据");
  }
  validateMemoryMetadata(*metadata, index.num_vectors);
  if (!std::isfinite(params.memory_semantic_weight) ||
      params.memory_semantic_weight <= 0.0F ||
      !std::isfinite(params.memory_importance_weight) ||
      params.memory_importance_weight < 0.0F ||
      !std::isfinite(params.memory_recency_weight) ||
      params.memory_recency_weight < 0.0F ||
      !std::isfinite(params.memory_time_scale) ||
      params.memory_time_scale <= 0.0F || params.memory_rerank_factor == 0) {
    throw std::runtime_error("记忆评分参数不合法");
  }
}

}  // namespace

IvfFlatIndex buildIvfFlatIndex(const VectorDatabase& database,
                               std::uint32_t nlist, std::uint32_t iterations,
                               std::uint64_t max_training_vectors,
                               std::uint32_t num_threads) {
  if (database.num_vectors == 0 || database.dim == 0) {
    throw std::runtime_error("IVF 建库输入不能为空");
  }
  if (nlist == 0 || nlist > database.num_vectors) {
    throw std::runtime_error("nlist 必须位于 [1, num_vectors]");
  }
  if (iterations == 0 || max_training_vectors == 0 || num_threads == 0) {
    throw std::runtime_error("IVF 训练迭代数、采样数和线程数必须大于 0");
  }
  const std::uint64_t value_count =
      checkedMultiply(database.num_vectors, database.dim, "向量库");
  if ((database.dtype == DataType::kFloat32 &&
       database.values.size() != value_count) ||
      (database.dtype == DataType::kFloat16 &&
       database.half_values.size() != value_count)) {
    throw std::runtime_error("IVF 建库数据长度与元数据不一致");
  }

  const std::uint64_t sample_count = std::min<std::uint64_t>(
      database.num_vectors,
      std::max<std::uint64_t>(nlist, max_training_vectors));
  std::vector<float> samples(sample_count * database.dim);
  for (std::uint64_t sample_id = 0; sample_id < sample_count; ++sample_id) {
    const std::uint64_t vector_id =
        sample_id * database.num_vectors / sample_count;
    for (std::uint32_t d = 0; d < database.dim; ++d) {
      samples[sample_id * database.dim + d] =
          databaseValue(database, vector_id * database.dim + d);
    }
  }

  std::vector<float> centers(static_cast<std::uint64_t>(nlist) * database.dim);
  for (std::uint32_t center_id = 0; center_id < nlist; ++center_id) {
    const std::uint64_t sample_id =
        static_cast<std::uint64_t>(center_id) * sample_count / nlist;
    std::copy_n(
        samples.data() + sample_id * database.dim, database.dim,
        centers.data() + static_cast<std::uint64_t>(center_id) * database.dim);
    if (database.metric != Metric::kL2) {
      normalize(
          centers.data() + static_cast<std::uint64_t>(center_id) * database.dim,
          database.dim);
    }
  }

  std::vector<float> sums(centers.size());
  std::vector<std::uint64_t> counts(nlist);
  const std::uint32_t training_threads = static_cast<std::uint32_t>(
      std::min<std::uint64_t>(num_threads, sample_count));
  std::vector<float> partial_sums(
      static_cast<std::size_t>(training_threads) * centers.size());
  std::vector<std::uint64_t> partial_counts(
      static_cast<std::size_t>(training_threads) * nlist);
  for (std::uint32_t iteration = 0; iteration < iterations; ++iteration) {
    std::fill(sums.begin(), sums.end(), 0.0F);
    std::fill(counts.begin(), counts.end(), 0);
    std::fill(partial_sums.begin(), partial_sums.end(), 0.0F);
    std::fill(partial_counts.begin(), partial_counts.end(), 0);
    parallelFor(sample_count, training_threads,
                [&](std::uint32_t worker, std::uint64_t begin,
                    std::uint64_t end) {
      float* worker_sums =
          partial_sums.data() +
          static_cast<std::size_t>(worker) * centers.size();
      std::uint64_t* worker_counts =
          partial_counts.data() + static_cast<std::size_t>(worker) * nlist;
      for (std::uint64_t sample_id = begin; sample_id < end; ++sample_id) {
        const std::uint32_t center_id =
            nearestCenter(samples.data() + sample_id * database.dim, centers,
                          nlist, database.dim, database.metric);
        ++worker_counts[center_id];
        for (std::uint32_t d = 0; d < database.dim; ++d) {
          worker_sums[static_cast<std::uint64_t>(center_id) * database.dim +
                      d] += samples[sample_id * database.dim + d];
        }
      }
    });
    for (std::uint32_t worker = 0; worker < training_threads; ++worker) {
      const float* worker_sums =
          partial_sums.data() +
          static_cast<std::size_t>(worker) * centers.size();
      const std::uint64_t* worker_counts =
          partial_counts.data() + static_cast<std::size_t>(worker) * nlist;
      for (std::uint32_t center_id = 0; center_id < nlist; ++center_id) {
        counts[center_id] += worker_counts[center_id];
      }
      for (std::size_t value = 0; value < centers.size(); ++value) {
        sums[value] += worker_sums[value];
      }
    }
    for (std::uint32_t center_id = 0; center_id < nlist; ++center_id) {
      float* center =
          centers.data() + static_cast<std::uint64_t>(center_id) * database.dim;
      if (counts[center_id] == 0) {
        const std::uint64_t replacement =
            (static_cast<std::uint64_t>(center_id) + iteration) % sample_count;
        std::copy_n(samples.data() + replacement * database.dim, database.dim,
                    center);
      } else {
        const float inverse_count =
            1.0F / static_cast<float>(counts[center_id]);
        for (std::uint32_t d = 0; d < database.dim; ++d) {
          center[d] =
              sums[static_cast<std::uint64_t>(center_id) * database.dim + d] *
              inverse_count;
        }
      }
      // 对 Cosine 和 Inner Product 使用 spherical k-means。若不归一化，
      // 高范数中心会在点积比较中吸收过多向量，nlist 较大时桶会严重倾斜。
      if (database.metric != Metric::kL2) {
        normalize(center, database.dim);
      }
    }
  }

  std::vector<std::uint32_t> database_assignments(database.num_vectors);
  counts.assign(nlist, 0);
  parallelFor(database.num_vectors, num_threads,
              [&](std::uint32_t, std::uint64_t begin, std::uint64_t end) {
    std::vector<float> vector(database.dim);
    for (std::uint64_t vector_id = begin; vector_id < end; ++vector_id) {
      for (std::uint32_t d = 0; d < database.dim; ++d) {
        vector[d] = databaseValue(database, vector_id * database.dim + d);
      }
      database_assignments[vector_id] = nearestCenter(
          vector.data(), centers, nlist, database.dim, database.metric);
    }
  });
  for (const std::uint32_t center_id : database_assignments) {
    ++counts[center_id];
  }

  IvfFlatIndex index;
  index.num_vectors = database.num_vectors;
  index.dim = database.dim;
  index.nlist = nlist;
  index.dtype = database.dtype;
  index.metric = database.metric;
  index.centers = std::move(centers);
  index.offsets.resize(static_cast<std::size_t>(nlist) + 1, 0);
  for (std::uint32_t center_id = 0; center_id < nlist; ++center_id) {
    index.offsets[center_id + 1] = index.offsets[center_id] + counts[center_id];
  }
  std::vector<std::uint64_t> positions = index.offsets;
  index.ids.resize(database.num_vectors);
  if (database.dtype == DataType::kFloat32) {
    index.values.resize(value_count);
  } else {
    index.half_values.resize(value_count);
  }
  for (std::uint64_t vector_id = 0; vector_id < database.num_vectors;
       ++vector_id) {
    const std::uint64_t position = positions[database_assignments[vector_id]]++;
    index.ids[position] = vector_id;
    const std::uint64_t source = vector_id * database.dim;
    const std::uint64_t destination = position * database.dim;
    if (database.dtype == DataType::kFloat32) {
      std::copy_n(database.values.data() + source, database.dim,
                  index.values.data() + destination);
    } else {
      std::copy_n(database.half_values.data() + source, database.dim,
                  index.half_values.data() + destination);
    }
  }
  validateIvfFlatIndex(index);
  return index;
}

void validateIvfFlatIndex(const IvfFlatIndex& index) {
  if (index.num_vectors == 0 || index.dim == 0 || index.nlist == 0 ||
      index.nlist > index.num_vectors) {
    throw std::runtime_error("IVF 索引元数据不合法");
  }
  const std::uint64_t center_count =
      checkedMultiply(index.nlist, index.dim, "IVF centers");
  const std::uint64_t value_count =
      checkedMultiply(index.num_vectors, index.dim, "IVF vectors");
  if (index.centers.size() != center_count ||
      index.offsets.size() != static_cast<std::size_t>(index.nlist) + 1 ||
      index.ids.size() != index.num_vectors || index.offsets.front() != 0 ||
      index.offsets.back() != index.num_vectors) {
    throw std::runtime_error("IVF 索引数组长度与元数据不一致");
  }
  if ((index.dtype == DataType::kFloat32 &&
       (index.values.size() != value_count || !index.half_values.empty())) ||
      (index.dtype == DataType::kFloat16 &&
       (index.half_values.size() != value_count || !index.values.empty()))) {
    throw std::runtime_error("IVF 索引向量 dtype 或长度不正确");
  }
  if (index.dtype != DataType::kFloat32 && index.dtype != DataType::kFloat16) {
    throw std::runtime_error("IVF 索引 dtype 不支持");
  }
  if (index.metric != Metric::kL2 && index.metric != Metric::kInnerProduct &&
      index.metric != Metric::kCosine) {
    throw std::runtime_error("IVF 索引 metric 不支持");
  }
  for (std::uint32_t center_id = 0; center_id < index.nlist; ++center_id) {
    if (index.offsets[center_id] > index.offsets[center_id + 1]) {
      throw std::runtime_error("IVF 桶 offsets 必须单调递增");
    }
  }
  std::vector<bool> seen(index.num_vectors, false);
  for (const std::uint64_t id : index.ids) {
    if (id >= index.num_vectors || seen[id]) {
      throw std::runtime_error("IVF 原始向量 ID 越界或重复");
    }
    seen[id] = true;
  }
}

void writeIvfFlatIndex(const std::string& path, const IvfFlatIndex& index) {
  validateIvfFlatIndex(index);
  std::ofstream output(path, std::ios::binary);
  if (!output) {
    throw std::runtime_error("无法写入 IVF 索引: " + path);
  }
  output.write(kIvfMagic.data(), kIvfMagic.size());
  writeValue(output, index.num_vectors);
  writeValue(output, index.dim);
  writeValue(output, static_cast<std::uint32_t>(index.dtype));
  writeValue(output, static_cast<std::uint32_t>(index.metric));
  writeValue(output, index.nlist);
  output.write(
      reinterpret_cast<const char*>(index.centers.data()),
      static_cast<std::streamsize>(index.centers.size() * sizeof(float)));
  output.write(reinterpret_cast<const char*>(index.offsets.data()),
               static_cast<std::streamsize>(index.offsets.size() *
                                            sizeof(std::uint64_t)));
  output.write(
      reinterpret_cast<const char*>(index.ids.data()),
      static_cast<std::streamsize>(index.ids.size() * sizeof(std::uint64_t)));
  const void* values = index.dtype == DataType::kFloat32
                           ? static_cast<const void*>(index.values.data())
                           : static_cast<const void*>(index.half_values.data());
  const std::uint64_t element_bytes =
      index.dtype == DataType::kFloat32 ? sizeof(float) : sizeof(std::uint16_t);
  output.write(reinterpret_cast<const char*>(values),
               static_cast<std::streamsize>(index.num_vectors * index.dim *
                                            element_bytes));
  if (!output) {
    throw std::runtime_error("写入 IVF 索引失败: " + path);
  }
}

IvfFlatIndex readIvfFlatIndex(const std::string& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("无法打开 IVF 索引: " + path);
  }
  std::array<char, 8> magic{};
  input.read(magic.data(), magic.size());
  if (magic != kIvfMagic) {
    throw std::runtime_error("IVF 索引 magic/version 不正确: " + path);
  }
  IvfFlatIndex index;
  index.num_vectors = readValue<std::uint64_t>(input, "num_vectors");
  index.dim = readValue<std::uint32_t>(input, "dim");
  index.dtype = static_cast<DataType>(readValue<std::uint32_t>(input, "dtype"));
  index.metric = static_cast<Metric>(readValue<std::uint32_t>(input, "metric"));
  index.nlist = readValue<std::uint32_t>(input, "nlist");
  if (index.num_vectors == 0 || index.dim == 0 || index.nlist == 0 ||
      index.nlist > index.num_vectors) {
    throw std::runtime_error("IVF 索引元数据不合法: " + path);
  }
  const std::uint64_t center_count =
      checkedMultiply(index.nlist, index.dim, "IVF centers");
  const std::uint64_t value_count =
      checkedMultiply(index.num_vectors, index.dim, "IVF vectors");
  const std::uint64_t element_bytes =
      index.dtype == DataType::kFloat32   ? sizeof(float)
      : index.dtype == DataType::kFloat16 ? sizeof(std::uint16_t)
                                          : 0;
  if (element_bytes == 0) {
    throw std::runtime_error("IVF 索引 dtype 不支持: " + path);
  }
  const std::uint64_t expected_size =
      kHeaderBytes + center_count * sizeof(float) +
      (static_cast<std::uint64_t>(index.nlist) + 1) * sizeof(std::uint64_t) +
      index.num_vectors * sizeof(std::uint64_t) + value_count * element_bytes;
  if (std::filesystem::file_size(path) != expected_size) {
    throw std::runtime_error("IVF 索引文件大小不正确: " + path);
  }
  index.centers.resize(center_count);
  index.offsets.resize(static_cast<std::size_t>(index.nlist) + 1);
  index.ids.resize(index.num_vectors);
  input.read(reinterpret_cast<char*>(index.centers.data()),
             static_cast<std::streamsize>(center_count * sizeof(float)));
  input.read(reinterpret_cast<char*>(index.offsets.data()),
             static_cast<std::streamsize>(index.offsets.size() *
                                          sizeof(std::uint64_t)));
  input.read(
      reinterpret_cast<char*>(index.ids.data()),
      static_cast<std::streamsize>(index.ids.size() * sizeof(std::uint64_t)));
  if (index.dtype == DataType::kFloat32) {
    index.values.resize(value_count);
    input.read(reinterpret_cast<char*>(index.values.data()),
               static_cast<std::streamsize>(value_count * sizeof(float)));
  } else {
    index.half_values.resize(value_count);
    input.read(
        reinterpret_cast<char*>(index.half_values.data()),
        static_cast<std::streamsize>(value_count * sizeof(std::uint16_t)));
  }
  if (!input) {
    throw std::runtime_error("读取 IVF 索引数据失败: " + path);
  }
  validateIvfFlatIndex(index);
  return index;
}

SearchResults cpuIvfFlatSearch(const IvfFlatIndex& index,
                               const QuerySet& queries,
                               const SearchParams& params,
                               const MemoryMetadata* metadata) {
  if (params.nprobe_policy != "fixed") {
    throw std::runtime_error("自适应 nprobe 当前只支持 GPU backend");
  }
  const std::vector<std::vector<std::uint64_t>> candidate_positions =
      selectIvfCandidatePositions(index, queries, params.nprobe);
  if (params.top_k == 0 || params.top_k > index.num_vectors) {
    throw std::runtime_error("IVF 的 top_k 不合法");
  }
  validateMemorySearch(index, params, metadata);

  SearchResults results(queries.num_queries);
  for (std::uint64_t query_id = 0; query_id < queries.num_queries; ++query_id) {
    std::vector<Neighbor> candidates;
    candidates.reserve(candidate_positions[query_id].size());
    for (const std::uint64_t position : candidate_positions[query_id]) {
      const std::uint64_t vector_id = index.ids[position];
      if (params.memory_mode == "fused" &&
          !memoryCandidateMatches(*metadata, vector_id, params)) {
        continue;
      }
      float score = candidateScore(index, queries, query_id, position);
      if (params.memory_mode == "fused") {
        score =
            applyMemoryScore(score, vector_id, index.metric, *metadata, params);
      }
      candidates.push_back({vector_id, score});
    }
    if (params.memory_mode == "disabled" && candidates.size() < params.top_k) {
      throw std::runtime_error("IVF 探测桶的候选数少于 top_k，请增大 nprobe");
    }
    const auto better = [&](const Neighbor& left, const Neighbor& right) {
      return isBetter(left.score, left.id, right.score, right.id, index.metric);
    };
    std::size_t selected_k =
        std::min<std::size_t>(params.top_k, candidates.size());
    if (params.memory_mode == "rerank") {
      const std::uint64_t requested = static_cast<std::uint64_t>(params.top_k) *
                                      params.memory_rerank_factor;
      const std::size_t rerank_k = static_cast<std::size_t>(
          std::min<std::uint64_t>({requested, 100, candidates.size()}));
      if (rerank_k < candidates.size()) {
        std::nth_element(candidates.begin(), candidates.begin() + rerank_k,
                         candidates.end(), better);
        candidates.resize(rerank_k);
      }
      candidates.erase(std::remove_if(candidates.begin(), candidates.end(),
                                      [&](const Neighbor& candidate) {
                                        return !memoryCandidateMatches(
                                            *metadata, candidate.id, params);
                                      }),
                       candidates.end());
      for (Neighbor& candidate : candidates) {
        candidate.score = applyMemoryScore(candidate.score, candidate.id,
                                           index.metric, *metadata, params);
      }
      selected_k = std::min<std::size_t>(params.top_k, candidates.size());
    }
    if (selected_k < candidates.size()) {
      std::nth_element(candidates.begin(), candidates.begin() + selected_k,
                       candidates.end(), better);
      candidates.resize(selected_k);
    }
    std::sort(candidates.begin(), candidates.end(), better);
    results[query_id] = std::move(candidates);
  }
  return results;
}

std::vector<std::vector<std::uint64_t>> selectIvfCandidatePositions(
    const IvfFlatIndex& index, const QuerySet& queries, std::uint32_t nprobe) {
  validateIvfFlatIndex(index);
  if (queries.num_queries == 0 || queries.dim != index.dim ||
      queries.dtype != index.dtype) {
    throw std::runtime_error("查询集与 IVF 索引的维度或 dtype 不匹配");
  }
  const std::uint64_t query_value_count =
      checkedMultiply(queries.num_queries, queries.dim, "IVF queries");
  if ((queries.dtype == DataType::kFloat32 &&
       queries.values.size() != query_value_count) ||
      (queries.dtype == DataType::kFloat16 &&
       queries.half_values.size() != query_value_count)) {
    throw std::runtime_error("IVF 查询数据长度与元数据不一致");
  }
  if (nprobe == 0 || nprobe > index.nlist) {
    throw std::runtime_error("IVF 的 nprobe 不合法");
  }

  std::vector<std::vector<std::uint64_t>> selected(queries.num_queries);
  std::vector<float> query(index.dim);
  for (std::uint64_t query_id = 0; query_id < queries.num_queries; ++query_id) {
    for (std::uint32_t d = 0; d < index.dim; ++d) {
      query[d] = queryValue(queries, query_id * index.dim + d);
    }
    std::vector<Neighbor> center_ranking(index.nlist);
    for (std::uint32_t center_id = 0; center_id < index.nlist; ++center_id) {
      center_ranking[center_id] = {
          center_id,
          centerScore(query.data(),
                      index.centers.data() +
                          static_cast<std::uint64_t>(center_id) * index.dim,
                      index.dim, index.metric)};
    }
    const auto better = [&](const Neighbor& left, const Neighbor& right) {
      return isBetter(left.score, left.id, right.score, right.id, index.metric);
    };
    std::partial_sort(center_ranking.begin(), center_ranking.begin() + nprobe,
                      center_ranking.end(), better);

    std::vector<std::uint64_t>& positions = selected[query_id];
    for (std::uint32_t probe = 0; probe < nprobe; ++probe) {
      const std::uint64_t center_id = center_ranking[probe].id;
      for (std::uint64_t position = index.offsets[center_id];
           position < index.offsets[center_id + 1]; ++position) {
        positions.push_back(position);
      }
    }
  }
  return selected;
}

double recallAtK(const SearchResults& exact, const SearchResults& approximate) {
  if (exact.size() != approximate.size()) {
    throw std::runtime_error("recall@K 的 query 数量不同");
  }
  std::uint64_t matches = 0;
  std::uint64_t total = 0;
  for (std::size_t query_id = 0; query_id < exact.size(); ++query_id) {
    if (exact[query_id].size() != approximate[query_id].size()) {
      throw std::runtime_error("recall@K 的 K 不同");
    }
    std::unordered_set<std::uint64_t> exact_ids;
    for (const Neighbor& neighbor : exact[query_id]) {
      exact_ids.insert(neighbor.id);
    }
    for (const Neighbor& neighbor : approximate[query_id]) {
      matches += exact_ids.count(neighbor.id);
    }
    total += exact[query_id].size();
  }
  return total == 0 ? 1.0 : static_cast<double>(matches) / total;
}
