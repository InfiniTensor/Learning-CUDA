#include "file_io.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace {

constexpr std::array<char, 8> kDatabaseMagic = {'P', 'P', 'V', 'E',
                                                'C', '0', '0', '1'};
constexpr std::array<char, 8> kQueryMagic = {'P', 'P', 'Q', 'R',
                                             'Y', '0', '0', '1'};
constexpr std::array<char, 8> kMemoryMetadataMagic = {'P', 'P', 'M', 'E',
                                                      'T', 'A', '0', '1'};

template <typename T>
void writeValue(std::ofstream& output, const T& value) {
  output.write(reinterpret_cast<const char*>(&value), sizeof(T));
}

template <typename T>
T readValue(std::ifstream& input, const char* field_name) {
  T value{};
  input.read(reinterpret_cast<char*>(&value), sizeof(T));
  if (!input) {
    throw std::runtime_error(std::string("读取字段失败: ") + field_name);
  }
  return value;
}

std::string trim(std::string text) {
  const auto not_space = [](unsigned char c) { return !std::isspace(c); };
  text.erase(text.begin(), std::find_if(text.begin(), text.end(), not_space));
  text.erase(std::find_if(text.rbegin(), text.rend(), not_space).base(),
             text.end());
  if (text.size() >= 2 && text.front() == '"' && text.back() == '"') {
    text = text.substr(1, text.size() - 2);
  }
  return text;
}

std::uint32_t parsePositiveInt(const std::string& value,
                               const std::string& key) {
  std::size_t parsed_chars = 0;
  const unsigned long parsed = std::stoul(value, &parsed_chars);
  if (parsed_chars != value.size() || parsed == 0 || parsed > UINT32_MAX) {
    throw std::runtime_error("参数 " + key + " 必须是正整数");
  }
  return static_cast<std::uint32_t>(parsed);
}

float parsePositiveFloat(const std::string& value, const std::string& key) {
  std::size_t parsed_chars = 0;
  const float parsed = std::stof(value, &parsed_chars);
  if (parsed_chars != value.size() || !std::isfinite(parsed) ||
      parsed <= 0.0F) {
    throw std::runtime_error("参数 " + key + " 必须是有限正数");
  }
  return parsed;
}

float parseNonnegativeFloat(const std::string& value, const std::string& key) {
  std::size_t parsed_chars = 0;
  const float parsed = std::stof(value, &parsed_chars);
  if (parsed_chars != value.size() || !std::isfinite(parsed) || parsed < 0.0F) {
    throw std::runtime_error("参数 " + key + " 必须是有限非负数");
  }
  return parsed;
}

std::uint64_t parseUint64(const std::string& value, const std::string& key) {
  if (value.empty() || value.front() == '-') {
    throw std::runtime_error("参数 " + key + " 必须是非负整数");
  }
  std::size_t parsed_chars = 0;
  const unsigned long long parsed = std::stoull(value, &parsed_chars);
  if (parsed_chars != value.size()) {
    throw std::runtime_error("参数 " + key + " 必须是非负整数");
  }
  return static_cast<std::uint64_t>(parsed);
}

std::uint32_t parseOptionalUint32(const std::string& value,
                                  const std::string& key) {
  if (value == "any") {
    return UINT32_MAX;
  }
  const std::uint64_t parsed = parseUint64(value, key);
  if (parsed >= UINT32_MAX) {
    throw std::runtime_error("参数 " + key + " 必须小于 UINT32_MAX");
  }
  return static_cast<std::uint32_t>(parsed);
}

std::uint64_t elementSize(DataType dtype) {
  if (dtype == DataType::kFloat32) {
    return sizeof(float);
  }
  if (dtype == DataType::kFloat16) {
    return sizeof(std::uint16_t);
  }
  throw std::runtime_error("不支持的 dtype 编号: " +
                           std::to_string(static_cast<std::uint32_t>(dtype)));
}

std::uint64_t checkedValueCount(std::uint64_t rows, std::uint32_t dim,
                                const std::string& path) {
  if (dim != 0 && rows > UINT64_MAX / dim) {
    throw std::runtime_error("向量数量与维度乘积溢出: " + path);
  }
  return rows * dim;
}

void checkFileSize(const std::string& path, std::uint64_t header_bytes,
                   std::uint64_t value_count, DataType dtype) {
  const std::uint64_t element_bytes = elementSize(dtype);
  if (value_count > (UINT64_MAX - header_bytes) / element_bytes) {
    throw std::runtime_error("文件大小计算溢出: " + path);
  }
  const std::uint64_t expected = header_bytes + value_count * element_bytes;
  const std::uint64_t actual = std::filesystem::file_size(path);
  if (actual != expected) {
    throw std::runtime_error("文件大小不正确: " + path + "，期望 " +
                             std::to_string(expected) + " 字节，实际 " +
                             std::to_string(actual) + " 字节");
  }
}

}  // namespace

const char* metricName(Metric metric) {
  switch (metric) {
    case Metric::kL2:
      return "l2";
    case Metric::kInnerProduct:
      return "inner_product";
    case Metric::kCosine:
      return "cosine";
  }
  throw std::runtime_error("未知的距离类型");
}

const char* dataTypeName(DataType dtype) {
  switch (dtype) {
    case DataType::kFloat32:
      return "fp32";
    case DataType::kFloat16:
      return "fp16";
  }
  throw std::runtime_error("未知的数据类型");
}

Metric parseMetric(const std::string& text) {
  if (text == "l2") {
    return Metric::kL2;
  }
  if (text == "inner_product") {
    return Metric::kInnerProduct;
  }
  if (text == "cosine") {
    return Metric::kCosine;
  }
  throw std::runtime_error("不支持的 metric: " + text);
}

VectorDatabase readVectorDatabase(const std::string& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("无法打开向量库: " + path);
  }

  std::array<char, 8> magic{};
  input.read(magic.data(), magic.size());
  if (magic != kDatabaseMagic) {
    throw std::runtime_error("向量库 magic/version 不正确: " + path);
  }

  VectorDatabase database;
  database.num_vectors = readValue<std::uint64_t>(input, "num_vectors");
  database.dim = readValue<std::uint32_t>(input, "dim");
  database.dtype =
      static_cast<DataType>(readValue<std::uint32_t>(input, "dtype"));
  database.metric =
      static_cast<Metric>(readValue<std::uint32_t>(input, "metric"));

  (void)dataTypeName(database.dtype);
  (void)metricName(database.metric);
  if (database.num_vectors == 0 || database.dim == 0) {
    throw std::runtime_error("向量库的数量和维度必须大于 0");
  }

  const std::uint64_t count =
      checkedValueCount(database.num_vectors, database.dim, path);
  checkFileSize(path, 8 + 8 + 4 + 4 + 4, count, database.dtype);
  if (database.dtype == DataType::kFloat32) {
    database.values.resize(count);
    input.read(reinterpret_cast<char*>(database.values.data()),
               static_cast<std::streamsize>(count * sizeof(float)));
  } else {
    database.half_values.resize(count);
    input.read(reinterpret_cast<char*>(database.half_values.data()),
               static_cast<std::streamsize>(count * sizeof(std::uint16_t)));
  }
  if (!input) {
    throw std::runtime_error("读取向量库数据失败: " + path);
  }
  return database;
}

QuerySet readQuerySet(const std::string& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("无法打开查询文件: " + path);
  }

  std::array<char, 8> magic{};
  input.read(magic.data(), magic.size());
  if (magic != kQueryMagic) {
    throw std::runtime_error("查询文件 magic/version 不正确: " + path);
  }

  QuerySet queries;
  queries.num_queries = readValue<std::uint64_t>(input, "num_queries");
  queries.dim = readValue<std::uint32_t>(input, "dim");
  queries.dtype =
      static_cast<DataType>(readValue<std::uint32_t>(input, "dtype"));
  (void)dataTypeName(queries.dtype);
  if (queries.num_queries == 0 || queries.dim == 0) {
    throw std::runtime_error("查询数量和维度必须大于 0");
  }

  const std::uint64_t count =
      checkedValueCount(queries.num_queries, queries.dim, path);
  checkFileSize(path, 8 + 8 + 4 + 4, count, queries.dtype);
  if (queries.dtype == DataType::kFloat32) {
    queries.values.resize(count);
    input.read(reinterpret_cast<char*>(queries.values.data()),
               static_cast<std::streamsize>(count * sizeof(float)));
  } else {
    queries.half_values.resize(count);
    input.read(reinterpret_cast<char*>(queries.half_values.data()),
               static_cast<std::streamsize>(count * sizeof(std::uint16_t)));
  }
  if (!input) {
    throw std::runtime_error("读取查询数据失败: " + path);
  }
  return queries;
}

void validateMemoryMetadata(const MemoryMetadata& metadata,
                            std::uint64_t expected_vectors) {
  if (metadata.num_vectors == 0 ||
      (expected_vectors != 0 && metadata.num_vectors != expected_vectors) ||
      metadata.timestamps.size() != metadata.num_vectors ||
      metadata.importance.size() != metadata.num_vectors ||
      metadata.session_ids.size() != metadata.num_vectors ||
      metadata.source_types.size() != metadata.num_vectors) {
    throw std::runtime_error("记忆元数据列长度或向量数量不一致");
  }
  for (const float value : metadata.importance) {
    if (!std::isfinite(value) || value < 0.0F || value > 1.0F) {
      throw std::runtime_error("记忆 importance 必须位于 [0, 1]");
    }
  }
}

void writeMemoryMetadata(const std::string& path,
                         const MemoryMetadata& metadata) {
  validateMemoryMetadata(metadata);
  std::ofstream output(path, std::ios::binary);
  if (!output) {
    throw std::runtime_error("无法写入记忆元数据: " + path);
  }
  output.write(kMemoryMetadataMagic.data(), kMemoryMetadataMagic.size());
  writeValue(output, metadata.num_vectors);
  output.write(reinterpret_cast<const char*>(metadata.timestamps.data()),
               static_cast<std::streamsize>(metadata.num_vectors *
                                            sizeof(std::uint64_t)));
  output.write(
      reinterpret_cast<const char*>(metadata.importance.data()),
      static_cast<std::streamsize>(metadata.num_vectors * sizeof(float)));
  output.write(reinterpret_cast<const char*>(metadata.session_ids.data()),
               static_cast<std::streamsize>(metadata.num_vectors *
                                            sizeof(std::uint32_t)));
  output.write(reinterpret_cast<const char*>(metadata.source_types.data()),
               static_cast<std::streamsize>(metadata.num_vectors *
                                            sizeof(std::uint32_t)));
  if (!output) {
    throw std::runtime_error("写入记忆元数据失败: " + path);
  }
}

MemoryMetadata readMemoryMetadata(const std::string& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("无法打开记忆元数据: " + path);
  }
  std::array<char, 8> magic{};
  input.read(magic.data(), magic.size());
  if (magic != kMemoryMetadataMagic) {
    throw std::runtime_error("记忆元数据 magic/version 不正确: " + path);
  }
  MemoryMetadata metadata;
  metadata.num_vectors = readValue<std::uint64_t>(input, "num_vectors");
  constexpr std::uint64_t kBytesPerVector =
      sizeof(std::uint64_t) + sizeof(float) + 2 * sizeof(std::uint32_t);
  if (metadata.num_vectors == 0 ||
      metadata.num_vectors > (UINT64_MAX - 16) / kBytesPerVector ||
      std::filesystem::file_size(path) !=
          16 + metadata.num_vectors * kBytesPerVector) {
    throw std::runtime_error("记忆元数据文件大小不正确: " + path);
  }
  metadata.timestamps.resize(metadata.num_vectors);
  metadata.importance.resize(metadata.num_vectors);
  metadata.session_ids.resize(metadata.num_vectors);
  metadata.source_types.resize(metadata.num_vectors);
  input.read(reinterpret_cast<char*>(metadata.timestamps.data()),
             static_cast<std::streamsize>(metadata.num_vectors *
                                          sizeof(std::uint64_t)));
  input.read(
      reinterpret_cast<char*>(metadata.importance.data()),
      static_cast<std::streamsize>(metadata.num_vectors * sizeof(float)));
  input.read(reinterpret_cast<char*>(metadata.session_ids.data()),
             static_cast<std::streamsize>(metadata.num_vectors *
                                          sizeof(std::uint32_t)));
  input.read(reinterpret_cast<char*>(metadata.source_types.data()),
             static_cast<std::streamsize>(metadata.num_vectors *
                                          sizeof(std::uint32_t)));
  if (!input) {
    throw std::runtime_error("读取记忆元数据失败: " + path);
  }
  validateMemoryMetadata(metadata);
  return metadata;
}

SearchParams readSearchParams(const std::string& path) {
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("无法打开参数文件: " + path);
  }

  SearchParams params;
  std::string line;
  while (std::getline(input, line)) {
    const std::size_t comment = line.find('#');
    if (comment != std::string::npos) {
      line.erase(comment);
    }
    line = trim(line);
    if (line.empty()) {
      continue;
    }

    const std::size_t equal = line.find('=');
    if (equal == std::string::npos) {
      throw std::runtime_error("参数行缺少 '=': " + line);
    }
    const std::string key = trim(line.substr(0, equal));
    const std::string value = trim(line.substr(equal + 1));

    if (key == "top_k") {
      params.top_k = parsePositiveInt(value, key);
    } else if (key == "search_mode") {
      params.search_mode = value;
    } else if (key == "batch_size") {
      params.batch_size = parsePositiveInt(value, key);
    } else if (key == "distance_mode") {
      params.distance_mode = value;
    } else if (key == "topk_mode") {
      params.topk_mode = value;
    } else if (key == "exact_query_tile") {
      params.exact_query_tile = parsePositiveInt(value, key);
    } else if (key == "exact_fused_layout") {
      params.exact_fused_layout = value;
    } else if (key == "exact_fused_blocks") {
      params.exact_fused_blocks = parsePositiveInt(value, key);
    } else if (key == "nlist") {
      params.nlist = parsePositiveInt(value, key);
    } else if (key == "nprobe") {
      params.nprobe = parsePositiveInt(value, key);
    } else if (key == "nprobe_policy") {
      params.nprobe_policy = value;
    } else if (key == "adaptive_execution") {
      params.adaptive_execution = value;
    } else if (key == "adaptive_nprobe_min") {
      params.adaptive_nprobe_min = parsePositiveInt(value, key);
    } else if (key == "adaptive_nprobe_step") {
      params.adaptive_nprobe_step = parsePositiveInt(value, key);
    } else if (key == "adaptive_target_mass") {
      params.adaptive_target_mass = parsePositiveFloat(value, key);
    } else if (key == "adaptive_temperature") {
      params.adaptive_temperature = parsePositiveFloat(value, key);
    } else if (key == "memory_mode") {
      params.memory_mode = value;
    } else if (key == "memory_semantic_weight") {
      params.memory_semantic_weight = parsePositiveFloat(value, key);
    } else if (key == "memory_importance_weight") {
      params.memory_importance_weight = parseNonnegativeFloat(value, key);
    } else if (key == "memory_recency_weight") {
      params.memory_recency_weight = parseNonnegativeFloat(value, key);
    } else if (key == "memory_time_scale") {
      params.memory_time_scale = parsePositiveFloat(value, key);
    } else if (key == "memory_now") {
      params.memory_now = parseUint64(value, key);
    } else if (key == "filter_min_timestamp") {
      params.filter_min_timestamp = parseUint64(value, key);
    } else if (key == "filter_session_id") {
      params.filter_session_id = parseOptionalUint32(value, key);
    } else if (key == "filter_source_type") {
      params.filter_source_type = parseOptionalUint32(value, key);
    } else if (key == "memory_rerank_factor") {
      params.memory_rerank_factor = parsePositiveInt(value, key);
    } else if (key == "pq_m") {
      params.pq_m = parsePositiveInt(value, key);
    } else {
      throw std::runtime_error("未知参数: " + key);
    }
  }
  return params;
}

void writeSearchResults(const std::string& path, const SearchResults& results) {
  std::ofstream output(path);
  if (!output) {
    throw std::runtime_error("无法写入结果文件: " + path);
  }
  output.precision(9);
  for (std::size_t query_id = 0; query_id < results.size(); ++query_id) {
    for (const Neighbor& neighbor : results[query_id]) {
      output << query_id << ' ' << neighbor.id << ' ' << neighbor.score << '\n';
    }
  }
}

void validateInputs(const VectorDatabase& database, const QuerySet& queries,
                    const SearchParams& params) {
  if (database.num_vectors == 0 || database.dim == 0) {
    throw std::runtime_error("向量库的数量和维度必须大于 0");
  }
  if (queries.num_queries == 0 || queries.dim == 0) {
    throw std::runtime_error("查询数量和维度必须大于 0");
  }
  if (database.dim != queries.dim) {
    throw std::runtime_error("向量库维度与查询维度不一致");
  }
  if (database.dtype != queries.dtype) {
    throw std::runtime_error("向量库与查询的数据类型不一致");
  }
  const std::uint64_t database_count =
      checkedValueCount(database.num_vectors, database.dim, "向量库");
  const std::uint64_t query_count =
      checkedValueCount(queries.num_queries, queries.dim, "查询集");
  if (database.dtype == DataType::kFloat32) {
    if (database.values.size() != database_count ||
        queries.values.size() != query_count) {
      throw std::runtime_error("FP32 数据长度与头部元数据不一致");
    }
  } else if (database.dtype == DataType::kFloat16) {
    if (database.half_values.size() != database_count ||
        queries.half_values.size() != query_count) {
      throw std::runtime_error("FP16 数据长度与头部元数据不一致");
    }
  } else {
    (void)elementSize(database.dtype);
  }
  if (params.top_k == 0) {
    throw std::runtime_error("top_k 必须大于 0");
  }
  if (params.top_k > database.num_vectors) {
    throw std::runtime_error("top_k 不能大于向量库大小");
  }
  if (params.top_k > 100) {
    throw std::runtime_error("当前按照项目要求支持 top_k <= 100");
  }
  if (params.batch_size == 0) {
    throw std::runtime_error("batch_size 必须大于 0");
  }
  if (params.search_mode != "exact") {
    throw std::runtime_error("当前只实现了 search_mode=exact，尚未实现: " +
                             params.search_mode);
  }
  if (params.topk_mode != "simple" && params.topk_mode != "block" &&
      params.topk_mode != "two_stage" && params.topk_mode != "fused") {
    throw std::runtime_error(
        "topk_mode 必须是 simple、block、two_stage 或 fused");
  }
  if (params.topk_mode == "two_stage" && params.top_k > 10) {
    throw std::runtime_error("当前 two_stage Top-K 支持 top_k <= 10");
  }
  if (params.distance_mode != "simple" && params.distance_mode != "warp") {
    throw std::runtime_error("distance_mode 必须是 simple 或 warp");
  }
  if (params.topk_mode == "fused" && params.distance_mode != "warp") {
    throw std::runtime_error("fused Top-K 要求 distance_mode=warp");
  }
  if (params.exact_query_tile != 1 && params.exact_query_tile != 2 &&
      params.exact_query_tile != 4 && params.exact_query_tile != 8) {
    throw std::runtime_error("exact_query_tile 必须是 1、2、4 或 8");
  }
  if (params.exact_query_tile != 1 && params.topk_mode != "fused") {
    throw std::runtime_error("exact_query_tile > 1 只支持 fused Top-K");
  }
  if (params.exact_fused_blocks != 0 && params.topk_mode != "fused") {
    throw std::runtime_error("exact_fused_blocks 只支持 fused Top-K");
  }
  if (params.exact_query_tile == 8 && params.top_k > 10) {
    throw std::runtime_error("exact_query_tile=8 当前只支持 top_k <= 10");
  }
  if (params.exact_fused_layout != "warp_query" &&
      params.exact_fused_layout != "block_tile") {
    throw std::runtime_error(
        "exact_fused_layout 必须是 warp_query 或 block_tile");
  }
  if (params.exact_fused_layout == "block_tile" &&
      (params.topk_mode != "fused" || params.exact_query_tile != 8 ||
       params.top_k > 10)) {
    throw std::runtime_error(
        "block_tile 当前要求 fused、exact_query_tile=8、top_k<=10");
  }
  if (params.exact_fused_layout == "block_tile") {
    const std::uint32_t maximum_dim =
        database.dtype == DataType::kFloat32 ? 384 : 768;
    if (database.dim > maximum_dim) {
      throw std::runtime_error("block_tile 当前共享内存预算不支持该维度");
    }
  }
}
