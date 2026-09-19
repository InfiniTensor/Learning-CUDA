#include "vector_search/search.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace vector_search {
namespace {

std::string trim(std::string value) {
  const auto is_space = [](unsigned char c) { return std::isspace(c) != 0; };
  value.erase(value.begin(),
              std::find_if(value.begin(), value.end(),
                           [&](unsigned char c) { return !is_space(c); }));
  value.erase(std::find_if(value.rbegin(), value.rend(),
                           [&](unsigned char c) { return !is_space(c); })
                  .base(),
              value.end());
  return value;
}

std::string unquote(std::string value) {
  value = trim(std::move(value));
  if (value.size() >= 2 && value.front() == '"' && value.back() == '"') {
    return value.substr(1, value.size() - 2);
  }
  return value;
}

std::unordered_map<std::string, std::string> load_values(
    const std::string& path) {
  std::ifstream input(path);
  if (!input) throw std::runtime_error("cannot open config: " + path);
  std::unordered_map<std::string, std::string> result;
  std::string line;
  std::size_t line_number = 0;
  while (std::getline(input, line)) {
    ++line_number;
    if (const auto comment = line.find('#'); comment != std::string::npos) {
      line.resize(comment);
    }
    line = trim(std::move(line));
    if (line.empty()) continue;
    const auto equals = line.find('=');
    if (equals == std::string::npos) {
      throw std::runtime_error("expected key = value at " + path + ":" +
                               std::to_string(line_number));
    }
    const std::string key = trim(line.substr(0, equals));
    const std::string value = unquote(line.substr(equals + 1));
    if (key.empty() || value.empty() || !result.emplace(key, value).second) {
      throw std::runtime_error("invalid or duplicate key at " + path + ":" +
                               std::to_string(line_number));
    }
  }
  return result;
}

const std::string& required(
    const std::unordered_map<std::string, std::string>& values,
    const std::string& key, const std::string& path) {
  const auto it = values.find(key);
  if (it == values.end()) {
    throw std::runtime_error("missing '" + key + "' in " + path);
  }
  return it->second;
}

int parse_int(const std::string& value, const std::string& key) {
  std::size_t consumed = 0;
  const int parsed = std::stoi(value, &consumed);
  if (consumed != value.size()) {
    throw std::runtime_error("invalid integer for '" + key + "'");
  }
  return parsed;
}

std::int64_t parse_int64(const std::string& value, const std::string& key) {
  std::size_t consumed = 0;
  const std::int64_t parsed = std::stoll(value, &consumed);
  if (consumed != value.size()) {
    throw std::runtime_error("invalid int64 for '" + key + "'");
  }
  return parsed;
}

std::uint64_t parse_uint64(const std::string& value, const std::string& key) {
  std::size_t consumed = 0;
  const auto parsed = std::stoull(value, &consumed);
  if (consumed != value.size()) {
    throw std::runtime_error("invalid uint64 for '" + key + "'");
  }
  return parsed;
}

SearchMode parse_mode(const std::string& value) {
  if (value == "exact") return SearchMode::kExact;
  if (value == "ivf_flat") return SearchMode::kIvfFlat;
  if (value == "ivf_pq") {
    throw std::runtime_error(
        "ivf_pq is not implemented; use ivf_flat for the required ANN path");
  }
  throw std::runtime_error("unsupported search_mode: " + value);
}

}  // namespace

SearchConfig load_search_config(const std::string& path) {
  // 解析配置并检查参数范围。
  const auto values = load_values(path);
  SearchConfig config;
  config.top_k = parse_int(required(values, "top_k", path), "top_k");
  config.mode = parse_mode(required(values, "search_mode", path));
  config.batch_size =
      parse_int(required(values, "batch_size", path), "batch_size");
  config.nlist = parse_int(required(values, "nlist", path), "nlist");
  config.nprobe = parse_int(required(values, "nprobe", path), "nprobe");
  config.pq_m = parse_int(required(values, "pq_m", path), "pq_m");
  if (const auto it = values.find("kmeans_iterations"); it != values.end()) {
    config.kmeans_iterations = parse_int(it->second, "kmeans_iterations");
  }
  if (const auto it = values.find("training_samples"); it != values.end()) {
    config.training_samples = parse_int64(it->second, "training_samples");
  }
  if (const auto it = values.find("quality_queries"); it != values.end()) {
    config.quality_queries = parse_int(it->second, "quality_queries");
  }
  if (const auto it = values.find("seed"); it != values.end()) {
    config.seed = parse_uint64(it->second, "seed");
  }

  if (config.top_k <= 0 || config.top_k > 100 || config.batch_size <= 0 ||
      config.nlist <= 0 || config.nprobe <= 0 ||
      config.nprobe > config.nlist || config.kmeans_iterations < 0 ||
      config.training_samples <= 0 || config.quality_queries <= 0) {
    throw std::runtime_error("invalid search configuration bounds");
  }
  return config;
}

std::string to_string(DType dtype) {
  switch (dtype) {
    case DType::kFloat32:
      return "fp32";
    case DType::kFloat16:
      return "fp16";
  }
  throw std::logic_error("unknown dtype");
}

std::string to_string(Metric metric) {
  switch (metric) {
    case Metric::kL2:
      return "l2";
    case Metric::kInnerProduct:
      return "inner_product";
    case Metric::kCosine:
      return "cosine";
  }
  throw std::logic_error("unknown metric");
}

std::string to_string(SearchMode mode) {
  switch (mode) {
    case SearchMode::kExact:
      return "exact";
    case SearchMode::kIvfFlat:
      return "ivf_flat";
  }
  throw std::logic_error("unknown search mode");
}

}  // namespace vector_search
