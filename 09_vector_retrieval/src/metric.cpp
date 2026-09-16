#include "vsearch/metric.hpp"

#include <algorithm>
#include <cctype>

namespace vs {

Metric parseMetric(const std::string& s) {
  std::string t = s;
  std::transform(t.begin(), t.end(), t.begin(),
                 [](unsigned char c) { return static_cast<char>(::tolower(c)); });
  if (t == "l2") return Metric::L2;
  if (t == "inner_product" || t == "ip") return Metric::InnerProduct;
  if (t == "cosine") return Metric::Cosine;
  if (t == "none") return Metric::None;
  throwRuntime("unknown metric '" + s + "'");
}

DataType parseDataType(const std::string& s) {
  std::string t = s;
  std::transform(t.begin(), t.end(), t.begin(),
                 [](unsigned char c) { return static_cast<char>(::tolower(c)); });
  if (t == "fp32" || t == "float") return DataType::Fp32;
  if (t == "fp16" || t == "half") return DataType::Fp16;
  throwRuntime("unknown dtype '" + s + "'");
}

SearchMode parseSearchMode(const std::string& s) {
  std::string t = s;
  std::transform(t.begin(), t.end(), t.begin(),
                 [](unsigned char c) { return static_cast<char>(::tolower(c)); });
  if (t == "exact") return SearchMode::Exact;
  if (t == "ivf_flat") return SearchMode::IvfFlat;
  if (t == "ivf_pq") return SearchMode::IvfPq;
  throwRuntime("unknown search_mode '" + s + "'");
}

IndexKind parseIndexKind(const std::string& s) {
  SearchMode m = parseSearchMode(s);
  if (m == SearchMode::IvfFlat) return IndexKind::IvfFlat;
  if (m == SearchMode::IvfPq) return IndexKind::IvfPq;
  throwRuntime("index kind must be ivf_flat or ivf_pq");
}

}  // namespace vs
