#include "vsearch/metric.hpp"

#include <algorithm>
#include <cctype>

namespace vs {

// 解析度量字符串（大小写不敏感）：l2 / inner_product(ip) / cosine / none。
// none 只允许出现在查询文件里（查询向量本身不带度量语义）。
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

// 解析数据类型字符串：fp32(float) / fp16(half)。
DataType parseDataType(const std::string& s) {
  std::string t = s;
  std::transform(t.begin(), t.end(), t.begin(),
                 [](unsigned char c) { return static_cast<char>(::tolower(c)); });
  if (t == "fp32" || t == "float") return DataType::Fp32;
  if (t == "fp16" || t == "half") return DataType::Fp16;
  throwRuntime("unknown dtype '" + s + "'");
}

// 解析检索模式字符串：exact / ivf_flat / ivf_pq。
SearchMode parseSearchMode(const std::string& s) {
  std::string t = s;
  std::transform(t.begin(), t.end(), t.begin(),
                 [](unsigned char c) { return static_cast<char>(::tolower(c)); });
  if (t == "exact") return SearchMode::Exact;
  if (t == "ivf_flat") return SearchMode::IvfFlat;
  if (t == "ivf_pq") return SearchMode::IvfPq;
  throwRuntime("unknown search_mode '" + s + "'");
}

// 索引类型只允许 ivf_flat / ivf_pq（exact 无需索引）。
IndexKind parseIndexKind(const std::string& s) {
  SearchMode m = parseSearchMode(s);
  if (m == SearchMode::IvfFlat) return IndexKind::IvfFlat;
  if (m == SearchMode::IvfPq) return IndexKind::IvfPq;
  throwRuntime("index kind must be ivf_flat or ivf_pq");
}

}  // namespace vs
