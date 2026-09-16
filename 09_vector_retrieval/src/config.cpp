#include "vsearch/config.hpp"

#include <cctype>
#include <fstream>
#include <sstream>

namespace vs {
namespace {

std::string trim(const std::string& s) {
  std::size_t b = 0, e = s.size();
  while (b < e && std::isspace(static_cast<unsigned char>(s[b]))) ++b;
  while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1]))) --e;
  return s.substr(b, e - b);
}

std::string unquote(const std::string& s) {
  if (s.size() >= 2 && ((s.front() == '"' && s.back() == '"') ||
                        (s.front() == '\'' && s.back() == '\'')))
    return s.substr(1, s.size() - 2);
  return s;
}

template <typename T>
bool fromString(const std::string& s, T* out) {
  std::istringstream in(s);
  in >> *out;
  return !in.fail();
}

void apply(SearchConfig* c, const std::string& keyRaw, const std::string& val) {
  std::string key = keyRaw;
  for (auto& ch : key) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
  const std::string v = unquote(trim(val));
  auto parseInt = [&](int* p) {
    if (!fromString(v, p)) throwRuntime("config: bad integer for " + keyRaw);
  };
  auto parseInt64 = [&](i64* p) {
    if (!fromString(v, p)) throwRuntime("config: bad integer for " + keyRaw);
  };

  if (key == "top_k") parseInt(&c->top_k);
  else if (key == "search_mode") c->search_mode = v;
  else if (key == "batch_size") parseInt(&c->batch_size);
  else if (key == "nlist") parseInt(&c->nlist);
  else if (key == "nprobe") parseInt(&c->nprobe);
  else if (key == "pq_m") parseInt(&c->pq_m);
  else if (key == "pq_ks") parseInt(&c->pq_ks);
  else if (key == "pq_rerank") parseInt(&c->pq_rerank);
  else if (key == "kmeans_iters") parseInt(&c->kmeans_iters);
  else if (key == "kmeans_sample") parseInt64(&c->kmeans_sample);
  else if (key == "kmeans_chunk_rows") parseInt64(&c->kmeans_chunk_rows);
  else if (key == "kmeans_seed") parseInt(&c->kmeans_seed);
  else if (key == "kmeans_assign_frac") {
    if (!fromString(v, &c->kmeans_assign_frac))
      throwRuntime("config: bad double for kmeans_assign_frac");
  } else if (key == "cosine_normalize_centers") {
    c->cosine_normalize_centers = (v == "1" || v == "true" || v == "yes");
  } else if (key == "nthreads") parseInt(&c->nthreads);
  else if (key == "index_path") c->index_path = v;
  else if (key == "result_path") c->result_path = v;
  else if (key == "perf_log_path") c->perf_log_path = v;
  else if (key == "quality_log_path") c->quality_log_path = v;
  else if (key == "force_rebuild") {
    c->force_rebuild = (v == "1" || v == "true" || v == "yes");
  } else if (key == "ref_query_limit") parseInt64(&c->ref_query_limit);
  else if (key == "exact_memory_mb") parseInt64(&c->exact_memory_mb);
  // Unknown extension keys are ignored.
}

}  // namespace

SearchConfig parseParamString(const std::string& text) {
  SearchConfig c;
  std::istringstream ss(text);
  std::string line;
  while (std::getline(ss, line)) {
    // strip inline comment only when '#' starts the token
    const auto hash = line.find('#');
    if (hash != std::string::npos) line.erase(hash);
    line = trim(line);
    if (line.empty()) continue;
    const auto eq = line.find('=');
    if (eq == std::string::npos) continue;
    apply(&c, trim(line.substr(0, eq)), trim(line.substr(eq + 1)));
  }
  if (c.top_k <= 0 || c.batch_size <= 0 || c.nprobe <= 0 || c.nlist <= 0 ||
      c.pq_m <= 0)
    throwRuntime("config: top_k/batch_size/nprobe/nlist/pq_m must be positive");
  return c;
}

SearchConfig parseParamFile(const std::string& path) {
  std::ifstream in(path);
  if (!in) throwRuntime("cannot open parameter file: " + path);
  std::ostringstream ss;
  ss << in.rdbuf();
  return parseParamString(ss.str());
}

}  // namespace vs
