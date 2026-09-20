#include "flow.hpp"
#include <algorithm>
#include <climits>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace trade {
Graph read_graph(const std::string &path) {
  static_assert(sizeof(int) == 4, "CSR format requires 32-bit int");
  const uint32_t endian = 1;
  if (*reinterpret_cast<const unsigned char *>(&endian) != 1)
    throw std::runtime_error("CSR input requires a little-endian host");
  std::ifstream in(path, std::ios::binary);
  if (!in)
    throw std::runtime_error("Cannot open graph: " + path);
  Graph g;
  in.read(reinterpret_cast<char *>(&g.n), 4);
  in.read(reinterpret_cast<char *>(&g.m), 4);
  if (!in || g.n < 2 || g.n == INT_MAX || g.m < 0 || g.m > INT_MAX / 2)
    throw std::runtime_error("Invalid N/M or residual index exceeds int32");
  const uint64_t bytes = 8ULL + 4ULL * (uint64_t(g.n) + 1 + 2ULL * g.m);
  if (std::filesystem::file_size(path) != bytes)
    throw std::runtime_error(
        "CSR byte length mismatch (truncated or trailing data)");
  g.row.resize(size_t(g.n) + 1);
  g.col.resize(g.m);
  g.cap.resize(g.m);
  in.read(reinterpret_cast<char *>(g.row.data()), 4ULL * g.row.size());
  in.read(reinterpret_cast<char *>(g.col.data()), 4ULL * g.col.size());
  in.read(reinterpret_cast<char *>(g.cap.data()), 4ULL * g.cap.size());
  if (!in || g.row.front() != 0 || g.row.back() != g.m)
    throw std::runtime_error("Invalid CSR row endpoints");
  for (int u = 0; u < g.n; ++u)
    if (g.row[u] < 0 || g.row[u] > g.row[u + 1])
      throw std::runtime_error("CSR row_ptr must be nondecreasing");
  for (int e = 0; e < g.m; ++e)
    if (g.col[e] < 0 || g.col[e] >= g.n || g.cap[e] < 0)
      throw std::runtime_error("Invalid destination or negative capacity");
  return g;
}

std::vector<std::pair<int, int>> read_queries(const std::string &path, int n) {
  std::ifstream in(path);
  if (!in)
    throw std::runtime_error("Cannot open queries: " + path);
  std::vector<std::pair<int, int>> queries;
  std::string line;
  int line_number = 0;
  while (std::getline(in, line)) {
    ++line_number;
    line = line.substr(0, line.find('#'));
    if (line.find_first_not_of(" \t\r") == std::string::npos)
      continue;
    std::istringstream stream(line);
    int s, t;
    std::string extra;
    if (!(stream >> s >> t) || (stream >> extra) || s < 0 || t < 0 || s >= n ||
        t >= n || s == t)
      throw std::runtime_error("Invalid distinct source/target at query line " +
                               std::to_string(line_number));
    queries.emplace_back(s, t);
  }
  if (queries.empty())
    throw std::runtime_error("Query file is empty");
  return queries;
}

ResidualGraph::ResidualGraph(const Graph &g) : n(g.n), row(size_t(n) + 1, 0) {
  for (int u = 0; u < n; ++u)
    for (int e = g.row[u]; e < g.row[u + 1]; ++e) {
      ++row[u + 1];
      ++row[g.col[e] + 1];
    }
  for (int u = 0; u < n; ++u)
    row[u + 1] += row[u];
  col.resize(row.back());
  rev.resize(row.back());
  cap.assign(row.back(), 0);
  auto next = row;
  for (int u = 0; u < n; ++u)
    for (int e = g.row[u]; e < g.row[u + 1]; ++e) {
      const int v = g.col[e], a = next[u]++, b = next[v]++;
      col[a] = v;
      col[b] = u;
      rev[a] = b;
      rev[b] = a;
      cap[a] = g.cap[e];
    }
}
} // namespace trade
