// io.cpp - 最大流 I/O 实现
#include "io.h"
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace mf {

GraphCSR load_graph(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f)
        throw std::runtime_error("无法打开图文件: " + path);

    int32_t N = 0, M = 0;
    f.read(reinterpret_cast<char*>(&N), sizeof(int32_t));
    f.read(reinterpret_cast<char*>(&M), sizeof(int32_t));
    if (!f)
        throw std::runtime_error("图文件头读取失败");
    if (N <= 0 || M < 0)
        throw std::runtime_error("图规模非法: N=" + std::to_string(N) + " M=" + std::to_string(M));

    GraphCSR g;
    g.num_nodes = N;
    g.num_edges = M;
    g.row_ptr.resize(N + 1);
    g.col_idx.resize(M);
    g.cap.resize(M);

    f.read(reinterpret_cast<char*>(g.row_ptr.data()), sizeof(int32_t) * (N + 1));
    f.read(reinterpret_cast<char*>(g.col_idx.data()), sizeof(int32_t) * M);
    f.read(reinterpret_cast<char*>(g.cap.data()), sizeof(int32_t) * M);
    if (!f)
        throw std::runtime_error("图数据读取失败");

    // 基本校验
    if (g.row_ptr[0] != 0)
        throw std::runtime_error("row_ptr[0] 必须为 0");
    if (g.row_ptr[N] != M)
        throw std::runtime_error("row_ptr[N] != M");
    for (int32_t v : g.col_idx) {
        if (v < 0 || v >= N)
            throw std::runtime_error("col_idx 越界");
    }
    for (int32_t c : g.cap) {
        if (c < 0)
            throw std::runtime_error("容量必须非负");
    }
    return g;
}

std::vector<Query> load_queries(const std::string& path) {
    std::ifstream f(path);
    if (!f)
        throw std::runtime_error("无法打开查询文件: " + path);

    std::vector<Query> queries;
    int32_t s, t;
    while (f >> s >> t) {
        queries.push_back({s, t});
    }
    if (queries.empty())
        throw std::runtime_error("查询文件为空或格式错误");
    return queries;
}

void write_results(const std::string& path, const std::vector<Query>& queries,
                   const std::vector<int64_t>& maxflows) {
    std::ofstream f(path);
    if (!f)
        throw std::runtime_error("无法写入结果文件: " + path);
    for (size_t i = 0; i < queries.size(); ++i) {
        f << queries[i].source << " " << queries[i].target << " " << maxflows[i] << "\n";
    }
}

void write_perf_log(const std::string& path, const PerfStats& stats) {
    std::ofstream f(path);
    if (!f)
        throw std::runtime_error("无法写入性能日志: " + path);
    f << "T_preprocess(ms) = " << stats.preprocess_ms << "\n";
    f << "TTFQ(ms)         = " << stats.ttfq_ms << "\n";
    f << "T_total(ms)      = " << stats.total_ms << "\n";
    f << "TPQ(ms)          = " << stats.tpq_ms << "\n";
    f << "num_queries      = " << stats.num_queries << "\n";
    f << "last_num_phases  = " << stats.num_phases << "\n";
}

ResidualGraph build_residual_graph(const GraphCSR& g) {
    const int32_t N = g.num_nodes;
    const int32_t M = g.num_edges;

    // 统计每个节点的出边数（正向）和入边数（反向）
    std::vector<int32_t> out_deg(N, 0);
    std::vector<int32_t> in_deg(N, 0);
    for (int32_t u = 0; u < N; ++u) {
        out_deg[u] = g.row_ptr[u + 1] - g.row_ptr[u];
    }
    for (int32_t k = 0; k < M; ++k) {
        in_deg[g.col_idx[k]]++;
    }

    ResidualGraph rg;
    rg.num_nodes = N;
    rg.num_edges = 2 * M;
    rg.row_ptr.resize(N + 1);
    rg.col_idx.resize(2 * M);
    rg.cap.resize(2 * M);
    rg.rev_edge.resize(2 * M);

    // row_ptr: 每个节点的边数 = 出度 + 入度
    rg.row_ptr[0] = 0;
    for (int32_t u = 0; u < N; ++u) {
        rg.row_ptr[u + 1] = rg.row_ptr[u] + out_deg[u] + in_deg[u];
    }

    // 用游标填充：先正向边，再反向边
    std::vector<int32_t> fwd_cur(N), rev_cur(N);
    for (int32_t u = 0; u < N; ++u) {
        fwd_cur[u] = rg.row_ptr[u];
        rev_cur[u] = rg.row_ptr[u] + out_deg[u];
    }

    for (int32_t u = 0; u < N; ++u) {
        for (int32_t p = g.row_ptr[u]; p < g.row_ptr[u + 1]; ++p) {
            int32_t v = g.col_idx[p];
            int32_t c = g.cap[p];
            // 正向边：u -> v，容量 c
            int32_t e_fwd = fwd_cur[u]++;
            rg.col_idx[e_fwd] = v;
            rg.cap[e_fwd] = c;
            // 反向边：v -> u，容量 0
            int32_t e_rev = rev_cur[v]++;
            rg.col_idx[e_rev] = u;
            rg.cap[e_rev] = 0;
            // 互为反向
            rg.rev_edge[e_fwd] = e_rev;
            rg.rev_edge[e_rev] = e_fwd;
        }
    }
    return rg;
}

} // namespace mf
