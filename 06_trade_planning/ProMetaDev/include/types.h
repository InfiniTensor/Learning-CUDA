// types.h - GPU 最大流的基础类型
#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace mf {

// ======================= CSR 图（正向边，输入格式） =======================
struct GraphCSR {
    int32_t num_nodes = 0;        // N
    int32_t num_edges = 0;        // M
    std::vector<int32_t> row_ptr; // 长度 N+1
    std::vector<int32_t> col_idx; // 长度 M
    std::vector<int32_t> cap;     // 长度 M，非负容量
};

// ======================= 残量图（含反向边，2M 条边） =======================
// 每条输入边 (u->v, c) 生成两条残量边：
//   正向边 e_fwd：u -> v，容量 c
//   反向边 e_rev：v -> u，容量 0
// 边 e 的反向边索引记录在 rev_edge[e] 中（互为反向）。
// 每个节点的出边列表先排正向边，再排反向边。
struct ResidualGraph {
    int32_t num_nodes = 0;
    int32_t num_edges = 0;         // = 2 * M
    std::vector<int32_t> row_ptr;  // 长度 N+1（含反向边）
    std::vector<int32_t> col_idx;  // 长度 2M
    std::vector<int32_t> cap;      // 长度 2M，原始容量
    std::vector<int32_t> rev_edge; // 长度 2M，rev_edge[e] = e 的反向边索引
};

// ======================= 查询 =======================
struct Query {
    int32_t source;
    int32_t target;
};

// ======================= 性能统计 =======================
struct PerfStats {
    double preprocess_ms = 0.0; // 读图 + 建残量图 + H2D
    double ttfq_ms = 0.0;       // 首次查询时间（含残量重置）
    double total_ms = 0.0;      // 所有查询总时间
    double tpq_ms = 0.0;        // 平均每查询时间
    int num_phases = 0;         // 最后一次查询的 push-relabel 阶段数
    int num_queries = 0;
};

} // namespace mf
