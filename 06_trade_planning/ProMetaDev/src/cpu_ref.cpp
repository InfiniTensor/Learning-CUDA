// cpu_ref.cpp - CPU 参考实现（Edmonds-Karp：每轮 BFS 找一条最短增广路，正确性基准）
#include "maxflow.h"
#include <algorithm>
#include <queue>
#include <vector>

namespace mf {

int64_t cpu_edmonds_karp_maxflow(const ResidualGraph& g, int32_t s, int32_t t) {
    if (s == t)
        return 0;
    const int32_t N = g.num_nodes;
    std::vector<int32_t> residual = g.cap; // 残量副本

    int64_t total = 0;
    while (true) {
        // BFS 找增广路径
        std::vector<int32_t> parent(N, -1);
        std::vector<int32_t> parent_edge(N, -1);
        std::queue<int32_t> q;
        parent[s] = s;
        q.push(s);
        while (!q.empty() && parent[t] == -1) {
            int32_t u = q.front();
            q.pop();
            for (int32_t p = g.row_ptr[u]; p < g.row_ptr[u + 1]; ++p) {
                int32_t v = g.col_idx[p];
                if (parent[v] == -1 && residual[p] > 0) {
                    parent[v] = u;
                    parent_edge[v] = p;
                    q.push(v);
                }
            }
        }
        if (parent[t] == -1)
            break; // 无增广路径

        // 找瓶颈容量
        int32_t bottleneck = INT32_MAX;
        for (int32_t v = t; v != s; v = parent[v]) {
            int32_t p = parent_edge[v];
            bottleneck = std::min(bottleneck, residual[p]);
        }

        // 更新残量
        for (int32_t v = t; v != s; v = parent[v]) {
            int32_t p = parent_edge[v];
            residual[p] -= bottleneck;
            residual[g.rev_edge[p]] += bottleneck;
        }
        total += bottleneck;
    }
    return total;
}

} // namespace mf
