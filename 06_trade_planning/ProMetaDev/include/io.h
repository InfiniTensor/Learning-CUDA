// io.h - 最大流 I/O 接口
#pragma once
#include "types.h"
#include <string>
#include <vector>

namespace mf {

// 读取 CSR 二进制图文件：
//   N(int32) M(int32) row_ptr[N+1] col_idx[M] values[M]
// 若文件头 magic 为 "GRPH" 则跳过 4 字节（兼容带 magic 的版本）
GraphCSR load_graph(const std::string& path);

// 读取查询文件：每行 "source target"
std::vector<Query> load_queries(const std::string& path);

// 写入结果文件：每行 "source target maxflow"
void write_results(const std::string& path, const std::vector<Query>& queries,
                   const std::vector<int64_t>& maxflows);

// 写入性能日志
void write_perf_log(const std::string& path, const PerfStats& stats);

// 生成带反向边的残量图 CSR（正向边 0..M-1，反向边 M..2M-1）
ResidualGraph build_residual_graph(const GraphCSR& g);

} // namespace mf
