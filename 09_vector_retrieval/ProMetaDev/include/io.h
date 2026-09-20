// io.h - 向量库/查询/参数文件、索引文件与结果文件的读写
#pragma once
#include "types.h"
#include <string>
#include <vector>

namespace vs {

// ======================= 配置 =======================
Config load_config(const std::string& path);

// ======================= 向量集合 =======================
// 文件格式（文本头 + 二进制数据段）：
//   [header]
//   num_vectors: int64        （查询文件为 num_queries）
//   dim: int32
//   dtype: fp32 | fp16
//   metric: l2 | inner_product | cosine   （查询文件可省略）
//
//   [data]
//   <num_vectors * dim 个元素，按 dtype 紧密排列，行主序>
VectorSet load_vectors(const std::string& path);
void save_vectors(const std::string& path, const VectorSet& vs);

// 生成合成向量库：kind = uniform | normal | clustered
//   clustered 会生成 nlist 个高斯簇，便于验证 IVF 的召回率
VectorSet gen_vectors(int64_t n, int32_t dim, const std::string& kind, uint32_t seed,
                      int nlist = 0);
// 从向量库中随机抽取查询（生成独立查询集）
VectorSet gen_queries(int64_t nq, const VectorSet& base, const std::string& kind, uint32_t seed);

// ======================= 检索结果 =======================
// 文本格式：每行 "query_id rank vector_id distance"
void save_result(const std::string& path, const SearchResult& r);
SearchResult load_result(const std::string& path, int64_t nq, int k);

// ======================= 索引文件 =======================
void save_index(const std::string& path, const IvfIndex& idx);
bool load_index(const std::string& path, IvfIndex& idx);

// ======================= 向量库 → 索引 的映射 =======================
// 索引文件不含原始向量，加载索引后需要配合原始向量库使用

} // namespace vs
