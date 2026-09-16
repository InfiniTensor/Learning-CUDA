#pragma once

#include "vsearch/common.hpp"

namespace vs {

// Brute-force CPU reference. `ids` and `scores` are nq*topK row-major; the K
// entries for a query are ordered best-first with deterministic id tie-breaks.
// L2 / cosine entries are Euclidean / cosine distances; inner product entries
// are raw dot products. This is deliberately simple (not vectorized) so it can
// serve as an independent correctness oracle.
//
// 单线程暴力精确检索（正确性 oracle）：输出 nq×topK 行主序结果，
// 每个 query 的 K 个结果按优劣排序，同分时按 id 升序。刻意保持朴素实现
// （不做向量化），以保证它与 GPU 实现相互独立，能真正验证 GPU 的正确性。
// 返回值为 CPU 耗时（毫秒），用于计算加速比。
double cpuExactSearch(const Dataset& ds, i64 nq, const float* queries,
                      int topK, std::vector<i32>* ids,
                      FloatVec* scores);

}  // namespace vs
