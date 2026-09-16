#pragma once

#include "vsearch/common.hpp"

namespace vs {

// Brute-force CPU reference. `ids` and `scores` are nq*topK row-major; the K
// entries for a query are ordered best-first with deterministic id tie-breaks.
// L2 / cosine entries are Euclidean / cosine distances; inner product entries
// are raw dot products. This is deliberately simple (not vectorized) so it can
// serve as an independent correctness oracle.
double cpuExactSearch(const Dataset& ds, i64 nq, const float* queries,
                      int topK, std::vector<i32>* ids,
                      FloatVec* scores);

}  // namespace vs
