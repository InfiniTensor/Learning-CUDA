#pragma once
#include "numeric.cuh"

namespace lp {
struct Layout {
  size_t rows, cols;
  int dtype, fmt, block;
  bool tensor, stochastic;
  uint32_t seed;
  size_t count() const { return rows * cols; }
  size_t groups() const { return tensor ? 1 : rows * ((cols + block - 1) / block); }
  size_t bytes() const { return fmt == MXFP8 ? count() : rows * ((cols + 1) / 2); }
};
#if !defined(LP_ASCEND)
__device__ inline float warp_max(float x, int width = 32) {
  for (int s = width / 2; s; s /= 2)
    x = fmaxf(x, __shfl_xor_sync(FULL_WARP_MASK, x, s, width));
  return x;
}
#endif
} // namespace lp
