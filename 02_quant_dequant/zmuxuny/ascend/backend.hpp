#pragma once
#include "../include/layout.cuh"
#include <algorithm>
extern "C" void launch_ascend_amax(void *, void *, void *, void *, uint64_t, int,
                                   unsigned);
extern "C" void launch_ascend_quant(void *, void *, void *, void *, void *, uint64_t,
                                    uint64_t, int, int, int, bool, bool, uint32_t,
                                    unsigned);
extern "C" void launch_ascend_dequant(void *, void *, void *, void *, float, uint64_t,
                                      uint64_t, int, int, bool, int, unsigned);
namespace lp {
struct AscendScratch {
  void *p{};
  AscendScratch() { lp_ascend::checked(cudaMalloc(&p, 48 * 64)); }
  ~AscendScratch() { cudaFree(p); }
};
inline unsigned ascend_cores(size_t work) {
  return unsigned(std::min<size_t>(48, work));
}
inline void quantize(const void *in, uint8_t *data, uint8_t *scales, float *amax,
                     Layout p) {
  static AscendScratch scratch;
  if (p.fmt == NVFP4 || p.tensor)
    launch_ascend_amax(lp_ascend::stream(), const_cast<void *>(in), scratch.p, amax,
                       p.count(), p.dtype, ascend_cores(p.count()));
  launch_ascend_quant(lp_ascend::stream(), const_cast<void *>(in), data, scales, amax,
                      p.rows, p.cols, p.dtype, p.fmt, p.block, p.tensor, p.stochastic,
                      p.seed,
                      ascend_cores(p.rows * ((p.cols + p.block - 1) / p.block)));
}
inline void dequantize(uint8_t *data, uint8_t *scales, float global, void *out,
                       int type, Layout p) {
  launch_ascend_dequant(lp_ascend::stream(), data, scales, out, global, p.rows, p.cols,
                        p.fmt, p.block, p.tensor, type,
                        ascend_cores(p.rows * ((p.cols + 511) / 512)));
}
} // namespace lp
