#pragma once

#include <memory>
extern "C" void launch_ascend_had(void *, void *, void *, void *, void *, void *,
                                  uint64_t, int, int, float, bool, uint32_t, int, bool,
                                  uint32_t, int, int, unsigned);
extern "C" void launch_ascend_prepare(void *, void *, void *, uint64_t, int, int, bool,
                                      uint32_t, unsigned);
extern "C" void launch_ascend_max_finish(void *, void *, void *, unsigned);
extern "C" void launch_ascend_cube(void *, void *, void *, void *, uint64_t, int, int,
                                   unsigned);
namespace lp {
inline void ascend_transform(const void *in, void *out, size_t rows, int d, int dtype,
                             bool norm, bool signs, uint32_t seed, int stage = 1) {
  launch_ascend_had(lp_ascend::stream(), const_cast<void *>(in), out, nullptr, nullptr,
                    nullptr, rows, d, dtype, norm ? float(1 / std::sqrt(double(d))) : 1,
                    signs, seed, 0, false, 0, stage, 0, ascend_cores(rows));
}
template <int MODE>
inline void launch_had(const void *in, void *out, size_t rows, int d, int dtype,
                       bool norm, bool signs, uint32_t seed, uint8_t *, uint8_t *,
                       float *, Layout) {
  static_assert(MODE == 0, "Ascend uses explicit reduction/fusion entrypoints");
  ascend_transform(in, out, rows, d, dtype, norm, signs, seed);
}
inline void ascend_fused(const void *in, size_t rows, int d, int dtype, bool norm,
                         bool signs, uint32_t signseed, uint8_t *data, uint8_t *scales,
                         float *amax, Layout p, int stage = 1) {
  static AscendScratch scratch;
  unsigned cores = ascend_cores(rows);
  float factor = norm ? float(1 / std::sqrt(double(d))) : 1;
  if (p.fmt == NVFP4) {
    launch_ascend_had(lp_ascend::stream(), const_cast<void *>(in), nullptr, nullptr,
                      nullptr, scratch.p, rows, d, dtype, factor, signs, signseed,
                      p.fmt, p.stochastic, p.seed, stage, 1, cores);
    // Reuse the reduction finalizer through a dedicated launcher (no host amax).
    launch_ascend_max_finish(lp_ascend::stream(), scratch.p, amax, cores);
  }
  launch_ascend_had(lp_ascend::stream(), const_cast<void *>(in), nullptr, data, scales,
                    amax, rows, d, dtype, factor, signs, signseed, p.fmt, p.stochastic,
                    p.seed, stage, 2, cores);
}
inline void fused_had(const void *in, size_t rows, int d, int dtype, bool norm,
                      bool signs, uint32_t seed, uint8_t *data, uint8_t *scales,
                      float *amax, Layout p) {
  ascend_fused(in, rows, d, dtype, norm, signs, seed, data, scales, amax, p);
}

// Native Cube implementation: original FP16/BF16 inputs, FP32 Mmad
// accumulation, and a separate FP32 butterfly/normalization epilogue.
struct CubePlan {
  size_t rows;
  int d, k, dtype;
  Device input, product, weight;
  CubePlan(size_t rows_, int d_, int k_, int dtype_)
      : rows(rows_), d(d_), k(k_), dtype(dtype_), input(rows * d * 2),
        product(rows * d * 4), weight(k * k * 2) {
    std::vector<uint16_t> h(k * k);
    for (int i = 0; i < k; ++i)
      for (int j = 0; j < k; ++j)
        save(h.data(), i * k + j, __builtin_parity(unsigned(i & j)) ? -1 : 1, dtype);
    weight.upload(h.data(), h.size() * 2);
  }
  void run(const void *in, int type, bool signs, uint32_t seed) {
    void *source = const_cast<void *>(in);
    if (signs) {
      launch_ascend_prepare(lp_ascend::stream(), source, input.ptr, rows * d, d, type,
                            true, seed, ascend_cores((rows * d + 511) / 512));
      source = input.ptr;
    }
    size_t matrixRows = rows * d / k, tiles = ((matrixRows + 15) / 16) * (k / 16);
    launch_ascend_cube(lp_ascend::stream(), source, weight.ptr, product.ptr, matrixRows,
                       k, type, unsigned(std::min<size_t>(24, tiles)));
  }
};
inline CubePlan &cube_plan(size_t rows, int d, int k, int dtype) {
  static std::unique_ptr<CubePlan> p;
  if (!p || p->rows != rows || p->d != d || p->k != k || p->dtype != dtype)
    p = std::make_unique<CubePlan>(rows, d, k, dtype);
  return *p;
}
inline void launch_had_mma(const void *in, void *out, size_t rows, int d, int dtype,
                           bool norm, bool signs, uint32_t seed) {
  auto &p = cube_plan(rows, d, 16, dtype);
  p.run(in, dtype, signs, seed);
  ascend_transform(p.product.ptr, out, rows, d, dtype, norm, false, seed, 16);
}
inline void fused_had_mma(const void *in, size_t rows, int d, int dtype, bool norm,
                          bool signs, uint32_t seed, uint8_t *data, uint8_t *scales,
                          float *amax, Layout layout) {
  auto &p = cube_plan(rows, d, 16, dtype);
  p.run(in, dtype, signs, seed);
  ascend_fused(p.product.ptr, rows, d, dtype, norm, false, seed, data, scales, amax,
               layout, 16);
}
inline void materialized_had_mma(const void *in, void *out, size_t rows, int d,
                                 int dtype, bool norm, bool signs, uint32_t seed,
                                 uint8_t *data, uint8_t *scales, float *amax,
                                 Layout p) {
  auto &plan = cube_plan(rows, d, 16, dtype);
  plan.run(in, dtype, signs, seed);
  static AscendScratch scratch;
  unsigned cores = ascend_cores(rows);
  launch_ascend_had(lp_ascend::stream(), plan.product.ptr, out, nullptr, nullptr,
                    scratch.p, rows, d, dtype,
                    norm ? float(1 / std::sqrt(double(d))) : 1, false, seed, p.fmt,
                    p.stochastic, p.seed, 16, 3, cores);
  launch_ascend_max_finish(lp_ascend::stream(), scratch.p, amax, cores);
  launch_ascend_quant(lp_ascend::stream(), out, data, scales, amax, rows, d, dtype,
                      p.fmt, p.block, p.tensor, p.stochastic, p.seed,
                      ascend_cores(rows));
}
} // namespace lp
