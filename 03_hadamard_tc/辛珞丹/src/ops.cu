// =============================================================================
//  ops.cu -- host side dispatch, launch configuration and device utilities.
// =============================================================================
#include "fhwt/fhwt.h"

#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>

#include <algorithm>
#include <cmath>
#include <memory>
#include <functional>
#include <vector>

#include "fhwt/common.cuh"
#include "fhwt/kernel_gemm.cuh"
#include "fhwt/kernel_quant.cuh"
#include "fhwt/kernel_reg.cuh"
#include "fhwt/kernel_smem.cuh"
#include "fhwt/kernel_tc.cuh"
#include "fhwt/probe.cuh"
#include "fhwt/types.cuh"

namespace fhwt {
namespace {

int ilog2_exact(int64_t v) {
  if (!is_pow2((int)v))
    throw std::invalid_argument("dim must be a power of two");
  int l = 0;
  while ((1LL << l) < v)
    ++l;
  return l;
}

[[noreturn]] void fail(const char *what) { throw std::runtime_error(what); }
[[noreturn]] void fail(const std::string &what) { throw std::runtime_error(what); }

// A per-block scale needs a power-of-two block of at least 8 elements (one packed
// store) and at most one row.  Checked on the host before any launch/capture.
//
// Until the final review only "power of two" was checked, and the tests only
// used bs = 0 / 128 at dim <= 256.  dim = 1024 with bs = 32 (the MX-style block)
// then gave a negative shift in the kernel and silently wrong scales.  The check
// also runs before a graph capture opens, because throwing inside a capture
// leaves the stream stuck in capture mode.
void check_quant_block(const Config &cfg, int64_t dim) {
  if (cfg.block_size > 0 &&
      (!is_pow2(cfg.block_size) || cfg.block_size < 8 || cfg.block_size > dim))
    fail("quant block size must be a power of two in [8, dim]");
}

template <typename T, typename Acc> struct RegLauncher {
  template <int kLogD, int kLogTPR>
  static void run(const void *x, void *y, int64_t rows, const Config &cfg, float scale,
                  cudaStream_t stream) {
    if constexpr ((1 << (kLogD - kLogTPR)) < 2) {
      fail("threads_per_row too large for this dim");
    } else {
      constexpr int kTPR = 1 << kLogTPR;
      int rpb = cfg.rows_per_block > 0 ? cfg.rows_per_block : 4;
      if (rpb < 1)
        rpb = 1;
      int block = rpb * kTPR;
      if (block > 1024) {
        rpb = 1024 / kTPR;
        if (rpb < 1)
          rpb = 1;
        block = rpb * kTPR;
      }
      int64_t grid = cfg.num_blocks > 0 ? cfg.num_blocks : (rows + rpb - 1) / rpb;
      if (grid > 2147483647LL)
        grid = 2147483647LL;
      fhwt_reg_kernel<T, Acc, kLogD, kLogTPR><<<(int)grid, block, 0, stream>>>(
          reinterpret_cast<const T *>(x), reinterpret_cast<T *>(y), rows, scale);
    }
  }
};

// Dispatch over log2(dim) for a fixed (element, accumulator) pair.
template <typename T, typename Acc>
void dispatch_dim(const void *x, void *y, int64_t rows, int logD, int logTPR, const Config &cfg,
                  float scale, cudaStream_t stream) {
  using L = RegLauncher<T, Acc>;
  switch (logD) {
  case 5:
    if (logTPR == 3)
      return L::template run<5, 3>(x, y, rows, cfg, scale, stream);
    return L::template run<5, 4>(x, y, rows, cfg, scale, stream);
  case 6:
    if (logTPR == 3)
      return L::template run<6, 3>(x, y, rows, cfg, scale, stream);
    if (logTPR == 4)
      return L::template run<6, 4>(x, y, rows, cfg, scale, stream);
    return L::template run<6, 5>(x, y, rows, cfg, scale, stream);
  case 7:
    if (logTPR == 3)
      return L::template run<7, 3>(x, y, rows, cfg, scale, stream);
    if (logTPR == 4)
      return L::template run<7, 4>(x, y, rows, cfg, scale, stream);
    return L::template run<7, 5>(x, y, rows, cfg, scale, stream);
  case 8:
    if (logTPR == 3)
      return L::template run<8, 3>(x, y, rows, cfg, scale, stream);
    if (logTPR == 4)
      return L::template run<8, 4>(x, y, rows, cfg, scale, stream);
    return L::template run<8, 5>(x, y, rows, cfg, scale, stream);
  case 9:
    if (logTPR == 4)
      return L::template run<9, 4>(x, y, rows, cfg, scale, stream);
    return L::template run<9, 5>(x, y, rows, cfg, scale, stream);
  case 10:
    return L::template run<10, 5>(x, y, rows, cfg, scale, stream);
  default:
    fail("dim outside the register-kernel range (32..1024)");
  }
}

int default_log_tpr(int logD) {
  // Keep 8 contiguous elements (one 128-bit access for fp16/bf16) per lane
  // whenever the row is long enough for a full warp to be used.
  if (logD <= 6)
    return logD - 3; // EPT = 8
  return 5;          // TPR = 32
}

int default_rows_per_block(int logD, int logTPR) {
  int tpr = 1 << logTPR;
  // Aim at 128..256 threads per block, but never fewer than 32 threads.
  int rpb = 256 / tpr;
  if (rpb < 1)
    rpb = 1;
  if (rpb > 32)
    rpb = 32;
  (void)logD;
  return rpb;
}

// The fused quantiser does not have the rotate kernels' balance.  It is bound by
// the per-element encoder, not by DRAM traffic, so it wants *fewer* elements per
// lane (the rounded values it now keeps for the packer cost registers) and
// *more* rows per block to keep the encoder pipelines busy.
//
// Measured on both platforms (rows=262144, f16, fp8 + int4, dim 64..1024):
// 8 threads/row with 16 rows/block stays within 1.07x of the best geometry for
// every shape, where the rotate-kernel default (32 threads/row, 4 rows/block)
// is up to 2.56x off it on MUSA.  On the RTX 5060 Ti the two are within 1.2x of
// each other the other way round, so one default serves both -- see the report,
// section 7.15.14.
int default_log_tpr_quant(int logD) {
  int k = logD - 3; // 8 elements per lane is the unit the packed store handles
  if (k < 2)
    k = 2; // dim = 32 needs at least 4 threads per row
  if (k > 3)
    k = 3; // 8 threads per row
  return k;
}

int default_rows_per_block_quant() { return 16; }

// -----------------------------------------------------------------------------
//  Tensor-core launch helper.  One warp owns one row, so `rows_per_block` is
//  reinterpreted as "warps per block" and the dynamic shared-memory footprint
//  is warps * dim * sizeof(T) bytes.
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
//  Two-level shared-memory launcher for large head dimensions.
//  CHUNK = 32 * EPT elements stays in registers; everything above goes through
//  a shared tile holding one full row.
// -----------------------------------------------------------------------------
constexpr int kSmemLogTPR = 5; // 32 lanes per chunk

template <typename T, typename Acc> struct SmemLauncher {
  template <int kLogD, int kLogEPT>
  static void run(const void *x, void *y, int64_t rows, const Config &cfg, float scale,
                  cudaStream_t stream) {
    constexpr int kD = 1 << kLogD;
    constexpr int kEPT = 1 << kLogEPT;
    const size_t smem = (size_t)kD * sizeof(T);
    auto *kern = fhwt_smem_kernel<T, Acc, kLogD, kLogEPT, kSmemLogTPR>;
    if (smem > 48 * 1024) {
      cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    }
    int warps = cfg.rows_per_block > 0 ? cfg.rows_per_block : 8;
    if (warps < 1)
      warps = 1;
    if (warps > 32)
      warps = 32;
    const int block = warps * 32;
    int64_t grid = cfg.num_blocks > 0 ? cfg.num_blocks : rows;
    if (grid < 1)
      grid = 1;
    if (grid > 2147483647LL)
      grid = 2147483647LL;
    kern<<<(int)grid, block, smem, stream>>>(reinterpret_cast<const T *>(x),
                                             reinterpret_cast<T *>(y), rows, scale);
  }
};

template <typename T, typename Acc>
void dispatch_smem_dim(const void *x, void *y, int64_t rows, int logD, const Config &cfg,
                       float scale, cudaStream_t stream) {
  using L = SmemLauncher<T, Acc>;
  // EPT is chosen as large as possible (8 elements, a 128-bit access) while
  // keeping CHUNK = 32*EPT inside the row.  The vector helpers need EPT >= 2,
  // so dim = 32 stays on the register kernel.
  switch (logD) {
  case 6:
    return L::template run<6, 1>(x, y, rows, cfg, scale, stream);
  case 7:
    return L::template run<7, 2>(x, y, rows, cfg, scale, stream);
  case 8:
    return L::template run<8, 3>(x, y, rows, cfg, scale, stream);
  case 9:
    return L::template run<9, 3>(x, y, rows, cfg, scale, stream);
  case 10:
    return L::template run<10, 3>(x, y, rows, cfg, scale, stream);
  case 11:
    return L::template run<11, 3>(x, y, rows, cfg, scale, stream);
  case 12:
    return L::template run<12, 3>(x, y, rows, cfg, scale, stream);
  case 13:
    return L::template run<13, 3>(x, y, rows, cfg, scale, stream);
  case 14:
    return L::template run<14, 3>(x, y, rows, cfg, scale, stream);
  default:
    fail("dim outside the shared-memory range (64..16384)");
  }
}

// The two-level kernel keeps the low bits of the transform in registers and the
// rest in a shared tile, so fp32 accumulation applies to the register stage.
// Honour the flag here as well instead of silently always using the packed
// native type.
template <typename T>
void dispatch_smem_acc(const void *x, void *y, int64_t rows, int logD, const Config &cfg,
                       float scale, cudaStream_t stream) {
  if (cfg.fp32_accum)
    dispatch_smem_dim<T, Fp32Acc<T>>(x, y, rows, logD, cfg, scale, stream);
  else
    dispatch_smem_dim<T, Native2<T>>(x, y, rows, logD, cfg, scale, stream);
}

template <typename T, typename Acc> struct TcLauncher {
  template <int kLogD>
  static void run(const void *x, void *y, int64_t rows, const Config &cfg, float scale,
                  cudaStream_t stream) {
    constexpr int kD = 1 << kLogD;
    constexpr size_t kPerWarp = (size_t)kD * sizeof(T);
    // 48 KiB is the guaranteed static per-block limit without an opt-in.
    const size_t budget = 48 * 1024;
    int warps = cfg.rows_per_block > 0 ? cfg.rows_per_block : 8;
    if (warps < 1)
      warps = 1;
    const int max_warps = (int)(budget / (kPerWarp ? kPerWarp : 1));
    if (warps > max_warps)
      warps = (max_warps < 1) ? 1 : max_warps;
    if (warps > 32)
      warps = 32;
    const int block = warps * 32;
    int64_t grid = cfg.num_blocks > 0 ? cfg.num_blocks : (rows + warps - 1) / warps;
    if (grid < 1)
      grid = 1;
    if (grid > 2147483647LL)
      grid = 2147483647LL;
    const size_t smem = (size_t)warps * kPerWarp;
    fhwt_tc_kernel<T, Acc, kLogD><<<(int)grid, block, smem, stream>>>(
        reinterpret_cast<const T *>(x), reinterpret_cast<T *>(y), rows, scale);
  }
};

template <typename T, typename Acc>
void dispatch_tc_dim(const void *x, void *y, int64_t rows, int logD, const Config &cfg, float scale,
                     cudaStream_t stream) {
  using L = TcLauncher<T, Acc>;
  switch (logD) {
  case 5:
    return L::template run<5>(x, y, rows, cfg, scale, stream);
  case 6:
    return L::template run<6>(x, y, rows, cfg, scale, stream);
  case 7:
    return L::template run<7>(x, y, rows, cfg, scale, stream);
  case 8:
    return L::template run<8>(x, y, rows, cfg, scale, stream);
  case 9:
    return L::template run<9>(x, y, rows, cfg, scale, stream);
  case 10:
    return L::template run<10>(x, y, rows, cfg, scale, stream);
  default:
    fail("dim outside the tensor-core range (32..1024)");
  }
}

} // namespace

const char *to_string(DType t) {
  switch (t) {
  case DType::kF16:
    return "fp16";
  case DType::kBF16:
    return "bf16";
  case DType::kF32:
    return "fp32";
  }
  return "?";
}

const char *to_string(KernelKind k) {
  switch (k) {
  case KernelKind::kAuto:
    return "auto";
  case KernelKind::kReg:
    return "reg";
  case KernelKind::kSmem:
    return "smem";
  case KernelKind::kTensorCore:
    return "tensorcore";
  case KernelKind::kQuant:
    return "quant";
  }
  return "?";
}

const char *to_string(QuantKind q) {
  switch (q) {
  case QuantKind::kNone:
    return "none";
  case QuantKind::kFp8E4M3:
    return "fp8e4m3";
  case QuantKind::kInt4:
    return "int4";
  }
  return "?";
}

size_t dtype_size(DType t) {
  switch (t) {
  case DType::kF16:
  case DType::kBF16:
    return 2;
  case DType::kF32:
    return 4;
  }
  return 0;
}

LaunchPlan plan(int64_t rows, int64_t dim, const Config &cfg) {
  LaunchPlan p;
  p.log_dim = ilog2_exact(dim);
  p.threads_per_row =
      cfg.threads_per_row > 0 ? cfg.threads_per_row : (1 << default_log_tpr(p.log_dim));
  int logTPR = 0;
  while ((1 << logTPR) < p.threads_per_row)
    ++logTPR;
  p.rows_per_block =
      cfg.rows_per_block > 0 ? cfg.rows_per_block : default_rows_per_block(p.log_dim, logTPR);
  p.block = p.rows_per_block * p.threads_per_row;
  if (p.block > 1024) {
    p.rows_per_block = 1024 / p.threads_per_row;
    if (p.rows_per_block < 1)
      p.rows_per_block = 1;
    p.block = p.rows_per_block * p.threads_per_row;
  }
  if ((cfg.kernel == KernelKind::kSmem && p.log_dim >= 6) || p.log_dim > 10) {
    int warps = cfg.rows_per_block > 0 ? cfg.rows_per_block : 8;
    if (warps < 1)
      warps = 1;
    if (warps > 32)
      warps = 32;
    p.threads_per_row = 32;
    p.rows_per_block = warps;
    p.block = warps * 32;
    p.smem_bytes = (int)((size_t)dim * dtype_size(cfg.dtype));
    int64_t g = cfg.num_blocks > 0 ? cfg.num_blocks : rows;
    p.grid = (int)(g < 1 ? 1 : g);
    p.kernel_name = "fhwt_smem";
    return p;
  }
  if (cfg.kernel == KernelKind::kTensorCore) {
    int warps = cfg.rows_per_block > 0 ? cfg.rows_per_block : 8;
    const size_t per_warp = (size_t)dim * dtype_size(cfg.dtype);
    int max_warps = (int)((48 * 1024) / (per_warp ? per_warp : 1));
    if (max_warps < 1)
      max_warps = 1;
    if (warps > max_warps)
      warps = max_warps;
    if (warps > 32)
      warps = 32;
    p.threads_per_row = 32;
    p.rows_per_block = warps;
    p.block = warps * 32;
    p.smem_bytes = (int)((size_t)warps * per_warp);
    int64_t g = cfg.num_blocks > 0 ? cfg.num_blocks : (rows + warps - 1) / warps;
    p.grid = (int)(g < 1 ? 1 : g);
    p.kernel_name = "fhwt_tc";
    return p;
  }
  int64_t grid =
      cfg.num_blocks > 0 ? cfg.num_blocks : (rows + p.rows_per_block - 1) / p.rows_per_block;
  p.grid = (int)(grid < 1 ? 1 : grid);
  p.kernel_name = "fhwt_reg";
  return p;
}

void hadamard(const void *x, void *y, int64_t rows, int64_t dim, const Config &cfg, float scale,
              cudaStream_t stream) {
  if (rows <= 0)
    return;
  if (dim < 32 || dim > 16384)
    fail("dim outside the supported range (32..16384)");
  const int logD = ilog2_exact(dim);
  const int logTPR =
      cfg.threads_per_row > 0 ? (int)(ilog2_exact(cfg.threads_per_row)) : default_log_tpr(logD);
  // Above one warp's register file (dim > 1024) the two-level shared-memory
  // kernel is the only option, so `auto` selects it there.
  if (logD > 10 || (cfg.kernel == KernelKind::kSmem && logD >= 6)) {
    if (cfg.dtype == DType::kF16)
      return dispatch_smem_acc<__half>(x, y, rows, logD, cfg, scale, stream);
    if (cfg.dtype == DType::kBF16)
      return dispatch_smem_acc<__nv_bfloat16>(x, y, rows, logD, cfg, scale, stream);
    return dispatch_smem_dim<float, Fp32Acc<float>>(x, y, rows, logD, cfg, scale, stream);
  }
  if (cfg.kernel == KernelKind::kSmem && logD >= 6) {
    // dim > 1024 only exists on the two-level shared-memory path.
    if (cfg.dtype == DType::kF16)
      return dispatch_smem_acc<__half>(x, y, rows, logD, cfg, scale, stream);
    if (cfg.dtype == DType::kBF16)
      return dispatch_smem_acc<__nv_bfloat16>(x, y, rows, logD, cfg, scale, stream);
    return dispatch_smem_dim<float, Fp32Acc<float>>(x, y, rows, logD, cfg, scale, stream);
  }
#if defined(__MUSACC__)
  // No mma.sync in the MUSA device ISA, so kernel_tc.cuh is a stub there (see the
  // note at the top of that file).  Reject the request instead of running a
  // kernel that writes nothing.
  if (cfg.kernel == KernelKind::kTensorCore)
    fail("--kernel tc is not available on this platform (no mma.sync in the MUSA "
         "device ISA); use --kernel reg or --kernel smem");
#endif
  if (cfg.kernel == KernelKind::kTensorCore) {
    // The MMA path covers fp16 / bf16; fp32 falls through to the register kernel.
    if (cfg.dtype == DType::kF16)
      return dispatch_tc_dim<__half, Native2<__half>>(x, y, rows, logD, cfg, scale, stream);
    if (cfg.dtype == DType::kBF16)
      return dispatch_tc_dim<__nv_bfloat16, Native2<__nv_bfloat16>>(x, y, rows, logD, cfg, scale,
                                                                    stream);
  }
  switch (cfg.dtype) {
  case DType::kF16:
    if (cfg.fp32_accum)
      dispatch_dim<__half, Fp32Acc<__half>>(x, y, rows, logD, logTPR, cfg, scale, stream);
    else
      dispatch_dim<__half, Native2<__half>>(x, y, rows, logD, logTPR, cfg, scale, stream);
    break;
  case DType::kBF16:
    if (cfg.fp32_accum)
      dispatch_dim<__nv_bfloat16, Fp32Acc<__nv_bfloat16>>(x, y, rows, logD, logTPR, cfg, scale,
                                                          stream);
    else
      dispatch_dim<__nv_bfloat16, Native2<__nv_bfloat16>>(x, y, rows, logD, logTPR, cfg, scale,
                                                          stream);
    break;
  case DType::kF32:
    dispatch_dim<float, Fp32Acc<float>>(x, y, rows, logD, logTPR, cfg, scale, stream);
    break;
  }
}

// -----------------------------------------------------------------------------
//  Fused transform + quantisation dispatch
// -----------------------------------------------------------------------------
template <typename T, typename Acc> struct QuantLauncher {
  template <int kLogD, int kLogTPR>
  static void run(const void *x, void *q, float *s, int64_t rows, const Config &cfg, float scale,
                  cudaStream_t stream) {
    if constexpr ((1 << (kLogD - kLogTPR)) < 8) {
      // The packed quantised store handles 8 elements per lane.
      fail("fused quantisation needs dim / threads_per_row >= 8");
    } else {
      constexpr int kTPR = 1 << kLogTPR;
      int rpb = cfg.rows_per_block > 0 ? cfg.rows_per_block : default_rows_per_block_quant();
      int block = rpb * kTPR;
      if (block > 1024) {
        rpb = 1024 / kTPR;
        block = rpb * kTPR;
      }
      int64_t grid = cfg.num_blocks > 0 ? cfg.num_blocks : (rows + rpb - 1) / rpb;
      if (grid > 2147483647LL)
        grid = 2147483647LL;
      int bslog = 0;
      if (cfg.block_size > 0) {
        bslog = 0;
        while ((1 << bslog) < cfg.block_size)
          ++bslog;
        if ((1 << bslog) != cfg.block_size)
          fail("quant block size must be a power of two");
        // One lane writes one scale, so a scale block may not be narrower than
        // the kEPT contiguous elements a lane owns (nor wider than the row).
        if (bslog < kLogD - kLogTPR || bslog > kLogD)
          fail("quant block size must lie in [dim / threads_per_row, dim]");
      }
      fhwt_quant_kernel<T, Acc, kLogD, kLogTPR, true><<<(int)grid, block, 0, stream>>>(
          reinterpret_cast<const T *>(x), q, s, rows, scale, (int)cfg.quant, bslog);
    }
  }
};

template <typename T, typename Acc>
void dispatch_quant_dim(const void *x, void *q, float *s, int64_t rows, int logD, int logTPR,
                        const Config &cfg, float scale, cudaStream_t stream) {
  using L = QuantLauncher<T, Acc>;
  switch (logD) {
  case 5:
    if (logTPR == 2)
      return L::template run<5, 2>(x, q, s, rows, cfg, scale, stream);
    return L::template run<5, 3>(x, q, s, rows, cfg, scale, stream);
  case 6:
    return L::template run<6, 3>(x, q, s, rows, cfg, scale, stream);
  case 7:
    if (logTPR == 3)
      return L::template run<7, 3>(x, q, s, rows, cfg, scale, stream);
    if (logTPR == 4)
      return L::template run<7, 4>(x, q, s, rows, cfg, scale, stream);
    return L::template run<7, 5>(x, q, s, rows, cfg, scale, stream);
  case 8:
    if (logTPR == 3)
      return L::template run<8, 3>(x, q, s, rows, cfg, scale, stream);
    if (logTPR == 4)
      return L::template run<8, 4>(x, q, s, rows, cfg, scale, stream);
    return L::template run<8, 5>(x, q, s, rows, cfg, scale, stream);
  case 9:
    if (logTPR == 4)
      return L::template run<9, 4>(x, q, s, rows, cfg, scale, stream);
    return L::template run<9, 5>(x, q, s, rows, cfg, scale, stream);
  case 10:
    return L::template run<10, 5>(x, q, s, rows, cfg, scale, stream);
  default:
    fail("dim outside the fused-quant range (32..1024)");
  }
}

int64_t quant_scale_count(int64_t rows, int64_t dim, const Config &cfg) {
  if (cfg.quant == QuantKind::kNone)
    return 0;
  int64_t per_row = 1;
  if (cfg.block_size > 0)
    per_row = dim / cfg.block_size;
  return rows * per_row;
}

void hadamard_quant(const void *x, void *q_out, float *scale_out, int64_t rows, int64_t dim,
                    const Config &cfg, float scale, cudaStream_t stream) {
  if (rows <= 0)
    return;
  if (cfg.quant == QuantKind::kNone)
    fail("hadamard_quant requires a quantisation format");
  if (dim < 32 || dim > 1024)
    fail("dim outside the supported range (32..1024)");
  const int logD = ilog2_exact(dim);
  int logTPR = cfg.threads_per_row > 0 ? (int)(ilog2_exact(cfg.threads_per_row))
                                       : default_log_tpr_quant(logD);
  if (logTPR > logD - 3)
    logTPR = logD - 3; // keep >= 8 elements per lane
  if (logTPR < 2)
    logTPR = 2;
  if (cfg.block_size > 0) {
    check_quant_block(cfg, dim);
    // Default geometry: use enough lanes per row that one lane never spans
    // more than one scale block (an explicit threads_per_row is left alone and
    // checked by the launcher instead).  E.g. dim 1024, bs 32: 8 -> 32 lanes.
    // Rejecting the case outright was the other option; widening keeps
    // `--block-size 32` usable at every dim, and for dim 512/1024 it lands on the
    // 32 lanes/row the dispatcher picks anyway.
    const int bslog = ilog2_exact(cfg.block_size);
    if (cfg.threads_per_row <= 0 && logD - logTPR > bslog)
      logTPR = logD - bslog;
  }
  switch (cfg.dtype) {
  case DType::kF16:
    if (cfg.fp32_accum)
      dispatch_quant_dim<__half, Fp32Acc<__half>>(x, q_out, scale_out, rows, logD, logTPR, cfg,
                                                  scale, stream);
    else
      dispatch_quant_dim<__half, Native2<__half>>(x, q_out, scale_out, rows, logD, logTPR, cfg,
                                                  scale, stream);
    break;
  case DType::kBF16:
    if (cfg.fp32_accum)
      dispatch_quant_dim<__nv_bfloat16, Fp32Acc<__nv_bfloat16>>(x, q_out, scale_out, rows, logD,
                                                                logTPR, cfg, scale, stream);
    else
      dispatch_quant_dim<__nv_bfloat16, Native2<__nv_bfloat16>>(x, q_out, scale_out, rows, logD,
                                                                logTPR, cfg, scale, stream);
    break;
  case DType::kF32:
    fail("fused quantisation supports fp16 / bf16 inputs");
  }
}

// Standalone quantiser: out = quant(in), used as the unfused reference path.
void quantize_rows(const void *x, void *q_out, float *scale_out, int64_t rows, int64_t dim,
                   const Config &cfg, cudaStream_t stream) {
  const int logD = ilog2_exact(dim);
  check_quant_block(cfg, dim);
  const int bslog = (cfg.block_size > 0) ? (int)(ilog2_exact(cfg.block_size)) : 0;
  const int nblocks = (cfg.block_size > 0) ? (int)(dim / cfg.block_size) : 1;
  const int threads = 128; // 4 warps, one row per warp
  dim3 grid((unsigned)((rows + 3) / 4));
  size_t smem = sizeof(float) * nblocks * (threads / 32);
  switch (cfg.dtype) {
  case DType::kF16:
    quantize_rows_kernel<__half><<<grid, threads, smem, stream>>>(
        reinterpret_cast<const __half *>(x), q_out, scale_out, rows, logD, (int)cfg.quant, bslog);
    break;
  case DType::kBF16:
    quantize_rows_kernel<__nv_bfloat16>
        <<<grid, threads, smem, stream>>>(reinterpret_cast<const __nv_bfloat16 *>(x), q_out,
                                          scale_out, rows, logD, (int)cfg.quant, bslog);
    break;
  case DType::kF32:
    fail("quantisation supports fp16 / bf16 inputs");
  }
}

DeviceInfo device_info(int device) {
  cudaDeviceProp prop{};
  cudaGetDeviceProperties(&prop, device);
  DeviceInfo info{};
  std::snprintf(info.name, sizeof(info.name), "%s", prop.name);
  info.major = prop.major;
  info.minor = prop.minor;
  info.multi_processor_count = prop.multiProcessorCount;
  info.max_threads_per_sm = prop.maxThreadsPerMultiProcessor;
  info.max_smem_per_sm = (int)prop.sharedMemPerMultiprocessor;
  info.clock_khz = prop.clockRate;
  info.memory_clock_khz = prop.memoryClockRate;
  info.memory_bus_width = prop.memoryBusWidth;
  info.l2_bytes = (int)prop.l2CacheSize;
  return info;
}

double device_bandwidth_gbps(const DeviceInfo &info) {
  // GDDR/HBM effective bandwidth: 2 transfers per clock, bus width in bits.
  return 2.0 * (double)info.memory_clock_khz * 1000.0 * (double)info.memory_bus_width / 8.0 / 1e9;
}

// -----------------------------------------------------------------------------
//  Bandwidth roof (see include/fhwt/probe.cuh for why the driver's theoretical
//  number is reported but not trusted).
// -----------------------------------------------------------------------------
RoofResult bandwidth_roof(size_t bytes, int iters, int warmup) {
  RoofResult r;
  r.bytes = (double)bytes;
  if (iters < 1)
    iters = 1;
  if (warmup < 0)
    warmup = 0;

  DeviceInfo info = device_info(0);
  int sm = info.multi_processor_count > 0 ? info.multi_processor_count : 1;

  void *a = nullptr;
  void *b = nullptr;
  unsigned *sink = nullptr;
  if (cudaMalloc(&a, bytes) != cudaSuccess || cudaMalloc(&b, bytes) != cudaSuccess) {
    cudaGetLastError();
    if (a)
      cudaFree(a);
    if (b)
      cudaFree(b);
    return r; // not enough memory: leave the fields at zero
  }
  if (cudaMalloc(&sink, (size_t)sm * 8 * sizeof(unsigned)) != cudaSuccess)
    sink = nullptr;

  cudaEvent_t s = nullptr, e = nullptr;
  cudaEventCreate(&s);
  cudaEventCreate(&e);
  auto ms_per_iter = [&](const std::function<void()> &fn) -> double {
    for (int i = 0; i < warmup; ++i)
      fn();
    cudaDeviceSynchronize();
    cudaEventRecord(s);
    for (int i = 0; i < iters; ++i)
      fn();
    cudaEventRecord(e);
    cudaEventSynchronize(e);
    float ms = 0.f;
    cudaEventElapsedTime(&ms, s, e);
    return (double)ms / (double)iters;
  };
  auto gbps = [&](double ms) { return r.bytes / (ms * 1e-3) / 1e9; };

  const int block = 256;
  const int grid = sm * 8;
  const long long n4 = (long long)(bytes / sizeof(uint4));

  // Random (not constant) content: see roof_fill_kernel.
  roof_fill_kernel<<<grid, block>>>(static_cast<uint4 *>(a), n4, 0x1234567u);
  roof_fill_kernel<<<grid, block>>>(static_cast<uint4 *>(b), n4, 0x7654321u);
  cudaDeviceSynchronize();

  // The copy path: does the driver move more or less than a SIMT kernel can?
  r.memcpy_async_gbps =
      2.0 * gbps(ms_per_iter([&] { cudaMemcpyAsync(b, a, bytes, cudaMemcpyDeviceToDevice); }));
  cudaDeviceSynchronize();
  r.memcpy_sync_gbps =
      2.0 * gbps(ms_per_iter([&] { cudaMemcpy(b, a, bytes, cudaMemcpyDeviceToDevice); }));

  r.copy_gbps = 2.0 * gbps(ms_per_iter([&] {
                  roof_copy_kernel<<<grid, block>>>(static_cast<const uint4 *>(a),
                                                    static_cast<uint4 *>(b), n4);
                }));
  r.read_gbps = gbps(ms_per_iter(
      [&] { roof_read_kernel<<<grid, block>>>(static_cast<const uint4 *>(a), sink, n4); }));
  r.write_gbps =
      gbps(ms_per_iter([&] { roof_write_kernel<<<grid, block>>>(static_cast<uint4 *>(b), n4); }));

  // Same copy kernel, working set small enough to stay in L2 (a quarter of it).
  size_t l2_bytes = (size_t)info.l2_bytes / 4;
  if (l2_bytes < (256u << 10))
    l2_bytes = 256u << 10;
  if (l2_bytes > (8u << 20))
    l2_bytes = 8u << 20;
  l2_bytes &= ~(size_t)15;
  if (l2_bytes != 0 && l2_bytes < bytes) {
    const long long n4l = (long long)(l2_bytes / sizeof(uint4));
    const int reps = 32; // amortise the launch: see roof_copy_loop_kernel
    const double saved = r.bytes;
    r.bytes = (double)l2_bytes * reps;
    r.l2_copy_gbps = 2.0 * gbps(ms_per_iter([&] {
                       roof_copy_loop_kernel<<<grid, block>>>(static_cast<const uint4 *>(a),
                                                              static_cast<uint4 *>(b), n4l, reps);
                     }));
    r.bytes = saved;
  }
  cudaDeviceSynchronize();

  cudaEventDestroy(s);
  cudaEventDestroy(e);
  cudaFree(a);
  cudaFree(b);
  if (sink)
    cudaFree(sink);
  return r;
}

// -----------------------------------------------------------------------------
//  Graph
// -----------------------------------------------------------------------------
// cudaGraphLaunch stages an unseen exec on first use.  Doing that upload here
// keeps it out of the first timed replay, and for a hand-assembled graph it is
// what stops the staging cost from being paid on *every* replay.
// CUDA 12 dropped the error-node / log-buffer out-parameters from
// cudaGraphInstantiate; MUSA 3.1 still exposes the original five-argument form.
static cudaError_t graph_instantiate(cudaGraphExec_t *exec, cudaGraph_t g) {
#if defined(__MUSACC__)
  return musaGraphInstantiate(exec, g, nullptr, nullptr, 0);
#else
  return cudaGraphInstantiate(exec, g, 0);
#endif
}

static void instantiate_and_upload(cudaGraph_t g, cudaGraphExec_t *exec, cudaStream_t stream) {
  if (graph_instantiate(exec, g) != cudaSuccess)
    fail("cudaGraphInstantiate failed");
#if defined(__MUSACC__)
  // MUSA 3.1 declares musaGraphUpload only inside a disabled block, so the exec
  // is staged lazily by the first launch instead.  The callers always warm up
  // before timing, so that cost never lands in a measured window.
  (void)stream;
#else
  if (cudaGraphUpload(*exec, stream) != cudaSuccess)
    fail("cudaGraphUpload failed");
#endif
}

Graph::~Graph() { reset(); }

void Graph::open_stream(cudaStream_t external) {
  if (external != nullptr) {
    stream_ = external;
    owns_stream_ = false;
  } else {
    // Stream capture is illegal on the legacy default stream, so a burst with no
    // caller-supplied stream is recorded on a private non-blocking one.
    if (cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking) != cudaSuccess) {
      fail("cudaStreamCreateWithFlags failed");
    }
    owns_stream_ = true;
  }
}

void Graph::close_capture() {
  if (cudaStreamEndCapture(stream_, &graph_) != cudaSuccess)
    fail("cudaStreamEndCapture failed");
  instantiate_and_upload(graph_, &exec_, stream_);
}

void Graph::reset() {
  if (exec_ != nullptr) {
    cudaGraphExecDestroy(exec_);
    exec_ = nullptr;
  }
  if (graph_ != nullptr) {
    cudaGraphDestroy(graph_);
    graph_ = nullptr;
  }
  if (stream_ != nullptr && owns_stream_)
    cudaStreamDestroy(stream_);
  stream_ = nullptr;
  owns_stream_ = false;
  repeats_ = 0;
}

void Graph::capture(const void *x, void *y, int64_t rows, int64_t dim, const Config &cfg,
                    float scale, int repeats, cudaStream_t stream) {
  if (repeats < 1)
    fail("Graph::capture requires repeats >= 1");
  if (rows <= 0)
    fail("Graph::capture requires rows > 0");
  // Everything is validated *before* the capture region opens: an exception
  // thrown inside a capture would leave the stream in capture mode and poison
  // every later CUDA call on it.
  (void)plan(rows, dim, cfg);
  if (dim < 32 || dim > 16384)
    fail("dim outside the supported range (32..16384)");
  reset();
  open_stream(stream);
  if (cudaStreamBeginCapture(stream_, cudaStreamCaptureModeThreadLocal) != cudaSuccess) {
    fail("cudaStreamBeginCapture failed");
  }
  for (int i = 0; i < repeats; ++i)
    hadamard(x, y, rows, dim, cfg, scale, stream_);
  close_capture();
  repeats_ = repeats;
}

void Graph::capture_quant(const void *x, void *q_out, float *scale_out, int64_t rows, int64_t dim,
                          const Config &cfg, float norm_scale, int repeats, cudaStream_t stream) {
  if (repeats < 1)
    fail("Graph::capture_quant requires repeats >= 1");
  if (rows <= 0)
    fail("Graph::capture_quant requires rows > 0");
  if (cfg.quant == QuantKind::kNone)
    fail("Graph::capture_quant requires a quantisation format");
  if (cfg.dtype == DType::kF32)
    fail("fused quantisation supports fp16 / bf16 inputs");
  if (dim < 32 || dim > 1024)
    fail("dim outside the fused quantisation range (32..1024)");
  check_quant_block(cfg, dim);
  reset();
  open_stream(stream);
  if (cudaStreamBeginCapture(stream_, cudaStreamCaptureModeThreadLocal) != cudaSuccess) {
    fail("cudaStreamBeginCapture failed");
  }
  for (int i = 0; i < repeats; ++i) {
    hadamard_quant(x, q_out, scale_out, rows, dim, cfg, norm_scale, stream_);
  }
  close_capture();
  repeats_ = repeats;
}

void Graph::launch(cudaStream_t stream) const {
  if (exec_ == nullptr)
    fail("Graph::launch called on an empty graph");
  if (cudaGraphLaunch(exec_, stream != nullptr ? stream : stream_) != cudaSuccess) {
    fail("cudaGraphLaunch failed");
  }
}

// -----------------------------------------------------------------------------
//  GraphSequence
// -----------------------------------------------------------------------------
void GraphSequence::open_stream(cudaStream_t external) {
  if (external != nullptr) {
    stream_ = external;
    owns_stream_ = false;
  } else {
    // Stream capture is illegal on the legacy default stream, so a sequence
    // with no caller-supplied stream is recorded on a private non-blocking one.
    if (cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking) != cudaSuccess) {
      fail("cudaStreamCreateWithFlags failed");
    }
    owns_stream_ = true;
  }
}

void GraphSequence::drop_graph() {
  if (exec_ != nullptr) {
    cudaGraphExecDestroy(exec_);
    exec_ = nullptr;
  }
  if (graph_ != nullptr) {
    cudaGraphDestroy(graph_);
    graph_ = nullptr;
  }
  for (size_t i = 0; i < children_.size(); ++i)
    cudaGraphDestroy(children_[i]);
  children_.clear();
  kernel_nodes_ = false;
  if (stream_ != nullptr && owns_stream_)
    cudaStreamDestroy(stream_);
  stream_ = nullptr;
  owns_stream_ = false;
}

GraphSequence::~GraphSequence() { reset(); }

void GraphSequence::reset() {
  drop_graph();
  steps_.clear();
  forks_ = 0;
}

void GraphSequence::validate() const {
  if (steps_.empty())
    fail("GraphSequence::record called with no steps");
  for (size_t i = 0; i < steps_.size(); ++i) {
    const Step &st = steps_[i];
    if (st.rows <= 0)
      fail("GraphSequence steps require rows > 0");
    // plan() checks that dim is a power of two and inside the kernel range;
    // the quantised path has a narrower range on top of that.
    (void)plan(st.rows, st.dim, st.cfg);
    if (st.quant) {
      if (st.cfg.quant == QuantKind::kNone)
        fail("quantised step without a format");
      if (st.cfg.dtype == DType::kF32)
        fail("fused quantisation supports fp16 / bf16 inputs");
      if (st.dim < 32 || st.dim > 1024) {
        fail("dim outside the fused quantisation range (32..1024)");
      }
      check_quant_block(st.cfg, st.dim);
    } else {
      if (st.cfg.quant != QuantKind::kNone)
        fail("GraphSequence::add requires quant == kNone");
    }
  }
}

void GraphSequence::push(Step &&st) {
  // Queuing another step invalidates any existing recording; dropping it here
  // keeps a stale graph from silently replaying the old sequence.
  if (exec_ != nullptr)
    drop_graph();
  if (st.independent)
    ++forks_;
  steps_.push_back(st);
}

void GraphSequence::add(const void *x, void *y, int64_t rows, int64_t dim, const Config &cfg,
                        float scale) {
  Step st{};
  st.x = x;
  st.y = y;
  st.scale_out = nullptr;
  st.rows = rows;
  st.dim = dim;
  st.cfg = cfg;
  st.scale = scale;
  st.quant = false;
  st.independent = false;
  push(std::move(st));
}

void GraphSequence::add_parallel(const void *x, void *y, int64_t rows, int64_t dim,
                                 const Config &cfg, float scale) {
  Step st{};
  st.x = x;
  st.y = y;
  st.scale_out = nullptr;
  st.rows = rows;
  st.dim = dim;
  st.cfg = cfg;
  st.scale = scale;
  st.quant = false;
  st.independent = true;
  push(std::move(st));
}

void GraphSequence::add_quant(const void *x, void *q_out, float *scale_out, int64_t rows,
                              int64_t dim, const Config &cfg, float norm_scale) {
  Step st{};
  st.x = x;
  st.y = q_out;
  st.scale_out = scale_out;
  st.rows = rows;
  st.dim = dim;
  st.cfg = cfg;
  st.scale = norm_scale;
  st.quant = true;
  st.independent = false;
  push(std::move(st));
}

void GraphSequence::add_quant_parallel(const void *x, void *q_out, float *scale_out, int64_t rows,
                                       int64_t dim, const Config &cfg, float norm_scale) {
  Step st{};
  st.x = x;
  st.y = q_out;
  st.scale_out = scale_out;
  st.rows = rows;
  st.dim = dim;
  st.cfg = cfg;
  st.scale = norm_scale;
  st.quant = true;
  st.independent = true;
  push(std::move(st));
}

void GraphSequence::record(cudaStream_t stream) {
  validate(); // before the capture opens: an exception inside it would poison the stream
  drop_graph();
  open_stream(stream);
  if (forks_ == 0) {
    record_linear();
  } else {
    record_fork_join();
  }
}

// The fast path: one stream capture reproduces the submission order exactly.
void GraphSequence::record_linear() {
  if (cudaStreamBeginCapture(stream_, cudaStreamCaptureModeThreadLocal) != cudaSuccess) {
    fail("cudaStreamBeginCapture failed");
  }
  for (size_t i = 0; i < steps_.size(); ++i) {
    const Step &st = steps_[i];
    if (st.quant) {
      hadamard_quant(st.x, st.y, st.scale_out, st.rows, st.dim, st.cfg, st.scale, stream_);
    } else {
      hadamard(st.x, st.y, st.rows, st.dim, st.cfg, st.scale, stream_);
    }
  }
  if (cudaStreamEndCapture(stream_, &graph_) != cudaSuccess)
    fail("cudaStreamEndCapture failed");
  instantiate_and_upload(graph_, &exec_, stream_);
}

// The fork/join path.  Stream capture can only ever produce the linear order it
// observed on the stream, so the shape is assembled by hand instead.  Capturing
// each step on its own is unavoidable -- capture is the only thing that knows how
// to turn a hadamard() / hadamard_quant() call into graph nodes -- but the
// captured one-node graph is then *unwrapped*: cudaGraphKernelNodeGetParams()
// reads its kernel node back and that node is re-added to the parent graph as a
// plain kernel node.
//
// Why unwrap: this started as a hypothesis.  The forked strategy lost to the
// linear one in report 7.11 by a fixed ~0.12-0.35 ms per replay while its
// kernels were measurably identical (8.6), and the obvious suspect was the
// child-graph node the parent had to descend into.  Unwrapping settled it: with
// every node a real kernel node the forked graph replays no faster (5.8), which
// rules the child-graph indirection out as the cause.  What did explain it is
// the number of *roots* in the graph -- see add_parallel() -- which is a
// property of the caller's step list rather than of this function.
//
// Fallback: if a captured step is not exactly one kernel node, the child graphs
// are used instead -- the previous behaviour -- so nothing that used to work
// stops working.  kernel_nodes() reports which form the recorded graph got.
//
// The edges follow the rule documented on add_parallel(): a parallel step waits
// on the last non-parallel step (the fork point); a non-parallel step waits on
// every sibling queued since that fork point plus the fork point itself (the
// join).
void GraphSequence::record_fork_join() {
  const size_t n = steps_.size();
  // Reading the kernel node back out of a captured one-node graph.  The returned
  // kernelParams point into the child graph's own storage (see the runtime docs),
  // which is why the children stay alive until drop_graph().
  auto unwrap = [](cudaGraph_t child, cudaKernelNodeParams *out) -> bool {
    cudaGraphNode_t node = nullptr;
    size_t count = 0;
    if (cudaGraphGetNodes(child, nullptr, &count) != cudaSuccess || count != 1)
      return false;
    if (cudaGraphGetNodes(child, &node, &count) != cudaSuccess)
      return false;
    cudaGraphNodeType type = cudaGraphNodeTypeEmpty;
    if (cudaGraphNodeGetType(node, &type) != cudaSuccess || type != cudaGraphNodeTypeKernel) {
      return false;
    }
    if (cudaGraphKernelNodeGetParams(node, out) != cudaSuccess)
      return false;
    return out->func != nullptr && (out->kernelParams != nullptr || out->extra != nullptr);
  };

  children_.reserve(n);
  std::vector<cudaKernelNodeParams> params(n);
  bool direct = true;
  for (size_t i = 0; i < n; ++i) {
    const Step &st = steps_[i];
    cudaGraph_t child = nullptr;
    if (cudaStreamBeginCapture(stream_, cudaStreamCaptureModeThreadLocal) != cudaSuccess) {
      fail("cudaStreamBeginCapture failed");
    }
    if (st.quant) {
      hadamard_quant(st.x, st.y, st.scale_out, st.rows, st.dim, st.cfg, st.scale, stream_);
    } else {
      hadamard(st.x, st.y, st.rows, st.dim, st.cfg, st.scale, stream_);
    }
    if (cudaStreamEndCapture(stream_, &child) != cudaSuccess)
      fail("cudaStreamEndCapture failed");
    children_.push_back(child);
    if (direct && !unwrap(child, &params[i]))
      direct = false;
  }
  std::vector<cudaGraphNode_t> nodes(n, (cudaGraphNode_t) nullptr);
  // Lays the DAG out with either the unwrapped kernel nodes or the captured child
  // graphs.  It is a lambda because the kernel-node route may have to be retried
  // as the child-graph one (see below), and the two passes share the dependency
  // logic verbatim -- only the node kind differs.
  auto build = [&](bool as_kernel) -> cudaError_t {
    std::vector<cudaGraphNode_t> deps;
    std::vector<int> fork_set;
    int join_base = -1;
    for (size_t i = 0; i < n; ++i) {
      deps.clear();
      if (steps_[i].independent) {
        if (join_base >= 0)
          deps.push_back(nodes[(size_t)join_base]);
        fork_set.push_back((int)i);
      } else {
        for (size_t k = 0; k < fork_set.size(); ++k)
          deps.push_back(nodes[(size_t)fork_set[k]]);
        if (join_base >= 0)
          deps.push_back(nodes[(size_t)join_base]);
        join_base = (int)i;
        fork_set.clear();
      }
      const cudaGraphNode_t *dep = deps.empty() ? nullptr : deps.data();
      const cudaError_t err =
          as_kernel ? cudaGraphAddKernelNode(&nodes[i], graph_, dep, deps.size(), &params[i])
                    : cudaGraphAddChildGraphNode(&nodes[i], graph_, dep, deps.size(), children_[i]);
      if (err != cudaSuccess)
        return err;
    }
    return cudaSuccess;
  };

  if (cudaGraphCreate(&graph_, 0) != cudaSuccess)
    fail("cudaGraphCreate failed");
  cudaError_t err = build(direct);
  if (err != cudaSuccess && direct) {
    // A kernel node rebuilt from cudaGraphKernelNodeGetParams() of a captured
    // graph is not universally accepted: MUSA 3.1 refuses it with "invalid device
    // function".  The child-graph route moves exactly the same work with exactly
    // the same dependencies, so fall back to it and rebuild the DAG from scratch.
    (void)cudaGetLastError(); // the refused add() leaves an error latched
    cudaGraphDestroy(graph_);
    if (cudaGraphCreate(&graph_, 0) != cudaSuccess)
      fail("cudaGraphCreate failed");
    std::fill(nodes.begin(), nodes.end(), (cudaGraphNode_t) nullptr);
    direct = false;
    err = build(false);
  }
  if (err != cudaSuccess) {
    char msg[192];
    std::snprintf(msg, sizeof(msg), "adding a fork/join node failed (%s): %s",
                  direct ? "kernel node" : "child graph", cudaGetErrorString(err));
    fail(msg);
  }
  kernel_nodes_ = direct;
  instantiate_and_upload(graph_, &exec_, stream_);
}

void GraphSequence::launch(cudaStream_t stream) const {
  if (exec_ == nullptr)
    fail("GraphSequence::launch called before record()");
  if (cudaGraphLaunch(exec_, stream != nullptr ? stream : stream_) != cudaSuccess) {
    fail("cudaGraphLaunch failed");
  }
}
bool inplace_pays_off(int64_t rows, int64_t dim, DType t, const DeviceInfo &info) {
  if (rows <= 0 || dim <= 0 || info.l2_bytes <= 0)
    return false;
  const double footprint = (double)rows * (double)dim * (double)dtype_size(t);
  const double l2 = (double)info.l2_bytes;
  // Out of place has to keep the input and the output resident at the same time;
  // in place only has to keep the tensor.  So the transform pays off exactly
  // when the tensor fits in L2 on its own while the pair does not.
  return footprint <= l2 && 2.0 * footprint > l2;
}

// -----------------------------------------------------------------------------
//  Tensor views
// -----------------------------------------------------------------------------
int64_t view_rows(const TensorView &v) { return v.shape[0] * v.shape[1] * v.shape[2]; }

int64_t view_dim(const TensorView &v) { return v.shape[3]; }

bool row_contiguous(const TensorView &v) {
  for (int i = 0; i < 4; ++i) {
    if (v.shape[i] <= 0)
      return false;
  }
  if (v.stride[3] != 1)
    return false;
  // With the last dim contiguous, the leading axes are flattened into "rows"
  // only if each one is laid out as tightly as the extents after it.  A size-1
  // axis constrains nothing, so its stride is free; a strided view (transposed
  // heads/seq, or a slice that skips part of a batch) fails here, which is what
  // stops an in-place transform from interleaving rows.
  int64_t expect = v.shape[3];
  for (int i = 2; i >= 0; --i) {
    if (v.shape[i] == 1)
      continue;
    if (v.stride[i] != expect)
      return false;
    expect *= v.shape[i];
  }
  return true;
}

bool inplace_pays_off(const TensorView &v, DType t, const DeviceInfo &info) {
  if (!row_contiguous(v))
    return false;
  return inplace_pays_off(view_rows(v), view_dim(v), t, info);
}

// -----------------------------------------------------------------------------
//  GraphExecCache
// -----------------------------------------------------------------------------
namespace {

// Captures one workload (a repeats-fold burst of one shape) on `stream`.
void record_workload(cudaStream_t stream, const void *x, void *y, float *scale_out, int64_t rows,
                     int64_t dim, const Config &cfg, float scale, int repeats, cudaGraph_t *out) {
  if (cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal) != cudaSuccess) {
    fail("cudaStreamBeginCapture failed");
  }
  for (int r = 0; r < repeats; ++r) {
    if (cfg.quant != QuantKind::kNone) {
      hadamard_quant(x, y, scale_out, rows, dim, cfg, scale, stream);
    } else {
      hadamard(x, y, rows, dim, cfg, scale, stream);
    }
  }
  if (cudaStreamEndCapture(stream, out) != cudaSuccess)
    fail("cudaStreamEndCapture failed");
}

} // namespace

GraphExecCache::GraphExecCache(int capacity) : capacity_(capacity < 1 ? 1 : capacity) {}

GraphExecCache::~GraphExecCache() { clear(); }

bool GraphExecCache::same_config(const Config &a, const Config &b) {
  return a.dtype == b.dtype && a.kernel == b.kernel && a.threads_per_row == b.threads_per_row &&
         a.rows_per_block == b.rows_per_block && a.fp32_accum == b.fp32_accum &&
         a.num_blocks == b.num_blocks && a.quant == b.quant && a.block_size == b.block_size;
}

bool GraphExecCache::matches(const Entry &e, const void *x, void *y, const float *scale_out,
                             int64_t rows, int64_t dim, const Config &cfg, float scale,
                             int repeats) {
  // The pointer comparison is the whole point.  A graph replays against the
  // addresses it was recorded with; a request that matches on every other field
  // but points somewhere else is a miss, not a hit with the wrong answer.
  return e.x == x && e.y == y && e.scale_out == scale_out && e.rows == rows && e.dim == dim &&
         e.repeats == repeats && e.scale == scale && same_config(e.cfg, cfg);
}

bool GraphExecCache::lookup(const void *x, void *y, const float *scale_out, int64_t rows,
                            int64_t dim, const Config &cfg, float scale, int repeats) const {
  for (size_t i = 0; i < entries_.size(); ++i) {
    if (matches(entries_[i], x, y, scale_out, rows, dim, cfg, scale, repeats))
      return true;
  }
  return false;
}

void GraphExecCache::evict_one() {
  if (entries_.empty())
    return;
  size_t victim = 0;
  for (size_t i = 1; i < entries_.size(); ++i) {
    if (entries_[i].stamp < entries_[victim].stamp)
      victim = i;
  }
  cudaGraphExecDestroy(entries_[victim].exec);
  cudaGraphDestroy(entries_[victim].graph);
  entries_.erase(entries_.begin() + (std::ptrdiff_t)victim);
  ++evictions_;
}

cudaGraphExec_t GraphExecCache::get_or_capture(const void *x, void *y, float *scale_out,
                                               int64_t rows, int64_t dim, const Config &cfg,
                                               float scale, int repeats) {
  for (size_t i = 0; i < entries_.size(); ++i) {
    Entry &e = entries_[i];
    if (matches(e, x, y, scale_out, rows, dim, cfg, scale, repeats)) {
      e.stamp = ++stamp_;
      ++hits_;
      return e.exec;
    }
  }
  ++misses_;
  if (repeats < 1)
    fail("GraphExecCache requires repeats >= 1");
  if (rows <= 0)
    fail("GraphExecCache requires rows > 0");
  if (dim < 32 || dim > 16384)
    fail("dim outside the supported range (32..16384)");
  (void)plan(rows, dim, cfg); // rejects a bad dim / threads_per_row before capture
  if (cfg.quant != QuantKind::kNone) {
    if (cfg.dtype == DType::kF32)
      fail("fused quantisation supports fp16 / bf16 inputs");
    if (dim < 32 || dim > 1024)
      fail("dim outside the fused quantisation range (32..1024)");
    if (scale_out == nullptr)
      fail("a quantised workload needs a scale_out buffer");
  }
  if (stream_ == nullptr) {
    if (cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking) != cudaSuccess) {
      fail("cudaStreamCreateWithFlags failed");
    }
  }
  Entry e{};
  e.x = x;
  e.y = y;
  e.scale_out = scale_out;
  e.rows = rows;
  e.dim = dim;
  e.cfg = cfg;
  e.scale = scale;
  e.repeats = repeats;
  e.stamp = ++stamp_;
  record_workload(stream_, x, y, scale_out, rows, dim, cfg, scale, repeats, &e.graph);
  if (graph_instantiate(&e.exec, e.graph) != cudaSuccess)
    fail("cudaGraphInstantiate failed");
  ++captures_;
  while ((int)entries_.size() >= capacity_)
    evict_one();
  entries_.push_back(e);
  return e.exec;
}

void GraphExecCache::launch(const void *x, void *y, float *scale_out, int64_t rows, int64_t dim,
                            const Config &cfg, float scale, int repeats, cudaStream_t stream) {
  cudaGraphExec_t exec = get_or_capture(x, y, scale_out, rows, dim, cfg, scale, repeats);
  if (cudaGraphLaunch(exec, stream != nullptr ? stream : stream_) != cudaSuccess) {
    fail("cudaGraphLaunch failed");
  }
}

void GraphExecCache::clear() {
  for (size_t i = 0; i < entries_.size(); ++i) {
    cudaGraphExecDestroy(entries_[i].exec);
    cudaGraphDestroy(entries_[i].graph);
  }
  entries_.clear();
  if (stream_ != nullptr) {
    cudaStreamDestroy(stream_);
    stream_ = nullptr;
  }
}

// -----------------------------------------------------------------------------
//  Rotation fused into the GEMM's A-tile load
// -----------------------------------------------------------------------------
namespace {

template <typename T, int kLogK, bool kFuse>
void gemm_launch(const void *a, const void *b, void *c, int64_t m, int64_t n, const GemmConfig &cfg,
                 cudaStream_t stream) {
  // Two tile heights exist.  The tall one holds a whole row of A plus a 32-row
  // micro-tile per thread, which needs an opt-in shared-memory carve-out above
  // 48 KiB once K = 1024; the short one costs half the tile and fits two blocks
  // per SM instead of one.  k == 512 is the crossover.
  const bool wide =
      (cfg.block_m > 0) ? (cfg.block_m >= 32) : (gemm_smem_bytes<kLogK, 32>() <= 48 * 1024);
  const int block_m = wide ? 32 : 16;
  const int nsplit = cfg.nsplit > 0 ? cfg.nsplit : 1;
  const size_t smem = wide ? gemm_smem_bytes<kLogK, 32>() : gemm_smem_bytes<kLogK, 16>();
  const dim3 grid((unsigned)((m + block_m - 1) / block_m), (unsigned)nsplit);
  const dim3 block(kGemmTx, kGemmTy);
  auto *kern = wide ? fhwt_gemm_kernel<T, Native2<T>, kLogK, 32, kFuse>
                    : fhwt_gemm_kernel<T, Native2<T>, kLogK, 16, kFuse>;
  if (smem > 48 * 1024 && cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                               (int)smem) != cudaSuccess) {
    fail("cannot raise the shared-memory limit for the fused GEMM");
  }
  kern<<<grid, block, smem, stream>>>(reinterpret_cast<const T *>(a),
                                      reinterpret_cast<const T *>(b), reinterpret_cast<T *>(c),
                                      (int)m, (int)n, cfg.scale);
}

template <typename T, int kLogK>
void gemm_dispatch_m(const void *a, const void *b, void *c, int64_t m, int64_t n,
                     const GemmConfig &cfg, cudaStream_t stream) {
  if (cfg.fuse_rotation) {
    gemm_launch<T, kLogK, true>(a, b, c, m, n, cfg, stream);
  } else {
    gemm_launch<T, kLogK, false>(a, b, c, m, n, cfg, stream);
  }
}

template <typename T>
void gemm_dispatch_k(const void *a, const void *b, void *c, int64_t m, int64_t n, int logK,
                     const GemmConfig &cfg, cudaStream_t stream) {
  switch (logK) {
  case 7:
    return gemm_dispatch_m<T, 7>(a, b, c, m, n, cfg, stream);
  case 8:
    return gemm_dispatch_m<T, 8>(a, b, c, m, n, cfg, stream);
  case 9:
    return gemm_dispatch_m<T, 9>(a, b, c, m, n, cfg, stream);
  case 10:
    return gemm_dispatch_m<T, 10>(a, b, c, m, n, cfg, stream);
  default:
    fail("hadamard_gemm requires K = 2^p with 7 <= p <= 10");
  }
}

} // namespace

void hadamard_gemm(const void *a, const void *b, void *c, int64_t m, int64_t n, int64_t k,
                   const GemmConfig &cfg, cudaStream_t stream) {
  if (m <= 0 || n <= 0 || k <= 0)
    return;
  if (cfg.dtype == DType::kF32)
    fail("hadamard_gemm supports fp16 / bf16 inputs");
  if (m > 2147483647LL || n > 2147483647LL)
    fail("hadamard_gemm: M or N exceeds INT_MAX");
  const int logK = ilog2_exact(k);
  if (logK < 7 || logK > 10)
    fail("hadamard_gemm requires K = 2^p with 7 <= p <= 10");
  if (cfg.dtype == DType::kF16) {
    gemm_dispatch_k<__half>(a, b, c, m, n, logK, cfg, stream);
  } else {
    gemm_dispatch_k<__nv_bfloat16>(a, b, c, m, n, logK, cfg, stream);
  }
}

} // namespace fhwt
