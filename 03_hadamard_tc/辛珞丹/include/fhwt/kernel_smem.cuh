// =============================================================================
//  kernel_smem.cuh -- two-level (register + shared memory) FWHT for large head
//  dimensions.
//
//  Why a second kernel
//  -------------------
//  The register/shuffle kernel (kernel_reg.cuh) keeps the *whole* butterfly in
//  registers and in the warp shuffle network, which is optimal while the row
//  still fits into one warp's register file:
//
//      dim <= 32 lanes * EPT(max) = 32 * 32 = 1024 elements
//
//  Beyond that the row has to be split, and the natural split is a two level
//  divide and conquer:
//
//    level 1 (intra-chunk)  bits [0, log2(CHUNK)) of the row index stay inside
//                           one warp: register-local butterflies for the low
//                           EPT bits, __shfl_xor_sync for the TPR bits above.
//    level 2 (inter-chunk)  bits [log2(CHUNK), log2(dim)) become a stride-2^s
//                           butterfly over a shared-memory tile holding the row.
//
//  Each chunk is transformed exactly once, so the extra shared-memory traffic
//  is 2 * dim * (log2(dim) - log2(CHUNK)) element accesses against dim elements
//  of global traffic.  With CHUNK = 256 and dim = 4096 those are 4 extra shared
//  round trips -- a few percent of the DRAM time, because shared memory
//  sustains roughly 20x the bandwidth of this GPU's GDDR7.
//
//  Bank conflicts
//  --------------
//  The inter-chunk stages use the bit-split index map
//
//      i = (j / d) * 2d + (j % d)          d = 2^s
//
//  so consecutive j map to consecutive addresses (coalesced, conflict free)
//  and the only discontinuity sits at a 2d boundary, i.e. at least CHUNK
//  elements away.  Since the low bits were already resolved in registers every
//  shared stage has d >= CHUNK >= 32, which is exactly the condition for the
//  split map to be conflict free.
//
//  In-place operation
//  ------------------
//  This kernel and the register kernel are both in-place safe: a lane owns a
//  contiguous element range, reads it, and writes the transformed value back to
//  the same addresses, so x == y is allowed.
// =============================================================================
#pragma once

#include "common.cuh"
#include "kernel_reg.cuh"
#include "types.cuh"

namespace fhwt {

// -----------------------------------------------------------------------------
//  Load / store two *consecutive* elements as one Acc::vec_t (a 32-bit word for
//  the 16-bit element types, a float2 for fp32).
// -----------------------------------------------------------------------------
template <typename T, typename Acc>
__device__ __forceinline__ typename Acc::vec_t load_pair(const T *p) {
  if constexpr (sizeof(T) == 2) {
    return Acc::from_word(*reinterpret_cast<const uint32_t *>(p));
  } else {
    return *reinterpret_cast<const float2 *>(p);
  }
}

template <typename T, typename Acc>
__device__ __forceinline__ void store_pair(T *p, typename Acc::vec_t v) {
  if constexpr (sizeof(T) == 2) {
    *reinterpret_cast<uint32_t *>(p) = Acc::to_word(v);
  } else {
    *reinterpret_cast<float2 *>(p) = v;
  }
}

// -----------------------------------------------------------------------------
//  Intra-chunk transform: EPT contiguous elements held by one lane, TPR lanes
//  per chunk.  Identical butterfly network to kernel_reg.cuh, reused verbatim
//  through Acc::butterfly01 / Acc::shfl_xor so both kernels stay numerically
//  interchangeable.
// -----------------------------------------------------------------------------
template <typename T, typename Acc, int kLogEPT, int kLogTPR>
__device__ __forceinline__ void smem_intra_chunk(typename Acc::vec_t v[1 << (kLogEPT - 1)],
                                                 int lane, unsigned group_mask) {
  constexpr int kNReg = 1 << (kLogEPT - 1);
  if constexpr (kLogEPT >= 1) {
#pragma unroll
    for (int i = 0; i < kNReg; ++i)
      v[i] = Acc::butterfly01(v[i]);
  }
#pragma unroll
  for (int s = 1; s < kLogEPT; ++s) {
    const int stride = 1 << (s - 1);
#pragma unroll
    for (int i = 0; i < kNReg; ++i) {
      if ((i & stride) == 0) {
        typename Acc::vec_t a = v[i];
        typename Acc::vec_t b = v[i + stride];
        v[i] = Acc::add(a, b);
        v[i + stride] = Acc::sub(a, b);
      }
    }
  }
#pragma unroll
  for (int s = kLogEPT; s < kLogEPT + kLogTPR; ++s) {
    const int step = 1 << (s - kLogEPT);
    const float sg = ((lane & step) != 0) ? -1.0f : 1.0f;
    const typename Acc::vec_t sgn = Acc::from_scalar(sg);
#pragma unroll
    for (int i = 0; i < kNReg; ++i) {
      typename Acc::vec_t p = Acc::shfl_xor(v[i], step, 1 << kLogTPR, group_mask);
      v[i] = Acc::mul_add(v[i], sgn, p);
    }
  }
}

// -----------------------------------------------------------------------------
//  One inter-chunk stage of distance d = 2^s, processed two elements per thread.
//      m enumerates the kDim/4 "pair slots"; the bit-split map turns m into the
//      element index i with bit s clear, and i+1 (same run) is handled too.
// -----------------------------------------------------------------------------
template <typename T, typename Acc, int kD>
__device__ __forceinline__ void smem_inter_stage(T *__restrict__ tile, int s, int tid,
                                                 int nthreads) {
  const int d = 1 << s;
  const int nslot = kD >> 2; // kD/4 pair slots
  for (int m = tid; m < nslot; m += nthreads) {
    const int i = (((m >> (s - 1)) << (s + 1)) | ((m & ((d >> 1) - 1)) << 1));
    typename Acc::vec_t a = load_pair<T, Acc>(tile + i);
    typename Acc::vec_t b = load_pair<T, Acc>(tile + i + d);
    store_pair<T, Acc>(tile + i, Acc::add(a, b));
    store_pair<T, Acc>(tile + i + d, Acc::sub(a, b));
  }
}

// -----------------------------------------------------------------------------
//  fhwt_smem_kernel
//    blockDim.x = warps * TPR threads; every warp owns one CHUNK of the row and
//    the block iterates over the row in strides of (blockDim.x / TPR) chunks.
//    Dynamic shared memory: one row of kDim elements.
// -----------------------------------------------------------------------------
template <typename T, typename Acc, int kLogD, int kLogEPT, int kLogTPR>
__global__ void fhwt_smem_kernel(const T *__restrict__ x, T *__restrict__ y, int64_t rows,
                                 float scale) {
  constexpr int kDim = 1 << kLogD;
  constexpr int kTPR = 1 << kLogTPR;
  constexpr int kEPT = 1 << kLogEPT;
  constexpr int kChunk = kTPR * kEPT;
  constexpr int kLogChunk = kLogTPR + kLogEPT;
  constexpr int kNChunk = kDim / kChunk;
  constexpr int kNReg = kEPT / 2;

  extern __shared__ __align__(16) unsigned char smem_raw[];
  T *tile = reinterpret_cast<T *>(smem_raw);

  const int lane = threadIdx.x & (kTPR - 1);
  const int width = blockDim.x >> kLogTPR;
  const int cw = threadIdx.x >> kLogTPR;
  // Lane-in-warp shift (report 5.10).  kSmemLogTPR is 5 today, so this branch is
  // not taken, but it had the same block-position bug as kernel_reg.cuh.
  const unsigned group_mask =
      (kTPR >= 32) ? 0xFFFFFFFFu : ((~0u >> (32 - kTPR)) << ((threadIdx.x & 31) & ~(kTPR - 1)));

  for (int64_t row = blockIdx.x; row < rows; row += gridDim.x) {
    const T *src = x + row * kDim;
    T *dst = y + row * kDim;

    // ---- level 1: the low log2(CHUNK) bits, entirely in registers --------
    for (int c = cw; c < kNChunk; c += width) {
      const int off = c * kChunk + lane * kEPT;
      typename Acc::vec_t v[kNReg];
      load_row_vec<kEPT, T, Acc>(src + off, v);
      smem_intra_chunk<T, Acc, kLogEPT, kLogTPR>(v, lane, group_mask);
      store_row_vec<kEPT, T, Acc>(tile + off, v, 1.0f);
    }
    __syncthreads();

    // ---- level 2: the high bits, over the shared tile --------------------
#pragma unroll
    for (int s = kLogChunk; s < kLogD; ++s) {
      smem_inter_stage<T, Acc, kDim>(tile, s, (int)threadIdx.x, (int)blockDim.x);
      __syncthreads();
    }

    // ---- write back ------------------------------------------------------
    for (int c = cw; c < kNChunk; c += width) {
      const int off = c * kChunk + lane * kEPT;
      typename Acc::vec_t v[kNReg];
      load_row_vec<kEPT, T, Acc>(tile + off, v);
      store_row_vec<kEPT, T, Acc>(dst + off, v, scale);
    }
    __syncthreads(); // the tile is reused by the next row
  }
}

} // namespace fhwt