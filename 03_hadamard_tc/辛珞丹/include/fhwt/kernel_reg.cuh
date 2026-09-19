// =============================================================================
//  kernel_reg.cuh -- register + warp-shuffle FWHT (the primary CUDA path).
//
//  Instead of the classic "one thread block per row, shared-memory butterfly"
//  layout (used by Tri Dao's fast_hadamard_transform), this kernel assigns one
//  *row group* of kThreadsPerRow lanes to every row and keeps the whole
//  butterfly in registers and in the shuffle network:
//
//      bit 0 .. log2(EPT)-1   -> lane-local  (in-register butterflies)
//      bit log2(EPT) .. log2D -> cross-lane  (__shfl_xor_sync, width = TPR)
//
//  Consequences:
//    * no shared memory at all  -> no bank conflicts, no __syncthreads()
//    * EPT contiguous elements per lane -> one or two 128-bit global accesses
//    * every lane holds 2 elements per 32-bit register (fp16/bf16), halving
//      both the ALU instruction count and the shuffle count versus an fp32
//      accumulator
//    * kRowsPerBlock rows per block keeps small head dims (64/128) from
//      degenerating into 8- or 16-thread blocks, which is what limits
//      occupancy in the reference implementation
//
//  The butterfly (a,b) -> (a+b, a-b) is realised with two instructions for a
//  packed pair:  v' = fma(v, (1,-1), swap(v))  (PRMT + HFMA2).
//  Cross-lane stages use the branch-free form  v' = fma(v, +-1, partner),
//  where the sign is +1 for the low lane and -1 for the high lane of the pair.
// =============================================================================
#pragma once

#include "common.cuh"
#include "types.cuh"

namespace fhwt {

// -----------------------------------------------------------------------------
//  Vectorised contiguous load / store of kEPT elements (kEPT >= 2, even).
//  The pointer is 16B aligned whenever kEPT * sizeof(T) is a multiple of 16.
// -----------------------------------------------------------------------------
template <int kEPT, typename T, typename Acc>
__device__ __forceinline__ void load_row_vec(const T *__restrict__ p,
                                             typename Acc::vec_t v[kEPT / 2]) {
  constexpr int kBytes = kEPT * sizeof(T);
  if constexpr (sizeof(T) == 2) {
    if constexpr (kBytes % 16 == 0) {
#pragma unroll
      for (int i = 0; i < kBytes / 16; ++i) {
        uint4 r = reinterpret_cast<const uint4 *>(p)[i];
        v[4 * i + 0] = Acc::from_word(r.x);
        v[4 * i + 1] = Acc::from_word(r.y);
        v[4 * i + 2] = Acc::from_word(r.z);
        v[4 * i + 3] = Acc::from_word(r.w);
      }
    } else if constexpr (kBytes % 8 == 0) {
#pragma unroll
      for (int i = 0; i < kBytes / 8; ++i) {
        uint2 r = reinterpret_cast<const uint2 *>(p)[i];
        v[2 * i + 0] = Acc::from_word(r.x);
        v[2 * i + 1] = Acc::from_word(r.y);
      }
    } else if constexpr (kBytes % 4 == 0) {
#pragma unroll
      for (int i = 0; i < kBytes / 4; ++i) {
        v[i] = Acc::from_word(reinterpret_cast<const uint32_t *>(p)[i]);
      }
    } else {
#pragma unroll
      for (int i = 0; i < kBytes / 2; ++i) {
        v[i] = Acc::from_word(reinterpret_cast<const uint16_t *>(p)[i]);
      }
    }
  } else { // fp32 data: Acc::vec_t is float2, i.e. exactly two elements.
    if constexpr (kBytes % 16 == 0) {
#pragma unroll
      for (int i = 0; i < kBytes / 16; ++i) {
        float4 r = reinterpret_cast<const float4 *>(p)[i];
        v[2 * i + 0] = make_float2(r.x, r.y);
        v[2 * i + 1] = make_float2(r.z, r.w);
      }
    } else {
#pragma unroll
      for (int i = 0; i < kBytes / 8; ++i) {
        v[i] = reinterpret_cast<const float2 *>(p)[i];
      }
    }
  }
}

// scale == 1.f (the caller's fast path) skips the multiply entirely.
template <int kEPT, typename T, typename Acc>
__device__ __forceinline__ void store_row_vec(T *__restrict__ p,
                                              const typename Acc::vec_t v[kEPT / 2], float scale) {
  constexpr int kBytes = kEPT * sizeof(T);
  const bool unit = (scale == 1.0f);
  if constexpr (sizeof(T) == 2) {
    if constexpr (kBytes % 16 == 0) {
#pragma unroll
      for (int i = 0; i < kBytes / 16; ++i) {
        uint4 r;
        r.x = Acc::to_word(unit ? v[4 * i + 0] : Acc::scale(v[4 * i + 0], scale));
        r.y = Acc::to_word(unit ? v[4 * i + 1] : Acc::scale(v[4 * i + 1], scale));
        r.z = Acc::to_word(unit ? v[4 * i + 2] : Acc::scale(v[4 * i + 2], scale));
        r.w = Acc::to_word(unit ? v[4 * i + 3] : Acc::scale(v[4 * i + 3], scale));
        reinterpret_cast<uint4 *>(p)[i] = r;
      }
    } else if constexpr (kBytes % 8 == 0) {
#pragma unroll
      for (int i = 0; i < kBytes / 8; ++i) {
        uint2 r;
        r.x = Acc::to_word(unit ? v[2 * i + 0] : Acc::scale(v[2 * i + 0], scale));
        r.y = Acc::to_word(unit ? v[2 * i + 1] : Acc::scale(v[2 * i + 1], scale));
        reinterpret_cast<uint2 *>(p)[i] = r;
      }
    } else if constexpr (kBytes % 4 == 0) {
#pragma unroll
      for (int i = 0; i < kBytes / 4; ++i) {
        reinterpret_cast<uint32_t *>(p)[i] = Acc::to_word(unit ? v[i] : Acc::scale(v[i], scale));
      }
    } else {
#pragma unroll
      for (int i = 0; i < kBytes / 2; ++i) {
        reinterpret_cast<uint16_t *>(p)[i] = Acc::to_word(unit ? v[i] : Acc::scale(v[i], scale));
      }
    }
  } else {
    if constexpr (kBytes % 16 == 0) {
#pragma unroll
      for (int i = 0; i < kBytes / 16; ++i) {
        float4 r;
        r.x = (unit ? v[2 * i + 0] : Acc::scale(v[2 * i + 0], scale)).x;
        r.y = (unit ? v[2 * i + 0] : Acc::scale(v[2 * i + 0], scale)).y;
        r.z = (unit ? v[2 * i + 1] : Acc::scale(v[2 * i + 1], scale)).x;
        r.w = (unit ? v[2 * i + 1] : Acc::scale(v[2 * i + 1], scale)).y;
        reinterpret_cast<float4 *>(p)[i] = r;
      }
    } else {
#pragma unroll
      for (int i = 0; i < kBytes / 8; ++i) {
        reinterpret_cast<float2 *>(p)[i] = unit ? v[i] : Acc::scale(v[i], scale);
      }
    }
  }
}
// -----------------------------------------------------------------------------
//  The complete butterfly, shared by the standalone kernel and the fused kernel.
// -----------------------------------------------------------------------------
template <typename T, typename Acc, int kLogD, int kLogTPR>
__device__ __forceinline__ void
fhwt_transform_vec(typename Acc::vec_t v[(1 << (kLogD - kLogTPR)) / 2], int lane,
                   unsigned group_mask) {
  constexpr int kTPR = 1 << kLogTPR;
  constexpr int kNReg = (1 << (kLogD - kLogTPR)) / 2;
  constexpr int kLogEPT = kLogD - kLogTPR;

  // --- lane-local stages: bits [0, kLogEPT) of the row index ---------------
  if constexpr (kLogEPT >= 1) {
#pragma unroll
    for (int i = 0; i < kNReg; ++i)
      v[i] = Acc::butterfly01(v[i]); // bit 0
  }
  // Bit s lives at distance 2^(s-1) in vec_t units (bit 0 is the lane swap that
  // butterfly01 already handled, so the loop starts at s = 1).
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

  // --- cross-lane stages: bits [kLogEPT, kLogD) ----------------------------
#pragma unroll
  for (int s = kLogEPT; s < kLogD; ++s) {
    const int step = 1 << (s - kLogEPT);
    const float sg = ((lane & step) != 0) ? -1.0f : 1.0f;
    const typename Acc::vec_t sgn = Acc::from_scalar(sg);
#pragma unroll
    for (int i = 0; i < kNReg; ++i) {
      typename Acc::vec_t p = Acc::shfl_xor(v[i], step, kTPR, group_mask);
      v[i] = Acc::mul_add(v[i], sgn, p);
    }
  }
}

// -----------------------------------------------------------------------------
//  Standalone kernel: y = scale * H_last_dim(x)
// -----------------------------------------------------------------------------
template <typename T, typename Acc, int kLogD, int kLogTPR>
__global__ void fhwt_reg_kernel(const T *__restrict__ x, T *__restrict__ y, int64_t rows,
                                float scale) {
  constexpr int kTPR = 1 << kLogTPR;
  constexpr int kEPT = 1 << (kLogD - kLogTPR);
  constexpr int kNReg = kEPT / 2;

  const int lane = threadIdx.x & (kTPR - 1);
  // Shuffle mask of this row group, as lane bits *inside the warp*.
  //
  // The first version shifted by (threadIdx.x & ~(kTPR - 1)), i.e. by the thread's
  // position in the block.  From the second warp on that shift is >= 32: undefined
  // in C++, and on the GPU it simply produced mask = 0, which __shfl_xor_sync does
  // not allow either.  The results were still right because the warp is fully
  // converged here, and the tests happened to launch 32-thread blocks for
  // tpr < 32, so nothing ever failed.  Found while re-reading the kernels before
  // submission (report 5.10); the default dim 32/64 geometry (256-thread blocks)
  // hit it on every launch.
  const unsigned group_mask =
      (kTPR == 32) ? 0xFFFFFFFFu : (((1u << kTPR) - 1u) << ((threadIdx.x & 31) & ~(kTPR - 1)));
  // rows per block is a launch parameter: blockDim.x / kTPR.
  const int rows_per_block = (int)blockDim.x >> kLogTPR;
  const int64_t stride = (int64_t)gridDim.x * rows_per_block;
  const int64_t first = (int64_t)blockIdx.x * rows_per_block + (threadIdx.x >> kLogTPR);

  for (int64_t row = first; row < rows; row += stride) {
    const T *__restrict__ src = x + (row << kLogD) + lane * kEPT;
    T *__restrict__ dst = y + (row << kLogD) + lane * kEPT;
    typename Acc::vec_t v[kNReg];
    load_row_vec<kEPT, T, Acc>(src, v);
    fhwt_transform_vec<T, Acc, kLogD, kLogTPR>(v, lane, group_mask);
    store_row_vec<kEPT, T, Acc>(dst, v, scale);
  }
}

} // namespace fhwt