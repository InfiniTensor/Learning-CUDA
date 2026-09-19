// =============================================================================
//  kernel_gemm.cuh -- Hadamard rotation fused into the A-tile load of a GEMM.
//
//  Why fuse
//  --------
//  A QuaRot-style layer rotates its activation and then multiplies it:
//
//      A_rot = A H_k        one full read of A, one full write of A_rot
//      C     = A_rot B^T    one more read of A_rot
//
//  A is moved three times.  The GEMM has to read A anyway, so if the rotation
//  happens on the way into the shared tile, A is moved twice and the A_rot
//  tensor never exists in DRAM: 2*M*K*sizeof(T) bytes and one kernel launch
//  saved, and nothing else about the arithmetic changes.
//
//  Layout of the A tile
//  --------------------
//  The butterfly hands lane l the kUnits = K/64 two-element units
//  [l*kUnits, (l+1)*kUnits) of a row (a warp holds a whole row: 32 lanes x
//  kEPT = K/32 elements).  Storing those units at their natural slot makes
//  every lane of the warp hit the same bank on every store -- the lanes differ
//  by kUnits words, which is a multiple of 32 for kUnits >= 16 -- so the store
//  serialises 16-way.  The tile is therefore written through the bijection
//
//      slot(u) = (u mod kUnits) * 32 + u / kUnits
//
//  which turns "lane l writes the words l*kUnits, l*kUnits+1, ..." into "lane l
//  writes the words l, l+32, l+64, ...": 32 distinct banks, no conflict.  The
//  inner loop reads the tile through the same bijection, which costs nothing --
//  a warp reads one address per row of A (broadcast) and the slot index is
//  uniform across the warp, so it folds into the LDS immediate.  Store and read
//  agree, so the GEMM sums exactly the same products in a different order,
//  which is why the fused result stays bit-identical to the unfused pipeline.
//
//  The unfused entry point runs the *same* kernel with kFuse = false: slot()
//  becomes the identity and the inner loop reads a pre-rotated [M, K] tensor.
//  Comparing the two isolates the fusion from the quality of the GEMM core.
// =============================================================================
#pragma once

#include "common.cuh"
#include <cstring>
#include "kernel_reg.cuh"
#include "types.cuh"

namespace fhwt {

// -----------------------------------------------------------------------------
//  Tile geometry.  256 threads as (32, 8): threadIdx.x walks the columns of C
//  and threadIdx.y the rows, so a warp's 32 lanes always share one row index.
//  The block computes kBlockM x 128 outputs, each thread owning a
//  (kBlockM/8) x 4 micro-tile.
// -----------------------------------------------------------------------------
constexpr int kGemmTx = 32;
constexpr int kGemmTy = 8;
constexpr int kGemmTileN = 4;
constexpr int kGemmChunkK = 32;
constexpr int kGemmBlockN = kGemmTx * kGemmTileN; // 128
constexpr int kGemmStrideB = kGemmBlockN + 4;     // pad keeps rows 8B aligned
constexpr int kGemmLogTPR = 5;                    // one warp per row of A

// Shared bytes of one (kBlockM, K) tile plus its B chunk, in bytes.
template <int kLogK, int kBlockM> constexpr size_t gemm_smem_bytes() {
  return (size_t)kBlockM * (size_t)(1 << kLogK) * 2 + (size_t)kGemmChunkK * kGemmStrideB * 2;
}

// The bank-conflict-free slot of unit u (see the file comment).
template <int kLogUnits, bool kFuse> __device__ __forceinline__ int gemm_slot(int u) {
  if constexpr (kFuse) {
    constexpr int kMask = (1 << kLogUnits) - 1;
    return ((u & kMask) << 5) | (u >> kLogUnits);
  } else {
    return u;
  }
}

// Writes one transformed row, held as lane-local vec_t units, into the tile.
template <int kLogK, typename T, typename Acc, bool kFuse>
__device__ __forceinline__ void gemm_store_row(T *__restrict__ row, const typename Acc::vec_t *v,
                                               int lane, float scale) {
  constexpr int kUnits = ((1 << kLogK) >> kGemmLogTPR) / 2;
  const bool unit = (scale == 1.0f);
#pragma unroll
  for (int j = 0; j < kUnits; ++j) {
    const typename Acc::vec_t val = unit ? v[j] : Acc::scale(v[j], scale);
    reinterpret_cast<typename Acc::vec_t *>(row)[gemm_slot<kLogK - 6, kFuse>(lane * kUnits + j)] =
        val;
  }
}

// Fills one row of the tile: rotate when kFuse, plain copy otherwise, zeros for
// the out-of-range rows of the last M tile (src == nullptr).
template <typename T, typename Acc, int kLogK, bool kFuse>
__device__ __forceinline__ void gemm_fill_row(T *__restrict__ dst, const T *__restrict__ src,
                                              int lane, float scale) {
  constexpr int kPT = (1 << kLogK) >> kGemmLogTPR;
  typename Acc::vec_t v[kPT / 2];
  if (src != nullptr) {
    load_row_vec<kPT, T, Acc>(src, v);
    if constexpr (kFuse)
      fhwt_transform_vec<T, Acc, kLogK, kGemmLogTPR>(v, lane, 0xFFFFFFFFu);
  } else {
#pragma unroll
    for (int i = 0; i < kPT / 2; ++i)
      v[i] = Acc::from_word(0u);
  }
  gemm_store_row<kLogK, T, Acc, kFuse>(dst, v, lane, kFuse ? scale : 1.0f);
}

// -----------------------------------------------------------------------------
//  C[m, n] = (A H_k)[m, :] . B[n, :]^T
//  a : [M, K] row-major, rotated on load when kFuse
//  b : [N, K] row-major
//  c : [M, N] row-major, fp32 accumulation rounded to T on store
// -----------------------------------------------------------------------------
template <typename T, typename Acc, int kLogK, int kBlockM, bool kFuse>
__global__ void __launch_bounds__(kGemmTx *kGemmTy)
    fhwt_gemm_kernel(const T *__restrict__ a, const T *__restrict__ b, T *__restrict__ c, int M,
                     int N, float scale) {
  constexpr int kK = 1 << kLogK;
  constexpr int kTM = kBlockM / kGemmTy; // rows of C per thread
  constexpr int kThreads = kGemmTx * kGemmTy;
  constexpr int kLoadsPerThread = kGemmChunkK * kGemmBlockN / 8 / kThreads;

  extern __shared__ __align__(16) unsigned char gemm_smem_raw[];
  T *a_s = reinterpret_cast<T *>(gemm_smem_raw);
  T *b_s = a_s + (size_t)kBlockM * kK;

  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int tid = ty * kGemmTx + tx;
  const int m0 = (int)blockIdx.x * kBlockM;
  // The epilogue stores 4 columns at once, which is only addressable when every
  // row of C starts on an 8-byte boundary.  Odd N falls back to scalar stores.
  const bool vec_store = ((N & 3) == 0);

  // ---- phase 1: build the A tile, rotating on the way in --------------------
  // Tile row r*kGemmTy + ty holds global row m0 + r*kGemmTy + ty, which is the
  // same row this thread reads back in phase 2 -- the two mappings have to
  // agree or the block would compute a permuted C.
#pragma unroll
  for (int i = 0; i < kTM; ++i) {
    const int row = ty * kTM + i;
    const int gm = m0 + row;
    const T *src = (gm < M) ? a + (int64_t)gm * kK + tx * (kK >> kGemmLogTPR) : nullptr;
    gemm_fill_row<T, Acc, kLogK, kFuse>(a_s + (size_t)row * kK, src, tx, scale);
  }

  // ---- phase 2: every N tile this block owns -------------------------------
  const int n_chunk = (N + (int)gridDim.y - 1) / (int)gridDim.y;
  const int n_beg = (int)blockIdx.y * n_chunk;
  const int n_end = (n_beg + n_chunk < N) ? (n_beg + n_chunk) : N;

  for (int n0 = n_beg; n0 < n_end; n0 += kGemmBlockN) {
    float acc[kTM][kGemmTileN];
#pragma unroll
    for (int i = 0; i < kTM; ++i)
#pragma unroll
      for (int j = 0; j < kGemmTileN; ++j)
        acc[i][j] = 0.0f;

#pragma unroll 4
    for (int k0 = 0; k0 < kK; k0 += kGemmChunkK) {
      __syncthreads();
#pragma unroll
      for (int it = 0; it < kLoadsPerThread; ++it) {
        const int i = it * kThreads + tid;
        const int nn = (i * 8) / kGemmChunkK;
        const int kk = (i * 8) % kGemmChunkK;
        const int gn = n0 + nn;
        uint4 raw = make_uint4(0u, 0u, 0u, 0u);
        if (gn < N)
          raw = *reinterpret_cast<const uint4 *>(b + (int64_t)gn * kK + k0 + kk);
        // The eight halves are peeled out of the vector through memcpy rather than
        // by aliasing a local uint4 as `const T*`.  That reinterpret_cast is a
        // strict-aliasing violation (the local's declared type is uint4) which mcc
        // miscompiles: at K = 128 it fed stale halves into the B tile, and rows of
        // C came back as inf.  memcpy is the defined way to reinterpret and folds
        // back into register moves, so the generated code is the same on nvcc.
        const uint32_t w4[4] = {raw.x, raw.y, raw.z, raw.w};
        T shv[8];
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          const uint16_t hbits = (uint16_t)(w4[e >> 1] >> ((e & 1) * 16));
          memcpy(&shv[e], &hbits, sizeof(uint16_t));
        }
#pragma unroll
        for (int e = 0; e < 8; ++e)
          b_s[(kk + e) * kGemmStrideB + nn] = shv[e];
      }
      __syncthreads();

#pragma unroll
      for (int kk = 0; kk < kGemmChunkK; kk += 2) {
        const int slot = gemm_slot<kLogK - 6, kFuse>((k0 + kk) >> 1);
        float2 av[kTM];
#pragma unroll
        for (int i = 0; i < kTM; ++i) {
          av[i] = Acc::to_float2(reinterpret_cast<const typename Acc::vec_t *>(
              a_s + (size_t)(ty * kTM + i) * kK)[slot]);
        }
        const uint2 r0 =
            *reinterpret_cast<const uint2 *>(&b_s[kk * kGemmStrideB + tx * kGemmTileN]);
        const uint2 r1 =
            *reinterpret_cast<const uint2 *>(&b_s[(kk + 1) * kGemmStrideB + tx * kGemmTileN]);
        const float2 u0 = Acc::to_float2(Acc::from_word(r0.x));
        const float2 u1 = Acc::to_float2(Acc::from_word(r0.y));
        const float2 w0 = Acc::to_float2(Acc::from_word(r1.x));
        const float2 w1 = Acc::to_float2(Acc::from_word(r1.y));
        const float bk[2][kGemmTileN] = {{u0.x, u0.y, u1.x, u1.y}, {w0.x, w0.y, w1.x, w1.y}};
#pragma unroll
        for (int i = 0; i < kTM; ++i)
#pragma unroll
          for (int j = 0; j < kGemmTileN; ++j) {
            acc[i][j] = fmaf(av[i].x, bk[0][j], acc[i][j]);
            acc[i][j] = fmaf(av[i].y, bk[1][j], acc[i][j]);
          }
      }
    }

    // ---- epilogue ----------------------------------------------------------
#pragma unroll
    for (int i = 0; i < kTM; ++i) {
      const int gm = m0 + ty * kTM + i;
      if (gm >= M)
        break;
      const int gn = n0 + tx * kGemmTileN;
      T *dst = c + (int64_t)gm * N + gn;
      if (vec_store && gn + kGemmTileN <= N) {
        uint2 out;
        out.x = Acc::to_word(Acc::from_floats(acc[i][0], acc[i][1]));
        out.y = Acc::to_word(Acc::from_floats(acc[i][2], acc[i][3]));
        *reinterpret_cast<uint2 *>(dst) = out;
      } else {
#pragma unroll
        for (int j = 0; j < kGemmTileN; ++j) {
          if (gn + j < N)
            dst[j] = ElemTraits<T>::from_float(acc[i][j]);
        }
      }
    }
  }
}

} // namespace fhwt
