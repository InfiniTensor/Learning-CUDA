// =============================================================================
//  kernel_quant.cuh -- fused Hadamard rotation + quantisation.
//
//  Motivation
//  ----------
//  QuaRot / SpinQuant / FlashInfer all rotate activations before quantising
//  them.  Running the rotation and the quantiser as two kernels costs four full
//  passes over the activation tensor:
//
//      rotate:    read 2 B + write 2 B
//      quantise:  read 2 B + write 1 B (fp8)          -> 7 B / element
//
//  Fusing them removes the intermediate tensor entirely:
//
//      fused:     read 2 B + write 1 B (fp8)          -> 3 B / element
//
//  i.e. 2.3x less DRAM traffic for FP8 output (2.6x for INT4: 6.5 B vs 2.5 B) plus one kernel
//  launch instead of two.  The per-token / per-block amax the quantiser needs is
//  produced by shuffle reductions over the row, so no extra pass over the data
//  is required.
//
//  Numerics: the fused kernel rounds the rotated values back to the storage type
//  T, which is exactly what the unfused pipeline sees after writing the rotated
//  tensor to DRAM.  Therefore the fused output equals the output of
//  "hadamard(); quantise();" bit for bit (verified by tests/test_fusion.cu).
// =============================================================================
#pragma once

#include "common.cuh"
#include "kernel_reg.cuh"
#include "quant_pack.cuh"
#include "types.cuh"

namespace fhwt {

// Rounds an accumulator value to the storage type T and back, reproducing the
// rounding performed when the rotated tensor is materialised in DRAM.
template <typename T, typename Acc>
__device__ __forceinline__ float round_to_storage(typename Acc::vec_t v, int which) {
  float f = which ? Acc::hi(v) : Acc::lo(v);
  if constexpr (sizeof(T) == 4) {
    return f;
  } else {
    return ElemTraits<T>::to_float(ElemTraits<T>::from_float(f));
  }
}

// -----------------------------------------------------------------------------
//  Fused kernel (also used with kDoTransform = false as the fast quantiser).
//    quant_kind     : kQuantFp8E4M3 / kQuantInt4 (uniform across the grid)
//    block_size_log : 0 -> one scale per row, else one scale per 2^bs columns
// -----------------------------------------------------------------------------
template <typename T, typename Acc, int kLogD, int kLogTPR, bool kDoTransform>
__global__ void fhwt_quant_kernel(const T *__restrict__ x, void *__restrict__ q_out,
                                  float *__restrict__ scale_out, int64_t rows, float scale,
                                  int quant_kind, int block_size_log) {
  constexpr int kTPR = 1 << kLogTPR;
  constexpr int kEPT = 1 << (kLogD - kLogTPR);
  constexpr int kNReg = kEPT / 2;
  constexpr int kDim = 1 << kLogD;
  static_assert(kEPT >= 8, "the packed store path requires at least 8 elements per lane");

  const int lane = threadIdx.x & (kTPR - 1);
  // Lane-in-warp shift, same fix as kernel_reg.cuh (report 5.10).  This kernel
  // needed it more: its default geometry is 8 threads/row x 16 rows = 128 threads,
  // so three of the four warps used to run with mask = 0 (dim 32..256; the
  // dispatcher picks 32 threads/row for dim 512/1024).
  const unsigned group_mask =
      (kTPR == 32) ? 0xFFFFFFFFu : (((1u << kTPR) - 1u) << ((threadIdx.x & 31) & ~(kTPR - 1)));
  const int rows_per_block = (int)blockDim.x >> kLogTPR;
  const int64_t stride = (int64_t)gridDim.x * rows_per_block;
  const int64_t first = (int64_t)blockIdx.x * rows_per_block + (threadIdx.x >> kLogTPR);

  const int red_width = (block_size_log > 0) ? (1 << (block_size_log - (kLogD - kLogTPR))) : kTPR;
  const int scales_per_row = (block_size_log > 0) ? (kDim >> block_size_log) : 1;
  const int scale_idx = (block_size_log > 0) ? ((lane * kEPT) >> block_size_log) : 0;

  for (int64_t row = first; row < rows; row += stride) {
    typename Acc::vec_t v[kNReg];
    load_row_vec<kEPT, T, Acc>(x + (row << kLogD) + lane * kEPT, v);
    if constexpr (kDoTransform)
      fhwt_transform_vec<T, Acc, kLogD, kLogTPR>(v, lane, group_mask);
    if (scale != 1.0f) {
#pragma unroll
      for (int i = 0; i < kNReg; ++i)
        v[i] = Acc::scale(v[i], scale);
    }

    // amax over this lane group, after rounding to the storage type.  The
    // rounded values are kept in `rv`: the packer below needs exactly these
    // floats, and rounding once here instead of once per use removes two
    // conversions per element (the amax loop and the packing loop used to round
    // the same value independently).
    typename Acc::vec_t rv[kNReg];
    float amax = 0.0f;
#pragma unroll
    for (int i = 0; i < kNReg; ++i) {
      rv[i] =
          Acc::from_floats(round_to_storage<T, Acc>(v[i], 0), round_to_storage<T, Acc>(v[i], 1));
      const float2 f = Acc::to_float2(rv[i]); // one conversion for both halves
      amax = fmaxf(amax, fmaxf(fabsf(f.x), fabsf(f.y)));
    }
    for (int o = red_width >> 1; o > 0; o >>= 1)
      amax = fmaxf(amax, __shfl_xor_sync(group_mask, amax, o, red_width));

    if (quant_kind == kQuantFp8E4M3) {
      const float sc = (amax > 0.0f) ? (amax / 448.0f) : 1.0f;
      const float inv = 1.0f / sc;
      uint8_t *dst = reinterpret_cast<uint8_t *>(q_out) + (row << kLogD) + lane * kEPT;
#pragma unroll
      for (int g = 0; g < kNReg / 4; ++g)
        store_fp8x8<Acc>(dst + 8 * g, rv + 4 * g, inv); // already rounded above
      if ((lane & (red_width - 1)) == 0)
        scale_out[row * scales_per_row + scale_idx] = sc;
    } else {
      const float sc = (amax > 0.0f) ? (amax / 7.0f) : 1.0f;
      const float inv = 1.0f / sc;
      uint8_t *dst = reinterpret_cast<uint8_t *>(q_out) + (row << (kLogD - 1)) + (lane * kEPT) / 2;
#pragma unroll
      for (int g = 0; g < kNReg / 4; ++g)
        store_int4x8<Acc>(dst + 4 * g, rv + 4 * g, inv); // already rounded above
      if ((lane & (red_width - 1)) == 0)
        scale_out[row * scales_per_row + scale_idx] = sc;
    }
  }
}

// -----------------------------------------------------------------------------
//  Standalone quantiser: "rotate then quantise" reference pipeline and the
//  baseline for the fusion speed-up measurement.  One warp per row, vectorised
//  128-bit loads, one smem atomicMax per 8 element chunk for the amax, then a
//  second (L2 resident) pass to quantise.  Written independently of the fused
//  kernel above.
// -----------------------------------------------------------------------------
// Element i of a 128-bit raw vector holding eight 16-bit elements.
template <typename T> __device__ __forceinline__ float elem_at(const uint4 &raw, int i) {
  Pair16<T> p;
  p.u = (&raw.x)[i >> 1];
  return ElemTraits<T>::to_float(p.h[i & 1]);
}

template <typename T>
__global__ void quantize_rows_kernel(const T *__restrict__ x, void *__restrict__ q_out,
                                     float *__restrict__ scale_out, int64_t rows, int dim_log,
                                     int quant_kind, int block_size_log) {
  // One amax scratch region per warp: four warps of a block quantise four
  // different rows and must not share their running maxima.
  extern __shared__ float bmax_all[];
  const int dim = 1 << dim_log;
  const int nblk = (block_size_log > 0) ? (dim >> block_size_log) : 1;
  const int wi = (int)((blockIdx.x * blockDim.x + threadIdx.x) >> 5) % (int)(blockDim.x >> 5);
  float *bmax = bmax_all + (size_t)wi * nblk;
  const int64_t row = (int64_t)((blockIdx.x * blockDim.x + threadIdx.x) >> 5);
  const int lane = threadIdx.x & 31;
  const bool valid = row < rows;
  const int64_t srow = valid ? row : 0; // keep every thread in the
  const T *src = x + (srow << dim_log); // same __syncthreads() region
  const int nblocks = nblk;
  const int bs = (block_size_log > 0) ? block_size_log : dim_log;
  // Lanes sharing one amax block: 2^bs elements / 8 elements per lane, capped at
  // the warp (a block wider than 256 elements is merged by the atomicMax below).
  // Without the cap, bs = 512 / 1024 asked for a shuffle width of 64 / 128 --
  // nobody had tried a scale block that wide until the range check went in.
  const int width = (block_size_log > 0 && bs < 8) ? (1 << (bs - 3)) : 32;

  for (int i = lane; i < nblocks; i += 32)
    bmax[i] = 0.0f;
  __syncthreads();

  // ---- pass 1: amax -------------------------------------------------------
  // Every lane executes every iteration: out of range offsets are clamped to
  // the last full chunk and masked out of the reduction.  This keeps the warp
  // converged across the __shfl_xor_sync calls (a divergent shuffle would be
  // undefined behaviour).
  for (int base = 0; base < dim; base += 256) {
    const int off = base + lane * 8;
    const bool in_range = (off + 8) <= dim;
    const int safe = in_range ? off : (dim - 8);
    uint4 raw = *reinterpret_cast<const uint4 *>(src + safe);
    float m = 0.0f;
#pragma unroll
    for (int i = 0; i < 8; ++i)
      m = fmaxf(m, in_range ? fabsf(elem_at<T>(raw, i)) : 0.0f);
    for (int o = width >> 1; o > 0; o >>= 1)
      m = fmaxf(m, __shfl_xor_sync(0xFFFFFFFFu, m, o, width));
    if (valid && in_range && (lane & (width - 1)) == 0)
      atomicMax(reinterpret_cast<int *>(&bmax[safe >> bs]), __float_as_int(m));
  }
  __syncthreads();

  // ---- pass 2: quantise (the row is in L2 by now) -------------------------
  if (quant_kind == kQuantFp8E4M3) {
    uint8_t *dst = reinterpret_cast<uint8_t *>(q_out) + ((int64_t)srow << dim_log);
    for (int base = 0; base < dim; base += 256) {
      const int off = base + lane * 8;
      if (off + 8 > dim)
        break;
      uint4 raw = *reinterpret_cast<const uint4 *>(src + off);
      float sc = (bmax[off >> bs] > 0.0f) ? (bmax[off >> bs] / 448.0f) : 1.0f;
      float inv = 1.0f / sc;
      uint2 out;
      out.x = pack_u32(to_e4m3x2(elem_at<T>(raw, 0) * inv, elem_at<T>(raw, 1) * inv),
                       to_e4m3x2(elem_at<T>(raw, 2) * inv, elem_at<T>(raw, 3) * inv));
      out.y = pack_u32(to_e4m3x2(elem_at<T>(raw, 4) * inv, elem_at<T>(raw, 5) * inv),
                       to_e4m3x2(elem_at<T>(raw, 6) * inv, elem_at<T>(raw, 7) * inv));
      if (valid)
        *reinterpret_cast<uint2 *>(dst + off) = out;
    }
  } else {
    uint8_t *dst = reinterpret_cast<uint8_t *>(q_out) + ((int64_t)srow << (dim_log - 1));
    for (int base = 0; base < dim; base += 256) {
      const int off = base + lane * 8;
      if (off + 8 > dim)
        break;
      uint4 raw = *reinterpret_cast<const uint4 *>(src + off);
      float sc = (bmax[off >> bs] > 0.0f) ? (bmax[off >> bs] / 7.0f) : 1.0f;
      float inv = 1.0f / sc;
      uint32_t w = 0;
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        int lo = quant_rne(elem_at<T>(raw, 2 * i), inv, 7) & 0xF;
        int hi = quant_rne(elem_at<T>(raw, 2 * i + 1), inv, 7) & 0xF;
        w |= (uint32_t)(lo | (hi << 4)) << (8 * i);
      }
      if (valid)
        *reinterpret_cast<uint32_t *>(dst + (off >> 1)) = w;
    }
  }

  // ---- scales -------------------------------------------------------------
  for (int i = lane; i < nblocks; i += 32) {
    float m = bmax[i];
    float denom = (quant_kind == kQuantFp8E4M3) ? 448.0f : 7.0f;
    if (valid)
      scale_out[srow * nblocks + i] = (m > 0.0f) ? (m / denom) : 1.0f;
  }
}

} // namespace fhwt