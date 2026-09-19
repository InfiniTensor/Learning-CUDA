// =============================================================================
//  kernel_tc.cuh -- Tensor Core (MMA) fast Hadamard transform.
//
//  Motivation
//  ----------
//  A naively "matmul-ised" Hadamard transform wastes the tensor core: a dense
//  D x D matrix product performs D^2 MACs per row whereas the FWHT performs
//  only (D/2) log2(D) butterflies, an execute/operate ratio of
//
//      2*D^2 / (D*log2 D) = 2D / log2 D        (64x for D = 256)
//
//  Dense MMA is therefore hopeless; the only structure a tensor core can
//  exploit is the *block diagonal* one.
//
//  Decomposition
//  -------------
//  With D = 2^r * 16^L  (r = kLogD % 4, L = kLogD / 4) the Kronecker algebra
//  gives
//
//      H_D = (H_16 (x) H_16 (x) ... (x) H_16) (x) H_{2^r}
//
//  Each radix-16 factor acts on exactly one "digit" i_l of the mixed-radix
//  index decomposition
//
//      i = i_res + 2^r * (i_1 + 16*i_2 + 256*i_3 + ...),   i_l, i_res < 16, 2^r
//
//  so every stage is a *block diagonal* operator: D/16 independent 16-point
//  Hadamard transforms whose 16 members sit s = 2^(r + 4*(l-1)) elements apart
//  inside a window of 16*s contiguous elements.
//
//  One independent 16-point transform maps onto exactly one MMA:
//
//      C(16x8) = A(16x16) * B(16x8),   A      = H_16 (a compile-time constant)
//                                      B[:,n] = the 16 elements of group n
//                                      C[m,n] = output m of group n
//
//  so a single m16n8k16 instruction replaces 16*16 = 256 MACs (128 butterflies)
//  at a cost of 1 executed instruction.  This is the HadaCore observation
//  (arXiv 2412.08832) applied to the register-free formulation.
//
//  Layout / data movement
//  ----------------------
//  The operand fragments the MMA needs are strided gathers, so the row is
//  staged once in shared memory and every stage performs
//
//      gather B fragment -> mma -> scatter C fragment
//
//  The per-warp shared buffer holds exactly one row, so no __syncthreads() is
//  required (a __syncwarp() memory barrier is enough) and every global access
//  stays fully vectorised.
//
//  Cost model: per radix-16 stage each element is read and written once in
//  shared memory, plus 1/8 mma-instruction per element (one mma covers 16*8).
//  Shared memory runs ~20x faster than DRAM, so the transform is expected to
//  stay DRAM bound; the kernel exists to locate the tensor-core crossover and
//  to explain why a memory bound FWHT never reaches it (see the report).
// =============================================================================
#pragma once

#if defined(__MUSACC__)
// -----------------------------------------------------------------------------
//  MUSA: this path is compiled out.
//
//  The kernel below is written in inline PTX (`mma.sync.aligned.m16n8k16...`),
//  which is the NVIDIA ISA; the MUSA device ISA has no such instruction and
//  mcc rejects the asm.  MUSA does ship an mma.h, but it exposes no m16n8k16
//  primitive with the fragment layout this kernel needs, and the path is worth
//  nothing anyway on the numbers this project measured (report 7.2 / 7.3: the
//  tensor-core route ties the register route in the DRAM-bound region and
//  loses in the L2-resident one, because this operator is bandwidth-bound).
//
//  So the launcher is replaced by a stub and ops.cu rejects --kernel tc with an
//  explicit message instead of silently returning wrong data.
// -----------------------------------------------------------------------------
namespace fhwt {
template <typename T, typename Acc, int kLogD>
__global__ void fhwt_tc_kernel(const T *, T *, int64_t, float) {}
} // namespace fhwt

#else

#include "common.cuh"
#include "kernel_reg.cuh" // load_row_vec / store_row_vec (vectorised staging)
#include "types.cuh"

namespace fhwt {

// =============================================================================
//  H_16 helpers
// =============================================================================
// H_16[row][col] = (-1)^popcount(row & col)
__device__ __forceinline__ float h16_sign(int row, int col) {
  return (__popc((unsigned)row & (unsigned)col) & 1) ? -1.0f : 1.0f;
}

// Packs two signs into one 32-bit register holding two elements of T.
template <typename T> __device__ __forceinline__ uint32_t pack_pair(float s0, float s1) {
  Pair16<T> p;
  p.h[0] = ElemTraits<T>::from_float(s0);
  p.h[1] = ElemTraits<T>::from_float(s1);
  return p.u;
}

// -----------------------------------------------------------------------------
//  A-fragment of the constant H_16 (m16n8k16, .row, 8 halves per thread).
//    a0: row g   , col 2t         a1: row g  , col 2t+1
//    a2: row g+8 , col 2t         a3: row g+8, col 2t+1
//    a4: row g   , col 2t+8       a5: row g  , col 2t+9
//    a6: row g+8 , col 2t+8       a7: row g+8, col 2t+9
//  with g = lane >> 2 and t = lane & 3.
// -----------------------------------------------------------------------------
template <typename T> __device__ __forceinline__ void h16_a_fragment(int lane, uint32_t a[4]) {
  const int g = lane >> 2;
  const int t = lane & 3;
  const int c0 = 2 * t, c1 = 2 * t + 1, c2 = 2 * t + 8, c3 = 2 * t + 9;
  a[0] = pack_pair<T>(h16_sign(g, c0), h16_sign(g, c1));
  a[1] = pack_pair<T>(h16_sign(g + 8, c0), h16_sign(g + 8, c1));
  a[2] = pack_pair<T>(h16_sign(g, c2), h16_sign(g, c3));
  a[3] = pack_pair<T>(h16_sign(g + 8, c2), h16_sign(g + 8, c3));
}

// -----------------------------------------------------------------------------
//  mma.sync.aligned.m16n8k16.row.col.<dtype>.<atype>.<btype>.<ctype>
//
//  Note the PTX type order: destination, A, B, C -- NOT the order in which the
//  operands are written.  Both fp16 and bf16 inputs use an fp32 accumulator,
//  which is also what the reference fast_hadamard_transform accumulates in.
//  The C/D fragment is therefore four .f32 registers per thread:
//      c0 = C[g  ][2t]   c1 = C[g  ][2t+1]
//      c2 = C[g+8][2t]   c3 = C[g+8][2t+1]
// -----------------------------------------------------------------------------
__device__ __forceinline__ void mma_m16n8k16_f16(float c[4], const uint32_t a[4],
                                                 const uint32_t b[2]) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
               "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__device__ __forceinline__ void mma_m16n8k16_bf16(float c[4], const uint32_t a[4],
                                                  const uint32_t b[2]) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
               "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// Compile-time dispatch on the element type.
template <typename T> struct MmaOp;
template <> struct MmaOp<__half> {
  static __device__ __forceinline__ void run(float c[4], const uint32_t a[4], const uint32_t b[2]) {
    mma_m16n8k16_f16(c, a, b);
  }
};
template <> struct MmaOp<__nv_bfloat16> {
  static __device__ __forceinline__ void run(float c[4], const uint32_t a[4], const uint32_t b[2]) {
    mma_m16n8k16_bf16(c, a, b);
  }
};

// =============================================================================
//  Stage strides
//  Stage l (l = 1..L) transforms the digit i_l, whose stride is
//      s_l = 2^(r + 4*(l-1))
// =============================================================================
__host__ __device__ constexpr int tc_stage_stride_log(int kLogD, int l) {
  return (kLogD % 4) + 4 * (l - 1);
}

// =============================================================================
//  One radix-16 stage of stride 2^kLogS.
//      base(n)  = (n / s) * 16 * s + (n % s)      (minimum position of group n)
//      P(n, k)  = base(n) + k * s
//  The D/16 groups are consumed 8 at a time along the n dimension of the MMA.
// =============================================================================
template <typename T, int kD, int kLogS>
__device__ __forceinline__ void tc_radix16_stage(T *__restrict__ buf, int lane) {
  constexpr int kS = 1 << kLogS;
  constexpr int kNGroup = kD / 16;        // independent 16-point transforms
  constexpr int kMma = (kNGroup + 7) / 8; // mma instructions per row
  const int g = lane >> 2;
  const int t = lane & 3;

  uint32_t a[4];
  h16_a_fragment<T>(lane, a);

  // ---------------------------------------------------------------------------
  //  Register blocking: the whole stage is issued as
  //      gather every B fragment -> mma everything -> scatter every C fragment
  //  rather than load/mma/store per mma.  Without this the compiler has to
  //  assume the scattered stores alias the next gather (same buffer) and the
  //  stage degenerates into a chain of round trips through shared memory.
  // ---------------------------------------------------------------------------
  uint32_t b[kMma][2];
  float c[kMma][4];

#pragma unroll
  for (int j = 0; j < kMma; ++j) {
    int n = 8 * j + g;
    if (n >= kNGroup)
      n = 0; // tail: harmless duplicate read
    const int nbase = (n / kS) * (16 * kS) + (n % kS);
    const int p0 = nbase + (2 * t + 0) * kS;
    const int p1 = nbase + (2 * t + 1) * kS;
    const int p2 = nbase + (2 * t + 8) * kS;
    const int p3 = nbase + (2 * t + 9) * kS;
    Pair16<T> lo, hi;
    lo.h[0] = buf[p0];
    lo.h[1] = buf[p1];
    hi.h[0] = buf[p2];
    hi.h[1] = buf[p3];
    b[j][0] = lo.u;
    b[j][1] = hi.u;
#pragma unroll
    for (int q = 0; q < 4; ++q)
      c[j][q] = 0.f;
  }

#pragma unroll
  for (int j = 0; j < kMma; ++j)
    MmaOp<T>::run(c[j], a, b[j]);

#pragma unroll
  for (int j = 0; j < kMma; ++j) {
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      const int ng = 8 * j + 2 * t + half;
      if (ng < kNGroup) {
        const int mbase = (ng / kS) * (16 * kS) + (ng % kS);
        buf[mbase + g * kS] = ElemTraits<T>::from_float(c[j][half]);
        buf[mbase + (g + 8) * kS] = ElemTraits<T>::from_float(c[j][2 + half]);
      }
    }
  }
}

// =============================================================================
//  Residual stage: H_{2^r} applied to every contiguous 2^r block
//  (r = kLogD % 4).  At most 3 butterfly stages, done in registers in fp32.
// =============================================================================
template <typename T, int kD, int kR>
__device__ __forceinline__ void tc_residual_stage(T *__restrict__ buf, int lane) {
  if constexpr (kR == 0) {
    (void)buf;
    (void)lane;
  } else {
    constexpr int kB = 1 << kR;     // block length
    constexpr int kNBlk = kD >> kR; // number of blocks in the row
    for (int b0 = 0; b0 < kNBlk; b0 += 32) {
      const int b = b0 + lane;
      if (b < kNBlk) {
        const int base = b * kB;
        float v[kB];
#pragma unroll
        for (int i = 0; i < kB; ++i)
          v[i] = ElemTraits<T>::to_float(buf[base + i]);
#pragma unroll
        for (int len = 1; len < kB; len <<= 1) {
#pragma unroll
          for (int i = 0; i < kB; i += 2 * len) {
#pragma unroll
            for (int q = 0; q < len; ++q) {
              const float u = v[i + q], w = v[i + q + len];
              v[i + q] = u + w;
              v[i + q + len] = u - w;
            }
          }
        }
#pragma unroll
        for (int i = 0; i < kB; ++i)
          buf[base + i] = ElemTraits<T>::from_float(v[i]);
      }
    }
  }
}

// =============================================================================
//  Vectorised global <-> shared staging of one row.
//  Lane `lane` owns the contiguous range [lane*kEPT, (lane+1)*kEPT).
// =============================================================================
template <typename T, int kEPT>
__device__ __forceinline__ void tc_stage_in(const T *__restrict__ src, T *__restrict__ buf,
                                            int lane) {
  constexpr int kBytes = kEPT * sizeof(T);
  const T *p = src + lane * kEPT;
  T *q = buf + lane * kEPT;
  if constexpr (kBytes % 16 == 0) {
#pragma unroll
    for (int i = 0; i < kBytes / 16; ++i)
      reinterpret_cast<uint4 *>(q)[i] = reinterpret_cast<const uint4 *>(p)[i];
  } else if constexpr (kBytes % 8 == 0) {
#pragma unroll
    for (int i = 0; i < kBytes / 8; ++i)
      reinterpret_cast<uint2 *>(q)[i] = reinterpret_cast<const uint2 *>(p)[i];
  } else if constexpr (kBytes % 4 == 0) {
#pragma unroll
    for (int i = 0; i < kBytes / 4; ++i)
      reinterpret_cast<uint32_t *>(q)[i] = reinterpret_cast<const uint32_t *>(p)[i];
  } else {
#pragma unroll
    for (int i = 0; i < kEPT; ++i)
      q[i] = p[i];
  }
}

template <typename T, typename Acc, int kEPT>
__device__ __forceinline__ void tc_stage_out(T *__restrict__ buf, T *__restrict__ dst, int lane,
                                             float scale) {
  // Reuse the vectorised register-kernel store: it keeps the 128-bit global
  // access *and* the scaling, whereas a naive scalar loop would turn the warp's
  // stores into 32 separate 2-byte accesses 64 bytes apart (16x write
  // amplification on the dominant DRAM stream).
  if constexpr (kEPT >= 2) {
    typename Acc::vec_t v[kEPT / 2];
    load_row_vec<kEPT, T, Acc>(buf + lane * kEPT, v);
    store_row_vec<kEPT, T, Acc>(dst + lane * kEPT, v, scale);
  } else { // dim == 32: one element per lane, the vector policy needs pairs
    const int i = lane;
    dst[i] = ElemTraits<T>::from_float(ElemTraits<T>::to_float(buf[i]) * scale);
  }
}

// =============================================================================
//  The complete tensor-core kernel: one warp per row.
//
//  The radix-16 stages are emitted by a recursive template helper so that every
//  stride is a compile-time constant (the smem index arithmetic collapses into
//  shifts and the __syncwarp() count is exact).
// =============================================================================
template <typename T, int kD, int kLogD, int kL> struct TcStages {
  static __device__ __forceinline__ void run(T *__restrict__ buf, int lane) {
    tc_radix16_stage<T, kD, tc_stage_stride_log(kLogD, kL)>(buf, lane);
    __syncwarp();
    TcStages<T, kD, kLogD, kL - 1>::run(buf, lane);
  }
};
template <typename T, int kD, int kLogD> struct TcStages<T, kD, kLogD, 0> {
  static __device__ __forceinline__ void run(T *__restrict__ buf, int lane) {
    (void)buf;
    (void)lane;
  }
};

template <typename T, typename Acc, int kLogD>
__global__ void fhwt_tc_kernel(const T *__restrict__ x, T *__restrict__ y, int64_t rows,
                               float scale) {
  constexpr int kD = 1 << kLogD;
  constexpr int kEPT = kD / 32; // elements per lane while staging
  constexpr int kR = kLogD % 4;
  constexpr int kL = kLogD / 4;

  extern __shared__ __align__(16) unsigned char smem_raw[];
  T *buf = reinterpret_cast<T *>(smem_raw) + (size_t)(threadIdx.x >> 5) * kD;

  const int lane = threadIdx.x & 31;
  const int warps_per_block = blockDim.x >> 5;
  const int64_t stride = (int64_t)gridDim.x * warps_per_block;
  int64_t row = (int64_t)blockIdx.x * warps_per_block + (threadIdx.x >> 5);

  for (; row < rows; row += stride) {
    tc_stage_in<T, kEPT>(x + row * kD, buf, lane);
    __syncwarp();
    tc_residual_stage<T, kD, kR>(buf, lane);
    __syncwarp();
    TcStages<T, kD, kLogD, kL>::run(buf, lane);
    tc_stage_out<T, Acc, kEPT>(buf, y + row * kD, lane, scale);
    __syncwarp(); // the buffer is reused by the next row of this warp
  }
}

} // namespace fhwt

#endif // !defined(__MUSACC__)
