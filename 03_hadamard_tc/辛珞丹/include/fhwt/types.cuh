// =============================================================================
//  types.cuh -- element traits and the 2-wide "Vec2" accumulator policies.
//
//  Every FWHT butterfly is a pair operation  (a, b) -> (a + b, a - b).
//  Two accumulator policies are provided:
//
//    * Native2<T>  keeps the data in the native packed 2-wide type
//                  (__half2 / __nv_bfloat162).  One 32-bit register holds two
//                  elements, so both ALU work and the warp-shuffle count are
//                  halved with respect to an fp32 accumulator.
//
//    * Fp32Acc<T>  promotes every pair to float2 and rounds back only at the
//                  store, matching the numerics of the reference
//                  fast_hadamard_transform (which also accumulates in fp32).
//                  Costs 2x ALU and 2x shuffle instructions.
//
//  A 32-bit word always carries exactly two elements of a 16-bit type, so the
//  load/store helpers can stay word-based for fp16/bf16 and fall back to native
//  4-byte float vectors for fp32 data (see load_row_vec / store_row_vec).
// =============================================================================
#pragma once

#include "common.cuh"

namespace fhwt {

// A 32-bit word holding two 16-bit elements (register to register type punning).
template <typename T> union Pair16 {
  uint32_t u;
  T h[2];
};

// -----------------------------------------------------------------------------
//  Element traits
// -----------------------------------------------------------------------------
template <typename T> struct ElemTraits;

template <> struct ElemTraits<__half> {
  static constexpr int kBytes = 2;
  static constexpr const char *kName = "fp16";
  static __device__ __forceinline__ __half from_float(float v) { return __float2half(v); }
  static __device__ __forceinline__ float to_float(__half v) { return __half2float(v); }
};

template <> struct ElemTraits<__nv_bfloat16> {
  static constexpr int kBytes = 2;
  static constexpr const char *kName = "bf16";
  static __device__ __forceinline__ __nv_bfloat16 from_float(float v) {
    return __float2bfloat16(v);
  }
  static __device__ __forceinline__ float to_float(__nv_bfloat16 v) { return __bfloat162float(v); }
};

template <> struct ElemTraits<float> {
  static constexpr int kBytes = 4;
  static constexpr const char *kName = "fp32";
  static __device__ __forceinline__ float from_float(float v) { return v; }
  static __device__ __forceinline__ float to_float(float v) { return v; }
};

// Bit-cast helpers (register to register, no memory traffic).
union Raw32 {
  uint32_t u;
  __half2 h;
  __nv_bfloat162 b;
  float f;
};
__device__ __forceinline__ uint32_t bits_of(__half2 v) {
  Raw32 r;
  r.h = v;
  return r.u;
}
__device__ __forceinline__ __half2 half2_of_bits(uint32_t r) {
  Raw32 u;
  u.u = r;
  return u.h;
}
__device__ __forceinline__ uint32_t bits_of(__nv_bfloat162 v) {
  Raw32 r;
  r.b = v;
  return r.u;
}
__device__ __forceinline__ __nv_bfloat162 bf162_of_bits(uint32_t r) {
  Raw32 u;
  u.u = r;
  return u.b;
}
// -----------------------------------------------------------------------------
//  bf16 pair primitives (bf162_add / sub / mul / fma / to_float2)
//
//  On CUDA these are the native packed instructions.  On MUSA the header
//  implementations cannot be used here:
//
//    * below __MUSA_ARCH__ 220 musa_bf16.h does not define them at all
//      (its guard is `__MUSA_ARCH__ >= 220 || !defined(__MUSA_ARCH__)`), and
//    * at 220 and above __hfma2 lowers to an inline `fma.rn.bf16x2` asm that
//      this compiler cannot register-allocate for mp_22 -- the build dies with
//      "couldn't allocate output register for constraint 'r'" inside
//      musa_bf16.hpp itself, so it is not something a call site can avoid by
//      being careful.
//
//  MUSA therefore gets a portable fp32-based pair implementation.  It rounds
//  once on the way out, so the accumulator behaves like the fp32 policy and
//  only the store to memory is bf16 -- numerically at least as good as the
//  native packed path, and it removes the dependency on an instruction mcc
//  cannot schedule here.  A CUDA build keeps the native instructions and is
//  bit-for-bit unchanged.
// -----------------------------------------------------------------------------
#if defined(__MUSACC__)
__device__ __forceinline__ __mt_bfloat162 bf162_add(__mt_bfloat162 a, __mt_bfloat162 b) {
  return __floats2bfloat162_rn(__low2float(a) + __low2float(b), __high2float(a) + __high2float(b));
}
__device__ __forceinline__ __mt_bfloat162 bf162_sub(__mt_bfloat162 a, __mt_bfloat162 b) {
  return __floats2bfloat162_rn(__low2float(a) - __low2float(b), __high2float(a) - __high2float(b));
}
__device__ __forceinline__ __mt_bfloat162 bf162_mul(__mt_bfloat162 a, __mt_bfloat162 b) {
  return __floats2bfloat162_rn(__low2float(a) * __low2float(b), __high2float(a) * __high2float(b));
}
__device__ __forceinline__ __mt_bfloat162 bf162_fma(__mt_bfloat162 a, __mt_bfloat162 b,
                                                    __mt_bfloat162 c) {
  return __floats2bfloat162_rn(__low2float(a) * __low2float(b) + __low2float(c),
                               __high2float(a) * __high2float(b) + __high2float(c));
}
__device__ __forceinline__ float2 bf162_to_float2(__mt_bfloat162 a) {
  return make_float2(__low2float(a), __high2float(a));
}
#else
__device__ __forceinline__ __nv_bfloat162 bf162_add(__nv_bfloat162 a, __nv_bfloat162 b) {
  return __hadd2(a, b);
}
__device__ __forceinline__ __nv_bfloat162 bf162_sub(__nv_bfloat162 a, __nv_bfloat162 b) {
  return __hsub2(a, b);
}
__device__ __forceinline__ __nv_bfloat162 bf162_mul(__nv_bfloat162 a, __nv_bfloat162 b) {
  return __hmul2(a, b);
}
__device__ __forceinline__ __nv_bfloat162 bf162_fma(__nv_bfloat162 a, __nv_bfloat162 b,
                                                    __nv_bfloat162 c) {
  return __hfma2(a, b, c);
}
__device__ __forceinline__ float2 bf162_to_float2(__nv_bfloat162 a) {
  return __bfloat1622float2(a);
}
#endif

// Swap the two 16-bit halves of a 32-bit register: one PRMT instruction.
__device__ __forceinline__ uint32_t swap_halves(uint32_t w) {
#if defined(__MUSACC__)
  // No __byte_perm on MUSA; the shift pair is what PRMT would have lowered to.
  return (w << 16) | (w >> 16);
#else
  return __byte_perm(w, w, 0x1032);
#endif
}
__device__ __forceinline__ __half2 half2_swap(__half2 v) {
  return half2_of_bits(swap_halves(bits_of(v)));
}
__device__ __forceinline__ __nv_bfloat162 bf162_swap(__nv_bfloat162 v) {
  return bf162_of_bits(swap_halves(bits_of(v)));
}

// -----------------------------------------------------------------------------
//  Policy: native packed 2-wide accumulation
// -----------------------------------------------------------------------------
template <typename T> struct Native2;

template <> struct Native2<__half> {
  using elem_t = __half;
  using vec_t = __half2;
  static constexpr int kElems = 2;

  static __device__ __forceinline__ vec_t add(vec_t a, vec_t b) { return __hadd2(a, b); }
  static __device__ __forceinline__ vec_t sub(vec_t a, vec_t b) { return __hsub2(a, b); }
  static __device__ __forceinline__ vec_t scale(vec_t a, float s) {
    return __hmul2(a, __float2half2_rn(s));
  }
  static __device__ __forceinline__ vec_t from_scalar(float s) { return __float2half2_rn(s); }
  // (a, b) -> (a + b, a - b): PRMT (lane swap) + HFMA2 with the constant (1,-1).
  static __device__ __forceinline__ vec_t butterfly01(vec_t v) {
    return __hfma2(v, __floats2half2_rn(1.0f, -1.0f), half2_swap(v));
  }
  static __device__ __forceinline__ vec_t mul_add(vec_t v, vec_t s, vec_t p) {
    return __hfma2(v, s, p);
  }
  static __device__ __forceinline__ vec_t shfl_xor(vec_t v, int k, int width, unsigned mask) {
    return half2_of_bits(__shfl_xor_sync(mask, bits_of(v), k, width));
  }
  static __device__ __forceinline__ vec_t from_word(uint32_t w) { return half2_of_bits(w); }
  static __device__ __forceinline__ uint32_t to_word(vec_t v) { return bits_of(v); }
  static __device__ __forceinline__ vec_t from_floats(float a, float b) {
    return __floats2half2_rn(a, b);
  }
  static __device__ __forceinline__ float lo(vec_t v) { return __low2float(v); }
  static __device__ __forceinline__ float hi(vec_t v) { return __high2float(v); }
  // One instruction for both halves; used by the GEMM inner loop, where the
  // per-element lo()/hi() pair would double the conversion count.
  static __device__ __forceinline__ float2 to_float2(vec_t v) { return __half22float2(v); }
};

template <> struct Native2<__nv_bfloat16> {
  using elem_t = __nv_bfloat16;
  using vec_t = __nv_bfloat162;
  static constexpr int kElems = 2;

  static __device__ __forceinline__ vec_t add(vec_t a, vec_t b) { return bf162_add(a, b); }
  static __device__ __forceinline__ vec_t sub(vec_t a, vec_t b) { return bf162_sub(a, b); }
  static __device__ __forceinline__ vec_t scale(vec_t a, float s) {
    return bf162_mul(a, __float2bfloat162_rn(s));
  }
  static __device__ __forceinline__ vec_t from_scalar(float s) { return __float2bfloat162_rn(s); }
  static __device__ __forceinline__ vec_t butterfly01(vec_t v) {
    return bf162_fma(v, __floats2bfloat162_rn(1.0f, -1.0f), bf162_swap(v));
  }
  static __device__ __forceinline__ vec_t mul_add(vec_t v, vec_t s, vec_t p) {
    return bf162_fma(v, s, p);
  }
  static __device__ __forceinline__ vec_t shfl_xor(vec_t v, int k, int width, unsigned mask) {
    return bf162_of_bits(__shfl_xor_sync(mask, bits_of(v), k, width));
  }
  static __device__ __forceinline__ vec_t from_floats(float a, float b) {
    return __floats2bfloat162_rn(a, b);
  }
  static __device__ __forceinline__ vec_t from_word(uint32_t w) { return bf162_of_bits(w); }
  static __device__ __forceinline__ uint32_t to_word(vec_t v) { return bits_of(v); }
  static __device__ __forceinline__ float lo(vec_t v) { return __low2float(v); }
  static __device__ __forceinline__ float hi(vec_t v) { return __high2float(v); }
  static __device__ __forceinline__ float2 to_float2(vec_t v) { return bf162_to_float2(v); }
};

// -----------------------------------------------------------------------------
//  Policy: fp32 promoted accumulation
// -----------------------------------------------------------------------------
template <typename T> struct Fp32Acc {
  using elem_t = T;
  using vec_t = float2;
  static constexpr int kElems = 2;

  static __device__ __forceinline__ vec_t add(vec_t a, vec_t b) {
    return make_float2(a.x + b.x, a.y + b.y);
  }
  static __device__ __forceinline__ vec_t sub(vec_t a, vec_t b) {
    return make_float2(a.x - b.x, a.y - b.y);
  }
  static __device__ __forceinline__ vec_t scale(vec_t a, float s) {
    return make_float2(a.x * s, a.y * s);
  }
  static __device__ __forceinline__ vec_t from_scalar(float s) { return make_float2(s, s); }
  static __device__ __forceinline__ vec_t butterfly01(vec_t v) {
    return make_float2(v.x + v.y, v.x - v.y);
  }
  static __device__ __forceinline__ vec_t mul_add(vec_t v, vec_t s, vec_t p) {
    return make_float2(fmaf(v.x, s.x, p.x), fmaf(v.y, s.y, p.y));
  }
  static __device__ __forceinline__ vec_t shfl_xor(vec_t v, int k, int width, unsigned mask) {
    return make_float2(__shfl_xor_sync(mask, v.x, k, width), __shfl_xor_sync(mask, v.y, k, width));
  }
  // Only valid for 16-bit element types; the two halves of the word are promoted.
  static __device__ __forceinline__ vec_t from_word(uint32_t w) {
    Pair16<T> p;
    p.u = w;
    return make_float2(ElemTraits<T>::to_float(p.h[0]), ElemTraits<T>::to_float(p.h[1]));
  }
  static __device__ __forceinline__ uint32_t to_word(vec_t v) {
    Pair16<T> p;
    p.h[0] = ElemTraits<T>::from_float(v.x);
    p.h[1] = ElemTraits<T>::from_float(v.y);
    return p.u;
  }
  static __device__ __forceinline__ vec_t from_floats(float a, float b) {
    return make_float2(a, b);
  }
  static __device__ __forceinline__ float lo(vec_t v) { return v.x; }
  static __device__ __forceinline__ float hi(vec_t v) { return v.y; }
  static __device__ __forceinline__ float2 to_float2(vec_t v) { return v; }
};

// fp32 data keeps its own precision; the policy is only used as a container.
template <> struct Fp32Acc<float> {
  using elem_t = float;
  using vec_t = float2;
  static constexpr int kElems = 2;

  static __device__ __forceinline__ vec_t add(vec_t a, vec_t b) {
    return make_float2(a.x + b.x, a.y + b.y);
  }
  static __device__ __forceinline__ vec_t sub(vec_t a, vec_t b) {
    return make_float2(a.x - b.x, a.y - b.y);
  }
  static __device__ __forceinline__ vec_t scale(vec_t a, float s) {
    return make_float2(a.x * s, a.y * s);
  }
  static __device__ __forceinline__ vec_t from_scalar(float s) { return make_float2(s, s); }
  static __device__ __forceinline__ vec_t butterfly01(vec_t v) {
    return make_float2(v.x + v.y, v.x - v.y);
  }
  static __device__ __forceinline__ vec_t mul_add(vec_t v, vec_t s, vec_t p) {
    return make_float2(fmaf(v.x, s.x, p.x), fmaf(v.y, s.y, p.y));
  }
  static __device__ __forceinline__ vec_t shfl_xor(vec_t v, int k, int width, unsigned mask) {
    return make_float2(__shfl_xor_sync(mask, v.x, k, width), __shfl_xor_sync(mask, v.y, k, width));
  }
  static __device__ __forceinline__ vec_t from_floats(float a, float b) {
    return make_float2(a, b);
  }
  static __device__ __forceinline__ float lo(vec_t v) { return v.x; }
  static __device__ __forceinline__ float hi(vec_t v) { return v.y; }
  static __device__ __forceinline__ float2 to_float2(vec_t v) { return v; }
};

} // namespace fhwt
