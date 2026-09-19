// =============================================================================
//  quant_pack.cuh -- element to byte packing for the quantised output.
//
//  Two encoders are provided for E4M3:
//    * the hardware `cvt.rn.satfinite.e4m3x2.f32` instruction (sm_89+), which
//      converts two floats per instruction, and
//    * a software fallback (see common.cuh) so the kernel stays portable.
//  Both agree bit for bit on the entire E4M3 range (checked by
//  tests/test_correctness.cu::fp8_encoder).
// =============================================================================
#pragma once

#include "common.cuh"

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 890)
#define FHWT_HW_FP8 1
#include <cuda_fp8.h>
#else
#define FHWT_HW_FP8 0
#endif

namespace fhwt {

// Hardware encoder (sm_89+): one instruction per element pair.
__device__ __forceinline__ uint16_t to_e4m3x2(float lo, float hi) {
#if FHWT_HW_FP8
  return (uint16_t)__nv_cvt_float2_to_fp8x2(make_float2(lo, hi), __NV_SATFINITE, __NV_E4M3);
#else
  return (uint16_t)(float_to_e4m3_rne(lo) | ((uint16_t)float_to_e4m3_rne(hi) << 8));
#endif
}

// Portable encoder, always available (used by the equivalence test).
__device__ __forceinline__ uint16_t to_e4m3x2_sw(float lo, float hi) {
  return (uint16_t)(float_to_e4m3_rne(lo) | ((uint16_t)float_to_e4m3_rne(hi) << 8));
}

// Pack kNBytes (1 or 2 or 4) little endian words into one 32 bit word.
__device__ __forceinline__ uint32_t pack_u32(uint16_t a, uint16_t b) {
  return (uint32_t)a | ((uint32_t)b << 16);
}

// -----------------------------------------------------------------------------
//  Store 8 consecutive quantised elements (fp8) as one 8 byte access.
// -----------------------------------------------------------------------------
template <typename Acc>
__device__ __forceinline__ void store_fp8x8(uint8_t *__restrict__ dst,
                                            const typename Acc::vec_t v[4], float inv) {
  uint2 p;
  p.x = pack_u32(to_e4m3x2(Acc::lo(v[0]) * inv, Acc::hi(v[0]) * inv),
                 to_e4m3x2(Acc::lo(v[1]) * inv, Acc::hi(v[1]) * inv));
  p.y = pack_u32(to_e4m3x2(Acc::lo(v[2]) * inv, Acc::hi(v[2]) * inv),
                 to_e4m3x2(Acc::lo(v[3]) * inv, Acc::hi(v[3]) * inv));
  *reinterpret_cast<uint2 *>(dst) = p;
}

// -----------------------------------------------------------------------------
//  Store 8 consecutive quantised elements (int4) as one 4 byte access.
//  Element 2i goes to the low nibble, element 2i+1 to the high nibble.
// -----------------------------------------------------------------------------
template <typename Acc>
__device__ __forceinline__ void store_int4x8(uint8_t *__restrict__ dst,
                                             const typename Acc::vec_t v[4], float inv) {
  uint32_t w = 0;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    int lo = quant_rne(Acc::lo(v[i]), inv, 7) & 0xF;
    int hi = quant_rne(Acc::hi(v[i]), inv, 7) & 0xF;
    w |= (uint32_t)((lo | (hi << 4)) << (8 * i));
  }
  *reinterpret_cast<uint32_t *>(dst) = w;
}

} // namespace fhwt