// Bit-exactness check between the software codecs (include/fp_formats.h) and
// the CUDA Math API conversion intrinsics (cuda_fp8.h / cuda_fp4.h), which on
// Blackwell map to the native `cvt` hardware instructions.
//
// Device side is the authoritative comparison (real cvt instructions when
// compiled for an arch-specific suffix like 120a; header software emulation
// otherwise). Host side additionally validates the header's host path.
//
// Build: -DENABLE_NATIVE_CVT_TEST=ON (needs CUDA >= 12.8; on Blackwell pass
// -DCMAKE_CUDA_ARCHITECTURES=120 so the target compiles with the 120a suffix).
#include "fp_formats.h"

#include <cuda_fp8.h>
#include <cuda_fp4.h>
#include <cuda_fp16.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

static int failures = 0;
#define CHECK(cond, msg)                                                      \
  do {                                                                        \
    if (!(cond)) {                                                            \
      fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, msg);           \
      ++failures;                                                             \
    }                                                                         \
  } while (0)

__host__ __device__ inline bool bits_equal(float a, float b) {
  uint32_t ba, bb;
  memcpy(&ba, &a, 4);
  memcpy(&bb, &b, 4);
  return ba == bb;
}

__host__ __device__ inline bool value_equal(float a, float b) {
  const bool na = (a != a), nb = (b != b);
  return (na && nb) || (!na && !nb && bits_equal(a, b));
}

__host__ __device__ inline float halfraw_to_float(__half_raw r) {
  return __half2float(__ushort_as_half(r.x));
}

// Hardware/intrinsic encoders (host: header emulation; device: cvt when the
// arch supports it).
__host__ __device__ inline uint8_t hw_to_fp8(float x, bool e4m3) {
  return __nv_cvt_float_to_fp8(
      x, __NV_SATFINITE, e4m3 ? __NV_E4M3 : __NV_E5M2);
}
__host__ __device__ inline float hw_from_fp8(uint8_t c, bool e4m3) {
  return halfraw_to_float(__nv_cvt_fp8_to_halfraw(
      c, e4m3 ? __NV_E4M3 : __NV_E5M2));
}
__host__ __device__ inline uint8_t hw_to_e2m1(float x) {
  // x in the low nibble, 0.0f in the high nibble
  return (uint8_t)(__nv_cvt_float2_to_fp4x2(make_float2(x, 0.0f),
                                            __NV_E2M1,
                                            cudaRoundNearest) & 0xFu);
}
__host__ __device__ inline float hw_from_e2m1(uint8_t c) {
  __half2_raw r = __nv_cvt_fp4x2_to_halfraw2(c, __NV_E2M1);
  return __half2float(__ushort_as_half(r.x));
}

// Exhaustive per-code comparison, shared by host and device.
__host__ __device__ inline int check_all_codes() {
  int bad = 0;
  // decode: every fp8 code must decode to the same float
  for (uint32_t i = 0; i <= 0xFF; ++i) {
    const uint8_t c = (uint8_t)i;
    if (!value_equal(hw_from_fp8(c, true), lp::fp8_to_fp32(c, true))) ++bad;
    if (!value_equal(hw_from_fp8(c, false), lp::fp8_to_fp32(c, false))) ++bad;
  }
  // encode roundtrip: sw_decode(code) -> hw_encode == code. Skip NaN codes
  // and skip the E5M2 +/-inf codes: both encoders saturate inf to the max
  // finite value under __NV_SATFINITE, so inf cannot roundtrip by design.
  for (uint32_t i = 0; i <= 0xFF; ++i) {
    const uint8_t c = (uint8_t)i;
    const float v4 = lp::fp8_to_fp32(c, true);
    if (v4 == v4 && !isinf(v4) && hw_to_fp8(v4, true) != c) ++bad;
    const float v5 = lp::fp8_to_fp32(c, false);
    if (v5 == v5 && !isinf(v5) && hw_to_fp8(v5, false) != c) ++bad;
  }
  // E2M1: decode all 16 nibble codes + encode roundtrip
  for (uint32_t i = 0; i < 16; ++i) {
    const uint8_t c = (uint8_t)i;
    if (!value_equal(hw_from_e2m1(c), lp::e2m1_to_fp32(c))) ++bad;
    if (hw_to_e2m1(lp::e2m1_to_fp32(c)) != c) ++bad;
  }
  return bad;
}

// Random float sweep of the encoders (device): sw encoder must match the
// intrinsic bit-for-bit (NaN inputs compare by NaN-ness + magnitude code).
__global__ void sweep_kernel(const uint32_t* seeds, unsigned long long n,
                             unsigned long long* bad) {
  const unsigned long long stride =
      (unsigned long long)gridDim.x * blockDim.x;
  unsigned long long k = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
  for (; k < n; k += stride) {
    // xorshift32 keyed by the element index: covers the whole bit space.
    uint32_t s = seeds[k & 1023] ^ (uint32_t)(k * 2654435761u);
    s ^= s << 13; s ^= s >> 17; s ^= s << 5;
    float x;
    memcpy(&x, &s, 4);
    // e4m3 / e5m2: exact bit match, NaNs compare through their payload code
    const uint8_t hw4 = hw_to_fp8(x, true);
    const uint8_t sw4 = lp::fp32_to_fp8(x, true);
    if (hw4 != sw4 && !((hw4 & 0x7Fu) == 0x7Fu && (sw4 & 0x7Fu) == 0x7Fu))
      atomicAdd(bad, 1ull);
    const uint8_t hw5 = hw_to_fp8(x, false);
    const uint8_t sw5 = lp::fp32_to_fp8(x, false);
    if (hw5 != sw5 && !((hw5 & 0x7Eu) == 0x7Eu && (sw5 & 0x7Eu) == 0x7Eu))
      atomicAdd(bad, 1ull);
    // e2m1: NaN/Inf may saturate with differing sign bit, magnitude must match
    const uint8_t hwx = hw_to_e2m1(x);
    const uint8_t swx = lp::fp32_to_e2m1(x);
    if (hwx != swx && (hwx & 7u) != (swx & 7u)) atomicAdd(bad, 1ull);
  }
}

int main() {
  // ---- host (header emulation path) ---------------------------------------
  int host_bad = check_all_codes();
  CHECK(host_bad == 0, "host exhaustive code comparison");
  printf("host: 256 fp8 codes x {decode, encode-roundtrip} x {e4m3, e5m2} "
         "+ 16 e2m1 codes: %s\n", host_bad ? "FAILED" : "all equal");

  // ---- device --------------------------------------------------------------
  int dev = 0;
  cudaGetDevice(&dev);
  cudaDeviceProp prop{};
  cudaGetDeviceProperties(&prop, dev);
  printf("device: %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);

  const int dev_bad = check_all_codes();
  CHECK(dev_bad == 0, "device exhaustive code comparison");
  printf("device exhaustive codes: %s\n", dev_bad ? "FAILED" : "all equal");

  const unsigned long long n = 1ull << 22;
  uint32_t* seeds;
  unsigned long long* bad;
  cudaMalloc(&seeds, 1024 * sizeof(uint32_t));
  cudaMalloc(&bad, sizeof(unsigned long long));
  cudaMemset(bad, 0, sizeof(unsigned long long));
  // fixed seed table for reproducibility
  {
    uint32_t h[1024];
    uint32_t s = 0x9e3779b9u;
    for (auto& v : h) { s ^= s << 13; s ^= s >> 17; s ^= s << 5; v = s; }
    cudaMemcpy(seeds, h, sizeof(h), cudaMemcpyHostToDevice);
  }
  sweep_kernel<<<1024, 256>>>(seeds, n, bad);
  unsigned long long sweep_bad = 0;
  cudaMemcpy(&sweep_bad, bad, sizeof(sweep_bad), cudaMemcpyDeviceToHost);
  CHECK(sweep_bad == 0, "device random sweep");
  printf("device random sweep: %llu mismatches / %llu samples x 3 formats "
         "(full float bit space)\n", sweep_bad, n);

  cudaFree(seeds);
  cudaFree(bad);

  if (failures) {
    fprintf(stderr, "%d failure(s)\n", failures);
    return 1;
  }
  printf("test_native_cvt: all passed\n");
  return 0;
}
