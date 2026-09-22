#pragma once
#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>

// Software E4M3 / E2M1 conversion. No FP8/FP4 instruction is required.
__device__ __forceinline__ float e4_decode(uint8_t u) {
  int a = u & 127, e = a >> 3, m = a & 7;
  if (a == 127)
    return nanf("");
  float v = e ? ldexpf(float(8 + m), e - 10) : ldexpf(float(m), -9);
  return (u & 128) ? -v : v;
}

__device__ __forceinline__ uint8_t e4_encode(float x) {
  unsigned sign = __float_as_uint(x) >> 31;
  float a = fabsf(x);
  int code;
  if (isnan(a))
    return 127;
  if (a >= 448)
    code = 126;
  else if (a < 0x1p-6f)
    code = __float2int_rn(ldexpf(a, 9));
  else {
    int e = ilogbf(a);
    code = (e + 7) * 8 + __float2int_rn(ldexpf(a, 3 - e)) - 8;
  }
  return uint8_t((sign << 7) | min(code, 126));
}

__device__ __forceinline__ float e2_decode(uint8_t u) {
  const float values[8] = {0, .5f, 1, 1.5f, 2, 3, 4, 6};
  float v = values[u & 7];
  return (u & 8) ? -v : v;
}

__device__ __forceinline__ uint8_t e2_encode(float x) {
  float a = fabsf(x), best = 1e30f;
  int code = 0;
  // Enumeration is intentionally simple; equal distances choose an even code.
  for (int k = 0; k < 8; ++k) {
    float error = fabsf(a - e2_decode(k));
    if (error < best || (error == best && !(k & 1))) {
      best = error;
      code = k;
    }
  }
  return code | ((__float_as_uint(x) >> 31) << 3);
}

__device__ __forceinline__ float random01(unsigned i, unsigned seed) {
  unsigned z = i + seed * 747796405u + 2891336453u;
  z = (z ^ (z >> 16)) * 2246822519u;
  z = (z ^ (z >> 13)) * 3266489917u;
  return ((z ^ (z >> 16)) >> 8) * 0x1p-24f;
}

__device__ __forceinline__ uint8_t stochastic(float x, bool fp4, float u) {
  float a = fabsf(x);
  uint8_t near = fp4 ? e2_encode(a) : e4_encode(a);
  float v = fp4 ? e2_decode(near) : e4_decode(near);
  int maxcode = fp4 ? 7 : 126;
  int lo = v > a ? max(0, int(near) - 1) : int(near);
  int hi = v < a ? min(maxcode, int(near) + 1) : int(near);
  float lv = fp4 ? e2_decode(lo) : e4_decode(lo);
  float hv = fp4 ? e2_decode(hi) : e4_decode(hi);
  int code = hi != lo && u < (a - lv) / (hv - lv) ? hi : lo;
  return code | ((__float_as_uint(x) >> 31) << (fp4 ? 3 : 7));
}
