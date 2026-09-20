// Unit tests for the software FP8 / E2M1 / E8M0 codecs + 4-bit packing rules.
// Compiled for both host and device paths where relevant; all checks here run
// the __host__ __device__ codec functions on the host for determinism.
#include "fp_formats.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>

static int failures = 0;
#define CHECK(cond, msg)                                     \
  do {                                                       \
    if (!(cond)) {                                           \
      fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, msg); \
      ++failures;                                            \
    }                                                        \
  } while (0)

static bool near(float a, float b, float tol) {
  return fabsf(a - b) <= tol * (fabsf(b) + 1e-9f) + 1e-9f;
}

int main() {
  using namespace lp;

  // ---- E4M3 basics -------------------------------------------------------
  CHECK(fp8_to_fp32(fp32_to_fp8(0.0f, true), true) == 0.0f, "e4m3 zero");
  CHECK(fp8_to_fp32(fp32_to_fp8(1.0f, true), true) == 1.0f, "e4m3 one");
  CHECK(fp8_to_fp32(fp32_to_fp8(-2.0f, true), true) == -2.0f, "e4m3 -2");
  CHECK(fp8_to_fp32(fp32_to_fp8(448.0f, true), true) == 448.0f, "e4m3 max");
  CHECK(fp8_to_fp32(fp32_to_fp8(1e9f, true), true) == 448.0f, "e4m3 saturate");
  CHECK(fp8_to_fp32(fp32_to_fp8(-1e9f, true), true) == -448.0f, "e4m3 -sat");
  // subnormal: 2^-9 = 0x01 encoding -> 1 * 2^-9
  CHECK(near(fp8_to_fp32(fp32_to_fp8(ldexpf(1.0f, -9), true), true),
             ldexpf(1.0f, -9), 0.0f), "e4m3 subnormal");
  CHECK(near(fp8_to_fp32(fp32_to_fp8(0.001953125f * 1.5f, true), true),
             0.001953125f * 1.5f, 0.51f), "e4m3 subnormal 1.5x");
  // round-to-nearest-even: 1.0625 (tie between 1.0 and 1.125) -> 1.0
  CHECK(fp8_to_fp32(fp32_to_fp8(1.0625f, true), true) == 1.0f, "e4m3 tie even");
  CHECK(fp8_to_fp32(fp32_to_fp8(1.1875f, true), true) == 1.25f, "e4m3 tie up");
  // NaN
  CHECK(std::isnan(fp8_to_fp32(fp32_to_fp8(NAN, true), true)), "e4m3 nan");

  // ---- E5M2 ---------------------------------------------------------------
  CHECK(fp8_to_fp32(fp32_to_fp8(57344.0f, false), false) == 57344.0f,
        "e5m2 max");
  CHECK(fp8_to_fp32(fp32_to_fp8(1e9f, false), false) == 57344.0f,
        "e5m2 saturate");
  CHECK(fp8_to_fp32(fp32_to_fp8(1.0f, false), false) == 1.0f, "e5m2 one");
  CHECK(near(fp8_to_fp32(fp32_to_fp8(1.25f, false), false), 1.25f, 0.01f),
        "e5m2 1.25");
  CHECK(std::isnan(fp8_to_fp32(fp32_to_fp8(NAN, false), false)), "e5m2 nan");

  // ---- E2M1 ---------------------------------------------------------------
  static const float tab[8] = {0, .5f, 1, 1.5f, 2, 3, 4, 6};
  for (int c = 0; c < 8; ++c) {
    CHECK(e2m1_to_fp32((uint8_t)c) == tab[c], "e2m1 decode table");
    CHECK((fp32_to_e2m1(tab[c]) & 7) == c, "e2m1 encode exact");
    CHECK(fp32_to_e2m1(-tab[c]) == (uint8_t)(c | 8), "e2m1 encode sign");
  }
  CHECK((fp32_to_e2m1(0.25f) & 7) == 0, "e2m1 tie .25 -> 0");
  CHECK((fp32_to_e2m1(0.75f) & 7) == 2, "e2m1 tie .75 -> 1");
  CHECK((fp32_to_e2m1(2.5f) & 7) == 4, "e2m1 tie 2.5 -> 2");
  CHECK((fp32_to_e2m1(5.0f) & 7) == 6, "e2m1 tie 5 -> 4");
  CHECK((fp32_to_e2m1(100.0f) & 7) == 7, "e2m1 saturate");
  CHECK(e2m1_to_fp32(0xF) == -6.0f, "e2m1 -6");

  // ---- E8M0 ---------------------------------------------------------------
  CHECK(e8m0_to_fp32(127) == 1.0f, "e8m0 one");
  CHECK(e8m0_to_fp32(128) == 2.0f, "e8m0 two");
  CHECK(fp32_to_e8m0_ceil(1.0f) == 127, "e8m0 ceil exact");
  CHECK(fp32_to_e8m0_ceil(1.5f) == 128, "e8m0 ceil 1.5");
  CHECK(fp32_to_e8m0_ceil(3.9f) == 129, "e8m0 ceil 3.9");

  // ---- exhaustive fp8 roundtrip sanity ------------------------------------
  for (uint32_t i = 0; i <= 0xFF; ++i) {
    const uint8_t code = (uint8_t)i;
    const float v = fp8_to_fp32(code, true);
    if (!std::isnan(v)) {
      const uint8_t back = fp32_to_fp8(v, true);
      CHECK(back == code, "e4m3 roundtrip stable");
      if (back != code) break;
    }
  }

  if (failures) {
    fprintf(stderr, "%d failure(s)\n", failures);
    return 1;
  }
  printf("test_formats: all passed\n");
  return 0;
}
