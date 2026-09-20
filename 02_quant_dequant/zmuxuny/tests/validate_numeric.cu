// Regression for lane-dependent hashing under optimized device compilation.
// Exercise native-wave boundaries, multiple CTAs and both halves of size_t.
#include "../include/numeric.cuh"
#include <cstdio>
#include <cstring>
#include <vector>

constexpr int COUNT = 4096;
#if !defined(LP_ASCEND)
__global__ void hashes(float *out, uint64_t base, uint32_t seed) {
  unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < COUNT)
    out[i] = lp::uniform(base + i, seed);
}

__global__ void products(const float *a, const float *b, float *out, bool division) {
  unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < COUNT)
    out[i] = division ? lp::divide_rn(a[i], b[i]) : lp::multiply_rn(a[i], b[i]);
}

#else
extern "C" void launch_ascend_numeric(void*,void*,void*,void*,uint64_t,uint32_t,int);
#endif

int main() {
  float *device = nullptr;
  auto check = [](cudaError_t e) {
    if (e == cudaSuccess)
      return true;
    std::fprintf(stderr, "%s\n", cudaGetErrorString(e));
    return false;
  };
  if (!check(cudaMalloc(&device, COUNT * sizeof(float))))
    return 1;
  std::vector<float> result(COUNT);
  size_t checked = 0;
  for (uint64_t base : {0ULL, (1ULL << 32) - 65, 1ULL << 40}) {
    for (uint32_t seed : {0U, 7U, 42U, 0xffffffffU}) {
#if defined(LP_ASCEND)
      launch_ascend_numeric(lp_ascend::stream(),nullptr,nullptr,device,base,seed,0);
#else
      hashes<<<COUNT / 256, 256>>>(device, base, seed);
#endif
      if (!check(cudaGetLastError()) ||
          !check(cudaMemcpy(result.data(), device, COUNT * sizeof(float),
                            cudaMemcpyDeviceToHost)))
        return 1;
      for (unsigned i = 0; i < COUNT; ++i) {
        float expected = lp::uniform(base + i, seed);
        if (result[i] != expected) {
          std::fprintf(
              stderr,
              "hash mismatch: base=%llu index=%u seed=%u expected=%.9g actual=%.9g\n",
              static_cast<unsigned long long>(base), i, seed, expected, result[i]);
          return 1;
        }
        ++checked;
      }
    }
  }
  // Cover signed zero, FP32 subnormals, underflow ties, normal boundaries and
  // ordinary products. Compare bits against host IEEE round-to-nearest-even.
  float *da = nullptr, *db = nullptr;
  if (!check(cudaMalloc(&da, COUNT * sizeof(float))) ||
      !check(cudaMalloc(&db, COUNT * sizeof(float))))
    return 1;
  const unsigned exponents[] = {0,   1,   2,   8,   32,  64,  100, 120,
                                126, 127, 128, 134, 150, 180, 220, 254};
  std::vector<float> a(COUNT), b(COUNT);
  size_t product_checks = 0;
  for (unsigned trial = 0; trial < 16; ++trial) {
    for (unsigned i = 0; i < COUNT; ++i) {
      unsigned ma = (i * 0x7feb352dU + trial * 97U) & 0x7fffffU;
      unsigned mb = (i * 0x846ca68bU + trial * 131U) & 0x7fffffU;
      if (i % 7 == 0)
        ma = 0;
      if (i % 11 == 0)
        mb = 0;
      unsigned abits = ((i & 1) << 31) | (exponents[(i / 16) % 16] << 23) | ma;
      unsigned bbits = ((i & 2) << 30) | (exponents[(i + trial) % 16] << 23) | mb;
      std::memcpy(&a[i], &abits, 4);
      std::memcpy(&b[i], &bbits, 4);
    }
    if (!check(
            cudaMemcpy(da, a.data(), COUNT * sizeof(float), cudaMemcpyHostToDevice)) ||
        !check(cudaMemcpy(db, b.data(), COUNT * sizeof(float), cudaMemcpyHostToDevice)))
      return 1;
#if defined(LP_ASCEND)
    launch_ascend_numeric(lp_ascend::stream(),da,db,device,0,0,1);
#else
    products<<<COUNT / 256, 256>>>(da, db, device, false);
#endif
    if (!check(cudaGetLastError()) ||
        !check(cudaMemcpy(result.data(), device, COUNT * sizeof(float),
                          cudaMemcpyDeviceToHost)))
      return 1;
    for (unsigned i = 0; i < COUNT; ++i) {
      float expected = a[i] * b[i];
      if (std::memcmp(&expected, &result[i], 4)) {
        std::fprintf(
            stderr,
            "product mismatch trial=%u index=%u a=%a b=%a expected=%a actual=%a\n",
            trial, i, double(a[i]), double(b[i]), double(expected), double(result[i]));
        return 1;
      }
      ++product_checks;
    }
  }
  size_t division_checks = 0;
  // Cover every exponent pair in the fast interval, random mantissas, signs,
  // exact powers of two, quotient boundaries and the precise fallback range.
  uint32_t rng = 0x12345678U;
  for (unsigned trial = 0; trial < 256; ++trial) {
    for (unsigned i = 0; i < COUNT; ++i) {
      rng = rng * 1664525U + 1013904223U;
      unsigned abits = (rng & 0x807fffffU) | ((1 + (i + trial) % 254) << 23);
      rng = rng * 1664525U + 1013904223U;
      unsigned bbits = (rng & 0x807fffffU) | ((1 + (i / 254 + trial) % 254) << 23);
      if (i % 7 == 0)
        abits &= 0xff800000U;
      if (i % 11 == 0)
        bbits &= 0xff800000U;
      std::memcpy(&a[i], &abits, 4);
      std::memcpy(&b[i], &bbits, 4);
    }
    if (!check(
            cudaMemcpy(da, a.data(), COUNT * sizeof(float), cudaMemcpyHostToDevice)) ||
        !check(cudaMemcpy(db, b.data(), COUNT * sizeof(float), cudaMemcpyHostToDevice)))
      return 1;
#if defined(LP_ASCEND)
    launch_ascend_numeric(lp_ascend::stream(),da,db,device,0,0,2);
#else
    products<<<COUNT / 256, 256>>>(da, db, device, true);
#endif
    if (!check(cudaGetLastError()) ||
        !check(cudaMemcpy(result.data(), device, COUNT * sizeof(float),
                          cudaMemcpyDeviceToHost)))
      return 1;
    for (unsigned i = 0; i < COUNT; ++i) {
      float expected = a[i] / b[i];
      if (std::memcmp(&expected, &result[i], 4)) {
        std::fprintf(
            stderr,
            "division mismatch trial=%u index=%u a=%a b=%a expected=%a actual=%a\n",
            trial, i, double(a[i]), double(b[i]), double(expected), double(result[i]));
        return 1;
      }
      ++division_checks;
    }
  }
  if (!check(cudaFree(da)) || !check(cudaFree(db)))
    return 1;
  if (!check(cudaFree(device)))
    return 1;
  std::printf("{\"status\":\"PASS\",\"exact_hash_checks\":%zu,\"exact_product_checks\":"
              "%zu,\"exact_division_checks\":%zu}\n",
              checked, product_checks, division_checks);
}
