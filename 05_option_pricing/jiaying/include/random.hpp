#pragma once
#include "engine.hpp"
#include <cmath>

namespace pricing {
struct Words {
  uint32_t x, y, z, w;
};

// Philox4x32-10 counter = (group, 0, path_low, path_high), key = seed.
// This is the same bit stream as cuRAND Philox(seed, path, offset=0).
OP_HD inline Words philox(uint64_t seed, uint64_t path, uint64_t group) {
  Words c{static_cast<uint32_t>(group), static_cast<uint32_t>(group >> 32),
          static_cast<uint32_t>(path), static_cast<uint32_t>(path >> 32)};
  uint32_t k0 = static_cast<uint32_t>(seed),
           k1 = static_cast<uint32_t>(seed >> 32);
  for (int i = 0; i < 10; ++i) {
    const uint64_t a = uint64_t(0xD2511F53U) * c.x;
    const uint64_t b = uint64_t(0xCD9E8D57U) * c.z;
    c = {static_cast<uint32_t>(b >> 32) ^ c.y ^ k0, static_cast<uint32_t>(b),
         static_cast<uint32_t>(a >> 32) ^ c.w ^ k1, static_cast<uint32_t>(a)};
    k0 += 0x9E3779B9U;
    k1 += 0xBB67AE85U;
  }
  return c;
}
struct CustomBits {
  uint64_t seed, path, group = 0;
  OP_HD CustomBits(uint64_t s, uint64_t p) : seed(s), path(p) {}
  OP_HD Words next() { return philox(seed, path, group++); }
};
template <class Bits> struct NormalGenerator {
  Bits bits;
  float cache[4] = {};
  int cursor = 4;
  OP_HD NormalGenerator(uint64_t seed, uint64_t path) : bits(seed, path) {}
  OP_HD float next() {
    if (cursor == 4) {
      const Words b = bits.next();
      const uint32_t u[4] = {b.x, b.y, b.z, b.w};
      for (int i = 0; i < 4; i += 2) {
        // Midpoints of 2^23 equal bins: strictly inside (0,1), exactly in FP32.
        const float a = (float(u[i] >> 9) + 0.5f) * 0x1p-23f;
        const float angle =
            (float(u[i + 1] >> 9) + 0.5f) * 0x1p-23f * 6.283185307179586f;
        const float radius = sqrtf(-2.0f * logf(a));
        cache[i] = radius * cosf(angle);
        cache[i + 1] = radius * sinf(angle);
      }
      cursor = 0;
    }
    return cache[cursor++];
  }
};
} // namespace pricing
