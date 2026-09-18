#pragma once

#include <cstdint>
#if defined(__CUDACC__) || defined(__MUSACC__)
#define LP_HOST_DEVICE __host__ __device__
#else
#define LP_HOST_DEVICE
#endif

namespace pipeline {

// 按 seed 和元素下标产生随机数，与线程调度无关；固定 seed 可逐字节重现。
LP_HOST_DEVICE inline double random_unit(std::uint32_t seed, std::uint64_t index) {
  std::uint32_t x = seed ^ static_cast<std::uint32_t>(index) ^
                    static_cast<std::uint32_t>(index >> 32) * 0x9e3779b9u;
  x += 0x9e3779b9u;
  x ^= x >> 16;
  x *= 0x85ebca6bu;
  x ^= x >> 13;
  x *= 0xc2b2ae35u;
  x ^= x >> 16;
  return (static_cast<double>(x) + 0.5) / 4294967296.0;
}

// E2M1 的低 3 位表示幅值，符号由调用方处理。
LP_HOST_DEVICE inline float magnitude2(unsigned code) {
  const float values[8] = {0, .5f, 1, 1.5f, 2, 3, 4, 6};
  return values[code];
}

}  // namespace pipeline

#undef LP_HOST_DEVICE
