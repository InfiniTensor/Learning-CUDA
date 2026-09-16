#pragma once

#include "vsearch/common.hpp"

namespace vs {

Metric parseMetric(const std::string& s);
DataType parseDataType(const std::string& s);
SearchMode parseSearchMode(const std::string& s);
IndexKind parseIndexKind(const std::string& s);

// Exact float representation sort is used as the ranking key.
inline i32 floatBits(float v) {
  i32 u;
  std::memcpy(&u, &v, sizeof(u));
  return u;
}

inline float bitsToFloat(i32 u) {
  float v;
  std::memcpy(&v, &u, sizeof(v));
  return v;
}

inline i32 f32ToSortableBits(float v) {
  const i32 u = floatBits(v);
  // IEEE-754 fp32 total order. Negative values invert all bits; the sign bit
  // is then cleared so the mapping is strictly monotone.
  return (u & 0x80000000) ? ~u : (u | 0x80000000);
}

inline float sortableBitsToF32(i32 u) {
  const i32 raw = (u & 0x80000000) ? (u & 0x7fffffff) : ~u;
  return bitsToFloat(raw);
}

}  // namespace vs
