#include "vsearch/binary_io.hpp"
#include "vsearch/metric.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>

namespace vs {
namespace {

void writeBytes(FILE* f, const void* p, std::size_t n) {
  if (fwrite(p, 1, n, f) != n) throwRuntime("binary_io: short write");
}

void readBytes(FILE* f, void* p, std::size_t n) {
  if (fread(p, 1, n, f) != n) throwRuntime("binary_io: short read / truncated file");
}

void writeI64(FILE* f, i64 v) {
  unsigned char b[8];
  for (int i = 0; i < 8; ++i) b[i] = static_cast<unsigned char>((v >> (8 * i)) & 0xff);
  writeBytes(f, b, 8);
}

i64 readI64(FILE* f) {
  unsigned char b[8];
  readBytes(f, b, 8);
  i64 v = 0;
  for (int i = 7; i >= 0; --i) v = (v << 8) | b[i];
  return v;
}

void writeI32(FILE* f, i32 v) {
  unsigned char b[4];
  for (int i = 0; i < 4; ++i) b[i] = static_cast<unsigned char>((v >> (8 * i)) & 0xff);
  writeBytes(f, b, 4);
}

i32 readI32(FILE* f) {
  unsigned char b[4];
  readBytes(f, b, 4);
  i32 v = 0;
  for (int i = 3; i >= 0; --i) v = (v << 8) | b[i];
  return v;
}

void writeString(FILE* f, const std::string& s) {
  const auto n = static_cast<unsigned char>(s.size());
  writeBytes(f, &n, 1);
  writeBytes(f, s.data(), s.size());
}

std::string readString(FILE* f) {
  unsigned char n = 0;
  readBytes(f, &n, 1);
  std::string s(n, '\0');
  readBytes(f, s.data(), n);
  return s;
}

std::uint16_t halfFromFloat(float f) {
  std::uint32_t u = 0;
  std::memcpy(&u, &f, sizeof(u));
  const std::uint32_t sign = (u >> 16) & 0x8000u;
  const std::int32_t exp = static_cast<std::int32_t>((u >> 23) & 0xff) - 127 + 15;
  const std::uint32_t mant = u & 0x7fffffu;
  if (exp >= 0x1f) {
    // Inf / NaN (NaN payload is intentionally truncated).
    return static_cast<std::uint16_t>(sign | 0x7c00u);
  }
  if (exp <= 0) {
    if (exp < -10) return static_cast<std::uint16_t>(sign);
    std::uint32_t m = mant | 0x800000u;
    const std::uint32_t shift = static_cast<std::uint32_t>(14 - exp);
    std::uint32_t half = m >> shift;
    const std::uint32_t rem = m & ((1u << shift) - 1u);
    const std::uint32_t halfway = 1u << (shift - 1u);
    if (rem > halfway || (rem == halfway && (half & 1u))) ++half;
    return static_cast<std::uint16_t>(sign | half);
  }
  std::uint32_t half =
      (static_cast<std::uint32_t>(exp) << 10) | (mant >> 13);
  const std::uint32_t rem = mant & 0x1fffu;
  if (rem > 0x1000u || (rem == 0x1000u && (half & 1u)))
    ++half;
  if ((half & 0x7c00u) == 0x7c00u)
    return static_cast<std::uint16_t>(sign | 0x7c00u);
  return static_cast<std::uint16_t>(sign | half);
}

}  // namespace

float halfBitsToFloat(std::uint16_t h) {
  const std::uint32_t sign = static_cast<std::uint32_t>(h & 0x8000u) << 16;
  const std::uint32_t exp = (h >> 10) & 0x1fu;
  const std::uint32_t mant = h & 0x3ffu;
  std::uint32_t u;
  if (exp == 0) {
    if (mant == 0) {
      u = sign;
    } else {
      int e = -1;
      std::uint32_t m = mant;
      do {
        ++e;
        m <<= 1;
      } while ((m & 0x400u) == 0);
      m &= 0x3ffu;
      u = sign | (static_cast<std::uint32_t>(127 - 15 - e) << 23) | (m << 13);
    }
  } else if (exp == 0x1f) {
    u = sign | 0x7f800000u | (mant << 13);
  } else {
    u = sign | ((exp - 15 + 127) << 23) | (mant << 13);
  }
  float f = 0.f;
  std::memcpy(&f, &u, sizeof(f));
  return f;
}

std::uint16_t floatToHalfBits(float f) { return halfFromFloat(f); }

void normalizeRowsToUnit(float* rows, i64 n, i32 dim) {
  for (i64 i = 0; i < n; ++i) {
    float* r = rows + static_cast<std::size_t>(i) * dim;
    double s = 0.0;
    for (i32 j = 0; j < dim; ++j) s += static_cast<double>(r[j]) * r[j];
    const double inv = 1.0 / std::sqrt(std::max(s, 1e-30));
    for (i32 j = 0; j < dim; ++j) r[j] = static_cast<float>(r[j] * inv);
  }
}

void cosineNormalize(Dataset* ds) {
  normalizeRowsToUnit(ds->data.data(), ds->n, ds->dim);
}

void readVectorFile(const std::string& path, Dataset* out) {
  FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) throwRuntime("cannot open vector file: " + path);
  try {
    const i32 magic = readI32(f);
    const i32 version = readI32(f);
    if (magic != kFileMagic || version != kFileVersion)
      throwRuntime("bad vector-file header in " + path);
    const i64 n = readI64(f);
    const i32 dim = readI32(f);
    if (n < 0 || dim <= 0 || dim > 65536)
      throwRuntime("implausible vector file shape in " + path);
    const DataType dt = parseDataType(readString(f));
    const Metric met = parseMetric(readString(f));
    if (n > 0 && (i64)(static_cast<std::size_t>(n) * dim) >
                     (i64)((1ull << 40) / sizeof(float)))
      throwRuntime("vector file too large in " + path);

    out->n = n;
    out->dim = dim;
    out->dtype = dt;
    out->metric = met;
    out->data.assign(static_cast<std::size_t>(n) * dim, 0.0f);

    const std::size_t total = static_cast<std::size_t>(n) * dim;
    if (dt == DataType::Fp32) {
      readBytes(f, out->data.data(), total * sizeof(float));
    } else {
      std::vector<std::uint16_t> half(total);
      readBytes(f, half.data(), total * sizeof(std::uint16_t));
      for (std::size_t i = 0; i < total; ++i)
        out->data[i] = halfBitsToFloat(half[i]);
    }
  } catch (...) {
    std::fclose(f);
    throw;
  }
  std::fclose(f);
}

void writeVectorFile(const std::string& path, i64 n, i32 dim, DataType dtype,
                     Metric metric, const float* rowMajor, i64 count) {
  if (count != static_cast<i64>(n) * dim)
    throwRuntime("writeVectorFile: element count mismatch");
  FILE* f = std::fopen(path.c_str(), "wb");
  if (!f) throwRuntime("cannot open output file: " + path);
  try {
    writeI32(f, kFileMagic);
    writeI32(f, kFileVersion);
    writeI64(f, n);
    writeI32(f, dim);
    writeString(f, dtype == DataType::Fp32 ? "fp32" : "fp16");
    writeString(f, metricName(metric));
    if (dtype == DataType::Fp32) {
      writeBytes(f, rowMajor, static_cast<std::size_t>(count) * sizeof(float));
    } else {
      const std::size_t total = static_cast<std::size_t>(count);
      std::vector<std::uint16_t> half(total);
      for (std::size_t i = 0; i < total; ++i)
        half[i] = floatToHalfBits(rowMajor[i]);
      writeBytes(f, half.data(), total * sizeof(std::uint16_t));
    }
  } catch (...) {
    std::fclose(f);
    throw;
  }
  std::fclose(f);
}

}  // namespace vs
