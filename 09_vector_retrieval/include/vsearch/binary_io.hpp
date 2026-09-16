#pragma once

#include "vsearch/common.hpp"

namespace vs {

// ---------------------------------------------------------------------------
// On-disk layout shared by the vector database and query files.
//
//   magic   : i32 = 0x56534348  ("VSCH")
//   version : i32 = 1
//   n       : i64               number of rows
//   dim     : i32
//   dtype   : u8 length + ASCII ("fp32" or "fp16")
//   metric  : u8 length + ASCII ("l2" / "inner_product" / "cosine" / "none")
//   payload : n*dim raw elements, row-major
//
// The reader always returns fp32 in memory; the writer optionally converts
// fp32 host data down to fp16 for realistic memory-bandwidth experiments.
// ---------------------------------------------------------------------------

constexpr i32 kFileMagic = 0x56534348;
constexpr i32 kFileVersion = 1;

void readVectorFile(const std::string& path, Dataset* out);
void writeVectorFile(const std::string& path, i64 n, i32 dim, DataType dtype,
                     Metric metric, const float* rowMajor, i64 count);

inline void writeVectorFile(const std::string& path, i64 n, i32 dim,
                            DataType dtype, Metric metric,
                            const FloatVec& rowMajor) {
  writeVectorFile(path, n, dim, dtype, metric, rowMajor.data(), rowMajor.size());
}

float halfBitsToFloat(std::uint16_t h);
std::uint16_t floatToHalfBits(float f);

// Normalize every row to unit L2 norm (used only for cosine metric).
void normalizeRowsToUnit(float* rows, i64 n, i32 dim);
void cosineNormalize(Dataset* ds);

}  // namespace vs
