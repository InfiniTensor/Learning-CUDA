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
//
// 中文说明：向量库与查询文件共用同一容器格式（查询文件的 metric 为 none）。
// 读取端统一还原为 fp32 再送 GPU，因此 fp16 只影响磁盘/带宽占用，不影响
// 检索语义；写端可选择降为 fp16 以做显存/带宽受限实验。
// ---------------------------------------------------------------------------

constexpr i32 kFileMagic = 0x56534348;
constexpr i32 kFileVersion = 1;

// 读取向量文件到 Dataset（自动完成 fp16->fp32 转换，并校验头部合法性）。
void readVectorFile(const std::string& path, Dataset* out);
// 写出向量文件（与读取端对称，供测试与数据生成工具使用）。
void writeVectorFile(const std::string& path, i64 n, i32 dim, DataType dtype,
                     Metric metric, const float* rowMajor, i64 count);

// FloatVec 重载：内部对齐分配器版本，方便直接传 Dataset::data。
inline void writeVectorFile(const std::string& path, i64 n, i32 dim,
                            DataType dtype, Metric metric,
                            const FloatVec& rowMajor) {
  writeVectorFile(path, n, dim, dtype, metric, rowMajor.data(), rowMajor.size());
}

// IEEE-754 half <-> float 的位级转换（不依赖 GPU / 半精度指令）。
float halfBitsToFloat(std::uint16_t h);
std::uint16_t floatToHalfBits(float f);

// Normalize every row to unit L2 norm (used only for cosine metric).
// 逐行 L2 归一化（仅 cosine 度量）：库向量与查询都在进入检索前归一化，
// 使距离计算退化为 1 - 内积。
void normalizeRowsToUnit(float* rows, i64 n, i32 dim);
void cosineNormalize(Dataset* ds);

}  // namespace vs
