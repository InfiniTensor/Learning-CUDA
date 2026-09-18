#pragma once

// 输入 FP32/FP16 文件及 v2 packed 文件读写；格式保持不变。
#include "common/types.h"
#include <cuda_fp16.h>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <set>

namespace pipeline {

namespace fs = std::filesystem;

// 当前目标平台为小端；逐字段读取文件头，避免直接读取含编译器填充的结构体。
template <typename T>
T read_scalar(std::istream& f) {
  T value{};
  if (!f.read(reinterpret_cast<char*>(&value), sizeof(T)))
    throw std::runtime_error("truncated header");
  return value;
}

// 与 read_scalar 对应；写入顺序和字段宽度决定磁盘格式。
template <typename T>
void write_scalar(std::ostream& f, T value) {
  f.write(reinterpret_cast<const char*>(&value), sizeof(T));
}

// 创建父目录并打开二进制输出；已有同名文件会被截断，防覆盖由上层入口处理。
inline std::ofstream output_file(const std::string& path) {
  if (!fs::path(path).parent_path().empty())
    fs::create_directories(fs::path(path).parent_path());
  std::ofstream out(path, std::ios::binary);
  if (!out)
    throw std::runtime_error("cannot write: " + path);
  return out;
}

// ===== 3. 输入张量文件 =====

// 读取 28 字节头（magic、version、rows、cols）及行主序数据。
// 校验完整文件长度，拒绝 NaN/Inf；FP16 转 FP32 后交给 GPU 量化。
inline Tensor read_input(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  char magic[8]{};
  if (!f.read(magic, 8))
    throw std::runtime_error("cannot read input: " + path);
  const std::string tag(magic, 8);
  if (tag != "FP32INP1" && tag != "FP16INP1")
    throw std::runtime_error("invalid input magic");
  if (read_scalar<std::uint32_t>(f) != 1)
    throw std::runtime_error("invalid input version");
  Tensor t;
  t.rows = read_scalar<std::uint64_t>(f);
  t.cols = read_scalar<std::uint64_t>(f);
  t.element_bytes = tag == "FP32INP1" ? 4 : 2;
  const auto n = checked_count(t.rows, t.cols);
  if (fs::file_size(path) < 28 || fs::file_size(path) - 28 != n * t.element_bytes)
    throw std::runtime_error("input payload length mismatch");
  t.values.resize(n);
  if (t.element_bytes == 4) {
    f.read(reinterpret_cast<char*>(t.values.data()), n * 4);
  } else {
    // FP16 输入先无损扩展成 FP32；GPU 使用同一量化公式，文件压缩率按原 dtype 计算。
    for (float& v : t.values) {
      const auto bits = read_scalar<std::uint16_t>(f);
      __half h;
      std::memcpy(&h, &bits, 2);
      v = __half2float(h);
    }
  }
  for (float v : t.values)
    if (!std::isfinite(v))
      throw std::runtime_error("quantization input must be finite (NaN/Inf rejected)");
  return t;
}

// ===== 4. v2 低精度权重文件 =====

// 写入顺序：72 字节头 -> packed data -> scales；global_scale 位于头中。
// 头包含格式、形状、分组大小、模式、舍入、seed 和两段数据的字节数。
inline void write_packed(const std::string& path, const Packed& q) {
  auto f = output_file(path);
  f.write("LPQUANT2", 8);
  write_scalar<std::uint32_t>(f, 2);
  write_scalar<std::uint32_t>(f, nvfp4 ? 4 : 8);
  write_scalar(f, q.rows);
  write_scalar(f, q.cols);
  write_scalar(f, q.group);
  write_scalar<std::uint32_t>(f, q.options.tensor);
  write_scalar<std::uint32_t>(f, q.options.stochastic);
  write_scalar(f, q.options.seed);
  write_scalar(f, q.global);
  write_scalar<std::uint64_t>(f, q.data.size());
  write_scalar<std::uint64_t>(f, q.scales.size());
  f.write(reinterpret_cast<const char*>(q.data.data()), q.data.size());
  f.write(reinterpret_cast<const char*>(q.scales.data()), q.scales.size());
  if (!f)
    throw std::runtime_error("packed output write failed");
}

// 读取 .lpq 并校验版本/格式、尺寸、scale 编码范围和 NVFP4 奇数尾部填充。
// 当前二进制只接受自身编译格式，不能用 MXFP8 程序读取 NVFP4 权重。
inline Packed read_packed(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  char magic[8]{};
  if (!f.read(magic, 8) || std::string(magic, 8) != "LPQUANT2")
    throw std::runtime_error("invalid packed magic");
  if (read_scalar<std::uint32_t>(f) != 2 || read_scalar<std::uint32_t>(f) != (nvfp4 ? 4u : 8u))
    throw std::runtime_error("packed version/format mismatch");
  Packed q;
  q.rows = read_scalar<std::uint64_t>(f);
  q.cols = read_scalar<std::uint64_t>(f);
  q.group = read_scalar<std::uint64_t>(f);
  const auto mode = read_scalar<std::uint32_t>(f), rounding = read_scalar<std::uint32_t>(f);
  if (mode > 1 || rounding > 1)
    throw std::runtime_error("invalid packed mode");
  q.options.tensor = mode;
  q.options.stochastic = rounding;
  q.options.seed = read_scalar<std::uint32_t>(f);
  q.global = read_scalar<float>(f);
  const auto bytes = read_scalar<std::uint64_t>(f), scales = read_scalar<std::uint64_t>(f);
  const auto n = checked_count(q.rows, q.cols);
  const auto expected_group = mode ? std::max<std::size_t>(1, n) : kBlockSize;
  if (q.group != expected_group || bytes != data_size(n) || scales != ceil_div(n, expected_group) ||
      !std::isfinite(q.global) || q.global <= 0 || (!nvfp4 && q.global != 1))
    throw std::runtime_error("invalid packed dimensions/scales");
  if (fs::file_size(path) < 72 || fs::file_size(path) - 72 != bytes + scales)
    throw std::runtime_error("packed payload length mismatch");
  q.data.resize(bytes);
  q.scales.resize(scales);
  f.read(reinterpret_cast<char*>(q.data.data()), bytes);
  f.read(reinterpret_cast<char*>(q.scales.data()), scales);
  for (auto s : q.scales)
    if (s > (nvfp4 ? 126 : 254))
      throw std::runtime_error("invalid scale code");
  if (nvfp4 && n % 2 && (q.data.back() & 0xf0))
    throw std::runtime_error("nonzero tail padding");
  return q;
}

}  // namespace pipeline
