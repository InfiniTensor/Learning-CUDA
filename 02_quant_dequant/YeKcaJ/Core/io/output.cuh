#pragma once

// 反量化文件写出；数值转换在 common/output_type.cuh，校验在 reference/。
#include "common/output_type.cuh"
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace cuda_output {

// 固定 36 字节头部后直接保存 T 数组；FP16/BF16 的每个元素占 2 字节。
template <typename T>
void write(const std::string& path, std::size_t rows, std::size_t cols,
           const std::vector<T>& values) {
  std::ofstream file(path, std::ios::binary);
  if (!file) throw std::runtime_error("cannot open CUDA output: " + path);
  const std::uint32_t version = 1;
  const std::uint64_t r = rows, c = cols, n = values.size();
  file.write(Format<T>::magic, 8);
  file.write(reinterpret_cast<const char*>(&version), sizeof(version));
  file.write(reinterpret_cast<const char*>(&r), sizeof(r));
  file.write(reinterpret_cast<const char*>(&c), sizeof(c));
  file.write(reinterpret_cast<const char*>(&n), sizeof(n));
  if (!values.empty())
    file.write(reinterpret_cast<const char*>(values.data()), values.size() * sizeof(T));
  if (!file) throw std::runtime_error("failed writing CUDA output: " + path);
}

}  // namespace cuda_output
