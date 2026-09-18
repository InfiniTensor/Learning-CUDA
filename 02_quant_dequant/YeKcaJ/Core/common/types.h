#pragma once

// 共享数据结构与格式常量；不含 CUDA kernel、文件读写或 CPU 算法。
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace pipeline {

#ifdef TEST_NVFP4
constexpr std::size_t kBlockSize = 16;
#else
constexpr std::size_t kBlockSize = 32;
#endif
// 同一套源代码编译两次，TEST_NVFP4 由 NVFP4 项目的构建配置定义。
#ifdef TEST_NVFP4
constexpr bool nvfp4 = true;
constexpr const char* format_name = "nvfp4";
#else
constexpr bool nvfp4 = false;
constexpr const char* format_name = "mxfp8";
#endif
// tensor=true 表示整张量共享 scale；stochastic=true 表示元素随机舍入。
struct Options {
  bool tensor = false, stochastic = false;
  std::size_t block = kBlockSize;
  std::uint32_t seed = 1234;
};

// 文件可以是 FP16 或 FP32；读取后统一用 FP32 values 存储，保留原元素字节数。
struct Tensor {
  std::uint64_t rows = 0, cols = 0;
  unsigned element_bytes = 4;
  std::vector<float> values;
};

// data 存低精度编码，scales 每分组一个字节；global 仅 NVFP4 使用。
// NVFP4 的 data 每字节两个元素，不能按每元素一个 uint8_t 存储。
struct Packed {
  std::uint64_t rows = 0, cols = 0, group = kBlockSize;
  Options options;
  float global = 1;
  std::vector<std::uint8_t> data, scales;
};
// 避免 (n+d-1)/d 中加法溢出；调用方保证 d 非零。
inline std::size_t ceil_div(std::size_t n, std::size_t d) {
  return n / d + (n % d != 0);
}

// 只计算 data 区字节数，不包含 scale 和文件头；奇数 NVFP4 长度向上取整。
inline std::size_t data_size(std::size_t n) {
  return nvfp4 ? ceil_div(n, 2) : n;
}

// 在分配内存之前检查行列乘积和 FP32 字节数，拒绝溢出的尺寸。
inline std::size_t checked_count(std::uint64_t rows, std::uint64_t cols) {
  if (rows > INT64_MAX || cols > INT64_MAX || (cols && rows > SIZE_MAX / cols))
    throw std::runtime_error("invalid/overflowing tensor dimensions");
  const auto n = static_cast<std::size_t>(rows * cols);
  if (n > SIZE_MAX / sizeof(float))
    throw std::runtime_error("tensor byte size overflow");
  return n;
}

// block 大小固定为格式规定值；tensor 模式通过模式字段表达，不改 block_size。
inline void validate(const Options& o) {
  if (o.block != kBlockSize)
    throw std::runtime_error(
        "block_size must be 32 for MXFP8 or 16 for NVFP4; use scale_mode=tensor for a shared scale");
}

}  // namespace pipeline
