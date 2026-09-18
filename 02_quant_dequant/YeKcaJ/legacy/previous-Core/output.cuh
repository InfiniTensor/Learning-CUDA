#pragma once

// 正式输出模块；旧 CUDACommon/output.cuh 仅转发到本文件。
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace cuda_output {

// 公共输出辅助代码，不含量化算子；T 决定转换方式和文件类型标识。
// 16 位输出使用 nearest-even，供反量化 kernel 和主机 golden 比较共同调用。
template <typename T> struct Format;
template <> struct Format<float> {
  static constexpr const char* name = "fp32";
  static constexpr const char* magic = "FP32DEQ1";
  __host__ __device__ static float convert(float x) { return x; }
};
template <> struct Format<__half> {
  static constexpr const char* name = "fp16";
  static constexpr const char* magic = "FP16DEQ1";
  __host__ __device__ static __half convert(float x) { return __float2half_rn(x); }
};
template <> struct Format<__nv_bfloat16> {
  static constexpr const char* name = "bf16";
  static constexpr const char* magic = "BF16DEQ1";
  __host__ __device__ static __nv_bfloat16 convert(float x) { return __float2bfloat16_rn(x); }
};

// 只接受末尾的输出类型选项，并缩短 argc，供原有位置参数解析继续使用。
inline std::string parse_dtype(int& argc, char** argv) {
  std::string dtype = "fp32";
  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) != "--output-dtype") continue;
    if (i != argc - 2) throw std::runtime_error("append --output-dtype fp32|fp16|bf16 at the end");
    dtype = argv[i + 1];
    if (dtype != "fp32" && dtype != "fp16" && dtype != "bf16")
      throw std::runtime_error("invalid output dtype: " + dtype);
    argc -= 2;
    if (argc > 1 && std::string(argv[1]) == "--quantize")
      throw std::runtime_error("--output-dtype applies to dequantization only");
    break;
  }
  return dtype;
}

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

// 比较的是 CUDA 与 CPU 反量化结果，不是原始输入的量化损失。
// 16 位结果与转换后的 golden 逐位比较；FP32 有限值使用绝对误差阈值。
template <typename T>
void compare(const std::vector<T>& actual, const std::vector<float>& golden) {
  if (actual.size() != golden.size()) throw std::runtime_error("dequant shape mismatch");
  std::size_t mismatches = 0;
  double max_diff = 0.0;
  for (std::size_t i = 0; i < actual.size(); ++i) {
    const T expected = Format<T>::convert(golden[i]);
    const float a = static_cast<float>(actual[i]);
    const float e = static_cast<float>(expected);
    // NaN 比较类别，不要求 payload 一致；Inf 必须同号，零保留符号。
    bool equal = false;
    if (std::isnan(e)) equal = std::isnan(a);
    else if constexpr (std::is_same_v<T, float>) {
      if (std::isfinite(a) && std::isfinite(e)) {
        const double diff = std::fabs(static_cast<double>(a) - e);
        max_diff = std::max(max_diff, diff);
        equal = diff <= 1e-6 && !(a == 0 && e == 0 && std::signbit(a) != std::signbit(e));
      } else equal = a == e;
    } else {
      equal = std::memcmp(&actual[i], &expected, sizeof(T)) == 0;
    }
    if (!equal) ++mismatches;
  }
  std::cout << "output_dtype=" << Format<T>::name << " element_mismatches=" << mismatches;
  if constexpr (std::is_same_v<T, float>) std::cout << " max_abs_diff=" << max_diff;
  else std::cout << " comparison=rounded_cpu_bits";
  std::cout << '\n';
  if (mismatches) throw std::runtime_error("CUDA/CPU dequant comparison failed");
}

}  // namespace cuda_output
