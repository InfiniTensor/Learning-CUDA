#include "reference.h"

// 单独编译目录外的冻结 CPU 源码，直接复用原函数，不移动或复制量化算法。
// 改名 CPU main 防止入口重名；TEST_NVFP4 由对应 CUDA 工程的 CMake 定义。
#define main frozen_reference_main
#ifdef TEST_NVFP4
#include "../../CPUNVFP4/main.cpp"
#else
#include "../../CPUMXFP8/main.cpp"
#endif
#undef main

// 把两种 CPU 结果转换为相同结构，隐藏 packed/data 等字段命名差异。
ReferenceResult cpu_reference(const std::vector<float>& values) {
  const auto q = quantize(values, 1, values.size());
#ifdef TEST_NVFP4
  return {q.global_scale, q.packed, q.block_scales, dequantize(q)};
#else
  return {1.0f, q.data, q.scales, dequantize(q)};
#endif
}
float cpu_decode4(std::uint8_t code) { return decode_e4m3(code); }
std::uint8_t cpu_encode4(float value) { return encode_e4m3(value); }
std::uint8_t cpu_encode2(float value) {
#ifdef TEST_NVFP4
  return encode_e2m1(value);
#else
  return 0;
#endif
}
