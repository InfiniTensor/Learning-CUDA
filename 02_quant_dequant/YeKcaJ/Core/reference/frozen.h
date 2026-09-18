#pragma once

#include <cstdint>
#include <vector>

// 两种 CPU 格式的统一测试视图：NVFP4 的 data 为 packed，scales 为 E4M3。
// MXFP8 的 data 为 E4M3，scales 为 E8M0，global_scale 占位为 1。
struct ReferenceResult {
  float global_scale = 1.0f;
  std::vector<std::uint8_t> data;
  std::vector<std::uint8_t> scales;
  std::vector<float> dequant;
};
// 量化并反量化生成预期结果；下列编码接口供舍入边界测试使用。
ReferenceResult cpu_reference(const std::vector<float>& values);
float cpu_decode4(std::uint8_t code);
std::uint8_t cpu_encode4(float value);
std::uint8_t cpu_encode2(float value);
