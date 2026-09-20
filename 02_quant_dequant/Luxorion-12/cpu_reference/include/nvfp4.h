#pragma once

// 模块接口：声明E2M1单值转换、NVFP4结果结构和CPU量化/反量化入口。
// NVFP4Result将有效长度、两级scale和打包字节一起交给恢复端，避免依赖原始输入。
// 具体查表、分块与填充处理见src/nvfp4_cpu.cpp。

#include <cstdint>
#include <cstddef>
#include <vector>

namespace luxorion {

// 最近邻，中点 ties-to-even；有限值饱和到 +/-6，保留零的符号。
// NaN/Inf 抛出 std::invalid_argument；返回值只使用低四位。
std::uint8_t encode_e2m1(float value);
// 只接受 0..15；更大的编码抛出 std::invalid_argument。
float decode_e2m1(std::uint8_t code);

struct NVFP4Result {
  // 有效元素数量，不含尾块补零；反量化输出严格保持这一长度。
  std::size_t element_count = 0;
  // 每 16 个有效元素一块，每块固定 8 字节；first 在高四位。
  // 最后一个块的无效编码全部填 0，原始长度由 element_count 决定。
  std::vector<std::uint8_t> packed_data;
  // 每块一个E4M3编码，而不是直接存浮点scale；第i个元素属于块i/16。
  std::vector<std::uint8_t> block_scales;
  // 整个输入共用一个FP32数值，直接存储，不经过E4M3/E8M0编码。
  // 第i个元素的有效scale=global_scale*decode_e4m3(block_scales[i/16])。
  float global_scale = 1.0F;
};

// FP32 全局 scale、E4M3 局部 scale、E2M1 元素；行主序连续分块。
// 拒绝空输入和 NaN/Inf；scale 下溢为零抛出 std::underflow_error。
NVFP4Result nvfp4_quantize_cpu(const std::vector<float>& input);
// 校验尺寸、正的有限 scale 和零填充；只恢复 element_count 个元素。
// scale 下溢或恢复溢出分别抛出 std::underflow_error / std::overflow_error。
std::vector<float> nvfp4_dequantize_cpu(const NVFP4Result& q);

}  // namespace luxorion
