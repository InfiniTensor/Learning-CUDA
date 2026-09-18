#pragma once

// MXFP8 默认量化算子：block + nearest。
#include "kernels/scale.cuh"

namespace pipeline {

// MXFP8 block+nearest 融合 kernel：一个 warp 处理 32 元素，输入只读取一次。
// 尾部补零并参与所有 shuffle；组首计算 scale 后广播，复用寄存器中的原值编码。
// scale 公式及 FP32 除法保持原样，不引入 fast-math 或倒数近似。
__global__ void mxfp8_quantize_fused_kernel(const float* input, std::uint8_t* data,
                                          std::uint8_t* scales, std::size_t n) {
  const std::size_t i = blockIdx.x * 256ull + threadIdx.x;
  const unsigned lane = threadIdx.x & 31;
  const float value = i < n ? input[i] : 0.0f;
  float maximum = fabsf(value);
  for (unsigned step = 16; step; step >>= 1)
    maximum = fmaxf(maximum, __shfl_down_sync(0xffffffffu, maximum, step));

  unsigned code = 127;
  if (lane == 0 && i < n) {
    code = scale_code(maximum, 1.0f);
    scales[i / 32] = code;
  }
  code = __shfl_sync(0xffffffffu, code, 0);
  if (i < n) {
    const float scale = scalbnf(1.0f, static_cast<int>(code) - 127);
    data[i] = encode_e4m3_nearest(value / scale);
  }
}

// 向量化版本：每个线程处理 4 个连续元素，一个 8-lane 子组覆盖 32 元素。
// float4 读取并将 4 个 E4M3 字节合并为一次 uint32 写入；尾部不足 4 个元素时回退标量读取。
__global__ void mxfp8_quantize_vectorized_kernel(const float* input, std::uint8_t* data,
                                                 std::uint8_t* scales, std::size_t n) {
  constexpr unsigned kLanesPerGroup = 8;
  const std::size_t item = blockIdx.x * 256ull + threadIdx.x;
  const std::size_t group = item / kLanesPerGroup;
  const unsigned lane = threadIdx.x & 31;
  const unsigned sublane = lane & (kLanesPerGroup - 1);
  const std::size_t base = group * 32ull + static_cast<std::size_t>(sublane) * 4ull;

  float values[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  if (base + 3 < n) {
    const float4 v = *reinterpret_cast<const float4*>(input + base);
    values[0] = v.x; values[1] = v.y; values[2] = v.z; values[3] = v.w;
  } else {
    for (unsigned j = 0; j < 4; ++j)
      if (base + j < n) values[j] = input[base + j];
  }

  float maximum = 0.0f;
  for (unsigned j = 0; j < 4; ++j) maximum = fmaxf(maximum, fabsf(values[j]));
  for (unsigned step = 4; step; step >>= 1)
    maximum = fmaxf(maximum, __shfl_down_sync(0xffffffffu, maximum, step, kLanesPerGroup));

  unsigned code = 127;
  if (sublane == 0 && base < n) {
    code = scale_code(maximum, 1.0f);
    scales[group] = static_cast<std::uint8_t>(code);
  }
  code = __shfl_sync(0xffffffffu, code, 0, kLanesPerGroup);
  const float scale = scalbnf(1.0f, static_cast<int>(code) - 127);

  std::uint32_t packed = 0;
  for (unsigned j = 0; j < 4; ++j)
    if (base + j < n)
      packed |= static_cast<std::uint32_t>(encode_e4m3_nearest(values[j] / scale)) << (8 * j);
  if (base + 3 < n) {
    *reinterpret_cast<std::uint32_t*>(data + base) = packed;
  } else {
    // 尾部不足 4 字节时逐字节写，避免 uint32 存储越过输出缓冲区。
    for (unsigned j = 0; j < 4 && base + j < n; ++j)
      data[base + j] = static_cast<std::uint8_t>(packed >> (8 * j));
  }
}

}  // namespace pipeline
