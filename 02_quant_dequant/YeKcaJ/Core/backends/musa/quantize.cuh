#pragma once
#include "kernels/scale.cuh"

namespace pipeline {

// 默认 block+nearest 路径；步骤 1 的标量版本保存在 records/musa-opt01/step1 源码快照。

// 每线程读取 float4：MXFP8 写 4 字节，NVFP4 写 2 字节，每组只计算一次 scale。
// 完整向量才使用宽读写；尾部逐元素读取、逐字节写出，保持奇数 FP4 的高 nibble 为零。
__global__ void musa_quantize_vectorized_kernel(const float* input, std::uint8_t* data,
                                               std::uint8_t* scales, const float* state,
                                               std::size_t n) {
  constexpr unsigned width = kBlockSize / 4;
  const unsigned tid = threadIdx.x;
  const std::size_t base = (blockIdx.x * 256ull + tid) * 4;
  float values[4] = {};
  if (base + 3 < n) {
    const float4 v = *reinterpret_cast<const float4*>(input + base);
    values[0] = v.x; values[1] = v.y; values[2] = v.z; values[3] = v.w;
  } else {
    for (unsigned j = 0; j < 4; ++j)
      if (base + j < n) values[j] = input[base + j];
  }
  float maximum = 0;
  for (unsigned j = 0; j < 4; ++j) maximum = fmaxf(maximum, fabsf(values[j]));
  for (unsigned step = width / 2; step; step >>= 1)
    maximum = fmaxf(maximum, __shfl_down_sync(0xffffffffu, maximum, step, width));

  // 只保留一次整块同步，用 shared scale 广播保证零和次正规数路径一致。
  __shared__ unsigned codes[256 / width];
  if (tid % width == 0) {
    const unsigned code = scale_code(maximum, nvfp4 ? state[1] : 1.0f);
    codes[tid / width] = code;
    if (base < n) scales[base / kBlockSize] = code;
  }
  __syncthreads();
  if (base >= n) return;
  const auto code = static_cast<std::uint8_t>(codes[tid / width]);
  const float scale = nvfp4 ? multiply_rn(state[1], decode_e4m3(code))
                           : scalbnf(1.0f, int(code) - 127);
  std::uint32_t packed = 0;
  for (unsigned j = 0; j < 4; ++j) {
    if (base + j < n) {
      const unsigned encoded = encode(scale > 0 ? divide_rn(values[j], scale) : 0,
                                       nvfp4, false, 0, base + j);
      packed |= encoded << (j * (nvfp4 ? 4 : 8));
    }
  }
  const std::size_t offset = nvfp4 ? base / 2 : base;
  if (base + 3 < n) {
    if (nvfp4) *reinterpret_cast<std::uint16_t*>(data + offset) = static_cast<std::uint16_t>(packed);
    else *reinterpret_cast<std::uint32_t*>(data + offset) = packed;
  } else {
    const unsigned remaining = static_cast<unsigned>(n - base);
    const unsigned bytes = nvfp4 ? (remaining + 1) / 2 : remaining;
    for (unsigned j = 0; j < bytes; ++j)
      data[offset + j] = static_cast<std::uint8_t>(packed >> (8 * j));
  }
}
}  // namespace pipeline
