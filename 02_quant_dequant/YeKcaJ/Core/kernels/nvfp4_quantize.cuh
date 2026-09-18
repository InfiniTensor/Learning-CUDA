#pragma once

// NVFP4 默认量化算子：block + nearest，每字节打包两个元素。
#include "kernels/scale.cuh"

namespace pipeline {

// NVFP4 block+nearest 融合 kernel：一个 warp 覆盖 4 个分组（每线程 2 元素）。
// 组内用 width=kBlockSize/2 的 shuffle 归约最大值，组首算 scale 后广播，
// 复用寄存器中的原值编码并打包，input 只读取一次。
// global_scale 由前面的 maximum/finalize_max 提供，此处只做 per-block 工作。
// scale 公式、FP32 除法与 s > 0 的守卫保持与原路径一致。
__global__ void nvfp4_quantize_fused_kernel(const float* input, std::uint8_t* data,
                                           std::uint8_t* scales, const float* state,
                                           std::size_t n) {
  // 每个线程负责一个输出字节，即两个连续元素。
  const std::size_t item = blockIdx.x * 256ull + threadIdx.x;
  const std::size_t i = item * 2;
  const unsigned lane = threadIdx.x & 31;
  constexpr unsigned kLanesPerGroup = kBlockSize / 2;

  const float low_value = i < n ? input[i] : 0.0f;
  const float high_value = i + 1 < n ? input[i + 1] : 0.0f;

  // 组内归约：width 限定使每个 kBlockSize/2 个 lane 独立归约。
  float maximum = fmaxf(fabsf(low_value), fabsf(high_value));
  for (unsigned step = kLanesPerGroup / 2; step; step >>= 1)
    maximum = fmaxf(maximum, __shfl_down_sync(0xffffffffu, maximum, step, kLanesPerGroup));

  // 组首线程写 scale；其余 lane 用广播值。i 是组内第一个元素，i < n 即组有效。
  unsigned code = 0;
  if (lane % kLanesPerGroup == 0 && i < n) {
    code = scale_code(maximum, state[1]);
    scales[i / kBlockSize] = code;
  }
  code = __shfl_sync(0xffffffffu, code, 0, kLanesPerGroup);

  if (i < n) {
    const float scale = state[1] * decode_e4m3(static_cast<std::uint8_t>(code));
    const unsigned low = encode(scale > 0 ? low_value / scale : 0, true, false, 0, 0, false);
    const unsigned high = i + 1 < n
                              ? encode(scale > 0 ? high_value / scale : 0, true, false, 0, 0, false)
                              : 0;
    // 偶数元素放低 4 位、奇数放高 4 位；奇数尾部的 nibble 保持为零。
    data[item] = static_cast<std::uint8_t>(low | (high << 4));
  }
}

}  // namespace pipeline
