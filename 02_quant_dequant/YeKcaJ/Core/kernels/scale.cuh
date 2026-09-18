#pragma once

// scale 字节编码与实际乘数恢复；两种格式共用。
#include "kernels/codec.cuh"
#include "common/device_math.cuh"

namespace pipeline {

// 把分组最大绝对值 m 转成 scale 字节：NVFP4 用 E4M3，MXFP8 用 E8M0。
__device__ inline std::uint8_t scale_code(float m, float global) {
  if (nvfp4) return encode_e4m3_nearest(m > 0 ? divide_rn(m, multiply_rn(6.0f, global)) : 0.0f);
  if (m == 0) return 127;
  const float ratio = divide_rn(m, 448.0f);
  if (ratio == 0) return 0;
  return max(0, min(254, static_cast<int>(ceilf(log2f(ratio))) + 127));
}

// 恢复实际乘数：NVFP4 为 global_scale * block_scale，MXFP8 为 2^(字节-127)。
__device__ inline float effective(const std::uint8_t* scales, const float* state,
                                 std::size_t group) {
  return nvfp4 ? multiply_rn(state[1], decode_e4m3(scales[group]))
               : scalbnf(1.0f, static_cast<int>(scales[group]) - 127);
}

}  // namespace pipeline
