#pragma once

// 两种格式共用的反量化算子，模板 T 选择 FP32/FP16/BF16。
#include "kernels/scale.cuh"
#include "common/output_type.cuh"

namespace pipeline {

// Kernel 6：读取低精度 data 和 scale，恢复数值并转换成目标输出类型 T。
// 固定 256 线程/块，每线程输出一个元素；NVFP4 先从 packed 字节提取对应 4 位。
// 先以 FP32 计算 decode(data) * effective_scale，再输出 FP32、FP16 或 BF16。
template <typename T>
__global__ void dequantize_kernel(const std::uint8_t* data, const std::uint8_t* scales, const float* state,
                                 T* output, std::size_t n, std::size_t group) {
  const auto i = blockIdx.x * 256ull + threadIdx.x;
  if (i >= n) return;

  const unsigned code = nvfp4 ? ((data[i / 2] >> (4 * (i % 2))) & 15) : data[i];
  const float value = nvfp4 ? ((code & 8) ? -magnitude2(code & 7) : magnitude2(code & 7))
                            : decode_e4m3(code);
  output[i] = cuda_output::Format<T>::convert(
      multiply_rn(value, effective(scales, state, group == kBlockSize ? i / kBlockSize : 0)));
}

}  // namespace pipeline
