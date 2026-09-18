#pragma once

// 反量化目标类型的数值转换；无文件操作、无 CPU 对照。
#include <cuda_fp16.h>
#include <cuda_bf16.h>

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

}  // namespace cuda_output
