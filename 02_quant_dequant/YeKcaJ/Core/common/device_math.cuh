#pragma once
#include <cuda_runtime.h>

namespace pipeline {
// MUSA 普通 FP32 除法不能满足冻结规则；使用宽精度商并以 RN 转回 FP32。
__device__ inline float divide_rn(float a, float b) {
#ifdef LP_BACKEND_MUSA
  // MUSA 的 __fdiv_rn 对每个线程代价很高；FP32 操作数转 FP64 后相除，
  // 53 位有效数足以保证一次转换到 FP32 与 IEEE RN 相同（含次正规结果）。
  return __double2float_rn(static_cast<double>(a) / static_cast<double>(b));
#else
  return a / b;
#endif
}

// FP32 的精确乘积最多 48 位有效位，可由 FP64 精确容纳，再只舍入一次到 FP32。
// MUSA 普通乘法会 flush 次正规数；此兼容路径不能用 fast-math 替换。
__device__ inline float multiply_rn(float a, float b) {
#ifdef LP_BACKEND_MUSA
  return __double2float_rn(static_cast<double>(a) * static_cast<double>(b));
#else
  return a * b;
#endif
}
}  // namespace pipeline
