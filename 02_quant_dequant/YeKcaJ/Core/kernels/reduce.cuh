#pragma once

// NVFP4 及 tensor 模式的全局最大值两级归约。
#include "common/types.h"
#include <cuda_runtime.h>
#include "common/device_math.cuh"

namespace pipeline {

// 仅用于固定 256 线程块：各 warp 先归约，再由首个 warp 合并 8 个结果。
// 所有线程必须参与；只有 threadIdx.x==0 的返回值是整块最大值。
// MUSA SDK 的同步 shuffle 使用 32-lane 逻辑子组，已由 musa_shuffle_test 独立验证。
__device__ inline float block_maximum(float value, float* warp_maxima) {
  const unsigned lane = threadIdx.x & 31;
  const unsigned warp = threadIdx.x >> 5;
  for (unsigned step = 16; step; step >>= 1)
    value = fmaxf(value, __shfl_down_sync(0xffffffffu, value, step, 32));
  if (lane == 0) warp_maxima[warp] = value;
  __syncthreads();
  if (warp == 0) {
    value = lane < 8 ? warp_maxima[lane] : 0.0f;
    for (unsigned step = 16; step; step >>= 1)
      value = fmaxf(value, __shfl_down_sync(0xffffffffu, value, step, 32));
  }
  return value;
}

constexpr unsigned kReductionShared = 8;

// Kernel 1：全张量绝对值最大值的第一阶段，用于 NVFP4 或 tensor scale 模式。
// 每个 CUDA 线程块固定 256 线程，跨步扫描 FP32 input，再合并各 warp 最大值。
// 输出 partial[blockIdx.x]：该线程块负责的数据的最大绝对值。
__global__ void maximum(const float* input, float* partial, std::size_t n) {
  __shared__ float s[kReductionShared];
  float v = 0;
  for (std::size_t i = blockIdx.x * 256ull + threadIdx.x; i < n; i += gridDim.x * 256ull)
    v = fmaxf(v, fabsf(input[i]));
  v = block_maximum(v, s);
  if (threadIdx.x == 0) partial[blockIdx.x] = v;
}

// Kernel 2：用一个 256 线程块归约 partial，得到全张量最大绝对值 M。
// 输出 state[0]=M；NVFP4 的 state[1]=global_scale=M/(6*448)，其他格式为 1。
// 全零输入使用 global_scale=1；正数下溢时保留最小正 FP32 值，避免除零。
__global__ void finalize_max(const float* partial, float* state, unsigned n) {
  __shared__ float s[kReductionShared];
  float v = 0;
  for (unsigned i = threadIdx.x; i < n; i += 256) v = fmaxf(v, partial[i]);
  v = block_maximum(v, s);
  if (!threadIdx.x) {
    state[0] = v;
    state[1] = nvfp4 && v > 0 ? fmaxf(divide_rn(v, 2688.0f), __int_as_float(1)) : 1.0f;
  }
}

}  // namespace pipeline
