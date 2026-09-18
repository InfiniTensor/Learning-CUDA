#pragma once

// tensor、随机舍入及内部枚举对照路径；默认融合算子不在此文件。
#include "kernels/scale.cuh"

namespace pipeline {

// Kernel 3：通用/内部对照版 scale 计算，每个 CUDA 线程块固定 32 线程。
// block 模式：一个线程块归约一个量化分组；tensor 模式：直接使用 state[0]。
// 输出 scales 中的编码字节；默认 block+nearest 由各格式的融合 kernel 处理。
__global__ void build_scales(const float* input, const float* state, std::uint8_t* scales,
                            std::size_t n, bool tensor) {
  __shared__ float s[32];
  const unsigned lane = threadIdx.x;
  const auto i = blockIdx.x * static_cast<std::size_t>(kBlockSize) + lane;
  s[lane] = !tensor && lane < kBlockSize && i < n ? fabsf(input[i]) : 0;
  __syncthreads();
  for (unsigned step = 16; step; step >>= 1) {
    if (lane < step) s[lane] = fmaxf(s[lane], s[lane + step]);
    __syncthreads();
  }
  if (!lane)
    scales[blockIdx.x] = scale_code(tensor ? state[0] : s[0], nvfp4 ? state[1] : 1.0f);
}

// Kernel 4：非融合 block 路径的 scale 计算，输入 FP32，输出 scales 编码字节。
// 每个 CUDA 线程块 256 线程，处理 8 个 MXFP8 分组或 16 个 NVFP4 分组。
// shuffle 的 width=32/16 使各组独立归约，组内首线程写出一个 scale。
// 尾部线程仍参与 shuffle，避免使用 full mask 时缺失参与者。
__global__ void build_block_scales(const float* input, const float* state,
                                 std::uint8_t* scales, std::size_t n) {
  const std::size_t i = blockIdx.x * 256ull + threadIdx.x;
  float v = i < n ? fabsf(input[i]) : 0;
  for (unsigned step = kBlockSize / 2; step; step >>= 1)
    v = fmaxf(v, __shfl_down_sync(0xffffffffu, v, step, kBlockSize));
  if (threadIdx.x % kBlockSize == 0 && i < n)
    scales[i / kBlockSize] = scale_code(v, nvfp4 ? state[1] : 1.0f);
}

// Kernel 5：用已计算的 scale 将 FP32 输入量化，输出低精度 data 字节。
// 固定 256 线程/块：MXFP8 每线程写一个 E4M3 元素；NVFP4 每线程写两个 E2M1 元素。
// NVFP4 偶数元素放低 4 位、奇数元素放高 4 位，避免多个线程竞争同一个字节。
// group 决定使用分组 scale 还是全张量 scale；Baseline 选择内部对照编码算法。
template <bool Baseline>
__global__ void quantize_kernel(const float* input, std::uint8_t* data, const std::uint8_t* scales,
                               const float* state, std::size_t n, std::size_t group,
                               bool stochastic, std::uint32_t seed) {
  const auto item = blockIdx.x * 256ull + threadIdx.x;
  const auto i = nvfp4 ? item * 2 : item;
  if (i >= n) return;

  const float s = effective(scales, state, group == kBlockSize ? i / kBlockSize : 0);
  const auto low = encode(s > 0 ? divide_rn(input[i], s) : 0, nvfp4, stochastic, seed, i, Baseline);
  if (nvfp4) {
    // 每线程独占 packed 字节；奇数尾部的高 nibble 保持为零。
    const auto high = i + 1 < n
                          ? encode(s > 0 ? divide_rn(input[i + 1], s) : 0, true, stochastic,
                                   seed, i + 1, Baseline)
                          : 0;
    data[item] = low | (high << 4);
  } else
    data[item] = low;
}

}  // namespace pipeline
