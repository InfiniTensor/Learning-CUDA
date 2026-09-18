// warp-per-token 快速 Walsh-Hadamard 变换（optimized kernel 的下一级优化）
//
// 动机：
//   优化版 kernel 每个线程只持有 2 个元素，因此 head_dim=128/256 时仍有
//   1~2 轮跨 warp 的 shared-memory 交换 + 2~4 次 __syncthreads()，并且跨 warp
//   阶段用 `if (partner > local_pair)` 只让一半线程干活。d=256 因此只拿到约
//   1.25x 加速。
//
// 本文件的做法：
//   令 ELEMS = head_dim / 32，一个 warp 恰好负责一个 token。
//     - 全局元素下标 = lane * ELEMS + i；
//     - stride < ELEMS 的低阶蝶形完全落在同一线程的寄存器数组内；
//     - stride >= ELEMS 的 5 个高阶蝶形用 __shfl_xor_sync(lane ^ off) 完成；
//   于是整个变换 0 shared memory、0 barrier，并且 global 访问宽度提升到
//   ELEMS*2 字节（d=128 为 64-bit，d=256 为 128-bit）。
//
// 数值语义与 hadamard.cu 中的两条路径完全一致：低精度输入 -> FP32 蝶形 ->
// 可选 1/sqrt(d) -> 末端一次舍入回低精度，因此可以逐位回归。

#include "hadamard.cuh"
#include "cuda_check.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cmath>
#include <cstdint>

namespace {

// ---------------------------------------------------------------------------
// dtype 相关的标量转换（与 hadamard.cu 中 PackedOps 的舍入行为保持一致）
// ---------------------------------------------------------------------------

template <typename T>
struct ScalarOps;

template <>
struct ScalarOps<__half> {
  __device__ static float to_float(__half value) { return __half2float(value); }
  __device__ static __half from_float(float value) { return __float2half_rn(value); }
  // 在寄存器里模拟一次“写回低精度再读出”的舍入，用于 fused 量化对齐 unfused。
  __device__ static float round_scalar(float value) {
    return __half2float(__float2half_rn(value));
  }
};

template <>
struct ScalarOps<__nv_bfloat16> {
  __device__ static float to_float(__nv_bfloat16 value) { return __bfloat162float(value); }
  __device__ static __nv_bfloat16 from_float(float value) {
    return __float2bfloat16_rn(value);
  }
  __device__ static float round_scalar(float value) {
    return __bfloat162float(__float2bfloat16_rn(value));
  }
};

// ---------------------------------------------------------------------------
// 按 ELEMS*sizeof(T) 字节宽度做向量化 load/store
// ---------------------------------------------------------------------------
//
// 起始地址 = base_ptr + token*HEAD_DIM + lane*ELEMS，偏移量本身就是 ELEMS 的
// 倍数，因此只要 cudaMalloc 返回的基地址 16B 对齐，下面的宽访问就是对齐的。
template <typename T, int ELEMS>
__device__ __forceinline__ void vector_copy(const T* __restrict__ src,
                                            T* __restrict__ dst) {
  constexpr int kBytes = ELEMS * static_cast<int>(sizeof(T));
  if constexpr (kBytes % 16 == 0) {
#pragma unroll
    for (int chunk = 0; chunk < kBytes / 16; ++chunk) {
      reinterpret_cast<int4*>(dst)[chunk] = reinterpret_cast<const int4*>(src)[chunk];
    }
  } else if constexpr (kBytes == 8) {
    *reinterpret_cast<int2*>(dst) = *reinterpret_cast<const int2*>(src);
  } else {
    *reinterpret_cast<int*>(dst) = *reinterpret_cast<const int*>(src);
  }
}

// ---------------------------------------------------------------------------
// 蝶形核心：寄存器内低阶 stage + warp shuffle 高阶 stage
// ---------------------------------------------------------------------------

template <int ELEMS>
__device__ __forceinline__ void butterfly_registers(float (&value)[ELEMS]) {
  // stride = 1 .. ELEMS/2：配对的两个元素都在本线程的寄存器数组里。
#pragma unroll
  for (int stride = 1; stride < ELEMS; stride <<= 1) {
#pragma unroll
    for (int i = 0; i < ELEMS; ++i) {
      if ((i & stride) == 0) {
        const float a = value[i];
        const float b = value[i | stride];
        value[i] = a + b;
        value[i | stride] = a - b;
      }
    }
  }
}

template <int ELEMS>
__device__ __forceinline__ void butterfly_shuffle(float (&value)[ELEMS], int lane) {
  // stride = ELEMS*1 .. ELEMS*16：配对元素在 lane ^ offset 上，i 保持不变。
  // 下标较小的一侧取 a+b，较大的一侧取 a-b，与 shared-memory 版本同序。
#pragma unroll
  for (int offset = 1; offset < 32; offset <<= 1) {
    const bool upper = (lane & offset) != 0;
#pragma unroll
    for (int i = 0; i < ELEMS; ++i) {
      const float other = __shfl_xor_sync(0xffffffffu, value[i], offset, 32);
      value[i] = upper ? other - value[i] : value[i] + other;
    }
  }
}

// 读入一个 token 中属于本 lane 的 ELEMS 个元素，做完整蝶形并可选归一化。
template <typename T, int HEAD_DIM, bool NORMALIZE>
__device__ __forceinline__ void transform_lane(const T* __restrict__ input,
                                               int token, int lane,
                                               float norm_scale,
                                               float (&value)[HEAD_DIM / 32]) {
  constexpr int kElems = HEAD_DIM / 32;
  alignas(16) T raw[kElems];
  const size_t base = static_cast<size_t>(token) * HEAD_DIM + lane * kElems;
  vector_copy<T, kElems>(input + base, raw);
#pragma unroll
  for (int i = 0; i < kElems; ++i) {
    value[i] = ScalarOps<T>::to_float(raw[i]);
  }
  butterfly_registers<kElems>(value);
  butterfly_shuffle<kElems>(value, lane);
  if constexpr (NORMALIZE) {
#pragma unroll
    for (int i = 0; i < kElems; ++i) value[i] *= norm_scale;
  }
}

// ---------------------------------------------------------------------------
// kernel
// ---------------------------------------------------------------------------

#ifndef HADAMARD_WARPS_PER_BLOCK
#define HADAMARD_WARPS_PER_BLOCK 4
#endif
#ifndef HADAMARD_FUSED_WARPS_PER_BLOCK
#define HADAMARD_FUSED_WARPS_PER_BLOCK HADAMARD_WARPS_PER_BLOCK
#endif
#ifndef HADAMARD_QUANT_VECTOR_STORE
#define HADAMARD_QUANT_VECTOR_STORE 0
#endif
static_assert(HADAMARD_QUANT_VECTOR_STORE >= 0 && HADAMARD_QUANT_VECTOR_STORE <= 2);
static_assert(HADAMARD_WARPS_PER_BLOCK == 0 || HADAMARD_WARPS_PER_BLOCK == 2 ||
              HADAMARD_WARPS_PER_BLOCK == 4 || HADAMARD_WARPS_PER_BLOCK == 8);
static_assert(HADAMARD_FUSED_WARPS_PER_BLOCK == 0 || HADAMARD_FUSED_WARPS_PER_BLOCK == 2 ||
              HADAMARD_FUSED_WARPS_PER_BLOCK == 4 || HADAMARD_FUSED_WARPS_PER_BLOCK == 8);
template <int HEAD_DIM, bool FUSED = false>
__host__ __device__ constexpr int warp_block_size() {
  // Opt-in measured H200 shape preset. Explicit 2/4/8 keeps A/B controls intact.
  constexpr int setting = FUSED ? HADAMARD_FUSED_WARPS_PER_BLOCK : HADAMARD_WARPS_PER_BLOCK;
  return setting == 0 ? (HEAD_DIM == 64 ? 8 : 4) : setting;
}

template <typename T, int HEAD_DIM, bool NORMALIZE>
__global__ void hadamard_warp_kernel(const T* __restrict__ input,
                                     T* __restrict__ output, int total_tokens,
                                     float norm_scale) {
  constexpr int kWarpsPerBlock = warp_block_size<HEAD_DIM>();
  constexpr int kElems = HEAD_DIM / 32;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int token = blockIdx.x * kWarpsPerBlock + warp;
  // 整个 warp 处理同一个 token，因此该分支对 warp 内所有 lane 一致，
  // 提前返回不会破坏后续 __shfl_xor_sync 的 full-mask 假设。
  if (token >= total_tokens) return;

  float value[kElems];
  transform_lane<T, HEAD_DIM, NORMALIZE>(input, token, lane, norm_scale, value);

  alignas(16) T out[kElems];
#pragma unroll
  for (int i = 0; i < kElems; ++i) out[i] = ScalarOps<T>::from_float(value[i]);
  const size_t base = static_cast<size_t>(token) * HEAD_DIM + lane * kElems;
  vector_copy<T, kElems>(out, output + base);
}

// per-token symmetric INT4：scale = max|x|/7，round-to-nearest-even 后 clamp 到
// [-7,7]，相邻两个元素打包进一个 byte 的低/高 nibble。协议与 hadamard.cu 一致。
__device__ __forceinline__ int quantize_symmetric_int4(float value, float inv_scale) {
  int q = __float2int_rn(value * inv_scale);
  q = q < -7 ? -7 : q;
  return q > 7 ? 7 : q;
}

template <typename T, int HEAD_DIM, bool NORMALIZE>
__global__ void hadamard_warp_fused_int4_kernel(
    const T* __restrict__ input, unsigned char* __restrict__ packed_output,
    float* __restrict__ scales, int total_tokens, float norm_scale) {
  constexpr int kWarpsPerBlock = warp_block_size<HEAD_DIM, true>();
  constexpr int kElems = HEAD_DIM / 32;
  constexpr int kBytesPerLane = kElems / 2;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int token = blockIdx.x * kWarpsPerBlock + warp;
  if (token >= total_tokens) return;

  float value[kElems];
  transform_lane<T, HEAD_DIM, NORMALIZE>(input, token, lane, norm_scale, value);
  // 先在寄存器里舍入到输出 dtype，使 fused 与 “先变换写回、再量化” 逐位一致。
#pragma unroll
  for (int i = 0; i < kElems; ++i) value[i] = ScalarOps<T>::round_scalar(value[i]);

  float max_abs = 0.0f;
#pragma unroll
  for (int i = 0; i < kElems; ++i) max_abs = fmaxf(max_abs, fabsf(value[i]));
  // 一个 warp 就是一个 token，因此 per-token max 是纯 warp 归约：无 shared、无 barrier。
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    max_abs = fmaxf(max_abs, __shfl_xor_sync(0xffffffffu, max_abs, offset, 32));
  }
  const float scale = max_abs > 0.0f ? max_abs / 7.0f : 1.0f;
  if (lane == 0) scales[token] = scale;

  const float inv_scale = 1.0f / scale;
  alignas(16) unsigned char bytes[kBytesPerLane];
#pragma unroll
  for (int pair = 0; pair < kBytesPerLane; ++pair) {
    const int low = quantize_symmetric_int4(value[pair * 2], inv_scale);
    const int high = quantize_symmetric_int4(value[pair * 2 + 1], inv_scale);
    bytes[pair] = static_cast<unsigned char>((low & 0x0f) | ((high & 0x0f) << 4));
  }
  unsigned char* dst =
      packed_output + static_cast<size_t>(token) * (HEAD_DIM / 2) + lane * kBytesPerLane;
  // Mode 2 keeps the verified d1024 wide store without the d512 regression.
  constexpr bool wide_store = HADAMARD_QUANT_VECTOR_STORE == 1 ||
      (HADAMARD_QUANT_VECTOR_STORE == 2 && HEAD_DIM == 1024);
  if constexpr (wide_store && kBytesPerLane == 16) {
    *reinterpret_cast<int4*>(dst) = *reinterpret_cast<const int4*>(bytes);
  } else if constexpr (wide_store && kBytesPerLane == 8) {
    *reinterpret_cast<int2*>(dst) = *reinterpret_cast<const int2*>(bytes);
  } else if constexpr (kBytesPerLane == 4) {
    *reinterpret_cast<unsigned int*>(dst) = *reinterpret_cast<const unsigned int*>(bytes);
  } else if constexpr (kBytesPerLane == 2) {
    *reinterpret_cast<unsigned short*>(dst) =
        *reinterpret_cast<const unsigned short*>(bytes);
  } else {
#pragma unroll
    for (int pair = 0; pair < kBytesPerLane; ++pair) dst[pair] = bytes[pair];
  }
}

// ---------------------------------------------------------------------------
// host 侧分发
// ---------------------------------------------------------------------------

template <typename T, int HEAD_DIM>
void launch_warp_dim(const void* input, void* output, int total_tokens,
                      bool normalize, float norm_scale, cudaStream_t stream) {
  constexpr int kWarpsPerBlock = warp_block_size<HEAD_DIM>();
  const dim3 grid(static_cast<unsigned>((total_tokens + kWarpsPerBlock - 1) /
                                        kWarpsPerBlock));
  const dim3 block(32 * kWarpsPerBlock);
  const T* in = static_cast<const T*>(input);
  T* out = static_cast<T*>(output);
  if (normalize) {
    hadamard_warp_kernel<T, HEAD_DIM, true>
        <<<grid, block, 0, stream>>>(in, out, total_tokens, norm_scale);
  } else {
    hadamard_warp_kernel<T, HEAD_DIM, false>
        <<<grid, block, 0, stream>>>(in, out, total_tokens, norm_scale);
  }
  CUDA_CHECK(cudaGetLastError());
}

template <typename T>
int launch_warp_typed(const void* input, void* output, int total_tokens,
                      int head_dim, bool normalize, float norm_scale,
                      cudaStream_t stream) {
  switch (head_dim) {
    case 64:
      launch_warp_dim<T, 64>(input, output, total_tokens, normalize, norm_scale, stream);
      return 0;
    case 128:
      launch_warp_dim<T, 128>(input, output, total_tokens, normalize, norm_scale, stream);
      return 0;
    case 256:
      launch_warp_dim<T, 256>(input, output, total_tokens, normalize, norm_scale, stream);
      return 0;
    case 512:
      launch_warp_dim<T, 512>(input, output, total_tokens, normalize, norm_scale, stream);
      return 0;
    case 1024:
      launch_warp_dim<T, 1024>(input, output, total_tokens, normalize, norm_scale, stream);
      return 0;
    default:
      return -1;  // head_dim < 64 时一个 warp 装不下 2 元素/线程
  }
}

template <typename T, int HEAD_DIM>
void launch_warp_fused_dim(const void* input, unsigned char* packed_output,
                           float* scales, int total_tokens, bool normalize,
                           float norm_scale, cudaStream_t stream) {
  constexpr int kWarpsPerBlock = warp_block_size<HEAD_DIM, true>();
  const dim3 grid(static_cast<unsigned>((total_tokens + kWarpsPerBlock - 1) /
                                        kWarpsPerBlock));
  const dim3 block(32 * kWarpsPerBlock);
  const T* in = static_cast<const T*>(input);
  if (normalize) {
    hadamard_warp_fused_int4_kernel<T, HEAD_DIM, true><<<grid, block, 0, stream>>>(
        in, packed_output, scales, total_tokens, norm_scale);
  } else {
    hadamard_warp_fused_int4_kernel<T, HEAD_DIM, false><<<grid, block, 0, stream>>>(
        in, packed_output, scales, total_tokens, norm_scale);
  }
  CUDA_CHECK(cudaGetLastError());
}

template <typename T>
int launch_warp_fused_typed(const void* input, unsigned char* packed_output,
                            float* scales, int total_tokens, int head_dim,
                            bool normalize, float norm_scale,
                            cudaStream_t stream) {
  switch (head_dim) {
    case 64:
      launch_warp_fused_dim<T, 64>(input, packed_output, scales, total_tokens,
                                   normalize, norm_scale, stream);
      return 0;
    case 128:
      launch_warp_fused_dim<T, 128>(input, packed_output, scales, total_tokens,
                                    normalize, norm_scale, stream);
      return 0;
    case 256:
      launch_warp_fused_dim<T, 256>(input, packed_output, scales, total_tokens,
                                    normalize, norm_scale, stream);
      return 0;
    case 512:
      launch_warp_fused_dim<T, 512>(input, packed_output, scales, total_tokens,
                                    normalize, norm_scale, stream);
      return 0;
    case 1024:
      launch_warp_fused_dim<T, 1024>(input, packed_output, scales, total_tokens,
                                     normalize, norm_scale, stream);
      return 0;
    default:
      return -1;
  }
}

}  // namespace

int launch_hadamard_warp(const void* input, void* output, int batch_size,
                         int seq_len, int num_heads, int head_dim,
                         DataType dtype, bool normalize, cudaStream_t stream) {
  const long long total_tokens_ll =
      static_cast<long long>(batch_size) * seq_len * num_heads;
  if (total_tokens_ll == 0) return 0;
  const int total_tokens = static_cast<int>(total_tokens_ll);
  const float norm_scale =
      static_cast<float>(1.0 / std::sqrt(static_cast<double>(head_dim)));
  switch (dtype) {
    case DataType::FP16:
      return launch_warp_typed<__half>(input, output, total_tokens, head_dim,
                                       normalize, norm_scale, stream);
    case DataType::BF16:
      return launch_warp_typed<__nv_bfloat16>(input, output, total_tokens,
                                              head_dim, normalize, norm_scale,
                                              stream);
    default:
      return -2;
  }
}

int launch_hadamard_fused_quant_int4_warp(
    const void* input, unsigned char* packed_output, float* scales,
    int batch_size, int seq_len, int num_heads, int head_dim, DataType dtype,
    bool normalize, cudaStream_t stream) {
  const long long total_tokens_ll =
      static_cast<long long>(batch_size) * seq_len * num_heads;
  if (total_tokens_ll == 0) return 0;
  const int total_tokens = static_cast<int>(total_tokens_ll);
  const float norm_scale =
      static_cast<float>(1.0 / std::sqrt(static_cast<double>(head_dim)));
  switch (dtype) {
    case DataType::FP16:
      return launch_warp_fused_typed<__half>(input, packed_output, scales,
                                             total_tokens, head_dim, normalize,
                                             norm_scale, stream);
    case DataType::BF16:
      return launch_warp_fused_typed<__nv_bfloat16>(
          input, packed_output, scales, total_tokens, head_dim, normalize,
          norm_scale, stream);
    default:
      return -2;
  }
}
