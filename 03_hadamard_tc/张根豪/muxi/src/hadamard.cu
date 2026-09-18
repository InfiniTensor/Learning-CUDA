// 实现策略：
//   - 输入逻辑展平为 [total_tokens, head_dim]；
//   - 每个 block 处理 1 个 token，blockDim.x = HEAD_DIM（一个线程一个元素，映射最直观）；
//   - 数据先从 global memory 读入并按元素转成 FP32，存放在 shared memory；
//   - 在 shared memory 上做 log2(HEAD_DIM) 轮蝶形操作（快速 Walsh-Hadamard），
//     全部使用 FP32 累加 —— BF16 尤其依赖这一点才能满足误差要求；
//   - 最后按 NORMALIZE 决定是否乘 1/sqrt(HEAD_DIM)，转回原始类型写回。
//
// 蝶形说明：
//   对 stride = 1, 2, 4, ..., HEAD_DIM/2，
//   对每对下标 (i, i^stride)（i 的第 log2(stride) 位为 0）执行
//       (a, b) -> (a + b, a - b)
//   等价于乘以 Sylvester 型 Hadamard 矩阵 H（H @ H^T = HEAD_DIM * I）。
#include "hadamard.cuh"
#include "cuda_check.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cmath>
#include <cstdint>

// ---------------------------------------------------------------------------
// kernel 计算逻辑
// ---------------------------------------------------------------------------

template <typename T, int HEAD_DIM, bool NORMALIZE>
__global__ void hadamard_kernel(const T* __restrict__ input,
                                T* __restrict__ output,
                                int total_tokens, // 逻辑上 [total_tokens, head_dim]
                                float norm_scale) {
  // 确定当前 block 处理的 token 和当前线程处理的元素，每个block处理一个token
  const int token = blockIdx.x;
  if (token >= total_tokens) {
    return;
  }

  // 每个线程处理一个元素，tid = [0, HEAD_DIM)
  const int tid = threadIdx.x;
  // 计算当前 token 在 input/output 中的起始下标
  const size_t base = static_cast<size_t>(token) * HEAD_DIM;

  // 分配 shared memory 并将 input 转为 FP32 存入
  __shared__ float smem[HEAD_DIM];
  smem[tid] = static_cast<float>(input[base + tid]);
  __syncthreads();

  // log2(HEAD_DIM) 轮蝶形操作，每轮 stride = 1, 2, 4, ..., HEAD_DIM/2
#pragma unroll
  for (int stride = 1; stride < HEAD_DIM; stride <<= 1) {
    const int partner = tid ^ stride;

    // 每对元素只由下标较小的线程处理一次
    if (partner > tid) {
      const float a = smem[tid];
      const float b = smem[partner];
      smem[tid] = a + b;
      smem[partner] = a - b;
    }
    __syncthreads();
  }

  // 按需归一化并写回原始类型
  const float scale = NORMALIZE ? norm_scale : 1.0f;
  output[base + tid] = static_cast<T>(smem[tid] * scale);
}

// ---------------------------------------------------------------------------
// Week3 optimized kernel: packed I/O + register/shuffle butterfly
// ---------------------------------------------------------------------------

template <typename T>
struct PackedOps;

template <>
struct PackedOps<__half> {
  using Packed = __half2;
  __device__ static float2 load(const __half* ptr, int pair) {
    return __half22float2(reinterpret_cast<const Packed*>(ptr)[pair]);
  }
  __device__ static void store(__half* ptr, int pair, float2 value) {
    reinterpret_cast<Packed*>(ptr)[pair] = __floats2half2_rn(value.x, value.y);
  }
  __device__ static float round_scalar(float value) {
    return __half2float(__float2half_rn(value));
  }
};

template <>
struct PackedOps<__nv_bfloat16> {
  using Packed = __nv_bfloat162;
  __device__ static float2 load(const __nv_bfloat16* ptr, int pair) {
    return __bfloat1622float2(reinterpret_cast<const Packed*>(ptr)[pair]);
  }
  __device__ static void store(__nv_bfloat16* ptr, int pair, float2 value) {
    reinterpret_cast<Packed*>(ptr)[pair] =
        __floats2bfloat162_rn(value.x, value.y);
  }
  __device__ static float round_scalar(float value) {
    return __bfloat162float(__float2bfloat16_rn(value));
  }
};

template <int HEAD_DIM>
struct OptimizedShape {
  static constexpr int kPairs = HEAD_DIM / 2;
  static constexpr int kTokensPerBlock =
      HEAD_DIM <= 32 ? 8 : (HEAD_DIM == 64 ? 4 : (HEAD_DIM == 128 ? 2 : 1));
  static constexpr int kThreads = kPairs * kTokensPerBlock;
};

// 每线程负责相邻两个元素。stride=1 在两个寄存器之间完成；stride=2..32
// 用 shuffle；只有 stride>=64 才经过 shared memory 和 block barrier。
template <typename T, int HEAD_DIM, int TOKENS_PER_BLOCK>
__device__ __forceinline__ float2 transform_pair(
    const T* __restrict__ input, int total_tokens, int token, int local_pair,
    int token_in_block, float2* shared) {
  constexpr int kPairs = HEAD_DIM / 2;
  constexpr int kShuffleWidth = kPairs < 32 ? kPairs : 32;
  const bool valid = token < total_tokens;
  const size_t base = static_cast<size_t>(token) * HEAD_DIM;
  float2 value = valid ? PackedOps<T>::load(input + base, local_pair)
                       : make_float2(0.0f, 0.0f);

  // stride=1: adjacent pair in the same register.
  const float first = value.x;
  value.x = first + value.y;
  value.y = first - value.y;

  const unsigned mask = __activemask();
#pragma unroll
  for (int stride = 2; stride < HEAD_DIM && stride <= 32; stride <<= 1) {
    const int lane_offset = stride >> 1;
    const float other_x =
        __shfl_xor_sync(mask, value.x, lane_offset, kShuffleWidth);
    const float other_y =
        __shfl_xor_sync(mask, value.y, lane_offset, kShuffleWidth);
    if (local_pair & lane_offset) {
      value.x = other_x - value.x;
      value.y = other_y - value.y;
    } else {
      value.x += other_x;
      value.y += other_y;
    }
  }

  if constexpr (HEAD_DIM > 64) {
    const int shared_index = token_in_block * kPairs + local_pair;
    shared[shared_index] = value;
    __syncthreads();
#pragma unroll
    for (int stride = 64; stride < HEAD_DIM; stride <<= 1) {
      const int pair_offset = stride >> 1;
      const int partner = local_pair ^ pair_offset;
      if (valid && partner > local_pair) {
        const int partner_index = token_in_block * kPairs + partner;
        const float2 a = shared[shared_index];
        const float2 b = shared[partner_index];
        shared[shared_index] = make_float2(a.x + b.x, a.y + b.y);
        shared[partner_index] = make_float2(a.x - b.x, a.y - b.y);
      }
      __syncthreads();
    }
    value = shared[shared_index];
  }
  return value;
}

template <typename T, int HEAD_DIM, bool NORMALIZE>
__global__ void hadamard_optimized_kernel(const T* __restrict__ input,
                                          T* __restrict__ output,
                                          int total_tokens,
                                          float norm_scale) {
  constexpr int kPairs = OptimizedShape<HEAD_DIM>::kPairs;
  constexpr int kTokensPerBlock = OptimizedShape<HEAD_DIM>::kTokensPerBlock;
  constexpr int kThreads = OptimizedShape<HEAD_DIM>::kThreads;
  __shared__ float2 shared[kThreads];

  const int token_in_block = threadIdx.x / kPairs;
  const int local_pair = threadIdx.x - token_in_block * kPairs;
  const int token = blockIdx.x * kTokensPerBlock + token_in_block;
  float2 value = transform_pair<T, HEAD_DIM, kTokensPerBlock>(
      input, total_tokens, token, local_pair, token_in_block, shared);
  if (token >= total_tokens) return;
  if constexpr (NORMALIZE) {
    value.x *= norm_scale;
    value.y *= norm_scale;
  }
  const size_t base = static_cast<size_t>(token) * HEAD_DIM;
  PackedOps<T>::store(output + base, local_pair, value);
}

__device__ __forceinline__ int quantize_symmetric_int4(float value,
                                                        float inv_scale) {
  int q = __float2int_rn(value * inv_scale);
  q = q < -7 ? -7 : q;
  return q > 7 ? 7 : q;
}

template <typename T, int HEAD_DIM>
__device__ __forceinline__ void quantize_pair_block(
    float2 value, unsigned char* packed_output, float* scales, int token,
    float* warp_maxima, float* token_scale) {
  constexpr int kPairs = HEAD_DIM / 2;
  constexpr int kWarps = (kPairs + 31) / 32;
  constexpr int kWidth = kPairs < 32 ? kPairs : 32;
  float local_max = fmaxf(fabsf(value.x), fabsf(value.y));
  const unsigned mask = __activemask();
#pragma unroll
  for (int offset = kWidth / 2; offset > 0; offset >>= 1) {
    local_max = fmaxf(
        local_max, __shfl_down_sync(mask, local_max, offset, kWidth));
  }
  if ((threadIdx.x & 31) == 0) {
    warp_maxima[threadIdx.x >> 5] = local_max;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    float max_abs = 0.0f;
#pragma unroll
    for (int warp = 0; warp < kWarps; ++warp) {
      max_abs = fmaxf(max_abs, warp_maxima[warp]);
    }
    *token_scale = max_abs > 0.0f ? max_abs / 7.0f : 1.0f;
    scales[token] = *token_scale;
  }
  __syncthreads();
  const float inv_scale = 1.0f / *token_scale;
  const int low = quantize_symmetric_int4(value.x, inv_scale);
  const int high = quantize_symmetric_int4(value.y, inv_scale);
  packed_output[static_cast<size_t>(token) * kPairs + threadIdx.x] =
      static_cast<unsigned char>((low & 0x0f) | ((high & 0x0f) << 4));
}

template <typename T, int HEAD_DIM>
__global__ void quantize_int4_kernel(const T* __restrict__ input,
                                     unsigned char* __restrict__ packed_output,
                                     float* __restrict__ scales,
                                     int total_tokens) {
  constexpr int kPairs = HEAD_DIM / 2;
  constexpr int kWarps = (kPairs + 31) / 32;
  __shared__ float warp_maxima[kWarps];
  __shared__ float token_scale;
  const int token = blockIdx.x;
  if (token >= total_tokens) return;
  const size_t base = static_cast<size_t>(token) * HEAD_DIM;
  const float2 value = PackedOps<T>::load(input + base, threadIdx.x);
  quantize_pair_block<T, HEAD_DIM>(value, packed_output, scales, token,
                                    warp_maxima, &token_scale);
}

template <typename T, int HEAD_DIM, bool NORMALIZE>
__global__ void hadamard_fused_quant_int4_kernel(
    const T* __restrict__ input, unsigned char* __restrict__ packed_output,
    float* __restrict__ scales, int total_tokens, float norm_scale) {
  constexpr int kPairs = HEAD_DIM / 2;
  constexpr int kWarps = (kPairs + 31) / 32;
  __shared__ float2 transform_shared[kPairs];
  __shared__ float warp_maxima[kWarps];
  __shared__ float token_scale;
  const int token = blockIdx.x;
  if (token >= total_tokens) return;
  float2 value = transform_pair<T, HEAD_DIM, 1>(
      input, total_tokens, token, threadIdx.x, 0, transform_shared);
  if constexpr (NORMALIZE) {
    value.x *= norm_scale;
    value.y *= norm_scale;
  }
  // Match the unfused T output exactly without materializing it in global memory.
  value.x = PackedOps<T>::round_scalar(value.x);
  value.y = PackedOps<T>::round_scalar(value.y);
  quantize_pair_block<T, HEAD_DIM>(value, packed_output, scales, token,
                                    warp_maxima, &token_scale);
}

// ---------------------------------------------------------------------------
// 分发辅助（Host-side helper）
// ---------------------------------------------------------------------------

namespace {

// 按 HEAD_DIM 实例化模板并启动。norm_scale 在 host 侧用双精度算好再取 float，
// 避免在 device 端引入不必要的近似。
template <typename T, int HEAD_DIM>
void launch_dim(const void* input, void* output, int total_tokens,
                bool normalize, float norm_scale, cudaStream_t stream) {
  const dim3 grid(static_cast<unsigned>(total_tokens));
  const dim3 block(HEAD_DIM);
  const T* in = static_cast<const T*>(input);
  T* out = static_cast<T*>(output);
  if (normalize) {
    hadamard_kernel<T, HEAD_DIM, true>
        <<<grid, block, 0, stream>>>(in, out, total_tokens, norm_scale);
  } else {
    hadamard_kernel<T, HEAD_DIM, false>
        <<<grid, block, 0, stream>>>(in, out, total_tokens, norm_scale);
  }
  CUDA_CHECK(cudaGetLastError());
}

template <typename T>
int launch_typed(const void* input, void* output, int total_tokens,
                 int head_dim, bool normalize, float norm_scale,
                 cudaStream_t stream) {
  switch (head_dim) {
    case 32:
      launch_dim<T, 32>(input, output, total_tokens, normalize, norm_scale, stream);
      return 0;
    case 64:
      launch_dim<T, 64>(input, output, total_tokens, normalize, norm_scale, stream);
      return 0;
    case 128:
      launch_dim<T, 128>(input, output, total_tokens, normalize, norm_scale, stream);
      return 0;
    case 256:
      launch_dim<T, 256>(input, output, total_tokens, normalize, norm_scale, stream);
      return 0;
    case 512:
      launch_dim<T, 512>(input, output, total_tokens, normalize, norm_scale, stream);
      return 0;
    case 1024:
      launch_dim<T, 1024>(input, output, total_tokens, normalize, norm_scale, stream);
      return 0;
    default:
      return -1;  // 不支持的 head_dim
  }
}

template <typename T, int HEAD_DIM>
void launch_optimized_dim(const void* input, void* output, int total_tokens,
                          bool normalize, float norm_scale,
                          cudaStream_t stream) {
  constexpr int kThreads = OptimizedShape<HEAD_DIM>::kThreads;
  constexpr int kTokensPerBlock = OptimizedShape<HEAD_DIM>::kTokensPerBlock;
  const dim3 grid(static_cast<unsigned>((total_tokens + kTokensPerBlock - 1) /
                                        kTokensPerBlock));
  const dim3 block(kThreads);
  const T* in = static_cast<const T*>(input);
  T* out = static_cast<T*>(output);
  if (normalize) {
    hadamard_optimized_kernel<T, HEAD_DIM, true>
        <<<grid, block, 0, stream>>>(in, out, total_tokens, norm_scale);
  } else {
    hadamard_optimized_kernel<T, HEAD_DIM, false>
        <<<grid, block, 0, stream>>>(in, out, total_tokens, norm_scale);
  }
  CUDA_CHECK(cudaGetLastError());
}

template <typename T>
int launch_optimized_typed(const void* input, void* output, int total_tokens,
                           int head_dim, bool normalize, float norm_scale,
                           cudaStream_t stream) {
  switch (head_dim) {
    case 32:
      launch_optimized_dim<T, 32>(input, output, total_tokens, normalize,
                                   norm_scale, stream);
      return 0;
    case 64:
      launch_optimized_dim<T, 64>(input, output, total_tokens, normalize,
                                   norm_scale, stream);
      return 0;
    case 128:
      launch_optimized_dim<T, 128>(input, output, total_tokens, normalize,
                                    norm_scale, stream);
      return 0;
    case 256:
      launch_optimized_dim<T, 256>(input, output, total_tokens, normalize,
                                    norm_scale, stream);
      return 0;
    case 512:
      launch_optimized_dim<T, 512>(input, output, total_tokens, normalize,
                                    norm_scale, stream);
      return 0;
    case 1024:
      launch_optimized_dim<T, 1024>(input, output, total_tokens, normalize,
                                     norm_scale, stream);
      return 0;
    default:
      return -1;
  }
}

template <typename T, int HEAD_DIM>
void launch_quantize_dim(const void* input, unsigned char* packed_output,
                         float* scales, int total_tokens,
                         cudaStream_t stream) {
  quantize_int4_kernel<T, HEAD_DIM>
      <<<static_cast<unsigned>(total_tokens), HEAD_DIM / 2, 0, stream>>>(
          static_cast<const T*>(input), packed_output, scales, total_tokens);
  CUDA_CHECK(cudaGetLastError());
}

template <typename T>
int launch_quantize_typed(const void* input, unsigned char* packed_output,
                          float* scales, int total_tokens, int head_dim,
                          cudaStream_t stream) {
  switch (head_dim) {
    case 32: launch_quantize_dim<T, 32>(input, packed_output, scales, total_tokens, stream); return 0;
    case 64: launch_quantize_dim<T, 64>(input, packed_output, scales, total_tokens, stream); return 0;
    case 128: launch_quantize_dim<T, 128>(input, packed_output, scales, total_tokens, stream); return 0;
    case 256: launch_quantize_dim<T, 256>(input, packed_output, scales, total_tokens, stream); return 0;
    case 512: launch_quantize_dim<T, 512>(input, packed_output, scales, total_tokens, stream); return 0;
    case 1024: launch_quantize_dim<T, 1024>(input, packed_output, scales, total_tokens, stream); return 0;
    default: return -1;
  }
}

template <typename T, int HEAD_DIM>
void launch_fused_quantize_dim(const void* input,
                               unsigned char* packed_output, float* scales,
                               int total_tokens, bool normalize,
                               float norm_scale, cudaStream_t stream) {
  if (normalize) {
    hadamard_fused_quant_int4_kernel<T, HEAD_DIM, true>
        <<<static_cast<unsigned>(total_tokens), HEAD_DIM / 2, 0, stream>>>(
            static_cast<const T*>(input), packed_output, scales, total_tokens,
            norm_scale);
  } else {
    hadamard_fused_quant_int4_kernel<T, HEAD_DIM, false>
        <<<static_cast<unsigned>(total_tokens), HEAD_DIM / 2, 0, stream>>>(
            static_cast<const T*>(input), packed_output, scales, total_tokens,
            norm_scale);
  }
  CUDA_CHECK(cudaGetLastError());
}

template <typename T>
int launch_fused_quantize_typed(
    const void* input, unsigned char* packed_output, float* scales,
    int total_tokens, int head_dim, bool normalize, float norm_scale,
    cudaStream_t stream) {
  switch (head_dim) {
    case 32: launch_fused_quantize_dim<T, 32>(input, packed_output, scales, total_tokens, normalize, norm_scale, stream); return 0;
    case 64: launch_fused_quantize_dim<T, 64>(input, packed_output, scales, total_tokens, normalize, norm_scale, stream); return 0;
    case 128: launch_fused_quantize_dim<T, 128>(input, packed_output, scales, total_tokens, normalize, norm_scale, stream); return 0;
    case 256: launch_fused_quantize_dim<T, 256>(input, packed_output, scales, total_tokens, normalize, norm_scale, stream); return 0;
    case 512: launch_fused_quantize_dim<T, 512>(input, packed_output, scales, total_tokens, normalize, norm_scale, stream); return 0;
    case 1024: launch_fused_quantize_dim<T, 1024>(input, packed_output, scales, total_tokens, normalize, norm_scale, stream); return 0;
    default: return -1;
  }
}

}  // namespace

// ---------------------------------------------------------------------------
// 公开接口，主要用于对外暴露的 C++ API
// ---------------------------------------------------------------------------

int launch_hadamard(const void* input, void* output, int batch_size,
                    int seq_len, int num_heads, int head_dim, DataType dtype,
                    bool normalize, cudaStream_t stream) {
  const long long total_tokens =
      static_cast<long long>(batch_size) * seq_len * num_heads;
  if (total_tokens == 0) {
    return 0;  // 空输入，直接成功返回
  }
  // host 侧计算归一化系数：1 / sqrt(head_dim)
  const float norm_scale =
      static_cast<float>(1.0 / std::sqrt(static_cast<double>(head_dim)));

  switch (dtype) {
    case DataType::FP16:
      return launch_typed<__half>(input, output, static_cast<int>(total_tokens),
                                  head_dim, normalize, norm_scale, stream);
    case DataType::BF16:
      return launch_typed<__nv_bfloat16>(input, output,
                                         static_cast<int>(total_tokens),
                                         head_dim, normalize, norm_scale, stream);
    default:
      return -2;  // 不支持的 dtype
  }
}

int launch_hadamard_optimized(const void* input, void* output, int batch_size,
                              int seq_len, int num_heads, int head_dim,
                              DataType dtype, bool normalize,
                              cudaStream_t stream) {
  const long long total_tokens_ll =
      static_cast<long long>(batch_size) * seq_len * num_heads;
  if (total_tokens_ll == 0) return 0;
  const int total_tokens = static_cast<int>(total_tokens_ll);
  const float norm_scale =
      static_cast<float>(1.0 / std::sqrt(static_cast<double>(head_dim)));
  switch (dtype) {
    case DataType::FP16:
      return launch_optimized_typed<__half>(input, output, total_tokens,
                                             head_dim, normalize, norm_scale,
                                             stream);
    case DataType::BF16:
      return launch_optimized_typed<__nv_bfloat16>(
          input, output, total_tokens, head_dim, normalize, norm_scale, stream);
    default:
      return -2;
  }
}

int launch_quantize_int4(const void* input, unsigned char* packed_output,
                         float* scales, int total_tokens, int head_dim,
                         DataType dtype, cudaStream_t stream) {
  if (total_tokens == 0) return 0;
  switch (dtype) {
    case DataType::FP16:
      return launch_quantize_typed<__half>(input, packed_output, scales,
                                            total_tokens, head_dim, stream);
    case DataType::BF16:
      return launch_quantize_typed<__nv_bfloat16>(
          input, packed_output, scales, total_tokens, head_dim, stream);
    default:
      return -2;
  }
}

int launch_hadamard_fused_quant_int4(
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
      return launch_fused_quantize_typed<__half>(
          input, packed_output, scales, total_tokens, head_dim, normalize,
          norm_scale, stream);
    case DataType::BF16:
      return launch_fused_quantize_typed<__nv_bfloat16>(
          input, packed_output, scales, total_tokens, head_dim, normalize,
          norm_scale, stream);
    default:
      return -2;
  }
}

const char* dtype_name(DataType dtype) {
  switch (dtype) {
    case DataType::FP16:
      return "fp16";
    case DataType::BF16:
      return "bf16";
  }
  return "unknown";
}
