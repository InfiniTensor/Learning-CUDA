// hadamard_tc.cu — 用 WMMA Tensor Core 实现的 Hadamard 变换与融合 INT4 量化
//
// 算法推导见 include/hadamard_tc.cuh 顶部注释。本文件的组织：
//   1. dtype 适配（FP16 / BF16 的标量转换与舍入）；
//   2. 形状常量 TcShape：由 head_dim 推出 tile 数、每 tile token 数和内层块大小；
//   3. 两个常量矩阵在 block 启动时写入 shared memory（H_16 与 A = I_k ⊗ H_m）；
//   4. transform_tiles()：每个 warp 的核心，做 kTiles 次 “两/三条 mma”，
//      必要时再对 accumulator 做跨 tile 蝶形；
//   5. 两个 kernel：写回低精度张量 / 融合 per-token INT4；
//   6. host 分发，尾部不足一个 tile 的 token 回落到 warp-per-token FHT。
//
// 硬件要求：FP16 WMMA 需要 SM >= 70，BF16 WMMA 需要 SM >= 80。本模块按 SM 80+
// 编译与验证。

#include "hadamard_tc.cuh"
#include "cuda_check.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <mma.h>

#include <cmath>
#include <cstdint>
#include <type_traits>

#ifndef HADAMARD_TC_VECTOR_SMEM
#define HADAMARD_TC_VECTOR_SMEM 0
#endif

using namespace nvcuda;

namespace {

// ---------------------------------------------------------------------------
// 1. dtype 适配
// ---------------------------------------------------------------------------

template <typename T>
struct TcScalar;

template <>
struct TcScalar<__half> {
  __device__ static __half from_float(float v) { return __float2half_rn(v); }
  __device__ static float to_float(__half v) { return __half2float(v); }
  __device__ static float round_scalar(float v) {
    return __half2float(__float2half_rn(v));
  }
};

template <>
struct TcScalar<__nv_bfloat16> {
  __device__ static __nv_bfloat16 from_float(float v) { return __float2bfloat16_rn(v); }
  __device__ static float to_float(__nv_bfloat16 v) { return __bfloat162float(v); }
  __device__ static float round_scalar(float v) {
    return __bfloat162float(__float2bfloat16_rn(v));
  }
};

// ---------------------------------------------------------------------------
// 2. 形状常量
// ---------------------------------------------------------------------------

constexpr int kTileDim = 16;              // WMMA m=n=k=16
constexpr int kTileElems = kTileDim * kTileDim;  // 一个 tile = 256 个元素
constexpr int kLdmOperand = kTileDim + 8;  // half/bf16 shared ldm，须为 8 的倍数
constexpr int kLdmAcc = kTileDim + 4;      // float shared ldm，须为 4 的倍数
#ifndef HADAMARD_TC_WARPS_PER_BLOCK
#define HADAMARD_TC_WARPS_PER_BLOCK 4
#endif
constexpr int kWarpsPerBlock = HADAMARD_TC_WARPS_PER_BLOCK;
static_assert(kWarpsPerBlock == 2 || kWarpsPerBlock == 4 || kWarpsPerBlock == 8);

template <int HEAD_DIM>
struct TcShape {
  // d > 256 时一个 token 跨多个 tile；d <= 256 时一个 tile 装多个 token。
  static constexpr int kTilesPerToken = HEAD_DIM > kTileElems ? HEAD_DIM / kTileElems : 1;
  static constexpr int kTokensPerTile = HEAD_DIM < kTileElems ? kTileElems / HEAD_DIM : 1;
  // 左乘块对角矩阵 A = I_k ⊗ H_m 的内层块大小 m。
  static constexpr int kInner = (HEAD_DIM < kTileElems ? HEAD_DIM : kTileElems) / kTileDim;
  // 每个 warp 负责的连续元素数。
  static constexpr int kElemsPerWarp = kTilesPerToken * kTileElems;
  // 每个 lane 固定负责 8 个连续元素（16 字节的 FP16/BF16 访问），因此
  // d <= 256 时一个 token 由连续的 kLanesPerToken 个 lane 覆盖。
  static constexpr int kLanesPerToken = HEAD_DIM < kTileElems ? HEAD_DIM / 8 : 32;
};

// ---------------------------------------------------------------------------
// 3. 常量矩阵
// ---------------------------------------------------------------------------
//
// Sylvester 矩阵元素满足 H[r][c] = (-1)^popcount(r & c)。
// h16 是右乘因子；amat 是左乘因子 A = I_k ⊗ H_m，块外为 0。
template <typename T, int HEAD_DIM>
__device__ __forceinline__ void build_constant_matrices(T* h16, T* amat) {
  constexpr int m = TcShape<HEAD_DIM>::kInner;
  for (int idx = threadIdx.x; idx < kTileElems; idx += blockDim.x) {
    const int r = idx >> 4;
    const int c = idx & 15;
    const float h = (__popc(r & c) & 1) ? -1.0f : 1.0f;
    h16[r * kLdmOperand + c] = TcScalar<T>::from_float(h);
    float a = 0.0f;
    if ((r / m) == (c / m)) {
      a = (__popc((r % m) & (c % m)) & 1) ? -1.0f : 1.0f;
    }
    amat[r * kLdmOperand + c] = TcScalar<T>::from_float(a);
  }
  __syncthreads();
}

// ---------------------------------------------------------------------------
// 4. warp 级核心：kTiles 次 WMMA + 可选跨 tile 蝶形
// ---------------------------------------------------------------------------
//
// 输出：本 lane 负责的 kTiles * 8 个 FP32 结果（已含可选归一化）。
// acc_scratch / operand_hi / operand_lo 都是调用方为本 warp 独占分配的
// shared memory：一块 FP32 tile 用于 accumulator 中转，一到两块低精度 tile
// 用于第 2/3 次 mma 的 operand。h16 / amat 是 block 内共享的常量矩阵。
template <typename T, int HEAD_DIM, bool NORMALIZE, bool SPLIT>
__device__ __forceinline__ void transform_tiles(
    const T* __restrict__ input, size_t warp_base, int lane, float norm_scale,
    float* acc_scratch, T* operand_hi, T* operand_lo,
    const T* h16, const T* amat,
    float (&result)[TcShape<HEAD_DIM>::kTilesPerToken * 8]) {
  using Shape = TcShape<HEAD_DIM>;
  constexpr int kTiles = Shape::kTilesPerToken;

  using AccFrag = wmma::fragment<wmma::accumulator, 16, 16, 16, float>;
  wmma::fragment<wmma::matrix_a, 16, 16, 16, T, wmma::row_major> x_frag;
  wmma::fragment<wmma::matrix_a, 16, 16, 16, T, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, T, wmma::row_major> h_frag;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, T, wmma::row_major> p_frag;
  AccFrag acc_out[kTiles];

  // 两个常量矩阵在整个 warp 的生命周期里不变，只在 tile 循环外加载一次。
  wmma::load_matrix_sync(a_frag, amat, kLdmOperand);
  wmma::load_matrix_sync(h_frag, h16, kLdmOperand);

  // 每个 lane 在 tile 内负责的位置：8 个连续元素落在同一行内。
  const int row = lane >> 1;
  const int col = (lane & 1) * 8;

#pragma unroll
  for (int tile = 0; tile < kTiles; ++tile) {
    // ---- 第 1 次 mma：P = X @ H_16 ----
    // X 直接从 global memory 以 row-major、ldm=16 读入；tile 起点是 256 元素
    // 的整数倍，天然满足 WMMA 的对齐要求，并且访问完全合并。
    AccFrag acc_p;
    wmma::fill_fragment(acc_p, 0.0f);
    wmma::load_matrix_sync(x_frag, input + warp_base + tile * kTileElems, kTileDim);
    wmma::mma_sync(acc_p, x_frag, h_frag, acc_p);

    // accumulator 与 matrix_b 的 fragment 内部布局不同，必须经 shared memory 中转。
    wmma::store_matrix_sync(acc_scratch, acc_p, kLdmAcc, wmma::mem_row_major);
    __syncwarp();
#if HADAMARD_TC_VECTOR_SMEM
    // Both padded row strides preserve 16-byte alignment. Coalesce each lane's
    // eight scalar shared accesses into two float4 loads and one int4 store.
    alignas(16) float p_values[8];
    alignas(16) T hi_values[8], lo_values[8];
    reinterpret_cast<float4*>(p_values)[0] =
        *reinterpret_cast<const float4*>(acc_scratch + row * kLdmAcc + col);
    reinterpret_cast<float4*>(p_values)[1] =
        *reinterpret_cast<const float4*>(acc_scratch + row * kLdmAcc + col + 4);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      hi_values[i] = TcScalar<T>::from_float(p_values[i]);
      if constexpr (SPLIT) lo_values[i] = TcScalar<T>::from_float(
          p_values[i] - TcScalar<T>::to_float(hi_values[i]));
    }
    *reinterpret_cast<int4*>(operand_hi + row * kLdmOperand + col) =
        *reinterpret_cast<const int4*>(hi_values);
    if constexpr (SPLIT) {
      *reinterpret_cast<int4*>(operand_lo + row * kLdmOperand + col) =
          *reinterpret_cast<const int4*>(lo_values);
    }
#else
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const float p = acc_scratch[row * kLdmAcc + col + i];
      const T hi = TcScalar<T>::from_float(p);
      operand_hi[row * kLdmOperand + col + i] = hi;
      if constexpr (SPLIT) {
        // 残差再舍入一次到低精度：hi + lo 的等效尾数从 11 bit 提升到约 22 bit
        // （BF16 是 8 -> 16 bit）。误差虽小，仍可能跨过最终 native 舍入中点。
        operand_lo[row * kLdmOperand + col + i] =
            TcScalar<T>::from_float(p - TcScalar<T>::to_float(hi));
      }
    }
 #endif
    __syncwarp();

    // ---- 第 2/3 次 mma：Y = A @ P_hi (+ A @ P_lo) ----
    wmma::fill_fragment(acc_out[tile], 0.0f);
    wmma::load_matrix_sync(p_frag, operand_hi, kLdmOperand);
    wmma::mma_sync(acc_out[tile], a_frag, p_frag, acc_out[tile]);
    if constexpr (SPLIT) {
      wmma::load_matrix_sync(p_frag, operand_lo, kLdmOperand);
      wmma::mma_sync(acc_out[tile], a_frag, p_frag, acc_out[tile]);
    }
    __syncwarp();
  }

  // ---- 跨 tile 蝶形（仅 d > 256）：H_d = H_{kTiles} ⊗ H_256 ----
  // 同一 fragment 类型的第 e 个元素在各 tile 中对应相同的 (row, col)，
  // 因此这一级完全是 accumulator 寄存器上的逐元素加减。
  if constexpr (kTiles > 1) {
#pragma unroll
    for (int stride = 1; stride < kTiles; stride <<= 1) {
#pragma unroll
      for (int tile = 0; tile < kTiles; ++tile) {
        if ((tile & stride) == 0) {
#pragma unroll
          for (int e = 0; e < AccFrag::num_elements; ++e) {
            const float a = acc_out[tile].x[e];
            const float b = acc_out[tile | stride].x[e];
            acc_out[tile].x[e] = a + b;
            acc_out[tile | stride].x[e] = a - b;
          }
        }
      }
    }
  }

  // ---- 写回：accumulator -> shared -> 每 lane 8 个 FP32 ----
#pragma unroll
  for (int tile = 0; tile < kTiles; ++tile) {
    wmma::store_matrix_sync(acc_scratch, acc_out[tile], kLdmAcc, wmma::mem_row_major);
    __syncwarp();
#if HADAMARD_TC_VECTOR_SMEM
    alignas(16) float output_values[8];
    reinterpret_cast<float4*>(output_values)[0] =
        *reinterpret_cast<const float4*>(acc_scratch + row * kLdmAcc + col);
    reinterpret_cast<float4*>(output_values)[1] =
        *reinterpret_cast<const float4*>(acc_scratch + row * kLdmAcc + col + 4);
#endif
#pragma unroll
    for (int i = 0; i < 8; ++i) {
#if HADAMARD_TC_VECTOR_SMEM
      float v = output_values[i];
#else
      float v = acc_scratch[row * kLdmAcc + col + i];
#endif
      if constexpr (NORMALIZE) v *= norm_scale;
      result[tile * 8 + i] = v;
    }
#if HADAMARD_TC_VECTOR_SMEM >= 2
    // Scratch is warp-private. A barrier is needed only before the next tile
    // overwrites it, not after the last read before returning register results.
    if (tile + 1 < kTiles) __syncwarp();
#else
    __syncwarp();
#endif
  }
}

// 每个 warp 需要的 shared memory 字节数（FP32 tile + 1~2 份低精度 operand tile）。
template <typename T, bool SPLIT>
__host__ __device__ constexpr int warp_scratch_floats() {
  // 以 float 为单位统一分配，低精度 operand tile 折算成 float 个数向上取整。
  constexpr int acc_floats = kLdmAcc * kTileDim;
  constexpr int operand_floats =
      (kLdmOperand * kTileDim * static_cast<int>(sizeof(T)) + 3) / 4;
  return acc_floats + operand_floats * (SPLIT ? 2 : 1);
}

// ---------------------------------------------------------------------------
// 5. kernel
// ---------------------------------------------------------------------------

template <typename T, int HEAD_DIM, bool NORMALIZE, bool SPLIT>
__global__ void hadamard_tc_kernel(const T* __restrict__ input,
                                   T* __restrict__ output, int total_warp_groups,
                                   float norm_scale) {
  using Shape = TcShape<HEAD_DIM>;
  constexpr int kTiles = Shape::kTilesPerToken;

  extern __shared__ __align__(32) float tc_smem[];
  // block 级常量矩阵放在最前面，之后是每个 warp 独占的 scratch。
  constexpr int const_floats = (kLdmOperand * kTileDim * static_cast<int>(sizeof(T)) + 3) / 4;
  T* h16 = reinterpret_cast<T*>(tc_smem);
  T* amat = reinterpret_cast<T*>(tc_smem + const_floats);
  build_constant_matrices<T, HEAD_DIM>(h16, amat);

  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  float* scratch = tc_smem + const_floats * 2 + warp * warp_scratch_floats<T, SPLIT>();
  float* acc_scratch = scratch;
  T* operand_hi = reinterpret_cast<T*>(scratch + kLdmAcc * kTileDim);
  T* operand_lo = operand_hi + (SPLIT ? kLdmOperand * kTileDim : 0);

  const int group = blockIdx.x * kWarpsPerBlock + warp;
  if (group >= total_warp_groups) return;
  const size_t warp_base = static_cast<size_t>(group) * Shape::kElemsPerWarp;

  float result[kTiles * 8];
  transform_tiles<T, HEAD_DIM, NORMALIZE, SPLIT>(
      input, warp_base, lane, norm_scale, acc_scratch, operand_hi, operand_lo,
      h16, amat, result);

  alignas(16) T packed[8];
#pragma unroll
  for (int tile = 0; tile < kTiles; ++tile) {
#pragma unroll
    for (int i = 0; i < 8; ++i) packed[i] = TcScalar<T>::from_float(result[tile * 8 + i]);
    // 每 lane 一次 16 字节写回，warp 内 32 个 lane 覆盖 tile 的 512 字节。
    T* dst = output + warp_base + tile * kTileElems + lane * 8;
    *reinterpret_cast<int4*>(dst) = *reinterpret_cast<const int4*>(packed);
  }
}

__device__ __forceinline__ int quantize_symmetric_int4(float value, float inv_scale) {
  int q = __float2int_rn(value * inv_scale);
  q = q < -7 ? -7 : q;
  return q > 7 ? 7 : q;
}

// 融合 INT4：仅 head_dim <= 256。此时一个 lane 的 8 个元素必定属于同一个 token，
// 且同一 token 由连续的 kLanesPerToken = head_dim/8 个 lane 覆盖，
// 所以 per-token max 就是一次宽度为 kLanesPerToken 的 sub-warp shuffle 归约。
template <typename T, int HEAD_DIM, bool NORMALIZE, bool SPLIT>
__global__ void hadamard_tc_fused_int4_kernel(
    const T* __restrict__ input, unsigned char* __restrict__ packed_output,
    float* __restrict__ scales, int total_warp_groups, float norm_scale) {
  using Shape = TcShape<HEAD_DIM>;
  constexpr int kLanesPerToken = Shape::kLanesPerToken;

  extern __shared__ __align__(32) float tc_smem[];
  constexpr int const_floats = (kLdmOperand * kTileDim * static_cast<int>(sizeof(T)) + 3) / 4;
  T* h16 = reinterpret_cast<T*>(tc_smem);
  T* amat = reinterpret_cast<T*>(tc_smem + const_floats);
  build_constant_matrices<T, HEAD_DIM>(h16, amat);

  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  float* scratch = tc_smem + const_floats * 2 + warp * warp_scratch_floats<T, SPLIT>();
  float* acc_scratch = scratch;
  T* operand_hi = reinterpret_cast<T*>(scratch + kLdmAcc * kTileDim);
  T* operand_lo = operand_hi + (SPLIT ? kLdmOperand * kTileDim : 0);

  const int group = blockIdx.x * kWarpsPerBlock + warp;
  if (group >= total_warp_groups) return;
  const size_t warp_base = static_cast<size_t>(group) * Shape::kElemsPerWarp;

  float result[8];
  transform_tiles<T, HEAD_DIM, NORMALIZE, SPLIT>(
      input, warp_base, lane, norm_scale, acc_scratch, operand_hi, operand_lo,
      h16, amat, result);

  // 与 unfused 路径逐位对齐：先模拟一次写回低精度的舍入，再求 max/scale。
#pragma unroll
  for (int i = 0; i < 8; ++i) result[i] = TcScalar<T>::round_scalar(result[i]);

  float max_abs = 0.0f;
#pragma unroll
  for (int i = 0; i < 8; ++i) max_abs = fmaxf(max_abs, fabsf(result[i]));
#pragma unroll
  for (int offset = kLanesPerToken / 2; offset > 0; offset >>= 1) {
    max_abs = fmaxf(max_abs,
                    __shfl_xor_sync(0xffffffffu, max_abs, offset, kLanesPerToken));
  }
  const float scale = max_abs > 0.0f ? max_abs / 7.0f : 1.0f;

  const int token_in_group = lane / kLanesPerToken;
  const int lane_in_token = lane % kLanesPerToken;
  const int token = group * Shape::kTokensPerTile + token_in_group;
  if (lane_in_token == 0) scales[token] = scale;

  const float inv_scale = 1.0f / scale;
  alignas(4) unsigned char bytes[4];
#pragma unroll
  for (int pair = 0; pair < 4; ++pair) {
    const int low = quantize_symmetric_int4(result[pair * 2], inv_scale);
    const int high = quantize_symmetric_int4(result[pair * 2 + 1], inv_scale);
    bytes[pair] = static_cast<unsigned char>((low & 0x0f) | ((high & 0x0f) << 4));
  }
  unsigned char* dst =
      packed_output + static_cast<size_t>(token) * (HEAD_DIM / 2) + lane_in_token * 4;
  *reinterpret_cast<unsigned int*>(dst) = *reinterpret_cast<const unsigned int*>(bytes);
}

// ---------------------------------------------------------------------------
// 6. host 分发
// ---------------------------------------------------------------------------

template <typename T, bool SPLIT>
int shared_bytes() {
  constexpr int const_floats = (kLdmOperand * kTileDim * static_cast<int>(sizeof(T)) + 3) / 4;
  return static_cast<int>(sizeof(float)) *
         (const_floats * 2 + kWarpsPerBlock * warp_scratch_floats<T, SPLIT>());
}

template <typename T, int HEAD_DIM, bool SPLIT>
void launch_tc_dim(const void* input, void* output, int total_tokens,
                   bool normalize, float norm_scale, cudaStream_t stream) {
  using Shape = TcShape<HEAD_DIM>;
  // 一个 warp 负责 kElemsPerWarp 个连续元素：d<=256 时是 kTokensPerTile 个
  // token，d>256 时正好是一个 token。
  const int groups = total_tokens / Shape::kTokensPerTile;
  if (groups > 0) {
    const dim3 grid(static_cast<unsigned>((groups + kWarpsPerBlock - 1) / kWarpsPerBlock));
    const dim3 block(32 * kWarpsPerBlock);
    const int smem = shared_bytes<T, SPLIT>();
    const T* in = static_cast<const T*>(input);
    T* out = static_cast<T*>(output);
    if (normalize) {
      hadamard_tc_kernel<T, HEAD_DIM, true, SPLIT>
          <<<grid, block, smem, stream>>>(in, out, groups, norm_scale);
    } else {
      hadamard_tc_kernel<T, HEAD_DIM, false, SPLIT>
          <<<grid, block, smem, stream>>>(in, out, groups, norm_scale);
    }
    CUDA_CHECK(cudaGetLastError());
  }
  // 尾部不足一个完整 tile 的 token 交给 warp-per-token FHT，语义完全一致。
  const int covered = total_tokens / Shape::kTokensPerTile * Shape::kTokensPerTile;
  if (covered < total_tokens) {
    const size_t offset = static_cast<size_t>(covered) * HEAD_DIM;
    launch_hadamard_warp(static_cast<const T*>(input) + offset,
                         static_cast<T*>(output) + offset, 1, 1,
                         total_tokens - covered, HEAD_DIM,
                         std::is_same<T, __half>::value ? DataType::FP16 : DataType::BF16,
                         normalize, stream);
  }
}

template <typename T, bool SPLIT>
int launch_tc_typed(const void* input, void* output, int total_tokens,
                    int head_dim, bool normalize, float norm_scale,
                    cudaStream_t stream) {
  switch (head_dim) {
    case 64:
      launch_tc_dim<T, 64, SPLIT>(input, output, total_tokens, normalize, norm_scale, stream);
      return 0;
    case 128:
      launch_tc_dim<T, 128, SPLIT>(input, output, total_tokens, normalize, norm_scale, stream);
      return 0;
    case 256:
      launch_tc_dim<T, 256, SPLIT>(input, output, total_tokens, normalize, norm_scale, stream);
      return 0;
    case 512:
      launch_tc_dim<T, 512, SPLIT>(input, output, total_tokens, normalize, norm_scale, stream);
      return 0;
    case 1024:
      launch_tc_dim<T, 1024, SPLIT>(input, output, total_tokens, normalize, norm_scale, stream);
      return 0;
    default:
      return -1;  // head_dim < 64 时凑不满一个 16x16 tile 的 Kronecker 分解
  }
}

template <typename T, int HEAD_DIM, bool SPLIT>
int launch_tc_fused_dim(const void* input, unsigned char* packed_output,
                        float* scales, int total_tokens, bool normalize,
                        float norm_scale, cudaStream_t stream) {
  using Shape = TcShape<HEAD_DIM>;
  const int groups = total_tokens / Shape::kTokensPerTile;
  if (groups > 0) {
  const dim3 grid(static_cast<unsigned>((groups + kWarpsPerBlock - 1) / kWarpsPerBlock));
  const dim3 block(32 * kWarpsPerBlock);
  const int smem = shared_bytes<T, SPLIT>();
  const T* in = static_cast<const T*>(input);
  if (normalize) {
    hadamard_tc_fused_int4_kernel<T, HEAD_DIM, true, SPLIT>
        <<<grid, block, smem, stream>>>(in, packed_output, scales, groups, norm_scale);
  } else {
    hadamard_tc_fused_int4_kernel<T, HEAD_DIM, false, SPLIT>
        <<<grid, block, smem, stream>>>(in, packed_output, scales, groups, norm_scale);
  }
  CUDA_CHECK(cudaGetLastError());
  }
  const int covered = groups * Shape::kTokensPerTile;
  if (covered < total_tokens) {
    const size_t offset = static_cast<size_t>(covered) * HEAD_DIM;
    return launch_hadamard_fused_quant_int4_warp(
        static_cast<const T*>(input) + offset, packed_output + offset / 2,
        scales + covered, 1, 1, total_tokens - covered, HEAD_DIM,
        std::is_same<T, __half>::value ? DataType::FP16 : DataType::BF16,
        normalize, stream);
  }
  return 0;
}

template <typename T, bool SPLIT>
int launch_tc_fused_typed(const void* input, unsigned char* packed_output,
                          float* scales, int total_tokens, int head_dim,
                          bool normalize, float norm_scale, cudaStream_t stream) {
  switch (head_dim) {
    case 64:
      return launch_tc_fused_dim<T, 64, SPLIT>(input, packed_output, scales,
                                               total_tokens, normalize, norm_scale, stream);
    case 128:
      return launch_tc_fused_dim<T, 128, SPLIT>(input, packed_output, scales,
                                                total_tokens, normalize, norm_scale, stream);
    case 256:
      return launch_tc_fused_dim<T, 256, SPLIT>(input, packed_output, scales,
                                                total_tokens, normalize, norm_scale, stream);
    default:
      return -1;  // d > 256 时一个 token 跨 tile，per-token max 无法只靠 sub-warp 归约
  }
}

}  // namespace

int launch_hadamard_tc(const void* input, void* output, int batch_size,
                       int seq_len, int num_heads, int head_dim, DataType dtype,
                       bool normalize, TcMode mode, cudaStream_t stream) {
  const long long total_tokens_ll =
      static_cast<long long>(batch_size) * seq_len * num_heads;
  if (total_tokens_ll == 0) return 0;
  const int total_tokens = static_cast<int>(total_tokens_ll);
  const float norm_scale =
      static_cast<float>(1.0 / std::sqrt(static_cast<double>(head_dim)));
  const bool split = mode == TcMode::Split;
  switch (dtype) {
    case DataType::FP16:
      return split ? launch_tc_typed<__half, true>(input, output, total_tokens, head_dim,
                                                   normalize, norm_scale, stream)
                   : launch_tc_typed<__half, false>(input, output, total_tokens, head_dim,
                                                    normalize, norm_scale, stream);
    case DataType::BF16:
      return split ? launch_tc_typed<__nv_bfloat16, true>(input, output, total_tokens,
                                                          head_dim, normalize,
                                                          norm_scale, stream)
                   : launch_tc_typed<__nv_bfloat16, false>(input, output, total_tokens,
                                                           head_dim, normalize,
                                                           norm_scale, stream);
    default:
      return -2;
  }
}

int launch_hadamard_tc_fused_quant_int4(const void* input,
                                        unsigned char* packed_output,
                                        float* scales, int batch_size,
                                        int seq_len, int num_heads, int head_dim,
                                        DataType dtype, bool normalize,
                                        TcMode mode, cudaStream_t stream) {
  const long long total_tokens_ll =
      static_cast<long long>(batch_size) * seq_len * num_heads;
  if (total_tokens_ll == 0) return 0;
  const int total_tokens = static_cast<int>(total_tokens_ll);
  const float norm_scale =
      static_cast<float>(1.0 / std::sqrt(static_cast<double>(head_dim)));
  const bool split = mode == TcMode::Split;
  switch (dtype) {
    case DataType::FP16:
      return split ? launch_tc_fused_typed<__half, true>(input, packed_output, scales,
                                                         total_tokens, head_dim,
                                                         normalize, norm_scale, stream)
                   : launch_tc_fused_typed<__half, false>(input, packed_output, scales,
                                                          total_tokens, head_dim,
                                                          normalize, norm_scale, stream);
    case DataType::BF16:
      return split ? launch_tc_fused_typed<__nv_bfloat16, true>(
                         input, packed_output, scales, total_tokens, head_dim,
                         normalize, norm_scale, stream)
                   : launch_tc_fused_typed<__nv_bfloat16, false>(
                         input, packed_output, scales, total_tokens, head_dim,
                         normalize, norm_scale, stream);
    default:
      return -2;
  }
}

const char* tc_mode_name(TcMode mode) {
  switch (mode) {
    case TcMode::Fast:
      return "tc_fast";
    case TcMode::Split:
      return "tc_split";
  }
  return "unknown";
}
