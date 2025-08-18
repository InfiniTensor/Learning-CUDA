#include <vector>
#include <cuda_runtime.h>
#include <assert.h>
#include <math.h>
#include <vector>
#include <stdio.h>
#include "../tester/utils.h"
#include <iostream>
#include <cstdint>
/**
 * @brief Find the k-th largest element in a vector using CUDA.
 * 
 * @tparam T Type of elements in the input vector (should support `int` and `float`).
 * @param h_input Host-side input vector.
 * @param k 1-based index of the element to find (e.g., `k=1` returns the largest element).
 * @return T The k-th largest element in `h_input`.

 * @note Must use CUDA kernels for all compute-intensive steps; no significant CPU allowed.
 * @note Library functions that can directly complete a significant part of the work are NOT allowed. 
 * @note For invalid cases, return T(-100).
 * @note Handles device memory management (allocate/copy/free) internally. Errors should be thrown.
 */
// ===================== 基础宏与工具 =====================

// 若未定义，定义一个“全满掩码”的 warp 活跃位掩码（32 位）
// __ballot_sync 的便捷封装：收集 warp 内各 lane 的布尔投票结果到一个 32-bit mask
#define WARP_BALLOT_1(pred)             __ballot_sync(0xffffffffu, (pred))
#define WARP_BALLOT_2(pred, activeMask) __ballot_sync((activeMask), (pred))
// 根据参数个数自动选择上面两个宏
#define _WARP_BALLOT_GET_MACRO(_1,_2,NAME,...) NAME
#define WARP_BALLOT(...) _WARP_BALLOT_GET_MACRO(__VA_ARGS__, WARP_BALLOT_2, WARP_BALLOT_1)(__VA_ARGS__)

// 设备端断言（可在调试期查出不该发生的路径）
#ifndef CUDA_KERNEL_ASSERT
#include <assert.h>
#define CUDA_KERNEL_ASSERT(cond) assert(cond)
#endif

// 设备侧的受限只读缓存 load（Kepler+ 支持 __ldg）
// 对 global memory 上的只读数据使用 __ldg 可提高带宽利用率
template <typename T>
__device__ __forceinline__ T doLdg(const T* p) {
#if __CUDA_ARCH__ >= 350
  return __ldg(p);
#else
  return *p;
#endif
}

/** 无返回的原子加：在我们这里只需要 int 版本（共享/全局内存均可） */
__device__ __forceinline__ void gpuAtomicAddNoReturn(int* addr, int val) {
  atomicAdd(addr, val);
}

/** 向上取整到 d 的整数倍：round_up(n, d) = ceil(n/d) * d */
template <typename I>
__host__ __device__ __forceinline__ I round_up(I n, I d) {
  return ((n + d - 1) / d) * d;
}

// ===================== Bitfield 小工具（替代 at::cuda::Bitfield） =====================
namespace mini {

/** 取出 v 的 [pos, pos+bits) 位段（无符号右移+掩码） */
template <typename T>
__device__ __forceinline__ T getBitfield(T v, int pos, int bits) {
  const T mask = (bits >= (int)(sizeof(T) * 8)) ? ~T(0) : ((T(1) << bits) - T(1));
  return (v >> pos) & mask;
}

/** 将 base 的 [pos, pos+bits) 位段设置为 value（其余位保持不变） */
template <typename T>
__device__ __forceinline__ T setBitfield(T base, T value, int pos, int bits) {
  const T maskBits = (bits >= (int)(sizeof(T) * 8)) ? ~T(0) : ((T(1) << bits) - T(1));
  const T clrMask  = ~(maskBits << pos);
  return (base & clrMask) | ((value & maskBits) << pos);
}

/** 获取当前线程在 warp 内的 lane id（0..31） */
__device__ __forceinline__ unsigned getLaneId() {
  return threadIdx.x & 31;
}

} // namespace mini

// ===================== TopKTypeConfig：数据到“可比较位型”的变换 =====================
// 通过“单调映射”把原始标量转为无符号位型，以支持基于基数（radix）的比较与选择

template <typename scalar_t>
struct TopKTypeConfig {};

// --- float 专用：把 float 映到 uint32_t，保证“有序可比较”（含 NaN 处理）
template <>
struct TopKTypeConfig<float> {
  using RadixType = uint32_t;

  // 将 float 映射为“可比较”的无符号整数：
  // 负数区间翻转、有符号位处理，以保证按无符号排序等价于按浮点排序
  static inline __device__ RadixType convert(float v) {
    RadixType x = __float_as_int(v);
    RadixType mask = (x & 0x80000000) ? 0xffffffffu : 0x80000000u;
    // 对 NaN：统一映成“最大值”，这样在取最大/最小时可按约定处理
    return (v == v) ? (x ^ mask) : 0xffffffffu;
  }

  // 逆映射：将“可比较”的无符号整数还原为 float
  static inline __device__ float deconvert(RadixType v) {
    RadixType mask = (v & 0x80000000u) ? 0x80000000u : 0xffffffffu;
    return __int_as_float(v ^ mask);
  }
};

// --- int 专用：把有符号 int 映到 uint32_t（加 2^31 偏移），使其按无符号比较等价于按有符号比较
template <>
struct TopKTypeConfig<int> {
  using RadixType = uint32_t;

  static inline __device__ RadixType convert(int v) {
    static_assert(sizeof(int) == 4, "int must be 4 bytes");
    return 2147483648u + static_cast<uint32_t>(v); // 加 2^31 偏移
  }

  static inline __device__ int deconvert(RadixType v) {
    return static_cast<int>(v - 2147483648u);
  }
};

// ===================== 基数计数（带掩码）核心内核 =====================
// 在一个给定的位段（radixDigitPos 开始，宽度 RadixBits）上，统计每个 digit（0..RadixSize-1）的出现次数。
// 只统计满足 (val & desiredMask) == desired 的元素（即“当前前缀”匹配的元素）。

template <
    typename scalar_t,   // 原始标量类型（float / int）
    typename bitwise_t,  // convert() 后的无符号位型（如 uint32_t）
    typename index_t,    // 索引类型（size_t / int）
    typename CountType,  // 计数类型（int）
    int RadixSize,       // 基数大小（= 2^RadixBits）
    int RadixBits>       // 位段宽度
__device__ void countRadixUsingMask(
    CountType counts[RadixSize],  // 输出：每个 digit 的计数（寄存器/局部数组）
    CountType* smem,              // 临时共享内存计数（RadixSize 大小）
    bitwise_t desired,            // 目标前缀位模式
    bitwise_t desiredMask,        // 目标前缀掩码
    int radixDigitPos,            // 本次统计所用的位段起始 bit 位置
    index_t sliceSize,            // 待统计的元素数量
    index_t withinSliceStride,    // 跨度（通常为 1）
    const scalar_t* data) {       // 输入数据（连续或跨距访问）

  // 1) 清零每线程的局部计数
#pragma unroll
  for (int i = 0; i < RadixSize; ++i) {
    counts[i] = 0;
  }

  // 2) 清零共享计数
  if (threadIdx.x < RadixSize) {
    smem[threadIdx.x] = 0;
  }
  __syncthreads();

  // 3) 记录 warp 活跃掩码（非 ROCm 下）
#if !defined(USE_ROCM)
  unsigned mask = WARP_BALLOT(threadIdx.x < sliceSize); // 仅线程 i<sliceSize 为活跃
#endif

  // 4) 以 blockDim.x 为步长扫描整段数据
  for (index_t i = threadIdx.x; i < sliceSize; ) {
    // 将元素变换到“可比较位型”
    bitwise_t val = TopKTypeConfig<scalar_t>::convert(doLdg(&data[i * withinSliceStride]));
    // 是否匹配当前的“前缀筛选”
    bool hasVal = ((val & desiredMask) == desired);
    // 取出当前位段的 digit
    bitwise_t digitInRadix = mini::getBitfield<bitwise_t>(val, radixDigitPos, RadixBits);

    // 5) 对每个 digit 做一次 ballot，统计该 digit 的匹配数量（warp 内 popcount）
#pragma unroll
    for (uint32_t j = 0; j < RadixSize; ++j) {
      bool vote = hasVal && (digitInRadix == j);
#if defined(USE_ROCM)
      counts[j] += __popcll(WARP_BALLOT(vote));
#else
      counts[j] += __popc(WARP_BALLOT(vote, mask));
#endif
    }

    // 前进到下一个“本线程负责”的元素
    i += blockDim.x;

#if !defined(USE_ROCM)
    // 更新当前迭代的活跃掩码（只有 i<sliceSize 的 lane 才继续参与 ballot）
    mask = WARP_BALLOT(i < sliceSize, mask);
#endif
  }

  // 6) 每个 warp 的 lane0 把自己局部 counts 原子加到共享计数 smem（避免写冲突）
  if (mini::getLaneId() == 0) {
#pragma unroll
    for (uint32_t i = 0; i < RadixSize; ++i) {
      gpuAtomicAddNoReturn(&smem[i], counts[i]);
    }
  }
  __syncthreads();

  // 7) 将共享计数拷回每线程的 counts（后续会被使用）
#pragma unroll
  for (uint32_t i = 0; i < RadixSize; ++i) {
    counts[i] = smem[i];
  }
  __syncthreads();
}

// ===================== Radix 选择的配置常量 =====================
constexpr int RADIX_BITS = 2;            // 每次处理 2 个比特（即 4 进制）
constexpr int RADIX_SIZE = 4;            // 2^RADIX_BITS
constexpr int RADIX_MASK = (RADIX_SIZE - 1);

// ===================== 在匹配的前缀下“找出一个实际值” =====================
// findPattern：在 (val & desiredMask) == desired 的集合中，返回遇到的第一个元素的原始值。
// 用于“当某一位段的计数恰好覆盖第 k 个元素且该段只含 1 个元素”时，直接取值。

template <typename scalar_t, typename bitwise_t, typename index_t>
__device__ scalar_t findPattern(
    scalar_t* smem,                 // 用作 (flag, value) 的共享缓存，至少 2 个标量位
    const scalar_t* data,
    index_t sliceSize,
    index_t withinSliceStride,
    bitwise_t desired,              // 匹配的前缀值
    bitwise_t desiredMask) {        // 匹配的前缀掩码

  // 用 smem[0] 做“找到标记”，smem[1] 存放找到的值
  if (threadIdx.x < 2) {
    smem[threadIdx.x] = static_cast<scalar_t>(0);
  }
  __syncthreads();

  // 为了简化循环次数，向上取整到 blockDim 的整数倍
  const index_t numIterations = round_up(sliceSize, static_cast<index_t>(blockDim.x));

  for (index_t i = threadIdx.x; i < numIterations; i += blockDim.x) {
    bool inRange = (i < sliceSize);
    scalar_t v   = inRange ? doLdg(&data[i * withinSliceStride]) : static_cast<scalar_t>(0);

    // 如果在范围内且满足“前缀筛选”，写共享内存（全体线程随后读取到相同结果）
    if (inRange && ((TopKTypeConfig<scalar_t>::convert(v) & desiredMask) == desired)) {
      smem[0] = static_cast<scalar_t>(1); // 标志
      smem[1] = v;                        // 值（不能用 v 作为 flag，因为 v 可能为 0）
    }

    __syncthreads();
    scalar_t found = smem[0];
    scalar_t val   = smem[1];
    __syncthreads();

    if (found != static_cast<scalar_t>(0)) {
      return val; // 全体线程返回同一值，提前结束
    }
  }

  // 正常流程应在上面返回，这里若触达说明逻辑有问题
  CUDA_KERNEL_ASSERT(false);
  return static_cast<scalar_t>(0);
}

// ===================== 基于基数的第 k 选择（Radix-Select） =====================
// 思路：从最高位段到最低位段，逐段统计每个 digit 的覆盖数量，决定该位段取哪一个 digit，
// 从而逐步收缩“可能落入第 k 个”的值域前缀；遇到“唯一命中且 k==1”时直接查找返回。

template <typename scalar_t, typename bitwise_t, typename index_t>
__device__ void radixSelect(
    const scalar_t* data,       // 输入数据
    index_t k,                  // 第 k（1-based）
    bool largest,               // true 求第 k 大；false 求第 k 小
    index_t sliceSize,          // 数据长度
    index_t withinSliceStride,  // 跨度（通常为 1）
    int* smem,                  // 共享计数（大小至少 RADIX_SIZE * sizeof(int)）
    scalar_t* topK) {           // 输出：第 k 值

  int counts[RADIX_SIZE];       // 当前位段上每个 digit 的计数（线程局部缓存）

  bitwise_t desired     = 0;    // 逐步构造的“前缀值”（在 bit 域上）
  bitwise_t desiredMask = 0;    // 已决定的前缀掩码
  int kToFind = static_cast<int>(k);

  // 从“最高位段”开始逐段向低位推进
  for (int digitPos = sizeof(scalar_t) * 8 - RADIX_BITS; digitPos >= 0; digitPos -= RADIX_BITS) {
    // 统计在当前位段 digitPos 上的 counts（只统计满足当前前缀的元素）
    countRadixUsingMask<
        scalar_t, bitwise_t, index_t, int, RADIX_SIZE, RADIX_BITS>(
        counts, smem, desired, desiredMask, digitPos, sliceSize, withinSliceStride, data);

    // 当某 digit 的计数恰好 == 1 且 kToFind==1，可以直接找回该唯一值并返回
    auto found_unique = [&](int i, int count) -> bool {
      if (count == 1 && kToFind == 1) {
        desired     = mini::setBitfield<bitwise_t>(desired, i,        digitPos, RADIX_BITS);
        desiredMask = mini::setBitfield<bitwise_t>(desiredMask, RADIX_MASK, digitPos, RADIX_BITS);
        *topK = findPattern<scalar_t, bitwise_t, index_t>(
            (scalar_t*)smem, data, sliceSize, withinSliceStride, desired, desiredMask);
        return true;
      }
      return false;
    };

    // 非唯一：若某 digit 的计数 >= kToFind，则第 k 落在这个 digit 里；否则减去并继续
    auto found_non_unique = [&](int i, int count) -> bool {
      if (count >= kToFind) {
        desired     = mini::setBitfield<bitwise_t>(desired, i,        digitPos, RADIX_BITS);
        desiredMask = mini::setBitfield<bitwise_t>(desiredMask, RADIX_MASK, digitPos, RADIX_BITS);
        return true; // 该位段 digit 决定，进入下一位段
      }
      kToFind -= count; // 第 k 位于剩余 digit
      return false;
    };

    // 求第 k 大：digit 从大到小扫描；求第 k 小：digit 从小到大扫描
    if (largest) {
#pragma unroll
      for (int i = RADIX_SIZE - 1; i >= 0; --i) {
        int count = counts[i];
        if (found_unique(i, count)) return;
        if (found_non_unique(i, count)) break;
      }
    } else {
#pragma unroll
      for (int i = 0; i < RADIX_SIZE; ++i) {
        int count = counts[i];
        if (found_unique(i, count)) return;
        if (found_non_unique(i, count)) break;
      }
    }
  }

  // 所有位段走完：desired 就是目标值的可比较表示，反变换得到标量
  *topK = TopKTypeConfig<scalar_t>::deconvert(desired);
}

// ===================== 启动器（kernel + host 包装） =====================

template <typename T>
__global__ void kth_select_kernel(
    const T* __restrict__ data, // 输入数组
    size_t sliceSize,           // 元素数量
    size_t k,                   // 第 k（1-based）
    bool largest,               // true=第 k 大，false=第 k 小
    size_t stride,              // withinSliceStride（通常=1，支持步长访问）
    T* __restrict__ out) {      // 输出指针（单个标量）

  extern __shared__ int smem[]; // 动态共享内存：至少 RADIX_SIZE * sizeof(int)
  using BitwiseT = typename TopKTypeConfig<T>::RadixType;

  T top;
  radixSelect<T, BitwiseT, size_t>(
      data,
      static_cast<size_t>(k),
      largest,
      static_cast<size_t>(sliceSize),
      static_cast<size_t>(stride),
      smem,
      &top);

  if (threadIdx.x == 0) {
    *out = top;
  }
}

/**
 * @brief 纯主机端包装：返回输入向量中的第 k 大元素（k 从 1 开始）
 *        - 若 k > n/2，则转化为求第 (n-k+1) 小，可减少扫描偏向
 *        - 输入必须非空，且 1 <= k <= n
 */
template <typename T>
T kthLargest(const std::vector<T>& h_input, size_t k) {
  if (h_input.empty() || k == 0 || k > h_input.size()) {
    return T(-100);  // 错误返回（你也可以选择抛异常/返回 NaN）
  }

  const size_t n      = h_input.size();
  const size_t stride = 1; // 连续内存访问

  // 设备侧申请内存并拷贝数据
  T *d_input = nullptr, *d_output = nullptr;
  CUDA_CHECK(cudaMalloc(&d_input,  n * sizeof(T)));
  CUDA_CHECK(cudaMalloc(&d_output, sizeof(T)));
  CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), n * sizeof(T), cudaMemcpyHostToDevice));

  // 启动参数：单 block（1024 线程）+ 动态共享内存（仅需 RADIX_SIZE * sizeof(int)）
  constexpr int blockSize = 1024;
  dim3 block(blockSize);
  dim3 grid(1);
  const size_t shmemBytes = RADIX_SIZE * sizeof(int);

  // 若 k 比较靠后，则转为求“第 (n-k+1) 小”，同时 largest=false
  bool largest = true;
  if (k > n / 2) {
    k = n - k + 1;
    largest = false;
  }

  // 启动核函数
  kth_select_kernel<T><<<grid, block, shmemBytes>>>(
      d_input,
      n,
      k,
      largest,
      stride,
      d_output);

  // 错误检查与同步（非常重要）
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  // 拷回结果并释放资源
  T result{};
  CUDA_CHECK(cudaMemcpy(&result, d_output, sizeof(T), cudaMemcpyDeviceToHost));
  cudaFree(d_input);
  cudaFree(d_output);

  return result;
}





/**
 * @brief Computes flash attention for given query, key, and value tensors.
 * 
 * @tparam T Data type (float) for input/output tensors
 * @param[in] h_q Query tensor of shape [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param[in] h_k Key tensor of shape [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[in] h_v Value tensor of shape [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[out] h_o Output attention tensor of shape [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param[in] batch_size Batch dimension size
 * @param[in] target_seq_len Target sequence length
 * @param[in] src_seq_len Source sequence length  
 * @param[in] query_heads Number of query attention heads
 * @param[in] kv_heads Number of key/value heads (supports grouped query attention)
 * @param[in] head_dim Dimension size of each attention head
 * @param[in] is_causal Whether to apply causal masking
 */

// ===== 数据结构：记录一个 tile 的 softmax 累计状态 =====
struct MD_F {
    float m; // 当前块（或累计）的最大值（log-sum-exp 的稳定化基准）
    float d; // 当前块（或累计）的归一化因子（∑exp(score - m)）
};

// ===== warp/block 归约求和函数 =====
__inline__ __device__ float blockAllReduceSum(float x) {
    // warp 内归约：通过 shuffle 指令在一个 warp 内做 sum
    for (int offset = 16; offset > 0; offset >>= 1)
        x += __shfl_down_sync(0xffffffff, x, offset);

    // 使用共享内存聚合各个 warp 的结果（假设 blockDim.x 是 32 的倍数）
    __shared__ float smem[32];              // 最多支持 1024 线程（32 warp）
    int lane = threadIdx.x & 31;            // 当前线程在 warp 内的 lane id
    int wid  = threadIdx.x >> 5;            // 当前线程所在 warp id
    if (lane == 0) smem[wid] = x;           // 每个 warp 的 lane0 写入共享内存
    __syncthreads();

    // 再让前几个 warp 的 lane 来做一次归约
    x = (threadIdx.x < (blockDim.x >> 5)) ? smem[lane] : 0.0f;
    if (wid == 0) {
        for (int offset = 16; offset > 0; offset >>= 1)
            x += __shfl_down_sync(0xffffffff, x, offset);
    }

    // 最终结果广播给 warp 内所有线程
    return __shfl_sync(0xffffffff, x, 0);
}
__inline__ __device__ float blockAllReduceMax( float m_local ) {
                  for (int off = 16; off > 0; off >>= 1)  m_local = fmaxf(m_local, __shfl_down_sync(0xffffffff, m_local, off));
                __shared__ float smax[32];
                int lane = threadIdx.x & 31;
                int wid  = threadIdx.x >> 5;
                if (lane == 0) smax[wid] = m_local;
                __syncthreads();
                float m_block = (threadIdx.x < (blockDim.x >> 5)) ? smax[lane] : -1e20f;
                if (wid == 0) {
                    for (int off = 16; off > 0; off >>= 1)
                        m_block = fmaxf(m_block, __shfl_down_sync(0xffffffff, m_block, off));
                }
                return  __shfl_sync(0xffffffff, m_block, 0);
}

// ===== 合并两个 (m, d) 的函数：用于在线 softmax =====
__inline__ __device__ MD_F md_merge(MD_F a, MD_F b) {
    float m = fmaxf(a.m, b.m);                        // 新的 m 是二者中的最大
    float d = a.d * __expf(a.m - m)                   // 将 a 的和缩放到新的基准
             + b.d * __expf(b.m - m);                 // 同理缩放 b 的和
    return {m, d};                                    // 返回合并后的 (m, d)
}


// ====== 核函数（按 K/V 的 Bc 列块、对每个 Q 的 N 行做在线 softmax） ======
template <int Bc, int Br>
__global__ void flashAttentionKernel(
    const float* __restrict__ Q,          // [B, Hq, N, d]
    const float* __restrict__ K,          // [B, Hk, M, d]
    const float* __restrict__ V,          // [B, Hk, M, d]
    float* __restrict__ O,                // [B, Hq, N, d]
    float* __restrict__ lbuf,             // [B, Hq, N] (online softmax 累加用)
    float* __restrict__ mbuf,             // [B, Hq, N]
    int B, int Hq, int Hk, int d,
    float softmax_scale, const uint8_t *causal_mask, // [B, N, M] 或 nullptr
    int N, int M)
{
    // 网格：x = query head，y = batch
    const int b  = blockIdx.y;
    const int hq = blockIdx.x;
    const int hk = (Hk == 1) ? 0 : (hq % Hk); // MQA/GQA 映射

    const size_t stride_qo = (size_t)N * d;
    const size_t stride_kv = (size_t)M * d;
    const size_t stride_lm = (size_t)N;

    const size_t qo_base = ((size_t)b * Hq + hq) * stride_qo;
    const size_t kv_base = ((size_t)b * Hk + hk) * stride_kv;
    const size_t lm_base = ((size_t)b * Hq + hq) * stride_lm;
    const size_t mask_base = (size_t)b * (size_t)N * (size_t)M;

    extern __shared__ float s_ptr[];
    float *s_Q = s_ptr;                // [d]
    float *s_K = s_Q + d;              // [Bc, d]
    float *s_V = s_K + Bc * d;         // [Bc, d]
    float *s_S = s_V + Bc * d;         // [Bc]
    __shared__ MD_F row_prev;

    for (int kv_col0 = 0; kv_col0 < M; kv_col0 += Bc) {
        // 加载一个 [Bc, d] 的 K/V tile 到共享内存
        for (int t = threadIdx.x; t < Bc * d; t += blockDim.x) {
            int row = t / d;
            int col = t % d;
            int gcol = kv_col0 + row;
            if (gcol < M) {
                s_K[row * d + col]= K[kv_base + (size_t)gcol * d + col];
                s_V[row * d + col]= V[kv_base + (size_t)gcol * d + col];
            }else{
              s_K[row * d + col] = 0.0f;
              s_V[row * d + col] = 0.0f;
            }
        }
        __syncthreads();

        // 遍历 Q 的 N 行
        for (int n = 0; n < N; ++n) {
            // 加载一行 Q 到共享内存
            for (int t = threadIdx.x; t < d; t += blockDim.x) {
                s_Q[t] = Q[qo_base + (size_t)n * d + t];
            }

            // 取上一个 tile 合并前的 (m, l)
            
            if (threadIdx.x == 0) {
                row_prev = {mbuf[lm_base + n], lbuf[lm_base + n]};
            }
            __syncthreads();

            // 当前 tile 的 (m, l)
            MD_F tile_ml = {-1e20f, 0.0f};
       
          
            // 计算本 tile 的分数 S = (Q·K^T) * scale，并做行内最大与 exp 累加
            for (int kc = 0; kc < Bc; ++kc) {
                int m_col = kv_col0 + kc;
              
                MD_F tmp_ml = {0.0f,1.0f};
                for (int t = threadIdx.x; t < d; t += blockDim.x) {
                    tmp_ml.m += s_Q[t] * s_K[kc * d + t];
                }
                

                // 因果 mask（如果提供）：被 mask 的位置直接设为 -inf
                
                if (causal_mask) {
                    uint8_t keep = causal_mask[mask_base + (size_t)n * M + m_col];
                    if (!keep) tmp_ml.m= -INFINITY;
                } 

                tmp_ml.m= tmp_ml.m* softmax_scale;
                __syncthreads();
                tmp_ml.m = blockAllReduceSum(tmp_ml.m) ;

                if (threadIdx.x == 0) s_S[kc] = tmp_ml.m;
                __syncthreads();

                // 用本 block 的所有线程更新 tile 的 (m, l)
                MD_F cur = {tmp_ml.m, 1.0f};
                // reduce 最大值
                float m_local = cur.m;
                // 用 warp reduce 求最大（借助 shfl）
                float m_block = blockAllReduceMax(m_local) ;

                // 所有人用统一 m_block 计算本元素的 e^(s - m)
                float e = __expf(tmp_ml.m- m_block);
                float e_sum = blockAllReduceSum(e);
                if (threadIdx.x == 0) {
                    MD_F tmp = {m_block, e_sum};
                    tile_ml = md_merge(tile_ml, tmp);
                }
                __syncthreads();
            }

            __shared__ MD_F row_new;
            if (threadIdx.x == 0) {
                row_new = md_merge(row_prev, tile_ml);
            }
            __syncthreads();

            // 计算 O 的一行（结合旧 O 与本 tile 的 V，加权）
            for (int t = threadIdx.x; t < d; t += blockDim.x) {
                // 本 tile 对 O 的增量 pv = sum_x softmax * V
                // float pv = 0.f;
                // for (int kc = 0; kc < Bc; ++kc) {
                //     if (s_S[kc] != -INFINITY) pv += __expf(s_S[kc] - tile_ml.m)* s_V[kc * d + t];
                // }

              
                // // 合并旧 O（缩放到新 m）+ 新增量（缩放到新 m）
                // float oldO = O[qo_base + (size_t)n * d + t];
                // float newO =
                //     (row_prev.d > 0.f ? oldO * (row_prev.d * __expf(row_prev.m - row_new.m)) : 0.f)
                //     + (tile_ml.d  > 0.f ? pv   * (        __expf(tile_ml .m - row_new.m)) : 0.f);
                // newO /= (row_new.d + 1e-6f);
                // O[qo_base + (size_t)n * d + t] = newO;
                // 当前 tile 的 (m, l, O)
float m_tile = tile_ml.m;
float l_tile = tile_ml.d;

// 计算 O_tile = sum_j exp(score_j - m_tile) * V_j
float pv = 0.f;
for (int kc = 0; kc < Bc; ++kc) {
    if (s_S[kc] != -INFINITY) {
        pv += __expf(s_S[kc] - m_tile) * s_V[kc * d + t];
    }
}
float O_tile = pv;  // 注意这里相当于按 softmax 分子加权 V

// 合并旧状态和当前 tile
float m_old = row_prev.m;
float l_old = row_prev.d;
float O_old = O[qo_base + (size_t)n * d + t];

float m_new = fmaxf(m_old, m_tile);
float l_new = __expf(m_old - m_new) * l_old + __expf(m_tile - m_new) * l_tile;

float O_new =
    (__expf(m_old - m_new) * l_old * O_old +
     __expf(m_tile - m_new) * O_tile) / (l_new + 1e-6f);

O[qo_base + (size_t)n * d + t] = O_new;

            }

            if (threadIdx.x == 0) {
                lbuf[lm_base + n] = row_new.d;
                mbuf[lm_base + n] = row_new.m;
            }
            __syncthreads();
        }
        __syncthreads();
    }
}
__global__ void InitML(float* m, float* l, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        m[i] = -INFINITY;  // 正确的初值：最大值为 -INF
        l[i] = 0.0f;       // 正确的初值：累加和为 0
    }
}


// ====== 因果掩码构建（修了 klen 赋值错误） ======
template<typename T>
__global__ void BuildCausalMasks(T* mask,  int max_q_len, int max_k_len){
    int qlen = max_q_len;
    int klen = max_k_len;
    mask += blockIdx.x * max_q_len * max_k_len; // 每个 batch 一片
    int offset = threadIdx.x;
    while (offset < max_q_len * max_k_len){
        int q = offset / max_k_len;
        int k = offset % max_k_len;
        // 允许右侧 padding 的通用三角（适配 qlen<=klen, 以及 qlen!=klen）
        bool keep = (q < qlen) && (k < klen) && (k <= q + (klen - qlen)) && (k >= klen - qlen);
        mask[offset] = static_cast<T>(keep);
        offset += blockDim.x;
    }
}

template <typename T>
void flashAttention(const std::vector<T>& h_q, const std::vector<T>& h_k,
                    const std::vector<T>& h_v, std::vector<T>& h_o,
                    int batch_size, int target_seq_len, int src_seq_len, 
                    int query_heads, int kv_heads, int head_dim, bool is_causal)
{
     // 打印 Q、K、V 和 O 的维度
    // std::cout << "Q dimensions: [" << batch_size << ", " << query_heads << ", " << target_seq_len << ", " << head_dim << "]" << std::endl;
    // std::cout << "K dimensions: [" << batch_size << ", " << kv_heads << ", " << src_seq_len << ", " << head_dim << "]" << std::endl;
    // std::cout << "V dimensions: [" << batch_size << ", " << kv_heads << ", " << src_seq_len << ", " << head_dim << "]" << std::endl;
    // std::cout << "O dimensions: [" << batch_size << ", " << query_heads << ", " << target_seq_len << ", " << head_dim << "]" << std::endl;
    // std::cout << "l dimensions: [" << batch_size << ", " << query_heads << ", " << target_seq_len << ", 1]" << std::endl;
    // std::cout << "m dimensions: [" << batch_size << ", " << query_heads << ", " << target_seq_len << ", 1]" << std::endl;

    // // 打印是否使用因果掩码
    // std::cout << "Using causal mask: " << (is_causal ? "Yes" : "No") << std::endl;

    // // 额外：打印目标和源序列长度
    // std::cout << "Target sequence length (N): " << target_seq_len << std::endl;
    // std::cout << "Source sequence length (M): " << src_seq_len << std::endl;
    // std::cout << "Batch size (B): " << batch_size << std::endl;
    // std::cout << "Query heads (Hq): " << query_heads << std::endl;
    // std::cout << "KV heads (Hk): " << kv_heads << std::endl;
    // std::cout << "Head dimension (d): " << head_dim << std::endl;

/*
Q dimensions: [1, 1, 1, 8]
K dimensions: [1, 1, 1, 8]
V dimensions: [1, 1, 1, 8]
O dimensions: [1, 1, 1, 8]
l dimensions: [1, 1, 1, 1]
m dimensions: [1, 1, 1, 1]
Using causal mask: Yes
Target sequence length (N): 1
Source sequence length (M): 1
Batch size (B): 1
Query heads (Hq): 1
KV heads (Hk): 1
Head dimension (d): 8
Testase #6
Data type:      float
Warm-up iters:  1
Profile iters:  10
Avg time:       0.193257 ms
Verification:   Failed

Q dimensions: [3, 6, 10, 24]
K dimensions: [3, 2, 1, 24]
V dimensions: [3, 2, 1, 24]
O dimensions: [3, 6, 10, 24]
l dimensions: [3, 6, 10, 1]
m dimensions: [3, 6, 10, 1]
Using causal mask: No
Target sequence length (N): 10
Source sequence length (M): 1
Batch size (B): 3
Query heads (Hq): 6
KV heads (Hk): 2
Head dimension (d): 24
Testase #10
Data type:      float
Warm-up iters:  1
Profile iters:  10
Avg time:       0.218154 ms
Verification:   Failed


*/

    const size_t q_elems  = (size_t)batch_size * query_heads * target_seq_len * head_dim;
    const size_t k_elems  = (size_t)batch_size * kv_heads   * src_seq_len    * head_dim;
    const size_t v_elems  = k_elems;
    const size_t o_elems  = q_elems;
    const size_t lm_elems = (size_t)batch_size * query_heads * target_seq_len;

    h_o.assign(o_elems, 0.f);

    size_t q_bytes = q_elems * sizeof(float);
    size_t k_bytes = k_elems * sizeof(float);
    size_t v_bytes = v_elems * sizeof(float);
    size_t o_bytes = o_elems * sizeof(float);
    size_t l_bytes = lm_elems * sizeof(float);
    size_t m_bytes = lm_elems * sizeof(float);

    size_t total_bytes = q_bytes + k_bytes + v_bytes + o_bytes + l_bytes + m_bytes;

    float* d_base = nullptr;
    cudaMalloc((void**)&d_base, total_bytes);

    float* d_Q = d_base;
    float* d_K = (float*)((char*)d_Q + q_bytes);
    float* d_V = (float*)((char*)d_K + k_bytes);
    float* d_O = (float*)((char*)d_V + v_bytes);
    float* d_l = (float*)((char*)d_O + o_bytes);
    float* d_m = (float*)((char*)d_l + l_bytes);

    // H2D
    cudaMemcpy(d_Q, h_q.data(), q_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_k.data(), k_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_v.data(), v_bytes, cudaMemcpyHostToDevice);
    cudaMemset(d_O, 0, o_bytes);

    
   
    int threads = 256;
    int blocks  = (int)((lm_elems + threads - 1) / threads);
    InitML<<<blocks, threads>>>(d_m, d_l, lm_elems); // 注意顺序：(m, l)
    cudaDeviceSynchronize();



    // 构造可选因果 mask
    uint8_t* d_mask = nullptr;
    if (is_causal) {
        cudaMalloc(&d_mask, sizeof(uint8_t) * (size_t)batch_size * target_seq_len * src_seq_len);
        BuildCausalMasks<uint8_t><<<batch_size, 256>>>(d_mask, target_seq_len, src_seq_len);
    }

    // launch 参数
    constexpr int Bc = 1; // KV 列块（可按硬件改）
    constexpr int Br = 1;  // 每 block 线程束个数的倍数（影响 blockDim）
    //(void)Br;

    // 模板块尺寸要求：Bc 为常量，不能在运行时修改；若不整除也没关系，代码内部已做越界保护
    const float softmax_scale = 1.0f / sqrtf((float)head_dim);

    // 共享内存：Q[d] + K[Bc,d] + V[Bc,d] + S[Bc]
    const size_t sram_floats  = head_dim + 2ull * Bc * head_dim + (size_t)Bc;
    const size_t sram_bytes   = sram_floats * sizeof(float);

    dim3 grid_dim(query_heads, batch_size);
    dim3 block_dim(128); // 4 warps，既能跑满也不易溢出共享内存

    flashAttentionKernel<Bc, Br><<<grid_dim, block_dim, sram_bytes>>>(
        d_Q, d_K, d_V, d_O, d_l, d_m,
        batch_size, query_heads, kv_heads, head_dim,
        softmax_scale, is_causal ? d_mask : nullptr,
        target_seq_len, src_seq_len);

    cudaDeviceSynchronize();
   

    // D2H
    cudaMemcpy(h_o.data(), d_O, o_bytes, cudaMemcpyDeviceToHost);

    // 释放
    if (d_mask) cudaFree(d_mask);
    cudaFree(d_base);
}


// *********************************************************************
// Explicit Template Instantiations (REQUIRED FOR LINKING WITH TESTER.O)
// DO NOT MODIFY THIS SECTION
// *********************************************************************
template int kthLargest<int>(const std::vector<int>&, size_t);
template float kthLargest<float>(const std::vector<float>&, size_t);
template void flashAttention<float>(const std::vector<float>&, const std::vector<float>&,
  const std::vector<float>&, std::vector<float>&,
  int, int, int, int, int, int, bool);
