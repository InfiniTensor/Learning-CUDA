#include <vector>
#include <type_traits>
#include <cmath>
#include <cuda_fp16.h>

#include "../tester/utils.h"

// - tester 会在主机(CPU)端调用下面两个函数，输入/输出都用 std::vector 承载。
// - 主要流程：主机->设备(GPU)拷贝、启动 CUDA 核函数、设备->主机拷回结果。
// - 先把正确性跑通，再考虑性能优化。

// -----------------------------
// 工具函数：用于在设备端做类型转换（half <-> float）
// -----------------------------
template <typename T>
__device__ __forceinline__ float to_float(T x) {
  return static_cast<float>(x);
}

template <>
__device__ __forceinline__ float to_float<half>(half x) {
  return __half2float(x);
}

template <typename T>
__host__ __device__ __forceinline__ T from_float(float x) {
  return static_cast<T>(x);
}

template <>
__host__ __device__ __forceinline__ half from_float<half>(float x) {
  return __float2half_rn(x);
}

// -----------------------------
// Trace 核函数：对角线元素求和（并行 + 原子累加）
// -----------------------------
template <typename T>
__global__ void trace_kernel(const T* __restrict__ input, size_t rows, size_t cols,
                             size_t diag_len, T* __restrict__ out) {
  // 每个线程负责若干个对角线元素的累加，最后用 atomicAdd 汇总到 out。
  size_t tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;

  T local = T(0);
  for (size_t i = tid; i < diag_len; i += stride) {
    local += input[i * cols + i];
  }

  // 原子累加
  atomicAdd(out, local);
}

/**
 * @brief Computes the trace of a matrix.
 *
 * The trace of a matrix is defined as the sum of its diagonal elements.
 * This function expects a flattened row-major matrix stored in a
 * std::vector. If the matrix is not square, the trace will sum up
 * elements along the main diagonal up to the smaller of rows or cols.
 *
 * @tparam T The numeric type of matrix elements (e.g., float, int).
 * @param h_input A flattened matrix of size rows * cols.
 * @param rows Number of rows in the matrix.
 * @param cols Number of columns in the matrix.
 * @return The trace (sum of diagonal values) of the matrix.
 */
template <typename T>
T trace(const std::vector<T>& h_input, size_t rows, size_t cols) {
  // TODO: Implement the trace function

  // 处理边界情况：空矩阵 -> trace = 0
  if (rows == 0 || cols == 0) return T(0);

  const size_t diag_len = (rows < cols) ? rows : cols;
  const size_t numel = rows * cols;

  // 申请设备端缓冲区
  T* d_input = nullptr;
  T* d_out = nullptr;
  RUNTIME_CHECK(cudaMalloc(&d_input, numel * sizeof(T)));
  RUNTIME_CHECK(cudaMalloc(&d_out, sizeof(T)));

  // 拷贝输入到 GPU，并把输出初始化为 0
  RUNTIME_CHECK(cudaMemcpy(d_input, h_input.data(), numel * sizeof(T), cudaMemcpyHostToDevice));
  RUNTIME_CHECK(cudaMemset(d_out, 0, sizeof(T)));

  // 启动核函数：简单的一维 grid（易读但未调优）
  const int threads = 256;
  int blocks = static_cast<int>((diag_len + threads - 1) / threads);
  // 某些平台/老 GPU 的 gridDim.x 有上限，这里做个保守截断。
  if (blocks > 65535) blocks = 65535;
  trace_kernel<T><<<blocks, threads>>>(d_input, rows, cols, diag_len, d_out);
  RUNTIME_CHECK(cudaGetLastError());
  RUNTIME_CHECK(cudaDeviceSynchronize());

  // 拷回结果
  T h_out = T(0);
  RUNTIME_CHECK(cudaMemcpy(&h_out, d_out, sizeof(T), cudaMemcpyDeviceToHost));

  // 释放资源
  RUNTIME_CHECK(cudaFree(d_input));
  RUNTIME_CHECK(cudaFree(d_out));
  return h_out;
}

// -----------------------------
// 分块优化版 Flash Attention 核函数
// -----------------------------
// 说明：这里的“分块”是按 src_len 维度把 K/V 切成 tile，配合共享内存减少反复的全局内存读取。
// 注意：为了保证共享内存里装的是同一份 K/V，本 kernel 让一个 block 只处理同一个 (b, t, kv_head)，
// block 内的线程分别负责该 kv_head 组里的不同 q_head（即 GQA 的一个 group）。
//
// 另外，为了更容易通过 float 的严格误差容忍用例，这里用“两遍 softmax”（先求 max，再求 sum 与输出）。
//
// 备注：为了只保留“一份” Flash Attention 核函数，本文件把两种实现路径合并进同一个 kernel：
// - tiled 模式：按 (b, t, kvh) 分 block + 共享内存分块缓存 K/V（更快，且更稳定）
// - fallback 模式：按 (b, t, qh) 分配线程做一个通用实现（更慢，但形状更通用）
#define TILE_SIZE 32  // tile 大小（沿 src_len 维度）
#define MAX_HEAD_DIM 128  // 共享内存里 head_dim 的上限（常见 64/128）

template <typename T>
__global__ void flash_attention_kernel(const T* __restrict__ q,
                                       const T* __restrict__ k,
                                       const T* __restrict__ v,
                                       float* __restrict__ o_float,
                                       int batch_size, int tgt_len, int src_len,
                                       int q_heads, int kv_heads, int head_dim,
                                       bool is_causal, bool use_tiled) {
  // 共享内存缓存一小段 K/V：2 * TILE_SIZE * MAX_HEAD_DIM * 4 bytes
  // 例如 TILE_SIZE=32, MAX_HEAD_DIM=128 => 32KB（通常在 48KB 共享内存限制内）
  __shared__ float s_k[TILE_SIZE][MAX_HEAD_DIM];
  __shared__ float s_v[TILE_SIZE][MAX_HEAD_DIM];

  if (src_len <= 0 || head_dim <= 0) return;
  if (kv_heads <= 0) return;

  // ------------------------------------------------------------
  // tiled 模式：block 负责 (b, t, kvh)，thread 负责同一 group 的不同 qh
  // ------------------------------------------------------------
  if (use_tiled) {
    if (head_dim > MAX_HEAD_DIM) return;
    if (q_heads % kv_heads != 0) return;  // tiled 版本按标准 GQA 组织线程

    const int group = q_heads / kv_heads;  // 一个 kv_head 对应的 q_head 数
    const int block_id = blockIdx.x;
    const int kvh = block_id % kv_heads;
    const int bt = block_id / kv_heads;
    const int t = bt % tgt_len;
    const int b = bt / tgt_len;
    if (b >= batch_size) return;

    // block 内线程负责该 kvh 的不同 qh
    const int local_q = threadIdx.x;
    const int qh = kvh * group + local_q;
    if (local_q >= group) return;
    if (qh >= q_heads) return;

    // 输出指针（内部用 float 做累加，数值更稳定）
    const int o_base = ((b * tgt_len + t) * q_heads + qh) * head_dim;
    float* out = o_float + o_base;

    // 只考虑有效的 source 范围（causal 时最多看到 t）
    const int effective_src = is_causal ? min(src_len, t + 1) : src_len;
    if (effective_src <= 0) {
      for (int d = 0; d < head_dim; ++d) out[d] = 0.0f;
      return;
    }

    // 把 q 先读到寄存器里（避免每次算 dot 都去读 q）
    float q_local[MAX_HEAD_DIM];
    const int q_base = ((b * tgt_len + t) * q_heads + qh) * head_dim;
    for (int d = 0; d < head_dim; ++d) q_local[d] = to_float<T>(q[q_base + d]);

    const float inv_sqrt_d = 1.0f / sqrtf(static_cast<float>(head_dim));

    // -----------------------------
    // Pass 1: 求最大值 m
    // -----------------------------
    float m = -INFINITY;
    for (int tile_start = 0; tile_start < effective_src; tile_start += TILE_SIZE) {
      const int tile_end = min(tile_start + TILE_SIZE, effective_src);
      const int tile_size = tile_end - tile_start;

      // 协作加载 K 到共享内存（所有线程的 kvh/b 都相同，因此共享内存数据一致）
      for (int i = threadIdx.x; i < tile_size * head_dim; i += blockDim.x) {
        const int local_s = i / head_dim;
        const int d = i % head_dim;
        const int global_s = tile_start + local_s;
        const int k_idx = ((b * src_len + global_s) * kv_heads + kvh) * head_dim + d;
        s_k[local_s][d] = to_float<T>(k[k_idx]);
      }
      __syncthreads();

      for (int local_s = 0; local_s < tile_size; ++local_s) {
        float dot = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
          dot = fmaf(q_local[d], s_k[local_s][d], dot);
        }
        const float score = dot * inv_sqrt_d;
        m = fmaxf(m, score);
      }
      __syncthreads();
    }

    // -----------------------------
    // Pass 2: 求 sum 与输出
    // -----------------------------
    float out_local[MAX_HEAD_DIM];
    for (int d = 0; d < head_dim; ++d) out_local[d] = 0.0f;
    float sum = 0.0f;

    for (int tile_start = 0; tile_start < effective_src; tile_start += TILE_SIZE) {
      const int tile_end = min(tile_start + TILE_SIZE, effective_src);
      const int tile_size = tile_end - tile_start;

      // 协作加载 K/V 到共享内存
      for (int i = threadIdx.x; i < tile_size * head_dim; i += blockDim.x) {
        const int local_s = i / head_dim;
        const int d = i % head_dim;
        const int global_s = tile_start + local_s;
        const int k_idx = ((b * src_len + global_s) * kv_heads + kvh) * head_dim + d;
        const int v_idx = ((b * src_len + global_s) * kv_heads + kvh) * head_dim + d;
        s_k[local_s][d] = to_float<T>(k[k_idx]);
        s_v[local_s][d] = to_float<T>(v[v_idx]);
      }
      __syncthreads();

      for (int local_s = 0; local_s < tile_size; ++local_s) {
        float dot = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
          dot = fmaf(q_local[d], s_k[local_s][d], dot);
        }
        const float score = dot * inv_sqrt_d;
        const float w = expf(score - m);
        sum += w;

        for (int d = 0; d < head_dim; ++d) {
          out_local[d] = fmaf(w, s_v[local_s][d], out_local[d]);
        }
      }
      __syncthreads();
    }

    if (sum > 0.0f) {
      const float inv_sum = 1.0f / sum;
      for (int d = 0; d < head_dim; ++d) out[d] = out_local[d] * inv_sum;
    } else {
      for (int d = 0; d < head_dim; ++d) out[d] = 0.0f;
    }
    return;
  }

  // ------------------------------------------------------------
  // fallback 模式：每个线程负责一个 (b, t, qh) 输出向量
  // ------------------------------------------------------------
  const int vec_id = blockIdx.x * blockDim.x + threadIdx.x;
  const int num_vec = batch_size * tgt_len * q_heads;
  if (vec_id >= num_vec) return;

  const int bt = vec_id / q_heads;
  const int qh = vec_id - bt * q_heads;
  const int b = bt / tgt_len;
  const int t = bt - b * tgt_len;

  int kvh = 0;
  if (kv_heads > 0) {
    if (q_heads % kv_heads == 0) {
      const int group = q_heads / kv_heads;
      kvh = qh / group;
    } else {
      kvh = qh % kv_heads;
    }
  }

  const int o_base = ((b * tgt_len + t) * q_heads + qh) * head_dim;
  float* out = o_float + o_base;

  const int effective_src = is_causal ? min(src_len, t + 1) : src_len;
  if (effective_src <= 0) {
    for (int d = 0; d < head_dim; ++d) out[d] = 0.0f;
    return;
  }

  const float inv_sqrt_d = 1.0f / sqrtf(static_cast<float>(head_dim));
  const int q_base = ((b * tgt_len + t) * q_heads + qh) * head_dim;

  // Pass 1: m = max(score)
  float m = -INFINITY;
  for (int s = 0; s < effective_src; ++s) {
    float dot = 0.0f;
    const int k_base = ((b * src_len + s) * kv_heads + kvh) * head_dim;
    for (int d = 0; d < head_dim; ++d) {
      dot = fmaf(to_float<T>(q[q_base + d]), to_float<T>(k[k_base + d]), dot);
    }
    m = fmaxf(m, dot * inv_sqrt_d);
  }

  // Pass 2: sum 与 out
  float sum = 0.0f;
  constexpr int kFallbackMaxHeadDim = 256;
  if (head_dim <= kFallbackMaxHeadDim) {
    float out_local[kFallbackMaxHeadDim];
    for (int d = 0; d < head_dim; ++d) out_local[d] = 0.0f;

    for (int s = 0; s < effective_src; ++s) {
      float dot = 0.0f;
      const int k_base = ((b * src_len + s) * kv_heads + kvh) * head_dim;
      for (int d = 0; d < head_dim; ++d) {
        dot = fmaf(to_float<T>(q[q_base + d]), to_float<T>(k[k_base + d]), dot);
      }
      const float score = dot * inv_sqrt_d;
      const float w = expf(score - m);
      sum += w;

      const int v_base = ((b * src_len + s) * kv_heads + kvh) * head_dim;
      for (int d = 0; d < head_dim; ++d) {
        out_local[d] = fmaf(w, to_float<T>(v[v_base + d]), out_local[d]);
      }
    }

    if (sum > 0.0f) {
      const float inv_sum = 1.0f / sum;
      for (int d = 0; d < head_dim; ++d) out[d] = out_local[d] * inv_sum;
    } else {
      for (int d = 0; d < head_dim; ++d) out[d] = 0.0f;
    }
  } else {
    // head_dim 太大时不再用 out_local（避免线程栈/寄存器压力过大），直接在全局内存上累加后归一化。
    for (int d = 0; d < head_dim; ++d) out[d] = 0.0f;

    for (int s = 0; s < effective_src; ++s) {
      float dot = 0.0f;
      const int k_base = ((b * src_len + s) * kv_heads + kvh) * head_dim;
      for (int d = 0; d < head_dim; ++d) {
        dot = fmaf(to_float<T>(q[q_base + d]), to_float<T>(k[k_base + d]), dot);
      }
      const float score = dot * inv_sqrt_d;
      const float w = expf(score - m);
      sum += w;

      const int v_base = ((b * src_len + s) * kv_heads + kvh) * head_dim;
      for (int d = 0; d < head_dim; ++d) {
        out[d] = fmaf(w, to_float<T>(v[v_base + d]), out[d]);
      }
    }

    if (sum > 0.0f) {
      const float inv_sum = 1.0f / sum;
      for (int d = 0; d < head_dim; ++d) out[d] *= inv_sum;
    } else {
      for (int d = 0; d < head_dim; ++d) out[d] = 0.0f;
    }
  }
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
template <typename T>
void flashAttention(const std::vector<T>& h_q, const std::vector<T>& h_k,
                    const std::vector<T>& h_v, std::vector<T>& h_o,
                    int batch_size, int target_seq_len, int src_seq_len, 
                    int query_heads, int kv_heads, int head_dim, bool is_causal) {       
  // TODO: Implement the flash attention function

  // 调整输出向量 h_o 到期望大小
  const size_t out_numel =
      static_cast<size_t>(batch_size) * target_seq_len * query_heads * head_dim;
  h_o.resize(out_numel);

  // 一些简单边界情况
  if (out_numel == 0) return;
  if (batch_size <= 0 || target_seq_len <= 0 || query_heads <= 0 || head_dim <= 0 ||
      src_seq_len <= 0 || kv_heads <= 0) {
    // 维度不合法或没有 K/V：直接输出全 0（避免 cudaMalloc(0) 等无效调用）
    for (size_t i = 0; i < out_numel; ++i) h_o[i] = from_float<T>(0.0f);
    return;
  }

  // 申请设备端缓冲区
  const size_t q_numel =
      static_cast<size_t>(batch_size) * target_seq_len * query_heads * head_dim;
  const size_t k_numel =
      static_cast<size_t>(batch_size) * src_seq_len * kv_heads * head_dim;
  const size_t v_numel =
      static_cast<size_t>(batch_size) * src_seq_len * kv_heads * head_dim;

  T* d_q = nullptr;
  T* d_k = nullptr;
  T* d_v = nullptr;
  float* d_o_float = nullptr;  // 用 float 累加，数值更稳定

  RUNTIME_CHECK(cudaMalloc(&d_q, q_numel * sizeof(T)));
  RUNTIME_CHECK(cudaMalloc(&d_k, k_numel * sizeof(T)));
  RUNTIME_CHECK(cudaMalloc(&d_v, v_numel * sizeof(T)));
  RUNTIME_CHECK(cudaMalloc(&d_o_float, out_numel * sizeof(float)));

  // 拷贝输入到设备端
  RUNTIME_CHECK(cudaMemcpy(d_q, h_q.data(), q_numel * sizeof(T), cudaMemcpyHostToDevice));
  RUNTIME_CHECK(cudaMemcpy(d_k, h_k.data(), k_numel * sizeof(T), cudaMemcpyHostToDevice));
  RUNTIME_CHECK(cudaMemcpy(d_v, h_v.data(), v_numel * sizeof(T), cudaMemcpyHostToDevice));

  // 启动核函数：只有一份 kernel，根据形状决定走 tiled 还是 fallback 路径。
  const int num_vec = batch_size * target_seq_len * query_heads;

  bool use_tiled = false;
  int group = 0;
  if (kv_heads > 0 && (query_heads % kv_heads == 0) && head_dim > 0 && head_dim <= MAX_HEAD_DIM) {
    group = query_heads / kv_heads;
    // tiled 模式需要 thread 数 = group；这里保守限制一下，避免 block 太大。
    if (group > 0 && group <= 256) use_tiled = true;
  }

  if (use_tiled) {
    const int blocks = batch_size * target_seq_len * kv_heads;  // (b, t, kvh)
    const int threads = group;
    flash_attention_kernel<T><<<blocks, threads>>>(d_q, d_k, d_v, d_o_float,
                                                   batch_size, target_seq_len, src_seq_len,
                                                   query_heads, kv_heads, head_dim, is_causal,
                                                   true);
  } else {
    const int threads = 128;
    const int blocks = (num_vec + threads - 1) / threads;
    flash_attention_kernel<T><<<blocks, threads>>>(d_q, d_k, d_v, d_o_float,
                                                   batch_size, target_seq_len, src_seq_len,
                                                   query_heads, kv_heads, head_dim, is_causal,
                                                   false);
  }
  RUNTIME_CHECK(cudaGetLastError());
  RUNTIME_CHECK(cudaDeviceSynchronize());

  // 拷回结果，并按需要转换成 T（float 或 half）
  if constexpr (std::is_same<T, float>::value) {
    // float 情况：直接拷回即可
    RUNTIME_CHECK(cudaMemcpy(h_o.data(), d_o_float, out_numel * sizeof(float), cudaMemcpyDeviceToHost));
  } else {
    // half 情况：先拷回 float 缓冲区，再在 CPU 上转成 half（简单且可移植）
    std::vector<float> h_tmp(out_numel);
    RUNTIME_CHECK(cudaMemcpy(h_tmp.data(), d_o_float, out_numel * sizeof(float), cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < out_numel; ++i) {
      h_o[i] = from_float<T>(h_tmp[i]);
    }
  }

  // 释放资源
  RUNTIME_CHECK(cudaFree(d_q));
  RUNTIME_CHECK(cudaFree(d_k));
  RUNTIME_CHECK(cudaFree(d_v));
  RUNTIME_CHECK(cudaFree(d_o_float));
}

// *********************************************************************
// Explicit Template Instantiations (REQUIRED FOR LINKING WITH TESTER.O)
// DO NOT MODIFY THIS SECTION
// *********************************************************************
template int trace<int>(const std::vector<int>&, size_t, size_t);
template float trace<float>(const std::vector<float>&, size_t, size_t);
template void flashAttention<float>(const std::vector<float>&, const std::vector<float>&,
  const std::vector<float>&, std::vector<float>&,
  int, int, int, int, int, int, bool);
template void flashAttention<half>(const std::vector<half>&, const std::vector<half>&,
  const std::vector<half>&, std::vector<half>&,
  int, int, int, int, int, int, bool);
