#include <vector>
#include <cuda_fp16.h>
#include <type_traits>

#include "../tester/utils.h"

// 使用英伟达（NVIDIA）平台
//trace_kernel / trace：在 GPU 上计算矩阵的迹（主对角线元素之和）
//flash_attn_kernel / flashAttention：在 GPU 上计算缩放点积注意力

template <typename T>
__global__ void trace_kernel(const T* in, size_t r, size_t c, size_t len, T* out) {
  // 每个线程处理若干对角线元素，最终通过 atomicAdd 累加到 out
  size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
  size_t stride = blockDim.x * gridDim.x;
  for (size_t i = tid; i < len; i += stride) {
    // 行主序下主对角线第 i 个元素的索引：i * cols + i
    size_t idx = i * c + i;
    if constexpr (std::is_same<T, float>::value) {
      atomicAdd(out, in[idx]);
    } else if constexpr (std::is_same<T, int>::value) {
      atomicAdd(out, in[idx]);
    }
  }
}

template <typename T>
__global__ void flash_attn_kernel(const T* q, const T* k, const T* v, T* o,
                                  int B, int Tt, int Ts, int Hq, int Hkv, int D, bool causal) {
  // 该 Kernel 的网格/块映射关系：
  // grid.x = batch，grid.y = query_heads，grid.z = target_seq_len
  // block.x 遍历 head_dim（D），线程按维度并行写输出
  // 数值稳定：先计算最大分数 smax，再做 softmax 归一化
  // GQA：通过 hq → hk 的映射，将多个 query head 共享到对应的 kv head
  auto fval = [] __device__(T x) -> float {
    if constexpr (std::is_same<T, float>::value) return x;
    else return __half2float(x);
  };
  auto cvt = [] __device__(float x) -> T {
    if constexpr (std::is_same<T, float>::value) return x;
    else return __float2half_rn(x);
  };
  auto off_q = [=] __device__(int b, int t, int h, int d) -> size_t {
    return (((size_t)b * Tt + t) * Hq + h) * D + d;
  };
  auto off_k = [=] __device__(int b, int s, int h, int d) -> size_t {
    return (((size_t)b * Ts + s) * Hkv + h) * D + d;
  };
  auto off_o = [=] __device__(int b, int t, int h, int d) -> size_t {
    return (((size_t)b * Tt + t) * Hq + h) * D + d;
  };
  int b = blockIdx.x;
  int hq = blockIdx.y;
  int t = blockIdx.z;
  int hk = (int)((1ll * hq * Hkv) / Hq);
  float scale = rsqrtf((float)D);
  __shared__ float smax;
  __shared__ float snorm;
  if (threadIdx.x == 0) {
    // 第一步：遍历所有 src 位置，求最大分数（数值稳定）
    float m = -1e30f;
    for (int s = 0; s < Ts; ++s) {
      if (causal && s > t) continue;
      float dot = 0.0f;
      for (int d = 0; d < D; ++d) {
        float qv = fval(q[off_q(b, t, hq, d)]);
        float kv = fval(k[off_k(b, s, hk, d)]);
        dot += qv * kv;
      }
      float sc = dot * scale;
      if (sc > m) m = sc;
    }
    smax = m;
    // 第二步：基于 smax 计算 softmax 的分母（归一化系数）
    float sumexp = 0.0f;
    for (int s = 0; s < Ts; ++s) {
      if (causal && s > t) continue;
      float dot = 0.0f;
      for (int d = 0; d < D; ++d) {
        float qv = fval(q[off_q(b, t, hq, d)]);
        float kv = fval(k[off_k(b, s, hk, d)]);
        dot += qv * kv;
      }
      sumexp += __expf(dot * scale - smax);
    }
    snorm = sumexp > 0 ? sumexp : 1.0f;
  }
  __syncthreads();
  // 第三步：按维度并行，累加 softmax 权重对应的 V 向量得到输出
  for (int d = threadIdx.x; d < D; d += blockDim.x) {
    float acc = 0.0f;
    for (int s = 0; s < Ts; ++s) {
      if (causal && s > t) continue;
      float dot = 0.0f;
      for (int dd = 0; dd < D; ++dd) {
        float qv = fval(q[off_q(b, t, hq, dd)]);
        float kv = fval(k[off_k(b, s, hk, dd)]);
        dot += qv * kv;
      }
      float w = __expf(dot * scale - smax) / snorm;
      float vv = fval(v[off_k(b, s, hk, d)]);
      acc += w * vv;
    }
    o[off_o(b, t, hq, d)] = cvt(acc);
  }
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
  size_t n = rows * cols;
  if (rows == 0 || cols == 0 || h_input.size() < n) return T(0);
  // 为简明起见，采用一次拷贝到设备并在 GPU 上累加的方式
  T* d_in = nullptr;
  T* d_out = nullptr;
  RUNTIME_CHECK(cudaMalloc(&d_in, n * sizeof(T)));
  RUNTIME_CHECK(cudaMalloc(&d_out, sizeof(T)));
  RUNTIME_CHECK(cudaMemcpy(d_in, h_input.data(), n * sizeof(T), cudaMemcpyHostToDevice));
  RUNTIME_CHECK(cudaMemset(d_out, 0, sizeof(T)));
  size_t diag_len = rows < cols ? rows : cols;
  dim3 blk(256);
  dim3 grd((unsigned)((diag_len + blk.x - 1) / blk.x));
  trace_kernel<T><<<grd, blk>>>(d_in, rows, cols, diag_len, d_out);
  RUNTIME_CHECK(cudaDeviceSynchronize());
  T h_out{};
  RUNTIME_CHECK(cudaMemcpy(&h_out, d_out, sizeof(T), cudaMemcpyDeviceToHost));
  RUNTIME_CHECK(cudaFree(d_in));
  RUNTIME_CHECK(cudaFree(d_out));
  return h_out;
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
  size_t qsz = (size_t)batch_size * target_seq_len * query_heads * head_dim;
  size_t kvsz = (size_t)batch_size * src_seq_len * kv_heads * head_dim;
  if (h_q.size() < qsz || h_k.size() < kvsz || h_v.size() < kvsz) return;
  // 输出向量预分配；随后进行 H2D 拷贝、Kernel 计算、D2H 回传
  h_o.resize(qsz);
  T* d_q = nullptr; T* d_k = nullptr; T* d_v = nullptr; T* d_o = nullptr;
  RUNTIME_CHECK(cudaMalloc(&d_q, qsz * sizeof(T)));
  RUNTIME_CHECK(cudaMalloc(&d_k, kvsz * sizeof(T)));
  RUNTIME_CHECK(cudaMalloc(&d_v, kvsz * sizeof(T)));
  RUNTIME_CHECK(cudaMalloc(&d_o, qsz * sizeof(T)));
  RUNTIME_CHECK(cudaMemcpy(d_q, h_q.data(), qsz * sizeof(T), cudaMemcpyHostToDevice));
  RUNTIME_CHECK(cudaMemcpy(d_k, h_k.data(), kvsz * sizeof(T), cudaMemcpyHostToDevice));
  RUNTIME_CHECK(cudaMemcpy(d_v, h_v.data(), kvsz * sizeof(T), cudaMemcpyHostToDevice));
  RUNTIME_CHECK(cudaMemset(d_o, 0, qsz * sizeof(T)));
  dim3 blk(min(256, head_dim));
  dim3 grd((unsigned)batch_size, (unsigned)query_heads, (unsigned)target_seq_len);
  flash_attn_kernel<T><<<grd, blk>>>(d_q, d_k, d_v, d_o,
                                     batch_size, target_seq_len, src_seq_len,
                                     query_heads, kv_heads, head_dim, is_causal);
  RUNTIME_CHECK(cudaDeviceSynchronize());
  RUNTIME_CHECK(cudaMemcpy(h_o.data(), d_o, qsz * sizeof(T), cudaMemcpyDeviceToHost));
  RUNTIME_CHECK(cudaFree(d_q));
  RUNTIME_CHECK(cudaFree(d_k));
  RUNTIME_CHECK(cudaFree(d_v));
  RUNTIME_CHECK(cudaFree(d_o));
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
