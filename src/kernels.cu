#include <vector>
#include <algorithm>
#include <cmath>
#include <limits>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "../tester/utils.h"

namespace {
template <typename T>
inline float to_float(T value) {
  return static_cast<float>(value);
}

template <>
inline float to_float<half>(half value) {
  return __half2float(value);
}

template <typename T>
inline T from_float(float value) {
  return static_cast<T>(value);
}

template <>
inline half from_float<half>(float value) {
  return __float2half_rn(value);
}
}  // namespace

template <typename T>
__device__ inline float to_float_device(T value) {
  return static_cast<float>(value);
}

template <>
__device__ inline float to_float_device<half>(half value) {
  return __half2float(value);
}

template <typename T>
__device__ inline T from_float_device(float value) {
  return static_cast<T>(value);
}

template <>
__device__ inline half from_float_device<half>(float value) {
  return __float2half_rn(value);
}

template <typename T>
__global__ void trace_kernel(const T* input, size_t rows, size_t cols, size_t diag, T* output) {
  size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= diag) {
    return;
  }
  // Each thread handles one diagonal element and atomically accumulates.
  size_t input_idx = idx * cols + idx;
  atomicAdd(output, input[input_idx]);
}

template <typename T>
__global__ void flash_attention_kernel(const T* q, const T* k, const T* v, T* o,
                                       int batch_size, int target_seq_len, int src_seq_len,
                                       int query_heads, int kv_heads, int head_dim,
                                       int group_size, float scale, bool is_causal) {
  if (threadIdx.x != 0) {
    return;
  }

  int idx = blockIdx.x;
  int total = batch_size * target_seq_len * query_heads;
  if (idx >= total) {
    return;
  }

  int qh = idx % query_heads;
  int tmp = idx / query_heads;
  int t = tmp % target_seq_len;
  int b = tmp / target_seq_len;

  int kvh = qh / group_size;
  if (kvh >= kv_heads) {
    kvh = kv_heads - 1;
  }
  // Softmax over source positions with optional causal masking.

  size_t q_stride_bt = static_cast<size_t>(query_heads) * head_dim;
  size_t k_stride_bs = static_cast<size_t>(kv_heads) * head_dim;

  float max_score = -INFINITY;
  for (int s = 0; s < src_seq_len; ++s) {
    if (is_causal && s > t) {
      continue;
    }

    float dot = 0.0f;
    size_t q_base = (static_cast<size_t>(b) * target_seq_len + t) * q_stride_bt +
                    static_cast<size_t>(qh) * head_dim;
    size_t k_base = (static_cast<size_t>(b) * src_seq_len + s) * k_stride_bs +
                    static_cast<size_t>(kvh) * head_dim;
    for (int d = 0; d < head_dim; ++d) {
      dot += to_float_device(q[q_base + d]) * to_float_device(k[k_base + d]);
    }

    float score = dot * scale;
    if (score > max_score) {
      max_score = score;
    }
  }

  float denom = 0.0f;
  for (int s = 0; s < src_seq_len; ++s) {
    if (is_causal && s > t) {
      continue;
    }

    float dot = 0.0f;
    size_t q_base = (static_cast<size_t>(b) * target_seq_len + t) * q_stride_bt +
                    static_cast<size_t>(qh) * head_dim;
    size_t k_base = (static_cast<size_t>(b) * src_seq_len + s) * k_stride_bs +
                    static_cast<size_t>(kvh) * head_dim;
    for (int d = 0; d < head_dim; ++d) {
      dot += to_float_device(q[q_base + d]) * to_float_device(k[k_base + d]);
    }

    float score = dot * scale;
    denom += expf(score - max_score);
  }

  if (denom == 0.0f) {
    denom = 1.0f;
  }

  size_t o_base = (static_cast<size_t>(b) * target_seq_len + t) * q_stride_bt +
                  static_cast<size_t>(qh) * head_dim;
  for (int d = 0; d < head_dim; ++d) {
    // Weighted sum over V using normalized attention weights.
    float acc = 0.0f;
    for (int s = 0; s < src_seq_len; ++s) {
      if (is_causal && s > t) {
        continue;
      }

      float dot = 0.0f;
      size_t q_base = (static_cast<size_t>(b) * target_seq_len + t) * q_stride_bt +
                      static_cast<size_t>(qh) * head_dim;
      size_t k_base = (static_cast<size_t>(b) * src_seq_len + s) * k_stride_bs +
                      static_cast<size_t>(kvh) * head_dim;
      for (int dk = 0; dk < head_dim; ++dk) {
        dot += to_float_device(q[q_base + dk]) * to_float_device(k[k_base + dk]);
      }

      float score = dot * scale;
      float weight = expf(score - max_score) / denom;
      size_t v_base = (static_cast<size_t>(b) * src_seq_len + s) * k_stride_bs +
                      static_cast<size_t>(kvh) * head_dim;
      acc += weight * to_float_device(v[v_base + d]);
    }
    o[o_base + d] = from_float_device<T>(acc);
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
  if (rows == 0 || cols == 0 || h_input.empty()) {
    return T(0);
  }

  const size_t max_diag = std::min(rows, cols);
  const size_t max_by_size = (cols == 0 || h_input.empty())
                                 ? 0
                                 : (h_input.size() - 1) / (cols + 1) + 1;
  const size_t diag = std::min(max_diag, max_by_size);
  if (diag == 0) {
    return T(0);
  }

  T* d_input = nullptr;
  T* d_output = nullptr;
  RUNTIME_CHECK(cudaMalloc(&d_input, h_input.size() * sizeof(T)));
  RUNTIME_CHECK(cudaMalloc(&d_output, sizeof(T)));
  RUNTIME_CHECK(cudaMemcpy(d_input, h_input.data(), h_input.size() * sizeof(T), cudaMemcpyHostToDevice));
  RUNTIME_CHECK(cudaMemset(d_output, 0, sizeof(T)));

  const int threads = 256;
  const int blocks = static_cast<int>((diag + threads - 1) / threads);
  trace_kernel<<<blocks, threads>>>(d_input, rows, cols, diag, d_output);
  RUNTIME_CHECK(cudaDeviceSynchronize());

  T sum = T(0);
  RUNTIME_CHECK(cudaMemcpy(&sum, d_output, sizeof(T), cudaMemcpyDeviceToHost));
  RUNTIME_CHECK(cudaFree(d_input));
  RUNTIME_CHECK(cudaFree(d_output));
  return sum;
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
  if (batch_size <= 0 || target_seq_len <= 0 || src_seq_len <= 0 ||
      query_heads <= 0 || kv_heads <= 0 || head_dim <= 0) {
    return;
  }

  const int group_size = std::max(1, query_heads / kv_heads);
  const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

  const size_t q_count = static_cast<size_t>(batch_size) * target_seq_len * query_heads * head_dim;
  const size_t k_count = static_cast<size_t>(batch_size) * src_seq_len * kv_heads * head_dim;
  const size_t o_count = static_cast<size_t>(batch_size) * target_seq_len * query_heads * head_dim;

  if (h_o.size() < o_count) {
    h_o.resize(o_count);
  }

  T* d_q = nullptr;
  T* d_k = nullptr;
  T* d_v = nullptr;
  T* d_o = nullptr;
  RUNTIME_CHECK(cudaMalloc(&d_q, q_count * sizeof(T)));
  RUNTIME_CHECK(cudaMalloc(&d_k, k_count * sizeof(T)));
  RUNTIME_CHECK(cudaMalloc(&d_v, k_count * sizeof(T)));
  RUNTIME_CHECK(cudaMalloc(&d_o, o_count * sizeof(T)));
  RUNTIME_CHECK(cudaMemcpy(d_q, h_q.data(), q_count * sizeof(T), cudaMemcpyHostToDevice));
  RUNTIME_CHECK(cudaMemcpy(d_k, h_k.data(), k_count * sizeof(T), cudaMemcpyHostToDevice));
  RUNTIME_CHECK(cudaMemcpy(d_v, h_v.data(), k_count * sizeof(T), cudaMemcpyHostToDevice));

  const int total = batch_size * target_seq_len * query_heads;
  const int threads = 1;
  const int blocks = total;
  flash_attention_kernel<<<blocks, threads>>>(d_q, d_k, d_v, d_o,
                                              batch_size, target_seq_len, src_seq_len,
                                              query_heads, kv_heads, head_dim,
                                              group_size, scale, is_causal);
  RUNTIME_CHECK(cudaDeviceSynchronize());

  RUNTIME_CHECK(cudaMemcpy(h_o.data(), d_o, o_count * sizeof(T), cudaMemcpyDeviceToHost));
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
