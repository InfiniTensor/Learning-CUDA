#include <vector>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cmath>

#include "../tester/utils.h"

/**
 * @brief CUDA kernel to compute the trace of a matrix.
 * 
 * Each thread processes one diagonal element and uses atomicAdd to accumulate the sum.
 * For a matrix stored in row-major format, the diagonal element at position (i, i)
 * is located at index i * cols + i in the flattened array.
 * 
 * @tparam T The numeric type of matrix elements (e.g., float, int).
 * @param d_input Device pointer to the flattened input matrix.
 * @param cols Number of columns in the matrix (used to calculate diagonal element index).
 * @param n_diagonal Number of diagonal elements to process (min(rows, cols)).
 * @param d_result Device pointer to store the result (single element).
 */
template <typename T>
__global__ void trace_kernel(const T* d_input, size_t cols, size_t n_diagonal, T* d_result) {
  // Get the thread index
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  
  // Each thread processes one diagonal element
  if (idx < n_diagonal) {
    // Calculate the index of diagonal element (i, i) in row-major format
    // For row i, column i: index = i * cols + i
    size_t diagonal_idx = idx * cols + idx;
    
    // Atomically add the diagonal element to the result
    atomicAdd(d_result, d_input[diagonal_idx]);
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
  // Calculate the number of diagonal elements (min of rows and cols)
  size_t n_diagonal = (rows < cols) ? rows : cols;
  
  // Handle edge case: empty matrix
  if (n_diagonal == 0) {
    return T(0);
  }
  
  // Allocate device memory for input matrix
  T* d_input;
  size_t input_size = rows * cols * sizeof(T);
  RUNTIME_CHECK(cudaMalloc(&d_input, input_size));
  
  // Copy input data from host to device
  RUNTIME_CHECK(cudaMemcpy(d_input, h_input.data(), input_size, cudaMemcpyHostToDevice));
  
  // Allocate device memory for result and initialize to zero
  T* d_result;
  RUNTIME_CHECK(cudaMalloc(&d_result, sizeof(T)));
  RUNTIME_CHECK(cudaMemset(d_result, 0, sizeof(T)));
  
  // Configure kernel launch parameters
  // Use 256 threads per block (common choice for good performance)
  const size_t threads_per_block = 256;
  const size_t blocks_per_grid = (n_diagonal + threads_per_block - 1) / threads_per_block;
  
  // Launch the CUDA kernel
  trace_kernel<T><<<blocks_per_grid, threads_per_block>>>(
    d_input, cols, n_diagonal, d_result
  );
  
  // Check for kernel launch errors
  RUNTIME_CHECK(cudaGetLastError());
  
  // Wait for kernel to complete
  RUNTIME_CHECK(cudaDeviceSynchronize());
  
  // Copy result back from device to host
  T h_result;
  RUNTIME_CHECK(cudaMemcpy(&h_result, d_result, sizeof(T), cudaMemcpyDeviceToHost));
  
  // Free device memory
  RUNTIME_CHECK(cudaFree(d_input));
  RUNTIME_CHECK(cudaFree(d_result));
  
  return h_result;
}

// Helper function to convert half to float
__device__ __forceinline__ float half_to_float(half h) {
  return __half2float(h);
}

__device__ __forceinline__ float half_to_float(float f) {
  return f;
}

// Helper function to convert float to half
template<typename T>
__device__ __forceinline__ T float_to_half(float f);

template<>
__device__ __forceinline__ half float_to_half<half>(float f) {
  return __float2half(f);
}

template<>
__device__ __forceinline__ float float_to_half<float>(float f) {
  return f;
}

/**
 * @brief CUDA kernel for Flash Attention
 * 
 * Each thread block processes one (batch, tgt_pos, q_head) combination.
 * Threads within the block cooperate to compute attention scores and accumulate output.
 * 
 * @tparam T Data type (float or half)
 */
template <typename T>
__global__ void flash_attention_kernel(
  const T* __restrict__ d_q,
  const T* __restrict__ d_k,
  const T* __restrict__ d_v,
  T* __restrict__ d_o,
  int batch_size, int target_seq_len, int src_seq_len,
  int query_heads, int kv_heads, int head_dim,
  bool is_causal, float scale) {

  // 使用 long long 防止大尺寸 tensor 索引溢出
  long long batch_idx = blockIdx.z;
  long long tgt_pos = blockIdx.y;
  long long q_head = blockIdx.x;

  if (batch_idx >= batch_size || tgt_pos >= target_seq_len || q_head >= query_heads) return;

  // GQA: q_head / group_size
  int group_size = query_heads / kv_heads;
  int kv_head = (int)q_head / group_size;

  extern __shared__ float s_mem[];
  float* s_q = s_mem;
  float* s_output = s_mem + head_dim;
  float* s_reduce = s_mem + head_dim * 2;
  float* s_max_score = s_mem + head_dim * 2 + blockDim.x;
  float* s_sum_exp = s_mem + head_dim * 2 + blockDim.x + 1;
  
  // 将共享变量移出循环
  __shared__ float s_exp_val;

  int tid = threadIdx.x;
  int num_threads = blockDim.x;

  // 初始化
  for (int d = tid; d < head_dim; d += num_threads) {
      s_output[d] = 0.0f;
  }
  if (tid == 0) {
      s_max_score[0] = -1e38f; 
      s_sum_exp[0] = 0.0f;
  }
  __syncthreads();

  // 加载 Query 到共享内存 (使用 long long 偏移)
  long long q_base_offset = ((batch_idx * target_seq_len + tgt_pos) * query_heads + q_head) * head_dim;
  for (int d = tid; d < head_dim; d += num_threads) {
      s_q[d] = half_to_float(d_q[q_base_offset + d]);
  }
  __syncthreads();

  // Pass 1: 计算 Max Score
  for (int src_pos = 0; src_pos < src_seq_len; src_pos++) {
      if (is_causal && src_pos > tgt_pos) continue;

      long long kv_base_offset = ((batch_idx * src_seq_len + src_pos) * kv_heads + kv_head) * head_dim;
      
      float local_dot = 0.0f;
      for (int d = tid; d < head_dim; d += num_threads) {
          local_dot += s_q[d] * half_to_float(d_k[kv_base_offset + d]);
      }

      s_reduce[tid] = local_dot;
      __syncthreads();
      // 规约求和
      for (int stride = num_threads / 2; stride > 0; stride >>= 1) {
          if (tid < stride) s_reduce[tid] += s_reduce[tid + stride];
          __syncthreads();
      }

      if (tid == 0) {
          float score = s_reduce[0] * scale;
          if (score > s_max_score[0]) s_max_score[0] = score;
      }
      __syncthreads();
  }

  // Pass 2: 计算 Softmax 和累加 Value
  for (int src_pos = 0; src_pos < src_seq_len; src_pos++) {
      if (is_causal && src_pos > tgt_pos) continue;

      long long kv_base_offset = ((batch_idx * src_seq_len + src_pos) * kv_heads + kv_head) * head_dim;

      float local_dot = 0.0f;
      for (int d = tid; d < head_dim; d += num_threads) {
          local_dot += s_q[d] * half_to_float(d_k[kv_base_offset + d]);
      }

      s_reduce[tid] = local_dot;
      __syncthreads();
      for (int stride = num_threads / 2; stride > 0; stride >>= 1) {
          if (tid < stride) s_reduce[tid] += s_reduce[tid + stride];
          __syncthreads();
      }

      if (tid == 0) {
          float score = s_reduce[0] * scale;
          s_exp_val = expf(score - s_max_score[0]);
          s_sum_exp[0] += s_exp_val;
      }
      __syncthreads();

      float exp_score = s_exp_val; // 读取到寄存器
      for (int d = tid; d < head_dim; d += num_threads) {
          s_output[d] += exp_score * half_to_float(d_v[kv_base_offset + d]);
      }
      __syncthreads();
  }

  // 写回结果 (使用 long long 偏移)
  long long o_base_offset = ((batch_idx * target_seq_len + tgt_pos) * query_heads + q_head) * head_dim;
  float final_sum_exp = s_sum_exp[0];
  float inv_sum = (final_sum_exp > 0.0f) ? (1.0f / final_sum_exp) : 0.0f;

  for (int d = tid; d < head_dim; d += num_threads) {
      d_o[o_base_offset + d] = float_to_half<T>(s_output[d] * inv_sum);
  }
}

/**
 * @brief Computes flash attention for given query, key, and value tensors.
 * 
 * @tparam T Data type (float or half) for input/output tensors
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
  // Resize output vector
  h_o.resize(batch_size * target_seq_len * query_heads * head_dim);
  
  // Calculate sizes
  size_t q_size = batch_size * target_seq_len * query_heads * head_dim * sizeof(T);
  size_t k_size = batch_size * src_seq_len * kv_heads * head_dim * sizeof(T);
  size_t v_size = batch_size * src_seq_len * kv_heads * head_dim * sizeof(T);
  size_t o_size = batch_size * target_seq_len * query_heads * head_dim * sizeof(T);
  
  // Allocate device memory
  T* d_q;
  T* d_k;
  T* d_v;
  T* d_o;
  
  RUNTIME_CHECK(cudaMalloc(&d_q, q_size));
  RUNTIME_CHECK(cudaMalloc(&d_k, k_size));
  RUNTIME_CHECK(cudaMalloc(&d_v, v_size));
  RUNTIME_CHECK(cudaMalloc(&d_o, o_size));
  
  // Copy data from host to device
  RUNTIME_CHECK(cudaMemcpy(d_q, h_q.data(), q_size, cudaMemcpyHostToDevice));
  RUNTIME_CHECK(cudaMemcpy(d_k, h_k.data(), k_size, cudaMemcpyHostToDevice));
  RUNTIME_CHECK(cudaMemcpy(d_v, h_v.data(), v_size, cudaMemcpyHostToDevice));
  
  // Calculate scale factor (1 / sqrt(head_dim))
  float scale = 1.0f / sqrtf(static_cast<float>(head_dim));
  
  // Configure kernel launch parameters
  // Each thread block processes one (batch, tgt_pos, q_head) combination
  dim3 grid_size(query_heads, target_seq_len, batch_size);
  int threads_per_block = 128;  // Threads per block
  // Shared memory: query (head_dim) + output (head_dim) + reduce buffer (256) + max_score (1) + sum_exp (1) + exp_score (1) + has_valid_pos (1 bool, aligned to float)
  size_t shared_mem_size = (head_dim * 2 + threads_per_block + 4) * sizeof(float);
  
  // Launch kernel
  flash_attention_kernel<T><<<grid_size, threads_per_block, shared_mem_size>>>(
      d_q, d_k, d_v, d_o,
      batch_size, target_seq_len, src_seq_len,
      query_heads, kv_heads, head_dim,
      is_causal, scale);
  
  // Check for kernel launch errors
  RUNTIME_CHECK(cudaGetLastError());
  
  // Wait for kernel to complete
  RUNTIME_CHECK(cudaDeviceSynchronize());
  
  // Copy result back from device to host
  RUNTIME_CHECK(cudaMemcpy(h_o.data(), d_o, o_size, cudaMemcpyDeviceToHost));
  
  // Free device memory
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
