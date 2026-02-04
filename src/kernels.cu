#include <vector>
#include <cuda_fp16.h>

#include "../tester/utils.h"

// ======================= Trace Kernel =======================

/**
 * @brief CUDA kernel to compute the trace of a matrix using parallel reduction.
 * 
 * Each thread handles multiple diagonal elements if needed, then performs
 * a tree-based reduction in shared memory to compute the final sum.
 */
template <typename T>
__global__ void traceKernel(const T* input, T* result, size_t diag_len, size_t cols) {
  extern __shared__ char shared_mem[];
  T* sdata = reinterpret_cast<T*>(shared_mem);

  unsigned int tid = threadIdx.x;
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // Each thread sums its assigned diagonal elements
  T local_sum = T(0);
  for (size_t i = idx; i < diag_len; i += blockDim.x * gridDim.x) {
    // Diagonal element at (i, i) in row-major layout is at index i * cols + i
    local_sum += input[i * cols + i];
  }

  sdata[tid] = local_sum;
  __syncthreads();

  // Tree-based reduction in shared memory
  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }

  // Thread 0 writes the block's partial sum to global memory
  if (tid == 0) {
    atomicAdd(result, sdata[0]);
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
  // Handle edge case: empty matrix
  if (rows == 0 || cols == 0) {
    return T(0);
  }

  // Diagonal length is the minimum of rows and cols
  size_t diag_len = (rows < cols) ? rows : cols;
  size_t input_size = rows * cols;

  // Allocate device memory
  T* d_input = nullptr;
  T* d_result = nullptr;
  RUNTIME_CHECK(cudaMalloc(&d_input, input_size * sizeof(T)));
  RUNTIME_CHECK(cudaMalloc(&d_result, sizeof(T)));

  // Copy input data to device
  RUNTIME_CHECK(cudaMemcpy(d_input, h_input.data(), input_size * sizeof(T), cudaMemcpyHostToDevice));
  
  // Initialize result to 0
  RUNTIME_CHECK(cudaMemset(d_result, 0, sizeof(T)));

  // Configure kernel launch parameters
  int blockSize = 256;
  int numBlocks = (diag_len + blockSize - 1) / blockSize;
  // Limit the number of blocks to avoid too many atomic operations
  if (numBlocks > 256) numBlocks = 256;
  size_t sharedMemSize = blockSize * sizeof(T);

  // Launch kernel
  traceKernel<T><<<numBlocks, blockSize, sharedMemSize>>>(d_input, d_result, diag_len, cols);
  RUNTIME_CHECK(cudaGetLastError());
  RUNTIME_CHECK(cudaDeviceSynchronize());

  // Copy result back to host
  T result;
  RUNTIME_CHECK(cudaMemcpy(&result, d_result, sizeof(T), cudaMemcpyDeviceToHost));

  // Free device memory
  RUNTIME_CHECK(cudaFree(d_input));
  RUNTIME_CHECK(cudaFree(d_result));

  return result;
}

// ======================= Flash Attention =======================

// Helper: convert to float for computation
template <typename T>
__device__ __forceinline__ float toFloat(T val) {
  return static_cast<float>(val);
}

template <>
__device__ __forceinline__ float toFloat<half>(half val) {
  return __half2float(val);
}

// Helper: convert from float to output type
template <typename T>
__device__ __forceinline__ T fromFloat(float val) {
  return static_cast<T>(val);
}

template <>
__device__ __forceinline__ half fromFloat<half>(float val) {
  return __float2half(val);
}

// Block sizes for Flash Attention
#define BLOCK_SIZE 256  // Threads per block

// ===================== DEBUG MODE =====================
// 设置为 1 启用调试输出，设置为 0 禁用
#define DEBUG_FLASH_ATTENTION 0

// 调试时只打印前 N 个线程的信息，避免输出过多
#define DEBUG_MAX_THREADS 5

// 调试宏：只在第一个线程打印参数信息
#define DEBUG_PRINT_PARAMS() \
  if (DEBUG_FLASH_ATTENTION && idx == 0) { \
    printf("\n========== Flash Attention Parameters ==========\n"); \
    printf("batch_size=%d, tgt_seq_len=%d, src_seq_len=%d\n", batch_size, tgt_seq_len, src_seq_len); \
    printf("query_heads=%d, kv_heads=%d, head_dim=%d\n", query_heads, kv_heads, head_dim); \
    printf("is_causal=%d, scale=%.6f\n", is_causal, scale); \
    printf("total_queries=%d\n", total_queries); \
    printf("================================================\n\n"); \
  }

// 调试宏：打印线程索引信息
#define DEBUG_PRINT_INDEX() \
  if (DEBUG_FLASH_ATTENTION && idx < DEBUG_MAX_THREADS) { \
    printf("[Thread %d] batch=%d, q_pos=%d, head=%d, kv_head=%d, kv_end=%d\n", \
           idx, batch_idx, q_pos, head_idx, kv_head_idx, kv_end); \
  }

// 调试宏：打印 Query 向量的前几个元素
#define DEBUG_PRINT_QUERY() \
  if (DEBUG_FLASH_ATTENTION && idx < DEBUG_MAX_THREADS) { \
    printf("[Thread %d] Q[0:3] = [%.4f, %.4f, %.4f, ...]\n", \
           idx, q_reg[0], head_dim > 1 ? q_reg[1] : 0.0f, head_dim > 2 ? q_reg[2] : 0.0f); \
  }

// 调试宏：打印每个 KV 位置的计算
#define DEBUG_PRINT_KV_STEP(kv_pos, score, m_new, l_new, weight) \
  if (DEBUG_FLASH_ATTENTION && idx < DEBUG_MAX_THREADS && kv_pos < 3) { \
    printf("[Thread %d] kv_pos=%d: score=%.4f, m=%.4f, l=%.4f, weight=%.4f\n", \
           idx, kv_pos, score, m_new, l_new, weight); \
  }

// 调试宏：打印最终输出
#define DEBUG_PRINT_OUTPUT() \
  if (DEBUG_FLASH_ATTENTION && idx < DEBUG_MAX_THREADS) { \
    printf("[Thread %d] O[0:3] = [%.4f, %.4f, %.4f, ...]\n", \
           idx, o_acc[0], head_dim > 1 ? o_acc[1] : 0.0f, head_dim > 2 ? o_acc[2] : 0.0f); \
  }

/**
 * @brief Flash Attention CUDA kernel - Simple version
 * 
 * Each thread handles one query position for one (batch, head) pair.
 * Uses online softmax to avoid storing the full attention matrix.
 */
template <typename T>
__global__ void flashAttentionKernel(
    const T* __restrict__ Q,
    const T* __restrict__ K,
    const T* __restrict__ V,
    T* __restrict__ O,
    int batch_size,
    int tgt_seq_len,
    int src_seq_len,
    int query_heads,
    int kv_heads,
    int head_dim,
    bool is_causal,
    float scale) {
  
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total_queries = batch_size * tgt_seq_len * query_heads;
  
  if (idx >= total_queries) return;
  
  // Decode indices
  int batch_idx = idx / (tgt_seq_len * query_heads);
  int remainder = idx % (tgt_seq_len * query_heads);
  int q_pos = remainder / query_heads;
  int head_idx = remainder % query_heads;
  
  // Determine KV head index (for GQA)
  int kv_head_idx;
  if (query_heads == kv_heads) {
    kv_head_idx = head_idx;
  } else {
    int heads_per_kv = query_heads / kv_heads;
    kv_head_idx = head_idx / heads_per_kv;
  }
  
  // Calculate strides
  long long q_batch_stride = (long long)tgt_seq_len * query_heads * head_dim;
  long long q_seq_stride = (long long)query_heads * head_dim;
  long long q_head_stride = head_dim;
  
  long long kv_batch_stride = (long long)src_seq_len * kv_heads * head_dim;
  long long kv_seq_stride = (long long)kv_heads * head_dim;
  long long kv_head_stride = head_dim;
  
  // Get pointers
  const T* q_ptr = Q + batch_idx * q_batch_stride + q_pos * q_seq_stride + head_idx * q_head_stride;
  T* o_ptr = O + batch_idx * q_batch_stride + q_pos * q_seq_stride + head_idx * q_head_stride;
  
  const T* k_base = K + batch_idx * kv_batch_stride;
  const T* v_base = V + batch_idx * kv_batch_stride;
  
  if (DEBUG_FLASH_ATTENTION && idx < DEBUG_MAX_THREADS) {
    printf("\n[Thread %d] Starting computation\n", idx);
    printf("  batch=%d, q_pos=%d, head=%d, kv_head=%d\n", batch_idx, q_pos, head_idx, kv_head_idx);
    printf("  q_ptr offset=%lld, o_ptr offset=%lld\n", 
           (long long)(q_ptr - Q), (long long)(o_ptr - O));
    printf("  k_base offset=%lld, v_base offset=%lld\n",
           (long long)(k_base - K), (long long)(v_base - V));
  }
  
  // Load query into registers
  float q_reg[256];
  for (int d = 0; d < head_dim; d++) {
    q_reg[d] = toFloat(q_ptr[d]);
  }
  
  if (DEBUG_FLASH_ATTENTION && idx < DEBUG_MAX_THREADS) {
    printf("  Q[0:3] = [%.4f, %.4f, %.4f, ...]\n", 
           q_reg[0], head_dim > 1 ? q_reg[1] : 0.0f, head_dim > 2 ? q_reg[2] : 0.0f);
  }
  
  // Initialize output accumulator and statistics
  float o_acc[256];
  for (int d = 0; d < head_dim; d++) {
    o_acc[d] = 0.0f;
  }
  float m_prev = -INFINITY;
  float l_prev = 0.0f;
  
  // Determine KV range (for causal masking)
  int kv_end = src_seq_len;
  if (is_causal) {
    // Causal mask: query at position q_pos can only attend to keys at positions 0..q_pos
    kv_end = min(src_seq_len, q_pos + 1);
  }
  
  if (DEBUG_FLASH_ATTENTION && idx < DEBUG_MAX_THREADS) {
    printf("  kv_end=%d (is_causal=%d, offset=%d)\n", kv_end, is_causal, src_seq_len - tgt_seq_len);
  }
  
  // Handle edge case
  if (kv_end <= 0) {
    if (DEBUG_FLASH_ATTENTION && idx < DEBUG_MAX_THREADS) {
      printf("  WARNING: kv_end <= 0, outputting zeros\n");
    }
    for (int d = 0; d < head_dim; d++) {
      o_ptr[d] = fromFloat<T>(0.0f);
    }
    return;
  }
  
  // Process each key-value position
  for (int kv_pos = 0; kv_pos < kv_end; kv_pos++) {
    const T* k_ptr = k_base + (long long)kv_pos * kv_seq_stride + kv_head_idx * kv_head_stride;
    const T* v_ptr = v_base + (long long)kv_pos * kv_seq_stride + kv_head_idx * kv_head_stride;
    
    // Compute attention score
    float score = 0.0f;
    for (int d = 0; d < head_dim; d++) {
      score += q_reg[d] * toFloat(k_ptr[d]);
    }
    score *= scale;
    
    // Online softmax update
    float m_new = fmaxf(m_prev, score);
    float exp_prev = expf(m_prev - m_new);
    float exp_curr = expf(score - m_new);
    float l_new = exp_prev * l_prev + exp_curr;
    
    // Debug: print first and last few positions
    if (DEBUG_FLASH_ATTENTION && idx < DEBUG_MAX_THREADS && 
        (kv_pos < 3 || kv_pos >= kv_end - 3)) {
      printf("  [kv_pos=%d] score=%.4f, m_new=%.4f, l_new=%.4f, v[0]=%.4f\n",
             kv_pos, score, m_new, l_new, toFloat(v_ptr[0]));
    }
    
    // Update output accumulator
    for (int d = 0; d < head_dim; d++) {
      o_acc[d] = o_acc[d] * exp_prev + toFloat(v_ptr[d]) * exp_curr;
    }
    
    m_prev = m_new;
    l_prev = l_new;
  }
  
  if (DEBUG_FLASH_ATTENTION && idx < DEBUG_MAX_THREADS) {
    printf("  BEFORE norm: o_acc[0]=%.4f, l_prev=%.4f\n", o_acc[0], l_prev);
  }
  
  // Final normalization and write output
  for (int d = 0; d < head_dim; d++) {
    o_ptr[d] = fromFloat<T>(o_acc[d] / l_prev);
  }
  
  if (DEBUG_FLASH_ATTENTION && idx < DEBUG_MAX_THREADS) {
    printf("  AFTER norm: o_ptr[0]=%.4f (expected %.4f)\n", 
           toFloat(o_ptr[0]), o_acc[0] / l_prev);
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
  // Calculate sizes
  size_t q_size = batch_size * target_seq_len * query_heads * head_dim;
  size_t kv_size = batch_size * src_seq_len * kv_heads * head_dim;
  size_t o_size = q_size;
  
  // Ensure output vector is properly sized
  h_o.resize(o_size);
  
  // Handle edge case: no output to produce
  if (o_size == 0) {
    return;
  }
  
  // Allocate device memory
  T *d_q, *d_k, *d_v, *d_o;
  RUNTIME_CHECK(cudaMalloc(&d_q, q_size * sizeof(T)));
  RUNTIME_CHECK(cudaMalloc(&d_k, kv_size * sizeof(T)));
  RUNTIME_CHECK(cudaMalloc(&d_v, kv_size * sizeof(T)));
  RUNTIME_CHECK(cudaMalloc(&d_o, o_size * sizeof(T)));
  
  // Copy input data to device
  RUNTIME_CHECK(cudaMemcpy(d_q, h_q.data(), q_size * sizeof(T), cudaMemcpyHostToDevice));
  RUNTIME_CHECK(cudaMemcpy(d_k, h_k.data(), kv_size * sizeof(T), cudaMemcpyHostToDevice));
  RUNTIME_CHECK(cudaMemcpy(d_v, h_v.data(), kv_size * sizeof(T), cudaMemcpyHostToDevice));
  
  // Calculate scaling factor: 1 / sqrt(head_dim)
  float scale = 1.0f / sqrtf(static_cast<float>(head_dim));
  
  // Configure kernel launch parameters
  // Each thread handles one query position
  int total_queries = batch_size * target_seq_len * query_heads;
  int num_blocks = (total_queries + BLOCK_SIZE - 1) / BLOCK_SIZE;
  
  // Launch kernel
  flashAttentionKernel<T><<<num_blocks, BLOCK_SIZE>>>(
      d_q, d_k, d_v, d_o,
      batch_size, target_seq_len, src_seq_len,
      query_heads, kv_heads, head_dim,
      is_causal, scale
  );
  
  RUNTIME_CHECK(cudaGetLastError());
  RUNTIME_CHECK(cudaDeviceSynchronize());
  
  // Copy result back to host
  RUNTIME_CHECK(cudaMemcpy(h_o.data(), d_o, o_size * sizeof(T), cudaMemcpyDeviceToHost));
  
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
