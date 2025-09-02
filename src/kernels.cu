#include <vector>
#include <iostream>

#include "../tester/utils.h"

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

template <typename T>
__global__ void bitonic_sort_kernel(T* input, int n, int desc, int k, int j) {
  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = x ^ j;
  if (x >= n || x > y) return;
  T valx = input[x];
  T valy = input[y];
  T val;
  if ((valx > valy) == (desc ^ ((x & k) != 0))) {
    if (x > y) {
      input[x] = valy;
      input[y] = valx;
    } 
  } else {
    if (x < y) {
      input[x] = valy;
      input[y] = valx;  
    }
  }
}

template <typename T>
__global__ void init_input(T* input, int n, int mx) {
  int x = blockDim.x * blockIdx.x + threadIdx.x;
  if (x < n && x >= mx) {
    input[x] = T(-1e9);
  }
}

template <typename T>
T kthLargest(const std::vector<T>& h_input, size_t k) {
  // TODO: Implement the kthLargest function
  int n = h_input.size();
  
  if (k < 1 || k > n) 
    return T(-100);
  
  while (__builtin_popcount(n) != 1) n += (n & (-n));

  T *d_input;

  size_t size_input = h_input.size()  * sizeof(T);
  size_t size_malloc = n * sizeof(T);

  CUDA_CHECK(cudaMalloc(&d_input, size_malloc));
  
  CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), size_input, cudaMemcpyHostToDevice));
  
  
  int blockSize = 256;
  int gridSize = (n + 255) / 256;
  
  init_input<T><<<gridSize, blockSize>>>(d_input, n, h_input.size());
  CUDA_CHECK(cudaDeviceSynchronize());
  

  for (int k = 2; k <= n; k <<= 1) {
    for (int j = (k >> 1); j > 0; j >>= 1) {
      bitonic_sort_kernel<T><<<gridSize, blockSize>>>(d_input, n, 1, k, j);
      CUDA_CHECK(cudaDeviceSynchronize());
    }
  }
  
  T result;
  CUDA_CHECK(cudaMemcpy(&result, d_input + (k - 1), sizeof(T), cudaMemcpyDeviceToHost));
  
  CUDA_CHECK(cudaFree(d_input));
  
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

template <typename T>
__device__ T block_reduce_max(T* smem, T val) {
  int tid = threadIdx.x;
  int lane = tid % 32;
  int warp_id = tid / 32;

  for (int offset = 16; offset > 0; offset >>= 1) {
    T other = __shfl_xor_sync(0xFFFFFFFF, val, offset);
    val = fmaxf(val, other);
  }

  if (lane == 0)
    smem[warp_id] = val;
  __syncthreads();

  if (warp_id == 0) {
    val = lane >= 8 ? -1e9 : smem[lane];
    for (int offset = 16; offset > 0; offset >>= 1) {
      T other = __shfl_xor_sync(0xFFFFFFFF, val, offset);
      val = fmaxf(val, other);
    }
    smem[0] = val;
  }
  __syncthreads();

  return smem[0];
}

template <typename T>
__device__ T block_reduce_sum(T* smem, T val) {
  int tid = threadIdx.x;
  int lane = tid % 32;
  int warp_id = tid / 32;

  for (int offset = 16; offset > 0; offset >>= 1) {
    T other = __shfl_xor_sync(0xFFFFFFFF, val, offset);
    val = val + other;
  }

  if (lane == 0)
    smem[warp_id] = val;
  __syncthreads();

  if (warp_id == 0) {
    val = lane >= 8 ? 0 : smem[lane];
    for (int offset = 16; offset > 0; offset >>= 1) {
      T other = __shfl_xor_sync(0xFFFFFFFF, val, offset);
      val = val + other;
    }
    smem[0] = val;
  }
  __syncthreads();

  return smem[0];
}

template <typename T>
__global__ void flashAttentionKernel(
  const T* query, const T* key,
  const T* value, T* output,
  int batch_size, int target_seq_len, int src_seq_len, 
  int query_heads, int kv_heads, int head_dim, bool is_causal
) {

  
  int batch_id = blockIdx.x;
  int heads_id = blockIdx.y;
  int kv_id = heads_id * kv_heads / query_heads;

  if (batch_id >= batch_size || heads_id >= query_heads) return;
  
  extern __shared__ unsigned char shared_mem[];
  T* scores = reinterpret_cast<T*>(shared_mem);
  T* smem = scores + src_seq_len;

  for (int tgt = 0; tgt < target_seq_len; tgt ++) {
    float mx = -1e9;
    for (int src = threadIdx.x; src < src_seq_len; src += blockDim.x) {
      if (is_causal && tgt < src) {
        scores[src] = -1e9f;
      } else {
        float sum = 0.;
        for (int dim = 0; dim < head_dim; dim ++) {
          // query[batch_id][tgt][heads_id][dim]
          int qid = dim + head_dim * (heads_id + query_heads * (tgt + target_seq_len * batch_id));
          // key[batch_id][src][kv_id][dim]
          int kid = dim + head_dim * (kv_id + kv_heads * (src + src_seq_len * batch_id));
          sum += query[qid] * key[kid];
        }
        scores[src] = sum / sqrtf(float(head_dim));
        mx = fmaxf(mx, scores[src]);
      }
    }
    __syncthreads();

    mx = block_reduce_max(smem, mx);
    
    T sum = 0.;
    for (int src = threadIdx.x; src < src_seq_len; src += blockDim.x) {
      scores[src] = expf(scores[src] - mx);
      sum += scores[src];
    }
    __syncthreads();

    sum = block_reduce_sum(smem, sum);

    for (int src = threadIdx.x; src < src_seq_len; src += blockDim.x) {
      scores[src] = scores[src] / (sum + 1e-8f);
    }
    __syncthreads();

    for (int dim = threadIdx.x; dim < head_dim; dim += blockDim.x) {
      float sum = 0.;
      for (int src = 0; src < src_seq_len; src ++) {
        // value[batch_id][src][kv_id][dim]
        int vidx = dim + head_dim * (kv_id + kv_heads * (src + src_seq_len * batch_id));
        sum += scores[src] * value[vidx];
      }
      // output[batch_id][tgt][heads_id][dim]
      int oidx = dim + head_dim * (heads_id + query_heads * (tgt + target_seq_len * batch_id));
      output[oidx] = sum;
    }
    __syncthreads();
  }
}


template <typename T>
void flashAttention(
  const std::vector<T>& h_q, const std::vector<T>& h_k,
  const std::vector<T>& h_v, std::vector<T>& h_o,
  int batch_size, int target_seq_len, int src_seq_len, 
  int query_heads, int kv_heads, int head_dim, bool is_causal
) {

  h_o.resize(batch_size * target_seq_len * query_heads * head_dim);

  T* d_q, *d_k, *d_v, *d_o;
  size_t size_q = h_q.size() * sizeof(T);
  size_t size_k = h_k.size() * sizeof(T);
  size_t size_v = h_v.size() * sizeof(T);
  size_t size_o = h_o.size() * sizeof(T);
  
  CUDA_CHECK(cudaMalloc(&d_q, size_q));
  CUDA_CHECK(cudaMalloc(&d_k, size_k));
  CUDA_CHECK(cudaMalloc(&d_v, size_v));
  CUDA_CHECK(cudaMalloc(&d_o, size_o));

  CUDA_CHECK(cudaMemcpy(d_q, h_q.data(), size_q, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_k, h_k.data(), size_k, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_v, h_v.data(), size_v, cudaMemcpyHostToDevice));

  dim3 gridSize(batch_size, query_heads);
  int blockSize = 256;
  size_t shared_mem = src_seq_len * sizeof(T) + 8 * sizeof(T);
  flashAttentionKernel<T><<<gridSize, blockSize, shared_mem>>>(
    d_q, d_k, d_v, d_o,
    batch_size, target_seq_len, src_seq_len, 
    query_heads, kv_heads, head_dim, is_causal
  );

  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(h_o.data(), d_o, size_o, cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(d_q));
  CUDA_CHECK(cudaFree(d_k));
  CUDA_CHECK(cudaFree(d_v));
  CUDA_CHECK(cudaFree(d_o));

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
