#include <vector>
#include <cuda_fp16.h>

#include "../tester/utils.h"
#define BLOCKSIZE 256

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
 __global__ void traceKernel(const T* input,size_t rows, size_t cols,T* output){
  __shared__ T cache[BLOCKSIZE];
  cache[threadIdx.x] = 0;
  for(int i = blockIdx.x * blockDim.x + threadIdx.x;i < cols && i<rows;i+=blockDim.x * gridDim.x ){
    cache[threadIdx.x] += input[i*cols + i];
  }
  __syncthreads();
  //上面就把跨block线程的迹加到cache里面去。
  for(int total = blockDim.x / 2 ;total > 0;total /= 2){
    if(threadIdx.x < total){
      cache[threadIdx.x] += cache[threadIdx.x + total];
    }
    __syncthreads();
  }
  if(threadIdx.x == 0){
    atomicAdd(output,cache[0]);
  }
 }

template <typename T>
T trace(const std::vector<T>& h_input, size_t rows, size_t cols) {
  // TODO: Implement the trace function
  int Blocknumber_x = (cols + BLOCKSIZE - 1) / BLOCKSIZE;

  T* d_input;
  size_t size = rows * cols * sizeof(T);
  cudaMalloc((void**)&d_input, size);
  cudaMemcpy(d_input, h_input.data(), size, cudaMemcpyHostToDevice);

  T* d_output;
  cudaMalloc((void**)&d_output, sizeof(T));
  cudaMemset(d_output, 0, sizeof(T));

  traceKernel<T><<<Blocknumber_x,BLOCKSIZE>>>(d_input, rows, cols,d_output);
  T h_output;
  cudaMemcpy(&h_output, d_output, sizeof(T), cudaMemcpyDeviceToHost);
  cudaFree(d_input);
  cudaFree(d_output);
  return h_output;
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
