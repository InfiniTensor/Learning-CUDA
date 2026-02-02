#include <vector>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

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
