#include <vector>
#include <cuda_fp16.h>

#include "../tester/utils.h"

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

#define FULLMASK 0xffffffff

template <typename T>
__device__ T WarpReduce(T val)
{
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
    {
        val += __shfl_down_sync(FULLMASK, val, offset);
    }
    return val;
}


template <typename T>
__global__ void trace_kernel(const T* d_input, T* d_output, const size_t N)
{

    extern __shared__ char shared_mem[];
    T* smem = reinterpret_cast<T*>(shared_mem);

    const size_t tid = threadIdx.x;
    const size_t idx = tid + blockIdx.x * blockDim.x;

    T sum = 0;
    for (int i = idx; i < N; i += blockDim.x * gridDim.x)
    {
        sum += d_input[i];
    }

    T warp_sum = WarpReduce<T>(sum);

    const size_t warpID = tid / warpSize;
    const size_t laneID = tid % warpSize;

    if (laneID == 0)
    {
        smem[warpID] = warp_sum;
    }
    __syncthreads();

    if (warpID == 0)
    {
        T block_sum = (tid < (blockDim.x + warpSize - 1) / warpSize) ? smem[laneID] : 0;
        block_sum = WarpReduce(block_sum);
        if (tid == 0) atomicAdd(d_output, block_sum);
    }
}

template <typename T>
T trace(const std::vector<T>& h_input, size_t rows, size_t cols) {

    const size_t n_diag = (rows < cols) ? rows : cols;
    std::vector<T> temp;
    temp.reserve(n_diag);
    for (size_t i = 0; i < n_diag; i++) {
        temp.push_back(h_input[(size_t)i * cols + i]);
    }


    const size_t N = temp.size();
    const size_t num_bytes = sizeof(T) * N;

    T *d_input, *d_output;
    cudaMalloc((void**)&d_input, num_bytes);
    cudaMalloc((void**)&d_output, sizeof(T));
    cudaMemset(d_output, 0, sizeof(T));

    cudaMemcpy(d_input, temp.data(), num_bytes, cudaMemcpyHostToDevice);

    int threadsPerBlock = 1024;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    size_t smem_size = threadsPerBlock * sizeof(T);

    trace_kernel<T><<<blocksPerGrid, threadsPerBlock, smem_size>>>(d_input, d_output, N);

    T h_res = 0;
    cudaDeviceSynchronize();

    cudaMemcpy(&h_res, d_output, sizeof(T), cudaMemcpyDeviceToHost);


    cudaFree(d_input);
    cudaFree(d_output);

    return h_res;
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
