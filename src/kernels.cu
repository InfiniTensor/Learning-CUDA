#include <vector>
#include <cuda_fp16.h>
#include <cooperative_groups.h>


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
   
#pragma unroll    
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
    {
        val += __shfl_down_sync(FULLMASK, val, offset);
    }
    return val;
}


namespace cg = cooperative_groups;

template <typename T>
__global__ void reduce_kernel(const T* d_input, T* d_output, T* d_workspace, const size_t N)
{
    cg::grid_group grid = cg::this_grid();
    T sum = (T)0;
    
    const size_t tid = threadIdx.x;
    const size_t bid = blockIdx.x;
    const size_t idx = tid + bid * blockDim.x;
    const size_t laneID = tid % warpSize;
    const size_t warpID = tid / warpSize;


    for (size_t i = idx; i < N; i += gridDim.x * blockDim.x)
    {
        sum += d_input[i];
    }

    // warp 内的和
    T warp_sum = WarpReduce(sum);

    __shared__ T smem[32];
    if (laneID == 0) smem[warpID] = warp_sum;
    __syncthreads();

    // 用一个warp 对整个block内的所有warp之和归约
    if (warpID == 0)
    {
        T block_sum = (tid < ((blockDim.x + warpSize - 1) / warpSize)) ? smem[laneID] : 0;
        block_sum = WarpReduce(block_sum);
        if(tid == 0) smem[0] = block_sum;
    }
    __syncthreads();

    if (tid == 0) d_workspace[bid] = smem[0];
    grid.sync();

    // 用一个block对d_workspace中所有元素归约
    if (bid == 0)
    {
        T final_sum = 0;
        for (size_t i = idx; i < gridDim.x; i += blockDim.x)
        {
            final_sum += d_workspace[i];
        }
        final_sum = WarpReduce(final_sum);

        if (laneID == 0) smem[warpID] = final_sum;
        __syncthreads();

        if (warpID == 0)
        {
            final_sum = (tid < ((blockDim.x + warpSize - 1) / warpSize)) ? smem[laneID] : 0;
            final_sum = WarpReduce(final_sum);
            if (tid == 0) *d_output = final_sum;
        }
    }

}

template <typename T>
T trace(const std::vector<T>& h_input, size_t rows, size_t cols) {

    const size_t n_diag = (rows < cols) ? rows : cols;
    std::vector<T> temp;
    temp.reserve(n_diag);

#pragma unroll    
    for (size_t i = 0; i < n_diag; i++) {
        temp.push_back(h_input[(size_t)i * cols + i]);
    }


    size_t N = temp.size();

    T *d_input, *d_output, *d_workspace;
    cudaMalloc(&d_input, sizeof(T) * N);
    cudaMalloc(&d_output, sizeof(T));
    cudaMemcpy(d_input, temp.data(), sizeof(T) * N, cudaMemcpyHostToDevice);
    cudaMemset(d_output, 0, sizeof(T));

    int dev = 0;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);

    int threadsPerBlock = prop.maxThreadsPerBlock;
    size_t smem_size = (threadsPerBlock / 32) * sizeof(T); 

    int numBlocksPerSm = 0;
    auto kernel_func = reduce_kernel<T>;

    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, kernel_func, threadsPerBlock, smem_size);

    int maxActiveBlocks = numBlocksPerSm * prop.multiProcessorCount;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    if (blocks > maxActiveBlocks) blocks = maxActiveBlocks;

    cudaMalloc(&d_workspace, blocks * sizeof(T));

    // 参数包
    void* kernelArgs[] = { &d_input, &d_output, &d_workspace, &N };

    cudaLaunchCooperativeKernel((void*)kernel_func, dim3(blocks), dim3(threadsPerBlock), kernelArgs, smem_size, 0);

    T res;
    cudaMemcpy(&res, d_output, sizeof(T), cudaMemcpyDeviceToHost);
    
    cudaFree(d_input); cudaFree(d_output); cudaFree(d_workspace);
    return res;
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
