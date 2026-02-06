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



#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>

template<typename T>
__global__ 
void flash_attn_kernel(
    const T* Q, const T* K, const T* V, 
    const int N_q, const int N_kv, const int d,
    const int Tc, const int Tr, const int Bc, const int Br, 
    const T softmax_scale, const int group_size, bool is_causal,
    float* l, float* m, T* O) 
{
    // 每个 thread 负责 Q 的一行 (blockDim.x == Br)
    const int tid = threadIdx.x;
    const int bx = blockIdx.x;  // batch index
    const int by = blockIdx.y;  // query head index (by)

    // GQA
    const int kv_head_idx = by / group_size;
    const int num_q_heads = gridDim.y;
    const int num_kv_heads = num_q_heads / group_size;

    // Qi(Br*d) + Kj(Bc*d) + Vj(Bc*d) + S(Br*Bc)
    extern __shared__ float sram[];
    float* Qi = sram;
    float* Kj = &sram[Br * d];
    float* Vj = &sram[Br * d + Bc * d];
    float* S  = &sram[Br * d + Bc * d * 2];

    // 全局内存偏移计算 [B, N, H, D]
    // 每个 Batch 的起始位置
    const size_t batch_stride_q = (size_t)N_q * num_q_heads * d;
    const size_t batch_stride_kv = (size_t)N_kv * num_kv_heads * d;
    
    // 定位到当前 Batch 和对应的 Head
    const T* Q_ptr = Q + (size_t)bx * batch_stride_q + (by * d);
    const T* K_ptr = K + (size_t)bx * batch_stride_kv + (kv_head_idx * d);
    const T* V_ptr = V + (size_t)bx * batch_stride_kv + (kv_head_idx * d);
    T* O_ptr = O + (size_t)bx * batch_stride_q + (by * d);

    // l, m 布局为 [B, H, N]
    float* l_start = l + (size_t)(bx * num_q_heads + by) * N_q;
    float* m_start = m + (size_t)(bx * num_q_heads + by) * N_q;

    // 跨行步长
    const int stride_q = num_q_heads * d;
    const int stride_kv = num_kv_heads * d;

    // Outer Loop: KV 分块
    for (int j = 0; j < Tc; j++) 
    {

        // 加载 Kj, Vj 到 Shared Memory
        if (tid < Bc) 
        {
            int kv_row_idx = j * Bc + tid;
            for (int x = 0; x < d; x++) 
            {
                if (kv_row_idx < N_kv) 
                {
                    Kj[tid * d + x] = (float)K_ptr[kv_row_idx * stride_kv + x];
                    Vj[tid * d + x] = (float)V_ptr[kv_row_idx * stride_kv + x];
                } 
                else 
                {
                    Kj[tid * d + x] = 0.0f;
                    Vj[tid * d + x] = 0.0f;
                }
            }
        }
        __syncthreads();

        // --- Inner Loop: Q 分块 ---
        for (int i = 0; i < Tr; i++) 
        {
            int q_row_idx = i * Br + tid;
            if (q_row_idx >= N_q) continue;

            // 2. 加载 Qi 到 Shared Memory
            for (int x = 0; x < d; x++) 
            {
                Qi[tid * d + x] = (float)Q_ptr[q_row_idx * stride_q + x];
            }

            // 3. 读取旧的统计量
            float m_prev = (j == 0) ? -INFINITY : (float)m_start[q_row_idx];
            float l_prev = (j == 0) ? 0.0f : (float)l_start[q_row_idx];

            // 4. 计算 S = QK^T 并找局部最大值
            float m_curr = -INFINITY;
            for (int y = 0; y < Bc; y++) 
            {
                int kv_idx = j * Bc + y;
                float score = -INFINITY;

                if (kv_idx < N_kv && !(is_causal && q_row_idx < kv_idx)) 
                {
                    float sum = 0.0f;
                    for (int x = 0; x < d; x++) 
                    {
                        sum += Qi[tid * d + x] * Kj[y * d + x];
                    }
                    score = sum * (float)softmax_scale;
                }
                S[tid * Bc + y] = score;
                m_curr = fmaxf(m_curr, score);
            }

            // 5. 计算局部累加和与 P 矩阵
            float l_curr = 0.0f;
            for (int y = 0; y < Bc; y++) 
            {
                float p = (S[tid * Bc + y] == -INFINITY) ? 0.0f : __expf(S[tid * Bc + y] - m_curr);
                S[tid * Bc + y] = p; 
                l_curr += p;
            }

            // 6. 更新全局统计量
            float m_new = fmaxf(m_prev, m_curr);
            float alpha = __expf(m_prev - m_new);
            float beta = __expf(m_curr - m_new);
            float l_new = alpha * l_prev + beta * l_curr;

            // 7. 更新 Output O
            for (int x = 0; x < d; x++) 
            {
                float pv = 0.0f;
                for (int y = 0; y < Bc; y++) 
                {
                    pv += S[tid * Bc + y] * Vj[y * d + x];
                }

                float o_prev = (j == 0) ? 0.0f : (float)O_ptr[q_row_idx * stride_q + x];
                float o_new = (alpha * l_prev * o_prev + beta * pv) / l_new;
                O_ptr[q_row_idx * stride_q + x] = (T)o_new;
            }

            // 8. 写回统计量
            m_start[q_row_idx] = m_new;
            l_start[q_row_idx] = l_new;
        }
        __syncthreads();
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
                    int query_heads, int kv_heads, int head_dim, bool is_causal) 
{
    
    // 1. 基础参数计算
    int group_size = query_heads / kv_heads; // GQA: 每个 KV Head 对应多少个 Query Head
    float softmax_scale = 1.0f / sqrtf(head_dim);

    // 设置分块大小（这里可以根据 L1 Cache/Shared Mem 大小动态调整）
    // 通常在现代 GPU 上，Br 和 Bc 选 32 或 64 是比较稳妥的
    const int Br = 32;
    const int Bc = 32;
    const int Tr = (target_seq_len + Br - 1) / Br; // Query 方向的分块数
    const int Tc = (src_seq_len + Bc - 1) / Bc;   // KV 方向的分块数

    // 2. GPU 内存分配 (Device Memory)
    T *d_q, *d_k, *d_v, *d_o;
    float *d_l, *d_m; // 用于存储 Online Softmax 的统计量 (Row Sum 和 Row Max)

    size_t q_size = batch_size * target_seq_len * query_heads * head_dim * sizeof(T);
    size_t kv_size = batch_size * src_seq_len * kv_heads * head_dim * sizeof(T);
    size_t stats_size = batch_size * query_heads * target_seq_len * sizeof(float);

    cudaMalloc(&d_q, q_size);
    cudaMalloc(&d_k, kv_size);
    cudaMalloc(&d_v, kv_size);
    cudaMalloc(&d_o, q_size);
    cudaMalloc(&d_l, stats_size);
    cudaMalloc(&d_m, stats_size);

    // 3. 将数据从 Host 搬运到 Device
    cudaMemcpy(d_q, h_q.data(), q_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_k, h_k.data(), kv_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, h_v.data(), kv_size, cudaMemcpyHostToDevice);
    
    
    // 初始化统计量：max 设为负无穷，sum 设为 0
    // (这里可以通过一个简单的 kernel 或 cudaMemset 初始化)
    // 假设我们有一个初始化函数，或者在 flash_attn_kernel 内部处理

    // 4. 配置 Kernel 启动参数
    // 每个 Block 处理一个 Batch 中的一个 Query Head
    dim3 grid(batch_size, query_heads);
    // 每个 Block 内的线程数，这里简单起见让一个线程处理一行 (Br)
    dim3 block(Br); 

    // 计算 Shared Memory 需要的大小 (Qi, Kj, Vj)
    size_t sram_elements = (Br * head_dim) + (Bc * head_dim) + (Bc * head_dim) + (Br * Bc);
    size_t shared_mem_size = sram_elements * sizeof(float);

    // 5. 启动 Kernel
    flash_attn_kernel<T><<<grid, block, shared_mem_size>>>(
        d_q, d_k, d_v, 
        target_seq_len, src_seq_len, head_dim,
        Tc, Tr, Bc, Br, 
        (T)softmax_scale, group_size, is_causal,
        d_l, d_m, d_o
    );

    // 6. 将结果搬回 Host
    cudaMemcpy(h_o.data(), d_o, q_size, cudaMemcpyDeviceToHost);

    cudaFree(d_q); cudaFree(d_k); cudaFree(d_v);
    cudaFree(d_o); cudaFree(d_l); cudaFree(d_m);
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
