#include <vector>
#include <cuda_fp16.h>

#include "../tester/utils.h"

/**
 * @brief Computes the trace of a matrix in GPU.
 * 
 * The trace of a matrix is defined as the sum of its diagonal elements.
 * This function expects a flattened row-major matrix stored in a
 * std::vector. If the matrix is not square, the trace will sum up
 * elements along the main diagonal up to the smaller of rows or cols.
 * 
 * @tparam T The numeric type of matrix elements (e.g., float, int).
 * @param d_input A flattened matrix of size rows * cols.
 * @param partial_sum 用于存储中间计算结果
 * @param rows Number of rows in the matrix.
 * @param cols Number of columns in the matrix.
 * @return 没有
 * */ 
template <typename T>
__global__ void trace_kernel(const T* d_input, T* partial_sum, size_t rows, size_t cols) {
  extern __shared__ char shared_mem[]; // 在kernel被调用时再传入
  T* sdata = reinterpret_cast<T*>(shared_mem);

  const size_t tid = threadIdx.x;
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t diag_len = min(cols, rows);

  // 初始化共享内存
  sdata[tid] = T{0};
  if (i < diag_len) sdata[tid] = d_input[i * cols + i];
  __syncthreads();

  // 归并求得当前block的和
  for (size_t s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid+s];
    }
    __syncthreads();
  }

  // 写回block的和
  if (tid == 0) {
    partial_sum[blockIdx.x] = sdata[0];
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
  if (h_input.size() == 0) return T{0};

  const size_t diag_len = std::min(rows, cols);
  if (diag_len == 0) return T{0};

  // 分配设备内存
  const size_t input_size = rows*cols;
  const size_t input_bytes = input_size * sizeof(T);

  T* d_input = nullptr;
  T* d_partial = nullptr;

  cudaMalloc(&d_input, input_bytes);
  cudaMemcpy(d_input, h_input.data(), input_bytes, cudaMemcpyHostToDevice);

  // 设备设置
  const int block_size = 256;
  const int grid_size = (static_cast<int>(diag_len) + block_size - 1) / block_size;

  // 分配部分求和结果的内存
  cudaMalloc(&d_partial, grid_size * sizeof(T));

  // 启动kernel
  const size_t shared_mem_size = block_size * sizeof(T);
  trace_kernel<<<grid_size, block_size, shared_mem_size>>>(
    d_input, d_partial, rows, cols
  );

  // 复制部分求和结果
  std::vector<T> h_partial(grid_size);
  cudaDeviceSynchronize();
  cudaMemcpy(h_partial.data(), d_partial, grid_size * sizeof(T), cudaMemcpyDeviceToHost);

  // 最终求和
  T total = T{0};
  for (T val : h_partial) total += val;

  cudaFree(d_input);
  cudaFree(d_partial);

  return total;

}

// T->float
template <typename T>
__device__ __host__ inline float to_float(T val) {
    if constexpr (std::is_same_v<T, __half>) {
        return __half2float(val);
    } else {
        return static_cast<float>(val);
    }
}
// T/double -> float
template <typename T>
__device__ __host__ inline T from_float(float val) {
    if constexpr (std::is_same_v<T, __half>) {
        return __float2half(val);
    } else {
        return static_cast<T>(val);
    }
}


#define MAX_SRC_SEQ 2048  // MAX_SRC_SEQ >= src_seq_len

// 每个 CUDA 线程负责计算 一个 (batch, target_position, query_head) 三元组对应的 输出向量 o[b][i][qh][:],即输出张量 o 的每一个“head 向量”由一个线程独立完成全部 attention 计算。
template <typename T>
__global__ void flashAttentionKernel(
    const T* q,
    const T* k,
    const T* v,
    T* o,
    int B, int Tgt, int Src, int QH, int KVH, int D, bool is_causal, double inf)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * Tgt * QH;
    if (tid >= total) return;

    // 解码 (b, i, qh)
    int b = tid / (Tgt * QH);
    int rem = tid % (Tgt * QH);
    int i = rem / QH;
    int qh = rem % QH;

    if (Src > MAX_SRC_SEQ) return; // 安全防护

    int group_size = QH / KVH;
    int kvh = qh / group_size;

    // 计算偏移
    int q_off = b * (Tgt * QH * D) + i * (QH * D) + qh * D;
    int kv_batch_off = b * (Src * KVH * D);

    // local memory for scores
    double scores[MAX_SRC_SEQ];
    double scale = 1.0 / sqrt(static_cast<double>(D));
    double max_score = -inf; 

    // Step 1: Compute QK^T + causal mask
    #pragma unroll 4 // 循环展开
    for (int j = 0; j < Src; ++j) {
        if (is_causal && j > i) {
            // scores[j] = -1e30;
            scores[j] = -inf;
            continue;
        }
        float dot = 0.0f;
        for (int d = 0; d < D; ++d) {
            dot += static_cast<double>(to_float(q[q_off + d])) *
                   static_cast<double>(to_float(k[kv_batch_off + j * (KVH * D) + kvh * D + d]));
        }
        scores[j] = dot * scale;
        if (scores[j] > max_score) max_score = scores[j];
    }

    // Step 2: Softmax with mask
    double sum_exp = 0.0;
    for (int j = 0; j < Src; ++j) {
        if (is_causal && j > i) {
            scores[j] = 0.0;
            continue;
        }
        scores[j] = exp(scores[j] - max_score);
        sum_exp += scores[j];
    }

    if (sum_exp > 0.0) {
        for (int j = 0; j < Src; ++j) {
            if (!(is_causal && j > i)) {
                scores[j] /= sum_exp;
            }
        }
    }

    // Step 3: Weighted sum of V
    for (int d = 0; d < D; ++d) {
        double acc = 0.0;
        for (int j = 0; j < Src; ++j) {
            if (is_causal && j > i) continue;
            acc += scores[j] * static_cast<double>(
                to_float(v[kv_batch_off + j * (KVH * D) + kvh * D + d]));
        }
        o[q_off + d] = from_float<T>(static_cast<float>(acc));
    }
}

// /**
//  * @brief Computes flash attention for given query, key, and value tensors.
//  * 
//  * @tparam T Data type (float) for input/output tensors
//  * @param[in] h_q Query tensor of shape [batch_size, tgt_seq_len, query_heads, head_dim]
//  * @param[in] h_k Key tensor of shape [batch_size, src_seq_len, kv_heads, head_dim]
//  * @param[in] h_v Value tensor of shape [batch_size, src_seq_len, kv_heads, head_dim]
//  * @param[out] h_o Output attention tensor of shape [batch_size, tgt_seq_len, query_heads, head_dim]
//  * @param[in] batch_size Batch dimension size
//  * @param[in] target_seq_len Target sequence length
//  * @param[in] src_seq_len Source sequence length  
//  * @param[in] query_heads Number of query attention heads
//  * @param[in] kv_heads Number of key/value heads (supports grouped query attention)
//  * @param[in] head_dim Dimension size of each attention head
//  * @param[in] is_causal Whether to apply causal masking
//  */
template <typename T>
void flashAttention(const std::vector<T>& h_q,
                    const std::vector<T>& h_k,
                    const std::vector<T>& h_v,
                    std::vector<T>& h_o,
                    int batch_size,
                    int target_seq_len,
                    int src_seq_len,
                    int query_heads,
                    int kv_heads,
                    int head_dim,
                    bool is_causal) {
    size_t q_size = h_q.size();
    size_t k_size = h_k.size();
    size_t v_size = h_v.size();
    size_t o_size = h_o.size();
    // 分配设备内存
    T *d_q = nullptr, *d_k = nullptr, *d_v = nullptr, *d_o = nullptr;
    size_t bytes_q = q_size * sizeof(T);
    size_t bytes_kv = k_size * sizeof(T); // assume k/v same size

    cudaMalloc(&d_q, bytes_q);
    cudaMalloc(&d_k, bytes_kv);
    cudaMalloc(&d_v, bytes_kv);
    cudaMalloc(&d_o, bytes_q);

    // 拷贝到设备
    cudaMemcpy(d_q, h_q.data(), bytes_q, cudaMemcpyHostToDevice);
    cudaMemcpy(d_k, h_k.data(), bytes_kv, cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, h_v.data(), bytes_kv, cudaMemcpyHostToDevice);

    // 启动配置
    int total_threads = batch_size * target_seq_len * query_heads;
    const int threads_per_block = 256;
    int blocks = (total_threads + threads_per_block - 1) / threads_per_block;

    // 启动 kernel
    flashAttentionKernel<<<blocks, threads_per_block>>>(
        d_q, d_k, d_v, d_o,
        batch_size, target_seq_len, src_seq_len,
        query_heads, kv_heads, head_dim, is_causal, std::numeric_limits<double>::infinity()
    );

    // 同步并检查错误
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        cudaFree(d_q); cudaFree(d_k); cudaFree(d_v); cudaFree(d_o);
        throw std::runtime_error("CUDA kernel failed: " + std::string(cudaGetErrorString(err)));
    }

    // 拷贝结果回 host
    cudaMemcpy(h_o.data(), d_o, bytes_q, cudaMemcpyDeviceToHost);

    // 释放设备内存
    cudaFree(d_q);
    cudaFree(d_k);
    cudaFree(d_v);
    cudaFree(d_o);
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
