#include <vector>
#include <cuda_fp16.h>
#include <iostream>

#include "../tester/utils.h"
// 清理函数 
template <typename T>
void cleanup(T* d_q, T* d_k, T* d_v, T* d_o) {
    if (d_q) cudaFree(d_q);
    if (d_k) cudaFree(d_k);
    if (d_v) cudaFree(d_v);
    if (d_o) cudaFree(d_o);
}

// 优化但保持正确性的kernel函数 

// float版本的优化kernel
__global__ void flash_attention_kernel_float(
    const float* Q, const float* K, const float* V, float* O,
    int batch_size, int target_seq_len, int src_seq_len,
    int query_heads, int kv_heads, int head_dim,
    bool is_causal, float scale) {
    
    // 每个线程处理一个输出位置
    int b = blockIdx.x;
    int t = blockIdx.y;
    int h = blockIdx.z;
    int d = threadIdx.x;
    
    if (b >= batch_size || t >= target_seq_len || h >= query_heads || d >= head_dim) return;
    
    int kvh = h / (query_heads / kv_heads);
    int valid_len = is_causal ? min(t + 1, src_seq_len) : src_seq_len;
    
    size_t q_base = ((b * target_seq_len + t) * query_heads + h) * head_dim;
    size_t kv_base = b * src_seq_len * kv_heads * head_dim + kvh * head_dim;
    
    // 使用共享内存存储查询向量
    extern __shared__ float shared_mem[];
    float* q_shared = shared_mem;
    
    // 协作加载查询向量到共享内存
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        q_shared[i] = Q[q_base + i];
    }
    __syncthreads();
    
    // 计算最大分数
    float max_score = -1e10f;
    for (int s = 0; s < valid_len; ++s) {
        float dot = 0.0f;
        size_t k_base = kv_base + s * kv_heads * head_dim;
        
        // 使用共享内存中的查询向量
        #pragma unroll(4)
        for (int i = 0; i < head_dim; ++i) {
            dot += q_shared[i] * K[k_base + i];
        }
        
        float score = dot * scale;
        if (score > max_score) max_score = score;
    }
    
    // 计算输出
    float sum_exp = 0.0f;
    float output = 0.0f;
    
    for (int s = 0; s < valid_len; ++s) {
        float dot = 0.0f;
        size_t k_base = kv_base + s * kv_heads * head_dim;
        size_t v_base = kv_base + s * kv_heads * head_dim + d;
        
        #pragma unroll(4)
        for (int i = 0; i < head_dim; ++i) {
            dot += q_shared[i] * K[k_base + i];
        }
        
        float score = dot * scale;
        float exp_val = expf(score - max_score);
        
        sum_exp += exp_val;
        output += exp_val * V[v_base];
    }
    
    // 归一化
    if (sum_exp > 1e-12f) {
        output = output / sum_exp;
    } else if (valid_len > 0) {
        output = 0.0f;
        for (int s = 0; s < valid_len; ++s) {
            size_t v_base = kv_base + s * kv_heads * head_dim + d;
            output += V[v_base];
        }
        output = output / valid_len;
    }
    
    O[q_base + d] = output;
}

// half版本的优化kernel
__global__ void flash_attention_kernel_half(
    const __half* Q, const __half* K, const __half* V, __half* O,
    int batch_size, int target_seq_len, int src_seq_len,
    int query_heads, int kv_heads, int head_dim,
    bool is_causal, __half scale) {
    
    int b = blockIdx.x;
    int t = blockIdx.y;
    int h = blockIdx.z;
    int d = threadIdx.x;
    
    if (b >= batch_size || t >= target_seq_len || h >= query_heads || d >= head_dim) return;
    
    int kvh = h / (query_heads / kv_heads);
    int valid_len = is_causal ? min(t + 1, src_seq_len) : src_seq_len;
    
    size_t q_base = ((b * target_seq_len + t) * query_heads + h) * head_dim;
    size_t kv_base = b * src_seq_len * kv_heads * head_dim + kvh * head_dim;
    
    // 使用共享内存存储查询向量
    extern __shared__ float shared_mem[];
    float* q_shared = shared_mem;
    
    float scale_f = __half2float(scale);
    
    // 协作加载查询向量到共享内存
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        q_shared[i] = __half2float(Q[q_base + i]);
    }
    __syncthreads();
    
    // 计算最大分数
    float max_score = -1e4f;
    for (int s = 0; s < valid_len; ++s) {
        float dot = 0.0f;
        size_t k_base = kv_base + s * kv_heads * head_dim;
        
        #pragma unroll(4)
        for (int i = 0; i < head_dim; ++i) {
            dot += q_shared[i] * __half2float(K[k_base + i]);
        }
        
        float score = dot * scale_f;
        if (score > max_score) max_score = score;
    }
    
    // 计算softmax和输出
    float sum_exp = 0.0f;
    float output_f = 0.0f;
    
    for (int s = 0; s < valid_len; ++s) {
        float dot = 0.0f;
        size_t k_base = kv_base + s * kv_heads * head_dim;
        size_t v_base = kv_base + s * kv_heads * head_dim + d;
        
        #pragma unroll(4)
        for (int i = 0; i < head_dim; ++i) {
            dot += q_shared[i] * __half2float(K[k_base + i]);
        }
        
        float score = dot * scale_f;
        float shifted = score - max_score;
        
        // 限制范围确保稳定性
        if (shifted > 10.0f) shifted = 10.0f;
        if (shifted < -20.0f) shifted = -20.0f;
        
        float exp_val = expf(shifted);
        sum_exp += exp_val;
        output_f += exp_val * __half2float(V[v_base]);
    }
    
    // 归一化
    if (sum_exp > 1e-7f) {
        output_f = output_f / sum_exp;
    } else if (valid_len > 0) {
        output_f = 0.0f;
        for (int s = 0; s < valid_len; ++s) {
            size_t v_base = kv_base + s * kv_heads * head_dim + d;
            output_f += __half2float(V[v_base]);
        }
        output_f = output_f / valid_len;
    }
    
    O[q_base + d] = __float2half(output_f);
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
    if (h_input.empty() || rows == 0 || cols == 0) return T(0);
    size_t n = rows < cols ? rows : cols;
    T sum = T(0);
    for (size_t i = 0; i < n; ++i) sum += h_input[i * cols + i];
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
    
    // 基本检查
    if (batch_size <= 0 || target_seq_len <= 0 || src_seq_len <= 0 || 
        query_heads <= 0 || kv_heads <= 0 || head_dim <= 0) {
        return;
    }
    
    if (query_heads % kv_heads != 0) return;
    
    // 计算大小
    size_t q_size = batch_size * target_seq_len * query_heads * head_dim;
    size_t kv_size = batch_size * src_seq_len * kv_heads * head_dim;
    size_t o_size = batch_size * target_seq_len * query_heads * head_dim;
    
    // 分配设备内存
    T *d_q = nullptr, *d_k = nullptr, *d_v = nullptr, *d_o = nullptr;
    
    cudaMalloc(&d_q, q_size * sizeof(T));
    cudaMalloc(&d_k, kv_size * sizeof(T));
    cudaMalloc(&d_v, kv_size * sizeof(T));
    cudaMalloc(&d_o, o_size * sizeof(T));
    
    // 拷贝数据
    cudaMemcpy(d_q, h_q.data(), q_size * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_k, h_k.data(), kv_size * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, h_v.data(), kv_size * sizeof(T), cudaMemcpyHostToDevice);
    
    // 缩放因子
    T scale;
    if constexpr (std::is_same_v<T, __half>) {
        float head_dim_f = static_cast<float>(head_dim);
        float scale_f = 1.0f / sqrtf(head_dim_f);
        if (scale_f > 5.0f) scale_f = 5.0f;
        scale = __float2half(scale_f);
    } else {
        scale = T(1.0 / sqrt(static_cast<double>(head_dim)));
    }
    
    // 启动对应的kernel
    dim3 grid(batch_size, target_seq_len, query_heads);
    
    // 优化block大小：确保是32的倍数（warp大小）
    int block_size = 256;
    if (head_dim < 256) {
        block_size = ((head_dim + 31) / 32) * 32; // 向上取整到32的倍数
    }
    
    // 计算共享内存大小
    size_t shared_mem_size = head_dim * sizeof(float);
    
    if constexpr (std::is_same_v<T, float>) {
        flash_attention_kernel_float<<<grid, block_size, shared_mem_size>>>(
            d_q, d_k, d_v, d_o,
            batch_size, target_seq_len, src_seq_len,
            query_heads, kv_heads, head_dim,
            is_causal, scale);
    } else {
        flash_attention_kernel_half<<<grid, block_size, shared_mem_size>>>(
            d_q, d_k, d_v, d_o,
            batch_size, target_seq_len, src_seq_len,
            query_heads, kv_heads, head_dim,
            is_causal, scale);
    }
    
    // 同步和错误检查
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
    }
    
    // 拷贝结果
    h_o.resize(o_size);
    cudaMemcpy(h_o.data(), d_o, o_size * sizeof(T), cudaMemcpyDeviceToHost);
    
    // 清理
    cleanup(d_q, d_k, d_v, d_o);
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




