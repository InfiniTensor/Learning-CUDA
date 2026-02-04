#include <vector>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cfloat>
#include <algorithm>

#include "../tester/utils.h"

// ============================================================================
// TRACE IMPLEMENTATION
// ============================================================================

template <typename T>
__global__ void trace_kernel(const T* d_input, T* d_partial_sums, 
                             size_t rows, size_t cols, size_t min_dim) {
    extern __shared__ char shared_mem[];
    T* sdata = reinterpret_cast<T*>(shared_mem);
    
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    T val = (idx < min_dim) ? d_input[idx * cols + idx] : T(0);
    sdata[tid] = val;
    __syncthreads();
    
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        d_partial_sums[blockIdx.x] = sdata[0];
    }
}

template <typename T>
T trace(const std::vector<T>& h_input, size_t rows, size_t cols) {
    size_t min_dim = (rows < cols) ? rows : cols;
    
    if (min_dim == 0) return T(0);
    
    if (min_dim < 1024) {
        T sum = T(0);
        for (size_t i = 0; i < min_dim; ++i) {
            sum += h_input[i * cols + i];
        }
        return sum;
    }
    
    T* d_input;
    size_t matrix_size = rows * cols * sizeof(T);
    cudaMalloc(&d_input, matrix_size);
    cudaMemcpy(d_input, h_input.data(), matrix_size, cudaMemcpyHostToDevice);
    
    int block_size = 256;
    int num_blocks = (min_dim + block_size - 1) / block_size;
    
    T* d_partial_sums;
    cudaMalloc(&d_partial_sums, num_blocks * sizeof(T));
    
    size_t shared_mem_size = block_size * sizeof(T);
    trace_kernel<<<num_blocks, block_size, shared_mem_size>>>(
        d_input, d_partial_sums, rows, cols, min_dim
    );
    
    std::vector<T> h_partial_sums(num_blocks);
    cudaMemcpy(h_partial_sums.data(), d_partial_sums, 
               num_blocks * sizeof(T), cudaMemcpyDeviceToHost);
    
    T result = T(0);
    for (int i = 0; i < num_blocks; ++i) {
        result += h_partial_sums[i];
    }
    
    cudaFree(d_input);
    cudaFree(d_partial_sums);
    
    return result;
}

// ============================================================================
// FLASH ATTENTION IMPLEMENTATION
// ============================================================================

template <typename T>
struct TypeConverter {
    __device__ __forceinline__ static float to_float(T val);
    __device__ __forceinline__ static T from_float(float val);
};

template <>
struct TypeConverter<float> {
    __device__ __forceinline__ static float to_float(float val) {
        return val;
    }
    __device__ __forceinline__ static float from_float(float val) {
        return val;
    }
};

template <>
struct TypeConverter<half> {
    __device__ __forceinline__ static float to_float(half val) {
        return __half2float(val);
    }
    __device__ __forceinline__ static half from_float(float val) {
        return __float2half(val);
    }
};

// Flash Attention Kernel
template <typename T>
__global__ void flash_attention_forward_kernel(
    const T* __restrict__ Q,
    const T* __restrict__ K, 
    const T* __restrict__ V,
    T* __restrict__ O,
    const int batch_size,
    const int tgt_seq_len,
    const int src_seq_len,
    const int query_heads,
    const int kv_heads,
    const int head_dim,
    const bool is_causal,
    const float softmax_scale
) {
    // 线程索引
    const int batch_idx = blockIdx.z;
    const int head_idx = blockIdx.y;
    const int global_tgt_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 边界检查
    if (global_tgt_idx >= tgt_seq_len) return;
    
    // GQA 支持
    const int kv_head_idx = (head_idx * kv_heads) / query_heads;
    
    // 计算偏移量
    const size_t q_batch_stride = tgt_seq_len * query_heads * head_dim;
    const size_t q_seq_stride = query_heads * head_dim;
    const size_t q_head_stride = head_dim;
    
    const int q_offset = batch_idx * q_batch_stride + 
                         global_tgt_idx * q_seq_stride + 
                         head_idx * q_head_stride;
    
    const size_t kv_batch_stride = src_seq_len * kv_heads * head_dim;
    const size_t kv_seq_stride = kv_heads * head_dim;
    const size_t kv_head_stride = head_dim;
    
    const int kv_batch_offset = batch_idx * kv_batch_stride + 
                                kv_head_idx * kv_head_stride;
    
    const int o_offset = batch_idx * q_batch_stride + 
                         global_tgt_idx * q_seq_stride + 
                         head_idx * q_head_stride;
    
    // 在线 Softmax 状态
    float max_score = -FLT_MAX;
    float sum_exp = 0.0f;
    
    // 动态分配输出累加器（避免固定大小假设）
    float* output_acc = new float[head_dim];
    float* q_vec = new float[head_dim];
    
    // 初始化
    for (int d = 0; d < head_dim; ++d) {
        output_acc[d] = 0.0f;
        q_vec[d] = TypeConverter<T>::to_float(Q[q_offset + d]);
    } 
    
    // 分块处理 Key 和 Value
    const int BLOCK_SIZE = 64;  // 固定分块大小
    
    for (int src_block_start = 0; src_block_start < src_seq_len; src_block_start += BLOCK_SIZE) {
        const int src_block_end = min(src_block_start + BLOCK_SIZE, src_seq_len);
        
        // 计算当前块的注意力分数
        float block_scores[64];  // 匹配 BLOCK_SIZE
        
        for (int i = 0; i < BLOCK_SIZE; ++i) {
            const int src_idx = src_block_start + i;
            
            if (src_idx >= src_block_end) {
                block_scores[i] = -FLT_MAX;
                continue;
            }
            
            // Causal masking
            if (is_causal && src_idx > global_tgt_idx) {
                block_scores[i] = -FLT_MAX;
                continue;
            }
            
            // 计算 Q·K^T
            const int k_offset = kv_batch_offset + src_idx * kv_seq_stride;
            
            float score = 0.0f;
            for (int d = 0; d < head_dim; ++d) {
                float k_val = TypeConverter<T>::to_float(K[k_offset + d]);
                score += q_vec[d] * k_val;
            }
            
            block_scores[i] = score * softmax_scale;
        }
        
        // 在线 Softmax 更新
        float prev_max = max_score;
        
        // 更新最大值
        for (int i = 0; i < BLOCK_SIZE; ++i) {
            if (block_scores[i] > -FLT_MAX) {
                max_score = fmaxf(max_score, block_scores[i]);
            }
        }
        
        // 重新缩放
        const float correction = expf(prev_max - max_score);
        sum_exp *= correction;
        
        for (int d = 0; d < head_dim; ++d) {
            output_acc[d] *= correction;
        }
        
        // 累加当前块的贡献
        for (int i = 0; i < BLOCK_SIZE; ++i) {
            const int src_idx = src_block_start + i;
            
            if (src_idx >= src_block_end || block_scores[i] == -FLT_MAX) {
                continue;
            }
            
            const float attn_weight = expf(block_scores[i] - max_score);
            sum_exp += attn_weight;
            
            const int v_offset = kv_batch_offset + src_idx * kv_seq_stride;
            
            for (int d = 0; d < head_dim; ++d) {
                float v_val = TypeConverter<T>::to_float(V[v_offset + d]);
                output_acc[d] += attn_weight * v_val;
            }
        }
    }
    
    // 最终归一化并写出
    const float inv_sum = (sum_exp > 1e-6f) ? (1.0f / sum_exp) : 0.0f;
    
    for (int d = 0; d < head_dim; ++d) {
        O[o_offset + d] = TypeConverter<T>::from_float(output_acc[d] * inv_sum);
    }
    
    // 清理
    delete[] output_acc;
    delete[] q_vec;
}

template <typename T>
void flashAttention(
    const std::vector<T>& h_q,
    const std::vector<T>& h_k,
    const std::vector<T>& h_v,
    std::vector<T>& h_o,
    int batch_size,
    int target_seq_len,
    int src_seq_len,
    int query_heads,
    int kv_heads,
    int head_dim,
    bool is_causal
) {
    const float softmax_scale = 1.0f / sqrtf(static_cast<float>(head_dim));
    
    const size_t q_size = batch_size * target_seq_len * query_heads * head_dim;
    const size_t kv_size = batch_size * src_seq_len * kv_heads * head_dim;
    const size_t o_size = q_size;
    
    T *d_q, *d_k, *d_v, *d_o;
    cudaMalloc(&d_q, q_size * sizeof(T));
    cudaMalloc(&d_k, kv_size * sizeof(T));
    cudaMalloc(&d_v, kv_size * sizeof(T));
    cudaMalloc(&d_o, o_size * sizeof(T));
    
    cudaMemcpy(d_q, h_q.data(), q_size * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_k, h_k.data(), kv_size * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, h_v.data(), kv_size * sizeof(T), cudaMemcpyHostToDevice);
    
    const int threads_per_block = 256;
    dim3 block_dim(threads_per_block);
    dim3 grid_dim(
        (target_seq_len + threads_per_block - 1) / threads_per_block,
        query_heads,
        batch_size
    );
    
    flash_attention_forward_kernel<T><<<grid_dim, block_dim>>>(
        d_q, d_k, d_v, d_o,
        batch_size, target_seq_len, src_seq_len,
        query_heads, kv_heads, head_dim,
        is_causal, softmax_scale
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }
    
    cudaMemcpy(h_o.data(), d_o, o_size * sizeof(T), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    
    cudaFree(d_q);
    cudaFree(d_k);
    cudaFree(d_v);
    cudaFree(d_o);
}

// ============================================================================
// 显式模板实例化
// ============================================================================

template int trace<int>(const std::vector<int>&, size_t, size_t);
template float trace<float>(const std::vector<float>&, size_t, size_t);

template void flashAttention<float>(
    const std::vector<float>&, const std::vector<float>&,
    const std::vector<float>&, std::vector<float>&,
    int, int, int, int, int, int, bool
);

template void flashAttention<half>(
    const std::vector<half>&, const std::vector<half>&,
    const std::vector<half>&, std::vector<half>&,
    int, int, int, int, int, int, bool
);

