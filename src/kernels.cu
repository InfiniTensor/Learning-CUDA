#include <vector>
#include <cuda_fp16.h>
#include <iostream>

// ================== 清理函数 ==================
template <typename T>
void cleanup(T* d_q, T* d_k, T* d_v, T* d_o) {
    if (d_q) cudaFree(d_q);
    if (d_k) cudaFree(d_k);
    if (d_v) cudaFree(d_v);
    if (d_o) cudaFree(d_o);
}

// ================== 辅助函数 ==================

// Warp级别归约求最大值
__device__ inline float warpReduceMax(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    }
    return val;
}

// Warp级别归约求和
__device__ inline float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

// Block级别归约求最大值
__device__ inline float blockReduceMax(float val) {
    __shared__ float shared[32];
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;
    
    val = warpReduceMax(val);
    
    if (lane == 0) shared[wid] = val;
    __syncthreads();
    
    val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : -1e30f;
    if (wid == 0) val = warpReduceMax(val);
    return val;
}

// Block级别归约求和
__device__ inline float blockReduceSum(float val) {
    __shared__ float shared[32];
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;
    
    val = warpReduceSum(val);
    
    if (lane == 0) shared[wid] = val;
    __syncthreads();
    
    val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : 0.0f;
    if (wid == 0) val = warpReduceSum(val);
    return val;
}

// ================== 优化后的kernel函数 ==================

// float版本的优化kernel
__global__ void flash_attention_kernel_float_optimized(
    const float* __restrict__ Q, const float* __restrict__ K, 
    const float* __restrict__ V, float* __restrict__ O,
    int batch_size, int target_seq_len, int src_seq_len,
    int query_heads, int kv_heads, int head_dim,
    bool is_causal, float scale) {
    
    // 使用2D block: blockDim.x处理head_dim, blockDim.y处理序列
    int b = blockIdx.x;
    int h = blockIdx.y;
    int t = blockIdx.z;
    
    if (b >= batch_size || t >= target_seq_len || h >= query_heads) return;
    
    int kvh = h / (query_heads / kv_heads);
    
    // 在device端计算valid_len
    int valid_len = src_seq_len;
    if (is_causal) {
        if (t + 1 < src_seq_len) {
            valid_len = t + 1;
        } else {
            valid_len = src_seq_len;
        }
    }
    
    // 计算基础偏移
    size_t q_base = ((b * target_seq_len + t) * query_heads + h) * head_dim;
    size_t kv_base = (b * src_seq_len * kv_heads + kvh) * head_dim;
    
    // 使用共享内存存储查询向量和部分结果
    extern __shared__ float shared_mem[];
    float* q_shared = shared_mem;
    
    // 协作加载查询向量到共享内存
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        q_shared[i] = Q[q_base + i];
    }
    __syncthreads();
    
    // 第一步：计算最大分数（使用并行归约）
    float max_score = -1e30f;
    
    // 每个线程处理多个序列位置
    for (int s_start = 0; s_start < valid_len; s_start += blockDim.y) {
        int s = s_start + threadIdx.y;
        if (s < valid_len) {
            size_t k_base = kv_base + s * kv_heads * head_dim;
            
            // 计算点积
            float dot = 0.0f;
            #pragma unroll(4)
            for (int i = 0; i < head_dim; i++) {
                dot += q_shared[i] * K[k_base + i];
            }
            
            float score = dot * scale;
            if (score > max_score) max_score = score;
        }
    }
    
    // Block内归约求最大分数
    __shared__ float shared_max;
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        shared_max = max_score;
    }
    __syncthreads();
    max_score = shared_max;
    
    // 第二步：计算softmax和输出
    __shared__ float shared_sum_exp;
    __shared__ float shared_output;
    
    float thread_sum_exp = 0.0f;
    float thread_output = 0.0f;
    int d = threadIdx.x;
    
    for (int s_start = 0; s_start < valid_len; s_start += blockDim.y) {
        int s = s_start + threadIdx.y;
        if (s < valid_len) {
            size_t k_base = kv_base + s * kv_heads * head_dim;
            size_t v_base = kv_base + s * kv_heads * head_dim + d;
            
            // 计算点积
            float dot = 0.0f;
            #pragma unroll(4)
            for (int i = 0; i < head_dim; i++) {
                dot += q_shared[i] * K[k_base + i];
            }
            
            float score = dot * scale;
            float shifted = score - max_score;
            float exp_val = expf(shifted);
            
            // 累加exp值和加权值
            thread_sum_exp += exp_val;
            thread_output += exp_val * V[v_base];
        }
    }
    
    // 使用原子操作累加到共享内存
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        atomicAdd(&shared_sum_exp, thread_sum_exp);
        atomicAdd(&shared_output, thread_output);
    }
    __syncthreads();
    
    // 归一化
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        float sum_exp = shared_sum_exp;
        float output = shared_output;
        
        if (sum_exp > 1e-12f) {
            output = output / sum_exp;
        } else if (valid_len > 0) {
            // 退回到平均值
            output = 0.0f;
            for (int s = 0; s < valid_len; ++s) {
                size_t v_base = kv_base + s * kv_heads * head_dim + d;
                output += V[v_base];
            }
            output = output / valid_len;
        }
        
        // 写入输出
        if (d < head_dim) {
            O[q_base + d] = output;
        }
    }
}

// half版本的优化kernel
__global__ void flash_attention_kernel_half_optimized(
    const __half* __restrict__ Q, const __half* __restrict__ K, 
    const __half* __restrict__ V, __half* __restrict__ O,
    int batch_size, int target_seq_len, int src_seq_len,
    int query_heads, int kv_heads, int head_dim,
    bool is_causal, __half scale) {
    
    int b = blockIdx.x;
    int h = blockIdx.y;
    int t = blockIdx.z;
    
    if (b >= batch_size || t >= target_seq_len || h >= query_heads) return;
    
    int kvh = h / (query_heads / kv_heads);
    
    // 在device端计算valid_len
    int valid_len = src_seq_len;
    if (is_causal) {
        if (t + 1 < src_seq_len) {
            valid_len = t + 1;
        } else {
            valid_len = src_seq_len;
        }
    }
    
    size_t q_base = ((b * target_seq_len + t) * query_heads + h) * head_dim;
    size_t kv_base = (b * src_seq_len * kv_heads + kvh) * head_dim;
    
    // 使用共享内存
    extern __shared__ float shared_mem[];
    float* q_shared = shared_mem;
    
    float scale_f = __half2float(scale);
    
    // 协作加载查询向量到共享内存（转换为float）
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        q_shared[i] = __half2float(Q[q_base + i]);
    }
    __syncthreads();
    
    // 第一步：计算最大分数
    float max_score = -1e30f;
    
    for (int s_start = 0; s_start < valid_len; s_start += blockDim.y) {
        int s = s_start + threadIdx.y;
        if (s < valid_len) {
            size_t k_base = kv_base + s * kv_heads * head_dim;
            
            float dot = 0.0f;
            #pragma unroll(4)
            for (int i = 0; i < head_dim; i++) {
                dot += q_shared[i] * __half2float(K[k_base + i]);
            }
            
            float score = dot * scale_f;
            if (score > max_score) max_score = score;
        }
    }
    
    // 归约求最大值
    __shared__ float shared_max;
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        shared_max = max_score;
    }
    __syncthreads();
    max_score = shared_max;
    
    // 第二步：计算softmax和输出
    __shared__ float shared_sum_exp;
    __shared__ float shared_output_f;
    
    float thread_sum_exp = 0.0f;
    float thread_output_f = 0.0f;
    int d = threadIdx.x;
    
    for (int s_start = 0; s_start < valid_len; s_start += blockDim.y) {
        int s = s_start + threadIdx.y;
        if (s < valid_len) {
            size_t k_base = kv_base + s * kv_heads * head_dim;
            size_t v_base = kv_base + s * kv_heads * head_dim + d;
            
            float dot = 0.0f;
            #pragma unroll(4)
            for (int i = 0; i < head_dim; i++) {
                dot += q_shared[i] * __half2float(K[k_base + i]);
            }
            
            float score = dot * scale_f;
            float shifted = score - max_score;
            // 限制范围确保稳定性
            if (shifted > 10.0f) shifted = 10.0f;
            if (shifted < -20.0f) shifted = -20.0f;
            
            float exp_val = expf(shifted);
            thread_sum_exp += exp_val;
            thread_output_f += exp_val * __half2float(V[v_base]);
        }
    }
    
    // 使用原子操作累加
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        atomicAdd(&shared_sum_exp, thread_sum_exp);
        atomicAdd(&shared_output_f, thread_output_f);
    }
    __syncthreads();
    
    // 归一化
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        float sum_exp = shared_sum_exp;
        float output_f = shared_output_f;
        
        if (sum_exp > 1e-12f) {
            output_f = output_f / sum_exp;
        } else if (valid_len > 0) {
            output_f = 0.0f;
            for (int s = 0; s < valid_len; ++s) {
                size_t v_base = kv_base + s * kv_heads * head_dim + d;
                output_f += __half2float(V[v_base]);
            }
            output_f = output_f / valid_len;
        }
        
        if (d < head_dim) {
            O[q_base + d] = __float2half(output_f);
        }
    }
}

// ================== Flash Attention主函数 ==================
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
    
    // 优化后的kernel配置
    // 使用2D block: x维度处理head_dim, y维度处理序列
    dim3 grid(batch_size, query_heads, target_seq_len);
    
    // 计算block大小 - 简化版本，不在host端使用valid_len
    int block_x = 256;
    if (head_dim < block_x) block_x = head_dim;
    int block_y = 4;  // 每个block处理4个序列位置
    
    dim3 block(block_x, block_y);
    
    // 计算共享内存大小
    size_t shared_mem_size = head_dim * sizeof(float);
    
    if constexpr (std::is_same_v<T, float>) {
        flash_attention_kernel_float_optimized<<<grid, block, shared_mem_size>>>(
            reinterpret_cast<const float*>(d_q),
            reinterpret_cast<const float*>(d_k),
            reinterpret_cast<const float*>(d_v),
            reinterpret_cast<float*>(d_o),
            batch_size, target_seq_len, src_seq_len,
            query_heads, kv_heads, head_dim,
            is_causal, scale);
    } else {
        flash_attention_kernel_half_optimized<<<grid, block, shared_mem_size>>>(
            reinterpret_cast<const __half*>(d_q),
            reinterpret_cast<const __half*>(d_k),
            reinterpret_cast<const __half*>(d_v),
            reinterpret_cast<__half*>(d_o),
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

// ================== Trace函数（保持原样） ==================
template <typename T>
T trace(const std::vector<T>& h_input, size_t rows, size_t cols) {
    if (h_input.empty() || rows == 0 || cols == 0) return T(0);
    size_t n = rows;
    if (cols < n) n = cols;
    T sum = T(0);
    for (size_t i = 0; i < n; ++i) sum += h_input[i * cols + i];
    return sum;
}

// ================== 显式实例化 ==================
template int trace<int>(const std::vector<int>&, size_t, size_t);
template float trace<float>(const std::vector<float>&, size_t, size_t);
template void flashAttention<float>(const std::vector<float>&, const std::vector<float>&,
                                    const std::vector<float>&, std::vector<float>&,
                                    int, int, int, int, int, int, bool);
template void flashAttention<__half>(const std::vector<__half>&, const std::vector<__half>&,
                                     const std::vector<__half>&, std::vector<__half>&,
                                     int, int, int, int, int, int, bool);
