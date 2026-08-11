#include <vector>
#include <musa_fp16.h>
#include <musa_runtime.h>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <type_traits>

#include "../tester/utils.h"

#define MUSA_CHECK(call) \
    do { \
        musaError_t err = call; \
        if (err != musaSuccess) { \
            std::cerr << "MUSA error at " << __FILE__ << ":" << __LINE__ \
                      << " - " << musaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// ==================== RMSNorm ====================
template <typename T>
__global__ void rmsNormKernel(const T* __restrict__ input,
                               const T* __restrict__ weight,
                               T* __restrict__ output,
                               size_t rows, size_t hidden_dim, float eps) {
    size_t row = blockIdx.x;
    if (row >= rows) return;
    
    __shared__ float shared_sum[256];
    size_t tid = threadIdx.x;
    float thread_sum = 0.0f;
    
    for (size_t i = tid; i < hidden_dim; i += blockDim.x) {
        size_t idx = row * hidden_dim + i;
        float val = static_cast<float>(input[idx]);
        thread_sum += val * val;
    }
    
    shared_sum[tid] = thread_sum;
    __syncthreads();
    
    for (size_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        __syncthreads();
    }
    
    float mean_square = shared_sum[0] / hidden_dim;
    float rms = rsqrtf(mean_square + eps);
    
    for (size_t i = tid; i < hidden_dim; i += blockDim.x) {
        size_t idx = row * hidden_dim + i;
        float val = static_cast<float>(input[idx]);
        float w = static_cast<float>(weight[i]);
        output[idx] = static_cast<T>(val * rms * w);
    }
}

template <typename T>
void rmsNorm(const std::vector<T>& h_input, const std::vector<T>& h_weight,
              std::vector<T>& h_output, size_t rows, size_t hidden_dim, float eps) {
    T *d_input, *d_weight, *d_output;
    size_t input_size = rows * hidden_dim * sizeof(T);
    size_t weight_size = hidden_dim * sizeof(T);
    
    MUSA_CHECK(musaMalloc(&d_input, input_size));
    MUSA_CHECK(musaMalloc(&d_weight, weight_size));
    MUSA_CHECK(musaMalloc(&d_output, input_size));
    
    MUSA_CHECK(musaMemcpy(d_input, h_input.data(), input_size, musaMemcpyHostToDevice));
    MUSA_CHECK(musaMemcpy(d_weight, h_weight.data(), weight_size, musaMemcpyHostToDevice));
    
    int threads_per_block = 256;
    int num_blocks = rows;
    rmsNormKernel<<<num_blocks, threads_per_block>>>(d_input, d_weight, d_output, rows, hidden_dim, eps);
    
    MUSA_CHECK(musaMemcpy(h_output.data(), d_output, input_size, musaMemcpyDeviceToHost));
    MUSA_CHECK(musaDeviceSynchronize());
    MUSA_CHECK(musaFree(d_input));
    MUSA_CHECK(musaFree(d_weight));
    MUSA_CHECK(musaFree(d_output));
}

// ==================== Flash Attention ====================
template <typename T>
void cleanup(T* d_q, T* d_k, T* d_v, T* d_o) {
    if (d_q) musaFree(d_q);
    if (d_k) musaFree(d_k);
    if (d_v) musaFree(d_v);
    if (d_o) musaFree(d_o);
}

// 升级后的 float kernel（双精度累加）
__global__ void flash_attention_kernel_float(
    const float* Q, const float* K, const float* V, float* O,
    int batch_size, int target_seq_len, int src_seq_len,
    int query_heads, int kv_heads, int head_dim,
    bool is_causal, float scale) {
    
    int b = blockIdx.x;
    int t = blockIdx.y;
    int h = blockIdx.z;
    int d = threadIdx.x;
    
    if (b >= batch_size || t >= target_seq_len || h >= query_heads || d >= head_dim) return;
    
    int kvh = h / (query_heads / kv_heads);
    int valid_len = is_causal ? min(t + 1, src_seq_len) : src_seq_len;
    
    size_t q_base = ((b * target_seq_len + t) * query_heads + h) * head_dim;
    size_t kv_base = b * src_seq_len * kv_heads * head_dim + kvh * head_dim;
    
    extern __shared__ float flash_attn_shared_float[];
    float* q_shared = flash_attn_shared_float;
    
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        q_shared[i] = Q[q_base + i];
    }
    __syncthreads();
    
    float max_score = -1e30f;
    for (int s = 0; s < valid_len; ++s) {
        float dot = 0.0f;
        size_t k_base = kv_base + s * kv_heads * head_dim;
        
        #pragma unroll 4
        for (int i = 0; i < head_dim; ++i) {
            dot += q_shared[i] * K[k_base + i];
        }
        float score = dot * scale;
        if (score > max_score) max_score = score;
    }
    
    double sum_exp = 0.0;
    double output = 0.0;
    
    for (int s = 0; s < valid_len; ++s) {
        float dot = 0.0f;
        size_t k_base = kv_base + s * kv_heads * head_dim;
        size_t v_base = kv_base + s * kv_heads * head_dim + d;
        
        #pragma unroll 4
        for (int i = 0; i < head_dim; ++i) {
            dot += q_shared[i] * K[k_base + i];
        }
        
        float score = dot * scale;
        float shifted = score - max_score;
        if (shifted > 10.0f) shifted = 10.0f;
        if (shifted < -20.0f) shifted = -20.0f;
        
        double exp_val = exp((double)shifted);
        sum_exp += exp_val;
        output += exp_val * (double)V[v_base];
    }
    
    if (sum_exp > 1e-12) {
        output = output / sum_exp;
    } else if (valid_len > 0) {
        output = 0.0;
        for (int s = 0; s < valid_len; ++s) {
            size_t v_base = kv_base + s * kv_heads * head_dim + d;
            output += (double)V[v_base];
        }
        output = output / valid_len;
    }
    
    O[q_base + d] = (float)output;
}

// half kernel（保持不变）
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
    
    extern __shared__ float flash_attn_shared_half[];
    float* q_shared = flash_attn_shared_half;
    
    float scale_f = __half2float(scale);
    
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        q_shared[i] = __half2float(Q[q_base + i]);
    }
    __syncthreads();
    
    float max_score = -1e4f;
    for (int s = 0; s < valid_len; ++s) {
        float dot = 0.0f;
        size_t k_base = kv_base + s * kv_heads * head_dim;
        
        #pragma unroll 4
        for (int i = 0; i < head_dim; ++i) {
            dot += q_shared[i] * __half2float(K[k_base + i]);
        }
        float score = dot * scale_f;
        if (score > max_score) max_score = score;
    }
    
    float sum_exp = 0.0f;
    float output_f = 0.0f;
    
    for (int s = 0; s < valid_len; ++s) {
        float dot = 0.0f;
        size_t k_base = kv_base + s * kv_heads * head_dim;
        size_t v_base = kv_base + s * kv_heads * head_dim + d;
        
        #pragma unroll 4
        for (int i = 0; i < head_dim; ++i) {
            dot += q_shared[i] * __half2float(K[k_base + i]);
        }
        
        float score = dot * scale_f;
        float shifted = score - max_score;
        if (shifted > 10.0f) shifted = 10.0f;
        if (shifted < -20.0f) shifted = -20.0f;
        
        float exp_val = expf(shifted);
        sum_exp += exp_val;
        output_f += exp_val * __half2float(V[v_base]);
    }
    
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

// 重载的启动函数
inline void launch_flash_attention(
    const float* d_q, const float* d_k, const float* d_v, float* d_o,
    int batch_size, int target_seq_len, int src_seq_len,
    int query_heads, int kv_heads, int head_dim, bool is_causal) {
    
    float scale = 1.0f / sqrtf(static_cast<float>(head_dim));
    dim3 grid(batch_size, target_seq_len, query_heads);
    int block_size = 256;
    if (head_dim < 256) block_size = ((head_dim + 31) / 32) * 32;
    size_t shared_mem_size = head_dim * sizeof(float);
    
    flash_attention_kernel_float<<<grid, block_size, shared_mem_size>>>(
        d_q, d_k, d_v, d_o,
        batch_size, target_seq_len, src_seq_len,
        query_heads, kv_heads, head_dim,
        is_causal, scale);
}

inline void launch_flash_attention(
    const __half* d_q, const __half* d_k, const __half* d_v, __half* d_o,
    int batch_size, int target_seq_len, int src_seq_len,
    int query_heads, int kv_heads, int head_dim, bool is_causal) {
    
    float scale_f = 1.0f / sqrtf(static_cast<float>(head_dim));
    if (scale_f > 5.0f) scale_f = 5.0f;
    __half scale = __float2half(scale_f);
    
    dim3 grid(batch_size, target_seq_len, query_heads);
    int block_size = 256;
    if (head_dim < 256) block_size = ((head_dim + 31) / 32) * 32;
    size_t shared_mem_size = head_dim * sizeof(float);
    
    flash_attention_kernel_half<<<grid, block_size, shared_mem_size>>>(
        d_q, d_k, d_v, d_o,
        batch_size, target_seq_len, src_seq_len,
        query_heads, kv_heads, head_dim,
        is_causal, scale);
}

template <typename T>
void flashAttention(const std::vector<T>& h_q, const std::vector<T>& h_k,
                    const std::vector<T>& h_v, std::vector<T>& h_o,
                    int batch_size, int target_seq_len, int src_seq_len,
                    int query_heads, int kv_heads, int head_dim, bool is_causal) {
    
    if (batch_size <= 0 || target_seq_len <= 0 || src_seq_len <= 0 || 
        query_heads <= 0 || kv_heads <= 0 || head_dim <= 0) return;
    if (query_heads % kv_heads != 0) return;
    
    size_t q_size = batch_size * target_seq_len * query_heads * head_dim;
    size_t kv_size = batch_size * src_seq_len * kv_heads * head_dim;
    size_t o_size = batch_size * target_seq_len * query_heads * head_dim;
    
    T *d_q = nullptr, *d_k = nullptr, *d_v = nullptr, *d_o = nullptr;
    MUSA_CHECK(musaMalloc(&d_q, q_size * sizeof(T)));
    MUSA_CHECK(musaMalloc(&d_k, kv_size * sizeof(T)));
    MUSA_CHECK(musaMalloc(&d_v, kv_size * sizeof(T)));
    MUSA_CHECK(musaMalloc(&d_o, o_size * sizeof(T)));
    
    MUSA_CHECK(musaMemcpy(d_q, h_q.data(), q_size * sizeof(T), musaMemcpyHostToDevice));
    MUSA_CHECK(musaMemcpy(d_k, h_k.data(), kv_size * sizeof(T), musaMemcpyHostToDevice));
    MUSA_CHECK(musaMemcpy(d_v, h_v.data(), kv_size * sizeof(T), musaMemcpyHostToDevice));
    
    launch_flash_attention(d_q, d_k, d_v, d_o,
                           batch_size, target_seq_len, src_seq_len,
                           query_heads, kv_heads, head_dim, is_causal);
    
    MUSA_CHECK(musaDeviceSynchronize());
    MUSA_CHECK(musaGetLastError());
    
    h_o.resize(o_size);
    MUSA_CHECK(musaMemcpy(h_o.data(), d_o, o_size * sizeof(T), musaMemcpyDeviceToHost));
    
    cleanup(d_q, d_k, d_v, d_o);
}

// ==================== 模板显式实例化 ====================
template void rmsNorm<float>(const std::vector<float>&, const std::vector<float>&,
  std::vector<float>&, size_t, size_t, float);
template void rmsNorm<half>(const std::vector<half>&, const std::vector<half>&,
  std::vector<half>&, size_t, size_t, float);
template void flashAttention<float>(const std::vector<float>&, const std::vector<float>&,
  const std::vector<float>&, std::vector<float>&,
  int, int, int, int, int, int, bool);
template void flashAttention<half>(const std::vector<half>&, const std::vector<half>&,
  const std::vector<half>&, std::vector<half>&,
  int, int, int, int, int, int, bool);
