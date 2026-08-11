#include <vector>
#include <cmath>
#include <iostream>
#include <algorithm>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

const float INFINITY_F = 1e30f;

/**
 * @brief Computes RMSNorm over the last dimension of a 2D tensor.
 *
 * The input is a row-major matrix with shape [rows, hidden_dim]. For each row
 * i and column j:
 *
 *   output[i, j] = input[i, j] * rsqrt(mean(input[i, :]^2) + eps) * weight[j]
 *
 * The output vector is preallocated with rows * hidden_dim elements.
 *
 * @tparam T Data type of input, weight, and output tensors.
 * @param[in] h_input Flattened input matrix of shape [rows, hidden_dim].
 * @param[in] h_weight Per-column scale vector of shape [hidden_dim].
 * @param[out] h_output Flattened output matrix of shape [rows, hidden_dim].
 * @param[in] rows Number of rows/tokens.
 * @param[in] hidden_dim Size of the normalized dimension.
 * @param[in] eps Numerical stability epsilon.
 */
template <typename T>
__global__ void rmsNormKernel(const T* __restrict__ input,
                               const T* __restrict__ weight,
                               T* __restrict__ output,
                               size_t rows,
                               size_t hidden_dim,
                               float eps) {
    // Each block processes one row
    size_t row = blockIdx.x;
    if (row >= rows) return;
    
    // Shared memory for reduction
    __shared__ float shared_sum[256];
    size_t tid = threadIdx.x;
    
    float thread_sum = 0.0f;
    
    // Step 1: Compute sum of squares for this row
    for (size_t i = tid; i < hidden_dim; i += blockDim.x) {
        size_t idx = row * hidden_dim + i;
        float val = static_cast<float>(input[idx]);
        thread_sum += val * val;
    }
    
    // Store partial sum to shared memory
    shared_sum[tid] = thread_sum;
    __syncthreads();
    
    // Parallel reduction in shared memory
    for (size_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        __syncthreads();
    }
    
    // Compute RMS normalization factor
    float mean_square = shared_sum[0] / hidden_dim;
    float rms = rsqrtf(mean_square + eps);
    
    // Step 2: Apply normalization and scaling
    for (size_t i = tid; i < hidden_dim; i += blockDim.x) {
        size_t idx = row * hidden_dim + i;
        float val = static_cast<float>(input[idx]);
        float w = static_cast<float>(weight[i]);
        output[idx] = static_cast<T>(val * rms * w);
    }
}

template <typename T>
void rmsNorm(const std::vector<T>& h_input, const std::vector<T>& h_weight,
              std::vector<T>& h_output, size_t rows, size_t hidden_dim,
              float eps) {
    // Allocate device memory
    T *d_input = nullptr, *d_weight = nullptr, *d_output = nullptr;
    size_t input_size = rows * hidden_dim * sizeof(T);
    size_t weight_size = hidden_dim * sizeof(T);
    
    cudaError_t err;
    
    err = cudaMalloc(&d_input, input_size);
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate d_input: " << cudaGetErrorString(err) << std::endl;
        return;
    }
    err = cudaMalloc(&d_weight, weight_size);
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate d_weight: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_input);
        return;
    }
    err = cudaMalloc(&d_output, input_size);
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate d_output: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_input);
        cudaFree(d_weight);
        return;
    }
    
    // Copy data to device
    cudaMemcpy(d_input, h_input.data(), input_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight, h_weight.data(), weight_size, cudaMemcpyHostToDevice);
    
    // Configure kernel launch parameters
    int threads_per_block = 256;
    int num_blocks = rows;
    
    // Launch kernel
    rmsNormKernel<T><<<num_blocks, threads_per_block>>>(
        d_input, d_weight, d_output, rows, hidden_dim, eps);
    
    // Synchronize and copy result back
    cudaDeviceSynchronize();
    cudaMemcpy(h_output.data(), d_output, input_size, cudaMemcpyDeviceToHost);
    
    // Free memory
    cudaFree(d_input);
    cudaFree(d_weight);
    cudaFree(d_output);
}

/**
 * @brief Helper to clean up device memory.
 */
template <typename T>
void cleanup(T* d_q, T* d_k, T* d_v, T* d_o) {
    if (d_q) cudaFree(d_q);
    if (d_k) cudaFree(d_k);
    if (d_v) cudaFree(d_v);
    if (d_o) cudaFree(d_o);
}

// float 版本的 flash attention kernel
__global__ void flash_attention_kernel_float(
    const float* Q, const float* K, const float* V, float* O,
    int batch_size, int target_seq_len, int src_seq_len,
    int query_heads, int kv_heads, int head_dim,
    bool is_causal, float scale) {
    
    // Each thread handles one output position
    int b = blockIdx.x;
    int t = blockIdx.y;
    int h = blockIdx.z;
    int d = threadIdx.x;
    
    if (b >= batch_size || t >= target_seq_len || h >= query_heads || d >= head_dim) return;
    
    int kvh = h / (query_heads / kv_heads);
    int valid_len = is_causal ? min(t + 1, src_seq_len) : src_seq_len;
    
    size_t q_base = ((b * target_seq_len + t) * query_heads + h) * head_dim;
    size_t kv_base = b * src_seq_len * kv_heads * head_dim + kvh * head_dim;
    
    // Use shared memory to store query vector
    extern __shared__ float flash_attn_shared_float[];
    float* q_shared = flash_attn_shared_float;
    
    // Cooperative loading of query vector into shared memory
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        q_shared[i] = Q[q_base + i];
    }
    __syncthreads();
    
    // Compute max score
    float max_score = -1e10f;
    for (int s = 0; s < valid_len; ++s) {
        float dot = 0.0f;
        size_t k_base = kv_base + s * kv_heads * head_dim;
        
        // Use query vector from shared memory
        #pragma unroll(4)
        for (int i = 0; i < head_dim; ++i) {
            dot += q_shared[i] * K[k_base + i];
        }
        
        float score = dot * scale;
        if (score > max_score) max_score = score;
    }
    
    // Compute output
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
    
    // Normalize
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

// half 版本的 flash attention kernel
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
    
    // Use shared memory to store query vector
    extern __shared__ float flash_attn_shared_half[];
    float* q_shared = flash_attn_shared_half;
    
    float scale_f = __half2float(scale);
    
    // Cooperative loading of query vector into shared memory
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        q_shared[i] = __half2float(Q[q_base + i]);
    }
    __syncthreads();
    
    // Compute max score
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
    
    // Compute softmax and output
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
        
        // Clamp for stability
        if (shifted > 10.0f) shifted = 10.0f;
        if (shifted < -20.0f) shifted = -20.0f;
        
        float exp_val = expf(shifted);
        sum_exp += exp_val;
        output_f += exp_val * __half2float(V[v_base]);
    }
    
    // Normalize
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
 * @brief Computes flash attention for given query, key, and value tensors.
 * 
 * @tparam T Data type (float or __half) for input/output tensors
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
    
    // Basic validation
    if (batch_size <= 0 || target_seq_len <= 0 || src_seq_len <= 0 || 
        query_heads <= 0 || kv_heads <= 0 || head_dim <= 0) {
        return;
    }
    
    if (query_heads % kv_heads != 0) return;
    
    // Calculate sizes
    size_t q_size = batch_size * target_seq_len * query_heads * head_dim;
    size_t kv_size = batch_size * src_seq_len * kv_heads * head_dim;
    size_t o_size = batch_size * target_seq_len * query_heads * head_dim;
    
    // Allocate device memory
    T *d_q = nullptr, *d_k = nullptr, *d_v = nullptr, *d_o = nullptr;
    
    cudaMalloc(&d_q, q_size * sizeof(T));
    cudaMalloc(&d_k, kv_size * sizeof(T));
    cudaMalloc(&d_v, kv_size * sizeof(T));
    cudaMalloc(&d_o, o_size * sizeof(T));
    
    // Copy data to device
    cudaMemcpy(d_q, h_q.data(), q_size * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_k, h_k.data(), kv_size * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, h_v.data(), kv_size * sizeof(T), cudaMemcpyHostToDevice);
    
    // Scale factor
    T scale;
    if constexpr (std::is_same_v<T, __half>) {
        float head_dim_f = static_cast<float>(head_dim);
        float scale_f = 1.0f / sqrtf(head_dim_f);
        if (scale_f > 5.0f) scale_f = 5.0f;
        scale = __float2half(scale_f);
    } else {
        scale = T(1.0 / sqrt(static_cast<double>(head_dim)));
    }
    
    // Launch configuration
    dim3 grid(batch_size, target_seq_len, query_heads);
    
    // Optimize block size: ensure multiple of 32 (warp size)
    int block_size = 256;
    if (head_dim < 256) {
        block_size = ((head_dim + 31) / 32) * 32; // Round up to multiple of 32
    }
    
    // Shared memory size
    size_t shared_mem_size = head_dim * sizeof(float);
    
    // Launch corresponding kernel
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
    
    // Synchronize and error check
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
    }
    
    // Copy result back
    h_o.resize(o_size);
    cudaMemcpy(h_o.data(), d_o, o_size * sizeof(T), cudaMemcpyDeviceToHost);
    
    // Cleanup
    cleanup(d_q, d_k, d_v, d_o);
}

// *********************************************************************
// Explicit Template Instantiations (REQUIRED FOR LINKING WITH TESTER.O)
// DO NOT MODIFY THIS SECTION
// *********************************************************************
template void rmsNorm<float>(const std::vector<float>&, const std::vector<float>&,
  std::vector<float>&, size_t, size_t, float);
template void rmsNorm<__half>(const std::vector<__half>&, const std::vector<__half>&,
  std::vector<__half>&, size_t, size_t, float);
template void flashAttention<float>(const std::vector<float>&, const std::vector<float>&,
  const std::vector<float>&, std::vector<float>&,
  int, int, int, int, int, int, bool);
template void flashAttention<__half>(const std::vector<__half>&, const std::vector<__half>&,
  const std::vector<__half>&, std::vector<__half>&,
  int, int, int, int, int, int, bool);
