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
__global__ void trace_kernel(const T* d_input, T* d_partial_sums, 
                             size_t rows, size_t cols, size_t min_dim) {
    extern __shared__ char shared_mem[];
    T* sdata = reinterpret_cast<T*>(shared_mem);
    
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    T val = (idx < min_dim) ? d_input[idx * cols + idx] : T(0);
    sdata[tid] = val;
    __syncthreads();
    
    // 并行 reduction 
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
    
    // 矩阵维度小，cpu 直接算
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
 struct TypeConverter;
 
 template <>
 struct TypeConverter<float> {
     __device__ __forceinline__ static float to_float(float v) {
         return v;
     }
     __device__ __forceinline__ static float from_float(float v) {
         return v;
     }
 };
 
 template <>
 struct TypeConverter<half> {
     __device__ __forceinline__ static float to_float(half v) {
         return __half2float(v);
     }
     __device__ __forceinline__ static half from_float(float v) {
         return __float2half(v);
     }
 };

 // 特殊 head_dim 模板特化
 template <typename T, int HEAD_DIM>
 __global__ void flash_attention_forward_kernel_fast(
     const T* __restrict__ Q,
     const T* __restrict__ K,
     const T* __restrict__ V,
     T* __restrict__ O,
     int batch_size,
     int tgt_seq_len,
     int src_seq_len,
     int query_heads,
     int kv_heads,
     bool is_causal,
     float softmax_scale
 ) {
     // 每个线程处理一个 Q 众多头中的一个头中的 token
     int tgt_idx   = blockIdx.x * blockDim.x + threadIdx.x;
     int head_idx  = blockIdx.y;
     int batch_idx = blockIdx.z;
 
     if (tgt_idx >= tgt_seq_len) return;
 
     int kv_head_idx = (head_idx * kv_heads) / query_heads;
    

     // 计算内存偏移量
     int q_offset =
         batch_idx * tgt_seq_len * query_heads * HEAD_DIM +
         tgt_idx   * query_heads * HEAD_DIM +
         head_idx  * HEAD_DIM;
 
     int kv_batch_offset =
         batch_idx * src_seq_len * kv_heads * HEAD_DIM +
         kv_head_idx * HEAD_DIM;
    
     // Q[i] 向量
     float q_vec[HEAD_DIM]; 
     float out[HEAD_DIM];
 
 #pragma unroll
     for (int d = 0; d < HEAD_DIM; ++d) {
         q_vec[d] = TypeConverter<T>::to_float(Q[q_offset + d]);
         out[d]   = 0.0f;
     }


    // Online Softmax + Attention
     float max_score = -FLT_MAX;
     float sum_exp   = 0.0f;    // Softmax 分母
 
     for (int s = 0; s < src_seq_len; ++s) {
         if (is_causal && s > tgt_idx) continue;
 
         int k_offset = kv_batch_offset + s * kv_heads * HEAD_DIM;
 
         float score = 0.0f;

         // 计算 Q * K ^ T
 #pragma unroll
         for (int d = 0; d < HEAD_DIM; ++d) {
             score += q_vec[d] *
                      TypeConverter<T>::to_float(K[k_offset + d]);
         }
         score *= softmax_scale;
         

         // Online Softmax 更新
         float prev_max = max_score;
         max_score = fmaxf(max_score, score);
 
         float scale = expf(prev_max - max_score);
         sum_exp *= scale;
 #pragma unroll
         for (int d = 0; d < HEAD_DIM; ++d) out[d] *= scale;
 
         float w = expf(score - max_score);
         sum_exp += w;
 
         int v_offset = kv_batch_offset + s * kv_heads * HEAD_DIM;
 #pragma unroll
         for (int d = 0; d < HEAD_DIM; ++d) {
             out[d] += w * TypeConverter<T>::to_float(V[v_offset + d]);
         }
     }
 
     float inv = (sum_exp > 1e-6f) ? 1.0f / sum_exp : 0.0f;
 
 #pragma unroll
     for (int d = 0; d < HEAD_DIM; ++d) {
         O[q_offset + d] =
             TypeConverter<T>::from_float(out[d] * inv);
     }
 }


 // 支持任意 head_dim
 template <typename T>
 __global__ void flash_attention_forward_kernel_generic(
     const T* __restrict__ Q,
     const T* __restrict__ K,
     const T* __restrict__ V,
     T* __restrict__ O,
     int batch_size,
     int tgt_seq_len,
     int src_seq_len,
     int query_heads,
     int kv_heads,
     int head_dim,
     bool is_causal,
     float softmax_scale
 ) {
     extern __shared__ float smem[];
 
     int tid      = threadIdx.x;
     int tgt_idx  = blockIdx.x * blockDim.x + tid;
     int head_idx = blockIdx.y;
     int batch_idx = blockIdx.z;
 
     if (tgt_idx >= tgt_seq_len) return;
 
     float* q_vec = smem + tid * head_dim;
     float* out   = q_vec + blockDim.x * head_dim;
 
     int kv_head_idx = (head_idx * kv_heads) / query_heads;
 
     int q_offset =
         batch_idx * tgt_seq_len * query_heads * head_dim +
         tgt_idx   * query_heads * head_dim +
         head_idx  * head_dim;
 
     int kv_batch_offset =
         batch_idx * src_seq_len * kv_heads * head_dim +
         kv_head_idx * head_dim;

     // 加载到 shared memory
     for (int d = 0; d < head_dim; ++d) {
         q_vec[d] = TypeConverter<T>::to_float(Q[q_offset + d]);
         out[d]   = 0.0f;
     }
 
     float max_score = -FLT_MAX;
     float sum_exp   = 0.0f;
 
     for (int s = 0; s < src_seq_len; ++s) {
         if (is_causal && s > tgt_idx) continue;
 
         int k_offset = kv_batch_offset + s * kv_heads * head_dim;
 
         float score = 0.0f;
         for (int d = 0; d < head_dim; ++d) {
             score += q_vec[d] *
                      TypeConverter<T>::to_float(K[k_offset + d]);
         }
         score *= softmax_scale;
 
         float prev_max = max_score;
         max_score = fmaxf(max_score, score);
 
         float scale = expf(prev_max - max_score);
         sum_exp *= scale;
         for (int d = 0; d < head_dim; ++d) out[d] *= scale;
 
         float w = expf(score - max_score);
         sum_exp += w;
 
         int v_offset = kv_batch_offset + s * kv_heads * head_dim;
         for (int d = 0; d < head_dim; ++d) {
             out[d] += w * TypeConverter<T>::to_float(V[v_offset + d]);
         }
     }
 
     float inv = (sum_exp > 1e-6f) ? 1.0f / sum_exp : 0.0f;
 
     for (int d = 0; d < head_dim; ++d) {
         O[q_offset + d] =
             TypeConverter<T>::from_float(out[d] * inv);
     }
 }
 
 template <typename T>
 void flashAttention(
     const std::vector<T>& h_q,
     const std::vector<T>& h_k,
     const std::vector<T>& h_v,
     std::vector<T>& h_o,
     int batch_size,
     int tgt_seq_len,
     int src_seq_len,
     int query_heads,
     int kv_heads,
     int head_dim,
     bool is_causal
 ) {
     float softmax_scale = 1.0f / sqrtf((float)head_dim);
 
     size_t q_size  = batch_size * tgt_seq_len * query_heads * head_dim;
     size_t kv_size = batch_size * src_seq_len * kv_heads * head_dim;
 
     T *d_q, *d_k, *d_v, *d_o;
     cudaMalloc(&d_q, q_size * sizeof(T));
     cudaMalloc(&d_k, kv_size * sizeof(T));
     cudaMalloc(&d_v, kv_size * sizeof(T));
     cudaMalloc(&d_o, q_size * sizeof(T));
 
     cudaMemcpy(d_q, h_q.data(), q_size * sizeof(T), cudaMemcpyHostToDevice);
     cudaMemcpy(d_k, h_k.data(), kv_size * sizeof(T), cudaMemcpyHostToDevice);
     cudaMemcpy(d_v, h_v.data(), kv_size * sizeof(T), cudaMemcpyHostToDevice);
 
     dim3 block(128);
     dim3 grid(
         (tgt_seq_len + block.x - 1) / block.x,
         query_heads,
         batch_size
     );

     // 选择 kernel 路径
     bool use_fast =
         (head_dim == 16 || head_dim == 32 ||
          head_dim == 64 || head_dim == 128);
 
     if (use_fast) {
        // 特殊 head_dim
         if (head_dim == 16)
             flash_attention_forward_kernel_fast<T,16><<<grid,block>>>(
                 d_q,d_k,d_v,d_o,
                 batch_size,tgt_seq_len,src_seq_len,
                 query_heads,kv_heads,is_causal,softmax_scale);
         else if (head_dim == 32)
             flash_attention_forward_kernel_fast<T,32><<<grid,block>>>(
                d_q,d_k,d_v,d_o,
                batch_size,tgt_seq_len,src_seq_len,
                query_heads,kv_heads,is_causal,softmax_scale
             );
         else if (head_dim == 64)
             flash_attention_forward_kernel_fast<T,64><<<grid,block>>>(
                d_q,d_k,d_v,d_o,
                batch_size,tgt_seq_len,src_seq_len,
                query_heads,kv_heads,is_causal,softmax_scale
             );
         else if (head_dim == 128)
             flash_attention_forward_kernel_fast<T,128><<<grid,block>>>(
                d_q,d_k,d_v,d_o,
                batch_size,tgt_seq_len,src_seq_len,
                query_heads,kv_heads,is_causal,softmax_scale
             );
     } else {

        // 任意 head_dim
         size_t smem_size =
             2 * block.x * head_dim * sizeof(float);
 
         flash_attention_forward_kernel_generic<T>
             <<<grid, block, smem_size>>>(
                 d_q, d_k, d_v, d_o,
                 batch_size, tgt_seq_len, src_seq_len,
                 query_heads, kv_heads,
                 head_dim, is_causal, softmax_scale);
     }
 
     cudaMemcpy(h_o.data(), d_o, q_size * sizeof(T),
                cudaMemcpyDeviceToHost);
 
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
