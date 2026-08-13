#include <vector>
#include <cuda_fp16.h>

#include "../tester/utils.h"

#include <stdexcept>
#include <limits>
#include <cmath>
#include <algorithm>
#include <iostream>
#define tileHeight 4

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
__device__ __forceinline__ T castFloatToT(float v) {
    return static_cast<T>(v);
}

// float 特化
template <>
__device__ __forceinline__ float castFloatToT<float>(float v) {
    return v;
}

// half 特化
template <>
__device__ __forceinline__ half castFloatToT<half>(float v) {
    return __float2half(v);
}

// 转float模板
template<typename T>
__device__ __forceinline__ float castToFloat(T v);

template <>
__device__ __forceinline__ float castToFloat<float>(float v) { return v; }

template <>
__device__ __forceinline__ float castToFloat<half>(half v) { return __half2float(v); }


extern __shared__ char smem_raw[];
 template <typename T>
 __global__ void RMSNormKernel(T* d_input, T* d_weight, T* d_output, size_t hidden_dim, size_t iterRounds, float eps, float* d_res){
    
  T *smem = reinterpret_cast<T*>(smem_raw);
    //load data from global mem to share mem
    int smemIdx = threadIdx.x;
    int gmemIdx = blockIdx.x * hidden_dim + smemIdx;
    for(int r = 0; r < iterRounds - 1; r++){
      smem[smemIdx] = d_input[gmemIdx];
      smemIdx += blockDim.x;
      gmemIdx += blockDim.x;
    }
    if(smemIdx < hidden_dim){
      smem[smemIdx] = d_input[gmemIdx];
    }
    __syncthreads();

    //let the first thread compute the sum, mean, rsqrt 
    if(threadIdx.x == 0){
      float tempRes = 0;
      for(int i = 0; i < hidden_dim; i++){
        float v = castToFloat<T>(smem[i]);
        tempRes += v * v;
      }
      tempRes /= (hidden_dim * 1.0f);
      tempRes = rsqrt(tempRes + eps);
      *d_res = tempRes;
    }
    __syncthreads();

    //让线程块的所有线程进行权重乘法，并写回全局内存
    float res = *d_res;
    smemIdx = threadIdx.x;
    gmemIdx = blockIdx.x * hidden_dim + smemIdx;

    for(int r = 0; r < iterRounds - 1; r++){
      float v = castToFloat<T>(smem[smemIdx]) * res * castToFloat<T>(d_weight[smemIdx]);
      d_output[gmemIdx] = castFloatToT<T>(v);
      smemIdx += blockDim.x;
      gmemIdx += blockDim.x;
    }
    if(smemIdx < hidden_dim){
      float v = castToFloat<T>(smem[smemIdx]) * res * castToFloat<T>(d_weight[smemIdx]);
      d_output[gmemIdx] = castFloatToT<T>(v);
    }
}



template <typename T>
void rmsNorm(const std::vector<T>& h_input, const std::vector<T>& h_weight,
              std::vector<T>& h_output, size_t rows, size_t hidden_dim,
              float eps) {
  // TODO: Implement the rmsNorm function

  T* d_input = nullptr;
  cudaMalloc(&d_input, rows * hidden_dim * sizeof(T));
  cudaMemcpyAsync(d_input, h_input.data(), rows * hidden_dim * sizeof(T), cudaMemcpyHostToDevice);

  int blksPerGrid = (int)rows;
  int threadsPerBlk = std::min(1024, (int)hidden_dim);

  T* d_weight = nullptr;
  cudaMalloc(&d_weight, hidden_dim * sizeof(T));
  cudaMemcpyAsync(d_weight, h_weight.data(), hidden_dim * sizeof(T), cudaMemcpyHostToDevice);

  T* d_output = nullptr;
  cudaMalloc(&d_output, rows * hidden_dim * sizeof(T));

  float* d_res = nullptr;
  cudaMalloc(&d_res, sizeof(float));

  size_t iterRounds = (hidden_dim - 1) / threadsPerBlk + 1;

  RMSNormKernel<T><<<blksPerGrid, threadsPerBlk, hidden_dim * sizeof(T)>>>(d_input, d_weight, d_output, hidden_dim, iterRounds, eps, d_res);
  cudaDeviceSynchronize();
  cudaMemcpyAsync(h_output.data(), d_output, rows * hidden_dim * sizeof(T), cudaMemcpyDeviceToHost);

  cudaFree(d_input);
  cudaFree(d_weight);
  cudaFree(d_output);
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
__global__ void flashAttentionKernel(
    T* q, T* k, T* v, T* o, int batch_size,
    int target_seq_len, int src_seq_len, int query_heads, int kv_heads, int head_dim, bool is_causal, bool isHalfType) {
        int b_idx = blockIdx.z;
        int qh_head_idx = blockIdx.y;
        int q_row_idx = blockIdx.x * tileHeight + threadIdx.y;          
        
        int kv_head_idx = qh_head_idx * kv_heads / query_heads;
        //int valid_seq_len = is_causal? min(q_row_idx + 1, src_seq_len) : src_seq_len;
        int valid_seq_len = src_seq_len;

        //assume head_dim <= 128
        __shared__ T q_patch[tileHeight][128]; 
        __shared__ T k_patch[tileHeight][128];
        __shared__ float intermediate[tileHeight][2048];
        __shared__ float softmaxSum[tileHeight];
    
        //load Q tile of current block to smem (smem:share memory)  
        if(q_row_idx < target_seq_len && threadIdx.x < head_dim){
            int q_offset = b_idx * (target_seq_len * query_heads * head_dim) + \
                           q_row_idx * (query_heads * head_dim) + qh_head_idx * head_dim + threadIdx.x;
            q_patch[threadIdx.y][threadIdx.x] = q[q_offset];
        }

        //load K to smem tile by tile
        int valid_row_curTile;
        for(int i = 0; i < (valid_seq_len - 1) / tileHeight + 1; i++){
            int k_row_idx = i * tileHeight + threadIdx.y;
 
            //load tile i to smem
            if(k_row_idx < valid_seq_len && threadIdx.x < head_dim){
                int k_offset = ((b_idx * src_seq_len + k_row_idx) * kv_heads + kv_head_idx) * head_dim + threadIdx.x;
                k_patch[threadIdx.y][threadIdx.x] = k[k_offset];
            }
            __syncthreads();

            //dot product: q patch * k patch
            valid_row_curTile = (i == (valid_seq_len - 1) / tileHeight)? valid_seq_len % tileHeight : tileHeight;
            if(valid_row_curTile == 0)
                valid_row_curTile = tileHeight;

            if(q_row_idx < target_seq_len && threadIdx.x < valid_row_curTile){
              if(is_causal && q_row_idx < threadIdx.x + i * tileHeight)
                intermediate[threadIdx.y][threadIdx.x + i * tileHeight] = -INFINITY;
              else{
                float sum = 0.0f;               
                for(int j = 0; j < head_dim; j++)
                      sum += castToFloat(q_patch[threadIdx.y][j]) * castToFloat(k_patch[threadIdx.x][j]);
                   
                intermediate[threadIdx.y][threadIdx.x + i * tileHeight] = sum; 
              }                                             
            }   
        }
        __syncthreads();

        valid_row_curTile = (blockIdx.x == (target_seq_len - 1) / tileHeight)? target_seq_len % tileHeight : tileHeight;
        if(valid_row_curTile == 0)
            valid_row_curTile = tileHeight;

        //find max value
        if(threadIdx.y == 0 && threadIdx.x < valid_row_curTile){
            float maxVal = -INFINITY;          
            for(int i = 0; i < valid_seq_len; i++){
                if(maxVal < intermediate[threadIdx.x][i])
                    maxVal = intermediate[threadIdx.x][i];
            }
            //scale, exp
            float softmax_sum = 0.0f;
            float scale_factor = 1.0f / sqrt(static_cast<float>(head_dim) + 1e-8f);
            for(int i = 0; i < valid_seq_len; i++){
                if(intermediate[threadIdx.x][i] > -INFINITY){
                  intermediate[threadIdx.x][i] = exp((intermediate[threadIdx.x][i] - maxVal) * scale_factor);
                  softmax_sum += intermediate[threadIdx.x][i];
                }
                else{
                  intermediate[threadIdx.x][i] = 0.0f;
                }
  
            }
            softmaxSum[threadIdx.x] = softmax_sum;
        }
        __syncthreads();

        //dot product intermediate with V       
        if(q_row_idx < target_seq_len && threadIdx.x < head_dim){
          float dotProduct = 0.0f;
            for(int i = 0; i < valid_seq_len; i++){
                int v_offset = ((b_idx * src_seq_len + i) * kv_heads + kv_head_idx) * head_dim + threadIdx.x;
                dotProduct += intermediate[threadIdx.y][i] * castToFloat(v[v_offset]);
            }
            //softmax
            if(softmaxSum[threadIdx.y] > 0.0f)
                dotProduct /= (softmaxSum[threadIdx.y] + 1e-12f);
            //store it to o
            int o_offset = b_idx * (target_seq_len * query_heads * head_dim) +  \
                            q_row_idx * (query_heads * head_dim) + qh_head_idx * head_dim + threadIdx.x;           
            o[o_offset] = castFloatToT<T>(dotProduct);
        }
}



template <typename T>
void flashAttention(const std::vector<T>& h_q, const std::vector<T>& h_k,
                    const std::vector<T>& h_v, std::vector<T>& h_o,
                    int batch_size, int target_seq_len, int src_seq_len, 
                    int query_heads, int kv_heads, int head_dim, bool is_causal) {       
  // TODO: Implement the flash attention function
  if (query_heads % kv_heads != 0) {
        return;
    }

    T *d_q = nullptr, *d_k = nullptr, *d_v = nullptr, *d_o = nullptr;
    cudaMalloc(&d_q, batch_size * target_seq_len * query_heads * head_dim * sizeof(T));
    cudaMalloc(&d_k, batch_size * src_seq_len * kv_heads * head_dim * sizeof(T));
    cudaMalloc(&d_v, batch_size * src_seq_len * kv_heads * head_dim * sizeof(T));
    cudaMalloc(&d_o, batch_size * target_seq_len * query_heads * head_dim * sizeof(T)); 

    cudaMemcpy(d_q, h_q.data(), batch_size * target_seq_len * query_heads * head_dim * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_k, h_k.data(), batch_size * src_seq_len * kv_heads * head_dim * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, h_v.data(), batch_size * src_seq_len * kv_heads * head_dim * sizeof(T), cudaMemcpyHostToDevice);   

    dim3 gridDim((target_seq_len + tileHeight - 1) / tileHeight, query_heads, batch_size);
    dim3 blockDim(max(tileHeight,head_dim), tileHeight);

    bool isHalfType = false;
    if constexpr (std::is_same<T, half>::value){
      isHalfType = true;
    }
    flashAttentionKernel<T><<<gridDim, blockDim>>>(
        d_q, d_k, d_v, d_o,
        batch_size, target_seq_len, src_seq_len,
        query_heads, kv_heads, head_dim,
        is_causal, isHalfType
    );

    cudaDeviceSynchronize();
    cudaMemcpy(h_o.data(), d_o, batch_size * target_seq_len * query_heads * head_dim * sizeof(T), cudaMemcpyDeviceToHost);

    cudaFree(d_q);
    cudaFree(d_k);
    cudaFree(d_v);
    cudaFree(d_o);     
}


// *********************************************************************
// Explicit Template Instantiations (REQUIRED FOR LINKING WITH TESTER.O)
// DO NOT MODIFY THIS SECTION
// *********************************************************************
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