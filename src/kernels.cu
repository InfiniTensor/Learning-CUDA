#include <vector>
#include <cuda_fp16.h>
#include <cmath>
#include "../tester/utils.h"

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
__device__ __forceinline__ float toFloat(T x) {
    return static_cast<float>(x);
}

template <>
__device__ __forceinline__ float toFloat<half>(half x) {
    return __half2float(x);
}

template <typename T>
__device__ __forceinline__ T fromFloat(float x) {
    return static_cast<T>(x);
}

template <>
__device__ __forceinline__ half fromFloat<half>(float x) {
    return __float2half(x);
}

template <typename T>
__global__ void rmsNormKernel(    
    const T* __restrict__ input,
    const T* __restrict__ weight,
    T* __restrict__ output,
    size_t rows,
    size_t hidden_dim,
    float eps){

    size_t row = blockIdx.x;

    if (row >= rows) {
        return;
    }

        // Shared memory for block-wide reduction.
    extern __shared__ float shared_sum[];

    float local_sum = 0.0f;

    const size_t row_offset = row * hidden_dim;

    for(size_t j = threadIdx.x;j<hidden_dim;j+=blockDim.x){
      float x = (input[row_offset + j]);
      local_sum += x*x;
    }

    shared_sum[threadIdx.x] = local_sum;
    __syncthreads();

    // 规约
    for (unsigned int stride = blockDim.x / 2;stride >0 ; stride>>=1){
      if(threadIdx.x < stride){
        shared_sum[threadIdx.x] += shared_sum[threadIdx.x + stride];
      }

    __syncthreads();
    }

        // RMS = sqrt(mean(x^2) + eps)
    float inv_rms = rsqrtf(
        shared_sum[0] / static_cast<float>(hidden_dim) + eps
    );

    for (size_t j = threadIdx.x; j < hidden_dim; j += blockDim.x) {
        float x = toFloat(input[row_offset + j]);
        float w = toFloat(weight[j]);

        float y = x * inv_rms * w;

        output[row_offset + j] = fromFloat<T>(y);
    }
}

template <typename T>
void rmsNorm(const std::vector<T>& h_input, const std::vector<T>& h_weight,
              std::vector<T>& h_output, size_t rows, size_t hidden_dim,
              float eps) {
  // TODO: Implement the rmsNorm function
  const size_t input_size = rows * hidden_dim;
  const size_t weight_size = hidden_dim;

  T* d_input = nullptr;
  T* d_weight = nullptr;
  T* d_output = nullptr;

  cudaMalloc(&d_input, input_size * sizeof(T));
  cudaMalloc(&d_weight, weight_size * sizeof(T));
  cudaMalloc(&d_output, input_size * sizeof(T));

  cudaMemcpy(
    d_input,
    h_input.data(),
    input_size * sizeof(T),
    cudaMemcpyHostToDevice
  );

  cudaMemcpy(
    d_weight,
    h_weight.data(),
    weight_size * sizeof(T),
    cudaMemcpyHostToDevice
  );

  constexpr int threads = 256;

  size_t shared_mem_size = threads * sizeof(float);

  rmsNormKernel<T><<<rows, threads, shared_mem_size>>>(
    d_input,
    d_weight,
    d_output,
    rows,
    hidden_dim,
    eps
  );

  cudaMemcpy(
    h_output.data(),
    d_output,
    input_size * sizeof(T),
    cudaMemcpyDeviceToHost
  );

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
  const T* q,
  const T* k,
  const T* v,
  T* o,
  int batch_size,
  int target_seq_len,
  int src_seq_len,
  int query_heads,
  int kv_heads,
  int head_dim,
  bool is_causal
)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  int total_heads = batch_size * target_seq_len * query_heads;

  if(idx >= total_heads) return;

  int q_head = idx % query_heads;
  int tmp = idx / query_heads;
  int tgt_pos = tmp % target_seq_len;
  int batch = tmp / target_seq_len;

  int kv_head = q_head / (query_heads / kv_heads);

  float scale = 1.0f / sqrtf((float)head_dim);

  float max_score = -1e30f;

    /*
        找 max
    */
  for(int src=0; src<src_seq_len; src++){

  // causal mask
  if(is_causal && src > tgt_pos)
    continue;

  float score = 0.0f;

  for(int d=0; d<head_dim; d++)
    {
      int q_idx = ((batch * target_seq_len + tgt_pos) * query_heads + q_head) * head_dim + d;
      int k_idx = ((batch * src_seq_len + src) * kv_heads + kv_head) * head_dim + d;
      score += toFloat(q[q_idx]) * toFloat(k[k_idx]);
    }
    score *= scale;
    max_score = max(max_score, score);
  }

    /*
        第二遍：
        softmax denominator
    */

    float sum_exp = 0.0f;

    for(int src=0; src<src_seq_len; src++)
    {
      if(is_causal && src > tgt_pos)
        continue;

      float score = 0.0f;
      for(int d=0; d<head_dim; d++){

      int q_idx =((batch * target_seq_len + tgt_pos) * query_heads + q_head) * head_dim + d;
      int k_idx =((batch * src_seq_len + src) * kv_heads + kv_head) * head_dim + d;
      score += toFloat(q[q_idx]) * toFloat(k[k_idx]);
      }
        score *= scale;
        sum_exp += expf(score - max_score);
    }

    /*
        第三遍：
        V
    */

    for(int d=0; d<head_dim; d++)
    {
      float out = 0.0f;
      for(int src=0; src<src_seq_len; src++){
        if(is_causal && src > tgt_pos)
          continue;

        float score = 0.0f;

        for(int x=0; x<head_dim; x++){
          int q_idx = ((batch * target_seq_len + tgt_pos) * query_heads + q_head) * head_dim + x;
          int k_idx = ((batch * src_seq_len + src) * kv_heads + kv_head) * head_dim + x;
          score += toFloat(q[q_idx]) * toFloat(k[k_idx]);
        }
        score *= scale;
        float prob = expf(score-max_score) / sum_exp;

        int v_idx = ((batch * src_seq_len + src) * kv_heads + kv_head) * head_dim + d;
        out += prob * toFloat(v[v_idx]);
        }

        int o_idx = ((batch * target_seq_len + tgt_pos) * query_heads + q_head) * head_dim + d;

        o[o_idx] = fromFloat<T>(out);
    }

      // for(int src=0; src<src_seq_len; src++){
      //   float out = 0.0f;
      //   if(is_causal && src > tgt_pos)
      //     continue;

      //   float score = 0.0f;

      //   for(int x=0; x<head_dim; x++){
      //     int q_idx = ((batch * target_seq_len + tgt_pos) * query_heads + q_head) * head_dim + x;
      //     int k_idx = ((batch * src_seq_len + src) * kv_heads + kv_head) * head_dim + x;
      //     score += toFloat(q[q_idx]) * toFloat(k[k_idx]);
      //   }
      //   score *= scale;
      //   float prob = expf(score-max_score) / sum_exp;

      //   for(int d=0; d<head_dim; d++){
      //     int v_idx = ((batch * src_seq_len + src) * kv_heads + kv_head) * head_dim + d;
      //     out = prob * toFloat(v[v_idx]);
      //     int o_idx = ((batch * target_seq_len + tgt_pos) * query_heads + q_head) * head_dim + d;

      //     o[o_idx] += fromFloat<T>(out);
      //   }
      //   }

}
template <typename T>
void flashAttention(const std::vector<T>& h_q, const std::vector<T>& h_k,
                    const std::vector<T>& h_v, std::vector<T>& h_o,
                    int batch_size, int target_seq_len, int src_seq_len, 
                    int query_heads, int kv_heads, int head_dim, bool is_causal) {       
  // TODO: Implement the flash attention function
  size_t q_size = batch_size * target_seq_len * query_heads * head_dim;
  size_t k_size = batch_size * src_seq_len * kv_heads * head_dim;
  size_t v_size = k_size;
  
  T* d_q;
  T* d_k;
  T* d_v;
  T* d_o;

  cudaMalloc(
    &d_q,
    q_size*sizeof(T)
  );

  cudaMalloc(
    &d_k,
    k_size*sizeof(T)
  );

  cudaMalloc(
    &d_v,
    v_size*sizeof(T)
  );

  cudaMalloc(
    &d_o,
    q_size*sizeof(T)
  );

  cudaMemcpy(
    d_q,
    h_q.data(),
    q_size*sizeof(T),
    cudaMemcpyHostToDevice
  );


  cudaMemcpy(
    d_k,
    h_k.data(),
    k_size*sizeof(T),
    cudaMemcpyHostToDevice
  );


  cudaMemcpy(
    d_v,
    h_v.data(),
    v_size*sizeof(T),
    cudaMemcpyHostToDevice
  );

  int total = batch_size * target_seq_len * query_heads;

  int threads = 256;

  int blocks = (total + threads - 1) / threads;



  flashAttentionKernel<T>
        <<<blocks,threads>>>
        (
        d_q,
        d_k,
        d_v,
        d_o,

        batch_size,
        target_seq_len,
        src_seq_len,

        query_heads,
        kv_heads,
        head_dim,

        is_causal
        );



  cudaMemcpy(
    h_o.data(),
    d_o,
    q_size*sizeof(T),
    cudaMemcpyDeviceToHost
    );


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
