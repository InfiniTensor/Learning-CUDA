// #include <__clang_cuda_builtin_vars.h>
// #include <__clang_cuda_math.h>
#include <cstddef>
#include <cfloat>
#include <vector>
#include <cuda_fp16.h>

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
__device__ float toFloat(T x) {
    return static_cast<float>(x);
}

template <>
__device__ float toFloat<half>(half x) {
    return __half2float(x);
}

template <typename T>
__device__ T fromFloat(float x) {
    return static_cast<T>(x);
}

template <>
__device__ half fromFloat<half>(float x) {
    return __float2half(x);
}

__device__ __forceinline__ float warpReduceMax(float value) {
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        value = fmaxf(value, __shfl_down_sync(0xffffffff, value, offset));
    }
    return value;
}

__device__ __forceinline__ float warpReduceSum(float value) {
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffff, value, offset);
    }
    return value;
}

__device__ __forceinline__ float blockReduceMax(float value, float* warp_values) {
    const int lane = threadIdx.x & (warpSize - 1);
    const int warp = threadIdx.x / warpSize;
    const int num_warps = blockDim.x / warpSize;

    value = warpReduceMax(value);
    if (lane == 0) {
        warp_values[warp] = value;
    }
    __syncthreads();

    value = (warp == 0 && lane < num_warps) ? warp_values[lane] : -FLT_MAX;
    if (warp == 0) {
        value = warpReduceMax(value);
    }
    if (threadIdx.x == 0) {
        warp_values[0] = value;
    }
    __syncthreads();
    return warp_values[0];
}

__device__ __forceinline__ float blockReduceSum(float value, float* warp_values) {
    const int lane = threadIdx.x & (warpSize - 1);
    const int warp = threadIdx.x / warpSize;
    const int num_warps = blockDim.x / warpSize;

    value = warpReduceSum(value);
    if (lane == 0) {
        warp_values[warp] = value;
    }
    __syncthreads();

    value = (warp == 0 && lane < num_warps) ? warp_values[lane] : 0.0f;
    if (warp == 0) {
        value = warpReduceSum(value);
    }
    if (threadIdx.x == 0) {
        warp_values[0] = value;
    }
    __syncthreads();
    return warp_values[0];
}

template <typename T>
 __global__ void rmsNormKernel(const T* input, const T* weight, T* output, size_t rows, size_t hidden_dim, float eps){
              extern __shared__ float smem[];

              // 分配行，分配线程每个线程一行元素
              size_t row = blockIdx.x;
              size_t tid = threadIdx.x;

              if (row >= rows) {
                      return;
              }

              float sum = 0.0f;

              for (size_t j =tid; j < hidden_dim; j += blockDim.x) {
                  float x = toFloat(input[row * hidden_dim + j]);
                  sum += x * x;
              }
              // 每个线程把自己的局部平方和放进去：
              smem[tid] = sum;
              __syncthreads();
              
              // 在 block 内做规约reduce，把所有线程的局部和加起来。分而治之二分法
              for (size_t s = blockDim.x / 2; s > 0; s >>= 1) {
                  if (tid < s) {
                      smem[tid] += smem[tid + s];
                  }
                  __syncthreads();
              }
              
              // 计算 RMSNorm 的缩放系数。
              float inv_rms = rsqrtf(smem[0] / hidden_dim + eps);
              
              // 每个线程计算 output。
              for (size_t j = tid; j < hidden_dim; j += blockDim.x) {
                  float x = toFloat(input[row * hidden_dim + j]);
                  float w = toFloat(weight[j]);
                  float y = x * inv_rms * w;
                  output[row * hidden_dim + j] = fromFloat<T>(y);
              }
}

template <typename T> 
void rmsNorm(const std::vector<T>& h_input, const std::vector<T>& h_weight,
              std::vector<T>& h_output, size_t rows, size_t hidden_dim,
              float eps) {
                // TODO: Implement the rmsNorm function
                size_t input_bytes = rows * hidden_dim * sizeof(T);
                size_t weight_bytes = hidden_dim * sizeof(T);

                // 分配指针
                T * d_input = nullptr;
                T * d_weight = nullptr;
                T * d_output = nullptr;

                // 给指针分配内存空间
                cudaMalloc(&d_input, input_bytes);
                cudaMalloc(&d_weight, weight_bytes);
                cudaMalloc(&d_output, input_bytes);

                // 复制数据
                cudaMemcpy(d_input, h_input.data(), input_bytes, cudaMemcpyHostToDevice);
                cudaMemcpy(d_weight, h_weight.data(), weight_bytes, cudaMemcpyHostToDevice);

                // mean_square = sum_j input[i, j]^2 / hidden_dim
                // output[i, j] = input[i, j] * rsqrt(mean_square + eps) * weight[j]
                // rom是对每一行操作，让一个block负责一行，一个grid负责所有blcok
                dim3 block(256);
                dim3 grid(rows);
                size_t smem_size = block.x * sizeof(float);
                rmsNormKernel<T><<<grid, block, smem_size>>>(d_input, d_weight, d_output, rows, hidden_dim, eps);

                cudaDeviceSynchronize();
                h_output.resize(rows * hidden_dim);
                cudaMemcpy(h_output.data(), d_output, input_bytes, cudaMemcpyDeviceToHost);

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
__global__ void flashAttentionKernel( const T * q, const T * k, const T * v, T* o, 
                int batch_size, int target_seq_len, int src_seq_len,
                int query_heads, int kv_heads, int head_dim, bool is_causal){
                        extern __shared__ float smem[];
                        int block_id = blockIdx.x;

                        // 计算当前blcok在计算哪个query_heads
                        int qh = block_id % query_heads;
                        // 计算当前block属于哪个query_seq
                        int t = (block_id / query_heads) % target_seq_len;
                        // 计算当前block属于哪个batch
                        int b = block_id / (query_heads * target_seq_len);

                        if (b >= batch_size) {
                            return;
                        }

                        int kvh = qh * kv_heads / query_heads;
                        int max_s = is_causal ? min(t + 1, src_seq_len) : src_seq_len;
                        float scale = rsqrtf(static_cast<float>(head_dim));

                        if (threadIdx.x == 0) {
                            // 只有 thread 0 一个线程串行计算所有 s 的 score，然后它自己在本线程里比较出 max_score
                            float  max_score = -3.402823466e+38F;

                            for (int s = 0; s < max_s; ++s) {
                                    float score = 0.0f;

                                    for (int d=0; d < head_dim;  ++d) {
                                        int q_idx = ((b * target_seq_len + t) * query_heads + qh) * head_dim + d;// Q[b][t][qh][d]
                                        int k_idx = ((b * src_seq_len + s) * kv_heads + kvh) * head_dim + d;//K[b][t][qh][d]

                                        float q_val = toFloat(q[q_idx]);
                                        float k_val = toFloat(k[k_idx]);

                                        score += q_val * k_val;
                                    }

                                    score *= scale;
                                    max_score = fmaxf(max_score, score);
                            }

                            float denom = 0.0f;

                            for (int s = 0; s < max_s; ++s) {
                                    float score = 0.0f;

                                    for (int d = 0; d < head_dim; ++d) {
                                            int q_idx = ((b * target_seq_len + t) * query_heads + qh) *head_dim + d;
                                            int k_idx = ((b * src_seq_len + s) * kv_heads + kvh) * head_dim +d;

                                            float q_val = toFloat(q[q_idx]);
                                            float k_val = toFloat(k[k_idx]);

                                            score += q_val * k_val;
                                    }

                                    score *= scale;
                                    /// 为了稳定数值，所有分子分母得分都减去最大值max_score,值不变不会出现溢出
                                    denom += expf(score - max_score);
                            }

                            smem[0] = max_score;
                            smem[1] = denom;
                        }

                        // 等待线程0计算出最大值得分max_score  和 sofamax的分母“sum(exp score all)”
                        __syncthreads();

                        float max_score = smem[0];
                        float denom = smem[1];

                        for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
                                float out = 0.0f;

                                for (int s = 0; s <max_s; ++s) {
                                        float score = 0.0f;
                                        
                                        for (int kd= 0; kd < head_dim; ++kd) {
                                                int q_idx = ((b * target_seq_len + t) * query_heads + qh) *head_dim + kd;
                                                int k_idx = ((b * src_seq_len + s) * kv_heads + kvh) * head_dim +kd;

                                                float q_val = toFloat(q[q_idx]);
                                                float k_val = toFloat(k[k_idx]);

                                                score += q_val * k_val;
                                        }

                                        score *= scale;

                                        float prob = expf(score - max_score) / denom;// prob = softmax(QK/sqrt(d))
                                        int v_idx = ((b * src_seq_len + s) * kv_heads + kvh) * head_dim + d;
                                        float v_val = toFloat(v[v_idx]);

                                        out += prob * v_val;
                                }

                                int o_idx = ((b * target_seq_len +t) * query_heads +qh) * head_dim + d;
                                o[o_idx] = fromFloat<T>(out);
                        }
                }


template <typename T>
__global__ void flashAttentionKernelOptimized(
                const T * q, const T * k, const T * v, T* o,
                int batch_size, int target_seq_len, int src_seq_len,
                        int query_heads, int kv_heads, int head_dim, bool is_causal){
                        extern __shared__ float smem[];
                        float* scores = smem;
                        __shared__ float warp_values[32];
                        int block_id = blockIdx.x;

                        int qh = block_id % query_heads;
                        int t = (block_id / query_heads) % target_seq_len;
                        int b = block_id / (query_heads * target_seq_len);

                        if (b >= batch_size) {
                            return;
                        }

                        int kvh = qh * kv_heads / query_heads;
                        int max_s = is_causal ? min(t + 1, src_seq_len) : src_seq_len;
                        float scale = rsqrtf(static_cast<float>(head_dim));
                        int q_base = ((b * target_seq_len + t) * query_heads + qh) * head_dim;
                        int kv_base = b * src_seq_len * kv_heads * head_dim;

                        for (int s = threadIdx.x; s < max_s; s += blockDim.x) {
                            float score = 0.0f;
                            int k_base = kv_base + (s * kv_heads + kvh) * head_dim;

                            for (int d = 0; d < head_dim; ++d) {
                                score += toFloat(q[q_base + d]) * toFloat(k[k_base + d]);
                            }

                            scores[s] = score * scale;
                        }
                        __syncthreads();

                        float local_max = -FLT_MAX;
                        for (int s = threadIdx.x; s < max_s; s += blockDim.x) {
                            local_max = fmaxf(local_max, scores[s]);
                        }
                        const float max_score = blockReduceMax(local_max, warp_values);

                        float local_sum = 0.0f;
                        for (int s = threadIdx.x; s < max_s; s += blockDim.x) {
                            const float numerator = expf(scores[s] - max_score);
                            scores[s] = numerator;
                            local_sum += numerator;
                        }
                        const float denom = blockReduceSum(local_sum, warp_values);

                        for (int s = threadIdx.x; s < max_s; s += blockDim.x) {
                            scores[s] /= denom;
                        }
                        __syncthreads();

                        for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
                            float out = 0.0f;

                            for (int s = 0; s < max_s; ++s) {
                                int v_idx = kv_base + (s * kv_heads + kvh) * head_dim + d;
                                out += scores[s] * toFloat(v[v_idx]);
                            }

                            int o_idx = ((b * target_seq_len + t) * query_heads + qh) * head_dim + d;
                            o[o_idx] = fromFloat<T>(out);
                        }
                }


template <typename T>
void flashAttention(const std::vector<T>& h_q, const std::vector<T>& h_k,
                    const std::vector<T>& h_v, std::vector<T>& h_o,
                    int batch_size, int target_seq_len, int src_seq_len, 
                    int query_heads, int kv_heads, int head_dim, bool is_causal) {       
  // TODO: Implement the flash attention function
                    size_t q_bytes = batch_size * target_seq_len * query_heads * head_dim * sizeof(T);
                    size_t k_bytes = batch_size * src_seq_len * kv_heads * head_dim * sizeof(T);
                    size_t v_bytes = batch_size * src_seq_len * kv_heads * head_dim * sizeof(T);

                    size_t o_elements = static_cast<size_t>(batch_size) * target_seq_len * query_heads * head_dim;

                    size_t o_bytes = batch_size * target_seq_len * query_heads * head_dim * sizeof(T);

                    T * d_q = nullptr;
                    T * d_k = nullptr;
                    T * d_v = nullptr;
                    T * d_o = nullptr;

                    cudaMalloc(&d_q, q_bytes);
                    cudaMalloc(&d_k, k_bytes);
                    cudaMalloc(&d_v, v_bytes);
                    cudaMalloc(&d_o,o_bytes);
                    
                    cudaMemcpy(d_q, h_q.data(), q_bytes, cudaMemcpyHostToDevice);
                    cudaMemcpy(d_k, h_k.data(), k_bytes, cudaMemcpyHostToDevice);
                    cudaMemcpy(d_v, h_v.data(), v_bytes, cudaMemcpyHostToDevice);
                    
                    // 一个 block 负责一个输出向量 O[b, t, qh, :]
                    // 一个输出向量对应三个维度：[batch target_token query_head]
                    int total_blocks  = batch_size * target_seq_len * query_heads;
                    dim3 grid(total_blocks );
                    int work_size = src_seq_len > head_dim ? src_seq_len : head_dim;
                    int block_size = 32;
                    while (block_size < work_size && block_size < 256) {
                        block_size <<= 1;
                    }
                    dim3 block(block_size);
                    
                    size_t smem_size = src_seq_len * sizeof(float);
                    
                    if (sizeof(T) == sizeof(half) && smem_size <= 48 * 1024) {
                        flashAttentionKernelOptimized<T><<<grid, block, smem_size>>>(d_q, d_k, d_v, d_o, batch_size, target_seq_len, src_seq_len, query_heads, kv_heads, head_dim, is_causal);
                    } else {
                        flashAttentionKernel<T><<<grid, block, 2 * sizeof(float)>>>(d_q, d_k, d_v, d_o, batch_size, target_seq_len, src_seq_len, query_heads, kv_heads, head_dim, is_causal);
                    }

                    h_o.resize(o_elements);

                    cudaMemcpy(h_o.data(), d_o, o_bytes, cudaMemcpyDeviceToHost);

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
