// final submission
#include <vector>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <iostream>

#include "../tester/utils.h"

#ifndef CHECK_CUDA
#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if(err != cudaSuccess) { \
        printf("CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)
#endif

// =====================================================================
// 修复点：Makefile中传入的是 PLATFORM_ILUVATAR，并且 64线程Warp 需要 64位 的 Mask
// =====================================================================
#if defined(PLATFORM_ILUVATAR) || defined(__ILUVATAR__)
#define WARP_SIZE 64
#define FULL_MASK 0xffffffffffffffffULL  // 必须是 64 位全 1 掩码
#define BLOCK_SIZE 64
#define Bc 128
#else
#define WARP_SIZE 32
#define FULL_MASK 0xffffffff
#define BLOCK_SIZE 128
#define Bc 64
#endif

#define MAX_HEAD_DIM 256
#define MAX_BC 128

// Layout Macros
#define QO_OFFSET(b,s,h,d) ((((b)*target_seq_len+(s))*query_heads+(h))*head_dim+(d))
#define KV_OFFSET(b,s,h,d) ((((b)*src_seq_len+(s))*kv_heads+(h))*head_dim+(d))

// =====================================================================
// Warp / Block Reduce Helpers
// =====================================================================
__inline__ __device__ float warpReduceSum(float val) {
    for(int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
         val += __shfl_down_sync(FULL_MASK, val, offset);
    }
    return val;
}

__inline__ __device__ float blockReduceSum(float val) {
    __shared__ float shared[64]; 
    int lane = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;  

    val = warpReduceSum(val);

    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads(); 

    val = (threadIdx.x < (blockDim.x + WARP_SIZE - 1)/WARP_SIZE) ? shared[lane] : 0.0f;
    
    if (wid == 0) {
        val = warpReduceSum(val);
    }
    return val;
}

// =====================================================================
// RMSNorm Kernel
// =====================================================================
template <typename T>
__global__ void rmsNormKernel(
    const T* __restrict__ input,   
    const T* __restrict__ weight,
    T* __restrict__ output,
    int rows,
    int hidden_dim,
    float eps
)
{
    int row = blockIdx.x;
    if(row >= rows) return;

    float sum = 0.0f;  

    for(int col = threadIdx.x; col < hidden_dim; col += blockDim.x) {
        float value = static_cast<float>(input[row * hidden_dim + col]);
        sum += value * value;
    }

    float total_sum = blockReduceSum(sum);

    __shared__ float s_scale;
    if (threadIdx.x == 0) {
        s_scale = rsqrtf(total_sum / hidden_dim + eps);
    }
    __syncthreads(); 

    float scale = s_scale;

    for(int col = threadIdx.x; col < hidden_dim; col += blockDim.x) {
        float value = static_cast<float>(input[row * hidden_dim + col]);
        float w = static_cast<float>(weight[col]);
        output[row * hidden_dim + col] = static_cast<T>(value * scale * w);
    }
}

template <typename T>
void rmsNorm(const std::vector<T>& h_input, const std::vector<T>& h_weight,
              std::vector<T>& h_output, size_t rows, size_t hidden_dim,
              float eps) {
    T* d_input;
    T* d_weight;
    T* d_output;

    size_t input_bytes = rows * hidden_dim * sizeof(T);
    size_t weight_bytes = hidden_dim * sizeof(T);

    CHECK_CUDA(cudaMalloc(&d_input, input_bytes));
    CHECK_CUDA(cudaMalloc(&d_output, input_bytes));
    CHECK_CUDA(cudaMalloc(&d_weight, weight_bytes));

    CHECK_CUDA(cudaMemcpy(d_input, h_input.data(), input_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_weight, h_weight.data(), weight_bytes, cudaMemcpyHostToDevice));

    dim3 grid(rows);
    dim3 block(256);

    rmsNormKernel<T><<<grid, block>>>(d_input, d_weight, d_output, rows, hidden_dim, eps);

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_output.data(), d_output, input_bytes, cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_weight));
    CHECK_CUDA(cudaFree(d_output));
}

// =====================================================================
// Reference Attention Kernel
// =====================================================================
template<typename T>
__global__ void referenceAttentionKernel(
    const T* __restrict__ q,
    const T* __restrict__ k,
    const T* __restrict__ v,
    T* __restrict__ o,
    int batch_size,
    int target_seq_len,
    int src_seq_len,
    int query_heads,
    int kv_heads,
    int head_dim,
    bool is_causal
)
{
    int total_queries = batch_size * target_seq_len * query_heads;
    int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(gtid >= total_queries) return;

    int q_head_idx = gtid % query_heads;
    int rem = gtid / query_heads;
    int t_idx = rem % target_seq_len;
    int b_idx = rem / target_seq_len;

    int kv_head_idx = q_head_idx / (query_heads / kv_heads);
    float scale = __frcp_rn(sqrtf((float)head_dim));

    float q_buf[MAX_HEAD_DIM];
    float o_buf[MAX_HEAD_DIM];

    int qo_base = QO_OFFSET(b_idx, t_idx, q_head_idx, 0);

    for(int d = 0; d < head_dim; d++) {
        q_buf[d] = static_cast<float>(q[qo_base + d]);
        o_buf[d] = 0.0f;
    }

    float global_max = -1e9f; 

    for(int s_idx = 0; s_idx < src_seq_len; s_idx++) {
        if(is_causal && s_idx > t_idx) continue;

        int kv_base = KV_OFFSET(b_idx, s_idx, kv_head_idx, 0);
        float score = 0.0f;
        
        for(int d = 0; d < head_dim; d++) {
            score = fmaf(q_buf[d], static_cast<float>(k[kv_base + d]), score);
        }
        
        global_max = fmaxf(global_max, score * scale);
    }

    float sum = 0.0f;

    for(int s_idx = 0; s_idx < src_seq_len; s_idx++) {
        if(is_causal && s_idx > t_idx) continue;

        int kv_base = KV_OFFSET(b_idx, s_idx, kv_head_idx, 0);
        float score = 0.0f;
        
        for(int d = 0; d < head_dim; d++) {
            score = fmaf(q_buf[d], static_cast<float>(k[kv_base + d]), score);
        }
        
        sum += expf(score * scale - global_max);
    }

    float inv_sum = (sum > 0.0f) ? __frcp_rn(sum) : 0.0f;

    for(int s_idx = 0; s_idx < src_seq_len; s_idx++) {
        if(is_causal && s_idx > t_idx) continue;

        int kv_base = KV_OFFSET(b_idx, s_idx, kv_head_idx, 0);
        float score = 0.0f;
        
        for(int d = 0; d < head_dim; d++) {
            score = fmaf(q_buf[d], static_cast<float>(k[kv_base + d]), score);
        }
        
        float prob = expf(score * scale - global_max);
        float factor = prob * inv_sum;

        for(int d = 0; d < head_dim; d++) {
            o_buf[d] = fmaf(factor, static_cast<float>(v[kv_base + d]), o_buf[d]);
        }
    }

    for(int d = 0; d < head_dim; d++) {
        o[qo_base + d] = static_cast<T>(o_buf[d]);
    }
}

// =====================================================================
// Host Wrapper Function
// =====================================================================
template<typename T>
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
)
{
    size_t q_size = batch_size * target_seq_len * query_heads * head_dim * sizeof(T);
    size_t k_size = batch_size * src_seq_len * kv_heads * head_dim * sizeof(T);
    size_t v_size = batch_size * src_seq_len * kv_heads * head_dim * sizeof(T);
    h_o.resize(batch_size * target_seq_len * query_heads * head_dim);

    T *d_q, *d_k, *d_v, *d_o;
    CHECK_CUDA(cudaMalloc(&d_q, q_size));
    CHECK_CUDA(cudaMalloc(&d_k, k_size));
    CHECK_CUDA(cudaMalloc(&d_v, v_size));
    CHECK_CUDA(cudaMalloc(&d_o, q_size));

    CHECK_CUDA(cudaMemcpy(d_q, h_q.data(), q_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_k, h_k.data(), k_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_v, h_v.data(), v_size, cudaMemcpyHostToDevice));

    int total_queries = batch_size * target_seq_len * query_heads;
    dim3 block(256);
    dim3 grid((total_queries + block.x - 1) / block.x);

    referenceAttentionKernel<T><<<grid, block>>>(
        d_q, d_k, d_v, d_o,
        batch_size, target_seq_len, src_seq_len,
        query_heads, kv_heads, head_dim, is_causal
    );

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_o.data(), d_o, q_size, cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaFree(d_q));
    CHECK_CUDA(cudaFree(d_k));
    CHECK_CUDA(cudaFree(d_v));
    CHECK_CUDA(cudaFree(d_o));
}

// =====================================================================
// Explicit Template Instantiations
// =====================================================================
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
