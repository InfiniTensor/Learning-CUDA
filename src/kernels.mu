// final submission
#include <vector>
#include <cmath>
#include <musa_fp16.h>
#include <musa_runtime.h>  // MUSA Runtime API

#include "../tester/utils.h"


extern "C" {
    // __attribute__((weak)) 的魔法：
    // 如果系统库里有这个函数（v2环境），链接器会忽略我们写的这段代码。
    // 如果系统库里没有这个函数（v1环境），链接器就会使用我们这里的实现兜底。
    __attribute__((weak)) musaError_t musaGetDeviceProperties_v2(musaDeviceProp *prop, int device) {
        return musaGetDeviceProperties(prop, device);
    }
}

// =====================================================================
// MUSA 错误检查宏
// =====================================================================
#ifndef CHECK_MUSA
#define CHECK_MUSA(call) do { \
    musaError_t err = call; \
    if(err != musaSuccess) { \
        printf("MUSA Error at %s:%d - %s\n", __FILE__, __LINE__, musaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)
#endif

// =====================================================================
// 规约函数：针对 MUSA S4000 WarpSize = 128 做了平台安全适配
// =====================================================================
__inline__ __device__ float blockReduceSum(float val) {
    // 最大支持 blockDim.x = 1024
    static __shared__ float shared[1024]; 
    int tid = threadIdx.x;
    
    // 将所有线程的数据存入共享内存
    shared[tid] = val;
    __syncthreads(); 

    // 标准树状归约 (Tree Reduction)，不依赖特定的 Warp Size 掩码，S4000 上绝对安全
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }
    return shared[0];
}

// =====================================================================
// RMSNorm 
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

    for(int col = threadIdx.x; col < hidden_dim; col += blockDim.x)
    {
        float value = static_cast<float>(input[row * hidden_dim + col]);
        sum += value * value;
    }

    // 调用硬件安全的 Block 归约
    float total_sum = blockReduceSum(sum);

    __shared__ float s_scale;
    if (threadIdx.x == 0)
    {
        s_scale = rsqrtf(total_sum / hidden_dim + eps);
    }
    __syncthreads(); 

    float scale = s_scale;

    for(int col = threadIdx.x; col < hidden_dim; col += blockDim.x)
    {
        float value = static_cast<float>(input[row * hidden_dim + col]);
        float w = static_cast<float>(weight[col]);
        output[row * hidden_dim + col] = static_cast<T>(value * scale * w);
    }
}

template <typename T>
void rmsNorm(
    const std::vector<T>& h_input, 
    const std::vector<T>& h_weight,
    std::vector<T>& h_output, 
    size_t rows, 
    size_t hidden_dim,
    float eps
) 
{
    T* d_input;
    T* d_weight;
    T* d_output;

    size_t input_bytes = rows * hidden_dim * sizeof(T);
    size_t weight_bytes = hidden_dim * sizeof(T);

    // MUSA 显存分配
    CHECK_MUSA(musaMalloc(&d_input, input_bytes));
    CHECK_MUSA(musaMalloc(&d_output, input_bytes));
    CHECK_MUSA(musaMalloc(&d_weight, weight_bytes));

    // Host -> Device
    CHECK_MUSA(musaMemcpy(d_input, h_input.data(), input_bytes, musaMemcpyHostToDevice));
    CHECK_MUSA(musaMemcpy(d_weight, h_weight.data(), weight_bytes, musaMemcpyHostToDevice));

    dim3 grid(rows);
    dim3 block(256); // 256 threads = S4000 下刚好 2 个 128-Warp，调度优良

    rmsNormKernel<T><<<grid, block>>>(
        d_input, d_weight, d_output, rows, hidden_dim, eps
    );

    CHECK_MUSA(musaGetLastError());
    CHECK_MUSA(musaDeviceSynchronize());

    // Device -> Host
    CHECK_MUSA(musaMemcpy(h_output.data(), d_output, input_bytes, musaMemcpyDeviceToHost));

    CHECK_MUSA(musaFree(d_input));
    CHECK_MUSA(musaFree(d_weight));
    CHECK_MUSA(musaFree(d_output));
}

// =====================================================================
// FlashAttention
// =====================================================================

#define QO_OFFSET(b,s,h,d) ((((b)*target_seq_len+(s))*query_heads+(h))*head_dim+(d))
#define KV_OFFSET(b,s,h,d) ((((b)*src_seq_len+(s))*kv_heads+(h))*head_dim+(d))

template<typename T>
__global__ void flashAttentionKernel(
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
    int total = batch_size * target_seq_len * query_heads;
    int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    if(gtid >= total)
        return;

    int q_head_idx = gtid % query_heads;
    int rem = gtid / query_heads;
    int t_idx = rem % target_seq_len;
    int b_idx = rem / target_seq_len;

    int kv_head_idx = q_head_idx / (query_heads / kv_heads);

    float scale = rsqrtf((float)head_dim);

    float q_buf[256];
    float o_buf[256];

    int qo_base = QO_OFFSET(b_idx, t_idx, q_head_idx, 0);

    for(int d = 0; d < head_dim; d++)
        q_buf[d] = (float)q[qo_base + d];

    for(int d = 0; d < head_dim; d++)
        o_buf[d] = 0.0f;

    // Pass 1: 寻找最大值 (使用 fmaxf 保证 32bit 单精度)
    float global_max = -INFINITY;
    for(int s_idx = 0; s_idx < src_seq_len; s_idx++)
    {
        if(is_causal && s_idx > t_idx) continue;

        int kv_base = KV_OFFSET(b_idx, s_idx, kv_head_idx, 0);

        float score = 0.0f;
        for(int d = 0; d < head_dim; d++)
            score = fmaf(q_buf[d], (float)k[kv_base + d], score);

        global_max = fmaxf(global_max, score * scale);
    }

    // Pass 2: 计算 Softmax 分母 (使用 expf)
    float sum = 0.0f;
    for(int s_idx = 0; s_idx < src_seq_len; s_idx++)
    {
        if(is_causal && s_idx > t_idx) continue;

        int kv_base = KV_OFFSET(b_idx, s_idx, kv_head_idx, 0);

        float score = 0.0f;
        for(int d = 0; d < head_dim; d++)
            score = fmaf(q_buf[d], (float)k[kv_base + d], score);

        sum += expf(score * scale - global_max);
    }

    float inv_sum = (sum != 0.0f) ? (1.0f / sum) : 0.0f;

    // Pass 3: 计算 O 矩阵
    for(int s_idx = 0; s_idx < src_seq_len; s_idx++)
    {
        if(is_causal && s_idx > t_idx) continue;

        int kv_base = KV_OFFSET(b_idx, s_idx, kv_head_idx, 0);

        float score = 0.0f;
        for(int d = 0; d < head_dim; d++)
            score = fmaf(q_buf[d], (float)k[kv_base + d], score);

        float prob = expf(score * scale - global_max);
        float factor = inv_sum * prob;

        for(int d = 0; d < head_dim; d++)
            o_buf[d] = fmaf(factor, (float)v[kv_base + d], o_buf[d]);
    }

    for(int d = 0; d < head_dim; d++)
        o[qo_base + d] = (T)o_buf[d];
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
) 
{
    T *d_q, *d_k, *d_v, *d_o;

    size_t q_size = h_q.size() * sizeof(T);
    size_t k_size = h_k.size() * sizeof(T);
    size_t v_size = h_v.size() * sizeof(T);
    size_t o_size = h_o.size() * sizeof(T);

    CHECK_MUSA(musaMalloc(&d_q, q_size));
    CHECK_MUSA(musaMalloc(&d_k, k_size));
    CHECK_MUSA(musaMalloc(&d_v, v_size));
    CHECK_MUSA(musaMalloc(&d_o, o_size));

    CHECK_MUSA(musaMemcpy(d_q, h_q.data(), q_size, musaMemcpyHostToDevice));
    CHECK_MUSA(musaMemcpy(d_k, h_k.data(), k_size, musaMemcpyHostToDevice));
    CHECK_MUSA(musaMemcpy(d_v, h_v.data(), v_size, musaMemcpyHostToDevice));

    int total = batch_size * target_seq_len * query_heads;

    dim3 block(256);
    dim3 grid((total + block.x - 1) / block.x);

    flashAttentionKernel<T><<<grid, block>>>(
        d_q, d_k, d_v, d_o,
        batch_size, target_seq_len, src_seq_len,
        query_heads, kv_heads,
        head_dim,
        is_causal
    );

    CHECK_MUSA(musaGetLastError());
    CHECK_MUSA(musaDeviceSynchronize());

    CHECK_MUSA(musaMemcpy(h_o.data(), d_o, o_size, musaMemcpyDeviceToHost));

    CHECK_MUSA(musaFree(d_q));
    CHECK_MUSA(musaFree(d_k));
    CHECK_MUSA(musaFree(d_v));
    CHECK_MUSA(musaFree(d_o));
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
