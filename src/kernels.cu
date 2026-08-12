// #include <__clang_cuda_builtin_vars.h>
// #include <__clang_cuda_math.h>
#include <cstddef>
#include <cfloat>
#include <vector>
#include <cuda_fp16.h>

#include "../tester/utils.h"
#define WARPSIZE 32
/**
 * @brief Computes RMSNorm over the last dimension of a 2D tensor.
 *
 * The input is a row-major matrix with shape [rows, hidden_dim]. For each row
 * i and column col:
 *
 *   output[i, col] = input[i, col] * rsqrt(mean(input[i, :]^2) + eps) * weight[col]
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

// 核心目的是：参数内部都已FP32的精度进行处理，保证精度的同时来去适应half，float类型

template <typename T>
__device__ float toFloat(T x)
{
    return static_cast<float>(x);
}

template <>
__device__ float toFloat<half>(half x)
{
    return __half2float(x);
}

template <typename T>
__device__ T fromFloat(float x)
{
    return static_cast<T>(x);
}

template <>
__device__ half fromFloat<half>(float x)
{
    return __float2half(x);
}

template <typename T>
__device__ __forceinline__ float warpReduceSum(float val)
{
#pragma unroll
    for (int offset = WARPSIZE / 2; offset > 0; offset >>= 1)
    {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

template <typename T>
__global__ void rmsNormKernel(const T *input, const T *weight,
                              T *output, size_t rows, size_t hidden_dim,
                              float eps)
{
    const size_t row = blockIdx.x;
    const int tid = threadIdx.x;

    if (row >= rows)
        return;

    const int lane = tid % 32;
    const int warp_id = tid / 32;
    const int num_warps = blockDim.x / WARPSIZE;

    // 最多支持 1024 threads = 32 warps
    __shared__ float warp_sums[32];
    __shared__ float rms_scale; // 一行全局共享

    const T *row_input = input + row * hidden_dim;
    T *row_output = output + row * hidden_dim;

    float square_sum = 0.0f;

    // 每个线程覆盖多个元素，如果一行的数据比较长，这里的stride = blockDim.x
    for (size_t col = tid; col < hidden_dim; col += blockDim.x)
    {
        float x = toFloat(row_input[col]);
        square_sum += x * x;
    }

    square_sum = warpReduceSum<T>(square_sum);

    // 每个 warp 只写一个值到 shared memory，完成第一层归约
    if (lane == 0)
    {
        warp_sums[warp_id] = square_sum;
    }

    __syncthreads();

    //  warp 0 对各个 warp 的结果继续 reduction，二层归约
    if (warp_id == 0)
    {
        float block_sum =
            lane < num_warps ? warp_sums[lane] : 0.0f;

        block_sum = warpReduceSum<T>(block_sum);

        if (lane == 0)
        {
            rms_scale = rsqrtf(block_sum / static_cast<float>(hidden_dim) + eps);
        }
    }

    __syncthreads();

    // 类似的线程处理自己负责行的多个数据
    for (size_t col = tid; col < hidden_dim; col += blockDim.x)
    {
        float x = toFloat(row_input[col]);
        float w = toFloat(weight[col]);

        row_output[col] = fromFloat<T>(x * rms_scale * w);
    }
}

template <typename T>
void rmsNorm(const std::vector<T> &h_input, const std::vector<T> &h_weight,
             std::vector<T> &h_output, size_t rows, size_t hidden_dim,
             float eps)
{
    size_t input_bytes = rows * hidden_dim * sizeof(T);

    size_t weight_bytes = hidden_dim * sizeof(T);

    // 分配指针
    T *d_input = nullptr;
    T *d_weight = nullptr;
    T *d_output = nullptr;

    // 给指针分配内存空间
    cudaMalloc(&d_input, input_bytes);
    cudaMalloc(&d_weight, weight_bytes);
    cudaMalloc(&d_output, input_bytes);

    // 复制数据
    cudaMemcpy(d_input, h_input.data(), input_bytes, cudaMemcpyHostToDevice);

    cudaMemcpy(d_weight, h_weight.data(), weight_bytes, cudaMemcpyHostToDevice);

    // 一个block负责一行
    dim3 block(256);
    dim3 grid(rows);

    size_t shared_bytes = block.x * sizeof(float);

    rmsNormKernel<T><<<grid, block, shared_bytes>>>(d_input, d_weight, d_output, rows, hidden_dim, eps);

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
 * @tparam T Data type for input/output tensors
 * @param[in] h_q Query tensor of shape
 * [batch_size, target_seq_len, query_heads, head_dim]
 * @param[in] h_k Key tensor of shape
 * [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[in] h_v Value tensor of shape
 * [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[out] h_o Output tensor of shape
 * [batch_size, target_seq_len, query_heads, head_dim]
 * @param[in] batch_size Batch dimension size
 * @param[in] target_seq_len Target sequence length
 * @param[in] src_seq_len Source sequence length
 * @param[in] query_heads Number of query attention heads
 * @param[in] kv_heads Number of key/value heads
 * @param[in] head_dim Dimension of each attention head
 * @param[in] is_causal Whether to apply causal masking
 */

// 一个Block负责一个输出向量:
//
// O[batch_id, q_pos, q_head, :]
//
// 当前版本主要以测试正确性和浮点结果一致性为目标，
// 不使用Online Softmax，也不保存完整Score。
//
// 计算过程分成三步:
//
// 1. 第一次QK：找到当前Attention Row的row_max
// 2. 第二次QK：计算Softmax归一化和
// 3. 第三次QK：计算weight并直接累加P * V
//
// QK存在重复计算，后续再考虑性能优化。
template <typename T>
__global__ void flashAttentionKernel(
    const T *q,
    const T *k,
    const T *v,
    T *o,
    int batch_size,
    int target_seq_len,
    int src_seq_len,
    int query_heads,
    int kv_heads,
    int head_dim,
    bool is_causal)
{
    extern __shared__ float smem[];

    int task_id = blockIdx.x;

    // 当前Block负责哪个Q head
    int q_head = task_id % query_heads;
    // 当前Block负责哪个Query Row
    int q_pos = (task_id / query_heads) % target_seq_len;
    // 当前Block属于哪个Batch
    int batch_id = task_id / (query_heads * target_seq_len);

    if (batch_id >= batch_size)
    {
        return;
    }
    // query_heads == kv_heads:
    //      一个Q head对应一个KV head --- MHA
    // query_heads > kv_heads:
    //      多个Q head共享一个KV head --- GQA
    int kv_head = q_head * kv_heads / query_heads;

    // Causal Attention只能访问当前位置以及之前的Key
    int key_count = is_causal ? min(q_pos + 1, src_seq_len) : src_seq_len;

    float attn_scale = rsqrtf(static_cast<float>(head_dim));

    const std::size_t q_offset = ((static_cast<std::size_t>(batch_id) * target_seq_len + q_pos) * query_heads + q_head) * head_dim;
    // 计算QK：寻找当前Attention Row中的最大即row_max
    if (threadIdx.x == 0)
    {
        float row_max = -INFINITY;
        for (int k_pos = 0; k_pos < key_count; k_pos++)
        {
            const std::size_t k_offset = ((static_cast<std::size_t>(batch_id) * src_seq_len + k_pos) * kv_heads + kv_head) * head_dim;
            float logit = 0.0f;

            // 一个Q Row和一个K Row进行dot
            for (int dim = 0; dim < head_dim; dim++)
            {
                float q_val = toFloat(q[q_offset + dim]);
                float k_val = toFloat(k[k_offset + dim]);

                logit += q_val * k_val;
            }

            logit *= attn_scale;

            row_max = fmaxf(row_max, logit);
        }
        // 计算QK，并记录Softmax归一化和

        float softmax_sum = 0.0f;

        for (int k_pos = 0; k_pos < key_count; k_pos++)
        {
            const std::size_t k_offset = ((static_cast<std::size_t>(batch_id) * src_seq_len + k_pos) * kv_heads + kv_head) * head_dim;
            float logit = 0.0f;

            for (int dim = 0; dim < head_dim; dim++)
            {
                float q_val = toFloat(q[q_offset + dim]);
                float k_val = toFloat(k[k_offset + dim]);

                logit += q_val * k_val;
            }

            logit *= attn_scale;

            softmax_sum += expf(logit - row_max); // safe softmax
        }

        // 后续Block内部线程都需要使用
        smem[0] = row_max;
        smem[1] = softmax_sum;
    }

    // 等待thread 0完成当前Attention Row的row_max和softmax_sum
    __syncthreads();

    // 每个线程在寄存器中独立持有
    float row_max = smem[0];
    float softmax_sum = smem[1];

    // 第三次计算QK，同时完成 P * V
    // 每个线程负责若干个head_dim位置
    for (int dim = threadIdx.x; dim < head_dim; dim += blockDim.x)
    {
        float out_val = 0.0f;

        // 当前输出维度需要遍历所有有效的KV Row
        for (int k_pos = 0; k_pos < key_count; k_pos++)
        {
            // K/V布局相同:
            // [B, Lkv, Hkv, D]
            // 因此可以共用同一个KV起始位置
            const std::size_t kv_offset = ((static_cast<std::size_t>(batch_id) * src_seq_len + k_pos) * kv_heads + kv_head) * head_dim;
            // 重新计算当前Q Row与K Row的logit
            float logit = 0.0f;

            for (int k_dim = 0; k_dim < head_dim; k_dim++)
            {
                float q_val = toFloat(q[q_offset + k_dim]);
                float k_val = toFloat(k[kv_offset + k_dim]);

                logit += q_val * k_val;
            }

            logit *= attn_scale;

            // weight = softmax(QK / sqrt(head_dim))
            float weight = expf(logit - row_max) / softmax_sum;
            float v_val = toFloat(v[kv_offset + dim]);

            out_val += weight * v_val;
        }

        // [B, Lq, Hq, D]
        o[q_offset + dim] = fromFloat<T>(out_val);
    }
}

template <typename T>
void flashAttention(const std::vector<T> &h_q, const std::vector<T> &h_k,
                    const std::vector<T> &h_v, std::vector<T> &h_o,
                    int batch_size, int target_seq_len, int src_seq_len,
                    int query_heads, int kv_heads, int head_dim, bool is_causal)
{
    const size_t query_elements = static_cast<size_t>(batch_size) * target_seq_len * query_heads * head_dim;
    const size_t kv_elements = static_cast<size_t>(batch_size) * src_seq_len * kv_heads * head_dim;
    const size_t query_bytes = query_elements * sizeof(T);
    const size_t kv_bytes = kv_elements * sizeof(T);

    T *d_q = nullptr;
    T *d_k = nullptr;
    T *d_v = nullptr;
    T *d_o = nullptr;

    // 分配GPU内存
    cudaMalloc(&d_q, query_bytes);
    cudaMalloc(&d_k, kv_bytes);
    cudaMalloc(&d_v, kv_bytes);
    cudaMalloc(&d_o, query_bytes);

    // Host -> Device
    cudaMemcpy(d_q, h_q.data(), query_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_k, h_k.data(), kv_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, h_v.data(), kv_bytes, cudaMemcpyHostToDevice);

    // 一个Block负责:
    // O[batch_id, q_pos, q_head, :]
    int total_blocks = batch_size * target_seq_len * query_heads;
    dim3 grid(total_blocks);

    // 当前主要的线程并行发生在输出的head_dim上。
    // 这里保留原版本的block_size选择方式，
    // 后续再单独进行性能上的调整。
    int parallel_size = src_seq_len > head_dim ? src_seq_len : head_dim;
    int block_size = 32;

    // 保证并行维度 足够
    while (block_size < parallel_size && block_size < 256)
    {
        block_size <<= 1;
    }

    dim3 block(block_size);

    // 暂时使用非online的形式处理结果
    // smem[0] = row_max
    // smem[1] = softmax_sum
    size_t shared_bytes = 2 * sizeof(float);

    flashAttentionKernel<T>
        <<<grid, block, shared_bytes>>>(d_q, d_k, d_v, d_o, batch_size, target_seq_len, src_seq_len, query_heads, kv_heads, head_dim, is_causal);

    h_o.resize(query_elements);

    // Device -> Host
    // 记录计算结果，便于后续处理
    cudaMemcpy(h_o.data(), d_o, query_bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_q);
    cudaFree(d_k);
    cudaFree(d_v);
    cudaFree(d_o);
}

// *********************************************************************
// Explicit Template Instantiations (REQUIRED FOR LINKING WITH TESTER.O)
// DO NOT MODIFY THIS SECTION
// *********************************************************************
template void rmsNorm<float>(const std::vector<float> &, const std::vector<float> &,
                             std::vector<float> &, size_t, size_t, float);
template void rmsNorm<half>(const std::vector<half> &, const std::vector<half> &,
                            std::vector<half> &, size_t, size_t, float);
template void flashAttention<float>(const std::vector<float> &, const std::vector<float> &,
                                    const std::vector<float> &, std::vector<float> &,
                                    int, int, int, int, int, int, bool);
template void flashAttention<half>(const std::vector<half> &, const std::vector<half> &,
                                   const std::vector<half> &, std::vector<half> &,
                                   int, int, int, int, int, int, bool);