#include <vector>
#include <musa_fp16.h>
#include <cfloat>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <string>
#include <iostream>
#include <cstdlib>

// Error checking macro
#define MUSA_CHECK(call) \
{ \
    musaError_t err = call; \
    if (err != musaSuccess) \
    { \
        std::cerr << "MUSA error at " << __FILE__ << ":" << __LINE__ \
                  << " - " << musaGetErrorString(err) << "\n"; \
        exit(1); \
    } \
}


constexpr int WARP_SIZE = 32;

template <typename T, const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ T warp_reduce_sum(T val) {
    #pragma unroll
    for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, mask);
    }
    return val;
}

template <typename T>
__global__ void trace_kernel(const T* d_input, int cols, int n_diag, T* d_sum) {
    constexpr int NUM_THREADS = 256;
    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    __shared__ T reduce_smem[NUM_WARPS];

    int tid = threadIdx.x;
    int idx = blockIdx.x * NUM_THREADS + tid;
    int warp = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;

    T sum = (idx < n_diag) ? d_input[idx * cols + idx] : T(0);
    sum = warp_reduce_sum<T, WARP_SIZE>(sum);

    if (lane == 0) {
        reduce_smem[warp] = sum;
    }
    __syncthreads();

    sum = (lane < NUM_WARPS) ? reduce_smem[lane] : T(0);
    if (warp == 0) {
        sum = warp_reduce_sum<T, NUM_WARPS>(sum);
    }

    if (tid == 0) {
        atomicAdd(d_sum, sum);
    }
}

template <typename T>
T trace(const std::vector<T>& h_input, size_t rows, size_t cols) {
    if (h_input.empty() || rows == 0 || cols == 0) {
        return T(0);
    }

    const int n_diag = static_cast<int>(std::min(rows, cols));
    const int block_size = 256;
    const int grid_size = (n_diag + block_size - 1) / block_size;

    T* d_input = nullptr;
    T* d_sum = nullptr;
    
    MUSA_CHECK(musaMalloc(&d_input, h_input.size() * sizeof(T)));
    MUSA_CHECK(musaMalloc(&d_sum, sizeof(T)));

    MUSA_CHECK(musaMemcpy(d_input, h_input.data(), h_input.size() * sizeof(T), musaMemcpyHostToDevice));
    MUSA_CHECK(musaMemset(d_sum, 0, sizeof(T)));

    trace_kernel<T><<<grid_size, block_size>>>(
        d_input, static_cast<int>(cols), n_diag, d_sum
    );
    MUSA_CHECK(musaGetLastError());
    MUSA_CHECK(musaDeviceSynchronize());

    T h_sum = T(0);
    MUSA_CHECK(musaMemcpy(&h_sum, d_sum, sizeof(T), musaMemcpyDeviceToHost));

    MUSA_CHECK(musaFree(d_input));
    MUSA_CHECK(musaFree(d_sum));

    return h_sum;
}

constexpr int FLASH_BLOCK_SIZE = 32;


template <typename T>
__global__ void flashAttentionKernel(
    const T* Q, const T* K, const T* V, T* O,
    int batch_size, int target_seq_len, int src_seq_len,
    int query_heads, int kv_heads, int head_dim, bool is_causal) {

    (void)batch_size;
    int batch_idx = blockIdx.z;
    int head_idx = blockIdx.y;
    int row_idx = blockIdx.x;
    int tid = threadIdx.x;

    if (row_idx >= target_seq_len) {
        return;
    }

    if (tid != 0) {
        return;
    }

    int kv_head_idx = 0;
    if (query_heads % kv_heads == 0) {
        int q_per_kv = query_heads / kv_heads;
        kv_head_idx = head_idx / q_per_kv;
    } else {
        kv_head_idx = head_idx % kv_heads;
    }
    int effective_src = is_causal ? min(src_seq_len, row_idx + 1) : src_seq_len;
    int out_base = ((batch_idx * target_seq_len + row_idx) * query_heads + head_idx) * head_dim;

    extern __shared__ float smem[];
    float* q_vec = smem;                 // [head_dim]，缓存当前 query 向量
    float* out_accum = q_vec + head_dim; // [head_dim]，累计 softmax 后的输出

    if (effective_src <= 0) {
        for (int d = 0; d < head_dim; ++d) {
            O[out_base + d] = static_cast<T>(0);
        }
        return;
    }

    // 缩放因子：scores = (QK^T) / sqrt(d)
    const float inv_sqrt_d = 1.0f / sqrtf(static_cast<float>(head_dim));
    for (int d = 0; d < head_dim; ++d) {
        int q_idx = ((batch_idx * target_seq_len + row_idx) * query_heads + head_idx) * head_dim + d;
        q_vec[d] = static_cast<float>(Q[q_idx]);
        out_accum[d] = 0.0f;
    }

    // Pass 1：求 m = max_j(score_j)，用于数值稳定 softmax。
    float m = -FLT_MAX;
    for (int s = 0; s < effective_src; ++s) {
        float dot = 0.0f;
        int k_base = ((batch_idx * src_seq_len + s) * kv_heads + kv_head_idx) * head_dim;
        for (int d = 0; d < head_dim; ++d) {
            dot = fmaf(q_vec[d], static_cast<float>(K[k_base + d]), dot);
        }
        m = fmaxf(m, dot * inv_sqrt_d);
    }

    // Pass 2：计算
    //   denom = sum_j exp(score_j - m)
    //   out   = sum_j exp(score_j - m) * V_j
    float denom = 0.0f;
    for (int s = 0; s < effective_src; ++s) {
        float dot = 0.0f;
        int k_base = ((batch_idx * src_seq_len + s) * kv_heads + kv_head_idx) * head_dim;
        for (int d = 0; d < head_dim; ++d) {
            dot = fmaf(q_vec[d], static_cast<float>(K[k_base + d]), dot);
        }

        float w = expf(dot * inv_sqrt_d - m);
        denom += w;

        int v_base = ((batch_idx * src_seq_len + s) * kv_heads + kv_head_idx) * head_dim;
        for (int d = 0; d < head_dim; ++d) {
            out_accum[d] += w * static_cast<float>(V[v_base + d]);
        }
    }

    // 最终归一化：out = out / denom
    float inv_denom = (denom > 0.0f) ? (1.0f / denom) : 0.0f;
    for (int d = 0; d < head_dim; ++d) {
        O[out_base + d] = static_cast<T>(out_accum[d] * inv_denom);
    }
}

// Host function for Flash Attention v1
template <typename T>
void flashAttention(const std::vector<T>& h_q, const std::vector<T>& h_k,
                    const std::vector<T>& h_v, std::vector<T>& h_o,
                    int batch_size, int target_seq_len, int src_seq_len, 
                    int query_heads, int kv_heads, int head_dim, bool is_causal) {
    // 输出形状：[B, Tq, Hq, D]
    const size_t o_elems =
        static_cast<size_t>(batch_size > 0 ? batch_size : 0) *
        static_cast<size_t>(target_seq_len > 0 ? target_seq_len : 0) *
        static_cast<size_t>(query_heads > 0 ? query_heads : 0) *
        static_cast<size_t>(head_dim > 0 ? head_dim : 0);
    if (h_o.size() != o_elems) {
        h_o.resize(o_elems);
    }

    if (o_elems == 0) {
        return;
    }
    if (batch_size <= 0 || target_seq_len <= 0 || src_seq_len <= 0 ||
        query_heads <= 0 || kv_heads <= 0 || head_dim <= 0) {
        // 输入维度非法时，按全 0 输出处理，避免非法 launch/malloc。
        std::fill(h_o.begin(), h_o.end(), T(0));
        return;
    }

    size_t elem_size = sizeof(T);
    // 输入布局：
    // Q: [B, Tq, Hq, D]
    // K: [B, Tk, Hkv, D]
    // V: [B, Tk, Hkv, D]
    size_t q_elems = (size_t)batch_size * target_seq_len * query_heads * head_dim;
    size_t k_elems = (size_t)batch_size * src_seq_len * kv_heads * head_dim;
    size_t v_elems = (size_t)batch_size * src_seq_len * kv_heads * head_dim;
    size_t q_size = q_elems * elem_size;
    size_t k_size = k_elems * elem_size;
    size_t v_size = v_elems * elem_size;
    size_t o_size = static_cast<size_t>(batch_size) * target_seq_len * query_heads * head_dim * elem_size;

    if (h_q.size() != q_elems || h_k.size() != k_elems || h_v.size() != v_elems) {
        throw std::invalid_argument("flashAttention: input tensor sizes do not match provided dimensions.");
    }

    T* d_q = nullptr;
    T* d_k = nullptr;
    T* d_v = nullptr;
    T* d_o = nullptr;
    
    try {
        MUSA_CHECK(musaMalloc(&d_q, q_size));
        MUSA_CHECK(musaMalloc(&d_k, k_size));
        MUSA_CHECK(musaMalloc(&d_v, v_size));
        MUSA_CHECK(musaMalloc(&d_o, o_size));

        MUSA_CHECK(musaMemcpy(d_q, h_q.data(), q_size, musaMemcpyHostToDevice));
        MUSA_CHECK(musaMemcpy(d_k, h_k.data(), k_size, musaMemcpyHostToDevice));
        MUSA_CHECK(musaMemcpy(d_v, h_v.data(), v_size, musaMemcpyHostToDevice));

        // 网格布局：
        // grid.x -> query 位置 t
        // grid.y -> query head h
        // grid.z -> batch b
        dim3 grid_dim(
            target_seq_len,
            query_heads,
            batch_size
        );
        
        int block_size = FLASH_BLOCK_SIZE;

        // 动态共享内存：q_vec + out_accum（各 head_dim 个 float）
        size_t smem_size = 0;
        smem_size += (2 * static_cast<size_t>(head_dim)) * sizeof(float);

        int device = 0;
        MUSA_CHECK(musaGetDevice(&device));

        int max_smem = 0;
        MUSA_CHECK(musaDeviceGetAttribute(&max_smem, musaDevAttrMaxSharedMemoryPerBlock, device));
        if (smem_size > static_cast<size_t>(max_smem)) {
            throw std::invalid_argument("flashAttention: shared memory requirement exceeds device limit.");
        }

        flashAttentionKernel<T><<<grid_dim, block_size, smem_size>>>(
            d_q, d_k, d_v, d_o,
            batch_size, target_seq_len, src_seq_len,
            query_heads, kv_heads, head_dim, is_causal
        );
        
        MUSA_CHECK(musaGetLastError());
        MUSA_CHECK(musaDeviceSynchronize());

        MUSA_CHECK(musaMemcpy(h_o.data(), d_o, o_size, musaMemcpyDeviceToHost));
    } catch (...) {
        if (d_q != nullptr) MUSA_CHECK(musaFree(d_q));
        if (d_k != nullptr) MUSA_CHECK(musaFree(d_k));
        if (d_v != nullptr) MUSA_CHECK(musaFree(d_v));
        if (d_o != nullptr) MUSA_CHECK(musaFree(d_o));
        throw;
    }

    MUSA_CHECK(musaFree(d_q));
    MUSA_CHECK(musaFree(d_k));
    MUSA_CHECK(musaFree(d_v));
    MUSA_CHECK(musaFree(d_o));
}

// *********************************************************************
// Explicit Template Instantiations
// *********************************************************************
template int trace<int>(const std::vector<int>&, size_t, size_t);
template float trace<float>(const std::vector<float>&, size_t, size_t);
template void flashAttention<float>(const std::vector<float>&, const std::vector<float>&,
  const std::vector<float>&, std::vector<float>&,
  int, int, int, int, int, int, bool);
template void flashAttention<half>(const std::vector<half>&, const std::vector<half>&,
  const std::vector<half>&, std::vector<half>&,
  int, int, int, int, int, int, bool);