#include <vector>
#include <cuda_fp16.h>
#include "../tester/utils.h"

template <typename T>
__global__ void trace_kernel(const T* __restrict__ input,
                             T* __restrict__ block_sums,
                             size_t rows,
                             size_t cols) {
    constexpr unsigned int block_size = 256;
    __shared__ T shared_sums[block_size];

    const size_t diag_size = rows < cols ? rows : cols;
    const unsigned int tid = threadIdx.x;
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 线程内累加本线程负责的对角元素
    T thread_sum = T(0);
    for (size_t i = idx; i < diag_size; i += gridDim.x * blockDim.x) {
        thread_sum += input[i * cols + i];
    }
    shared_sums[tid] = thread_sum;
    __syncthreads();

    // block 内归约为一个部分和
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sums[tid] += shared_sums[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        block_sums[blockIdx.x] = shared_sums[0];
    }
}

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
T trace(const std::vector<T>& h_input, size_t rows, size_t cols) {
    if (h_input.empty() || rows == 0 || cols == 0) {
        return T(0);
    }

    T* d_input = nullptr;
    T* d_output = nullptr;
    T h_output = T(0);

    const size_t input_size = h_input.size() * sizeof(T);
    const size_t diag_size = rows < cols ? rows : cols;

    constexpr unsigned int block_size = 256;
    const unsigned int grid_size = (diag_size + block_size - 1) / block_size;

    cudaMalloc(&d_input, input_size);
    cudaMalloc(&d_output, grid_size * sizeof(T));

    cudaMemcpy(d_input, h_input.data(), input_size, cudaMemcpyHostToDevice);
    cudaMemset(d_output, 0, grid_size * sizeof(T));

    trace_kernel<T><<<grid_size, block_size>>>(d_input, d_output, rows, cols);

    // 回到 host 上做一次确定性的最终归约
    std::vector<T> h_block_sums(grid_size);
    cudaMemcpy(h_block_sums.data(),d_output,grid_size * sizeof(T),cudaMemcpyDeviceToHost);

    for (unsigned int i = 0; i < grid_size; ++i) {
        h_output += h_block_sums[i];
    }

    cudaFree(d_input);
    cudaFree(d_output);

    return h_output;
}

template <typename T>
__global__ void flash_attention_kernel(
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
    int block_size_q,         // 一个block处理的query数量
    int block_size_kv,        // KV tile的宽度
    bool is_causal,
    float scale) {
    
    const int batch_idx = blockIdx.z;
    const int q_head_idx = blockIdx.y;    
    const int q_block_idx = blockIdx.x;       // 当前block负责的Q tile index
    const int q_start = q_block_idx * block_size_q;
    const int q_end = min(q_start + block_size_q, tgt_seq_len);
    const int q_block_len = q_end - q_start;
    const int kv_head_idx = (q_head_idx * kv_heads) / query_heads;       // GQA映射：先乘后除，避免kv_head/query_heads等于0
    
    extern __shared__ char shared_mem[];
    size_t offset = 0;

    // Q tile：[block_size_q, head_dim]
    T* q_shared = reinterpret_cast<T*>(shared_mem + offset);
    offset += block_size_q * head_dim * sizeof(T);
    
    // K tile：[block_size_kv, head_dim]
    T* k_shared = reinterpret_cast<T*>(shared_mem + offset);
    offset += block_size_kv * head_dim * sizeof(T);
    
    // V tile：[block_size_kv, head_dim]
    T* v_shared = reinterpret_cast<T*>(shared_mem + offset);
    offset += block_size_kv * head_dim * sizeof(T);
    
    // s_shared[block_size_q, block_size_kv]: 当前Q/KV tile的所有score，用doulbe存储，避免精度丢失
    offset = (offset + sizeof(double) - 1) / sizeof(double) * sizeof(double);  // 对齐到double的边界，避免多次访存
    double* s_shared = reinterpret_cast<double*>(shared_mem + offset);
    offset += block_size_q * block_size_kv * sizeof(double);
    
    // o_shared[block_size_q, head_dim]: 当前Q位置、head维度的未归一化输出累计值，用double存储，避免精度丢失
    offset = (offset + sizeof(double) - 1) / sizeof(double) * sizeof(double);
    double* o_shared = reinterpret_cast<double*>(shared_mem + offset);
    offset += block_size_q * head_dim * sizeof(double);
    
    // online softmax对应当前全局最大的logit
    offset = (offset + sizeof(double) - 1) / sizeof(double) * sizeof(double);
    double* m_shared = reinterpret_cast<double*>(shared_mem + offset);
    offset += block_size_q * sizeof(double);
    
    // online softmax对应的当前全局归一化分母
    double* l_shared = reinterpret_cast<double*>(shared_mem + offset);
    
    const int tid = threadIdx.x;
    const int num_threads = blockDim.x;
    
    // 初始化m，l，o
    for (int idx = tid; idx < q_block_len; idx += num_threads) {
        m_shared[idx] = -INFINITY;
        l_shared[idx] = 0.0;
        for (int d = 0; d < head_dim; d++) {
            o_shared[idx * head_dim + d] = 0.0;  // 未归一化的输出累加器（double）
        }
    }
    __syncthreads();
    
    // 将Q tile 从global memory加载到shared memory
    for (int idx = tid; idx < q_block_len * head_dim; idx += num_threads) {
        int q_idx = idx / head_dim;
        int d_idx = idx % head_dim;
        int global_q_idx = q_start + q_idx;
        if (global_q_idx < tgt_seq_len) {
            int offset = batch_idx * tgt_seq_len * query_heads * head_dim +
                         global_q_idx * query_heads * head_dim +
                         q_head_idx * head_dim + d_idx;
            q_shared[idx] = Q[offset];
        } else {
            q_shared[idx] = T(0);
        }
    }
    __syncthreads();
    
    const int num_kv_blocks = (src_seq_len + block_size_kv - 1) / block_size_kv;
    
    // 外循环：遍历KV tile
    for (int kv_block_idx = 0; kv_block_idx < num_kv_blocks; kv_block_idx++) {
        const int kv_start = kv_block_idx * block_size_kv;
        const int kv_end = min(kv_start + block_size_kv, src_seq_len);
        const int kv_block_len = kv_end - kv_start;
        const double scale_d = static_cast<double>(scale);
        
        // 内循环：加载当前KV tile到shared memory
        for (int idx = tid; idx < kv_block_len * head_dim; idx += num_threads) {
            int kv_idx = idx / head_dim;
            int d_idx = idx % head_dim;
            int global_kv_idx = kv_start + kv_idx;
            if (global_kv_idx < src_seq_len) {
                int k_offset = batch_idx * src_seq_len * kv_heads * head_dim +
                              global_kv_idx * kv_heads * head_dim +
                              kv_head_idx * head_dim + d_idx;
                int v_offset = batch_idx * src_seq_len * kv_heads * head_dim +
                              global_kv_idx * kv_heads * head_dim +
                              kv_head_idx * head_dim + d_idx;
                k_shared[idx] = K[k_offset];
                v_shared[idx] = V[v_offset];
            } else {
                k_shared[idx] = T(0);
                v_shared[idx] = T(0);
            }
        }
        __syncthreads();
        
        // 计算局部score
        for (int idx = tid; idx < q_block_len * kv_block_len; idx += num_threads) {
            int q_idx = idx / kv_block_len;
            int kv_idx = idx % kv_block_len;
            int global_q_idx = q_start + q_idx;
            int global_kv_idx = kv_start + kv_idx;
            bool mask = (is_causal && global_q_idx < global_kv_idx);

            if (mask) {
                s_shared[q_idx * block_size_kv + kv_idx] = -INFINITY;
                continue;
            }

            // 均使用float作为中间量，避免精度丢失
            float sum = 0.0;
            for (int d = 0; d < head_dim; d++) {
                float q_val = static_cast<float>(q_shared[q_idx * head_dim + d]);
                float k_val = static_cast<float>(k_shared[kv_idx * head_dim + d]);
                sum += q_val * k_val;
            }
            s_shared[q_idx * block_size_kv + kv_idx] = sum * scale_d;
        }
        __syncthreads();
        
        // ─────────────────────────────────────────────────────────────────
        // Online softmax：每处理一个 KV tile，增量更新 m/l/o
        // 流程：① 取当前 tile 的 max → ② 算 rescaling 因子 alpha
        //      ③ 累加 p_sum（softmax 分母）→ ④ 累加 o（softmax 分子 × V）
        // ─────────────────────────────────────────────────────────────────
        for (int q_idx = tid; q_idx < q_block_len; q_idx += num_threads) {
            int global_q_idx = q_start + q_idx;

            // ① 读取上一 tile 的状态
            float m_old = static_cast<float>(m_shared[q_idx]);
            float l_old = static_cast<float>(l_shared[q_idx]);

            // ② 当前 tile 内的 max
            float m_ij = -static_cast<float>(INFINITY);
            for (int kv_idx = 0; kv_idx < kv_block_len; kv_idx++) {
                int global_kv_idx = kv_start + kv_idx;
                if (is_causal && global_q_idx < global_kv_idx) continue;
                float s_val = s_shared[q_idx * block_size_kv + kv_idx];
                m_ij = fmax(m_ij, s_val);
            }

            double m_new = fmax(m_old, m_ij);
            if (m_new == -INFINITY) continue;

            // ③ rescaling 因子：max 变大时，旧值需整体缩小
            double alpha = exp(m_old - m_new);

            // ④ 计算并缓存 p_ij = exp(s_ij - m_new)（类似 v2 的 sSafeE）
            //    然后复用 p_ij 同时完成分母 p_sum 的累加，避免在后续 O 累加里重复计算 exp()
            const int score_row = q_idx * block_size_kv;
            double p_sum = 0.0;
            double p_err = 0.0;
            for (int kv_idx = 0; kv_idx < kv_block_len; kv_idx++) {
                const int global_kv_idx = kv_start + kv_idx;
                if (is_causal && global_q_idx < global_kv_idx) {
                    // masked 的位置等价于 exp(-inf) = 0
                    s_shared[score_row + kv_idx] = 0.0;
                    continue;
                }
                const double p_ij = exp(s_shared[score_row + kv_idx] - m_new);
                s_shared[score_row + kv_idx] = p_ij;
                // Kahan summation for p_sum
                const double y = p_ij - p_err;
                const double t = p_sum + y;
                p_err = (t - p_sum) - y;
                p_sum = t;
            }

            // ⑤ o_new = alpha * o_old + Σ p_ij * v，当前 tile 的输出贡献（Kahan 累加）
            for (int d = 0; d < head_dim; d++) {
                double o_new_val = alpha * o_shared[q_idx * head_dim + d];
                double o_err = 0.0;
                for (int kv_idx = 0; kv_idx < kv_block_len; kv_idx++) {
                    const int global_kv_idx = kv_start + kv_idx;
                    if (is_causal && global_q_idx < global_kv_idx) continue;
                    // 复用缓存的 p_ij（已写回 s_shared）
                    const double p_ij = s_shared[score_row + kv_idx];
                    const double term = p_ij * static_cast<double>(v_shared[kv_idx * head_dim + d]);
                    const double y = term - o_err;
                    const double t = o_new_val + y;
                    o_err = (t - o_new_val) - y;
                    o_new_val = t;
                }
                o_shared[q_idx * head_dim + d] = o_new_val;
            }

            // ⑥ 更新 m、l，供下一 tile 使用
            m_shared[q_idx] = m_new;
            l_shared[q_idx] = alpha * l_old + p_sum;
        }
        __syncthreads();
    }
    
    // 用 l_shared 做归一化，并把结果写回全局内存
    // o_shared 存的是未归一化的累积值，这里统一除以 l_shared
    for (int idx = tid; idx < q_block_len * head_dim; idx += num_threads) {
        int q_idx = idx / head_dim;
        int d_idx = idx % head_dim;
        int global_q_idx = q_start + q_idx;
        
        if (global_q_idx < tgt_seq_len && l_shared[q_idx] > 0.0) {
            double o_unnorm = o_shared[idx];
            double o_val = o_unnorm / l_shared[q_idx];
            int offset = batch_idx * tgt_seq_len * query_heads * head_dim +
                        global_q_idx * query_heads * head_dim +
                        q_head_idx * head_dim + d_idx;
            O[offset] = static_cast<T>(o_val);
        }
    }
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
void flashAttention(const std::vector<T>& h_q, const std::vector<T>& h_k,
                    const std::vector<T>& h_v, std::vector<T>& h_o,
                    int batch_size, int target_seq_len, int src_seq_len, 
                    int query_heads, int kv_heads, int head_dim, bool is_causal) {
    
    // 校验输入参数是否合法
    if (h_q.empty() || h_k.empty() || h_v.empty() ||
        batch_size <= 0 || target_seq_len <= 0 || src_seq_len <= 0 ||
        query_heads <= 0 || kv_heads <= 0 || head_dim <= 0) {
        h_o.resize(batch_size * target_seq_len * query_heads * head_dim, T(0));
        return;
    }
    
    // 调整输出向量大小
    h_o.resize(batch_size * target_seq_len * query_heads * head_dim);
    
    // 申请 device 端显存
    T* d_q = nullptr;
    T* d_k = nullptr;
    T* d_v = nullptr;
    T* d_o = nullptr;
    
    const size_t q_size = batch_size * target_seq_len * query_heads * head_dim * sizeof(T);
    const size_t k_size = batch_size * src_seq_len * kv_heads * head_dim * sizeof(T);
    const size_t v_size = batch_size * src_seq_len * kv_heads * head_dim * sizeof(T);
    const size_t o_size = batch_size * target_seq_len * query_heads * head_dim * sizeof(T);
    
    RUNTIME_CHECK(cudaMalloc(&d_q, q_size));
    RUNTIME_CHECK(cudaMalloc(&d_k, k_size));
    RUNTIME_CHECK(cudaMalloc(&d_v, v_size));
    RUNTIME_CHECK(cudaMalloc(&d_o, o_size));
    
    // 将 Host 数据拷贝到 Device
    RUNTIME_CHECK(cudaMemcpy(d_q, h_q.data(), q_size, cudaMemcpyHostToDevice));
    RUNTIME_CHECK(cudaMemcpy(d_k, h_k.data(), k_size, cudaMemcpyHostToDevice));
    RUNTIME_CHECK(cudaMemcpy(d_v, h_v.data(), v_size, cudaMemcpyHostToDevice));
    RUNTIME_CHECK(cudaMemset(d_o, 0, o_size));
    
    // float 和 half 都使用 double 作为中间量以保证精度
    // 计算转成double所需的空间
    auto align_double = [](size_t x) {
        return (x + sizeof(double) - 1) / sizeof(double) * sizeof(double);
    };
    auto smem_needed = [&](int bq, int bkv) {
        size_t off = 0;
        off += (static_cast<size_t>(bq) + 2ull * static_cast<size_t>(bkv)) *
               static_cast<size_t>(head_dim) * sizeof(T); // Q/K/V
        off = align_double(off);
        off += static_cast<size_t>(bq) * static_cast<size_t>(bkv) *
               sizeof(double);                              // scores
        off = align_double(off);
        off += static_cast<size_t>(bq) * static_cast<size_t>(head_dim) *
               sizeof(double);                              // o accumulator
        off = align_double(off);
        off += 2ull * static_cast<size_t>(bq) * sizeof(double); // m / l
        return off;
    };

    // 根据类型选择合适的 tile 大小
    // block_size_q:一个block负责的query个数，即每个block负责的Q的行数
    // block_size_kv:一个KV tile里的key/value个数，即每个block负责的KV的行数
    int block_size_q, block_size_kv;
    if constexpr (std::is_same_v<T, float>) {
        // float 路径使用更大的 tile 以减少重标定次数
        block_size_q  = (head_dim >= 128) ? 16 : ((head_dim > 64) ? 32 : 32);
        block_size_kv = (head_dim >= 128) ? 32 : ((head_dim > 64) ? 64 : 64);
    } else {
        // half 路径使用标准 tile 大小
        block_size_q  = (head_dim >= 128) ? 8 : ((head_dim > 64) ? 8 : 16);
        block_size_kv = (head_dim >= 128) ? 16 : ((head_dim > 64) ? 32 : 32);
    }

    size_t shared_mem_size = smem_needed(block_size_q, block_size_kv);
    const size_t max_shared_mem = 48 * 1024;
    if (shared_mem_size > max_shared_mem) {
        block_size_q = 4;
        block_size_kv = 8;
        shared_mem_size = smem_needed(block_size_q, block_size_kv);
    }
    
    // 缩放因子：1/sqrt(head_dim)
    float scale = 1.0f / sqrtf(static_cast<float>(head_dim));
    
    const int num_q_blocks = (target_seq_len + block_size_q - 1) / block_size_q;
    dim3 grid(num_q_blocks, query_heads, batch_size);
    dim3 block(256);  // 每个 block 的线程数
    
    // 统一使用 Flash Attention kernel（float 和 half 都使用 double 作为中间量）
    flash_attention_kernel<T><<<grid, block, shared_mem_size>>>(
        d_q, d_k, d_v, d_o,
        batch_size, target_seq_len, src_seq_len,
        query_heads, kv_heads, head_dim,
        block_size_q, block_size_kv,
        is_causal, scale
    );
    
    RUNTIME_CHECK(cudaGetLastError());
    RUNTIME_CHECK(cudaDeviceSynchronize());
    
    // 将结果从 Device 拷贝回 Host
    RUNTIME_CHECK(cudaMemcpy(h_o.data(), d_o, o_size, cudaMemcpyDeviceToHost));
    
    // 释放 device 端显存
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
