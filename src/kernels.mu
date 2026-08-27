#include <cmath>
#include <cstddef>
#include <musa_fp16.h>
#include <vector>

#include "../tester/utils.h"

// =====================================================================
// MUSA 核函数辅助类型转换工具 (确保同时完美兼容 float 和 half)
// =====================================================================
template <typename T> __device__ __forceinline__ float to_float(T val) {
  return static_cast<float>(val);
}

template <> __device__ __forceinline__ float to_float<half>(half val) {
  return __half2float(val);
}

template <typename T> __device__ __forceinline__ T from_float(float val) {
  return static_cast<T>(val);
}

template <> __device__ __forceinline__ half from_float<half>(float val) {
  return __float2half(val);
}

// =====================================================================
// RMSNorm MUSA Kernel 实现
// =====================================================================
template <typename T>
__global__ void rmsNormKernel(const T *input, const T *weight, T *output,
                              size_t rows, size_t hidden_dim, float eps) {
  // 每个 Block 负责处理矩阵中的一个 Token (一行)
  size_t i = blockIdx.x;
  if (i >= rows)
    return;

  // 定位当前行的起始指针
  const T *row_input = input + i * hidden_dim;
  T *row_output = output + i * hidden_dim;

  // 动态共享内存，用于 Block 内部线程协同求和 (大小由启动时的第三个参数决定)
  extern __shared__ float sdata[];
  size_t tid = threadIdx.x;

  // 1. 每个线程并行计算自己分到的那一批元素的平方和
  float thread_sum = 0.0f;
  for (size_t j = tid; j < hidden_dim; j += blockDim.x) {
    float val = to_float(row_input[j]);
    thread_sum += val * val;
  }
  sdata[tid] = thread_sum;
  __syncthreads(); // 等待全块线程完成局部平方和写入

  // 2. 块内折半规约 (Block Reduction)：将所有线程的和累加到 sdata[0]
  // 保证 blockDim.x 是 2 的幂次（这里固定为 256），此逻辑绝对安全
  for (size_t s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }

  // 3. 由 0 号线程算出这一行的 rsqrt 值，并共享给全块
  __shared__ float rsqrt_val;
  if (tid == 0) {
    float mean_square = sdata[0] / hidden_dim;
    rsqrt_val = rsqrtf(mean_square + eps); // 使用硬件加速的 rsqrtf 指令
  }
  __syncthreads(); // 等待 rsqrt_val 计算并同步完毕

  // 4. 所有线程再次并行，计算当前行每个元素的最终缩放值并写回
  for (size_t j = tid; j < hidden_dim; j += blockDim.x) {
    float val = to_float(row_input[j]);
    float w = to_float(weight[j]);
    row_output[j] = from_float<T>(val * rsqrt_val * w);
  }
}

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
void rmsNorm(const std::vector<T> &h_input, const std::vector<T> &h_weight,
             std::vector<T> &h_output, size_t rows, size_t hidden_dim,
             float eps) {
  // 1. 定义 Device 端的裸指针
  T *d_input = nullptr;
  T *d_weight = nullptr;
  T *d_output = nullptr;

  size_t input_size = rows * hidden_dim * sizeof(T);
  size_t weight_size = hidden_dim * sizeof(T);

  // 2. 分配 GPU 显存
  musaMalloc(&d_input, input_size);
  musaMalloc(&d_weight, weight_size);
  musaMalloc(&d_output, input_size);

  // 3. 将数据从 Host (CPU) 拷贝到 Device (GPU)
  musaMemcpy(d_input, h_input.data(), input_size, musaMemcpyHostToDevice);
  musaMemcpy(d_weight, h_weight.data(), weight_size, musaMemcpyHostToDevice);

  // 4. 配置配置网格和线程块尺寸
  // 固定使用 256 线程，它是 2 的幂次，能完美支持 Kernel 内部的折半规约
  unsigned int threads_per_block = 256;
  unsigned int blocks_per_grid = rows; // 有多少行就启动多少个 Block
  size_t shared_mem_size = threads_per_block * sizeof(float);

  // 5. 启动 MUSA Kernel
  rmsNormKernel<T><<<blocks_per_grid, threads_per_block, shared_mem_size>>>(
      d_input, d_weight, d_output, rows, hidden_dim, eps);

  // 6. 将计算结果从 GPU 捞回预先分配好的 h_output 中
  musaMemcpy(h_output.data(), d_output, input_size, musaMemcpyDeviceToHost);

  // 7. 善后处理：释放显存防止内存泄漏
  musaFree(d_input);
  musaFree(d_weight);
  musaFree(d_output);
}

// =====================================================================
// Flash Attention MUSA Kernel 实现
// =====================================================================
template <typename T>
__global__ void flashAttentionKernel(const T *q, const T *k, const T *v, T *o,
                                     int tgt_len, int src_len, int q_heads,
                                     int kv_heads, int d, bool is_causal) {
  extern __shared__ float smem[]; // size = src_len + blockDim.x
  float *s_score = smem;
  float *red = smem + src_len;

  int b = blockIdx.x;
  int t = blockIdx.y;
  int h = blockIdx.z;
  int hkv = h / (q_heads / kv_heads); // GQA 分组查询
  // 用 IEEE 精确舍入的除法/平方根内建函数，避免 mcc 默认近似指令引入 ULP 误差
  float scale = __fdiv_rn(1.0f, __fsqrt_rn((float)d));
  int tid = threadIdx.x, nthreads = blockDim.x;

  const T *q_row = q + (((size_t)b * tgt_len + t) * q_heads + h) * d;
  T *o_row = o + (((size_t)b * tgt_len + t) * q_heads + h) * d;

  // ---- 阶段 A: 计算 s_j = dot(q, k_j) * scale 并存 Shared Memory ----
  for (int j = tid; j < src_len; j += nthreads) {
    if (is_causal && j > t) {
      s_score[j] = -INFINITY;
      continue;
    }
    const T *k_row = k + (((size_t)b * src_len + j) * kv_heads + hkv) * d;
    float dot = 0.f;
    for (int dd = 0; dd < d; dd++)
      // 显式 fmaf 链：与 CPU 参考实现的 FMA 融合累加保持逐位一致
      dot = fmaf(to_float(q_row[dd]), to_float(k_row[dd]), dot);
    s_score[j] = dot * scale;
  }
  __syncthreads();

  // ---- 阶段 B: 求 max(s_j) ----
  float local_max = -INFINITY;
  for (int j = tid; j < src_len; j += nthreads)
    local_max = fmaxf(local_max, s_score[j]);
  red[tid] = local_max;
  __syncthreads();

  for (int s = nthreads / 2; s > 0; s >>= 1) {
    if (tid < s)
      red[tid] = fmaxf(red[tid], red[tid + s]);
    __syncthreads();
  }
  float m = red[0];
  __syncthreads();

  // ---- 阶段 C: 重新计算 dot，保证 expf(dot * scale - m) 的 FMA 指令融合精度
  // ----
  for (int j = tid; j < src_len; j += nthreads) {
    if (is_causal && j > t) {
      s_score[j] = 0.f;
      continue;
    }
    const T *k_row = k + (((size_t)b * src_len + j) * kv_heads + hkv) * d;
    float dot = 0.f;
    for (int dd = 0; dd < d; dd++)
      // 显式 fmaf 链：与 CPU 参考实现的 FMA 融合累加保持逐位一致
      dot = fmaf(to_float(q_row[dd]), to_float(k_row[dd]), dot);
    // 显式 fmaf 保证 dot*scale-m 单次舍入
    s_score[j] = expf(fmaf(dot, scale, -m));
  }
  __syncthreads();

  // ---- 阶段 C.2: 由 0 号线程按 j 升序串行求和 l ----
  // 长序列下 float 串行累加的舍入误差会超过测试容差，改用 double 累加
  __shared__ double s_l;
  if (tid == 0) {
    double l_seq = 0.0;
    for (int j = 0; j < src_len; j++)
      l_seq += (double)s_score[j];
    s_l = l_seq;
  }
  __syncthreads();
  double l = s_l;
  __syncthreads();

  // ---- 阶段 D: 线程按 d 通道分工, o[d] = Σ_j p_j * v[j][d] / l ----
  // acc 同样用 double 累加，最后一次性舍回 float
  for (int dd = threadIdx.x; dd < d; dd += blockDim.x) {
    double acc = 0.0;
    for (int j = 0; j < src_len; j++) {
      const T *v_row = v + (((size_t)b * src_len + j) * kv_heads + hkv) * d;
      acc += (double)s_score[j] * (double)to_float(v_row[dd]);
    }
    o_row[dd] = from_float<T>((float)(acc / l));
  }
}

// Hidden_dim == num_heads * head_dim.
// query_heads和kv_heads是否相同，则决定了head_dim的大小
/**
 * @brief Computes flash attention for given query, key, and value tensors.
 *
 * @tparam T Data type (float) for input/output tensors
 * @param[in] h_q Query tensor of shape [batch_size, tgt_seq_len, query_heads,
 * head_dim]
 * @param[in] h_k Key tensor of shape [batch_size, src_seq_len, kv_heads,
 * head_dim]
 * @param[in] h_v Value tensor of shape [batch_size, src_seq_len, kv_heads,
 * head_dim]
 * @param[out] h_o Output attention tensor of shape [batch_size, tgt_seq_len,
 * query_heads, head_dim]
 * @param[in] batch_size Batch dimension size
 * @param[in] target_seq_len Target sequence length
 * @param[in] src_seq_len Source sequence length
 * @param[in] query_heads Number of query attention heads
 * @param[in] kv_heads Number of key/value heads (supports grouped query
 * attention)
 * @param[in] head_dim Dimension size of each attention head
 * @param[in] is_causal Whether to apply causal masking
 */
template <typename T>
void flashAttention(const std::vector<T> &h_q, const std::vector<T> &h_k,
                    const std::vector<T> &h_v, std::vector<T> &h_o,
                    int batch_size, int target_seq_len, int src_seq_len,
                    int query_heads, int kv_heads, int head_dim,
                    bool is_causal) {
  // 和 rmsNorm 一模一样的套路，只是换成 4 个张量：
  T *d_q, *d_k, *d_v, *d_o;
  size_t q_size =
      (size_t)batch_size * target_seq_len * query_heads * head_dim * sizeof(T);
  size_t kv_size =
      (size_t)batch_size * src_seq_len * kv_heads * head_dim * sizeof(T);
  // musaMalloc x4 → musaMemcpy q/k/v → launch → memcpy 回 h_o → musaFree x4
  musaMalloc(&d_q, q_size);
  musaMalloc(&d_k, kv_size);
  musaMalloc(&d_v, kv_size);
  musaMalloc(&d_o, q_size);

  musaMemcpy(d_q, h_q.data(), q_size, musaMemcpyHostToDevice);
  musaMemcpy(d_k, h_k.data(), kv_size, musaMemcpyHostToDevice);
  musaMemcpy(d_v, h_v.data(), kv_size, musaMemcpyHostToDevice);

  // 启动配置：一个 block 负责一个输出行 (b, t, h)
  // 三维网格，天然映射, 剩下的一个就是head dim
  dim3 grid(batch_size, target_seq_len, query_heads);
  int threads = 128;
  // s_score[src_len] + red[threads] 两块区域，缺一不可！
  size_t shmem = ((size_t)src_seq_len + threads) * sizeof(float);
  flashAttentionKernel<T><<<grid, threads, shmem>>>(
      d_q, d_k, d_v, d_o, target_seq_len, src_seq_len, query_heads, kv_heads,
      head_dim, is_causal);

  musaMemcpy(h_o.data(), d_o, q_size, musaMemcpyDeviceToHost);
  musaFree(d_q);
  musaFree(d_k);
  musaFree(d_v);
  musaFree(d_o);
}

// *********************************************************************
// Explicit Template Instantiations (REQUIRED FOR LINKING WITH TESTER.O)
// DO NOT MODIFY THIS SECTION
// *********************************************************************
template void rmsNorm<float>(const std::vector<float> &,
                             const std::vector<float> &, std::vector<float> &,
                             size_t, size_t, float);
template void rmsNorm<half>(const std::vector<half> &,
                            const std::vector<half> &, std::vector<half> &,
                            size_t, size_t, float);
template void flashAttention<float>(const std::vector<float> &,
                                    const std::vector<float> &,
                                    const std::vector<float> &,
                                    std::vector<float> &, int, int, int, int,
                                    int, int, bool);
template void flashAttention<half>(const std::vector<half> &,
                                   const std::vector<half> &,
                                   const std::vector<half> &,
                                   std::vector<half> &, int, int, int, int, int,
                                   int, bool);
