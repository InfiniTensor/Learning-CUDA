#include <vector>
#include <cmath>
#include <cstdlib>
#include <type_traits>
#include <cuda_fp16.h>

#include "../tester/utils.h"

// warp 内 shuffle 归约，结果落在 lane 0
__device__ __forceinline__ float warpReduce(float v) {
  for (int offset = 16; offset > 0; offset >>= 1)
    v += __shfl_down_sync(0xffffffff, v, offset);
  return v;
}

// 简单版：一个 block 管一行，任何 hidden_dim 都能跑
template <typename T>
__global__ void rmsNormKernelNaive(const T* input, const T* weight, T* output,
                                   size_t rows, size_t hidden_dim, float eps) {
  const size_t row = blockIdx.x;
  if (row >= rows) return;

  const T* row_in  = input  + row * hidden_dim;
  T*       row_out = output + row * hidden_dim;

  __shared__ float s_sum[256];               // 和线程数一致

  float local_sum = 0.f;
  for (size_t j = threadIdx.x; j < hidden_dim; j += blockDim.x) {
    float v = (float)row_in[j];
    local_sum += v * v;
  }
  s_sum[threadIdx.x] = local_sum;
  __syncthreads();

  // 树状归约，结果归到线程 0
  for (size_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      s_sum[threadIdx.x] += s_sum[threadIdx.x + stride];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    s_sum[0] = rsqrtf(s_sum[0] / (float)hidden_dim + eps);
  }
  __syncthreads();

  const float scale = s_sum[0];
  for (size_t j = threadIdx.x; j < hidden_dim; j += blockDim.x) {
    row_out[j] = (T)((float)row_in[j] * scale * (float)weight[j]);
  }
}

// 优化版：整行缓存进 shared，float4/half2 向量化，warp 归约
// 仅当 hidden_dim 是 4（float）/2（half）的倍数且 shared <= 48KB 时启用
template <typename T>
__global__ void rmsNormKernel(const T* input, const T* weight, T* output,
                              size_t rows, size_t hidden_dim, float eps) {
  const size_t row = blockIdx.x;
  if (row >= rows) return;

  const T* row_in  = input  + row * hidden_dim;
  T*       row_out = output + row * hidden_dim;

  // 前 hidden_dim 个 float 存整行（half 先转 float），后面 32 个放各 warp 的和
  extern __shared__ float s_data[];
  float* s_row  = s_data;
  float* s_warp = s_data + hidden_dim;

  const int tid  = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;

  float local_sum = 0.f;
  if constexpr (std::is_same<T, float>::value) {
    const float4* in4 = reinterpret_cast<const float4*>(row_in);
    float4* s4 = reinterpret_cast<float4*>(s_row);
    for (size_t j = tid; j < hidden_dim / 4; j += blockDim.x) {
      float4 v = in4[j];
      s4[j] = v;
      local_sum += v.x*v.x + v.y*v.y + v.z*v.z + v.w*v.w;
    }
  } else {
    const half2* in2 = reinterpret_cast<const half2*>(row_in);
    float2* s2 = reinterpret_cast<float2*>(s_row);
    for (size_t j = tid; j < hidden_dim / 2; j += blockDim.x) {
      float2 v = __half22float2(in2[j]);
      s2[j] = v;
      local_sum += v.x*v.x + v.y*v.y;
    }
  }
  __syncthreads();                           // 行数据写完了才能读

  local_sum = warpReduce(local_sum);
  if (lane == 0) s_warp[warp] = local_sum;
  __syncthreads();

  if (tid == 0) {
    float total = 0.f;
    for (int w = 0; w < blockDim.x / 32; ++w) total += s_warp[w];
    s_warp[0] = rsqrtf(total * (1.f / (float)hidden_dim) + eps);
  }
  __syncthreads();

  // 直接读 shared，不用再碰 global input
  const float scale = s_warp[0];
  if constexpr (std::is_same<T, float>::value) {
    float4* s4   = reinterpret_cast<float4*>(s_row);
    float4* out4 = reinterpret_cast<float4*>(row_out);
    for (size_t j = tid; j < hidden_dim / 4; j += blockDim.x) {
      float4 v = s4[j];
      float4 o;
      o.x = v.x * scale * (float)weight[j*4+0];
      o.y = v.y * scale * (float)weight[j*4+1];
      o.z = v.z * scale * (float)weight[j*4+2];
      o.w = v.w * scale * (float)weight[j*4+3];
      out4[j] = o;
    }
  } else {
    float2* s2  = reinterpret_cast<float2*>(s_row);
    half2*  out2 = reinterpret_cast<half2*>(row_out);
    for (size_t j = tid; j < hidden_dim / 2; j += blockDim.x) {
      float2 v = s2[j];
      float2 o;
      o.x = v.x * scale * (float)weight[j*2+0];
      o.y = v.y * scale * (float)weight[j*2+1];
      out2[j] = __float22half2_rn(o);
    }
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
void rmsNorm(const std::vector<T>& h_input, const std::vector<T>& h_weight,
             std::vector<T>& h_output, size_t rows, size_t hidden_dim,
             float eps) {
  if (rows == 0 || hidden_dim == 0) return;

  const size_t numel = rows * hidden_dim;

  T *d_input = nullptr, *d_weight = nullptr, *d_output = nullptr;
  RUNTIME_CHECK(cudaMalloc(&d_input,  numel      * sizeof(T)));
  RUNTIME_CHECK(cudaMalloc(&d_weight, hidden_dim * sizeof(T)));
  RUNTIME_CHECK(cudaMalloc(&d_output, numel      * sizeof(T)));

  RUNTIME_CHECK(cudaMemcpy(d_input,  h_input.data(),  numel      * sizeof(T),
                           cudaMemcpyHostToDevice));
  RUNTIME_CHECK(cudaMemcpy(d_weight, h_weight.data(), hidden_dim * sizeof(T),
                           cudaMemcpyHostToDevice));

  const int threads = 256;
  // 对齐且 shared 够用才走优化版，否则朴素版兜住
  const bool aligned = (std::is_same<T, float>::value) ? (hidden_dim % 4 == 0)
                                                       : (hidden_dim % 2 == 0);
  const size_t smem = hidden_dim * sizeof(float) + 32 * sizeof(float);
  if (aligned && smem <= 48 * 1024) {
    rmsNormKernel<<<(unsigned)rows, threads, smem>>>(d_input, d_weight,
                                                     d_output, rows,
                                                     hidden_dim, eps);
  } else {
    rmsNormKernelNaive<<<(unsigned)rows, threads>>>(d_input, d_weight,
                                                    d_output, rows,
                                                    hidden_dim, eps);
  }

  RUNTIME_CHECK(cudaGetLastError());
  RUNTIME_CHECK(cudaDeviceSynchronize());

  RUNTIME_CHECK(cudaMemcpy(h_output.data(), d_output, numel * sizeof(T),
                           cudaMemcpyDeviceToHost));

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
// head_dim 做成模板参数，寄存器数组按实际大小分配，不然按最大尺寸开数组太浪费
template <typename T, int DIM>
__global__ void flashAttnKernel(const T* q, const T* k, const T* v, T* o,
                                int T_, int S,
                                int QH, int KVH,
                                bool is_causal, float scale) {
  // head_dim >= 256 时 shared 会超 48KB，KV 块缩小到 16
  constexpr int KV_BLOCK = (DIM >= 256) ? 16 : 32;

  const int qblk = blockIdx.x;
  const int b    = blockIdx.y;
  const int h    = blockIdx.z;
  const int tid  = threadIdx.x;
  const int t    = qblk * blockDim.x + tid;

  // 尾块不能提前 return，不然 block 里其他线程会在 __syncthreads 死锁
  const bool active = (t < T_);

  // GQA：第 h 个 query head 用第 h/(QH/KVH) 个 kv head。
  // 注意 K/V 行距是 KVH*D 而不是 D
  const int kv_h = active ? h / (QH / KVH) : 0;
  const T* qrow  = active ? q + (((size_t)b * T_ + t) * QH + h) * DIM : nullptr;
  const T* kbase = active ? k + ((size_t)b * S * KVH + kv_h) * DIM : nullptr;
  const T* vbase = active ? v + ((size_t)b * S * KVH + kv_h) * DIM : nullptr;
  T* orow = active ? o + (((size_t)b * T_ + t) * QH + h) * DIM : nullptr;

  extern __shared__ float s_kv[];
  T* sk = reinterpret_cast<T*>(s_kv);
  T* sv = sk + KV_BLOCK * DIM;

  // Q 行常驻寄存器，KV 循环里不用反复读
  float qq[DIM];
  float oo[DIM];
  for (int d = 0; d < DIM; ++d) { qq[d] = 0.f; oo[d] = 0.f; }
  if (active) {
    for (int d = 0; d < DIM; ++d) qq[d] = (float)qrow[d];
  }
  float m = -1e30f;
  float l = 0.f;

  const int n_kv_blocks = (S + KV_BLOCK - 1) / KV_BLOCK;
  for (int kvblk = 0; kvblk < n_kv_blocks; ++kvblk) {
    // causal 时 key 全在 t 后面的块不用算，但同步还是要等
    const bool skip = !active || (is_causal && kvblk * KV_BLOCK > t);

    if (!skip) {
      // 协作加载 K/V 块
      for (int idx = tid; idx < KV_BLOCK * DIM; idx += blockDim.x) {
        int j = idx / DIM, d = idx % DIM;
        sk[idx] = kbase[(size_t)(kvblk * KV_BLOCK + j) * KVH * DIM + d];
      }
      for (int idx = tid; idx < KV_BLOCK * DIM; idx += blockDim.x) {
        int j = idx / DIM, d = idx % DIM;
        sv[idx] = vbase[(size_t)(kvblk * KV_BLOCK + j) * KVH * DIM + d];
      }
    }
    __syncthreads();

    if (!skip) {
      // 第一遍：score 和块内 max，mask 掉的位置塞 -inf
      float ss[KV_BLOCK];
      float block_max = -1e30f;
      for (int j = 0; j < KV_BLOCK; ++j) {
        const int key_j = kvblk * KV_BLOCK + j;
        const bool valid = (key_j < S) && (!is_causal || key_j <= t);
        float s = 0.f;
        for (int d = 0; d < DIM; ++d) s += qq[d] * (float)sk[j * DIM + d];
        s *= scale;
        ss[j] = valid ? s : -1e30f;
        if (valid && s > block_max) block_max = s;
      }

      // online softmax：基准换成 m_new 后，之前累加的 O、l 要跟着缩放
      const float m_new = fmaxf(m, block_max);
      const float alpha = expf(m - m_new);
      for (int d = 0; d < DIM; ++d) oo[d] *= alpha;
      l *= alpha;

      // 第二遍：p、行和、O 累加
      float row_sum = 0.f;
      for (int j = 0; j < KV_BLOCK; ++j) {
        const float p = expf(ss[j] - m_new);
        row_sum += p;
        for (int d = 0; d < DIM; ++d) oo[d] += p * (float)sv[j * DIM + d];
      }
      l += row_sum;
      m = m_new;
    }
    __syncthreads();                         // 等读完本块再覆写 shared
  }

  if (active) {
    const float inv_l = 1.f / l;
    for (int d = 0; d < DIM; ++d) orow[d] = (T)(oo[d] * inv_l);
  }
}

template <typename T>
void flashAttention(const std::vector<T>& h_q, const std::vector<T>& h_k,
                    const std::vector<T>& h_v, std::vector<T>& h_o,
                    int batch_size, int target_seq_len, int src_seq_len,
                    int query_heads, int kv_heads, int head_dim, bool is_causal) {
  // head_dim 支持 16/32/64/128/256；GQA 要求 QH 是 KVH 的整数倍
  const bool dim_ok = (head_dim == 16 || head_dim == 32 || head_dim == 64 ||
                       head_dim == 128 || head_dim == 256);
  if (!dim_ok || query_heads % kv_heads != 0) {
    std::cerr << "flashAttention: unsupported config (head_dim=" << head_dim
              << ", QH=" << query_heads << ", KVH=" << kv_heads << ")\n";
    exit(EXIT_FAILURE);
  }
  if (batch_size <= 0 || target_seq_len <= 0 || src_seq_len <= 0) return;

  const size_t q_numel = (size_t)batch_size * target_seq_len * query_heads * head_dim;
  const size_t kv_numel = (size_t)batch_size * src_seq_len * kv_heads * head_dim;

  T *d_q = nullptr, *d_k = nullptr, *d_v = nullptr, *d_o = nullptr;
  RUNTIME_CHECK(cudaMalloc(&d_q, q_numel * sizeof(T)));
  RUNTIME_CHECK(cudaMalloc(&d_k, kv_numel * sizeof(T)));
  RUNTIME_CHECK(cudaMalloc(&d_v, kv_numel * sizeof(T)));
  RUNTIME_CHECK(cudaMalloc(&d_o, q_numel * sizeof(T)));

  RUNTIME_CHECK(cudaMemcpy(d_q, h_q.data(), q_numel * sizeof(T), cudaMemcpyHostToDevice));
  RUNTIME_CHECK(cudaMemcpy(d_k, h_k.data(), kv_numel * sizeof(T), cudaMemcpyHostToDevice));
  RUNTIME_CHECK(cudaMemcpy(d_v, h_v.data(), kv_numel * sizeof(T), cudaMemcpyHostToDevice));

  const int threads = 128;
  dim3 grid((unsigned)((target_seq_len + threads - 1) / threads),
            (unsigned)batch_size, (unsigned)query_heads);
  const float scale = 1.f / sqrtf((float)head_dim);

#define LAUNCH_FLASH(DIM)                                                     \
  do {                                                                        \
    constexpr int KVB = ((DIM) >= 256) ? 16 : 32;                             \
    const size_t smem = 2 * KVB * (DIM) * sizeof(T);                          \
    flashAttnKernel<T, DIM><<<grid, threads, smem>>>(                         \
        d_q, d_k, d_v, d_o, target_seq_len, src_seq_len,                      \
        query_heads, kv_heads, is_causal, scale);                             \
  } while (0)

  switch (head_dim) {
    case 16:  LAUNCH_FLASH(16);  break;
    case 32:  LAUNCH_FLASH(32);  break;
    case 64:  LAUNCH_FLASH(64);  break;
    case 128: LAUNCH_FLASH(128); break;
    case 256: LAUNCH_FLASH(256); break;
    default:  break;  // dim_ok 已挡住
  }
#undef LAUNCH_FLASH

  RUNTIME_CHECK(cudaGetLastError());
  RUNTIME_CHECK(cudaDeviceSynchronize());

  RUNTIME_CHECK(cudaMemcpy(h_o.data(), d_o, q_numel * sizeof(T), cudaMemcpyDeviceToHost));

  cudaFree(d_q); cudaFree(d_k); cudaFree(d_v); cudaFree(d_o);
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
