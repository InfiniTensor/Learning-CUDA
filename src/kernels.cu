#include <vector>
#include <cuda_fp16.h>

#include "../tester/utils.h"

//定义一个错误检查函数
#define CUDA_CHECK(call)   cudaCheck(call, __FILE__, __LINE__)

static void cudaCheck(cudaError_t err, const char* file, const int line) {
    if (err != cudaSuccess) {
        printf("ERROR: %s, %d,", file, line);
        printf("CODE: %s, DETAIL: %s\n", cudaGetErrorName(err), cudaGetErrorString(err));
        exit(EXIT_FAILURE);
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
__global__ void rmsNorm_kernel(const T *d_input, const T *d_weight, T *d_output, size_t rows, size_t hidden_dim, float eps) {
    const size_t row = blockIdx.x;//一个block处理一行，多个block一起
    const size_t tid = threadIdx.x;//一行多个线程一起处理

    //求每个元素的平方和
    // __shared__ T sum_temp[1024]; 啊”T= half“会出错
    __shared__ float sum_temp[1024];
    float sum = 0.0f;
    for (int i = tid; i < hidden_dim; i += blockDim.x) {
      //hidden_dim应该会比较大，单个线程求取对应的元素平方和
        float val = static_cast<float>(d_input[row*hidden_dim + i]);//取值
        sum += val * val;
    }

    sum_temp[tid] = sum;//存储每个线程单独计算的平方和
    __syncthreads();

    //规约求和
    for (int i = blockDim.x / 2; i > 0; i >>= 1) {
      if (tid < i) {
        sum_temp[tid] += sum_temp[tid+i];
      }
      __syncthreads();
    }

    //计算公式
    float temp = rsqrt(sum_temp[0] / static_cast<float>(hidden_dim) + eps);

    for (int i = tid; i < hidden_dim; i += blockDim.x) {
      float x = static_cast<float>(d_input[row*hidden_dim + i]);
      float w = static_cast<float>(d_weight[i]);
      d_output[row*hidden_dim + i] = x * w * temp;
    }
}

template <typename T>
void rmsNorm(const std::vector<T>& h_input, const std::vector<T>& h_weight,
              std::vector<T>& h_output, size_t rows, size_t hidden_dim,
              float eps) {
  // TODO: Implement the rmsNorm function
  //主机内存初始化
  size_t input_bytes = rows * hidden_dim * sizeof(T);
  size_t weight_bytes = hidden_dim * sizeof(T);
  size_t output_bytes = input_bytes;
  T *d_input = nullptr; 
  T *d_weight = nullptr; 
  T *d_output = nullptr;

  //分配GPU显存
  CUDA_CHECK(cudaMalloc(&d_input, input_bytes));
  CUDA_CHECK(cudaMalloc(&d_weight, weight_bytes));
  CUDA_CHECK(cudaMalloc(&d_output, output_bytes));

  //将主机数据放入显存上
  CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), input_bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_weight, h_weight.data(), weight_bytes, cudaMemcpyHostToDevice));

  //核函数
  dim3 block(1024);//一个block最多1024个线程
  dim3 grid(rows);//一个block处理一行数据
  rmsNorm_kernel<T><<<grid, block>>>(d_input, d_weight, d_output, rows, hidden_dim, eps);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  //将显存数据拷回CPU
  CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, output_bytes, cudaMemcpyDeviceToHost));

  //释放显存内存
  CUDA_CHECK(cudaFree(d_input));
  CUDA_CHECK(cudaFree(d_weight));
  CUDA_CHECK(cudaFree(d_output));
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
/**constexpr int K = 32; //分块的大小，不能太大了，容易出错
template <typename T>
__global__ void flashAttention_kernel(const T* d_q, const T*d_k, const T*d_v, T* d_o, int batch_size, int target_seq_len, int src_seq_len, 
                    int query_heads, int kv_heads, int head_dim, bool is_causal ) {
    if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0)
    {
        printf("DEBUG kernel sizeof(T)=%ld\n", sizeof(T));
    }
    int batch = blockIdx.x;
    int headId = blockIdx.y;
    int tid = threadIdx.x; //一个线程处理一块的一整行
    
    //GQA分组
    int groups = query_heads / kv_heads;
    int kv_head = headId / groups;

    float alpha = 1.0f / sqrtf(static_cast<float>(head_dim)); //α系数

    //分配共享内存给qkv
    extern __shared__ float smem[];
    float* q_tile = smem;
    float* k_tile = smem + K * head_dim;
    float* v_tile = smem + 2 * K * head_dim;

    //计算循环次数
    int q_blocks  = (target_seq_len + K - 1) / K; //向上取整
    int kv_blocks  = (src_seq_len + K - 1) / K;

    //双重循环，外层Q，内层KV
    for (int q_block = 0; q_block < q_blocks; q_block++) {
      int global_q_row = q_block * K + tid;
      bool flag = global_q_row < target_seq_len; 
      for (int d = 0; d < head_dim; d++) {
        if (flag) {
          q_tile[tid * head_dim + d] = static_cast<float>(d_q[batch * target_seq_len * query_heads * head_dim
                                                          + global_q_row * query_heads * head_dim
                                                          + headId * head_dim + d]); //取出分块的一行
        } else {
          q_tile[tid * head_dim + d] = 0.0f;
        }
      }
      __syncthreads();

      //if (!flag) continue;

      float o_reg[128];
      for (int d = 0; d < head_dim; d++) {
            o_reg[d] = 0.0;
      }
      // float old_max = -INFINITY;
      // float old_l = 0.0f;
      float old_max = -1e20f;
      float old_l = 0;

      //内层循环，KV
      for (int kv_block = 0; kv_block < kv_blocks; kv_block++) {
        int kv_start = kv_block * K;
        int global_k_row = kv_start + tid;
        bool flag2 = global_k_row < src_seq_len;
        
        for (int d = 0; d < head_dim; d++) {
          if (flag2) {
              k_tile[tid * head_dim + d] = static_cast<float>(
                  d_k[batch * src_seq_len * kv_heads * head_dim
                      + global_k_row * kv_heads * head_dim
                      + kv_head * head_dim + d]);
              v_tile[tid * head_dim + d] = static_cast<float>(
                  d_v[batch * src_seq_len * kv_heads * head_dim
                      + global_k_row * kv_heads * head_dim
                      + kv_head * head_dim + d]);
          } else {
              k_tile[tid * head_dim + d] = 0.0f;
              v_tile[tid * head_dim + d] = 0.0f;
          }
        }
        __syncthreads();

        if (flag) {
          float s_row[K]; 
          float s_max=-1e20f;

          for (int j = 0; j < K; j++) { //第global_q_row行 X 第一列， 第global_q_row行 X 第二列
            int global_k_idx = kv_start + j;
            if (global_k_idx >= src_seq_len) {
                s_row[j] = -INFINITY;      // 越界行忽略
                continue;
            }

            if (is_causal && global_k_idx > global_q_row) {
                s_row[j] = -INFINITY;      
                continue;
            }

            float sum = 0.0f;
            for (int d =0; d < head_dim; d++) {
              sum += q_tile[tid * head_dim + d] * k_tile[j * head_dim + d];
            }
            sum *= alpha;
            s_row[j] = sum;
            s_max = max(s_max, sum);
          } //求出s矩阵的一行

          // if(s_max==-INFINITY) continue;

          // float new_max = max (old_max, s_max);
          // float scale = expf(old_max - new_max);

          // for (int d = 0; d < head_dim; d++) {
          //   o_reg[d] *= scale;
          // }
          // float new_l = scale * old_l;

          // //softmax
          // for (int j = 0; j < K; j++) {
          //   if (s_row[j] == -INFINITY) continue;
          //   float p = expf(s_row[j] - new_max);
          //   new_l += p;
          //   for (int d = 0; d < head_dim; d++) {
          //     o_reg[d] += p * v_tile[j * head_dim + d];
          //   }
          // }

          // old_max = new_max;
          // old_l = new_l;
          if (s_max > -INFINITY) {
              float new_max = max(old_max, s_max);
              float scale = expf(old_max - new_max);
              // float scale = (old_max == -INFINITY) ? 0.0f : expf(old_max - new_max);

              for (int d = 0; d < head_dim; d++) {
                  o_reg[d] *= scale;
              }
              float new_l = scale * old_l;

              for (int j = 0; j < K; j++) {
                  if (s_row[j] == -INFINITY) continue;
                  float p = expf(s_row[j] - new_max);
                  new_l += p;
                  for (int d = 0; d < head_dim; d++) {
                      o_reg[d] += p * v_tile[j * head_dim + d];
                  }
              }

              old_max = new_max;
              old_l = new_l;
          }
        }
        __syncthreads();
      }

      if (flag) {
        float l = (old_l > 0.0f) ? (1.0f / old_l) : 0.0f;
        for (int d = 0; d < head_dim; d++) {
          d_o[batch * target_seq_len * query_heads * head_dim
            + global_q_row * query_heads * head_dim
            + headId * head_dim + d] = static_cast<T>(o_reg[d] * l);
        }
      }
      __syncthreads();
    }  
}*/

constexpr int K = 32;
constexpr int MAX_HEAD_DIM = 128;
template <typename T>
__global__ void flashAttention_kernel(const T* d_q, const T* d_k, const T* d_v, T* d_o, int batch_size, int target_seq_len, int src_seq_len, int query_heads, int kv_heads, int head_dim, bool is_causal) {
  int batch  = blockIdx.x;
  int headId = blockIdx.y;
  int tid    = threadIdx.x;

  //GQA分组
  int groups  = query_heads / kv_heads;
  int kv_head = headId / groups;

  float alpha = 1.0f / sqrtf(static_cast<float>(head_dim)); //α系数

  //分配smem
  extern __shared__ float smem[];
  float* q_tile = smem;
  float* k_tile = q_tile + K * head_dim;
  float* v_tile = k_tile + K * head_dim;

  //循环多少次
  int q_blocks  = (target_seq_len + K - 1) / K;
  int kv_blocks = (src_seq_len + K - 1) / K;

  //双层循环，外层q，内层KV
  for (int q_block = 0; q_block < q_blocks; q_block++) {
      int global_q_row = q_block * K + tid;
      bool valid_q = global_q_row < target_seq_len; //target_seq_len不一定是分块的倍数，下面同理
      for (int d = 0; d < head_dim; d++) {
          if (valid_q) {
              size_t q_idx = ((size_t)batch * target_seq_len * query_heads + (size_t)global_q_row * query_heads + headId) * head_dim + d;
              q_tile[tid * head_dim + d] = static_cast<float>(d_q[q_idx]);
          } else {
              q_tile[tid * head_dim + d] = 0.0f;
          }
      }
      __syncthreads();

      //加载K计算QK得到一整行的最大值
      float row_max = -INFINITY;
      for (int kv_block = 0; kv_block < kv_blocks; kv_block++) {
          int kv_start = kv_block * K;
          int global_k_row = kv_start + tid;
          bool valid_k = global_k_row < src_seq_len;

          for (int d = 0; d < head_dim; d++) {
              if (valid_k) {
                  size_t k_idx = ((size_t)batch * src_seq_len * kv_heads + (size_t)global_k_row * kv_heads + kv_head) * head_dim + d;
                  k_tile[tid * head_dim + d] = static_cast<float>(d_k[k_idx]);
              } else {
                  k_tile[tid * head_dim + d] = 0.0f;
              }
          }
          __syncthreads();

          if (valid_q) {
              for (int j = 0; j < K; j++) {
                  int global_k_idx = kv_start + j;

                  //KV越界
                  if (global_k_idx >= src_seq_len)
                      continue;

                  //因果掩码
                  if (is_causal && global_k_idx > global_q_row)
                      continue;

                  float s = 0.0f;
                  for (int d = 0; d < head_dim; ++d) {
                      s += q_tile[tid * head_dim + d] * k_tile[j * head_dim + d];
                  }
                  s *= alpha;
                  row_max = fmaxf(row_max, s);
              }
          }
          __syncthreads();
      } //第global_q_row一整行的值求出来了，也就是矩阵s的一行

      float o_reg[MAX_HEAD_DIM];
      for (int d = 0; d < head_dim; ++d) {
          o_reg[d] = 0.0f;
      }

      float row_sum = 0.0f;
      for (int kv_block = 0; kv_block < kv_blocks; kv_block++) {
          int kv_start = kv_block * K;
          int global_k_row = kv_start + tid;
          bool valid_k = global_k_row < src_seq_len;

          //加载KV
          for (int d = 0; d < head_dim; d++) {
              if (valid_k) {
                  size_t k_idx = ((size_t)batch * src_seq_len * kv_heads + (size_t)global_k_row * kv_heads + kv_head) * head_dim + d;
                  k_tile[tid * head_dim + d] = static_cast<float>(d_k[k_idx]);
                  v_tile[tid * head_dim + d] =static_cast<float>(d_v[k_idx]);
              } else {
                  k_tile[tid * head_dim + d] = 0.0f;
                  v_tile[tid * head_dim + d] = 0.0f;
              }
          }
          __syncthreads();

          if (valid_q && row_max != -INFINITY) {
              for (int j = 0; j < K; j++) {
                  int global_k_idx = kv_start + j;
                  if (global_k_idx >= src_seq_len)
                      continue;

                  if (is_causal && global_k_idx > global_q_row)
                      continue;

                  //重新计算s
                  float s = 0.0f;

                  for (int d = 0; d < head_dim; ++d) {
                      s += q_tile[tid * head_dim + d] * k_tile[j * head_dim + d];
                  }
                  s *= alpha;

                  //sotfmax
                  float p = expf(s - row_max);
                  row_sum += p; //一整行的和

                  //计算PV
                  for (int d = 0; d < head_dim; d++) {
                      o_reg[d] += p * v_tile[j * head_dim + d];
                  }
              }
          }
          __syncthreads();
      }

      //写回d_o
      if (valid_q) {
          float inv_sum = row_sum > 0.0f ? 1.0f / row_sum : 0.0f;
          for (int d = 0; d < head_dim; d++) {
              size_t o_idx = ((size_t)batch * target_seq_len * query_heads + (size_t)global_q_row * query_heads + headId) * head_dim + d;
              d_o[o_idx] = static_cast<T>(o_reg[d] * inv_sum);
          }
      }
      __syncthreads();
  }
}

template <typename T>
void flashAttention(const std::vector<T>& h_q, const std::vector<T>& h_k,
                    const std::vector<T>& h_v, std::vector<T>& h_o,
                    int batch_size, int target_seq_len, int src_seq_len, 
                    int query_heads, int kv_heads, int head_dim, bool is_causal) {       
  // TODO: Implement the flash attention function
  //主机内存初始化
  size_t q_bytes = (size_t)batch_size * target_seq_len * query_heads * head_dim * sizeof(T);
  size_t kv_bytes = (size_t)batch_size * src_seq_len * kv_heads * head_dim * sizeof(T);
  size_t o_bytes = q_bytes;

  T *d_q = nullptr, *d_k = nullptr, *d_v = nullptr, *d_o = nullptr;

  //分配GPU显存globalmem
  CUDA_CHECK(cudaMalloc(&d_q, q_bytes));
  CUDA_CHECK(cudaMalloc(&d_k, kv_bytes));
  CUDA_CHECK(cudaMalloc(&d_v, kv_bytes));
  CUDA_CHECK(cudaMalloc(&d_o, q_bytes));

  //将主机的数据赋值给显存
  CUDA_CHECK(cudaMemcpy(d_q, h_q.data(), q_bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_k, h_k.data(), kv_bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_v, h_v.data(), kv_bytes, cudaMemcpyHostToDevice));

  // printf("Attention config: batch=%d, q_len=%d, kv_len=%d, q_heads=%d, kv_heads=%d, head_dim=%d, causal=%d\n",
  //      batch_size,
  //      target_seq_len,
  //      src_seq_len,
  //      query_heads,
  //      kv_heads,
  //      head_dim,
  //      is_causal);

  //核函数
  dim3 block(K); //一个线程处理一块的行
  dim3 grid(batch_size, query_heads);
  // 动态共享内存大小
  size_t shared_mem_size = (3 * K * head_dim) * sizeof(float);

  flashAttention_kernel<T><<<grid, block, shared_mem_size>>>(d_q, d_k, d_v, d_o, batch_size, target_seq_len, src_seq_len, query_heads, kv_heads, head_dim, is_causal);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  //将显存数据拷回CPU
  CUDA_CHECK(cudaMemcpy(h_o.data(), d_o, o_bytes, cudaMemcpyDeviceToHost));

  //释放显存内存
  CUDA_CHECK(cudaFree(d_q));
  CUDA_CHECK(cudaFree(d_k));
  CUDA_CHECK(cudaFree(d_v));
  CUDA_CHECK(cudaFree(d_o));
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
