#include <vector>
#include <iostream>
#include <algorithm>
#include <limits>
#include "../tester/utils.h"


// 定义一个CUDA核函数，用于比较和交换元素
template <typename T>
__global__ void compareAndSwap(T* data, int j, int k, size_t n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    int ixj = idx ^ j;
    // 确保ixj也在有效范围内且ixj > idx（只处理一次交换）
    if (ixj < n && ixj > idx) {
      // 根据双调排序网络的规则决定是升序还是降序比较
      bool ascending = ((idx & k) != 0);
      
      // 如果需要升序排序，且data[idx] > data[ixj]，则交换
      // 如果需要降序排序，且data[idx] < data[ixj]，则交换
      if ((ascending && data[idx] > data[ixj]) || 
          (!ascending && data[idx] < data[ixj])) {
        T temp = data[idx];
        data[idx] = data[ixj];
        data[ixj] = temp;
      }
    }
  }
}

/**
 * @brief Find the k-th largest element in a vector using CUDA.
 * 
 * @tparam T Type of elements in the input vector (should support `int` and `float`).
 * @param h_input Host-side input vector.
 * @param k 1-based index of the element to find (e.g., `k=1` returns the largest element).
 * @return T The k-th largest element in `h_input`.

 * @note Must use CUDA kernels for all compute-intensive steps; no significant CPU allowed.
 * @note Library functions that can directly complete a significant part of the work are NOT allowed. 
 * @note For invalid cases, return T(-100).
 * @note Handles device memory management (allocate/copy/free) internally. Errors should be thrown.
 */
template <typename T>
T kthLargest(const std::vector<T>& h_input, size_t k) {
  // 检查输入是否有效
  if (h_input.empty() || k == 0 || k > h_input.size()) {
    return T(-100); // 无效情况返回-100
  }

  size_t n = h_input.size();
  
  // 使数组大小为2的幂，便于双调排序
  size_t powerOfTwoSize = 1;
  while (powerOfTwoSize < n) {
    powerOfTwoSize <<= 1;
  }

  // 创建一个主机端的填充向量，用数据类型的最小值填充
  std::vector<T> h_padded_input(powerOfTwoSize, std::numeric_limits<T>::lowest());
  std::copy(h_input.begin(), h_input.end(), h_padded_input.begin());
  
  // 分配设备内存（使用填充后的大小）
  T* d_input = nullptr;
  CUDA_CHECK(cudaMalloc(&d_input, powerOfTwoSize * sizeof(T)));
  
  // 将填充后的输入数据复制到设备
  CUDA_CHECK(cudaMemcpy(d_input, h_padded_input.data(), powerOfTwoSize * sizeof(T), cudaMemcpyHostToDevice));
  
  // 定义每个线程块的线程数（基于填充后的大小）
  const int threadsPerBlock = 256;
  const int numBlocks = (powerOfTwoSize + threadsPerBlock - 1) / threadsPerBlock;
  
  // 双调排序 - 先构建双调序列，然后进行合并排序
  for (size_t k_step = 2; k_step <= powerOfTwoSize; k_step <<= 1) {
    for (size_t j = k_step >> 1; j > 0; j >>= 1) {
      // 向核函数传递填充后的大小
      compareAndSwap<T><<<numBlocks, threadsPerBlock>>>(d_input, j, k_step, powerOfTwoSize);
      CUDA_CHECK(cudaDeviceSynchronize());
    }
  }
  
  // 将排序后的第k大元素复制回主机（注意：排序后是降序，所以第k大元素在索引k-1处）
  T result;
  CUDA_CHECK(cudaMemcpy(&result, d_input + k - 1, sizeof(T), cudaMemcpyDeviceToHost));
  
  // 释放设备内存
  CUDA_CHECK(cudaFree(d_input));
  
  return result;
}


// CUDA核函数，用于计算注意力分数和输出
template <typename T>
__global__ void flashAttentionKernel(
    const T* q, const T* k, const T* v, T* o,
    int batch_size, int target_seq_len, int src_seq_len,
    int query_heads, int kv_heads, int head_dim,
    bool is_causal) {
    
    // 计算当前线程处理的位置
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 计算总元素数
    int total_elements = batch_size * target_seq_len * query_heads * head_dim;
    
    if (idx < total_elements) {
        // 计算当前元素的坐标
        int h_idx = idx % head_dim;
        int qh_idx = (idx / head_dim) % query_heads;
        int t_idx = (idx / (head_dim * query_heads)) % target_seq_len;
        int b_idx = idx / (head_dim * query_heads * target_seq_len);
        
        // 计算对应的kv头索引（支持GQA）
        int kv_head_idx = qh_idx * kv_heads / query_heads;
        
        // 初始化输出
        T sum = 0.0f;
        T scale_factor = 1.0f / sqrt(static_cast<T>(head_dim));
        T max_val = -1e30f;
        
        // 计算有效序列长度（考虑因果掩码）
        int valid_seq_len = is_causal ? min(t_idx + 1, src_seq_len) : src_seq_len;
        
        // 首先找到最大值，用于数值稳定性
        for (int s_idx = 0; s_idx < valid_seq_len; s_idx++) {
            T score = 0.0f;
            
            // 计算q和k的点积
            for (int d = 0; d < head_dim; d++) {
                int q_offset = ((b_idx * target_seq_len + t_idx) * query_heads + qh_idx) * head_dim + d;
                int k_offset = ((b_idx * src_seq_len + s_idx) * kv_heads + kv_head_idx) * head_dim + d;
                score += q[q_offset] * k[k_offset];
            }
            
            // 应用缩放因子
            score *= scale_factor;
            
            // 更新最大值
            if (score > max_val) {
                max_val = score;
            }
        }
        
        // 计算softmax和加权和
        T softmax_sum = 0.0f;
        
        for (int s_idx = 0; s_idx < valid_seq_len; s_idx++) {
            T score = 0.0f;
            
            // 重新计算q和k的点积
            for (int d = 0; d < head_dim; d++) {
                int q_offset = ((b_idx * target_seq_len + t_idx) * query_heads + qh_idx) * head_dim + d;
                int k_offset = ((b_idx * src_seq_len + s_idx) * kv_heads + kv_head_idx) * head_dim + d;
                score += q[q_offset] * k[k_offset];
            }
            
            // 应用缩放因子
            score *= scale_factor;
            
            // 应用softmax（减去最大值以提高数值稳定性）
            T exp_score = exp(score - max_val);
            softmax_sum += exp_score;
            
            // 累加加权值
            int v_offset = ((b_idx * src_seq_len + s_idx) * kv_heads + kv_head_idx) * head_dim + h_idx;
            sum += exp_score * v[v_offset];
        }
        
        // 归一化并写入输出
        if (softmax_sum > 0.0f) {
            sum /= softmax_sum;
        }
        o[idx] = sum;
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
    // 检查输入是否有效
    if (batch_size <= 0 || target_seq_len <= 0 || src_seq_len <= 0 ||
        query_heads <= 0 || kv_heads <= 0 || head_dim <= 0) {
        return;
    }
    
    // 检查GQA约束：query_heads必须是kv_heads的整数倍
    if (query_heads % kv_heads != 0) {
        return;
    }
    
    // 计算输入和输出的大小
    size_t q_size = batch_size * target_seq_len * query_heads * head_dim;
    size_t k_size = batch_size * src_seq_len * kv_heads * head_dim;
    size_t v_size = batch_size * src_seq_len * kv_heads * head_dim;
    size_t o_size = batch_size * target_seq_len * query_heads * head_dim;
    
    // 检查输入向量的大小是否正确
    if (h_q.size() != q_size || h_k.size() != k_size || h_v.size() != v_size) {
        return;
    }
    
    // 调整输出向量的大小
    h_o.resize(o_size);
    
    // 分配设备内存
    T *d_q = nullptr, *d_k = nullptr, *d_v = nullptr, *d_o = nullptr;
    CUDA_CHECK(cudaMalloc(&d_q, q_size * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_k, k_size * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_v, v_size * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_o, o_size * sizeof(T)));
    
    // 将输入数据复制到设备
    CUDA_CHECK(cudaMemcpy(d_q, h_q.data(), q_size * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_k, h_k.data(), k_size * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_v, h_v.data(), v_size * sizeof(T), cudaMemcpyHostToDevice));
    
    // 设置CUDA核函数的参数
    int threadsPerBlock = 256;
    int numBlocks = (o_size + threadsPerBlock - 1) / threadsPerBlock;
    
    // 启动CUDA核函数
    flashAttentionKernel<T><<<numBlocks, threadsPerBlock>>>(
        d_q, d_k, d_v, d_o,
        batch_size, target_seq_len, src_seq_len,
        query_heads, kv_heads, head_dim,
        is_causal
    );
    
    // 同步设备
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // 将结果复制回主机
    CUDA_CHECK(cudaMemcpy(h_o.data(), d_o, o_size * sizeof(T), cudaMemcpyDeviceToHost));
    
    // 释放设备内存
    CUDA_CHECK(cudaFree(d_q));
    CUDA_CHECK(cudaFree(d_k));
    CUDA_CHECK(cudaFree(d_v));
    CUDA_CHECK(cudaFree(d_o));
}

// *********************************************************************
// Explicit Template Instantiations (REQUIRED FOR LINKING WITH TESTER.O)
// DO NOT MODIFY THIS SECTION
// *********************************************************************
template int kthLargest<int>(const std::vector<int>&, size_t);
template float kthLargest<float>(const std::vector<float>&, size_t);
template void flashAttention<float>(const std::vector<float>&, const std::vector<float>&,
  const std::vector<float>&, std::vector<float>&,
  int, int, int, int, int, int, bool);
