#include <vector>
#include <algorithm>
#include <cmath>
#include <type_traits>
#include <limits>
#include <musa_fp16.h>

#include "../tester/utils.h"

template <typename T>
struct FloatAdapter {
  static inline float toFloat(T v) { return static_cast<float>(v); }
  static inline T fromFloat(float v) { return static_cast<T>(v); }
};

template <>
struct FloatAdapter<half> {
  static inline float toFloat(half v) { return __half2float(v); }
  static inline half fromFloat(float v) { return __float2half(v); }
};

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
  const size_t diag = std::min(rows, cols);
  using AccT = typename std::conditional<std::is_floating_point<T>::value, double, long long>::type;
  AccT acc = 0;
  for (size_t i = 0; i < diag; ++i) {
    const size_t idx = i * cols + i;
    acc += static_cast<AccT>(h_input[idx]);
  }
  return static_cast<T>(acc);
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
  // CPU reference implementation matching torch.nn.functional.scaled_dot_product_attention
  const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
  const float neg_inf = -1e20f;

  auto q_idx = [=](int b, int t, int qh, int d) {
    return ((b * target_seq_len + t) * query_heads + qh) * head_dim + d;
  };
  auto kv_idx = [=](int b, int s, int kvh, int d) {
    return ((b * src_seq_len + s) * kv_heads + kvh) * head_dim + d;
  };
  auto kv_for_q = [=](int qh) { return (kv_heads == 1) ? 0 : (qh * kv_heads) / query_heads; };

  std::vector<float> acc(head_dim);

  for (int b = 0; b < batch_size; ++b) {
    for (int t = 0; t < target_seq_len; ++t) {
      for (int qh = 0; qh < query_heads; ++qh) {
        const int kvh = kv_for_q(qh);

        float max_score = -std::numeric_limits<float>::infinity();
        for (int s = 0; s < src_seq_len; ++s) {
          if (is_causal && s > t) continue;
          float dot = 0.0f;
          for (int d = 0; d < head_dim; ++d) {
            float q = FloatAdapter<T>::toFloat(h_q[q_idx(b, t, qh, d)]);
            float k = FloatAdapter<T>::toFloat(h_k[kv_idx(b, s, kvh, d)]);
            dot += q * k;
          }
          float score = dot * scale;
          if (score > max_score) max_score = score;
        }
        if (max_score == -std::numeric_limits<float>::infinity()) max_score = neg_inf;

        std::fill(acc.begin(), acc.end(), 0.0f);
        float denom = 0.0f;
        for (int s = 0; s < src_seq_len; ++s) {
          if (is_causal && s > t) continue;
          float dot = 0.0f;
          for (int d = 0; d < head_dim; ++d) {
            float q = FloatAdapter<T>::toFloat(h_q[q_idx(b, t, qh, d)]);
            float k = FloatAdapter<T>::toFloat(h_k[kv_idx(b, s, kvh, d)]);
            dot += q * k;
          }
          float score = dot * scale;
          float w = std::exp(score - max_score);
          denom += w;
          for (int d = 0; d < head_dim; ++d) {
            float v = FloatAdapter<T>::toFloat(h_v[kv_idx(b, s, kvh, d)]);
            acc[d] += w * v;
          }
        }

        float inv_denom = (denom == 0.0f) ? 0.0f : 1.0f / denom;
        for (int d = 0; d < head_dim; ++d) {
          float out_val = acc[d] * inv_denom;
          h_o[q_idx(b, t, qh, d)] = FloatAdapter<T>::fromFloat(out_val);
        }
      }
    }
  }
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
