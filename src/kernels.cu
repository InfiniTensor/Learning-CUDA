#include <vector>
#include <algorithm>
#include <cmath>
#include <limits>
#include <cuda_fp16.h>

#include "../tester/utils.h"

namespace {
template <typename T>
inline float to_float(T value) {
  return static_cast<float>(value);
}

template <>
inline float to_float<half>(half value) {
  return __half2float(value);
}

template <typename T>
inline T from_float(float value) {
  return static_cast<T>(value);
}

template <>
inline half from_float<half>(float value) {
  return __float2half_rn(value);
}
}  // namespace

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
  if (rows == 0 || cols == 0 || h_input.empty()) {
    return T(0);
  }

  const size_t diag = std::min(rows, cols);
  T sum = T(0);
  for (size_t i = 0; i < diag; ++i) {
    const size_t idx = i * cols + i;
    if (idx >= h_input.size()) {
      break;
    }
    sum += h_input[idx];
  }
  return sum;
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
  if (batch_size <= 0 || target_seq_len <= 0 || src_seq_len <= 0 ||
      query_heads <= 0 || kv_heads <= 0 || head_dim <= 0) {
    return;
  }

  const int group_size = std::max(1, query_heads / kv_heads);
  const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

  const size_t q_stride_bt = static_cast<size_t>(query_heads) * head_dim;
  const size_t k_stride_bs = static_cast<size_t>(kv_heads) * head_dim;

  for (int b = 0; b < batch_size; ++b) {
    for (int t = 0; t < target_seq_len; ++t) {
      for (int qh = 0; qh < query_heads; ++qh) {
        const int kvh = std::min(kv_heads - 1, qh / group_size);

        std::vector<float> scores(static_cast<size_t>(src_seq_len), -std::numeric_limits<float>::infinity());
        float max_score = -std::numeric_limits<float>::infinity();

        for (int s = 0; s < src_seq_len; ++s) {
          if (is_causal && s > t) {
            continue;
          }

          float dot = 0.0f;
          const size_t q_base = (static_cast<size_t>(b) * target_seq_len + t) * q_stride_bt +
                                static_cast<size_t>(qh) * head_dim;
          const size_t k_base = (static_cast<size_t>(b) * src_seq_len + s) * k_stride_bs +
                                static_cast<size_t>(kvh) * head_dim;
          for (int d = 0; d < head_dim; ++d) {
            dot += to_float(h_q[q_base + d]) * to_float(h_k[k_base + d]);
          }

          const float score = dot * scale;
          scores[static_cast<size_t>(s)] = score;
          if (score > max_score) {
            max_score = score;
          }
        }

        float denom = 0.0f;
        for (int s = 0; s < src_seq_len; ++s) {
          const float score = scores[static_cast<size_t>(s)];
          if (score == -std::numeric_limits<float>::infinity()) {
            continue;
          }
          const float exp_val = std::exp(score - max_score);
          scores[static_cast<size_t>(s)] = exp_val;
          denom += exp_val;
        }

        if (denom == 0.0f) {
          denom = 1.0f;
        }

        const size_t o_base = (static_cast<size_t>(b) * target_seq_len + t) * q_stride_bt +
                              static_cast<size_t>(qh) * head_dim;
        for (int d = 0; d < head_dim; ++d) {
          float acc = 0.0f;
          for (int s = 0; s < src_seq_len; ++s) {
            const float weight = scores[static_cast<size_t>(s)] / denom;
            if (weight == 0.0f) {
              continue;
            }
            const size_t v_base = (static_cast<size_t>(b) * src_seq_len + s) * k_stride_bs +
                                  static_cast<size_t>(kvh) * head_dim;
            acc += weight * to_float(h_v[v_base + d]);
          }
          h_o[o_base + d] = from_float<T>(acc);
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
