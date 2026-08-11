#include <vector>
#include <musa_fp16.h>
#include <cfloat>
#include <cmath>

#include "../tester/utils.h"

// ============================================================================
// Platform-agnostic negative infinity for masking
// ============================================================================
// MUSA's <cmath> / <cfloat> provides INFINITY.  Cast to float explicitly
// so that the kernel uses f32 negative infinity throughout.
#define NEG_INF (-static_cast<float>(INFINITY))

// ============================================================================
// Helper: type-safe conversion between float and half
// ============================================================================
// All arithmetic inside kernels is performed in f32 to avoid precision loss.
// These helpers centralise the platform-specific intrinsics so the kernel
// bodies are clean and identical for both float and half instantiations.
// For T = float they are simple static_casts (no-op at runtime).
//
// NOTE: If the MUSA toolchain uses different half-conversion intrinsics
// (e.g. musaHalf2Float / musaFloat2Half), replace the specialisations below.

template <typename T>
__device__ __forceinline__ float to_float(T val) {
  return static_cast<float>(val);
}

template <>
__device__ __forceinline__ float to_float<half>(half val) {
  return __half2float(val);            // MUSA intrinsic: half → float
}

template <typename T>
__device__ __forceinline__ T from_float(float val) {
  return static_cast<T>(val);
}

template <>
__device__ __forceinline__ half from_float<half>(float val) {
  return __float2half(val);            // MUSA intrinsic: float → half
}

// ============================================================================
// Task 1:  rmsNorm  —  Root Mean Square Layer Normalisation
// ============================================================================

/**
 * @brief GPU kernel for RMS Normalisation.
 *
 * **Algorithm (per row):**
 *   1. Compute  mean_sq = (1 / hidden_dim) * Σⱼ xⱼ²
 *   2. Compute  outputⱼ = xⱼ * (1 / √(mean_sq + ε)) * weightⱼ
 *
 * **Parallelisation strategy:**
 *   - Grid:  one block per row  (gridDim.x = rows)
 *   - Block: each thread strides across hidden_dim to compute a partial
 *            sum of squares, then all threads cooperate in a tree reduction
 *            via shared memory to obtain the full row sum.
 *   - After reduction, every thread applies the normalisation to its
 *     assigned elements using the scalar weight vector.
 *
 * **Why accumulate in float?**
 *   For `half` inputs, squaring a half value can underflow/overflow its
 *   limited range (~5.96e-8 to 65504).  Accumulating partial sums in f32
 *   preserves precision.  The final store converts back to half.
 *
 * @param input     Row-major matrix  [rows × hidden_dim]
 * @param weight    Per-column scale  [hidden_dim]
 * @param output    Row-major matrix  [rows × hidden_dim]  (pre-allocated)
 * @param rows      Number of rows (tokens)
 * @param hidden_dim Feature dimension
 * @param eps       Numerical stability term added before rsqrt
 */
template <typename T>
__global__ void rmsNormKernel(const T* __restrict__ input,
                               const T* __restrict__ weight,
                               T* __restrict__ output,
                               size_t rows, size_t hidden_dim, float eps) {
  int row = blockIdx.x;
  if (row >= rows) return;                  // guard against excess blocks

  // Dynamic shared memory: one f32 slot per thread for the reduction tree
  extern __shared__ float sdata[];

  // ---- Phase 1: per-thread partial sum of squares (strided loop) ----
  float sum_sq = 0.0f;
  for (int j = threadIdx.x; j < hidden_dim; j += blockDim.x) {
    float val = to_float(input[row * hidden_dim + j]);
    sum_sq += val * val;
  }

  // ---- Phase 2: parallel reduction (binary tree, in shared memory) ----
  // Writes each thread's partial sum into shared memory, then iteratively
  // halves the active set until only thread 0 holds the total.
  sdata[threadIdx.x] = sum_sq;
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      sdata[threadIdx.x] += sdata[threadIdx.x + s];
    }
    __syncthreads();                        // synchronise between tree levels
  }

  // ---- Phase 3: apply RMSNorm using the reduced total ----
  // Only thread 0's slot holds the full sum; all threads read it.
  float inv_rms = rsqrtf(sdata[0] / static_cast<float>(hidden_dim) + eps);

  for (int j = threadIdx.x; j < hidden_dim; j += blockDim.x) {
    float val = to_float(input[row * hidden_dim + j]);
    float w   = to_float(weight[j]);
    output[row * hidden_dim + j] = from_float<T>(val * inv_rms * w);
  }
}

/**
 * @brief Host-side driver for rmsNorm.
 *
 * Allocates device memory via the MUSA runtime API, copies inputs, launches
 * the kernel, copies the result back, and cleans up.  Error checking is
 * performed via the RUNTIME_CHECK macro (defined in utils.h).
 */
template <typename T>
void rmsNorm(const std::vector<T>& h_input, const std::vector<T>& h_weight,
              std::vector<T>& h_output, size_t rows, size_t hidden_dim,
              float eps) {
  size_t input_size  = rows * hidden_dim;
  size_t weight_size = hidden_dim;
  size_t output_size = rows * hidden_dim;

  // Device allocation  (MUSA runtime API: musa-prefixed)
  T *d_input, *d_weight, *d_output;
  RUNTIME_CHECK(musaMalloc(&d_input,  input_size  * sizeof(T)));
  RUNTIME_CHECK(musaMalloc(&d_weight, weight_size * sizeof(T)));
  RUNTIME_CHECK(musaMalloc(&d_output, output_size * sizeof(T)));

  // Host → device
  RUNTIME_CHECK(musaMemcpy(d_input,  h_input.data(),  input_size  * sizeof(T),
                           musaMemcpyHostToDevice));
  RUNTIME_CHECK(musaMemcpy(d_weight, h_weight.data(), weight_size * sizeof(T),
                           musaMemcpyHostToDevice));

  // Launch: one block per row, 256 threads per block
  int block_size    = 256;
  int grid_size     = static_cast<int>(rows);
  size_t shared_mem = block_size * sizeof(float);

  rmsNormKernel<<<grid_size, block_size, shared_mem>>>(
      d_input, d_weight, d_output, rows, hidden_dim, eps);
  RUNTIME_CHECK(musaGetLastError());
  RUNTIME_CHECK(musaDeviceSynchronize());

  // Device → host
  RUNTIME_CHECK(musaMemcpy(h_output.data(), d_output, output_size * sizeof(T),
                           musaMemcpyDeviceToHost));

  // Cleanup
  RUNTIME_CHECK(musaFree(d_input));
  RUNTIME_CHECK(musaFree(d_weight));
  RUNTIME_CHECK(musaFree(d_output));
}

// ============================================================================
// Task 2:  flashAttention  —  Tiled Online-Softmax Attention
// ============================================================================

// Tile sizes (tuned for typical head_dim ∈ [32, 128] and SM shared memory)
#define FA_TILE_Q  32   // query rows processed per block  (= blockDim.x)
#define FA_TILE_KV 32   // key / value rows processed per inner-loop iteration

/**
 * @brief Flash-Attention kernel using tiled online softmax.
 *
 * The naive attention  softmax(Q·Kᵀ / √d)·V  materialises the full
 * [target_seq_len × src_seq_len] score matrix, which is O(n²) memory.
 *
 * **Key insight (online softmax):**
 *   Instead of storing all scores, we iterate over K/V tiles and maintain
 *   three running statistics per query row:
 *     m  — running maximum of scores seen so far
 *     l  — running sum of exp(score - m)
 *     O_acc — running output accumulator (weighted sum of V rows)
 *   When a new tile arrives we rescale the old accumulator by
 *   exp(m_old - m_new) so that the contribution of past tiles is correctly
 *   weighted relative to the new tile.  This avoids materialising the
 *   attention matrix entirely — only a single K/V tile and the O
 *   accumulator reside in shared memory.
 *
 * **Grid & block layout:**
 *   grid.x  = ⌈target_seq_len / TILE_Q⌉        Q-sequence tiles
 *   grid.y  = batch_size × query_heads           (batch, head) pairs
 *   block.x = TILE_Q threads                     one thread per Q row
 *
 * **GQA (Grouped Query Attention):**
 *   When kv_heads < query_heads, multiple query heads share one KV head.
 *   Mapping:  kv_head = q_head × kv_heads ÷ query_heads  (integer div).
 *
 * **Causal mask:**
 *   When `is_causal` is true, query position i may only attend to key
 *   positions j where j ≤ i.  Masked positions receive score = -∞ so
 *   their softmax weight becomes 0.
 *
 * **Shared memory layout:**
 *   kv_buf  [TILE_KV × head_dim] f32  — loaded with K, then overwritten with V
 *   o_acc   [TILE_Q  × head_dim] f32  — running output accumulator
 *
 * **Tensor shapes (row-major, flattened):**
 *   Q: [batch_size, target_seq_len, query_heads, head_dim]
 *   K: [batch_size, src_seq_len,    kv_heads,    head_dim]
 *   V: [batch_size, src_seq_len,    kv_heads,    head_dim]
 *   O: [batch_size, target_seq_len, query_heads, head_dim]
 */
template <typename T>
__global__ void flashAttentionKernel(
    const T* __restrict__ Q,
    const T* __restrict__ K,
    const T* __restrict__ V,
    T* __restrict__ O,
    int batch_size, int target_seq_len, int src_seq_len,
    int query_heads, int kv_heads, int head_dim, bool is_causal) {

  // ---- Block indexing ----
  int q_tile_idx   = blockIdx.x;
  int q_tile_start = q_tile_idx * FA_TILE_Q;
  int batch_idx    = blockIdx.y / query_heads;
  int q_head_idx   = blockIdx.y % query_heads;

  // GQA: map query head → key/value head (grouped-query attention)
  int kv_head_idx  = q_head_idx * kv_heads / query_heads;

  int tid     = threadIdx.x;
  int q_idx   = q_tile_start + tid;               // global query position
  bool valid  = (q_idx < target_seq_len);         // guard for last partial tile

  // Dynamic shared memory partition
  extern __shared__ float smem[];
  float* kv_buf = smem;                           // [TILE_KV × head_dim]
  float* o_acc  = smem + FA_TILE_KV * head_dim;   // [TILE_Q  × head_dim]

  // 1/√d — scaling factor applied to dot products before softmax
  float scale = rsqrtf(static_cast<float>(head_dim));

  // Base pointers into the flat tensors for this (batch, head) pair
  // Strides are computed from the shape to handle the multi-dimensional indexing
  const T* Q_base = Q + batch_idx * target_seq_len * query_heads * head_dim;
  const T* K_base = K + batch_idx * src_seq_len    * kv_heads    * head_dim;
  const T* V_base = V + batch_idx * src_seq_len    * kv_heads    * head_dim;
  T*       O_base = O + batch_idx * target_seq_len * query_heads * head_dim;

  // ---- Initialise output accumulator to zero ----
  for (int d = tid; d < head_dim; d += FA_TILE_Q) {
    o_acc[tid * head_dim + d] = 0.0f;
  }

  // ---- Online softmax state (one set per query row) ----
  float m_prev = NEG_INF;     // running maximum score
  float l_prev = 0.0f;        // running denominator  Σ exp(score - m)

  // =========================================================================
  // Main loop: iterate over K/V tiles along the source sequence
  // =========================================================================
  for (int kv_start = 0; kv_start < src_seq_len; kv_start += FA_TILE_KV) {
    int kv_len = min(FA_TILE_KV, src_seq_len - kv_start);

    // ------------------------------------------------------------------
    // Step 1 — Load K tile into shared memory  (cooperative load)
    // ------------------------------------------------------------------
    for (int i = tid; i < kv_len * head_dim; i += FA_TILE_Q) {
      int kv_idx = kv_start + i / head_dim;
      int d      = i % head_dim;
      size_t off = kv_idx * kv_heads * head_dim + kv_head_idx * head_dim + d;
      kv_buf[i]  = to_float(K_base[off]);
    }
    __syncthreads();  // all threads: K tile ready to read

    // ------------------------------------------------------------------
    // Step 2 — Compute  S = Q[row] · Kᵀ / √d   and update online stats
    // ------------------------------------------------------------------
    // Each thread handles exactly one query row within the Q tile.
    // s_vals[k] stores the (scaled, optionally masked) dot product
    // between this thread's Q row and the k-th K row in the tile.

    float m_new = m_prev;
    float l_new = 0.0f;
    float s_vals[FA_TILE_KV];   // per-KV-row scores  (kept in registers)

    if (valid) {
      float s_max = NEG_INF;

      // Compute Q·Kᵀ for every KV row in this tile
      for (int k = 0; k < kv_len; k++) {
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++) {
          size_t q_off = q_idx * query_heads * head_dim
                       + q_head_idx * head_dim + d;
          dot += to_float(Q_base[q_off]) * kv_buf[k * head_dim + d];
        }
        dot *= scale;

        // Causal masking: query at position i cannot attend to key j > i
        if (is_causal && q_idx < kv_start + k) {
          dot = NEG_INF;
        }

        s_vals[k] = dot;
        if (dot > s_max) s_max = dot;
      }

      // Online-softmax update:
      //   m_new   = max(m_prev, max(S_tile))          — running max
      //   l_new   = exp(m_prev - m_new)·l_prev + Σ exp(S_k - m_new)
      // The exp(m_prev - m_new) factor correctly re-weights the
      // contributions from all previous tiles under the new maximum.
      m_new = fmaxf(m_prev, s_max);
      float correction = expf(m_prev - m_new);
      l_new = correction * l_prev;
      for (int k = 0; k < kv_len; k++) {
        l_new += expf(s_vals[k] - m_new);
      }
    }

    // Barrier: must finish reading K before any thread overwrites it with V
    __syncthreads();

    // ------------------------------------------------------------------
    // Step 3 — Load V tile into shared memory  (reuse kv_buf)
    // ------------------------------------------------------------------
    for (int i = tid; i < kv_len * head_dim; i += FA_TILE_Q) {
      int kv_idx = kv_start + i / head_dim;
      int d      = i % head_dim;
      size_t off = kv_idx * kv_heads * head_dim + kv_head_idx * head_dim + d;
      kv_buf[i]  = to_float(V_base[off]);
    }
    __syncthreads();  // all threads: V tile ready to read

    // ------------------------------------------------------------------
    // Step 4 — Accumulate  O += softmax(S) · V   (online update)
    // ------------------------------------------------------------------
    //   P_k = exp(S_k - m_new)                     ← softmax numerator
    //   O_acc = correction · O_acc_old  +  P · V   ← rescale + new tile
    if (valid) {
      float correction = expf(m_prev - m_new);

      // Rescale old accumulator so its contribution is relative to m_new
      for (int d = 0; d < head_dim; d++) {
        o_acc[tid * head_dim + d] *= correction;
      }

      // Accumulate weighted sum of V rows for this tile
      for (int k = 0; k < kv_len; k++) {
        float p_val = expf(s_vals[k] - m_new);     // softmax weight
        for (int d = 0; d < head_dim; d++) {
          o_acc[tid * head_dim + d] += p_val * kv_buf[k * head_dim + d];
        }
      }
    }

    // Advance running statistics for the next tile iteration
    m_prev = m_new;
    l_prev = l_new;
  }  // end KV-tile loop

  // =========================================================================
  // Finalise:  O = O_acc / l    (divide by softmax denominator)
  // =========================================================================
  if (valid) {
    float inv_l = (l_prev > 0.0f) ? (1.0f / l_prev) : 0.0f;
    for (int d = 0; d < head_dim; d++) {
      float val = o_acc[tid * head_dim + d] * inv_l;
      size_t off = q_idx * query_heads * head_dim + q_head_idx * head_dim + d;
      O_base[off] = from_float<T>(val);
    }
  }
}

/**
 * @brief Host-side driver for flashAttention.
 *
 * Allocates device memory for Q, K, V, O tensors via the MUSA runtime API,
 * copies inputs, launches the tiled attention kernel, copies the result
 * back, and cleans up.
 */
template <typename T>
void flashAttention(const std::vector<T>& h_q, const std::vector<T>& h_k,
                    const std::vector<T>& h_v, std::vector<T>& h_o,
                    int batch_size, int target_seq_len, int src_seq_len,
                    int query_heads, int kv_heads, int head_dim, bool is_causal) {
  size_t q_size  = static_cast<size_t>(batch_size) * target_seq_len
                 * query_heads * head_dim;
  size_t kv_size = static_cast<size_t>(batch_size) * src_seq_len
                 * kv_heads * head_dim;
  size_t o_size  = q_size;   // output has same shape as query

  // Device allocation  (MUSA runtime API: musa-prefixed)
  T *d_q, *d_k, *d_v, *d_o;
  RUNTIME_CHECK(musaMalloc(&d_q, q_size  * sizeof(T)));
  RUNTIME_CHECK(musaMalloc(&d_k, kv_size * sizeof(T)));
  RUNTIME_CHECK(musaMalloc(&d_v, kv_size * sizeof(T)));
  RUNTIME_CHECK(musaMalloc(&d_o, o_size  * sizeof(T)));

  // Host → device
  RUNTIME_CHECK(musaMemcpy(d_q, h_q.data(), q_size  * sizeof(T),
                           musaMemcpyHostToDevice));
  RUNTIME_CHECK(musaMemcpy(d_k, h_k.data(), kv_size * sizeof(T),
                           musaMemcpyHostToDevice));
  RUNTIME_CHECK(musaMemcpy(d_v, h_v.data(), kv_size * sizeof(T),
                           musaMemcpyHostToDevice));

  // Launch configuration
  int q_tiles  = (target_seq_len + FA_TILE_Q - 1) / FA_TILE_Q;
  dim3 grid(q_tiles, batch_size * query_heads);
  int block_dim = FA_TILE_Q;

  // Shared memory = KV tile buffer + O accumulator  (both in f32)
  size_t shared_bytes = static_cast<size_t>(FA_TILE_KV + FA_TILE_Q)
                      * head_dim * sizeof(float);

  flashAttentionKernel<<<grid, block_dim, shared_bytes>>>(
      d_q, d_k, d_v, d_o,
      batch_size, target_seq_len, src_seq_len,
      query_heads, kv_heads, head_dim, is_causal);
  RUNTIME_CHECK(musaGetLastError());
  RUNTIME_CHECK(musaDeviceSynchronize());

  // Device → host
  RUNTIME_CHECK(musaMemcpy(h_o.data(), d_o, o_size * sizeof(T),
                           musaMemcpyDeviceToHost));

  // Cleanup
  RUNTIME_CHECK(musaFree(d_q));
  RUNTIME_CHECK(musaFree(d_k));
  RUNTIME_CHECK(musaFree(d_v));
  RUNTIME_CHECK(musaFree(d_o));
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
