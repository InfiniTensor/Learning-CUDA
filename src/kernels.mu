/**
 * @file kernels.mu
 * @brief CUDA kernel implementations for Moore Threads (摩尔线程) GPU platform
 * @author Training Camp Student
 * @date 2026-02
 * 
 * This file contains implementations adapted for Moore Threads GPU:
 * 1. Matrix trace computation with parallel reduction
 * 2. Flash Attention with causal masking and GQA support
 */

#include <vector>
#include <musa_fp16.h>
#include <cmath>
#include <algorithm>

#include "../tester/utils.h"

// ============================================================================
// CONSTANTS
// ============================================================================

constexpr int WARP_SIZE = 32;           // Moore Threads uses 32-thread warps
constexpr int TRACE_BLOCK_SIZE = 256;

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

/**
 * @brief Warp-level reduction using shuffle instructions
 */
template <typename T>
__device__ __forceinline__ T warpReduceSum(T val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

/**
 * @brief Block-level reduction
 */
template <typename T>
__device__ __forceinline__ T blockReduceSum(T val, T* shared) {
    const int lane = threadIdx.x % WARP_SIZE;
    const int wid = threadIdx.x / WARP_SIZE;
    
    val = warpReduceSum(val);
    
    if (lane == 0) shared[wid] = val;
    __syncthreads();
    
    const int numWarps = blockDim.x / WARP_SIZE;
    val = (threadIdx.x < numWarps) ? shared[threadIdx.x] : T(0);
    
    if (wid == 0) val = warpReduceSum(val);
    
    return val;
}

// ============================================================================
// TRACE KERNEL
// ============================================================================

template <typename T>
__global__ void traceKernel(const T* __restrict__ input, 
                            T* __restrict__ output, 
                            size_t rows, 
                            size_t cols) {
    __shared__ T sharedMem[TRACE_BLOCK_SIZE / WARP_SIZE];
    
    const size_t diagLen = min(rows, cols);
    const size_t stride = gridDim.x * blockDim.x;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    T localSum = T(0);
    while (idx < diagLen) {
        localSum += input[idx * cols + idx];
        idx += stride;
    }
    
    localSum = blockReduceSum(localSum, sharedMem);
    
    if (threadIdx.x == 0) {
        atomicAdd(output, localSum);
    }
}

/**
 * @brief Computes the trace of a matrix using GPU
 */
template <typename T>
T trace(const std::vector<T>& h_input, size_t rows, size_t cols) {
    const size_t diagLen = std::min(rows, cols);
    if (diagLen == 0) return T(0);
    
    const size_t inputBytes = rows * cols * sizeof(T);
    const size_t outputBytes = sizeof(T);
    
    T* d_input = nullptr;
    T* d_output = nullptr;
    musaMalloc(&d_input, inputBytes);
    musaMalloc(&d_output, outputBytes);
    
    musaMemcpy(d_input, h_input.data(), inputBytes, musaMemcpyHostToDevice);
    musaMemset(d_output, 0, outputBytes);
    
    const int blockSize = TRACE_BLOCK_SIZE;
    const int numBlocks = std::min((size_t)((diagLen + blockSize - 1) / blockSize), (size_t)128);
    
    traceKernel<T><<<numBlocks, blockSize>>>(d_input, d_output, rows, cols);
    
    T result;
    musaMemcpy(&result, d_output, outputBytes, musaMemcpyDeviceToHost);
    
    musaFree(d_input);
    musaFree(d_output);
    
    return result;
}

// ============================================================================
// FLASH ATTENTION IMPLEMENTATION
// ============================================================================

/**
 * @brief Type conversion utilities for mixed-precision computation
 */
template <typename T>
struct TypeConverter {
    __device__ __forceinline__ static float toFloat(T val);
    __device__ __forceinline__ static T fromFloat(float val);
};

template <>
struct TypeConverter<float> {
    __device__ __forceinline__ static float toFloat(float val) { return val; }
    __device__ __forceinline__ static float fromFloat(float val) { return val; }
};

template <>
struct TypeConverter<half> {
    __device__ __forceinline__ static float toFloat(half val) { return __half2float(val); }
    __device__ __forceinline__ static half fromFloat(float val) { return __float2half(val); }
};

/**
 * @brief Flash Attention kernel with online softmax
 */
template <typename T>
__global__ void flashAttentionKernel(
    const T* __restrict__ Q,
    const T* __restrict__ K,
    const T* __restrict__ V,
    T* __restrict__ O,
    const int batchSize,
    const int tgtSeqLen,
    const int srcSeqLen,
    const int queryHeads,
    const int kvHeads,
    const int headDim,
    const bool isCausal,
    const float scale) {
    
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batchSize * tgtSeqLen * queryHeads * headDim;
    if (idx >= total) return;
    
    const int d = idx % headDim;
    const int h = (idx / headDim) % queryHeads;
    const int t = (idx / (headDim * queryHeads)) % tgtSeqLen;
    const int b = idx / (headDim * queryHeads * tgtSeqLen);
    
    const int kvH = h / (queryHeads / kvHeads);
    const int maxSrc = isCausal ? min(t + 1, srcSeqLen) : srcSeqLen;
    
    // Online softmax
    float maxVal = -INFINITY;
    float sumExp = 0.0f;
    float result = 0.0f;
    
    for (int s = 0; s < maxSrc; s++) {
        float dot = 0.0f;
        for (int dd = 0; dd < headDim; dd++) {
            int qIdx = ((b * tgtSeqLen + t) * queryHeads + h) * headDim + dd;
            int kIdx = ((b * srcSeqLen + s) * kvHeads + kvH) * headDim + dd;
            dot += TypeConverter<T>::toFloat(Q[qIdx]) * TypeConverter<T>::toFloat(K[kIdx]);
        }
        dot *= scale;
        
        float prevMax = maxVal;
        maxVal = fmaxf(maxVal, dot);
        float correction = (prevMax == -INFINITY) ? 0.0f : expf(prevMax - maxVal);
        float weight = expf(dot - maxVal);
        sumExp = sumExp * correction + weight;
        
        int vIdx = ((b * srcSeqLen + s) * kvHeads + kvH) * headDim + d;
        result = result * correction + weight * TypeConverter<T>::toFloat(V[vIdx]);
    }
    
    int oIdx = ((b * tgtSeqLen + t) * queryHeads + h) * headDim + d;
    O[oIdx] = TypeConverter<T>::fromFloat((sumExp > 0.0f) ? (result / sumExp) : 0.0f);
}

/**
 * @brief Computes Flash Attention for given query, key, and value tensors
 */
template <typename T>
void flashAttention(const std::vector<T>& h_q, const std::vector<T>& h_k,
                    const std::vector<T>& h_v, std::vector<T>& h_o,
                    int batch_size, int target_seq_len, int src_seq_len, 
                    int query_heads, int kv_heads, int head_dim, bool is_causal) {
    
    const size_t qSize = batch_size * target_seq_len * query_heads * head_dim;
    const size_t kvSize = batch_size * src_seq_len * kv_heads * head_dim;
    
    h_o.resize(qSize);
    
    T *d_q, *d_k, *d_v, *d_o;
    musaMalloc(&d_q, qSize * sizeof(T));
    musaMalloc(&d_k, kvSize * sizeof(T));
    musaMalloc(&d_v, kvSize * sizeof(T));
    musaMalloc(&d_o, qSize * sizeof(T));
    
    musaMemcpy(d_q, h_q.data(), qSize * sizeof(T), musaMemcpyHostToDevice);
    musaMemcpy(d_k, h_k.data(), kvSize * sizeof(T), musaMemcpyHostToDevice);
    musaMemcpy(d_v, h_v.data(), kvSize * sizeof(T), musaMemcpyHostToDevice);
    
    const float scale = 1.0f / sqrtf(static_cast<float>(head_dim));
    
    const int total = batch_size * target_seq_len * query_heads * head_dim;
    const int blockSize = 256;
    const int numBlocks = (total + blockSize - 1) / blockSize;
    
    flashAttentionKernel<T><<<numBlocks, blockSize>>>(
        d_q, d_k, d_v, d_o,
        batch_size, target_seq_len, src_seq_len,
        query_heads, kv_heads, head_dim,
        is_causal, scale);
    
    musaMemcpy(h_o.data(), d_o, qSize * sizeof(T), musaMemcpyDeviceToHost);
    
    musaFree(d_q);
    musaFree(d_k);
    musaFree(d_v);
    musaFree(d_o);
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
