#include <vector>
#include <stdexcept>
#include <limits>
#include <cmath>
#include <algorithm>
#include <iostream>

#include "../tester/utils.h"

#define smemCapa 8192
#define threadsNumPerBlock 1024
#define twiceTNPB 2048
#define tileHeight 4

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
__global__ void BitonicSort_Prolog(T* d_arr, int eleNumInSharedMem){
    //extern __shared__ T s_arr[];       
    __shared__ T s_arr[smemCapa];  
    int preBlksEleNum = blockIdx.x * eleNumInSharedMem;   //the num of array elements in previous blocks
    int idx = preBlksEleNum + threadIdx.x;  //idx:index in d_arr;   i: index in s_arr
    //load data from global mem to shared mem
    for(int i = threadIdx.x; i < eleNumInSharedMem; i += threadsNumPerBlock){
        s_arr[i] = d_arr[idx];
        idx += threadsNumPerBlock;
    }        

    for(int len = 2; len <= eleNumInSharedMem; len <<= 1){         
        for(int stride = len >> 1; stride > 0; stride >>= 1){
            __syncthreads();

            //compare s_arr[idx1] and s_arr[idx2]
            int idx1 = (stride > threadsNumPerBlock)? threadIdx.x : 2 * threadIdx.x - (threadIdx.x & (stride - 1));
            int idx2 = idx1 + stride;
            bool descend = (len & (preBlksEleNum + idx1)) != 0; 
            if((descend && s_arr[idx1] < s_arr[idx2]) || (!descend && s_arr[idx1] > s_arr[idx2])){ 
                T tem = s_arr[idx1];
                s_arr[idx1] = s_arr[idx2];
                s_arr[idx2] = tem;
            }

            int flag = stride / threadsNumPerBlock;
            for(int round = 1; round < eleNumInSharedMem / twiceTNPB; round++){
                if(stride <= threadsNumPerBlock){
                    idx1 += twiceTNPB;   //twiceTNPB: 2*threadNumPerBlock
                }
                else{
                    if(round % flag == 0)
                        idx1 += (stride + threadsNumPerBlock);
                    else
                        idx1 += threadsNumPerBlock;
                }
                idx2 = idx1 + stride;
                descend = (len & (preBlksEleNum + idx1)) != 0;
                if((descend && s_arr[idx1] < s_arr[idx2]) || (!descend && s_arr[idx1] > s_arr[idx2])){ 
                    T tem = s_arr[idx1];
                    s_arr[idx1] = s_arr[idx2];
                    s_arr[idx2] = tem;
                }
            }
        }
    }
    //copy sorted sublist from shared memory to global memory
    __syncthreads();
    idx = preBlksEleNum + threadIdx.x; 
    for(int i = threadIdx.x; i < eleNumInSharedMem; i += threadsNumPerBlock){
        d_arr[idx] = s_arr[i];
        idx += threadsNumPerBlock;
    }        
}


template <typename T>
__global__ void BitonicSort_GlobalMem(T* d_arr, int stride, int len, size_t dsize){
    int totalThreadNum = gridDim.x * threadsNumPerBlock; 
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    bool smallStride = stride < totalThreadNum;
    int idx1 = (smallStride)? 2 * tid - (tid & (stride - 1)) : tid;
    int idx2 = idx1 + stride; 
    bool descend = (len & idx1) != 0;

    if((descend && d_arr[idx1] < d_arr[idx2]) || (!descend && d_arr[idx1] > d_arr[idx2])){
        T tem = d_arr[idx1];
        d_arr[idx1] = d_arr[idx2];
        d_arr[idx2] = tem;
    }

    int range = stride / totalThreadNum;
    int round;
    for(round = 1; round < dsize / (totalThreadNum * 2); round++){
        if(smallStride){
            idx1 += totalThreadNum * 2;
        }
        else{
            if(round % range == 0)
                idx1 = idx1 + stride + totalThreadNum;
            else
                idx1 += totalThreadNum;
        }
        idx2 = idx1 + stride;
        descend = (len & idx1) != 0;
        if((descend && d_arr[idx1] < d_arr[idx2]) || (!descend && d_arr[idx1] > d_arr[idx2])){
            T tem = d_arr[idx1];
            d_arr[idx1] = d_arr[idx2];
            d_arr[idx2] = tem;
        }
    }
}



template <typename T>
__global__ void BitonicSort_SharedMem(T* d_arr, int stride, int len, int eleNumInSharedMem){
    __shared__ T s_Arr[smemCapa];
    //extern __shared__ T s_Arr[];
    int preBlksEleNum = blockIdx.x * eleNumInSharedMem;   //the num of array elements in previous blocks
    int idx = preBlksEleNum + threadIdx.x; 
    for(int i = threadIdx.x; i < eleNumInSharedMem; i += threadsNumPerBlock){ 
        s_Arr[i] = d_arr[idx];
        idx += threadsNumPerBlock;
    }        
    while(stride > 0){
        __syncthreads();
        int idx1 = (stride > threadsNumPerBlock)? threadIdx.x : 2 * threadIdx.x - (threadIdx.x & (stride - 1));
        int idx2 = idx1 + stride;
        bool descend = (len & (preBlksEleNum + idx1)) != 0; 
        if((descend && s_Arr[idx1] < s_Arr[idx2]) || (!descend && s_Arr[idx1] > s_Arr[idx2])){ 
            T tem = s_Arr[idx1];
            s_Arr[idx1] = s_Arr[idx2];
            s_Arr[idx2] = tem;
        }

        int flag = stride / threadsNumPerBlock;
        for(int round = 1; round < eleNumInSharedMem / twiceTNPB; round++){
            if(stride <= threadsNumPerBlock){
                idx1 += twiceTNPB;   //twiceTNPB: 2*threadNumPerBlock
            }
            else{
                if(round % flag == 0)
                    idx1 += (stride + threadsNumPerBlock);
                else
                    idx1 += threadsNumPerBlock;
            }
            idx2 = idx1 + stride;
            descend = (len & (preBlksEleNum + idx1)) != 0;
            if((descend && s_Arr[idx1] < s_Arr[idx2]) || (!descend && s_Arr[idx1] > s_Arr[idx2])){ 
                T tem = s_Arr[idx1];
                s_Arr[idx1] = s_Arr[idx2];
                s_Arr[idx2] = tem;
            }
        }
        stride >>= 1;
    }
    //copy sorted sublist from shared memory to global memory
    __syncthreads();
    idx = preBlksEleNum + threadIdx.x;
    for(int i = threadIdx.x; i < eleNumInSharedMem; i += threadsNumPerBlock){
        d_arr[idx] = s_Arr[i];
        idx += threadsNumPerBlock;
    }      
}



template <typename T>
T kthLargest(const std::vector<T>& h_input, size_t k) {
  const size_t n= h_input.size();
  if(k == 0 || k > n || n == 0){
    return T(-100);
  }

  size_t m = 1;
  while(m < n) m <<= 1;
  std::vector<T> h_padded_input(m);
  std::copy(h_input.begin(), h_input.end(), h_padded_input.begin());
  std::fill(h_padded_input.begin() + n, h_padded_input.end(), std::numeric_limits<T>::lowest());

  T* d_input = nullptr;
  CUDA_CHECK(cudaMalloc(&d_input, m * sizeof(T)));
  CUDA_CHECK(cudaMemcpy(d_input, h_padded_input.data(), m * sizeof(T), cudaMemcpyHostToDevice));
    
    int eleNumInSharedMem = smemCapa; //the number of array elements in the shared memory of a block
    if(eleNumInSharedMem > m)
        eleNumInSharedMem = m;
    int blocksNumPerGrid = m / eleNumInSharedMem;
    BitonicSort_Prolog<T><<<blocksNumPerGrid, threadsNumPerBlock>>>(d_input, eleNumInSharedMem);
    CUDA_CHECK(cudaGetLastError());

    for(int stage_size = eleNumInSharedMem * 2; stage_size <= m; stage_size <<= 1){
      for(int pass_size = stage_size >> 1; pass_size > 0; pass_size >>= 1){
            if(pass_size >= eleNumInSharedMem){
                BitonicSort_GlobalMem<T><<<blocksNumPerGrid, threadsNumPerBlock>>>(d_input, pass_size, stage_size, m);
                CUDA_CHECK(cudaGetLastError());
            }
            else{
                BitonicSort_SharedMem<T><<<blocksNumPerGrid, threadsNumPerBlock>>>(d_input, pass_size, stage_size, eleNumInSharedMem);
                CUDA_CHECK(cudaGetLastError());
                break;
            }       
      }
    }
    CUDA_CHECK(cudaDeviceSynchronize());

  T result;
  CUDA_CHECK(cudaMemcpy(&result, d_input + (m - k), sizeof(T), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d_input));
  return result;
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
__global__ void flashAttentionKernel(
    T* q, T* k, T* v, T* o, int batch_size,
    int target_seq_len, int src_seq_len, int query_heads, int kv_heads, int head_dim, bool is_causal) {

        int b_idx = blockIdx.z;
        int qh_head_idx = blockIdx.y;
        int q_row_idx = blockIdx.x * tileHeight + threadIdx.y;          
        
        int kv_head_idx = qh_head_idx * kv_heads / query_heads;
        int valid_seq_len = src_seq_len;


        //assume head_dim <= 128
        __shared__ T q_patch[tileHeight][128]; 
        __shared__ T k_patch[tileHeight][128];
        __shared__ T intermediate[tileHeight][2048];
        __shared__ T softmaxSum[tileHeight];
    
        //load Q tile of current block to smem  
        if(q_row_idx < target_seq_len && threadIdx.x < head_dim){
            int q_offset = b_idx * (target_seq_len * query_heads * head_dim) +
                           q_row_idx * (query_heads * head_dim) + qh_head_idx * head_dim + threadIdx.x;
            q_patch[threadIdx.y][threadIdx.x] = q[q_offset];
        }

        //load K to smem tile by tile
        int valid_row_curTile;
        for(int i = 0; i < (valid_seq_len - 1) / tileHeight + 1; i++){
            int k_row_idx = i * tileHeight + threadIdx.y;
 
            //load tile i to smem
            if(k_row_idx < valid_seq_len && threadIdx.x < head_dim){
                int k_offset = ((b_idx * src_seq_len + k_row_idx) * kv_heads + kv_head_idx) * head_dim + threadIdx.x;
                k_patch[threadIdx.y][threadIdx.x] = k[k_offset];
            }
            __syncthreads();

            //dot product: q patch * k patch
            valid_row_curTile = (i == (valid_seq_len - 1) / tileHeight)? valid_seq_len % tileHeight : tileHeight;
            if(valid_row_curTile == 0)
                valid_row_curTile = tileHeight;

            if(q_row_idx < target_seq_len && threadIdx.x < valid_row_curTile){
            	if(is_causal && q_row_idx < threadIdx.x + i * tileHeight)
            		intermediate[threadIdx.y][threadIdx.x + i * tileHeight] = -INFINITY;
            	else{
            		T sum = 0.0f;
            		for(int j = 0; j < head_dim; j++)
                    	sum += q_patch[threadIdx.y][j] * k_patch[threadIdx.x][j];
                	
                	intermediate[threadIdx.y][threadIdx.x + i * tileHeight] = sum; 
            	}                                             
            }   
        }
        __syncthreads();


        valid_row_curTile = (blockIdx.x == (target_seq_len - 1) / tileHeight)? target_seq_len % tileHeight : tileHeight;
        if(valid_row_curTile == 0)
            valid_row_curTile = tileHeight;

        //find max value
        if(threadIdx.y == 0 && threadIdx.x < valid_row_curTile){
            T maxVal = -INFINITY;          
            for(int i = 0; i < valid_seq_len; i++){
                if(maxVal < intermediate[threadIdx.x][i])
                    maxVal = intermediate[threadIdx.x][i];
            }
            //scale, exp
            T softmax_sum = 0.0f;
            T scale_factor = 1.0f / sqrt(static_cast<T>(head_dim));
            for(int i = 0; i < valid_seq_len; i++){
                intermediate[threadIdx.x][i] = (intermediate[threadIdx.x][i] > -INFINITY)? 
                								exp((intermediate[threadIdx.x][i] - maxVal) * scale_factor) : 0.0f;
                softmax_sum += intermediate[threadIdx.x][i];
            }
            softmaxSum[threadIdx.x] = softmax_sum;
        }
        __syncthreads();


        //dot product intermediate with V       
        if(q_row_idx < target_seq_len && threadIdx.x < head_dim){
          T dotProduct = 0.0f;
            for(int i = 0; i < valid_seq_len; i++){
                int v_offset = ((b_idx * src_seq_len + i) * kv_heads + kv_head_idx) * head_dim + threadIdx.x;
                dotProduct += intermediate[threadIdx.y][i] * v[v_offset];
            }
            //softmax
            if(softmaxSum[threadIdx.y] > 0.0f)
                dotProduct /= softmaxSum[threadIdx.y];
            //store it to o
            int o_offset = b_idx * (target_seq_len * query_heads * head_dim) +
                            q_row_idx * (query_heads * head_dim) + qh_head_idx * head_dim + threadIdx.x;
            o[o_offset] = dotProduct;
        }
}



template <typename T>
void flashAttention(const std::vector<T>& h_q, const std::vector<T>& h_k,
                    const std::vector<T>& h_v, std::vector<T>& h_o,
                    int batch_size, int target_seq_len, int src_seq_len, 
                    int query_heads, int kv_heads, int head_dim, bool is_causal) {
                    
    if (query_heads % kv_heads != 0) {
        return;
    }

    T *d_q = nullptr, *d_k = nullptr, *d_v = nullptr, *d_o = nullptr;
    CUDA_CHECK(cudaMalloc(&d_q, batch_size * target_seq_len * query_heads * head_dim * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_k, batch_size * src_seq_len * kv_heads * head_dim * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_v, batch_size * src_seq_len * kv_heads * head_dim * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_o, batch_size * target_seq_len * query_heads * head_dim * sizeof(T))); 

    CUDA_CHECK(cudaMemcpy(d_q, h_q.data(), batch_size * target_seq_len * query_heads * head_dim * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_k, h_k.data(), batch_size * src_seq_len * kv_heads * head_dim * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_v, h_v.data(), batch_size * src_seq_len * kv_heads * head_dim * sizeof(T), cudaMemcpyHostToDevice));   

    dim3 gridDim((target_seq_len + tileHeight - 1) / tileHeight, query_heads, batch_size);
    dim3 blockDim(max(tileHeight,head_dim), tileHeight);

    flashAttentionKernel<T><<<gridDim, blockDim>>>(
        d_q, d_k, d_v, d_o,
        batch_size, target_seq_len, src_seq_len,
        query_heads, kv_heads, head_dim,
        is_causal
    );

    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_o.data(), d_o, batch_size * target_seq_len * query_heads * head_dim * sizeof(T), cudaMemcpyDeviceToHost));

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
