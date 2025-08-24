#include <vector>

#include "../tester/utils.h"

#define threadsNumPerBlock 1024  //todo
#define twiceTNPB 2048    //2 * threadsNumPerBlock

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
  // TODO: Implement the kthLargest function

  uint h_size = h_input.size();
  if(k <= 0 || k > h_size)
  	return T(-100);

  uint smemCapa = 1 << 14;  //96KB shared memory can store 2^14 float/int
  uint d_arrSize = 1;
  while(d_arrSize < h_size){
        d_arrSize <<= 1;
  }
  T* d_arr;
  cudaMalloc((void**)&d_arr, d_arrSize * sizeof(T));
  //cudaHostRegister(h_input, d_size * sizeof(T), cudaHostAllocDefault);
  cudaMemcpyAsync(d_arr, h_input, h_size * sizeof(T), cudaMemcpyHostToDevice);
  cudaMemset(d_arr + h_size, 0xfe, sizeof(T) * (d_arrSize - h_size));  //0xfe may error
  
  uint eleNumInSharedMem = min(16384, d_arrSize); //todo.the number of array elements in the shared memory of a block
    uint blocksNumPerGrid = d_arrSize / eleNumInSharedMem;
    //printf("d_arrSize %d,  blockNum %d,  eleNumInBlock %d\n", d_arrSize, blocksNumPerGrid, eleNumInSharedMem);
    BitonicSort_Prolog<<<blocksNumPerGrid, threadsNumPerBlock>>>(d_arr, size, eleNumInSharedMem);

    for(uint len = eleNumInSharedMem << 1; len <= d_arrSize; len <<= 1){
        //printf("============== stage ============ length: %d\n", len);
        for(uint stride = len >> 1; stride > 0; stride >>= 1){
            //printf("..........step.........: stride: %d\n",stride);
            if(stride >= eleNumInSharedMem){
                BitonicSort_GlobalMem<<<blocksNumPerGrid, threadsNumPerBlock>>>(d_arr, stride, len, d_arrSize, eleNumInSharedMem);
            }
            else{
                BitonicSort_SharedMem<<<blocksNumPerGrid, threadsNumPerBlock>>>(d_arr, stride, len, d_arrSize, eleNumInSharedMem);
                continue;
            }
        }
    }

    T res;
    cudaMemcpy(&res, d_arr + d_arrSize - k, sizeof(T), cudaMemcpyDeviceToHost);
    return res;
  }    

__global__ void BitonicSort_Prolog(short* d_arr, uint h_arrSize, uint eleNumInSharedMem){
    //extern __shared__ short s_arr[];        //todo:create 2 arrays to avoid bank conflict?
    __shared__ short s_arr[16384];  //todo
    uint preBlksEleNum = blockIdx.x * eleNumInSharedMem;   //the num of array elements in previous blocks
    uint idx = preBlksEleNum + threadIdx.x;  //idx: index in d_arr
    for(uint i = threadIdx.x; i < eleNumInSharedMem; i += threadsNumPerBlock){
        if(idx >= h_arrSize)
            s_arr[i] = 1001;  //pad with 1001
        else
            s_arr[i] = d_arr[idx];
        idx += threadsNumPerBlock;
    }        

    //each block processes a sublist in its shared memory
    for(int len = 2; len <= eleNumInSharedMem; len <<= 1){         
        for(uint stride = len >> 1; stride > 0; stride >>= 1){
            __syncthreads();

            //compare s_arr[idx1] and s_arr[idx2]
            uint idx1 = (stride > threadsNumPerBlock)? threadIdx.x : 2 * threadIdx.x - (threadIdx.x & (stride - 1));
            uint idx2 = idx1 + stride;
            bool descend = (len & (preBlksEleNum + idx1)) != 0; 
            if((descend && s_arr[idx1] < s_arr[idx2]) || (!descend && s_arr[idx1] > s_arr[idx2])){ 
                short tem = s_arr[idx1];
                s_arr[idx1] = s_arr[idx2];
                s_arr[idx2] = tem;
            }

            uint flag = stride / threadsNumPerBlock;
            for(uint round = 1; round < eleNumInSharedMem / twiceTNPB; round++){
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
                    short tem = s_arr[idx1];
                    s_arr[idx1] = s_arr[idx2];
                    s_arr[idx2] = tem;
                }
            }
        }
    }
    //copy sorted sublist from shared memory to global memory
    __syncthreads();
    idx = preBlksEleNum + threadIdx.x;  //idx: index in d_arr
    for(uint i = threadIdx.x; i < eleNumInSharedMem; i += threadsNumPerBlock){
        d_arr[idx] = s_arr[i];
        idx += threadsNumPerBlock;
    }        
}



__global__ void BitonicSort_GlobalMem(short* d_arr, uint stride, uint len, uint dsize, uint eleNumInSharedMem){
    uint totalThreadNum = gridDim.x * threadsNumPerBlock; 
    uint tid = blockIdx.x * blockDim.x + threadIdx.x;
    bool smallStride = stride < totalThreadNum;
    uint idx1 = (smallStride)? 2 * tid - (tid & (stride - 1)) : tid;
    uint idx2 = idx1 + stride; 
    bool descend = (len & idx1) != 0;

    /*debug
    short* arr1 = (short*)malloc(64 * sizeof(short));
    short* arr2 = (short*)malloc(64 * sizeof(short));
    bool* signal = (bool*)malloc(64 * sizeof(bool));
    uint p = 0;
    arr1[p] = idx1; arr2[p] = idx2; signal[p] = descend;
    **/

    if((descend && d_arr[idx1] < d_arr[idx2]) || (!descend && d_arr[idx1] > d_arr[idx2])){
        short tem = d_arr[idx1];
        d_arr[idx1] = d_arr[idx2];
        d_arr[idx2] = tem;
    }

    uint range = stride / totalThreadNum;
    uint round;
    for(round = 1; round < dsize / (totalThreadNum * 2); round++){
        if(smallStride){
            idx1 += (totalThreadNum << 1);
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
            short tem = d_arr[idx1];
            d_arr[idx1] = d_arr[idx2];
            d_arr[idx2] = tem;
        }
        /*
        p++;
        arr1[p] = idx1; arr2[p] = idx2; signal[p] = descend;**/
    }

    /* debug
    for(uint e = 0; e <= p; e++){
        printf("bid %d : tid %d: global swap, stride= %d.....,     %d  vs  %d,  down? %d\n",  blockIdx.x, threadIdx.x, stride, arr1[e], arr2[e], signal[e]);
    } 
    **/ 
}



__global__ void BitonicSort_SharedMem(short* d_arr, uint stride, uint len, uint dsize, uint eleNumInSharedMem){
    __shared__ short s_arr[16384];
    //extern __shared__ short s_arr[];
    uint preBlksEleNum = blockIdx.x * eleNumInSharedMem;   //the num of array elements in previous blocks
    uint idx = preBlksEleNum + threadIdx.x;  //idx: index in d_arr
    for(uint i = threadIdx.x; i < eleNumInSharedMem; i += threadsNumPerBlock){ 
        s_arr[i] = d_arr[idx];
        idx += threadsNumPerBlock;
    }        
    while(stride > 0){
        __syncthreads();
        uint idx1 = (stride > threadsNumPerBlock)? threadIdx.x : 2 * threadIdx.x - (threadIdx.x & (stride - 1));
        uint idx2 = idx1 + stride;
        bool descend = (len & (preBlksEleNum + idx1)) != 0; 
        if((descend && s_arr[idx1] < s_arr[idx2]) || (!descend && s_arr[idx1] > s_arr[idx2])){ 
            short tem = s_arr[idx1];
            s_arr[idx1] = s_arr[idx2];
            s_arr[idx2] = tem;
        }

        uint flag = stride / threadsNumPerBlock;
        for(uint round = 1; round < eleNumInSharedMem / twiceTNPB; round++){
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
                short tem = s_arr[idx1];
                s_arr[idx1] = s_arr[idx2];
                s_arr[idx2] = tem;
            }
        }
        stride >>= 1;
    }
    //copy sorted sublist from shared memory to global memory
    __syncthreads();
    idx = preBlksEleNum + threadIdx.x;  //idx: index in d_arr
    for(uint i = threadIdx.x; i < eleNumInSharedMem; i += threadsNumPerBlock){
        d_arr[idx] = s_arr[i];
        idx += threadsNumPerBlock;
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
    float scaleFactor = 1.0 / sqrt(head_dim);   
    T* d_q;
    uint sizeQ = batch_size * tgt_seq_len * query_heads * head_dim;
    cudaMalloc((void**)&d_q, sizeQ * sizeof(T)); 
    cudaMemcpyAsync(d_q, h_q, sizeQ * sizeof(T), cudaMemcpyHostToDevice);   
    T* d_k;
    uint sizeK = batch_size * src_seq_len * kv_heads * head_dim;
    cudaMalloc((void**)&d_k, sizeK * sizeof(T)); 
    cudaMemcpyAsync(d_k, h_k, sizeK * sizeof(T), cudaMemcpyHostToDevice);       
    T* d_v;
    cudaMalloc((void**)&d_v, sizeK * sizeof(T)); 
    cudaMemcpyAsync(d_v, h_v, sizeK * sizeof(T), cudaMemcpyHostToDevice);
    T* d_o;
    cudaMalloc((void**)&d_o, sizeQ * sizeof(T));
    cudaMemsetAsync(d_o, 0x00, sizeQ * sizeof(T));

    T* d_l;
    uint sizeL = batch_size * tgt_seq_len * query_heads;
    cudaMalloc((void**)&d_l,  sizeL * sizeof(T));
    cudaMemsetAsync(d_l, 0x00, sizeL * sizeof(T));
    
    T* d_m;
    cudaMalloc((void**)&d_m,  sizeL * sizeof(T));
    cudaMemsetAsync(d_m, 0xfe, sizeL * sizeof(T));

    dim3 grid_dim(batch_size, tgt_seq_len);
    dim3 block_dim(32);
    int Tcr = (query_heads - 1) / 32 + 1;

    attentionKernel<<<grid_dim, block_dim, sram_size>>>(d_q,d_k,d_v,d_o,d_l,d_m,scaleFactor,Tcr, batch_size, target_seq_len, src_seq_len, query_heads, kv_heads, head_dim, s_causal);

    cudaMemcpyAsync(h_o, d_o, sizeQ * sizeof(T), cudaMemcpyDeviceToHost);  
}






__global__ void attentionKernel(T* d_q, T* d_k, T* d_v, T* d_o, T* l, T* m, float scaleFactor, int Tcr, int batch_size, int target_seq_len, int src_seq_len, int query_heads, int kv_heads, int head_dim, bool s_causal){
  extern __shared__ float sram[];
  int tx = threadIdx.x;
  int bx = blockIdx.x; 
  int by = blockIdx.y;
  int qkv_offset = (bx * gridDim.y * query_heads * head_dim) + (by * query_heads * head_dim);
  int lm_offset = (bx * gridDim.y * query_heads) + (by * query_heads);
  int tile_size = 32 * head_dim;
  float* Qi = sram;
    float* Kj = &sram[tile_size];
    float* Vj = &sram[tile_size * 2];
    float* S = &sram[tile_size * 3];

    for (int j = 0; j < Tcr; j++) {
        for (int x = 0; x < head_dim; x++) {
            Kj[(tx * head_dim) + x] = d_k[qkv_offset + (tile_size * j) + (tx * head_dim) + x];
            Vj[(tx * head_dim) + x] = d_v[qkv_offset + (tile_size * j) + (tx * head_dim) + x];
        }
        __syncthreads();  // such that the inner loop can use the correct Kj, Vj

        for (int i = 0; i < Tcr; i++)  {

            // Load Qi to SRAM, l and m to registers
            for (int x = 0; x < head_dim; x++) {
                Qi[(tx * head_dim) + x] = d_q[qkv_offset + (tile_size * i) + (tx * head_dim) + x];
            }
            float row_m_prev = m[lm_offset + (Br * i) + tx];
            float row_l_prev = l[lm_offset + (Br * i) + tx];

            // S = QK^T, row_m = rowmax(S)
            float row_m = -INFINITY;
            for (int y = 0; y < 32; y++) {
                float sum = 0;
                for (int x = 0; x < head_dim; x++) {
                    sum += Qi[(tx * head_dim) + x] * Kj[(y * head_dim) + x];
                }
                sum *= scaleFactor;
                S[(32 * tx) + y] = sum;

                if (sum > row_m)
                    row_m = sum;
            }

            // P = exp(S - row_m), row_l = rowsum(P)
            float row_l = 0;
            for (int y = 0; y < 32; y++) {
                S[(32 * tx) + y] = __expf(S[(32 * tx) + y] - row_m);
                row_l += S[(32 * tx) + y];
            }

            // Compute new m and l
            float row_m_new = max(row_m_prev, row_m);
            float row_l_new = (__expf(row_m_prev - row_m_new) * row_l_prev) + (__expf(row_m - row_m_new) * row_l);

            // Write O, l, m to HBM
            for (int x = 0; x < head_dim; x++) {
                float pv = 0;  
                for (int y = 0; y < 32; y++) {
                    pv += S[(32 * tx) + y] * Vj[(y * head_dim) + x];
                }
                d_o[qkv_offset + (tile_size * i) + (tx * head_dim) + x] = (1 / row_l_new) \
                    * ((row_l_prev * __expf(row_m_prev - row_m_new) * d_o[qkv_offset + (tile_size * i) + (tx * head_dim) + x]) \
                    + (__expf(row_m - row_m_new) * pv));
            }
            m[lm_offset + (Br * i) + tx] = row_m_new;
            l[lm_offset + (Br * i) + tx] = row_l_new;
        }
        __syncthreads();  // otherwise, thread can use the wrong Kj, Vj in inner loop
    }

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
