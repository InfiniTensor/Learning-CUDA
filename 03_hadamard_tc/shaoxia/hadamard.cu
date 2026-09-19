#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <cassert>
#include <type_traits>

#define WARP_SIZE 32
#define LD128BIT_CONST(value) (*(reinterpret_cast<const float4*>(&value)))
#define LD128BIT(value) (*(reinterpret_cast<float4*>(&value)))


struct alignas(16) Half8 {
    __half x[8];
};

struct alignas(16) BFloat16_8 {
    __nv_bfloat16 x[8];
};


static_assert(sizeof(Half8) == 16);
static_assert(sizeof(BFloat16_8) == 16);

__device__ __forceinline__
void load_8_to_float8(
    const __half* ptr,
    float (&v)[8]
) {
    Half8 h = *reinterpret_cast<const Half8*>(ptr);

    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        v[i] = __half2float(h.x[i]);
    }
}


__device__ __forceinline__
void load_8_to_float8(
    const __nv_bfloat16* ptr,
    float (&v)[8]
) {
    BFloat16_8 h =
        *reinterpret_cast<const BFloat16_8*>(ptr);

    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        v[i] = __bfloat162float(h.x[i]);
    }
}



__device__ __forceinline__
void store_float8_as_8(
    const float (&v)[8],
    __half* ptr
) {
    __half2 h2[4];

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        h2[i] = __floats2half2_rn(
            v[2 * i],
            v[2 * i + 1]
        );
    }

    uint4 packed =
        *reinterpret_cast<const uint4*>(h2);

    *reinterpret_cast<uint4*>(ptr) = packed;
}

__device__ __forceinline__
void store_float8_as_8(
    const float (&v)[8],
    __nv_bfloat16* ptr
) {
    __nv_bfloat162 b2[4];

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        b2[i] = __floats2bfloat162_rn(
            v[2 * i],
            v[2 * i + 1]
        );
    }

    uint4 packed =
        *reinterpret_cast<const uint4*>(b2);

    *reinterpret_cast<uint4*>(ptr) = packed;
}

template <
    typename T,
    const int COLS,
    const int ROWS_PER_BLOCK
>
__global__ void hadamard_kernel_scalar(const T* input, T* output, const int rows) {
    static_assert(COLS >= WARP_SIZE, "COLS must be >= WARP_SIZE");
    constexpr int NUM_PER_THREAD = COLS / WARP_SIZE;
    float local[NUM_PER_THREAD];

    const int tid = threadIdx.x;
    const int lane_id = tid % WARP_SIZE;
    const int warp_id = tid / WARP_SIZE;
    const int row = blockIdx.x * ROWS_PER_BLOCK + warp_id;

    if (row >= rows) {
        return;
    }

    #pragma unroll
    for (int i = 0; i < NUM_PER_THREAD; ++i) {
        local[i] = float(input[row * COLS + lane_id + i * WARP_SIZE]);
        #pragma unroll
        for (int j = 1 ; j < WARP_SIZE ; j <<= 1) {
            float other = __shfl_xor_sync(0xffffffff, local[i], j);
            if (lane_id & j) {
                local[i] = other - local[i];
            } else {
                local[i] = local[i] + other;
            }
        }
    }

    #pragma unroll
    for (int i = 1 ; i < NUM_PER_THREAD ; i <<= 1) {
        #pragma unroll
        for (int j = 0 ; j < NUM_PER_THREAD ; ++j) {
            if ((j & i) == 0) {
                float a = local[j];
                float b = local[j ^ i];
                local[j] = a + b;
                local[j ^ i] = a - b;
            }
        }
    }
    #pragma unroll
    for (int i = 0 ; i < NUM_PER_THREAD; ++i) {
        output[row * COLS + lane_id + i * WARP_SIZE] = T(local[i]);
    }
}

template <typename T, const int COLS>
__global__ void hadamard_kernel_small(const T* input, T* output, const int rows) {
    // One warp handles WARP_SIZE / COLS rows; each row uses COLS lanes (one element
    // per lane), so the FWHT butterfly is done with __shfl_xor_sync. COLS must be a
    // power of 2 that divides WARP_SIZE (2/4/8/16).
    static_assert(COLS < WARP_SIZE && WARP_SIZE % COLS == 0,
                  "COLS must be a power of 2 dividing WARP_SIZE and < WARP_SIZE");
    constexpr int ROWS_PER_WARP = WARP_SIZE / COLS;

    const int tid = threadIdx.x;
    const int lane_id = tid % WARP_SIZE;
    const int warp_id = tid / WARP_SIZE;
    const int lane_in_row = lane_id % COLS;          // element index within the row
    const int row = (blockIdx.x * (blockDim.x / WARP_SIZE) + warp_id) * ROWS_PER_WARP
                    + lane_id / COLS;

    // Keep every lane in the shuffle (mask 0xffffffff) even when its row is out of
    // range: a full row group (COLS lanes) is either all-active or all-inactive, and
    // the XOR shuffle only couples lanes within the same group, so inactive lanes
    // never corrupt an active row. Only the load/store are guarded.
    const bool active = row < rows;
    float local = active ? float(input[row * COLS + lane_in_row]) : 0.0f;

    #pragma unroll
    for (int j = 1; j < COLS; j <<= 1) {
        float other = __shfl_xor_sync(0xffffffff, local, j);
        if (lane_in_row & j) {
            local = other - local;
        } else {
            local = local + other;
        }
    }

    if (active) {
        output[row * COLS + lane_in_row] = T(local);
    }
}



template <
    typename T,
    const int COLS,
    const int ROWS_PER_BLOCK
>
__global__ void hadamard_kernel_vec(const T* input, T* output, const int rows) {
    constexpr int VEC_SIZE = 8;
    constexpr int ELEMS_PER_WARP = VEC_SIZE * WARP_SIZE;
    constexpr int VEC_PER_THREAD = COLS / VEC_SIZE / WARP_SIZE;
    float local[VEC_PER_THREAD][8];
    static_assert(COLS >= ELEMS_PER_WARP,
              "COLS must be >= 256");
    static_assert(COLS % ELEMS_PER_WARP == 0,
                "COLS must be divisible by 256");
    static_assert((VEC_PER_THREAD & (VEC_PER_THREAD - 1)) == 0,
                "VEC_PER_THREAD must be power of two");

    const int tid = threadIdx.x;
    const int lane_id = tid % WARP_SIZE;
    const int warp_id = tid / WARP_SIZE;
    const int row = blockIdx.x * ROWS_PER_BLOCK + warp_id;

    if (row >= rows) {
        return;
    }

    #pragma unroll
    for (int i = 0; i < VEC_PER_THREAD; ++i) {
        load_8_to_float8(&input[row * COLS + lane_id * VEC_SIZE + i * WARP_SIZE * VEC_SIZE], local[i]);
        #pragma unroll
        for (int j = 1 ; j < 8 ; j <<= 1) {
            #pragma unroll
            for (int k = 0; k < 8; ++k) {
                if (k & j){
                    float b = local[i][k];
                    float a = local[i][k ^ j];
                    local[i][k] = a - b;
                    local[i][k ^ j] = a + b;
                }   
            }
        }

        #pragma unroll
        for (int j = 1 ; j < WARP_SIZE ; j <<= 1) {
            #pragma unroll
            for (int k = 0; k < 8; ++k) {
                float other = __shfl_xor_sync(0xffffffff, local[i][k], j);
                if (lane_id & j) {
                    local[i][k] = other - local[i][k];
                } else {
                    local[i][k] = local[i][k] + other;
                }
            }
        }
    }

    #pragma unroll
    for (int i = 1 ; i < VEC_PER_THREAD ; i <<= 1) {
        #pragma unroll
        for (int j = 0 ; j < VEC_PER_THREAD ; ++j) {
            #pragma unroll
            for (int k = 0; k < 8; ++k) {
                if ((j & i) == 0) {
                    float a = local[j][k];
                    float b = local[j ^ i][k];
                    local[j][k] = a + b;
                    local[j ^ i][k] = a - b;
                }
            }
            
        }
    }
    
    #pragma unroll
    for (int i = 0 ; i < VEC_PER_THREAD; ++i) {
        store_float8_as_8(local[i], &output[row * COLS + lane_id * VEC_SIZE + i * WARP_SIZE * VEC_SIZE]);
    }
}

template<
    typename T,
    const int COLS,
    const int WARPS_PER_ROW
>
__global__ void hadamard_kernel_multi_warp_per_row(const T *input, T *output, const int rows){
    constexpr int VEC_SIZE = 8;
    constexpr int ELEMENT_NUM_PER_WARP = COLS / WARPS_PER_ROW;
    constexpr int NUM_VEC = ELEMENT_NUM_PER_WARP / WARP_SIZE / VEC_SIZE;
    __shared__ float smem[WARPS_PER_ROW][NUM_VEC][VEC_SIZE][WARP_SIZE];

    const int row = blockIdx.x;
    if (row >= rows)
        return;
    const int tid = threadIdx.x;
    const int lane_id = tid % WARP_SIZE;
    const int warp_id = tid / WARP_SIZE;
    const int col_offset = warp_id * ELEMENT_NUM_PER_WARP;

    float local[NUM_VEC][VEC_SIZE];

    #pragma unroll
    for (int i = 0; i < NUM_VEC; ++i) {
        load_8_to_float8(&input[row * COLS + col_offset + i * WARP_SIZE * VEC_SIZE + lane_id * VEC_SIZE], local[i]);
        #pragma unroll
        for (int j = 1; j < VEC_SIZE; j <<= 1){
            #pragma unroll
            for (int k = 0; k < VEC_SIZE; ++k){
                if (k & j){
                    float a = local[i][k ^ j];
                    float b = local[i][k];
                    local[i][k ^ j] = a + b;
                    local[i][k] = a - b;
                }
            }
        }

        #pragma unroll
        for (int j = 1; j < WARP_SIZE; j <<= 1) {
            #pragma unroll
            for (int k = 0; k < VEC_SIZE; ++k) {
                float other = __shfl_xor_sync(0xffffffff, local[i][k], j);
                if (lane_id & j) {
                    local[i][k] = other - local[i][k];
                }else {
                    local[i][k] = local[i][k] + other;
                }
            }
        }
    }

    #pragma unroll
    for (int i = 1 ; i < NUM_VEC ; i <<= 1) {
        #pragma unroll
        for (int j = 0; j < NUM_VEC; ++j)
        {
            #pragma unroll
            for (int k = 0; k < VEC_SIZE; ++k) {
                if ((j & i) == 0) {
                    float a = local[j][k];
                    float b = local[j ^ i][k];
                    local[j][k] = a + b;
                    local[j ^ i][k] = a - b;
                }
            }
        }
    }

    for (int i = 0; i < NUM_VEC; i++)
        for (int k = 0; k < VEC_SIZE; k++)
            smem[warp_id][i][k][lane_id] = local[i][k];


    __syncthreads();

    #pragma unroll
    for (int i = 1; i < WARPS_PER_ROW; i <<= 1) {
        #pragma unroll
        for (int j = 0; j < NUM_VEC; ++j) {
            #pragma unroll
            for (int k = 0; k < VEC_SIZE; ++k){
                float other = smem[warp_id ^ i][j][k][lane_id];
                if (warp_id & i) {
                    local[j][k] = other - local[j][k];
                }else {
                    local[j][k] = local[j][k] + other;
                }
                
            }
        }
        __syncthreads();
        #pragma unroll
        for (int j = 0; j < NUM_VEC; ++j) {
            #pragma unroll
            for (int k = 0; k < VEC_SIZE; ++k){
                smem[warp_id][j][k][lane_id] = local[j][k];
            }
        }
        __syncthreads();
    }
    
    for (int i = 0; i < NUM_VEC; ++i){
        store_float8_as_8(local[i], &output[row * COLS + col_offset + i * WARP_SIZE * VEC_SIZE + lane_id * VEC_SIZE]);
    }
}

template<
    typename T,
    const int COLS,
    const int WARPS_PER_ROW
>
__global__ void hadamard_kernel_multi_warp_per_row_chunked(const T *input, T *output, const int rows){
    constexpr int VEC_SIZE = 8;
    constexpr int ELEMENT_NUM_PER_WARP = COLS / WARPS_PER_ROW;
    constexpr int NUM_VEC = ELEMENT_NUM_PER_WARP / WARP_SIZE / VEC_SIZE;
    // Fixed 32 KB of shared memory for the cross-warp exchange. For COLS > 8192 the
    // whole row no longer fits, so the exchange is done in rounds of CHUNKS_PER_ROUND
    // chunks at a time instead of all NUM_VEC chunks. Butterfly stages act on
    // orthogonal bit ranges, so chunks are independent and can be batched.
    constexpr int SMEM_BYTES = 32 * 1024;
    constexpr int CHUNKS_PER_ROUND = SMEM_BYTES / (WARPS_PER_ROW * VEC_SIZE * WARP_SIZE * (int)sizeof(float));
    constexpr int NUM_ROUNDS = NUM_VEC / CHUNKS_PER_ROUND;
    static_assert(CHUNKS_PER_ROUND >= 1, "CHUNKS_PER_ROUND must be >= 1");
    static_assert(NUM_VEC % CHUNKS_PER_ROUND == 0, "NUM_VEC must be divisible by CHUNKS_PER_ROUND");
    __shared__ float smem[WARPS_PER_ROW][CHUNKS_PER_ROUND][VEC_SIZE][WARP_SIZE];

    const int row = blockIdx.x;
    if (row >= rows)
        return;
    const int tid = threadIdx.x;
    const int lane_id = tid % WARP_SIZE;
    const int warp_id = tid / WARP_SIZE;
    const int col_offset = warp_id * ELEMENT_NUM_PER_WARP;

    float local[NUM_VEC][VEC_SIZE];

    #pragma unroll
    for (int i = 0; i < NUM_VEC; ++i) {
        load_8_to_float8(&input[row * COLS + col_offset + i * WARP_SIZE * VEC_SIZE + lane_id * VEC_SIZE], local[i]);
        #pragma unroll
        for (int j = 1; j < VEC_SIZE; j <<= 1){
            #pragma unroll
            for (int k = 0; k < VEC_SIZE; ++k){
                if (k & j){
                    float a = local[i][k ^ j];
                    float b = local[i][k];
                    local[i][k ^ j] = a + b;
                    local[i][k] = a - b;
                }
            }
        }

        #pragma unroll
        for (int j = 1; j < WARP_SIZE; j <<= 1) {
            #pragma unroll
            for (int k = 0; k < VEC_SIZE; ++k) {
                float other = __shfl_xor_sync(0xffffffff, local[i][k], j);
                if (lane_id & j) {
                    local[i][k] = other - local[i][k];
                }else {
                    local[i][k] = local[i][k] + other;
                }
            }
        }
    }

    #pragma unroll
    for (int i = 1 ; i < NUM_VEC ; i <<= 1) {
        #pragma unroll
        for (int j = 0; j < NUM_VEC; ++j)
        {
            #pragma unroll
            for (int k = 0; k < VEC_SIZE; ++k) {
                if ((j & i) == 0) {
                    float a = local[j][k];
                    float b = local[j ^ i][k];
                    local[j][k] = a + b;
                    local[j ^ i][k] = a - b;
                }
            }
        }
    }

    // Cross-warp exchange, batched so smem stays fixed at 32 KB.
    #pragma unroll
    for (int r = 0; r < NUM_ROUNDS; ++r) {
        const int base = r * CHUNKS_PER_ROUND;
        #pragma unroll
        for (int c = 0; c < CHUNKS_PER_ROUND; ++c)
            #pragma unroll
            for (int k = 0; k < VEC_SIZE; ++k)
                smem[warp_id][c][k][lane_id] = local[base + c][k];

        __syncthreads();

        #pragma unroll
        for (int i = 1; i < WARPS_PER_ROW; i <<= 1) {
            #pragma unroll
            for (int c = 0; c < CHUNKS_PER_ROUND; ++c) {
                #pragma unroll
                for (int k = 0; k < VEC_SIZE; ++k){
                    float other = smem[warp_id ^ i][c][k][lane_id];
                    if (warp_id & i) {
                        local[base + c][k] = other - local[base + c][k];
                    }else {
                        local[base + c][k] = local[base + c][k] + other;
                    }
                }
            }
            __syncthreads();
            #pragma unroll
            for (int c = 0; c < CHUNKS_PER_ROUND; ++c) {
                #pragma unroll
                for (int k = 0; k < VEC_SIZE; ++k){
                    smem[warp_id][c][k][lane_id] = local[base + c][k];
                }
            }
            __syncthreads();
        }
    }

    #pragma unroll
    for (int i = 0; i < NUM_VEC; ++i){
        store_float8_as_8(local[i], &output[row * COLS + col_offset + i * WARP_SIZE * VEC_SIZE + lane_id * VEC_SIZE]);
    }
}

template <typename T>
void hadamard_v1(const T *input, T *output, int rows, int cols,
                  cudaStream_t stream = 0){
    static_assert(std::is_same<T, __half>::value ||
                      std::is_same<T, __nv_bfloat16>::value,
                  "T must be __half or __nv_bfloat16");
    int log2_n = 0;
    while ((1 << log2_n) < cols) {
        ++log2_n;
    }
    assert((1 << log2_n) == cols && "cols must be a power of 2");

    constexpr int ROWS_PER_BLOCK = 4;
    constexpr int THREADS_PER_BLOCK = ROWS_PER_BLOCK * WARP_SIZE;
    dim3 block(THREADS_PER_BLOCK);
    dim3 grid((rows + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK);

    switch (cols) {
        case 32:
            hadamard_kernel_scalar<T, 32, ROWS_PER_BLOCK><<<grid, block, 0, stream>>>(input, output, rows);
            break;
        case 64:
            hadamard_kernel_scalar<T, 64, ROWS_PER_BLOCK><<<grid, block, 0, stream>>>(input, output, rows);
            break;
        case 128:
            hadamard_kernel_scalar<T, 128, ROWS_PER_BLOCK><<<grid, block, 0, stream>>>(input, output, rows);
            break;
        case 256:
            hadamard_kernel_scalar<T, 256, ROWS_PER_BLOCK><<<grid, block, 0, stream>>>(input, output, rows);
            break;
        case 512:
            hadamard_kernel_scalar<T, 512, ROWS_PER_BLOCK><<<grid, block, 0, stream>>>(input, output, rows);
            break;
        case 1024:
            hadamard_kernel_scalar<T, 1024, ROWS_PER_BLOCK><<<grid, block, 0, stream>>>(input, output, rows);
            break;
        case 2048:
            hadamard_kernel_scalar<T, 2048, ROWS_PER_BLOCK><<<grid, block, 0, stream>>>(input, output, rows);
            break;
        case 4096:
            hadamard_kernel_scalar<T, 4096, ROWS_PER_BLOCK><<<grid, block, 0, stream>>>(input, output, rows);
            break;
        case 8192:
            hadamard_kernel_scalar<T, 8192, ROWS_PER_BLOCK><<<grid, block, 0, stream>>>(input, output, rows);
            break;
    default:
        assert(false && "cols must be a power of 2 in [32, 1024]");
    }
}

template <typename T>
void hadamard_v2(const T *input, T *output, int rows, int cols,
                 cudaStream_t stream = 0)
{
    static_assert(std::is_same<T, __half>::value ||
                      std::is_same<T, __nv_bfloat16>::value,
                  "T must be __half or __nv_bfloat16");
    int log2_n = 0;
    while ((1 << log2_n) < cols)
    {
        ++log2_n;
    }
    assert((1 << log2_n) == cols && "cols must be a power of 2");

    constexpr int ROWS_PER_BLOCK = 4;
    constexpr int THREADS_PER_BLOCK = ROWS_PER_BLOCK * WARP_SIZE;
    dim3 block(THREADS_PER_BLOCK);
    dim3 grid((rows + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK);

    switch (cols)
    {
    case 256:
        hadamard_kernel_vec<T, 256, ROWS_PER_BLOCK><<<grid, block, 0, stream>>>(input, output, rows);
        break;
    case 512:
        hadamard_kernel_vec<T, 512, ROWS_PER_BLOCK><<<grid, block, 0, stream>>>(input, output, rows);
        break;
    case 1024:
        hadamard_kernel_vec<T, 1024, ROWS_PER_BLOCK><<<grid, block, 0, stream>>>(input, output, rows);
        break;
    case 2048:
        hadamard_kernel_vec<T, 2048, ROWS_PER_BLOCK><<<grid, block, 0, stream>>>(input, output, rows);
        break;
    case 4096:
        hadamard_kernel_vec<T, 4096, ROWS_PER_BLOCK><<<grid, block, 0, stream>>>(input, output, rows);
        break;
    case 8192:
        hadamard_kernel_vec<T, 8192, ROWS_PER_BLOCK><<<grid, block, 0, stream>>>(input, output, rows);
        break;
    default:
        assert(false && "cols must be a power of 2 in [32, 1024]");
    }
}

template <typename T>
void hadamard_v3(const T *input, T *output, int rows, int cols,
                 cudaStream_t stream = 0)
{
    static_assert(std::is_same<T, __half>::value ||
                      std::is_same<T, __nv_bfloat16>::value,
                  "T must be __half or __nv_bfloat16");
    int log2_n = 0;
    while ((1 << log2_n) < cols)
    {
        ++log2_n;
    }
    assert((1 << log2_n) == cols && "cols must be a power of 2");

    if (cols <= 2048) {
        constexpr int ROWS_PER_BLOCK = 4;
        constexpr int THREADS_PER_BLOCK = ROWS_PER_BLOCK * WARP_SIZE;
        dim3 block(THREADS_PER_BLOCK);
        dim3 grid((rows + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK);
        switch (cols)
        {
        case 1024:
            hadamard_kernel_vec<T, 1024, ROWS_PER_BLOCK><<<grid, block, 0, stream>>>(input, output, rows);
            break;
        case 2048:
            hadamard_kernel_vec<T, 2048, ROWS_PER_BLOCK><<<grid, block, 0, stream>>>(input, output, rows);
            break;
        default:
            assert(false && "cols no");
        }
    }else {
        constexpr int THREADS_PER_BLOCK = 128;
        constexpr int WARPS_PER_ROW = THREADS_PER_BLOCK / WARP_SIZE;
        dim3 block(THREADS_PER_BLOCK);
        dim3 grid(rows);
        switch (cols)
        {
        case 4096:
            hadamard_kernel_multi_warp_per_row<T, 4096, WARPS_PER_ROW><<<grid, block, 0, stream>>>(input, output, rows);
            break;
        case 8192: {
            hadamard_kernel_multi_warp_per_row<T, 8192, WARPS_PER_ROW><<<grid, block, 0, stream>>>(input, output, rows);
            break;
        }
        case 16384: {
            constexpr int WARPS_PER_ROW = 8;
            constexpr int THREADS_PER_BLOCK = WARPS_PER_ROW * WARP_SIZE;
            dim3 block(THREADS_PER_BLOCK);
            dim3 grid(rows);
            hadamard_kernel_multi_warp_per_row_chunked<T, 16384, WARPS_PER_ROW><<<grid, block, 0, stream>>>(input, output, rows);
            break;
        }
        case 32768: {
            constexpr int WARPS_PER_ROW = 16;
            constexpr int THREADS_PER_BLOCK = WARPS_PER_ROW * WARP_SIZE;
            dim3 block(THREADS_PER_BLOCK);
            dim3 grid(rows);
            hadamard_kernel_multi_warp_per_row_chunked<T, 32768, WARPS_PER_ROW><<<grid, block, 0, stream>>>(input, output, rows);
            break;
        }
        default:
            assert(false && "cols no");
        }
    }
}

template <typename T>
void hadamard_v4(const T *input, T *output, int rows, int cols,
                 cudaStream_t stream = 0)
{
    static_assert(std::is_same<T, __half>::value ||
                      std::is_same<T, __nv_bfloat16>::value,
                  "T must be __half or __nv_bfloat16");
    int log2_n = 0;
    while ((1 << log2_n) < cols)
    {
        ++log2_n;
    }
    assert((1 << log2_n) == cols && "cols must be a power of 2");

    // Select by cols and, for the middle sizes, by row count:
    //   2/4/8/16    -> one thread per row (row fits in one thread's registers)
    //   32/64/128   -> scalar warp-shuffle (one warp per row)
    //   256..8192   -> multi-warp-per-row for rows <= 64, otherwise vec
    //   16384/32768 -> chunked multi-warp-per-row
    if (cols < WARP_SIZE) {
        constexpr int THREADS_PER_BLOCK = 256;
        // One warp handles WARP_SIZE / cols rows, so a block handles
        // THREADS_PER_BLOCK / cols rows (cols is 2/4/8/16 here, all divide 256).
        dim3 block(THREADS_PER_BLOCK);
        dim3 grid((rows + THREADS_PER_BLOCK / cols - 1) / (THREADS_PER_BLOCK / cols));
        switch (cols)
        {
        case 2:
            hadamard_kernel_small<T, 2><<<grid, block, 0, stream>>>(input, output, rows);
            break;
        case 4:
            hadamard_kernel_small<T, 4><<<grid, block, 0, stream>>>(input, output, rows);
            break;
        case 8:
            hadamard_kernel_small<T, 8><<<grid, block, 0, stream>>>(input, output, rows);
            break;
        case 16:
            hadamard_kernel_small<T, 16><<<grid, block, 0, stream>>>(input, output, rows);
            break;
        default:
            assert(false && "cols no");
        }
    } else if (cols < 256) {
        constexpr int ROWS_PER_BLOCK = 4;
        constexpr int THREADS_PER_BLOCK = ROWS_PER_BLOCK * WARP_SIZE;
        dim3 block(THREADS_PER_BLOCK);
        dim3 grid((rows + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK);
        switch (cols)
        {
        case 32:
            hadamard_kernel_scalar<T, 32, ROWS_PER_BLOCK><<<grid, block, 0, stream>>>(input, output, rows);
            break;
        case 64:
            hadamard_kernel_scalar<T, 64, ROWS_PER_BLOCK><<<grid, block, 0, stream>>>(input, output, rows);
            break;
        case 128:
            hadamard_kernel_scalar<T, 128, ROWS_PER_BLOCK><<<grid, block, 0, stream>>>(input, output, rows);
            break;
        default:
            assert(false && "cols no");
        }
    } else if (cols <= 8192) {
        if (rows <= 64) {
            dim3 grid(rows);
            switch (cols)
            {
            case 256: {
                constexpr int WARPS_PER_ROW = 1;
                dim3 block(WARPS_PER_ROW * WARP_SIZE);
                hadamard_kernel_multi_warp_per_row<T, 256, WARPS_PER_ROW><<<grid, block, 0, stream>>>(input, output, rows);
                break;
            }
            case 512: {
                constexpr int WARPS_PER_ROW = 2;
                dim3 block(WARPS_PER_ROW * WARP_SIZE);
                hadamard_kernel_multi_warp_per_row<T, 512, WARPS_PER_ROW><<<grid, block, 0, stream>>>(input, output, rows);
                break;
            }
            case 1024:
            case 2048:
            case 4096:
            case 8192: {
                constexpr int WARPS_PER_ROW = 4;
                dim3 block(WARPS_PER_ROW * WARP_SIZE);
                if (cols == 1024)
                    hadamard_kernel_multi_warp_per_row<T, 1024, WARPS_PER_ROW><<<grid, block, 0, stream>>>(input, output, rows);
                else if (cols == 2048)
                    hadamard_kernel_multi_warp_per_row<T, 2048, WARPS_PER_ROW><<<grid, block, 0, stream>>>(input, output, rows);
                else if (cols == 4096)
                    hadamard_kernel_multi_warp_per_row<T, 4096, WARPS_PER_ROW><<<grid, block, 0, stream>>>(input, output, rows);
                else
                    hadamard_kernel_multi_warp_per_row<T, 8192, WARPS_PER_ROW><<<grid, block, 0, stream>>>(input, output, rows);
                break;
            }
            default:
                assert(false && "cols no");
            }
        } else {
            constexpr int ROWS_PER_BLOCK = 4;
            dim3 block(ROWS_PER_BLOCK * WARP_SIZE);
            dim3 grid((rows + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK);
            switch (cols)
            {
            case 256:
                hadamard_kernel_vec<T, 256, ROWS_PER_BLOCK><<<grid, block, 0, stream>>>(input, output, rows);
                break;
            case 512:
                hadamard_kernel_vec<T, 512, ROWS_PER_BLOCK><<<grid, block, 0, stream>>>(input, output, rows);
                break;
            case 1024:
                hadamard_kernel_vec<T, 1024, ROWS_PER_BLOCK><<<grid, block, 0, stream>>>(input, output, rows);
                break;
            case 2048:
                hadamard_kernel_vec<T, 2048, ROWS_PER_BLOCK><<<grid, block, 0, stream>>>(input, output, rows);
                break;
            case 4096:
                hadamard_kernel_vec<T, 4096, ROWS_PER_BLOCK><<<grid, block, 0, stream>>>(input, output, rows);
                break;
            case 8192:
                hadamard_kernel_vec<T, 8192, ROWS_PER_BLOCK><<<grid, block, 0, stream>>>(input, output, rows);
                break;
            default:
                assert(false && "cols no");
            }
        }
    } else {
        // WARPS_PER_ROW grows with cols to keep per-thread register usage <= 64
        // floats (no spills) in the chunked kernel.
        switch (cols)
        {
        case 16384: {
            constexpr int WARPS_PER_ROW = 8;
            constexpr int THREADS_PER_BLOCK = WARPS_PER_ROW * WARP_SIZE;
            dim3 block(THREADS_PER_BLOCK);
            dim3 grid(rows);
            hadamard_kernel_multi_warp_per_row_chunked<T, 16384, WARPS_PER_ROW><<<grid, block, 0, stream>>>(input, output, rows);
            break;
        }
        case 32768: {
            constexpr int WARPS_PER_ROW = 16;
            constexpr int THREADS_PER_BLOCK = WARPS_PER_ROW * WARP_SIZE;
            dim3 block(THREADS_PER_BLOCK);
            dim3 grid(rows);
            hadamard_kernel_multi_warp_per_row_chunked<T, 32768, WARPS_PER_ROW><<<grid, block, 0, stream>>>(input, output, rows);
            break;
        }
        default:
            assert(false && "cols no");
        }
    }
}

// Select the implementation at compile time:
//   1 = hadamard_v1 (scalar warp-shuffle), 2 = hadamard_v2 (vectorized 8-wide),
//   3 = hadamard_v3 (multi-warp-per-row), 4 = hadamard_v4 (auto-select by cols).
// Override with -DHADAMARD_IMPL=N in extra_cuda_cflags.
#ifndef HADAMARD_IMPL
#define HADAMARD_IMPL 2
#endif

template <typename T>
void hadamard(const T *input, T *output, int rows, int cols,
              cudaStream_t stream = 0)
{
#if HADAMARD_IMPL == 1
    hadamard_v1<T>(input, output, rows, cols, stream);
#elif HADAMARD_IMPL == 2
    hadamard_v2<T>(input, output, rows, cols, stream);
// #elif HADAMARD_IMPL == 3
//     hadamard_v3<T>(input, output, rows, cols, stream);
// #elif HADAMARD_IMPL == 4
//     hadamard_v4<T>(input, output, rows, cols, stream);
#else
#error "HADAMARD_IMPL must be 1 (scalar), 2 (vec), 3 (multi-warp), or 4 (auto)"
#endif
}

template void hadamard<__half>(const __half*, __half*, int, int, cudaStream_t);
template void hadamard<__nv_bfloat16>(const __nv_bfloat16*, __nv_bfloat16*, int, int, cudaStream_t);

// Explicit instantiations of each implementation so the host can select the
// impl at runtime (instead of pinning it at compile time via HADAMARD_IMPL).
template void hadamard_v1<__half>(const __half*, __half*, int, int, cudaStream_t);
template void hadamard_v1<__nv_bfloat16>(const __nv_bfloat16*, __nv_bfloat16*, int, int, cudaStream_t);
template void hadamard_v2<__half>(const __half*, __half*, int, int, cudaStream_t);
template void hadamard_v2<__nv_bfloat16>(const __nv_bfloat16*, __nv_bfloat16*, int, int, cudaStream_t);
template void hadamard_v3<__half>(const __half*, __half*, int, int, cudaStream_t);
template void hadamard_v3<__nv_bfloat16>(const __nv_bfloat16*, __nv_bfloat16*, int, int, cudaStream_t);
template void hadamard_v4<__half>(const __half*, __half*, int, int, cudaStream_t);
template void hadamard_v4<__nv_bfloat16>(const __nv_bfloat16*, __nv_bfloat16*, int, int, cudaStream_t);
