#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <cassert>
#include <type_traits>

#include "utils.h"

typedef uint32_t b32;

// Build the H16 matrix as an mma A-fragment for this lane (mirrors the same
// Build the constant fragment directly so the hybrid kernel is self-contained.
template <typename T>
__device__ __forceinline__ void make_h16_a_frag(b32 (&frag)[4], int lane_id)
{
    constexpr b32 p1[4] = {0xFFAACC99u, 0xFFAACC99u, 0xFFAACC99u, 0x00553366u};
    constexpr b32 p2[4] = {0xF0A5C396u, 0xF0A5C396u, 0xF0A5C396u, 0x0F5A3C69u};

    b32 pp, pn, np, nn;
    if constexpr (std::is_same<T, __half>::value) {
        pp = 0x3C003C00u; pn = 0xBC003C00u; np = 0x3C00BC00u; nn = 0xBC00BC00u;
    } else {
        pp = 0x3F803F80u; pn = 0xBF803F80u; np = 0x3F80BF80u; nn = 0xBF80BF80u;
    }

    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        bool lo = (p1[j] >> (31 - lane_id)) & 1u;
        bool hi = (p2[j] >> (31 - lane_id)) & 1u;
        frag[j] = lo ? (hi ? pp : pn) : (hi ? np : nn);
    }
}

// mma.m16n8k16 with the raw f32 accumulator returned (no cvt back to fp16/bf16).
// D(16x8) = A(16x16) @ B(16x8), all four accumulator words kept in f32.
template <typename T>
__device__ __forceinline__ void mma_m16_n8_k16_f32(
    b32 a0, b32 a1, b32 a2, b32 a3,
    b32 b0, b32 b1,
    float &c0, float &c1, float &c2, float &c3)
{
    if constexpr (std::is_same<T, __half>::value) {
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
            : "=f"(c0), "=f"(c1), "=f"(c2), "=f"(c3)
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
              "r"(b0), "r"(b1),
              "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
    } else {
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
            : "=f"(c0), "=f"(c1), "=f"(c2), "=f"(c3)
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
              "r"(b0), "r"(b1),
              "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
    }
}


// Hybrid FWHT: the innermost H16 (bits 0..3) is applied with a single
// tensor-core mma that accumulates in f32, then every higher bit is handled by
// intra-warp __shfl_xor_sync (bits 4..7), a per-thread local loop
// (bits 8..8+log2(NUM_VEC)-1), and a shared-memory cross-warp butterfly (the
// rest). The whole transform stays in f32 until a single final rounding to T,
// so it reproduces fht's single-rounding precision while offloading H16 to the
// tensor cores.
//
// Each warp owns ELEMENT_NUM_PER_WARP = NUM_VEC * 256 elements, laid out as
// NUM_VEC chunks of 256 with 8 elements per lane (lane l holds elements
// [8l, 8l+8) of the chunk).
//
// Dynamic shared memory (flattened) layout: smem[warp_id][vec][k][lane], i.e.
// index ((warp_id*NUM_VEC + vec)*VEC_SIZE + k)*WARP_SIZE + lane. The first
// 256-float slot of each warp doubles as the H16 scatter scratch.
template <typename T, const int COLS, const int WARPS_PER_ROW>
__global__ void hadamard_hybrid_kernel(const T *input, T *output, const int rows)
{
    constexpr int VEC_SIZE = 8;
    constexpr int ELEMENT_NUM_PER_WARP = COLS / WARPS_PER_ROW;
    constexpr int NUM_VEC = ELEMENT_NUM_PER_WARP / 256;
    static_assert(ELEMENT_NUM_PER_WARP % 256 == 0,
                  "each warp must own whole 256-element chunks");

    extern __shared__ float smem[];

    const int row = blockIdx.x;
    if (row >= rows)
        return;
    const int tid = threadIdx.x;
    const int lane_id = tid % WARP_SIZE;
    const int warp_id = tid / WARP_SIZE;
    const int col_offset = warp_id * ELEMENT_NUM_PER_WARP;

    float local[NUM_VEC][VEC_SIZE];

    b32 h_frag[4];
    make_h16_a_frag<T>(h_frag, lane_id);

    // Tensor-core H16 + intra-warp shuffle (bits 4..7) for each 256-chunk.
    #pragma unroll
    for (int i = 0; i < NUM_VEC; ++i) {
        const b32 *in32 = reinterpret_cast<const b32 *>(
            input + row * COLS + col_offset + i * 256);

        const int g = lane_id >> 2;   // group 0..7
        const int t = lane_id & 3;    // thread-in-group 0..3

        // B-fragments for a 16x16 tile B[k][n] = input[16*n + k]: n is the
        // 16-element group index, k the position within the group.
        b32 x0 = in32[8 * g + t];           // groups 0..7,  rows 2t, 2t+1
        b32 x1 = in32[8 * g + t + 4];       // groups 0..7,  rows 2t+8, 2t+9
        b32 x2 = in32[8 * (g + 8) + t];     // groups 8..15, rows 2t, 2t+1
        b32 x3 = in32[8 * (g + 8) + t + 4]; // groups 8..15, rows 2t+8, 2t+9

        float c[8];
        mma_m16_n8_k16_f32<T>(h_frag[0], h_frag[1], h_frag[2], h_frag[3],
                              x0, x1, c[0], c[1], c[2], c[3]);
        mma_m16_n8_k16_f32<T>(h_frag[0], h_frag[1], h_frag[2], h_frag[3],
                              x2, x3, c[4], c[5], c[6], c[7]);

        // D[m][n] = (H16 applied to group n)[m]. Scatter into a scratch buffer
        // at linear position e = 16*n + m, then read back in the lane-major
        // layout used by the shuffle stages.
        float *buf = &smem[warp_id * NUM_VEC * 256];
        buf[16 * (2 * t) + g] = c[0];
        buf[16 * (2 * t + 1) + g] = c[1];
        buf[16 * (2 * t) + g + 8] = c[2];
        buf[16 * (2 * t + 1) + g + 8] = c[3];
        buf[16 * (8 + 2 * t) + g] = c[4];
        buf[16 * (8 + 2 * t + 1) + g] = c[5];
        buf[16 * (8 + 2 * t) + g + 8] = c[6];
        buf[16 * (8 + 2 * t + 1) + g + 8] = c[7];

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < VEC_SIZE; ++k)
            local[i][k] = buf[lane_id * VEC_SIZE + k];

        // Bits 4..7 (element strides 16, 32, 64, 128) are lane strides 2, 4, 8,
        // 16 in the lane-major layout. Bit 3 (stride 8) was already done by H16.
        #pragma unroll
        for (int j = 2; j < WARP_SIZE; j <<= 1) {
            #pragma unroll
            for (int k = 0; k < VEC_SIZE; ++k) {
                float other = __shfl_xor_sync(0xffffffff, local[i][k], j);
                if (lane_id & j)
                    local[i][k] = other - local[i][k];
                else
                    local[i][k] = local[i][k] + other;
            }
        }

        // Keep the scratch buffer safe for the next chunk's write.
        __syncthreads();
    }

    // Bits 8..8+log2(NUM_VEC)-1: across the NUM_VEC chunks held by this thread.
    #pragma unroll
    for (int i = 1; i < NUM_VEC; i <<= 1) {
        #pragma unroll
        for (int j = 0; j < NUM_VEC; ++j) {
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

    // Remaining bits: cross-warp butterfly through shared memory.
    #pragma unroll
    for (int i = 0; i < NUM_VEC; ++i)
        #pragma unroll
        for (int k = 0; k < VEC_SIZE; ++k)
            smem[((warp_id * NUM_VEC + i) * VEC_SIZE + k) * WARP_SIZE + lane_id] =
                local[i][k];

    __syncthreads();

    #pragma unroll
    for (int i = 1; i < WARPS_PER_ROW; i <<= 1) {
        #pragma unroll
        for (int j = 0; j < NUM_VEC; ++j) {
            #pragma unroll
            for (int k = 0; k < VEC_SIZE; ++k) {
                float other =
                    smem[(((warp_id ^ i) * NUM_VEC + j) * VEC_SIZE + k) * WARP_SIZE + lane_id];
                if (warp_id & i)
                    local[j][k] = other - local[j][k];
                else
                    local[j][k] = local[j][k] + other;
            }
        }
        __syncthreads();
        #pragma unroll
        for (int j = 0; j < NUM_VEC; ++j)
            #pragma unroll
            for (int k = 0; k < VEC_SIZE; ++k)
                smem[((warp_id * NUM_VEC + j) * VEC_SIZE + k) * WARP_SIZE + lane_id] =
                    local[j][k];
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < NUM_VEC; ++i)
        #pragma unroll
        for (int k = 0; k < VEC_SIZE; ++k)
            output[row * COLS + col_offset + i * 256 + lane_id * VEC_SIZE + k] =
                T(local[i][k]);
}


template <typename T, const int COLS, const int WARPS_PER_ROW>
void launch_hybrid(const T *input, T *output, int rows, cudaStream_t stream)
{
    constexpr int ELEMENT_NUM_PER_WARP = COLS / WARPS_PER_ROW;
    constexpr int NUM_VEC = ELEMENT_NUM_PER_WARP / 256;
    constexpr int SMEM = WARPS_PER_ROW * NUM_VEC * 256 * (int)sizeof(float);

    if constexpr (SMEM > 48 * 1024)
        cudaFuncSetAttribute(hadamard_hybrid_kernel<T, COLS, WARPS_PER_ROW>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);

    dim3 grid(rows), block(WARPS_PER_ROW * WARP_SIZE);
    hadamard_hybrid_kernel<T, COLS, WARPS_PER_ROW><<<grid, block, SMEM, stream>>>(
        input, output, rows);
}

template <typename T>
void hadamard_hybrid(const T *input, T *output, int rows, int cols,
                     cudaStream_t stream = 0)
{
    static_assert(std::is_same<T, __half>::value ||
                      std::is_same<T, __nv_bfloat16>::value,
                  "T must be __half or __nv_bfloat16");

    int log_cols = 0;
    while ((1 << log_cols) < cols)
        ++log_cols;
    assert((1 << log_cols) == cols && "cols must be a power of 2");
    assert(cols >= 256 && cols <= 32768 && "cols must be in [256, 32768]");

    switch (cols) {
        case 256:   launch_hybrid<T, 256, 1>(input, output, rows, stream); break;
        case 512:   launch_hybrid<T, 512, 2>(input, output, rows, stream); break;
        case 1024:  launch_hybrid<T, 1024, 4>(input, output, rows, stream); break;
        case 2048:  launch_hybrid<T, 2048, 8>(input, output, rows, stream); break;
        case 4096:  launch_hybrid<T, 4096, 16>(input, output, rows, stream); break;
        case 8192:  launch_hybrid<T, 8192, 32>(input, output, rows, stream); break;
        case 16384: launch_hybrid<T, 16384, 16>(input, output, rows, stream); break;
        case 32768: launch_hybrid<T, 32768, 16>(input, output, rows, stream); break;
        default:
            assert(false && "cols must be a power of 2 in [256, 32768]");
    }
}

template void hadamard_hybrid<__half>(const __half*, __half*, int, int, cudaStream_t);
template void hadamard_hybrid<__nv_bfloat16>(const __nv_bfloat16*, __nv_bfloat16*, int, int, cudaStream_t);
