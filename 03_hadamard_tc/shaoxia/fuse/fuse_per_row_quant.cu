#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include "fuse_per_row_quant.cuh"

#include <cassert>
#include <cstdint>
#include <type_traits>

#define WARP_SIZE 32

// ============================================================================
// Fused Hadamard transform + per-row quantization  (template)
// ============================================================================
//
//   input  : fp16 / bf16, shape [rows, cols]   (cols must be a power of 2)
//   output : int8 / fp8 E4M3, shape [rows, cols] (one byte per element), or
//            packed int4, shape [rows, cols / 2]
//   scale  : fp32,        shape [rows]         (per-row scale)
//
//   Per row r:
//       y        = FWHT(x[r])                  // natural-order Hadamard transform
//       scale[r] = absmax(y) / qmax            // qmax = 127 or 448
//       out[r][c] = round(y[c] / scale[r])     // clamped to int8
//
// The FWHT body and launch policy mirror hadamard_v4. Before quantization, the
// FP32 butterfly result is rounded once to the input storage type so this fused
// path is byte-for-byte equivalent to hadamard_v4 followed by per-row INT8
// quantization.
//
// Python side (torch.utils.cpp_extension.load_inline) needs a binding like:
//   void fuse_hadamard_per_row_quant_fp16(torch::Tensor x, torch::Tensor out, torch::Tensor scale) {
//       fused_hadamard_per_row_quant<__half>(
//           reinterpret_cast<const __half*>(x.data_ptr()),
//           reinterpret_cast<int8_t*>(out.data_ptr()),
//           scale.data_ptr<float>(),
//           static_cast<int>(x.size(0)), static_cast<int>(x.size(1)),
//           at::cuda::getCurrentCUDAStream().stream());
//   }
// (plus the __nv_bfloat16 twin), and m.def(...) for each.


// ---------- fp16 / bf16 -> float8 load helpers -------------------------------
struct alignas(16) Half8
{
    __half x[8];
};

struct alignas(16) BFloat16_8
{
    __nv_bfloat16 x[8];
};

static_assert(sizeof(Half8) == 16);
static_assert(sizeof(BFloat16_8) == 16);

__device__ __forceinline__ void load_8_to_float8(
    const __half *ptr,
    float (&v)[8])
{
    Half8 h = *reinterpret_cast<const Half8 *>(ptr);
#pragma unroll
    for (int i = 0; i < 8; ++i)
    {
        v[i] = __half2float(h.x[i]);
    }
}

// Match the unfused path exactly: hadamard_v4 stores its FP32 butterfly result
// as T before a standalone quantizer reads it back.  The fused kernel has no
// intermediate tensor, so explicitly perform the same storage-type rounding.
template <typename T>
__device__ __forceinline__ float round_to_storage(float v);

template <>
__device__ __forceinline__ float round_to_storage<__half>(float v)
{
    return __half2float(__float2half_rn(v));
}

template <>
__device__ __forceinline__ float round_to_storage<__nv_bfloat16>(float v)
{
    return __bfloat162float(__float2bfloat16_rn(v));
}

__device__ __forceinline__ void load_8_to_float8(
    const __nv_bfloat16 *ptr,
    float (&v)[8])
{
    BFloat16_8 h = *reinterpret_cast<const BFloat16_8 *>(ptr);
#pragma unroll
    for (int i = 0; i < 8; ++i)
    {
        v[i] = __bfloat162float(h.x[i]);
    }
}



template <FuseQuantScheme S>
__device__ __forceinline__ uint8_t quantize_int8(
    float v, float inv_scale, uint8_t zero_point)
{
    float q = rintf(v * inv_scale);
    if constexpr (S == FuseQuantScheme::Asymmetric) q += float(zero_point);
    if constexpr (S == FuseQuantScheme::Asymmetric) {
        q = fminf(255.0f, fmaxf(0.0f, q));
        return static_cast<uint8_t>(q);
    } else {
        q = fminf(127.0f, fmaxf(-127.0f, q));
        return static_cast<uint8_t>(static_cast<int8_t>(q));
    }
}

template <FuseQuantScheme S>
__device__ __forceinline__ uint8_t quantize_int4(
    float v, float inv_scale, uint8_t zero_point)
{
    float q = rintf(v * inv_scale);
    if constexpr (S == FuseQuantScheme::Asymmetric) {
        q = fminf(15.0f, fmaxf(0.0f, q + float(zero_point)));
    } else {
        q = fminf(7.0f, fmaxf(-7.0f, q));
    }
    return static_cast<uint8_t>(static_cast<int>(q)) & 0x0f;
}

__device__ __forceinline__ uint8_t quantize_fp8(
    float v, float inv_scale)
{
    return __nv_cvt_float_to_fp8(
        v * inv_scale, __NV_SATFINITE, __NV_E4M3);
}

template <FuseQuantScheme S>
__device__ __forceinline__ void quantize_store_int8x8(
    const float (&v)[8], float inv_scale, uint8_t *ptr, uint8_t zero_point)
{
    uint8_t out[8];
#pragma unroll
    for (int i = 0; i < 8; ++i)
        out[i] = quantize_int8<S>(v[i], inv_scale, zero_point);
    *reinterpret_cast<unsigned long long *>(ptr) =
        *reinterpret_cast<const unsigned long long *>(out);
}

__device__ __forceinline__ void quantize_store_fp8x8(
    const float (&v)[8], float inv_scale, uint8_t *ptr)
{
    __nv_fp8x2_storage_t out[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float2 pair = make_float2(
            v[2 * i] * inv_scale, v[2 * i + 1] * inv_scale);
        out[i] = __nv_cvt_float2_to_fp8x2(
            pair, __NV_SATFINITE, __NV_E4M3);
    }
    *reinterpret_cast<unsigned long long *>(ptr) =
        *reinterpret_cast<const unsigned long long *>(out);
}

template <FuseQuantScheme S>
__device__ __forceinline__ void quantize_store_int4x8(
    const float (&v)[8], float inv_scale, uint8_t *ptr, uint8_t zero_point)
{
    uint32_t packed = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const uint8_t lo = quantize_int4<S>(v[2 * i], inv_scale, zero_point);
        const uint8_t hi = quantize_int4<S>(v[2 * i + 1], inv_scale, zero_point);
        packed |= uint32_t(lo | (hi << 4)) << (8 * i);
    }
    *reinterpret_cast<uint32_t *>(ptr) = packed;
}

template <FuseQuantType Q>
__device__ __forceinline__ constexpr float symmetric_qmax()
{
    if constexpr (Q == FuseQuantType::Fp8E4M3) return 448.0f;
    if constexpr (Q == FuseQuantType::Int4) return 7.0f;
    return 127.0f;
}

__device__ __forceinline__ float warp_reduce_absmax(float v)
{
    v = fabsf(v);
#pragma unroll
    for (int j = WARP_SIZE / 2; j >= 1; j >>= 1)
    {
        v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, j));
    }
    return v;
}

__device__ __forceinline__ float warp_reduce_min(float v)
{
#pragma unroll
    for (int j = WARP_SIZE / 2; j >= 1; j >>= 1)
        v = fminf(v, __shfl_xor_sync(0xffffffff, v, j));
    return v;
}

__device__ __forceinline__ float warp_reduce_max(float v)
{
#pragma unroll
    for (int j = WARP_SIZE / 2; j >= 1; j >>= 1)
        v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, j));
    return v;
}

template <FuseQuantType Q, FuseQuantScheme S>
__device__ __forceinline__ void make_quant_params(
    float amax, float row_min, float row_max,
    float &scale, float &inv_scale, uint8_t &zero_point)
{
    if constexpr (S != FuseQuantScheme::Asymmetric) {
        if (amax > 0.0f) {
            constexpr float qmax = symmetric_qmax<Q>();
            scale = amax / qmax;
            inv_scale = qmax / amax;
        } else {
            scale = inv_scale = 1.0f;
        }
        zero_point = 0;
        return;
    }
    constexpr float qmax = Q == FuseQuantType::Int4 ? 15.0f : 255.0f;
    const float range = row_max - row_min;
    if (range > 0.0f) {
        scale = range / qmax;
        inv_scale = qmax / range;
    } else {
        scale = inv_scale = 1.0f;
    }
    zero_point = static_cast<uint8_t>(fminf(
        qmax, fmaxf(0.0f, rintf(-row_min * inv_scale))));
}


// Flattened index into the 1-D shared-memory exchange buffer. Mirrors the old
// 4-D layout smem[warp][vec][k][lane] (VEC_SIZE is fixed at 8). `vec_per_warp`
// is NUM_VEC (or CHUNKS_PER_ROUND when the exchange is batched in rounds). After
// the FWHT, the first WARPS_PER_ROW entries are reused for the per-warp absmax
// reduction, so one buffer serves both phases.
__device__ __forceinline__ int exchange_smem_idx(int warp_id, int vec, int k, int lane_id, int vec_per_warp)
{
    return (((warp_id * vec_per_warp + vec) * 8 + k) * WARP_SIZE + lane_id);
}


// One warp handles WARP_SIZE / COLS rows (COLS = 2/4/8/16); each row uses COLS
// lanes (one element per lane), so the FWHT butterfly is done with
// __shfl_xor_sync, followed by a per-row absmax scale and int8 quantize.
// Mirrors hadamard_kernel_small + per-row quant.
template <typename T, const int COLS, FuseQuantType Q, FuseQuantScheme S>
__global__ void fuse_hadamard_per_row_quant_kernel_small(
    const T *__restrict__ input,
    uint8_t *__restrict__ output,
    float *__restrict__ output_scale,
    uint8_t *__restrict__ output_zero_points,
    const int rows)
{
    static_assert(COLS < WARP_SIZE && WARP_SIZE % COLS == 0,
                  "COLS must be a power of 2 dividing WARP_SIZE and < WARP_SIZE");
    constexpr int ROWS_PER_WARP = WARP_SIZE / COLS;

    const int tid = threadIdx.x;
    const int lane_id = tid % WARP_SIZE;
    const int warp_id = tid / WARP_SIZE;
    const int lane_in_row = lane_id % COLS;
    const int row = (blockIdx.x * (blockDim.x / WARP_SIZE) + warp_id) * ROWS_PER_WARP
                    + lane_id / COLS;

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

    local = round_to_storage<T>(local);

    // Reduce within each COLS-lane row group.
    float amax = fabsf(local);
    float row_min = 0.0f;
    float row_max = 0.0f;
    if constexpr (S == FuseQuantScheme::Asymmetric)
        row_min = row_max = local;
    #pragma unroll
    for (int j = COLS >> 1; j >= 1; j >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, j));
        if constexpr (S == FuseQuantScheme::Asymmetric) {
            row_min = fminf(row_min, __shfl_xor_sync(0xffffffff, row_min, j));
            row_max = fmaxf(row_max, __shfl_xor_sync(0xffffffff, row_max, j));
        }
    }
    float scale;
    float inv_scale;
    uint8_t zero_point;
    make_quant_params<Q, S>(
        amax, row_min, row_max, scale, inv_scale, zero_point);

    if (active) {
        if constexpr (Q == FuseQuantType::Int4) {
            const uint8_t q = quantize_int4<S>(local, inv_scale, zero_point);
            const uint8_t neighbor = __shfl_down_sync(0xffffffff, q, 1);
            if ((lane_in_row & 1) == 0)
                output[row * (COLS / 2) + lane_in_row / 2] =
                    q | (neighbor << 4);
        } else {
            if constexpr (Q == FuseQuantType::Fp8E4M3)
                output[row * COLS + lane_in_row] =
                    quantize_fp8(local, inv_scale);
            else
                output[row * COLS + lane_in_row] =
                    quantize_int8<S>(local, inv_scale, zero_point);
        }
        if (lane_in_row == 0) {
            output_scale[row] = scale;
            if (output_zero_points) output_zero_points[row] = zero_point;
        }
    }
}


template <
    typename T,
    const int COLS,
    const int ROWS_PER_BLOCK,
    FuseQuantType Q,
    FuseQuantScheme S
>
__global__ void fuse_hadamard_per_row_quant_kernel_1warp1row_scalar(
    const T *__restrict__ input,
    uint8_t *__restrict__ output,
    float *__restrict__ output_scale,
    uint8_t *__restrict__ output_zero_points,
    const int rows)
{
    static_assert(COLS >= WARP_SIZE, "COLS must be >= WARP_SIZE");
    constexpr int NUM_PER_THREAD = COLS / WARP_SIZE;
    float local[NUM_PER_THREAD];

    const int tid = threadIdx.x;
    const int lane_id = tid % WARP_SIZE;
    const int warp_id = tid / WARP_SIZE;
    const int row = blockIdx.x * ROWS_PER_BLOCK + warp_id;

    if (row >= rows)
    {
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
    for (int i = 0; i < NUM_PER_THREAD; ++i)
        local[i] = round_to_storage<T>(local[i]);

    float amax = -1.0f;
    float row_min = 0.0f;
    float row_max = 0.0f;
    if constexpr (S == FuseQuantScheme::Asymmetric)
        row_min = row_max = local[0];
    #pragma unroll
    for (int i = 0; i < NUM_PER_THREAD; ++i){
        amax = fmaxf(fabsf(local[i]), amax);
        if constexpr (S == FuseQuantScheme::Asymmetric) {
            row_min = fminf(row_min, local[i]);
            row_max = fmaxf(row_max, local[i]);
        }
    }
    amax = warp_reduce_absmax(amax);
    if constexpr (S == FuseQuantScheme::Asymmetric) {
        row_min = warp_reduce_min(row_min);
        row_max = warp_reduce_max(row_max);
    }
    float scale;
    float inv_scale;
    uint8_t zero_point;
    make_quant_params<Q, S>(
        amax, row_min, row_max, scale, inv_scale, zero_point);

    #pragma unroll
    for (int i = 0; i < NUM_PER_THREAD; ++i) {
        uint8_t q;
        if constexpr (Q == FuseQuantType::Int4)
            q = quantize_int4<S>(local[i], inv_scale, zero_point);
        else if constexpr (Q == FuseQuantType::Fp8E4M3)
            q = quantize_fp8(local[i], inv_scale);
        else
            q = quantize_int8<S>(local[i], inv_scale, zero_point);
        if constexpr (Q == FuseQuantType::Int4) {
            const uint8_t neighbor = __shfl_down_sync(0xffffffff, q, 1);
            if ((lane_id & 1) == 0)
                output[row * (COLS / 2) + i * 16 + lane_id / 2] =
                    q | (neighbor << 4);
        } else {
            output[row * COLS + lane_id + i * WARP_SIZE] = q;
        }
    }

    if (lane_id == 0)
    {
        output_scale[row] = scale;
        if (output_zero_points) output_zero_points[row] = zero_point;
    }
}

template <
    typename T,
    const int COLS,
    const int ROWS_PER_BLOCK,
    FuseQuantType Q,
    FuseQuantScheme S>
__global__ void fuse_hadamard_per_row_quant_kernel_1warp1row_vec(
    const T *__restrict__ input,
    uint8_t *__restrict__ output,
    float *__restrict__ output_scale,
    uint8_t *__restrict__ output_zero_points,
    const int rows)
{
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
    for (int i = 0; i < VEC_PER_THREAD; ++i)
        #pragma unroll
        for (int j = 0; j < VEC_SIZE; ++j)
            local[i][j] = round_to_storage<T>(local[i][j]);

    float amax = -1.0f;
    float row_min = 0.0f;
    float row_max = 0.0f;
    if constexpr (S == FuseQuantScheme::Asymmetric)
        row_min = row_max = local[0][0];
    #pragma unroll
    for (int i = 0; i < VEC_PER_THREAD; ++i)
        #pragma unroll
        for (int j = 0; j < VEC_SIZE; ++j)
        {
            amax = fmaxf(fabsf(local[i][j]), amax);
            if constexpr (S == FuseQuantScheme::Asymmetric) {
                row_min = fminf(row_min, local[i][j]);
                row_max = fmaxf(row_max, local[i][j]);
            }
        }
    
    amax = warp_reduce_absmax(amax);
    if constexpr (S == FuseQuantScheme::Asymmetric) {
        row_min = warp_reduce_min(row_min);
        row_max = warp_reduce_max(row_max);
    }
    float scale;
    float inv_scale;
    uint8_t zero_point;
    make_quant_params<Q, S>(
        amax, row_min, row_max, scale, inv_scale, zero_point);

    #pragma unroll
    for (int i = 0; i < VEC_PER_THREAD; ++i)
    {
        uint8_t *out_ptr = &output[(Q == FuseQuantType::Int4
                         ? row * (COLS / 2) +
                               (lane_id * VEC_SIZE + i * WARP_SIZE * VEC_SIZE) / 2
                         : row * COLS + lane_id * VEC_SIZE +
                               i * WARP_SIZE * VEC_SIZE)];
        if constexpr (Q == FuseQuantType::Int8)
            quantize_store_int8x8<S>(local[i], inv_scale, out_ptr, zero_point);
        else if constexpr (Q == FuseQuantType::Fp8E4M3)
            quantize_store_fp8x8(local[i], inv_scale, out_ptr);
        else
            quantize_store_int4x8<S>(local[i], inv_scale, out_ptr, zero_point);
    }

    if (lane_id == 0)
    {
        output_scale[row] = scale;
        if (output_zero_points) output_zero_points[row] = zero_point;
    }

}

template <typename T, const int COLS, const int WARPS_PER_ROW,
          FuseQuantType Q, FuseQuantScheme S>
__global__ void fuse_hadamard_per_row_quant_kernel_multi_warp_per_row(
    const T *__restrict__ input,
    uint8_t *__restrict__ output,
    float *__restrict__ output_scale,
    uint8_t *__restrict__ output_zero_points,
    const int rows)
{
    constexpr int VEC_SIZE = 8;
    constexpr int ELEMENT_NUM_PER_WARP = COLS / WARPS_PER_ROW;
    constexpr int NUM_VEC = ELEMENT_NUM_PER_WARP / WARP_SIZE / VEC_SIZE;
    // 1-D shared memory: cross-warp FWHT exchange first, then reused (its first
    // WARPS_PER_ROW entries) for the per-warp absmax reduction.
    constexpr int SMEM_FLOATS = WARPS_PER_ROW * NUM_VEC * VEC_SIZE * WARP_SIZE;
    __shared__ float smem[SMEM_FLOATS];

    const int row = blockIdx.x;
    if (row >= rows)
        return;
    const int tid = threadIdx.x;
    const int lane_id = tid % WARP_SIZE;
    const int warp_id = tid / WARP_SIZE;
    const int col_offset = warp_id * ELEMENT_NUM_PER_WARP;

    float local[NUM_VEC][VEC_SIZE];

    // FWHT: intra-vec -> intra-warp -> intra-thread -> cross-warp (smem).
    #pragma unroll
    for (int i = 0; i < NUM_VEC; ++i) {
        load_8_to_float8(&input[row * COLS + col_offset + i * WARP_SIZE * VEC_SIZE + lane_id * VEC_SIZE], local[i]);
        #pragma unroll
        for (int j = 1; j < VEC_SIZE; j <<= 1) {
            #pragma unroll
            for (int k = 0; k < VEC_SIZE; ++k) {
                if (k & j) {
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
                } else {
                    local[i][k] = local[i][k] + other;
                }
            }
        }
    }

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

    for (int i = 0; i < NUM_VEC; ++i)
        for (int k = 0; k < VEC_SIZE; ++k)
            smem[exchange_smem_idx(warp_id, i, k, lane_id, NUM_VEC)] = local[i][k];
    __syncthreads();

    #pragma unroll
    for (int i = 1; i < WARPS_PER_ROW; i <<= 1) {
        #pragma unroll
        for (int j = 0; j < NUM_VEC; ++j) {
            #pragma unroll
            for (int k = 0; k < VEC_SIZE; ++k) {
                float other = smem[exchange_smem_idx(warp_id ^ i, j, k, lane_id, NUM_VEC)];
                if (warp_id & i) {
                    local[j][k] = other - local[j][k];
                } else {
                    local[j][k] = local[j][k] + other;
                }
            }
        }
        __syncthreads();
        #pragma unroll
        for (int j = 0; j < NUM_VEC; ++j) {
            #pragma unroll
            for (int k = 0; k < VEC_SIZE; ++k) {
                smem[exchange_smem_idx(warp_id, j, k, lane_id, NUM_VEC)] = local[j][k];
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < NUM_VEC; ++i)
        #pragma unroll
        for (int k = 0; k < VEC_SIZE; ++k)
            local[i][k] = round_to_storage<T>(local[i][k]);

    // Per-row scale: warp absmax, then cross-warp max via the now-free smem prefix.
    float amax = -1.0f;
    float row_min = 0.0f;
    float row_max = 0.0f;
    if constexpr (S == FuseQuantScheme::Asymmetric)
        row_min = row_max = local[0][0];
    #pragma unroll
    for (int i = 0; i < NUM_VEC; ++i)
        #pragma unroll
        for (int k = 0; k < VEC_SIZE; ++k)
        {
            amax = fmaxf(fabsf(local[i][k]), amax);
            if constexpr (S == FuseQuantScheme::Asymmetric) {
                row_min = fminf(row_min, local[i][k]);
                row_max = fmaxf(row_max, local[i][k]);
            }
        }
    amax = warp_reduce_absmax(amax);
    if constexpr (S == FuseQuantScheme::Asymmetric) {
        row_min = warp_reduce_min(row_min);
        row_max = warp_reduce_max(row_max);
    }

    if (lane_id == 0) {
        smem[warp_id] = amax;
        if constexpr (S == FuseQuantScheme::Asymmetric) {
            smem[WARPS_PER_ROW + warp_id] = row_min;
            smem[2 * WARPS_PER_ROW + warp_id] = row_max;
        }
    }
    __syncthreads();

    float scale;
    if (warp_id == 0) {
        // 0.0f sentinel: values are already non-negative absmax, so a plain max
        // (warp_reduce_absmax's fabsf is a no-op) keeps invalid lanes from
        // polluting the result.
        float w_amax = (lane_id < WARPS_PER_ROW) ? smem[lane_id] : 0.0f;
        float w_min = 0.0f;
        float w_max = 0.0f;
        if constexpr (S == FuseQuantScheme::Asymmetric) {
            w_min = (lane_id < WARPS_PER_ROW)
                        ? smem[WARPS_PER_ROW + lane_id] : __int_as_float(0x7f800000);
            w_max = (lane_id < WARPS_PER_ROW)
                        ? smem[2 * WARPS_PER_ROW + lane_id] : -__int_as_float(0x7f800000);
        }
        w_amax = warp_reduce_absmax(w_amax);
        if constexpr (S == FuseQuantScheme::Asymmetric) {
            w_min = warp_reduce_min(w_min);
            w_max = warp_reduce_max(w_max);
        }
        if (lane_id == 0) {
            uint8_t zp;
            make_quant_params<Q, S>(
                w_amax, w_min, w_max, smem[0], smem[2], zp);
            smem[1] = float(zp);
        }
    }
    __syncthreads();
    scale = smem[0];
    const float inv_scale = smem[2];
    const uint8_t zero_point = static_cast<uint8_t>(smem[1]);

    #pragma unroll
    for (int i = 0; i < NUM_VEC; ++i) {
        uint8_t *out_ptr = &output[(Q == FuseQuantType::Int4
                         ? row * (COLS / 2) +
                               (col_offset + i * WARP_SIZE * VEC_SIZE +
                                lane_id * VEC_SIZE) / 2
                         : row * COLS + col_offset +
                               i * WARP_SIZE * VEC_SIZE + lane_id * VEC_SIZE)];
        if constexpr (Q == FuseQuantType::Int8)
            quantize_store_int8x8<S>(local[i], inv_scale, out_ptr, zero_point);
        else if constexpr (Q == FuseQuantType::Fp8E4M3)
            quantize_store_fp8x8(local[i], inv_scale, out_ptr);
        else
            quantize_store_int4x8<S>(local[i], inv_scale, out_ptr, zero_point);
    }

    if (tid == 0) {
        output_scale[row] = scale;
        if (output_zero_points) output_zero_points[row] = zero_point;
    }
}


template <typename T, const int COLS, const int WARPS_PER_ROW,
          FuseQuantType Q, FuseQuantScheme S>
__global__ void fuse_hadamard_per_row_quant_kernel_multi_warp_per_row_chunked(
    const T *__restrict__ input,
    uint8_t *__restrict__ output,
    float *__restrict__ output_scale,
    uint8_t *__restrict__ output_zero_points,
    const int rows)
{
    constexpr int VEC_SIZE = 8;
    constexpr int ELEMENT_NUM_PER_WARP = COLS / WARPS_PER_ROW;
    constexpr int NUM_VEC = ELEMENT_NUM_PER_WARP / WARP_SIZE / VEC_SIZE;
    // Fixed 32 KB of shared memory: the cross-warp exchange is batched in rounds
    // of CHUNKS_PER_ROUND chunks; after the FWHT the buffer prefix is reused for
    // the per-warp absmax reduction.
    constexpr int SMEM_BYTES = 32 * 1024;
    constexpr int CHUNKS_PER_ROUND = SMEM_BYTES / (WARPS_PER_ROW * VEC_SIZE * WARP_SIZE * (int)sizeof(float));
    constexpr int NUM_ROUNDS = NUM_VEC / CHUNKS_PER_ROUND;
    static_assert(CHUNKS_PER_ROUND >= 1, "CHUNKS_PER_ROUND must be >= 1");
    static_assert(NUM_VEC % CHUNKS_PER_ROUND == 0, "NUM_VEC must be divisible by CHUNKS_PER_ROUND");
    constexpr int SMEM_FLOATS = WARPS_PER_ROW * CHUNKS_PER_ROUND * VEC_SIZE * WARP_SIZE;
    __shared__ float smem[SMEM_FLOATS];

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
        for (int j = 1; j < VEC_SIZE; j <<= 1) {
            #pragma unroll
            for (int k = 0; k < VEC_SIZE; ++k) {
                if (k & j) {
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
                } else {
                    local[i][k] = local[i][k] + other;
                }
            }
        }
    }

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

    // Cross-warp exchange, batched so smem stays fixed at 32 KB.
    #pragma unroll
    for (int r = 0; r < NUM_ROUNDS; ++r) {
        const int base = r * CHUNKS_PER_ROUND;
        #pragma unroll
        for (int c = 0; c < CHUNKS_PER_ROUND; ++c)
            #pragma unroll
            for (int k = 0; k < VEC_SIZE; ++k)
                smem[exchange_smem_idx(warp_id, c, k, lane_id, CHUNKS_PER_ROUND)] = local[base + c][k];
        __syncthreads();

        #pragma unroll
        for (int i = 1; i < WARPS_PER_ROW; i <<= 1) {
            #pragma unroll
            for (int c = 0; c < CHUNKS_PER_ROUND; ++c) {
                #pragma unroll
                for (int k = 0; k < VEC_SIZE; ++k) {
                    float other = smem[exchange_smem_idx(warp_id ^ i, c, k, lane_id, CHUNKS_PER_ROUND)];
                    if (warp_id & i) {
                        local[base + c][k] = other - local[base + c][k];
                    } else {
                        local[base + c][k] = local[base + c][k] + other;
                    }
                }
            }
            __syncthreads();
            #pragma unroll
            for (int c = 0; c < CHUNKS_PER_ROUND; ++c) {
                #pragma unroll
                for (int k = 0; k < VEC_SIZE; ++k) {
                    smem[exchange_smem_idx(warp_id, c, k, lane_id, CHUNKS_PER_ROUND)] = local[base + c][k];
                }
            }
            __syncthreads();
        }
    }

    #pragma unroll
    for (int i = 0; i < NUM_VEC; ++i)
        #pragma unroll
        for (int k = 0; k < VEC_SIZE; ++k)
            local[i][k] = round_to_storage<T>(local[i][k]);

    // Per-row scale (same as the non-chunked multi-warp kernel).
    float amax = -1.0f;
    float row_min = 0.0f;
    float row_max = 0.0f;
    if constexpr (S == FuseQuantScheme::Asymmetric)
        row_min = row_max = local[0][0];
    #pragma unroll
    for (int i = 0; i < NUM_VEC; ++i)
        #pragma unroll
        for (int k = 0; k < VEC_SIZE; ++k)
        {
            amax = fmaxf(fabsf(local[i][k]), amax);
            if constexpr (S == FuseQuantScheme::Asymmetric) {
                row_min = fminf(row_min, local[i][k]);
                row_max = fmaxf(row_max, local[i][k]);
            }
        }
    amax = warp_reduce_absmax(amax);
    if constexpr (S == FuseQuantScheme::Asymmetric) {
        row_min = warp_reduce_min(row_min);
        row_max = warp_reduce_max(row_max);
    }

    if (lane_id == 0) {
        smem[warp_id] = amax;
        if constexpr (S == FuseQuantScheme::Asymmetric) {
            smem[WARPS_PER_ROW + warp_id] = row_min;
            smem[2 * WARPS_PER_ROW + warp_id] = row_max;
        }
    }
    __syncthreads();

    float scale;
    if (warp_id == 0) {
        float w_amax = (lane_id < WARPS_PER_ROW) ? smem[lane_id] : 0.0f;
        float w_min = 0.0f;
        float w_max = 0.0f;
        if constexpr (S == FuseQuantScheme::Asymmetric) {
            w_min = (lane_id < WARPS_PER_ROW)
                        ? smem[WARPS_PER_ROW + lane_id] : __int_as_float(0x7f800000);
            w_max = (lane_id < WARPS_PER_ROW)
                        ? smem[2 * WARPS_PER_ROW + lane_id] : -__int_as_float(0x7f800000);
        }
        w_amax = warp_reduce_absmax(w_amax);
        if constexpr (S == FuseQuantScheme::Asymmetric) {
            w_min = warp_reduce_min(w_min);
            w_max = warp_reduce_max(w_max);
        }
        if (lane_id == 0) {
            uint8_t zp;
            make_quant_params<Q, S>(
                w_amax, w_min, w_max, smem[0], smem[2], zp);
            smem[1] = float(zp);
        }
    }
    __syncthreads();
    scale = smem[0];
    const float inv_scale = smem[2];
    const uint8_t zero_point = static_cast<uint8_t>(smem[1]);

    #pragma unroll
    for (int i = 0; i < NUM_VEC; ++i) {
        uint8_t *out_ptr = &output[(Q == FuseQuantType::Int4
                         ? row * (COLS / 2) +
                               (col_offset + i * WARP_SIZE * VEC_SIZE +
                                lane_id * VEC_SIZE) / 2
                         : row * COLS + col_offset +
                               i * WARP_SIZE * VEC_SIZE + lane_id * VEC_SIZE)];
        if constexpr (Q == FuseQuantType::Int8)
            quantize_store_int8x8<S>(local[i], inv_scale, out_ptr, zero_point);
        else if constexpr (Q == FuseQuantType::Fp8E4M3)
            quantize_store_fp8x8(local[i], inv_scale, out_ptr);
        else
            quantize_store_int4x8<S>(local[i], inv_scale, out_ptr, zero_point);
    }

    if (tid == 0) {
        output_scale[row] = scale;
        if (output_zero_points) output_zero_points[row] = zero_point;
    }
}


// ---------- host launcher ----------------------------------------------------
template <typename T, FuseQuantType Q, FuseQuantScheme S>
void fused_hadamard_per_row_quant_impl(
    const T *input, uint8_t *output, float *output_scale,
    uint8_t *output_zero_points, int rows, int cols,
    FuseKernelPolicy policy, cudaStream_t stream)
{
    static_assert(std::is_same<T, __half>::value ||
                      std::is_same<T, __nv_bfloat16>::value,
                  "T must be __half or __nv_bfloat16");
    assert(cols > 0 && (cols & (cols - 1)) == 0 && "cols must be a power of 2");

    // Dispatch on cols exactly like hadamard_v4, mapping each tier to its fused
    // hadamard + per-row-quant kernel.
    if (cols < WARP_SIZE) {
        constexpr int THREADS_PER_BLOCK = 256;
        dim3 block(THREADS_PER_BLOCK);
        dim3 grid((rows + THREADS_PER_BLOCK / cols - 1) / (THREADS_PER_BLOCK / cols));
        switch (cols) {
        case 2:
            fuse_hadamard_per_row_quant_kernel_small<T, 2, Q, S><<<grid, block, 0, stream>>>(input, output, output_scale, output_zero_points, rows);
            break;
        case 4:
            fuse_hadamard_per_row_quant_kernel_small<T, 4, Q, S><<<grid, block, 0, stream>>>(input, output, output_scale, output_zero_points, rows);
            break;
        case 8:
            fuse_hadamard_per_row_quant_kernel_small<T, 8, Q, S><<<grid, block, 0, stream>>>(input, output, output_scale, output_zero_points, rows);
            break;
        case 16:
            fuse_hadamard_per_row_quant_kernel_small<T, 16, Q, S><<<grid, block, 0, stream>>>(input, output, output_scale, output_zero_points, rows);
            break;
        default:
            assert(false && "cols not yet handled; extend the switch");
        }
    } else if (cols < 256) {
        constexpr int ROWS_PER_BLOCK = 4;
        constexpr int THREADS_PER_BLOCK = ROWS_PER_BLOCK * WARP_SIZE;
        dim3 block(THREADS_PER_BLOCK);
        dim3 grid((rows + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK);
        switch (cols) {
        case 32:
            fuse_hadamard_per_row_quant_kernel_1warp1row_scalar<T, 32, ROWS_PER_BLOCK, Q, S><<<grid, block, 0, stream>>>(input, output, output_scale, output_zero_points, rows);
            break;
        case 64:
            fuse_hadamard_per_row_quant_kernel_1warp1row_scalar<T, 64, ROWS_PER_BLOCK, Q, S><<<grid, block, 0, stream>>>(input, output, output_scale, output_zero_points, rows);
            break;
        case 128:
            fuse_hadamard_per_row_quant_kernel_1warp1row_scalar<T, 128, ROWS_PER_BLOCK, Q, S><<<grid, block, 0, stream>>>(input, output, output_scale, output_zero_points, rows);
            break;
        default:
            assert(false && "cols not yet handled; extend the switch");
        }
    } else if (cols <= 8192) {
        // Auto is tuned independently for FP8 and integer output because their
        // vector conversion/store costs produce different crossover points.
        // 8192 always uses multi-warp: the vec specialization has excessive
        // register pressure at that size, for every quantization mode.
        bool auto_use_multi;
        if constexpr (Q == FuseQuantType::Fp8E4M3) {
            auto_use_multi = cols == 8192 || cols == 4096 ||
                (cols == 2048 && rows <= 1024) ||
                (cols == 1024 && rows <= 512) ||
                (cols == 512 && rows <= 256);
        } else {
            auto_use_multi = cols == 8192 ||
                (cols >= 1024 && cols <= 4096 && rows <= 64);
        }
        const bool use_multi = policy == FuseKernelPolicy::MultiWarp ||
            (policy == FuseKernelPolicy::Auto && auto_use_multi);
        if (use_multi) {
            dim3 grid(rows);
            switch (cols) {
            case 256: {
                constexpr int WARPS_PER_ROW = 1;
                dim3 block(WARPS_PER_ROW * WARP_SIZE);
                fuse_hadamard_per_row_quant_kernel_multi_warp_per_row<T, 256, WARPS_PER_ROW, Q, S><<<grid, block, 0, stream>>>(input, output, output_scale, output_zero_points, rows);
                break;
            }
            case 512: {
                constexpr int WARPS_PER_ROW = 2;
                dim3 block(WARPS_PER_ROW * WARP_SIZE);
                fuse_hadamard_per_row_quant_kernel_multi_warp_per_row<T, 512, WARPS_PER_ROW, Q, S><<<grid, block, 0, stream>>>(input, output, output_scale, output_zero_points, rows);
                break;
            }
            case 1024:
            case 2048:
            case 4096:
            case 8192: {
                constexpr int WARPS_PER_ROW = 4;
                dim3 block(WARPS_PER_ROW * WARP_SIZE);
                if (cols == 1024)
                    fuse_hadamard_per_row_quant_kernel_multi_warp_per_row<T, 1024, WARPS_PER_ROW, Q, S><<<grid, block, 0, stream>>>(input, output, output_scale, output_zero_points, rows);
                else if (cols == 2048)
                    fuse_hadamard_per_row_quant_kernel_multi_warp_per_row<T, 2048, WARPS_PER_ROW, Q, S><<<grid, block, 0, stream>>>(input, output, output_scale, output_zero_points, rows);
                else if (cols == 4096)
                    fuse_hadamard_per_row_quant_kernel_multi_warp_per_row<T, 4096, WARPS_PER_ROW, Q, S><<<grid, block, 0, stream>>>(input, output, output_scale, output_zero_points, rows);
                else
                    fuse_hadamard_per_row_quant_kernel_multi_warp_per_row<T, 8192, WARPS_PER_ROW, Q, S><<<grid, block, 0, stream>>>(input, output, output_scale, output_zero_points, rows);
                break;
            }
            default:
                assert(false && "cols not yet handled; extend the switch");
            }
        } else {
            constexpr int ROWS_PER_BLOCK = 4;
            dim3 block(ROWS_PER_BLOCK * WARP_SIZE);
            dim3 grid((rows + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK);
            switch (cols) {
            case 256:
                fuse_hadamard_per_row_quant_kernel_1warp1row_vec<T, 256, ROWS_PER_BLOCK, Q, S><<<grid, block, 0, stream>>>(input, output, output_scale, output_zero_points, rows);
                break;
            case 512:
                fuse_hadamard_per_row_quant_kernel_1warp1row_vec<T, 512, ROWS_PER_BLOCK, Q, S><<<grid, block, 0, stream>>>(input, output, output_scale, output_zero_points, rows);
                break;
            case 1024:
                fuse_hadamard_per_row_quant_kernel_1warp1row_vec<T, 1024, ROWS_PER_BLOCK, Q, S><<<grid, block, 0, stream>>>(input, output, output_scale, output_zero_points, rows);
                break;
            case 2048:
                fuse_hadamard_per_row_quant_kernel_1warp1row_vec<T, 2048, ROWS_PER_BLOCK, Q, S><<<grid, block, 0, stream>>>(input, output, output_scale, output_zero_points, rows);
                break;
            case 4096:
                fuse_hadamard_per_row_quant_kernel_1warp1row_vec<T, 4096, ROWS_PER_BLOCK, Q, S><<<grid, block, 0, stream>>>(input, output, output_scale, output_zero_points, rows);
                break;
            case 8192:
                fuse_hadamard_per_row_quant_kernel_1warp1row_vec<T, 8192, ROWS_PER_BLOCK, Q, S><<<grid, block, 0, stream>>>(input, output, output_scale, output_zero_points, rows);
                break;
            default:
                assert(false && "cols not yet handled; extend the switch");
            }
        }
    } else {
        switch (cols) {
        case 16384: {
            constexpr int WARPS_PER_ROW = 8;
            constexpr int THREADS_PER_BLOCK = WARPS_PER_ROW * WARP_SIZE;
            dim3 block(THREADS_PER_BLOCK);
            dim3 grid(rows);
            fuse_hadamard_per_row_quant_kernel_multi_warp_per_row_chunked<T, 16384, WARPS_PER_ROW, Q, S><<<grid, block, 0, stream>>>(input, output, output_scale, output_zero_points, rows);
            break;
        }
        case 32768: {
            constexpr int WARPS_PER_ROW = 16;
            constexpr int THREADS_PER_BLOCK = WARPS_PER_ROW * WARP_SIZE;
            dim3 block(THREADS_PER_BLOCK);
            dim3 grid(rows);
            fuse_hadamard_per_row_quant_kernel_multi_warp_per_row_chunked<T, 32768, WARPS_PER_ROW, Q, S><<<grid, block, 0, stream>>>(input, output, output_scale, output_zero_points, rows);
            break;
        }
        default:
            assert(false && "cols not yet handled; extend the switch");
        }
    }
}

template <typename T>
void fused_hadamard_per_row_quant(
    const T *input, int8_t *output, float *output_scale,
    int rows, int cols, cudaStream_t stream)
{
    fused_hadamard_per_row_quant_impl<
        T, FuseQuantType::Int8, FuseQuantScheme::Symmetric>(
        input, reinterpret_cast<uint8_t *>(output), output_scale,
        nullptr, rows, cols, FuseKernelPolicy::Auto, stream);
}

template <typename T>
void fused_hadamard_per_row_quant_fp8(
    const T *input, __nv_fp8_storage_t *output, float *output_scale,
    int rows, int cols, cudaStream_t stream)
{
    fused_hadamard_per_row_quant_impl<
        T, FuseQuantType::Fp8E4M3, FuseQuantScheme::ScaleOnly>(
        input, reinterpret_cast<uint8_t *>(output), output_scale,
        nullptr, rows, cols, FuseKernelPolicy::Auto, stream);
}

template <typename T>
void fused_hadamard_per_row_quantize_with_policy(
    const T *input, void *output, float *output_scale,
    uint8_t *zero_points, int rows, int cols,
    FuseQuantType quant, FuseQuantScheme scheme,
    FuseKernelPolicy policy, cudaStream_t stream)
{
    assert(quant != FuseQuantType::Fp8E4M3 ||
           scheme == FuseQuantScheme::ScaleOnly);
    assert(quant == FuseQuantType::Fp8E4M3 ||
           scheme != FuseQuantScheme::ScaleOnly);
    assert(scheme != FuseQuantScheme::Asymmetric || zero_points != nullptr);
    auto *bytes = reinterpret_cast<uint8_t *>(output);
    if (quant == FuseQuantType::Fp8E4M3) {
        fused_hadamard_per_row_quant_impl<
            T, FuseQuantType::Fp8E4M3, FuseQuantScheme::ScaleOnly>(
            input, bytes, output_scale, zero_points, rows, cols, policy, stream);
    } else if (quant == FuseQuantType::Int8 &&
               scheme == FuseQuantScheme::Symmetric) {
        fused_hadamard_per_row_quant_impl<
            T, FuseQuantType::Int8, FuseQuantScheme::Symmetric>(
            input, bytes, output_scale, zero_points, rows, cols, policy, stream);
    } else if (quant == FuseQuantType::Int8) {
        fused_hadamard_per_row_quant_impl<
            T, FuseQuantType::Int8, FuseQuantScheme::Asymmetric>(
            input, bytes, output_scale, zero_points, rows, cols, policy, stream);
    } else if (scheme == FuseQuantScheme::Symmetric) {
        fused_hadamard_per_row_quant_impl<
            T, FuseQuantType::Int4, FuseQuantScheme::Symmetric>(
            input, bytes, output_scale, zero_points, rows, cols, policy, stream);
    } else {
        fused_hadamard_per_row_quant_impl<
            T, FuseQuantType::Int4, FuseQuantScheme::Asymmetric>(
            input, bytes, output_scale, zero_points, rows, cols, policy, stream);
    }
}

template <typename T>
void fused_hadamard_per_row_quantize(
    const T *input, void *output, float *output_scale,
    uint8_t *zero_points, int rows, int cols,
    FuseQuantType quant, FuseQuantScheme scheme, cudaStream_t stream)
{
    fused_hadamard_per_row_quantize_with_policy<T>(
        input, output, output_scale, zero_points, rows, cols,
        quant, scheme, FuseKernelPolicy::Auto, stream);
}

template void fused_hadamard_per_row_quant<__half>(const __half *, int8_t *, float *, int, int, cudaStream_t);
template void fused_hadamard_per_row_quant<__nv_bfloat16>(const __nv_bfloat16 *, int8_t *, float *, int, int, cudaStream_t);
template void fused_hadamard_per_row_quant_fp8<__half>(const __half *, __nv_fp8_storage_t *, float *, int, int, cudaStream_t);
template void fused_hadamard_per_row_quant_fp8<__nv_bfloat16>(const __nv_bfloat16 *, __nv_fp8_storage_t *, float *, int, int, cudaStream_t);
template void fused_hadamard_per_row_quantize<__half>(const __half *, void *, float *, uint8_t *, int, int, FuseQuantType, FuseQuantScheme, cudaStream_t);
template void fused_hadamard_per_row_quantize<__nv_bfloat16>(const __nv_bfloat16 *, void *, float *, uint8_t *, int, int, FuseQuantType, FuseQuantScheme, cudaStream_t);
template void fused_hadamard_per_row_quantize_with_policy<__half>(const __half *, void *, float *, uint8_t *, int, int, FuseQuantType, FuseQuantScheme, FuseKernelPolicy, cudaStream_t);
template void fused_hadamard_per_row_quantize_with_policy<__nv_bfloat16>(const __nv_bfloat16 *, void *, float *, uint8_t *, int, int, FuseQuantType, FuseQuantScheme, FuseKernelPolicy, cudaStream_t);
