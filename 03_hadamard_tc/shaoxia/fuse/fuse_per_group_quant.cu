#include "fuse_per_group_quant.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cassert>
#include <cstdint>
#include <type_traits>

namespace {

constexpr int kWarpSize = 32;
constexpr int kVecSize = 8;
constexpr int kElementsPerSegment = kWarpSize * kVecSize;

struct alignas(16) Half8 {
    __half x[kVecSize];
};

struct alignas(16) BFloat16x8 {
    __nv_bfloat16 x[kVecSize];
};

static_assert(sizeof(Half8) == 16);
static_assert(sizeof(BFloat16x8) == 16);

__device__ __forceinline__ void load_float8(
    const __half* ptr, float (&value)[kVecSize])
{
    const Half8 packed = *reinterpret_cast<const Half8*>(ptr);
#pragma unroll
    for (int i = 0; i < kVecSize; ++i)
        value[i] = __half2float(packed.x[i]);
}

__device__ __forceinline__ void load_float8(
    const __nv_bfloat16* ptr, float (&value)[kVecSize])
{
    const BFloat16x8 packed = *reinterpret_cast<const BFloat16x8*>(ptr);
#pragma unroll
    for (int i = 0; i < kVecSize; ++i)
        value[i] = __bfloat162float(packed.x[i]);
}

template <typename T>
__device__ __forceinline__ float round_to_storage(float value);

template <>
__device__ __forceinline__ float round_to_storage<__half>(float value)
{
    return __half2float(__float2half_rn(value));
}

template <>
__device__ __forceinline__ float round_to_storage<__nv_bfloat16>(float value)
{
    return __bfloat162float(__float2bfloat16_rn(value));
}

template <FuseQuantType Q>
__device__ __forceinline__ constexpr float symmetric_qmax()
{
    if constexpr (Q == FuseQuantType::Fp8E4M3) return 448.0f;
    if constexpr (Q == FuseQuantType::Int4) return 7.0f;
    return 127.0f;
}

template <FuseQuantType Q, FuseQuantScheme S>
__device__ __forceinline__ void make_quant_params(
    float amax, float group_min, float group_max,
    float& scale, float& inv_scale, uint8_t& zero_point)
{
    if constexpr (S != FuseQuantScheme::Asymmetric) {
        if (amax > 0.0f) {
            constexpr float qmax = symmetric_qmax<Q>();
            scale = amax / qmax;
            inv_scale = qmax / amax;
        } else {
            scale = 1.0f;
            inv_scale = 1.0f;
        }
        zero_point = 0;
    } else {
        constexpr float qmax = Q == FuseQuantType::Int4 ? 15.0f : 255.0f;
        const float range = group_max - group_min;
        if (range > 0.0f) {
            scale = range / qmax;
            inv_scale = qmax / range;
        } else {
            scale = 1.0f;
            inv_scale = 1.0f;
        }
        zero_point = static_cast<uint8_t>(fminf(
            qmax, fmaxf(0.0f, rintf(-group_min * inv_scale))));
    }
}

template <FuseQuantScheme S>
__device__ __forceinline__ uint8_t quantize_int8(
    float value, float inv_scale, uint8_t zero_point)
{
    float q = rintf(value * inv_scale);
    if constexpr (S == FuseQuantScheme::Asymmetric) {
        q = fminf(255.0f, fmaxf(0.0f, q + float(zero_point)));
        return static_cast<uint8_t>(q);
    } else {
        q = fminf(127.0f, fmaxf(-127.0f, q));
        return static_cast<uint8_t>(static_cast<int8_t>(q));
    }
}

template <FuseQuantScheme S>
__device__ __forceinline__ uint8_t quantize_int4(
    float value, float inv_scale, uint8_t zero_point)
{
    float q = rintf(value * inv_scale);
    if constexpr (S == FuseQuantScheme::Asymmetric)
        q = fminf(15.0f, fmaxf(0.0f, q + float(zero_point)));
    else
        q = fminf(7.0f, fmaxf(-7.0f, q));
    return static_cast<uint8_t>(static_cast<int>(q)) & 0x0f;
}

__device__ __forceinline__ uint8_t quantize_fp8(
    float value, float inv_scale)
{
    return __nv_cvt_float_to_fp8(
        value * inv_scale, __NV_SATFINITE, __NV_E4M3);
}

template <FuseQuantScheme S>
__device__ __forceinline__ void store_int8x8(
    const float (&value)[kVecSize], float inv_scale,
    uint8_t* output, uint8_t zero_point)
{
    uint8_t packed[kVecSize];
#pragma unroll
    for (int i = 0; i < kVecSize; ++i)
        packed[i] = quantize_int8<S>(value[i], inv_scale, zero_point);
    *reinterpret_cast<unsigned long long*>(output) =
        *reinterpret_cast<const unsigned long long*>(packed);
}

__device__ __forceinline__ void store_fp8x8(
    const float (&value)[kVecSize], float inv_scale, uint8_t* output)
{
    __nv_fp8x2_storage_t packed[kVecSize / 2];
#pragma unroll
    for (int i = 0; i < kVecSize / 2; ++i) {
        const float2 pair = make_float2(
            value[2 * i] * inv_scale,
            value[2 * i + 1] * inv_scale);
        packed[i] = __nv_cvt_float2_to_fp8x2(
            pair, __NV_SATFINITE, __NV_E4M3);
    }
    *reinterpret_cast<unsigned long long*>(output) =
        *reinterpret_cast<const unsigned long long*>(packed);
}

template <FuseQuantScheme S>
__device__ __forceinline__ void store_int4x8(
    const float (&value)[kVecSize], float inv_scale,
    uint8_t* output, uint8_t zero_point)
{
    uint32_t packed = 0;
#pragma unroll
    for (int i = 0; i < kVecSize / 2; ++i) {
        const uint8_t low =
            quantize_int4<S>(value[2 * i], inv_scale, zero_point);
        const uint8_t high =
            quantize_int4<S>(value[2 * i + 1], inv_scale, zero_point);
        packed |= uint32_t(low | (high << 4)) << (8 * i);
    }
    *reinterpret_cast<uint32_t*>(output) = packed;
}

template <FuseQuantType Q, FuseQuantScheme S>
__device__ __forceinline__ void store_vector(
    const float (&value)[kVecSize], float inv_scale,
    uint8_t* output, uint8_t zero_point)
{
    if constexpr (Q == FuseQuantType::Int8)
        store_int8x8<S>(value, inv_scale, output, zero_point);
    else if constexpr (Q == FuseQuantType::Fp8E4M3)
        store_fp8x8(value, inv_scale, output);
    else
        store_int4x8<S>(value, inv_scale, output, zero_point);
}

__device__ __forceinline__ int exchange_index(
    int warp, int vec, int element, int lane, int vectors_per_warp)
{
    return (((warp * vectors_per_warp + vec) * kVecSize + element) *
            kWarpSize + lane);
}

// Simple path for the dimensions below one 256-element vector segment.
template <typename T, int COLS, FuseQuantType Q, FuseQuantScheme S>
__global__ void fused_hadamard_per_group_small_kernel(
    const T* __restrict__ input, uint8_t* __restrict__ output,
    float* __restrict__ scales, uint8_t* __restrict__ zero_points,
    int rows, int group_size)
{
    __shared__ float value[COLS];
    __shared__ float params[3 * COLS];
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    if (row >= rows) return;

    value[tid] = float(input[row * COLS + tid]);
    __syncthreads();

#pragma unroll
    for (int stride = 1; stride < COLS; stride <<= 1) {
        const float self = value[tid];
        const float other = value[tid ^ stride];
        __syncthreads();
        value[tid] = (tid & stride) ? other - self : self + other;
        __syncthreads();
    }
    value[tid] = round_to_storage<T>(value[tid]);
    __syncthreads();

    const int groups_per_row = COLS / group_size;
    const int group = tid / group_size;
    const int lane_in_group = tid % group_size;
    if (lane_in_group == 0) {
        float amax = 0.0f;
        float group_min = value[tid];
        float group_max = value[tid];
        for (int i = 0; i < group_size; ++i) {
            const float x = value[tid + i];
            amax = fmaxf(amax, fabsf(x));
            if constexpr (S == FuseQuantScheme::Asymmetric) {
                group_min = fminf(group_min, x);
                group_max = fmaxf(group_max, x);
            }
        }
        uint8_t zero_point;
        make_quant_params<Q, S>(
            amax, group_min, group_max,
            params[group], params[groups_per_row + group], zero_point);
        params[2 * groups_per_row + group] = float(zero_point);
        scales[row * groups_per_row + group] = params[group];
        if (zero_points)
            zero_points[row * groups_per_row + group] = zero_point;
    }
    __syncthreads();

    const float inv_scale = params[groups_per_row + group];
    const uint8_t zero_point =
        static_cast<uint8_t>(params[2 * groups_per_row + group]);
    if constexpr (Q == FuseQuantType::Int8) {
        output[row * COLS + tid] =
            quantize_int8<S>(value[tid], inv_scale, zero_point);
    } else if constexpr (Q == FuseQuantType::Fp8E4M3) {
        output[row * COLS + tid] = quantize_fp8(value[tid], inv_scale);
    } else if ((tid & 1) == 0) {
        const int next_group = (tid + 1) / group_size;
        const float next_inv = params[groups_per_row + next_group];
        const uint8_t next_zero =
            static_cast<uint8_t>(params[2 * groups_per_row + next_group]);
        const uint8_t low =
            quantize_int4<S>(value[tid], inv_scale, zero_point);
        const uint8_t high =
            quantize_int4<S>(value[tid + 1], next_inv, next_zero);
        output[row * (COLS / 2) + tid / 2] = low | (high << 4);
    }
}

// One block processes one row. The Hadamard part mirrors the chunked
// multi-warp per-row kernel. Quantization statistics are computed only after
// every Hadamard stage and storage-type rounding have completed.
template <typename T, int COLS, int WARPS_PER_ROW,
          FuseQuantType Q, FuseQuantScheme S>
__global__ void fused_hadamard_per_group_multi_warp_kernel(
    const T* __restrict__ input, uint8_t* __restrict__ output,
    float* __restrict__ scales, uint8_t* __restrict__ zero_points,
    int rows, int group_size)
{
    constexpr int elements_per_warp = COLS / WARPS_PER_ROW;
    constexpr int num_vec = elements_per_warp / kElementsPerSegment;
    constexpr int max_chunks_by_smem =
        (32 * 1024) /
        (WARPS_PER_ROW * kVecSize * kWarpSize * int(sizeof(float)));
    constexpr int chunks_per_round =
        num_vec < max_chunks_by_smem ? num_vec : max_chunks_by_smem;
    constexpr int num_rounds = num_vec / chunks_per_round;
    constexpr int smem_floats =
        WARPS_PER_ROW * chunks_per_round * kVecSize * kWarpSize;
    static_assert(num_vec >= 1);
    static_assert(chunks_per_round >= 1);
    static_assert(num_vec % chunks_per_round == 0);
    __shared__ float smem[smem_floats];

    const int row = blockIdx.x;
    if (row >= rows) return;
    const int tid = threadIdx.x;
    const int lane = tid & (kWarpSize - 1);
    const int warp = tid / kWarpSize;
    const int col_offset = warp * elements_per_warp;
    float local[num_vec][kVecSize];

#pragma unroll
    for (int i = 0; i < num_vec; ++i) {
        load_float8(
            input + row * COLS + col_offset +
                i * kElementsPerSegment + lane * kVecSize,
            local[i]);
#pragma unroll
        for (int stride = 1; stride < kVecSize; stride <<= 1) {
#pragma unroll
            for (int k = 0; k < kVecSize; ++k) {
                if (k & stride) {
                    const float lower = local[i][k ^ stride];
                    const float upper = local[i][k];
                    local[i][k ^ stride] = lower + upper;
                    local[i][k] = lower - upper;
                }
            }
        }
#pragma unroll
        for (int stride = 1; stride < kWarpSize; stride <<= 1) {
#pragma unroll
            for (int k = 0; k < kVecSize; ++k) {
                const float other =
                    __shfl_xor_sync(0xffffffffu, local[i][k], stride);
                local[i][k] =
                    (lane & stride) ? other - local[i][k]
                                    : local[i][k] + other;
            }
        }
    }

#pragma unroll
    for (int stride = 1; stride < num_vec; stride <<= 1) {
#pragma unroll
        for (int i = 0; i < num_vec; ++i) {
#pragma unroll
            for (int k = 0; k < kVecSize; ++k) {
                if ((i & stride) == 0) {
                    const float lower = local[i][k];
                    const float upper = local[i ^ stride][k];
                    local[i][k] = lower + upper;
                    local[i ^ stride][k] = lower - upper;
                }
            }
        }
    }

#pragma unroll
    for (int round = 0; round < num_rounds; ++round) {
        const int base = round * chunks_per_round;
#pragma unroll
        for (int chunk = 0; chunk < chunks_per_round; ++chunk)
#pragma unroll
            for (int k = 0; k < kVecSize; ++k)
                smem[exchange_index(
                    warp, chunk, k, lane, chunks_per_round)] =
                    local[base + chunk][k];
        __syncthreads();

#pragma unroll
        for (int stride = 1; stride < WARPS_PER_ROW; stride <<= 1) {
#pragma unroll
            for (int chunk = 0; chunk < chunks_per_round; ++chunk) {
#pragma unroll
                for (int k = 0; k < kVecSize; ++k) {
                    const float other = smem[exchange_index(
                        warp ^ stride, chunk, k, lane, chunks_per_round)];
                    local[base + chunk][k] =
                        (warp & stride) ? other - local[base + chunk][k]
                                        : local[base + chunk][k] + other;
                }
            }
            __syncthreads();
#pragma unroll
            for (int chunk = 0; chunk < chunks_per_round; ++chunk)
#pragma unroll
                for (int k = 0; k < kVecSize; ++k)
                    smem[exchange_index(
                        warp, chunk, k, lane, chunks_per_round)] =
                        local[base + chunk][k];
            __syncthreads();
        }
    }

#pragma unroll
    for (int i = 0; i < num_vec; ++i)
#pragma unroll
        for (int k = 0; k < kVecSize; ++k)
            local[i][k] = round_to_storage<T>(local[i][k]);

    const int groups_per_row = COLS / group_size;

    if (group_size < kVecSize) {
        // A group is wholly contained in one lane's contiguous float8.
#pragma unroll
        for (int i = 0; i < num_vec; ++i) {
            const int base_col = col_offset + i * kElementsPerSegment +
                                 lane * kVecSize;
            for (int begin = 0; begin < kVecSize; begin += group_size) {
                float amax = 0.0f;
                float group_min = local[i][begin];
                float group_max = local[i][begin];
                for (int k = 0; k < group_size; ++k) {
                    const float x = local[i][begin + k];
                    amax = fmaxf(amax, fabsf(x));
                    if constexpr (S == FuseQuantScheme::Asymmetric) {
                        group_min = fminf(group_min, x);
                        group_max = fmaxf(group_max, x);
                    }
                }
                float scale;
                float inv_scale;
                uint8_t zero_point;
                make_quant_params<Q, S>(
                    amax, group_min, group_max,
                    scale, inv_scale, zero_point);
                const int group = (base_col + begin) / group_size;
                scales[row * groups_per_row + group] = scale;
                if (zero_points)
                    zero_points[row * groups_per_row + group] = zero_point;

                if constexpr (Q == FuseQuantType::Int8) {
                    for (int k = 0; k < group_size; ++k)
                        output[row * COLS + base_col + begin + k] =
                            quantize_int8<S>(
                                local[i][begin + k], inv_scale, zero_point);
                } else if constexpr (Q == FuseQuantType::Fp8E4M3) {
                    for (int k = 0; k < group_size; ++k)
                        output[row * COLS + base_col + begin + k] =
                            quantize_fp8(local[i][begin + k], inv_scale);
                } else {
                    for (int k = 0; k < group_size; k += 2) {
                        const uint8_t low = quantize_int4<S>(
                            local[i][begin + k], inv_scale, zero_point);
                        const uint8_t high = quantize_int4<S>(
                            local[i][begin + k + 1], inv_scale, zero_point);
                        output[row * (COLS / 2) +
                               (base_col + begin + k) / 2] =
                            low | (high << 4);
                    }
                }
            }
        }
        return;
    }

    if (group_size <= kElementsPerSegment) {
        // Each 256-element segment contains one or more complete groups.
        const int lanes_per_group = group_size / kVecSize;
#pragma unroll
        for (int i = 0; i < num_vec; ++i) {
            float amax = 0.0f;
            float group_min = local[i][0];
            float group_max = local[i][0];
#pragma unroll
            for (int k = 0; k < kVecSize; ++k) {
                const float x = local[i][k];
                amax = fmaxf(amax, fabsf(x));
                if constexpr (S == FuseQuantScheme::Asymmetric) {
                    group_min = fminf(group_min, x);
                    group_max = fmaxf(group_max, x);
                }
            }
            for (int offset = lanes_per_group / 2;
                 offset > 0; offset >>= 1) {
                amax = fmaxf(amax, __shfl_xor_sync(
                    0xffffffffu, amax, offset, lanes_per_group));
                if constexpr (S == FuseQuantScheme::Asymmetric) {
                    group_min = fminf(group_min, __shfl_xor_sync(
                        0xffffffffu, group_min, offset, lanes_per_group));
                    group_max = fmaxf(group_max, __shfl_xor_sync(
                        0xffffffffu, group_max, offset, lanes_per_group));
                }
            }
            float scale;
            float inv_scale;
            uint8_t zero_point;
            make_quant_params<Q, S>(
                amax, group_min, group_max,
                scale, inv_scale, zero_point);
            const int segment = warp * num_vec + i;
            const int group = segment *
                (kElementsPerSegment / group_size) + lane / lanes_per_group;
            if ((lane % lanes_per_group) == 0) {
                scales[row * groups_per_row + group] = scale;
                if (zero_points)
                    zero_points[row * groups_per_row + group] = zero_point;
            }
            const int base_col = col_offset + i * kElementsPerSegment +
                                 lane * kVecSize;
            uint8_t* out = output +
                (Q == FuseQuantType::Int4
                    ? row * (COLS / 2) + base_col / 2
                    : row * COLS + base_col);
            store_vector<Q, S>(local[i], inv_scale, out, zero_point);
        }
        return;
    }

    // Large groups combine one partial statistic per contiguous 256-element
    // segment. The Hadamard exchange buffer is no longer needed and is reused.
    constexpr int total_segments = COLS / kElementsPerSegment;
#pragma unroll
    for (int i = 0; i < num_vec; ++i) {
        float amax = 0.0f;
        float group_min = local[i][0];
        float group_max = local[i][0];
#pragma unroll
        for (int k = 0; k < kVecSize; ++k) {
            const float x = local[i][k];
            amax = fmaxf(amax, fabsf(x));
            if constexpr (S == FuseQuantScheme::Asymmetric) {
                group_min = fminf(group_min, x);
                group_max = fmaxf(group_max, x);
            }
        }
#pragma unroll
        for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
            amax = fmaxf(
                amax, __shfl_xor_sync(0xffffffffu, amax, offset));
            if constexpr (S == FuseQuantScheme::Asymmetric) {
                group_min = fminf(group_min, __shfl_xor_sync(
                    0xffffffffu, group_min, offset));
                group_max = fmaxf(group_max, __shfl_xor_sync(
                    0xffffffffu, group_max, offset));
            }
        }
        if (lane == 0) {
            const int segment = warp * num_vec + i;
            smem[segment] = amax;
            if constexpr (S == FuseQuantScheme::Asymmetric) {
                smem[total_segments + segment] = group_min;
                smem[2 * total_segments + segment] = group_max;
            }
        }
    }
    __syncthreads();

    const int segments_per_group =
        group_size / kElementsPerSegment;
    constexpr int params_base = 3 * total_segments;
    if (tid < groups_per_row) {
        const int first_segment = tid * segments_per_group;
        float amax = 0.0f;
        float group_min = smem[total_segments + first_segment];
        float group_max = smem[2 * total_segments + first_segment];
        for (int i = 0; i < segments_per_group; ++i) {
            const int segment = first_segment + i;
            amax = fmaxf(amax, smem[segment]);
            if constexpr (S == FuseQuantScheme::Asymmetric) {
                group_min = fminf(
                    group_min, smem[total_segments + segment]);
                group_max = fmaxf(
                    group_max, smem[2 * total_segments + segment]);
            }
        }
        uint8_t zero_point;
        make_quant_params<Q, S>(
            amax, group_min, group_max,
            smem[params_base + tid],
            smem[params_base + groups_per_row + tid], zero_point);
        smem[params_base + 2 * groups_per_row + tid] = float(zero_point);
        scales[row * groups_per_row + tid] = smem[params_base + tid];
        if (zero_points)
            zero_points[row * groups_per_row + tid] = zero_point;
    }
    __syncthreads();

#pragma unroll
    for (int i = 0; i < num_vec; ++i) {
        const int segment = warp * num_vec + i;
        const int group = segment / segments_per_group;
        const float inv_scale =
            smem[params_base + groups_per_row + group];
        const uint8_t zero_point = static_cast<uint8_t>(
            smem[params_base + 2 * groups_per_row + group]);
        const int base_col = col_offset + i * kElementsPerSegment +
                             lane * kVecSize;
        uint8_t* out = output +
            (Q == FuseQuantType::Int4
                ? row * (COLS / 2) + base_col / 2
                : row * COLS + base_col);
        store_vector<Q, S>(local[i], inv_scale, out, zero_point);
    }
}

template <typename T, FuseQuantType Q, FuseQuantScheme S>
void launch_per_group(
    const T* input, uint8_t* output, float* scales, uint8_t* zero_points,
    int rows, int cols, int group_size, cudaStream_t stream)
{
#define LAUNCH_SMALL(C)                                                       \
    fused_hadamard_per_group_small_kernel<T, C, Q, S>                         \
        <<<rows, C, 0, stream>>>(                                             \
            input, output, scales, zero_points, rows, group_size)
#define LAUNCH_MULTI(C, W)                                                    \
    fused_hadamard_per_group_multi_warp_kernel<T, C, W, Q, S>                 \
        <<<rows, W * kWarpSize, 0, stream>>>(                                 \
            input, output, scales, zero_points, rows, group_size)

    switch (cols) {
        case 2: LAUNCH_SMALL(2); break;
        case 4: LAUNCH_SMALL(4); break;
        case 8: LAUNCH_SMALL(8); break;
        case 16: LAUNCH_SMALL(16); break;
        case 32: LAUNCH_SMALL(32); break;
        case 64: LAUNCH_SMALL(64); break;
        case 128: LAUNCH_SMALL(128); break;
        case 256: LAUNCH_MULTI(256, 1); break;
        case 512: LAUNCH_MULTI(512, 2); break;
        case 1024: LAUNCH_MULTI(1024, 4); break;
        case 2048: LAUNCH_MULTI(2048, 4); break;
        case 4096: LAUNCH_MULTI(4096, 4); break;
        case 8192: LAUNCH_MULTI(8192, 4); break;
        case 16384: LAUNCH_MULTI(16384, 8); break;
        case 32768: LAUNCH_MULTI(32768, 16); break;
        default: assert(false && "unsupported cols");
    }
#undef LAUNCH_MULTI
#undef LAUNCH_SMALL
}

template <typename T>
void dispatch_per_group(
    const T* input, uint8_t* output, float* scales, uint8_t* zero_points,
    int rows, int cols, int group_size,
    FuseQuantType quant, FuseQuantScheme scheme, cudaStream_t stream)
{
    if (quant == FuseQuantType::Fp8E4M3) {
        launch_per_group<
            T, FuseQuantType::Fp8E4M3, FuseQuantScheme::ScaleOnly>(
            input, output, scales, zero_points,
            rows, cols, group_size, stream);
    } else if (quant == FuseQuantType::Int8 &&
               scheme == FuseQuantScheme::Symmetric) {
        launch_per_group<
            T, FuseQuantType::Int8, FuseQuantScheme::Symmetric>(
            input, output, scales, zero_points,
            rows, cols, group_size, stream);
    } else if (quant == FuseQuantType::Int8) {
        launch_per_group<
            T, FuseQuantType::Int8, FuseQuantScheme::Asymmetric>(
            input, output, scales, zero_points,
            rows, cols, group_size, stream);
    } else if (scheme == FuseQuantScheme::Symmetric) {
        launch_per_group<
            T, FuseQuantType::Int4, FuseQuantScheme::Symmetric>(
            input, output, scales, zero_points,
            rows, cols, group_size, stream);
    } else {
        launch_per_group<
            T, FuseQuantType::Int4, FuseQuantScheme::Asymmetric>(
            input, output, scales, zero_points,
            rows, cols, group_size, stream);
    }
}

}  // namespace

template <typename T>
void fused_hadamard_per_group_quantize(
    const T* input, void* output, float* scales, uint8_t* zero_points,
    int rows, int cols, int group_size,
    FuseQuantType quant, FuseQuantScheme scheme, cudaStream_t stream)
{
    static_assert(std::is_same<T, __half>::value ||
                      std::is_same<T, __nv_bfloat16>::value,
                  "T must be __half or __nv_bfloat16");
    assert(input && output && scales);
    assert(rows >= 0);
    assert(cols >= 2 && cols <= 32768 && (cols & (cols - 1)) == 0);
    assert(group_size >= 1 && group_size <= cols &&
           (group_size & (group_size - 1)) == 0 &&
           cols % group_size == 0);
    assert(quant != FuseQuantType::Fp8E4M3 ||
           scheme == FuseQuantScheme::ScaleOnly);
    assert(quant == FuseQuantType::Fp8E4M3 ||
           scheme != FuseQuantScheme::ScaleOnly);
    assert(scheme != FuseQuantScheme::Asymmetric || zero_points);
    assert(quant != FuseQuantType::Int4 || group_size >= 2);
    if (rows == 0) return;

    dispatch_per_group<T>(
        input, reinterpret_cast<uint8_t*>(output), scales, zero_points,
        rows, cols, group_size, quant, scheme, stream);
}

template void fused_hadamard_per_group_quantize<__half>(
    const __half*, void*, float*, uint8_t*, int, int, int,
    FuseQuantType, FuseQuantScheme, cudaStream_t);
template void fused_hadamard_per_group_quantize<__nv_bfloat16>(
    const __nv_bfloat16*, void*, float*, uint8_t*, int, int, int,
    FuseQuantType, FuseQuantScheme, cudaStream_t);

