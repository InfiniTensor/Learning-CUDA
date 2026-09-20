#include "nvfp4.cuh"

#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>

namespace nvfp4 {
namespace {

__device__ __forceinline__
uint32_t round_shift_rne(uint32_t value, int shift) {
    if (shift <= 0) return value;
    if (shift >= 32) return 0;

    uint32_t kept = value >> shift;
    uint32_t remainder = value & ((1u << shift) - 1u);
    uint32_t halfway = 1u << (shift - 1);

    if (remainder > halfway ||
        (remainder == halfway && (kept & 1u))) {
        ++kept;
    }
    return kept;
}

__device__ __forceinline__
uint32_t round_shift_up(uint32_t value, int shift) {
    if (shift <= 0) return value;
    if (shift >= 32) return value ? 1u : 0u;

    uint32_t kept = value >> shift;
    uint32_t remainder = value & ((1u << shift) - 1u);
    return kept + (remainder != 0);
}

// FP16 / BF16 软件转换。
__device__ __forceinline__
float fp16_bits_to_float(uint16_t h) {
    uint32_t sign = static_cast<uint32_t>(h >> 15);
    uint32_t exp = (h >> 10) & 0x1fu;
    uint32_t frac = h & 0x03ffu;

    if (exp == 0) {
        if (frac == 0) {
            return __uint_as_float(sign << 31);
        }

        float value = ldexpf(static_cast<float>(frac), -24);
        return sign ? -value : value;
    }

    if (exp == 0x1fu) {
        uint32_t bits = (sign << 31) | 0x7f800000u;
        if (frac != 0) bits |= 0x00400000u;
        return __uint_as_float(bits);
    }

    float mantissa = 1.0f + static_cast<float>(frac) / 1024.0f;
    float value = ldexpf(mantissa, static_cast<int>(exp) - 15);
    return sign ? -value : value;
}

__device__ __forceinline__
uint16_t float_to_fp16_bits_rne(float x) {
    uint32_t bits = __float_as_uint(x);
    uint32_t sign = (bits >> 16) & 0x8000u;
    uint32_t e32 = (bits >> 23) & 0xffu;
    uint32_t frac32 = bits & 0x7fffffu;

    if (e32 == 0xffu) {
        if (frac32 != 0) {
            return static_cast<uint16_t>(sign | 0x7e00u);
        }
        return static_cast<uint16_t>(sign | 0x7c00u);
    }

    if (e32 == 0) {

        return static_cast<uint16_t>(sign);
    }

    int E = static_cast<int>(e32) - 127;
    uint32_t sig = (1u << 23) | frac32;

    if (E > 15) {
        return static_cast<uint16_t>(sign | 0x7c00u);
    }

    if (E >= -14) {
        uint32_t rounded_sig = round_shift_rne(sig, 13);

        if (rounded_sig == 2048u) {
            rounded_sig = 1024u;
            ++E;
            if (E > 15) {
                return static_cast<uint16_t>(sign | 0x7c00u);
            }
        }

        uint32_t exp16 = static_cast<uint32_t>(E + 15);
        uint32_t frac16 = rounded_sig & 0x03ffu;
        return static_cast<uint16_t>(sign | (exp16 << 10) | frac16);
    }

    int shift = -E - 1;
    uint32_t q = round_shift_rne(sig, shift);

    if (q >= 1024u) {
        return static_cast<uint16_t>(sign | 0x0400u);
    }

    return static_cast<uint16_t>(sign | q);
}

__device__ __forceinline__
uint16_t float_to_bf16_bits_rne(float x) {
    uint32_t bits = __float_as_uint(x);
    uint32_t exp = bits & 0x7f800000u;
    uint32_t frac = bits & 0x007fffffu;

    if (exp == 0x7f800000u && frac != 0) {

        return static_cast<uint16_t>((bits >> 16) | 0x0040u);
    }

    uint32_t lsb = (bits >> 16) & 1u;
    bits += 0x7fffu + lsb;
    return static_cast<uint16_t>(bits >> 16);
}

// E2M1 编码：RNE 舍入，非有限值饱和到 ±6。
__device__ __forceinline__
uint8_t float_to_e2m1(float x) {
    const uint32_t bits = __float_as_uint(x);

    const uint8_t sign =
        static_cast<uint8_t>(((bits >> 31) & 1u) << 3);

    const uint32_t exp =
        (bits >> 23) & 0xffu;

    if (exp == 0xffu) {
        return static_cast<uint8_t>(sign | 0x7u);
    }

    const float a =
        __uint_as_float(bits & 0x7fffffffu);

    uint32_t code = 0;

    code += static_cast<uint32_t>(a >  0.25f);
    code += static_cast<uint32_t>(a >= 0.75f);
    code += static_cast<uint32_t>(a >  1.25f);
    code += static_cast<uint32_t>(a >= 1.75f);
    code += static_cast<uint32_t>(a >  2.50f);
    code += static_cast<uint32_t>(a >= 3.50f);
    code += static_cast<uint32_t>(a >  5.00f);

    return static_cast<uint8_t>(
        sign | static_cast<uint8_t>(code));
}

// 直接构造 FP32 位表示完成 E2M1 解码。
__device__ __forceinline__
float e2m1_to_float_v3(uint8_t q) {
    q &= 0x0fu;

    const uint32_t sign =
        static_cast<uint32_t>((q >> 3) & 1u);

    const uint32_t exp =
        static_cast<uint32_t>((q >> 1) & 0x3u);

    const uint32_t frac =
        static_cast<uint32_t>(q & 0x1u);

    const uint32_t fp_exp = 126u + exp;

    const uint32_t fp_frac =
        (exp != 0u) ? (frac << 22) : 0u;

    uint32_t magnitude_bits =
        (fp_exp << 23) | fp_frac;

    if ((exp | frac) == 0u) {
        magnitude_bits = 0u;
    }

    const uint32_t bits =
        (sign << 31) | magnitude_bits;

    return __uint_as_float(bits);
}

// 直接构造 FP32 位表示完成 E4M3 scale 解码。
__device__ __forceinline__
float e4m3_scale_to_float_v4(uint8_t q) {
    const uint32_t exp =
        (static_cast<uint32_t>(q) >> 3) & 0x0fu;

    const uint32_t frac =
        static_cast<uint32_t>(q) & 0x07u;

    if (exp != 0u) {

        if (exp == 0x0fu && frac == 0x07u) {
            return NAN;
        }

        const uint32_t bits =
            ((exp + 120u) << 23) |
            (frac << 20);

        return __uint_as_float(bits);
    }

    if (frac == 0u) {
        return 0.0f;
    }

    const int msb = 31 - __clz(frac);
    const uint32_t leading = 1u << msb;
    const uint32_t remainder = frac - leading;

    const uint32_t fp_exp =
        static_cast<uint32_t>(118 + msb);

    const uint32_t fp_frac =
        remainder << (23 - msb);

    const uint32_t bits =
        (fp_exp << 23) | fp_frac;

    return __uint_as_float(bits);
}

// E4M3 block scale 使用向上舍入。
__device__ __forceinline__
uint8_t float_to_e4m3_round_up(float x) {
    if (!(x > 0.0f)) return 0;
    if (isinf(x)) return 0x7eu;

    uint32_t bits = __float_as_uint(x);
    uint32_t e32 = (bits >> 23) & 0xffu;
    uint32_t frac32 = bits & 0x7fffffu;

    uint32_t sig;
    int e8;

    if (e32 != 0) {
        sig = (1u << 23) | frac32;
        e8 = static_cast<int>(e32) - 120;
    } else {
        int norm_shift = __clz(frac32) - 8;
        sig = frac32 << norm_shift;
        int E = -126 - norm_shift;
        e8 = E + 7;
    }

    uint32_t e4;
    uint32_t f3;

    if (e8 >= 1) {
        uint32_t rounded_sig = round_shift_up(sig, 20);

        if (rounded_sig == 16) {
            rounded_sig = 8;
            ++e8;
        }

        e4 = static_cast<uint32_t>(e8);
        f3 = rounded_sig & 0x7u;

        if (e4 > 15 || (e4 == 15 && f3 == 7)) {
            e4 = 15;
            f3 = 6;
        }
    } else {
        int shift = 21 - e8;
        uint32_t q = round_shift_up(sig, shift);

        if (q >= 8) {
            e4 = 1;
            f3 = 0;
        } else {
            e4 = 0;
            f3 = q;
        }
    }

    return static_cast<uint8_t>((e4 << 3) | f3);
}

__device__ __forceinline__
float e4m3_to_float(uint8_t q) {
    uint32_t sign = (q >> 7) & 1u;
    uint32_t exp = (q >> 3) & 0xfu;
    uint32_t frac = q & 0x7u;

    float value;
    if (exp == 0) {
        value = frac ? ldexpf(static_cast<float>(frac) / 8.0f, -6) : 0.0f;
    } else if (exp == 0xfu && frac == 0x7u) {
        value = NAN;
    } else {
        float mantissa = 1.0f + static_cast<float>(frac) / 8.0f;
        value = ldexpf(mantissa, static_cast<int>(exp) - 7);
    }

    return sign ? -value : value;
}

// 输入转换与向量化加载。
template <typename T>
struct InputConvert;

template <>
struct InputConvert<float> {

    __device__ __forceinline__ static float load(const float* ptr, int i) {
        return ptr[i];
    }

    __device__ __forceinline__ static void load2(
        const float* ptr,
        int pair_idx,
        float& x0,
        float& x1) {

        const float2 v =
            reinterpret_cast<const float2*>(ptr)[pair_idx];

        x0 = v.x;
        x1 = v.y;
    }

    __device__ __forceinline__ static void load4(
        const float* ptr,
        int group_idx,
        float& x0,
        float& x1,
        float& x2,
        float& x3) {

        const float4 v =
            reinterpret_cast<const float4*>(ptr)[group_idx];

        x0 = v.x;
        x1 = v.y;
        x2 = v.z;
        x3 = v.w;
    }

    __device__ __forceinline__ static void load8(
        const float* ptr,
        int group_idx,
        float& x0,
        float& x1,
        float& x2,
        float& x3,
        float& x4,
        float& x5,
        float& x6,
        float& x7) {

        const float4* p4 = reinterpret_cast<const float4*>(ptr);
        const int base = group_idx << 1;
        const float4 v0 = p4[base];
        const float4 v1 = p4[base + 1];

        x0 = v0.x;
        x1 = v0.y;
        x2 = v0.z;
        x3 = v0.w;
        x4 = v1.x;
        x5 = v1.y;
        x6 = v1.z;
        x7 = v1.w;
    }
};

template <>
struct InputConvert<uint16_t> {

    __device__ __forceinline__ static float load(const uint16_t* ptr, int i) {
        return fp16_bits_to_float(ptr[i]);
    }

    __device__ __forceinline__ static void load2(
        const uint16_t* ptr,
        int pair_idx,
        float& x0,
        float& x1) {

        const ushort2 v =
            reinterpret_cast<const ushort2*>(ptr)[pair_idx];

        x0 = fp16_bits_to_float(v.x);
        x1 = fp16_bits_to_float(v.y);
    }

    __device__ __forceinline__ static void load4(
        const uint16_t* ptr,
        int group_idx,
        float& x0,
        float& x1,
        float& x2,
        float& x3) {

        const ushort4 v =
            reinterpret_cast<const ushort4*>(ptr)[group_idx];

        x0 = fp16_bits_to_float(v.x);
        x1 = fp16_bits_to_float(v.y);
        x2 = fp16_bits_to_float(v.z);
        x3 = fp16_bits_to_float(v.w);
    }

    __device__ __forceinline__ static void load8(
        const uint16_t* ptr,
        int group_idx,
        float& x0,
        float& x1,
        float& x2,
        float& x3,
        float& x4,
        float& x5,
        float& x6,
        float& x7) {

        const uint4 v =
            reinterpret_cast<const uint4*>(ptr)[group_idx];

        x0 = fp16_bits_to_float(static_cast<uint16_t>(v.x & 0xffffu));
        x1 = fp16_bits_to_float(static_cast<uint16_t>(v.x >> 16));
        x2 = fp16_bits_to_float(static_cast<uint16_t>(v.y & 0xffffu));
        x3 = fp16_bits_to_float(static_cast<uint16_t>(v.y >> 16));
        x4 = fp16_bits_to_float(static_cast<uint16_t>(v.z & 0xffffu));
        x5 = fp16_bits_to_float(static_cast<uint16_t>(v.z >> 16));
        x6 = fp16_bits_to_float(static_cast<uint16_t>(v.w & 0xffffu));
        x7 = fp16_bits_to_float(static_cast<uint16_t>(v.w >> 16));
    }
};

// 不同输出类型共用同一反量化 kernel。
struct Fp32Output {
    using Storage = float;

    __device__ __forceinline__ static void store(Storage* ptr, int i, float x) {
        ptr[i] = x;
    }

    __device__ __forceinline__ static void store2(
        Storage* ptr,
        int pair_idx,
        float x0,
        float x1) {

        reinterpret_cast<float2*>(ptr)[pair_idx] =
            make_float2(x0, x1);
    }
};

struct Fp16Output {
    using Storage = uint16_t;

    __device__ __forceinline__ static void store(Storage* ptr, int i, float x) {
        ptr[i] = float_to_fp16_bits_rne(x);
    }

    __device__ __forceinline__ static void store2(
        Storage* ptr,
        int pair_idx,
        float x0,
        float x1) {

        const uint16_t h0 = float_to_fp16_bits_rne(x0);
        const uint16_t h1 = float_to_fp16_bits_rne(x1);

        reinterpret_cast<ushort2*>(ptr)[pair_idx] =
            make_ushort2(h0, h1);
    }
};

struct Bf16Output {
    using Storage = uint16_t;

    __device__ __forceinline__ static void store(Storage* ptr, int i, float x) {
        ptr[i] = float_to_bf16_bits_rne(x);
    }

    __device__ __forceinline__ static void store2(
        Storage* ptr,
        int pair_idx,
        float x0,
        float x1) {

        const uint16_t b0 = float_to_bf16_bits_rne(x0);
        const uint16_t b1 = float_to_bf16_bits_rne(x1);

        reinterpret_cast<ushort2*>(ptr)[pair_idx] =
            make_ushort2(b0, b1);
    }
};

// Warp 内最大值归约。
__device__ __forceinline__
float warp_reduce_max(float v) {
    constexpr unsigned mask = 0xffffffffu;

    v = fmaxf(v, __shfl_down_sync(mask, v, 16));
    v = fmaxf(v, __shfl_down_sync(mask, v, 8));
    v = fmaxf(v, __shfl_down_sync(mask, v, 4));
    v = fmaxf(v, __shfl_down_sync(mask, v, 2));
    v = fmaxf(v, __shfl_down_sync(mask, v, 1));

    return v;
}

// 每线程先求局部最大值，再进行 warp / CTA 归约。
template <typename InputT>
__global__
void global_amax_kernel_v4(
    const InputT* __restrict__ input,
    uint32_t* __restrict__ global_amax_bits,
    int n) {

    extern __shared__ float warp_max[];

    const int tid =
        blockIdx.x * blockDim.x + threadIdx.x;

    const int stride =
        blockDim.x * gridDim.x;

    const int lane =
        threadIdx.x & 31;

    const int warp_id =
        threadIdx.x >> 5;

    constexpr int kWarpSize = 32;

    const int num_warps =
        blockDim.x / kWarpSize;

    float local_max = 0.0f;

    for (int i = tid; i < n; i += stride) {
        float x =
            InputConvert<InputT>::load(input, i);

        if (isfinite(x)) {
            local_max =
                fmaxf(local_max, fabsf(x));
        }
    }

    local_max =
        warp_reduce_max(local_max);

    if (lane == 0) {
        warp_max[warp_id] = local_max;
    }

    __syncthreads();

    if (warp_id == 0) {
        float block_max =
            (lane < num_warps)
                ? warp_max[lane]
                : 0.0f;

        block_max =
            warp_reduce_max(block_max);

        if (lane == 0) {
            atomicMax(
                global_amax_bits,
                __float_as_uint(block_max));
        }
    }
}

// global_scale = global_amax / (448 × 6)。
__global__
void compute_global_scale_kernel(
    const uint32_t* __restrict__ global_amax_bits,
    float* __restrict__ global_scale) {

    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    float amax = __uint_as_float(*global_amax_bits);
    if (amax == 0.0f) {
        *global_scale = 1.0f;
        return;
    }

    float scale = amax / 2688.0f;
    if (scale == 0.0f) {
        scale = 0x1p-149f;
    }
    *global_scale = scale;
}

__device__ __forceinline__
float subgroup2_reduce_max(float v, unsigned mask) {
    v = fmaxf(v, __shfl_down_sync(mask, v, 1, 2));
    return v;
}

// V6：每线程处理 8 个输入值，每 2 个线程负责一个 16 元素 block。
template <typename InputT>
__global__
void quantize_kernel_v6(
    const InputT* __restrict__ input,
    uint8_t* __restrict__ packed_output,
    uint8_t* __restrict__ block_scales,
    const float* __restrict__ global_scale_ptr,
    int n_padded) {

    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int num_threads = n_padded >> 3;
    if (tid >= num_threads) return;

    const int lane = threadIdx.x & 31;
    const int subgroup_id = lane >> 1;
    const int subgroup_lane = lane & 1;

    const int nv_block_id = tid >> 1;

    float x0, x1, x2, x3;
    float x4, x5, x6, x7;
    InputConvert<InputT>::load8(
        input, tid,
        x0, x1, x2, x3,
        x4, x5, x6, x7);

    // block scale 只由有限值决定。
    const float a0 = isfinite(x0) ? fabsf(x0) : 0.0f;
    const float a1 = isfinite(x1) ? fabsf(x1) : 0.0f;
    const float a2 = isfinite(x2) ? fabsf(x2) : 0.0f;
    const float a3 = isfinite(x3) ? fabsf(x3) : 0.0f;
    const float a4 = isfinite(x4) ? fabsf(x4) : 0.0f;
    const float a5 = isfinite(x5) ? fabsf(x5) : 0.0f;
    const float a6 = isfinite(x6) ? fabsf(x6) : 0.0f;
    const float a7 = isfinite(x7) ? fabsf(x7) : 0.0f;

    const float m01 = fmaxf(a0, a1);
    const float m23 = fmaxf(a2, a3);
    const float m45 = fmaxf(a4, a5);
    const float m67 = fmaxf(a6, a7);
    float local_max = fmaxf(fmaxf(m01, m23), fmaxf(m45, m67));

    const unsigned subgroup_mask = 0x3u << (subgroup_id * 2);
    const float block_amax = subgroup2_reduce_max(local_max, subgroup_mask);
    const float global_scale = *global_scale_ptr;

    uint8_t block_scale_code = 0;
    float inv_scale = 0.0f;

    if (subgroup_lane == 0) {
        if (block_amax != 0.0f) {
            float ideal_block_scale = (block_amax / global_scale) / 6.0f;
            ideal_block_scale = fminf(ideal_block_scale, 448.0f);

            block_scale_code = float_to_e4m3_round_up(ideal_block_scale);
            const float block_scale = e4m3_to_float(block_scale_code);

            const float combined_scale = global_scale * block_scale;
            inv_scale = 1.0f / combined_scale;
        }

        block_scales[nv_block_id] = block_scale_code;
    }

    inv_scale = __shfl_sync(subgroup_mask, inv_scale, 0, 2);

    uint8_t q0 = 0;
    uint8_t q1 = 0;
    uint8_t q2 = 0;
    uint8_t q3 = 0;
    uint8_t q4 = 0;
    uint8_t q5 = 0;
    uint8_t q6 = 0;
    uint8_t q7 = 0;

    if (!isfinite(x0)) q0 = float_to_e2m1(x0);
    else if (inv_scale != 0.0f) q0 = float_to_e2m1(x0 * inv_scale);

    if (!isfinite(x1)) q1 = float_to_e2m1(x1);
    else if (inv_scale != 0.0f) q1 = float_to_e2m1(x1 * inv_scale);

    if (!isfinite(x2)) q2 = float_to_e2m1(x2);
    else if (inv_scale != 0.0f) q2 = float_to_e2m1(x2 * inv_scale);

    if (!isfinite(x3)) q3 = float_to_e2m1(x3);
    else if (inv_scale != 0.0f) q3 = float_to_e2m1(x3 * inv_scale);

    if (!isfinite(x4)) q4 = float_to_e2m1(x4);
    else if (inv_scale != 0.0f) q4 = float_to_e2m1(x4 * inv_scale);

    if (!isfinite(x5)) q5 = float_to_e2m1(x5);
    else if (inv_scale != 0.0f) q5 = float_to_e2m1(x5 * inv_scale);

    if (!isfinite(x6)) q6 = float_to_e2m1(x6);
    else if (inv_scale != 0.0f) q6 = float_to_e2m1(x6 * inv_scale);

    if (!isfinite(x7)) q7 = float_to_e2m1(x7);
    else if (inv_scale != 0.0f) q7 = float_to_e2m1(x7 * inv_scale);

    // 8 个 E2M1 值打包为 4 个连续字节。
    const uint8_t packed0 = static_cast<uint8_t>(((q1 & 0x0fu) << 4) | (q0 & 0x0fu));
    const uint8_t packed1 = static_cast<uint8_t>(((q3 & 0x0fu) << 4) | (q2 & 0x0fu));
    const uint8_t packed2 = static_cast<uint8_t>(((q5 & 0x0fu) << 4) | (q4 & 0x0fu));
    const uint8_t packed3 = static_cast<uint8_t>(((q7 & 0x0fu) << 4) | (q6 & 0x0fu));

    reinterpret_cast<uchar4*>(packed_output)[tid] =
        make_uchar4(packed0, packed1, packed2, packed3);
}

// V4：每线程解包并反量化两个连续值。
template <typename OutputPolicy>
__global__
void dequantize_kernel_v4(
    const uint8_t* __restrict__ packed_input,
    const uint8_t* __restrict__ block_scales,
    const float* __restrict__ global_scale_ptr,
    typename OutputPolicy::Storage* __restrict__ output,
    int n_padded) {

    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int num_packed = n_padded >> 1;
    if (tid >= num_packed) return;

    const int nv_block_id = tid >> 3;

    const uint8_t packed = packed_input[tid];
    const uint8_t q0 = packed & 0x0fu;
    const uint8_t q1 = (packed >> 4) & 0x0fu;

    const float y0 = e2m1_to_float_v3(q0);
    const float y1 = e2m1_to_float_v3(q1);

    const float block_scale =
        e4m3_scale_to_float_v4(block_scales[nv_block_id]);

    const float global_scale = *global_scale_ptr;
    const float scale = block_scale * global_scale;

    const float out0 = y0 * scale;
    const float out1 = y1 * scale;

    OutputPolicy::store2(
        output,
        tid,
        out0,
        out1);
}

// 量化启动逻辑；workspace 可由调用方复用。
template <typename InputT>
cudaError_t quantize_impl(
    const InputT* d_input,
    int n_padded,
    uint8_t* d_packed,
    uint8_t* d_block_scales,
    float* d_global_scale,
    cudaStream_t stream,
    uint32_t* d_global_amax_bits_workspace) {

    if (!d_input || !d_packed || !d_block_scales || !d_global_scale) {
        return cudaErrorInvalidValue;
    }
    if (n_padded <= 0 || (n_padded % kWarpValues) != 0) {
        return cudaErrorInvalidValue;
    }

    // 优先复用持久化 workspace。
    uint32_t* d_global_amax_bits = d_global_amax_bits_workspace;
    bool owns_workspace = false;

    if (!d_global_amax_bits) {
        cudaError_t alloc_err = cudaMalloc(
            reinterpret_cast<void**>(&d_global_amax_bits),
            sizeof(uint32_t));
        if (alloc_err != cudaSuccess) return alloc_err;
        owns_workspace = true;
    }

    cudaError_t err = cudaMemsetAsync(
        d_global_amax_bits,
        0,
        sizeof(uint32_t),
        stream);
    if (err != cudaSuccess) {
        if (owns_workspace) cudaFree(d_global_amax_bits);
        return err;
    }

    constexpr int kReduceThreads = 256;
    constexpr int kMaxReduceBlocks = 864;
    constexpr int kWarpsPerBlock = kReduceThreads / 32;

    int reduce_blocks =
    (n_padded + kReduceThreads - 1) / kReduceThreads;

    if (reduce_blocks > kMaxReduceBlocks)
        reduce_blocks = kMaxReduceBlocks;

    global_amax_kernel_v4<InputT><<<
        reduce_blocks,
        kReduceThreads,
        kWarpsPerBlock * sizeof(float),
        stream>>>(
            d_input,
            d_global_amax_bits,
            n_padded);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        if (owns_workspace) cudaFree(d_global_amax_bits);
        return err;
    }

    compute_global_scale_kernel<<<1, 1, 0, stream>>>(
        d_global_amax_bits,
        d_global_scale);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        if (owns_workspace) cudaFree(d_global_amax_bits);
        return err;
    }

    constexpr int kThreads = 256;
    int num_threads = n_padded / 8;
    int blocks = (num_threads + kThreads - 1) / kThreads;

    quantize_kernel_v6<InputT><<<blocks, kThreads, 0, stream>>>(
        d_input,
        d_packed,
        d_block_scales,
        d_global_scale,
        n_padded);

    err = cudaGetLastError();

    if (owns_workspace) {
        cudaError_t free_err = cudaFree(d_global_amax_bits);
        if (err == cudaSuccess) err = free_err;
    }

    return err;
}

// 反量化启动逻辑。
template <typename OutputPolicy>
cudaError_t dequantize_impl(
    const uint8_t* d_packed,
    const uint8_t* d_block_scales,
    const float* d_global_scale,
    int n_padded,
    typename OutputPolicy::Storage* d_output,
    cudaStream_t stream) {

    if (!d_packed || !d_block_scales || !d_global_scale || !d_output) {
        return cudaErrorInvalidValue;
    }
    if (n_padded <= 0 || (n_padded % kWarpValues) != 0) {
        return cudaErrorInvalidValue;
    }

    constexpr int kThreads = 256;
    int num_threads = n_padded / 2;
    int blocks = (num_threads + kThreads - 1) / kThreads;

    dequantize_kernel_v4<OutputPolicy><<<blocks, kThreads, 0, stream>>>(
        d_packed,
        d_block_scales,
        d_global_scale,
        d_output,
        n_padded);

    return cudaGetLastError();
}

}  // 匿名命名空间

cudaError_t quantize(
    const float* d_input,
    int n_padded,
    uint8_t* d_packed,
    uint8_t* d_block_scales,
    float* d_global_scale,
    cudaStream_t stream,
    uint32_t* d_global_amax_bits_workspace) {

    return quantize_impl(
        d_input,
        n_padded,
        d_packed,
        d_block_scales,
        d_global_scale,
        stream,
        d_global_amax_bits_workspace);
}

cudaError_t quantize_fp16(
    const uint16_t* d_input_fp16,
    int n_padded,
    uint8_t* d_packed,
    uint8_t* d_block_scales,
    float* d_global_scale,
    cudaStream_t stream,
    uint32_t* d_global_amax_bits_workspace) {

    return quantize_impl(
        d_input_fp16,
        n_padded,
        d_packed,
        d_block_scales,
        d_global_scale,
        stream,
        d_global_amax_bits_workspace);
}

cudaError_t dequantize(
    const uint8_t* d_packed,
    const uint8_t* d_block_scales,
    const float* d_global_scale,
    int n_padded,
    float* d_output,
    cudaStream_t stream) {

    return dequantize_impl<Fp32Output>(
        d_packed,
        d_block_scales,
        d_global_scale,
        n_padded,
        d_output,
        stream);
}

cudaError_t dequantize_fp16(
    const uint8_t* d_packed,
    const uint8_t* d_block_scales,
    const float* d_global_scale,
    int n_padded,
    uint16_t* d_output_fp16,
    cudaStream_t stream) {

    return dequantize_impl<Fp16Output>(
        d_packed,
        d_block_scales,
        d_global_scale,
        n_padded,
        d_output_fp16,
        stream);
}

cudaError_t dequantize_bf16(
    const uint8_t* d_packed,
    const uint8_t* d_block_scales,
    const float* d_global_scale,
    int n_padded,
    uint16_t* d_output_bf16,
    cudaStream_t stream) {

    return dequantize_impl<Bf16Output>(
        d_packed,
        d_block_scales,
        d_global_scale,
        n_padded,
        d_output_bf16,
        stream);
}

}  // namespace nvfp4
