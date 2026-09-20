// =============================================================================
//  vec_io.cuh -- 设备侧向量化装载/存储与元素类型 traits
//
//  量化/反量化 kernel 统一按「每线程处理 4 个连续元素」组织:
//    * fp32 输入 -> float4  (16B)
//    * fp16 输入 -> 2 x half2 (8B)
//    * fp16/bf16 输出 -> 4 个 16bit 编码打包成 uint2 (8B)
//  这样可以保证完全合并的访存,同时 4 是 16/32 的约数,与 block_size 对齐.
// =============================================================================
#pragma once

#include <cuda_runtime.h>

#include "lpq/formats.hpp"
#include "lpq/platform.hpp"

// 低精度类型头文件. 除英伟达 nvcc 外的平台若没有同名兼容头, 用
// -DLPQ_FP16_HEADER=<musa_fp16.h> / -DLPQ_BF16_HEADER=<...> 指定平台头,
// 只要其中同样定义了 __half / __half2 / __nv_bfloat16 即可.
#if defined(LPQ_FP16_HEADER)
#include LPQ_FP16_HEADER
#else
#include <cuda_fp16.h>
#endif

#if defined(LPQ_BF16_HEADER)
#include LPQ_BF16_HEADER
#else
#include <cuda_bf16.h>
#endif

namespace lpq {

// -----------------------------------------------------------------------------
// 输入类型(fp32 / fp16 / bf16)
// -----------------------------------------------------------------------------
template <class T>
struct InTraits;

template <>
struct InTraits<float> {
    static constexpr int kVec = 4;
    LPQ_DEV static void load4(const float* p, float v[4]) {
        const float4 t = *reinterpret_cast<const float4*>(p);
        v[0] = t.x;
        v[1] = t.y;
        v[2] = t.z;
        v[3] = t.w;
    }
    LPQ_DEV static float to_float(float x) { return x; }
};

template <>
struct InTraits<__nv_bfloat16> {
    static constexpr int kVec = 4;
    LPQ_DEV static void load4(const __nv_bfloat16* p, float v[4]) {
        const __nv_bfloat162* h2 = reinterpret_cast<const __nv_bfloat162*>(p);
        const __nv_bfloat162 a = h2[0];
        const __nv_bfloat162 b = h2[1];
        v[0] = __low2float(a);
        v[1] = __high2float(a);
        v[2] = __low2float(b);
        v[3] = __high2float(b);
    }
    // 不用 __bfloat162float: 它在 MUSA 上会把 bf16 次正规输入刷成 0, 并把 NaN
    // 规范化成 0x7FC00000; 走位模式解码则与 CPU 参考逐位一致(fmt_to_f32<FmtBF16>
    // 本身就是纯移位实现).
    LPQ_DEV static float to_float(__nv_bfloat16 x) {
        return f32_from_bf16_bits((uint16_t)__bfloat16_as_ushort(x));
    }
};

template <>
struct InTraits<__half> {
    static constexpr int kVec = 4;
    LPQ_DEV static void load4(const __half* p, float v[4]) {
        const __half2* h2 = reinterpret_cast<const __half2*>(p);
        const __half2 a = h2[0];
        const __half2 b = h2[1];
        v[0] = __low2float(a);
        v[1] = __high2float(a);
        v[2] = __low2float(b);
        v[3] = __high2float(b);
    }
    LPQ_DEV static float to_float(__half x) { return __half2float(x); }
};

// -----------------------------------------------------------------------------
// fp16/bf16 输出编码的快速路径
//
// f32_bits_to_fmt_mag<> 是通用的「位域提取 + RNE + 饱和」实现, 语义上要同时覆盖
// 次正规规格化(一个 while 循环)、上溢饱和、NaN 传播, 因此在 dequant 这种每个元素
// 都要编码一次的场合相当贵(十几条指令 + 若干分支).
//
// 但如果结果本身就落在目标格式的**有限可表示范围内**, 硬件转换指令的 RNE 结果与
// 通用实现逐位相同(两者都是"舍入到 11/8 bit 有效位, ties-to-even", 次正规的进位
// 规则也一样). 所以这里只对需要饱和 / NaN 传播的极少数输入回退到通用实现.
// -----------------------------------------------------------------------------
LPQ_DEV uint32_t f32_to_f16_bits(float x) {
    const uint32_t b = f32_to_bits(x) & 0x7FFFFFFFu;
    if (b < 0x477FE000u)  // |x| < 65504: 结果必为有限 fp16
        return (uint32_t)__half_as_ushort(__float2half_rn(x));
    return f32_to_fmt<FmtF16>(x);
}

LPQ_DEV uint32_t f32_to_bf16_bits(float x) {
    const uint32_t b = f32_to_bits(x) & 0x7FFFFFFFu;
    if (b < 0x7F7F0000u)  // |x| < 3.39e38: 结果必为有限 bf16
        return (uint32_t)__bfloat16_as_ushort(__float2bfloat16_rn(x));
    return f32_to_fmt<FmtBF16>(x);
}

// -----------------------------------------------------------------------------
// 输出类型(fp16 / bf16 / fp32)
//   16bit 输出同样走 f32_to_f16_bits/f32_to_bf16_bits(内含与 CPU 参考实现逐位
//   一致的通用实现作为回退),保证与 CPU 参考实现逐位一致
// -----------------------------------------------------------------------------
template <class T>
struct OutTraits;

template <>
struct OutTraits<float> {
    static constexpr int kVec = 4;
    LPQ_DEV static void store4(float* p, const float v[4]) {
        *reinterpret_cast<float4*>(p) = make_float4(v[0], v[1], v[2], v[3]);
    }
    LPQ_DEV static float to_float(float x) { return x; }
    LPQ_DEV static float from_float(float v) { return v; }
};

template <>
struct OutTraits<__half> {
    static constexpr int kVec = 4;
    LPQ_DEV static void store4(__half* p, const float v[4]) {
        const uint32_t lo = f32_to_f16_bits(v[0]) | (f32_to_f16_bits(v[1]) << 16);
        const uint32_t hi = f32_to_f16_bits(v[2]) | (f32_to_f16_bits(v[3]) << 16);
        *reinterpret_cast<uint2*>(p) = make_uint2(lo, hi);
    }
    LPQ_DEV static float to_float(__half x) { return __half2float(x); }
    LPQ_DEV static __half from_float(float v) {
        return __ushort_as_half((unsigned short)f32_to_f16_bits(v));
    }
};

template <>
struct OutTraits<__nv_bfloat16> {
    static constexpr int kVec = 4;
    LPQ_DEV static void store4(__nv_bfloat16* p, const float v[4]) {
        const uint32_t lo = f32_to_bf16_bits(v[0]) | (f32_to_bf16_bits(v[1]) << 16);
        const uint32_t hi = f32_to_bf16_bits(v[2]) | (f32_to_bf16_bits(v[3]) << 16);
        *reinterpret_cast<uint2*>(p) = make_uint2(lo, hi);
    }
    LPQ_DEV static float to_float(__nv_bfloat16 x) {
        return f32_from_bf16_bits((uint16_t)__bfloat16_as_ushort(x));
    }
    LPQ_DEV static __nv_bfloat16 from_float(float v) {
        return __ushort_as_bfloat16((unsigned short)f32_to_bf16_bits(v));
    }
};

// -----------------------------------------------------------------------------
// 4 个元素是否落在同一个缩放块内(决定能否只做一次 scale 查表)
// -----------------------------------------------------------------------------
LPQ_DEV bool same_block(int64_t i0, int64_t i1, int64_t block_size) {
    return (i0 / block_size) == (i1 / block_size);
}

// 元素下标 -> 缩放块下标; shift >= 0 表示 block_size 是 2 的幂, 可用移位代替除法
LPQ_DEV int64_t block_index(int64_t i, int64_t block_size, int shift) {
    return (shift >= 0) ? (i >> shift) : (i / block_size);
}

}  // namespace lpq