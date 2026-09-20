// fp8.h - FP8 (E4M3 / E5M2) 与 MX 共享缩放因子 E8M0 的纯软件编解码
//
// 与题目要求一致：不依赖 Hopper/Blackwell/Ada 的新指令，也不依赖硬件原生 FP8 类型。
// 采用「块内求 amax -> 取 2 的整数次幂作为共享缩放（E8M0）-> 逐元素编码」的
// microscaling 方案，与 QuaRot / SpinQuant 等旋转后量化的做法一致。
#pragma once
#include <cstdint>
#include <cmath>
#include <cstring>

#ifdef __CUDACC__
#define HD_HD __host__ __device__
#else
#define HD_HD
#endif

namespace hd {

// ======================= 基础位操作 =======================
HD_HD inline float bits_to_f32(uint32_t b) {
#ifdef __CUDA_ARCH__
    return __uint_as_float(b);
#else
    float f;
    std::memcpy(&f, &b, 4);
    return f;
#endif
}
HD_HD inline uint32_t f32_to_bits(float f) {
#ifdef __CUDA_ARCH__
    return __float_as_uint(f);
#else
    uint32_t b;
    std::memcpy(&b, &f, 4);
    return b;
#endif
}
HD_HD inline int32_t rn_int(float v) {
#ifdef __CUDA_ARCH__
    return __float2int_rn(v);
#else
    return (int32_t)nearbyintf(v);
#endif
}
// a * 2^k 的精确位运算实现（结果仍为正规数时与乘法等价）
HD_HD inline float scale_pow2(float a, int k) {
    return bits_to_f32(f32_to_bits(a) + ((uint32_t)k << 23));
}

// ======================= E4M3 =======================
HD_HD inline float e4m3_to_f32(uint8_t c) {
    uint32_t s = (c >> 7) & 1u, E = (c >> 3) & 0xFu, M = c & 0x7u;
    float v = (E == 0) ? ldexpf((float)M, -9) : bits_to_f32(((E + 120u) << 23) | (M << 20));
    return s ? -v : v;
}
HD_HD inline uint8_t f32_to_e4m3(float x) {
    uint32_t u = f32_to_bits(x);
    uint32_t s = u >> 31;
    float a = bits_to_f32(u & 0x7fffffffu);
    if (a != a)
        return (uint8_t)((s << 7) | 0x7Fu);
    if (a == 0.0f)
        return (uint8_t)(s << 7);
    if ((u & 0x7fffffffu) >= 0x7f800000u || a >= 448.0f)
        return (uint8_t)((s << 7) | 0x7Eu); // 饱和到 448
    int32_t e = (int32_t)((u >> 23) & 0xFFu) - 127;
    uint8_t mag;
    if (e < -6) {
        int32_t q = rn_int(a * 512.0f);
        mag = (q >= 8) ? 0x08 : (uint8_t)q;
    } else {
        float scaled = scale_pow2(a, -(e - 3)); // 归一化到 [8,16)
        int32_t R = rn_int(scaled);
        if (R >= 16) {
            R = 8;
            e += 1;
        }
        int32_t E = e + 7, M = R - 8;
        if (E > 15)
            return (uint8_t)((s << 7) | 0x7Eu);
        if (E == 15 && M > 6)
            M = 6; // 0x7F 保留给 NaN
        mag = (uint8_t)((E << 3) | M);
    }
    return (uint8_t)((s << 7) | mag);
}

// ======================= E5M2 =======================
HD_HD inline float e5m2_to_f32(uint8_t c) {
    uint32_t s = (c >> 7) & 1u, E = (c >> 2) & 0x1Fu, M = c & 0x3u;
    if (E == 31)
        return 0.0f;
    float v = (E == 0) ? ldexpf((float)M, -16) : bits_to_f32(((E + 112u) << 23) | (M << 21));
    return s ? -v : v;
}
HD_HD inline uint8_t f32_to_e5m2(float x) {
    uint32_t u = f32_to_bits(x);
    uint32_t s = u >> 31;
    float a = bits_to_f32(u & 0x7fffffffu);
    if (a != a)
        return (uint8_t)((s << 7) | 0x7Fu);
    if (a == 0.0f)
        return (uint8_t)(s << 7);
    if ((u & 0x7fffffffu) >= 0x7f800000u || a >= 57344.0f)
        return (uint8_t)((s << 7) | 0x7Bu);
    int32_t e = (int32_t)((u >> 23) & 0xFFu) - 127;
    uint8_t mag;
    if (e < -14) {
        int32_t q = rn_int(a * 65536.0f);
        mag = (q >= 4) ? 0x04 : (uint8_t)q;
    } else {
        float scaled = scale_pow2(a, -(e - 2)); // 归一化到 [4,8)
        int32_t R = rn_int(scaled);
        if (R >= 8) {
            R = 4;
            e += 1;
        }
        int32_t E = e + 15, M = R - 4;
        if (E > 30)
            return (uint8_t)((s << 7) | 0x7Bu);
        mag = (uint8_t)((E << 2) | M);
    }
    return (uint8_t)((s << 7) | mag);
}

// ======================= E8M0 (MX 共享缩放) =======================
HD_HD inline float e8m0_to_f32(uint8_t c) {
    if (c == 0)
        return ldexpf(1.0f, -127);
    return bits_to_f32((uint32_t)c << 23); // 2^(c-127)
}
HD_HD inline uint8_t e8m0_from_exp(int e) {
    if (e < -127)
        e = -127;
    if (e > 127)
        e = 127;
    return (uint8_t)(e + 127);
}
// 元素格式的最大可表示幅值
HD_HD inline float elem_max_val(int is_e5m2) {
    return is_e5m2 ? 57344.0f : 448.0f;
}

// 由块内 amax 推导共享指数 e（保证不溢出且尽量用满量程）。
// 用 frexpf 做精确拆分，保证 host 与 device 结果完全一致。
HD_HD inline int mx_scale_exp(float amax, int is_e5m2) {
    if (!(amax > 0.0f))
        return -127;
    float r = amax / elem_max_val(is_e5m2);
    int e2 = 0;
    float m = frexpf(r, &e2);
    int e = (m == 0.5f) ? (e2 - 1) : e2; // ceil(log2(r))
    if (e < -127)
        e = -127;
    if (e > 127)
        e = 127;
    return e;
}

} // namespace hd
