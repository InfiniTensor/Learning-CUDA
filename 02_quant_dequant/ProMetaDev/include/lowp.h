// lowp.h - 低精度浮点格式编解码核心 (MXFP8 / NVFP4)
//
// 全部为纯软件位运算实现，不依赖 Hopper/Blackwell/Ada 的任何新硬件指令，
// 可在任意支持 CUDA 的 GPU 上运行（也可用于纯 CPU 编译）。
//
// 支持的数值格式：
//   E4M3 : 1 符号 + 4 指数(bias 7)  + 3 尾数，max = 448
//   E5M2 : 1 符号 + 5 指数(bias 15) + 2 尾数，max = 57344
//   E2M1 : 1 符号 + 2 指数(bias 1)  + 1 尾数，max = 6      (FP4)
//   E8M0 : 仅 8 位指数(bias 127)，value = 2^(e-127)        (MX 共享缩放因子)
#pragma once
#include <cstdint>
#include <cmath>
#include <cstring>
#include <string>

#ifdef __CUDACC__
#define LP_HD __host__ __device__
#else
#define LP_HD
#endif

namespace lowp {

// ======================= 枚举定义 =======================
enum class LowFormat { MXFP8 = 0, NVFP4 = 1 };
enum class ElemFmt { E4M3 = 0, E5M2 = 1 };
enum class ScaleMode { TENSOR = 0, BLOCK = 1 };
enum class RoundMode { NEAREST = 0, STOCHASTIC = 1 };
enum class OutDtype { FP16 = 0, BF16 = 1, FP32 = 2 };

LP_HD inline const char* format_name(LowFormat f) {
    return f == LowFormat::MXFP8 ? "mxfp8" : "nvfp4";
}
LP_HD inline const char* elem_name(ElemFmt e) {
    return e == ElemFmt::E4M3 ? "e4m3" : "e5m2";
}

// ======================= 基础位操作 =======================
LP_HD inline float bits_to_f32(uint32_t b) {
#ifdef __CUDA_ARCH__
    return __uint_as_float(b);
#else
    float f;
    std::memcpy(&f, &b, 4);
    return f;
#endif
}
LP_HD inline uint32_t f32_to_bits(float f) {
#ifdef __CUDA_ARCH__
    return __float_as_uint(f);
#else
    uint32_t b;
    std::memcpy(&b, &f, 4);
    return b;
#endif
}
// 四舍五入到最近偶数 (round-to-nearest-even)
LP_HD inline int32_t rn_int(float v) {
#ifdef __CUDA_ARCH__
    return __float2int_rn(v);
#else
    return (int32_t)nearbyintf(v);
#endif
}
// 统一的取整入口：mode=0 最近偶数；mode=1 随机舍入(stochastic)
// rnd 为 [0,1) 均匀随机数，仅在随机舍入时使用
LP_HD inline int32_t round_to_int(float v, int mode, float rnd) {
    if (mode == 0)
        return rn_int(v);
    float fl = floorf(v);
    return (int32_t)fl + ((rnd < (v - fl)) ? 1 : 0);
}

// a * 2^k 的快速精确实现（当结果仍为正规数时与乘法完全等价）。
// 直接在指数域做定点加法，避免了昂贵的软件 ldexpf 调用。
// 无符号加法下的模 2^32 溢出恰好等价于有符号的 k 偏移。
LP_HD inline float scale_pow2(float a, int k) {
    return bits_to_f32(f32_to_bits(a) + ((uint32_t)k << 23));
}

// ======================= 随机数 (随机舍入用) =======================
LP_HD inline uint32_t rng_hash(uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}
LP_HD inline float rng_uniform(uint32_t seed, uint64_t idx) {
    uint32_t h = rng_hash((uint32_t)idx ^ (seed * 0x9E3779B9u));
    return (float)(h >> 8) * (1.0f / 16777216.0f); // [0,1)
}

// ======================= E4M3 =======================
// 编码值 -> float
LP_HD inline float e4m3_to_f32(uint8_t c) {
    uint32_t s = (c >> 7) & 1u;
    uint32_t E = (c >> 3) & 0xFu;
    uint32_t M = c & 0x7u;
    float v = (E == 0) ? ldexpf((float)M, -9)                         // 次正规数, 步长 2^-9
                       : bits_to_f32(((E + 120u) << 23) | (M << 20)); // (8+M) * 2^(E-10)
    return s ? -v : v;
}
// float -> 编码值 (饱和, 支持最近偶数/随机舍入)
LP_HD inline uint8_t f32_to_e4m3(float x, int mode, float rnd) {
    uint32_t u = f32_to_bits(x);
    uint32_t s = u >> 31;
    float a = bits_to_f32(u & 0x7fffffffu);
    if (a != a)
        return (uint8_t)((s << 7) | 0x7Fu); // NaN
    if (a == 0.0f)
        return (uint8_t)(s << 7); // ±0
    if ((u & 0x7fffffffu) >= 0x7f800000u || a >= 448.0f)
        return (uint8_t)((s << 7) | 0x7Eu); // 饱和到 448
    int32_t e = (int32_t)((u >> 23) & 0xFFu) - 127;
    uint8_t mag;
    if (e < -6) {
        // 次正规区
        int32_t q = round_to_int(a * 512.0f, mode, rnd);
        mag = (q >= 8) ? 0x08 : (uint8_t)q;
    } else {
        float scaled = scale_pow2(a, -(e - 3)); // 归一化到 [8,16)，结果必为正规数
        int32_t R = round_to_int(scaled, mode, rnd);
        if (R >= 16) {
            R = 8;
            e += 1;
        } // 进位
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
LP_HD inline float e5m2_to_f32(uint8_t c) {
    uint32_t s = (c >> 7) & 1u;
    uint32_t E = (c >> 2) & 0x1Fu;
    uint32_t M = c & 0x3u;
    if (E == 31)
        return 0.0f;                                                  // inf/NaN 容错
    float v = (E == 0) ? ldexpf((float)M, -16)                        // 步长 2^-16
                       : bits_to_f32(((E + 112u) << 23) | (M << 21)); // (4+M) * 2^(E-17)
    return s ? -v : v;
}
LP_HD inline uint8_t f32_to_e5m2(float x, int mode, float rnd) {
    uint32_t u = f32_to_bits(x);
    uint32_t s = u >> 31;
    float a = bits_to_f32(u & 0x7fffffffu);
    if (a != a)
        return (uint8_t)((s << 7) | 0x7Fu);
    if (a == 0.0f)
        return (uint8_t)(s << 7);
    if ((u & 0x7fffffffu) >= 0x7f800000u || a >= 57344.0f)
        return (uint8_t)((s << 7) | 0x7Bu); // 饱和
    int32_t e = (int32_t)((u >> 23) & 0xFFu) - 127;
    uint8_t mag;
    if (e < -14) {
        int32_t q = round_to_int(a * 65536.0f, mode, rnd);
        mag = (q >= 4) ? 0x04 : (uint8_t)q;
    } else {
        float scaled = scale_pow2(a, -(e - 2)); // 归一化到 [4,8)，结果必为正规数
        int32_t R = round_to_int(scaled, mode, rnd);
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

// ======================= E2M1 (FP4) =======================
// 3 位幅值编码 -> float：{0, 0.5, 1, 1.5, 2, 3, 4, 6}
LP_HD inline float e2m1_mag_to_f32(uint32_t m) {
    switch (m & 7u) {
    case 0:
        return 0.0f;
    case 1:
        return 0.5f;
    case 2:
        return 1.0f;
    case 3:
        return 1.5f;
    case 4:
        return 2.0f;
    case 5:
        return 3.0f;
    case 6:
        return 4.0f;
    default:
        return 6.0f;
    }
}
LP_HD inline float e2m1_to_f32(uint8_t c) {
    float v = e2m1_mag_to_f32(c & 7u);
    return (c & 8u) ? -v : v;
}
// 幅值编码：最近偶数 / 随机舍入
LP_HD inline uint32_t e2m1_encode_mag(float a, int mode, float rnd) {
    if (mode == 0) {
        // 阈值取相邻可表示值的中点；中点处取编码为偶数的一侧(round-half-to-even)
        if (a <= 0.25f)
            return 0;
        if (a < 0.75f)
            return 1;
        if (a <= 1.25f)
            return 2;
        if (a < 1.75f)
            return 3;
        if (a <= 2.5f)
            return 4;
        if (a < 3.5f)
            return 5;
        if (a <= 5.0f)
            return 6;
        return 7;
    }
    if (a >= 6.0f)
        return 7; // 饱和
    float lo, hi;
    int ilo; // 找到相邻区间后按概率取整
    if (a < 0.5f) {
        lo = 0.0f;
        hi = 0.5f;
        ilo = 0;
    } else if (a < 1.0f) {
        lo = 0.5f;
        hi = 1.0f;
        ilo = 1;
    } else if (a < 1.5f) {
        lo = 1.0f;
        hi = 1.5f;
        ilo = 2;
    } else if (a < 2.0f) {
        lo = 1.5f;
        hi = 2.0f;
        ilo = 3;
    } else if (a < 3.0f) {
        lo = 2.0f;
        hi = 3.0f;
        ilo = 4;
    } else if (a < 4.0f) {
        lo = 3.0f;
        hi = 4.0f;
        ilo = 5;
    } else {
        lo = 4.0f;
        hi = 6.0f;
        ilo = 6;
    }
    float p = (a - lo) / (hi - lo);
    return (uint32_t)((rnd < p) ? (ilo + 1) : ilo);
}
LP_HD inline uint8_t f32_to_e2m1(float x, int mode, float rnd) {
    uint32_t u = f32_to_bits(x);
    if (x != x)
        return 0x7u;
    uint32_t s = u >> 31;
    float a = bits_to_f32(u & 0x7fffffffu);
    uint32_t m = e2m1_encode_mag(a, mode, rnd);
    return (uint8_t)((s << 3) | m);
}

// ======================= E8M0 (MX 共享缩放) =======================
LP_HD inline float e8m0_to_f32(uint8_t c) {
    // 2^(c-127)：c∈[1,254] 时对应正规数，指数域恰好等于 c
    if (c == 0)
        return ldexpf(1.0f, -127); // 极小值路径，避免下溢成 0
    return bits_to_f32((uint32_t)c << 23);
}
LP_HD inline uint8_t e8m0_from_exp(int e) {
    if (e < -127)
        e = -127;
    if (e > 127)
        e = 127;
    return (uint8_t)(e + 127);
}
// 根据 block 内 amax 与元素格式最大可表示值，求共享指数 e，
// 使得 amax / 2^e <= maxv（不溢出），且尽量用满元素格式的动态范围。
// 采用 frexpf 做精确的指数/尾数拆分，保证 host 与 device 结果完全一致
// （若用 log2f+ceilf，宿主机 libm 与 CUDA libm 在末位可能不一致）。
LP_HD inline int mxfp8_scale_exp(float amax, float maxv) {
    if (!(amax > 0.0f))
        return -127;
    float r = amax / maxv; // IEEE 除法，两端完全一致
    int e2 = 0;
    float m = frexpf(r, &e2); // r = m * 2^e2, m ∈ [0.5, 1)
    // ceil(log2(r))：m == 0.5 时 r 恰为 2 的幂，需减一；否则即为 e2
    int e = (m == 0.5f) ? (e2 - 1) : e2;
    if (e < -127)
        e = -127;
    if (e > 127)
        e = 127;
    return e;
}
// 某元素格式的最大可表示幅值
LP_HD inline float elem_max_val(ElemFmt e) {
    return (e == ElemFmt::E4M3) ? 448.0f : 57344.0f;
}
// NVFP4 全局缩放：让最大的 block scale 恰好落满 E4M3 量程
LP_HD inline float nvfp4_global_scale(float gmax) {
    return (gmax > 0.0f) ? (gmax / (6.0f * 448.0f)) : 1.0f;
}

// FP4 元素格式固定最大值为 6
static const float NVFP4_ELEM_MAX = 6.0f;

} // namespace lowp
