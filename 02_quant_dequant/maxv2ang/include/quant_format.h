// ============================================================================
// 选题二「MXFP8 / NVFP4 低精度模拟与反量化」—— 格式语义（host/device 共用）
// ============================================================================
// 本头文件是设备编译唯一需要看到的格式层：元素编解码、缩放指数、舍入、FP16/BF16
// 位级转换、4bit 打包所需的位运算。
//
// 函数限定符统一走 QC_HD 宏，表访问器走 QT 宏 —— 在含 kernel 的 TU 里编译出
// host + device 两份（设备版被 kernel 调用，主机版被主机端参考实现调用，共用同一份
// 舍入语义所以可以逐字节对拍）；在纯主机 TU 里只编译主机版本。见下方开关说明。
//
// 主机侧的配置解析、文件 I/O、指标统计、主机参考实现见 quant_io.h（不进设备编译）。
// ============================================================================

#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>

#include "quant_tables.h"

// ============================================================================
// 编译阶段 / TU 相关的开关
// ============================================================================
// 本项目的编译单元分两类：
//   - src/quant_gpu.cu   含 kernel，定义 QC_DEVICE_TABLES —— 格式层编译成 host+device
//   - 其它（src/quant_io.cu、test/*）纯主机 —— 格式层只编译主机版本
//
// QT：表访问器取哪份表。两个条件缺一不可：
//       __CUDA_ARCH__    区分设备/主机编译阶段；
//       QC_DEVICE_TABLES 区分 TU（只有 src/quant_gpu.cu 定义了设备表）。
//     后者保证其它 TU 绝不引用设备表符号，因此不需要（也不该写）
//     extern __constant__ 声明 —— CUDA 的 __constant__ 是 per-TU 静态存储，
//     extern 会被 nvcc 当作静态定义并在该 TU 里造一份零初始化副本。
//
// QC_HD：函数限定符。必须整体切换而不是只切表访问器 —— nvcc 对 .cu 一定会跑
//       一遍设备编译，若此时格式层函数仍是 __host__ __device__ 而表访问器已
//       降级为主机函数，设备版本就会「从 __host__ __device__ 函数调用 __host__
//       函数」（warning #20011 / #20014）。一个 TU 要么整体有设备语义，要么整体没有。
// ============================================================================

#if defined(QC_DEVICE_TABLES) && defined(__CUDA_ARCH__)
#define QT(a) a##Dev
#else
#define QT(a) a##Host
#endif

#ifdef QC_DEVICE_TABLES
#define QC_HD __host__ __device__
#else
#define QC_HD
#endif

// 取 IEEE754 符号位。等价于 signbit()（对 -0.0 与带符号 NaN 同样为真），
// 但直接看最高位：不依赖 <cmath> 把 signbit 放在哪个命名空间，主机/设备一致。
QC_HD inline bool sign_bit(double x) {
    uint64_t b = 0;
    memcpy(&b, &x, sizeof(b));
    return (b >> 63) != 0;
}

// ---- 常量与枚举 ----

static const double E4M3_MAX = 448.0; // OCP E4M3 最大有限值
static const double E2M1_MAX = 6.0;   // E2M1 最大有限值

enum FmtId { FMT_E4M3 = 0, FMT_E5M2 = 1, FMT_E2M1 = 2 };
enum OutTypeId { OUT_FP32 = 0, OUT_FP16 = 1, OUT_BF16 = 2 };
enum RoundMode { MODE_NEAREST = 0, MODE_STOCHASTIC = 1 };

// ============================================================================
// 1. 表访问器：设备编译读 __constant__ 副本，主机读 host 副本
//    （主机代码不能直接访问 __constant__ 符号，故按 QT 切换）
// ============================================================================

QC_HD inline const double *e4m3_mags() {
    return QT(kE4M3Mags);
}
QC_HD inline const double *e5m2_mags() {
    return QT(kE5M2Mags);
}
QC_HD inline const double *e2m1_mags() {
    return QT(kE2M1Mags);
}
QC_HD inline const double *e4m3_decode() {
    return QT(kE4M3Decode);
}
QC_HD inline const double *e5m2_decode() {
    return QT(kE5M2Decode);
}
QC_HD inline const double *e2m1_decode() {
    return QT(kE2M1Decode);
}
QC_HD inline int e4m3_mags_k() {
    return 127;
}
QC_HD inline int e5m2_mags_k() {
    return 124;
}
QC_HD inline int e2m1_mags_k() {
    return 8;
}

// ============================================================================
// 2. host/device 工具函数：二分舍入 / 编解码 / FP16·BF16 转换 / E8M0 指数 / 随机数
// ============================================================================

// 首个 >= v 的下标（语义同 std::lower_bound，即左侧插入位）
QC_HD inline int lower_bound_idx(const double *mags, int K, double v) {
    int lo = 0, hi = K;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (mags[mid] < v)
            lo = mid + 1;
        else
            hi = mid;
    }
    return lo;
}

// 向上取整下标（>= v 的最近可表示值）；超出则饱和到最大值。
// 注：v 为 NaN 时（gs 欠流为 0 且块 amax 为 0 的极端角案）二分收敛到 0，
// 与 std::lower_bound 行为一致。
QC_HD inline int round_up_idx(const double *mags, int K, double v) {
    int idx = lower_bound_idx(mags, K, v);
    return idx < K ? idx : K - 1;
}

// splitmix64：确定性随机数（主机/设备结果一致）
QC_HD inline uint64_t splitmix64(uint64_t z) {
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}
// 每元素独立的随机数状态（与元素全局下标绑定 → 主机/设备逐元素一致）
QC_HD inline uint64_t elem_rng(uint64_t seed, int64_t idx) {
    return splitmix64(seed ^ (uint64_t)idx * 0x9E3779B97F4A7C15ULL);
}
QC_HD inline double rand01(uint64_t *state) {
    *state += 0x9E3779B97F4A7C15ULL;
    uint64_t z = splitmix64(*state);
    return (double)(z >> 11) * 0x1.0p-53; // [0,1) 均匀分布
}

// 浮点值 → 码点。mode: MODE_NEAREST(RNE) / MODE_STOCHASTIC(严格无偏)
QC_HD inline uint8_t fmt_encode(int fmt_id, double x, int mode, uint64_t rng) {
    const double *mags;
    int K, shift;
    switch (fmt_id) {
    case FMT_E4M3:
        mags = e4m3_mags();
        K = e4m3_mags_k();
        shift = 7;
        break;
    case FMT_E5M2:
        mags = e5m2_mags();
        K = e5m2_mags_k();
        shift = 7;
        break;
    default:
        mags = e2m1_mags();
        K = e2m1_mags_k();
        shift = 3;
        break;
    }
    uint8_t sign = sign_bit(x) ? (uint8_t)1 : (uint8_t)0;
    double v = fabs(x);
    int idx = lower_bound_idx(mags, K, v);
    int lo = idx > 0 ? idx - 1 : 0;
    int hi = idx < K ? idx : K - 1;
    double lower = mags[lo], upper = mags[hi];
    bool pick_hi;
    if (mode == MODE_NEAREST) {
        // RNE：更近者胜；平局取偶码点。码点低 3 位即尾数，码点偶 ⟺ 尾数偶，
        // 故「平局取偶码点」正好就是就近舍入的 tie-to-even 语义。
        // 超出表上限时 hi 饱和到 K-1（量化语义，避免产生 NaN）。
        double d_lo = v - lower, d_hi = upper - v;
        pick_hi = (d_hi < d_lo) || (d_hi == d_lo && (hi % 2) == 0);
    } else {
        double span = upper - lower;
        double frac = span > 0.0 ? (v - lower) / span : 0.0;
        pick_hi = rand01(&rng) < frac;
    }
    int i = pick_hi ? hi : lo;
    return (uint8_t)((uint32_t)i | ((uint32_t)sign << shift));
}

// 码点 → 浮点值（查全量解码表，含 NaN/Inf 语义）
QC_HD inline double fmt_decode(int fmt_id, uint8_t code) {
    switch (fmt_id) {
    case FMT_E4M3:
        return e4m3_decode()[code];
    case FMT_E5M2:
        return e5m2_decode()[code];
    default:
        return e2m1_decode()[code];
    }
}

// MX 块缩放指数：e = floor(log2(amax)) - p，裁剪到 [-127,128]。
// 用 frexp 而不是 log2：frexp 是精确位运算，floor(log2(x)) = e_frexp - 1
// 对一切正数成立（x = m×2^e, m∈[0.5,1) ⇒ log2(x) ∈ [e-1,e)），
// 避免 log2 实现的 1ulp 差异在幂次边界翻转 floor。
QC_HD inline long long mx_scale_exponent(double amax, int p) {
    if (!(amax > 0.0))
        return 0; // 全零块：指数 0（scale=1.0），码点全 0
    int e = 0;
    frexp(amax, &e);
    long long r = (long long)e - 1 - p;
    if (r < -127)
        r = -127;
    if (r > 128)
        r = 128;
    return r;
}

// ---- FP16 / BF16 位级转换（主机端与设备端共用同一算法，保证逐字节一致）----

QC_HD inline float f16_bits_to_f32(uint16_t h) {
    uint32_t sign = ((uint32_t)(h >> 15)) << 31;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t man = h & 0x3FF;
    uint32_t bits;
    if (exp == 0) {
        if (man == 0) {
            bits = sign; // ±0
        } else {         // subnormal: man × 2^-24，规格化
            int k = 9;
            while (!((man >> k) & 1))
                k--;
            bits = sign | ((uint32_t)(127 + k - 24) << 23) | ((man << (23 - k)) & 0x7FFFFF);
        }
    } else if (exp == 0x1F) {
        bits = sign | 0x7F800000 | ((uint32_t)man << 13); // Inf / NaN
    } else {
        bits = sign | ((exp - 15 + 127) << 23) | (man << 13);
    }
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

// FP64 → FP16（RNE 正确舍入；nearbyint 默认舍入模式 = 就近偶数）
QC_HD inline uint16_t f64_to_f16_bits(double x) {
    uint16_t sign = sign_bit(x) ? (uint16_t)0x8000 : (uint16_t)0;
    double a = fabs(x);
    if (a == 0.0)
        return sign;
    int e = 0;
    frexp(a, &e);   // a = m × 2^e, m ∈ [0.5,1) → a ∈ [2^(e-1), 2^e)
    int he = e - 1; // 半精度无偏指数
    if (he > 15)
        return sign | 0x7C00;         // ≥ 2^16 → Inf
    if (he >= -14) {                  // 正常数
        double s = ldexp(a, 10 - he); // ∈ [1024, 2048)
        double r = nearbyint(s);
        if (r >= 2048.0) { // 舍入进位到下一指数
            he++;
            if (he > 15)
                return sign | 0x7C00;
            r = 1024.0;
        }
        uint32_t m = (uint32_t)r;
        return sign | (uint16_t)(((he + 15) << 10) | (m & 0x3FF));
    }
    // subnormal / 下溢：可表示值 = k × 2^-24, k ∈ [0,1023]
    double s = ldexp(a, 24);
    double r = nearbyint(s);
    if (r >= 1024.0)
        return sign | 0x0400; // 舍入到最小正常数
    return sign | (uint16_t)(uint32_t)r;
}

// FP32 → BF16 位模式（RNE 截断）
QC_HD inline uint16_t f32_to_bf16_bits(float f) {
    uint32_t u;
    memcpy(&u, &f, 4);
    uint32_t rounded = (u + 0x7FFFu + ((u >> 16) & 1u)) & 0xFFFF0000u;
    return (uint16_t)(rounded >> 16);
}
QC_HD inline float bf16_bits_to_f32(uint16_t h) {
    uint32_t u = (uint32_t)h << 16;
    float f;
    memcpy(&f, &u, 4);
    return f;
}

