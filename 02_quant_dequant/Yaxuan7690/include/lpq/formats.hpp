// =============================================================================
//  formats.hpp -- 低精度浮点格式的软件模拟(纯位运算,不依赖任何新架构指令)
//
//  覆盖的格式:
//    * E2M1 (1-2-1, bias=1)   : NVFP4 的元素格式,最大值 6.0,无 NaN/Inf
//    * E4M3 (1-4-3, bias=7)   : MXFP8 的元素格式 / NVFP4 的块缩放因子格式,
//                               最大值 448.0,0x7F/0xFF 为 NaN,无 Inf
//    * E5M2 (1-5-2, bias=15)  : MXFP8 的可选元素格式,最大值 57344.0
//    * E8M0 (8 bit 纯指数)     : MX 的共享缩放因子,值 = 2^(code-127),255 为 NaN
//    * FP16 / BF16            : 仅作为反量化输出类型
//
//  所有转换函数都同时标注 __host__ / __device__,因此 GPU 实现与 CPU 参考实现
//  使用同一份代码,从根本上排除了"两端公式不一致"的风险;另有一份完全独立
//  的查表法参考实现(见 reference.hpp / tools/reference_mx.py)用于交叉校验.
//
//  转换语义(与 OCP Microscaling Formats v1.0 及 NVIDIA 参考实现一致):
//    * 默认使用「就近舍入,偶数优先」(RNE);
//    * 可选「随机舍入」(SR):在目标的量化格子 [q, q+1)*ulp 上按剩余量成比例
//      随机进位,数学期望无偏;
//    * 有限输入一律饱和(saturating),不会产生 Inf/NaN:
//      例如 E4M3 输入 480 -> 448(而不是 NaN,即 cvt.satfinite 语义).
// =============================================================================
#pragma once

#include "lpq/common.hpp"

namespace lpq {

// -----------------------------------------------------------------------------
// 格式特征
//   emax : 最大规格化指数(无偏)
//   emin : 最小规格化指数(无偏),次正规指数 = emin - mbits
//   total_bits = 1(sign) + ebits + mbits
// -----------------------------------------------------------------------------
struct FmtE2M1 {  // NVFP4 元素
    static constexpr int ebits = 2, mbits = 1, bias = 1;
    static constexpr int emax = 2, emin = 0;
    static constexpr bool has_inf = false, has_nan = false;
    static constexpr uint32_t max_code = 0x7u;  // 6.0
    static constexpr uint32_t inf_code = 0x0u;  // 无 Inf
    static constexpr uint32_t nan_code = 0x0u;  // 无 NaN
    static constexpr uint32_t qnan_code = 0x0u;
    static constexpr float max_norm = 6.0f;
    static constexpr const char* name = "e2m1";
};

struct FmtE4M3 {
    static constexpr int ebits = 4, mbits = 3, bias = 7;
    static constexpr int emax = 8, emin = -6;
    static constexpr bool has_inf = false, has_nan = true;
    static constexpr uint32_t max_code = 0x7Eu;  // 448.0
    static constexpr uint32_t inf_code = 0x0u;
    static constexpr uint32_t nan_code = 0x7Fu;   // 首个 NaN 编码
    static constexpr uint32_t qnan_code = 0x7Fu;  // 典型静默 NaN
    static constexpr float max_norm = 448.0f;
    static constexpr const char* name = "e4m3";
};

struct FmtE5M2 {
    static constexpr int ebits = 5, mbits = 2, bias = 15;
    static constexpr int emax = 15, emin = -14;
    static constexpr bool has_inf = true, has_nan = true;
    static constexpr uint32_t max_code = 0x7Bu;  // 57344.0
    static constexpr uint32_t inf_code = 0x7Cu;
    static constexpr uint32_t nan_code = 0x7Du;   // 首个 NaN 编码
    static constexpr uint32_t qnan_code = 0x7Fu;  // 典型静默 NaN
    static constexpr float max_norm = 57344.0f;
    static constexpr const char* name = "e5m2";
};

struct FmtF16 {
    static constexpr int ebits = 5, mbits = 10, bias = 15;
    static constexpr int emax = 15, emin = -14;
    static constexpr bool has_inf = true, has_nan = true;
    static constexpr uint32_t max_code = 0x7BFFu;  // 65504
    static constexpr uint32_t inf_code = 0x7C00u;
    static constexpr uint32_t nan_code = 0x7C01u;   // 首个 NaN 编码
    static constexpr uint32_t qnan_code = 0x7E00u;  // 典型静默 NaN
    static constexpr float max_norm = 65504.0f;
    static constexpr const char* name = "fp16";
};

struct FmtBF16 {
    static constexpr int ebits = 8, mbits = 7, bias = 127;
    static constexpr int emax = 127, emin = -126;
    static constexpr bool has_inf = true, has_nan = true;
    static constexpr uint32_t max_code = 0x7F7Fu;
    static constexpr uint32_t inf_code = 0x7F80u;
    static constexpr uint32_t nan_code = 0x7F81u;   // 首个 NaN 编码
    static constexpr uint32_t qnan_code = 0x7FC0u;  // 典型静默 NaN
    static constexpr float max_norm = 3.38953139e38f;
    static constexpr const char* name = "bf16";
};

// -----------------------------------------------------------------------------
// 舍入模式
// -----------------------------------------------------------------------------
enum class Rounding : int {
    Nearest = 0,     // round-to-nearest, ties-to-even
    Stochastic = 1,  // 随机舍入,需提供 64bit 随机数
};

// floor(log2(x)),x > 0 且有限;x == 0 时返回 INT32_MIN 表示 -inf
LPQ_HD int ilog2_floor(float x) {
    uint32_t b = f32_to_bits(x) & 0x7FFFFFFFu;
    if (b == 0)
        return INT32_MIN;
    int e = (int)(b >> 23) - 127;
    if (e == -127) {  // fp32 次正规:value = frac * 2^-149
        uint32_t frac = b & 0x7FFFFFu;
        int lz = 0;
        while (!(frac & 0x800000u)) {
            frac <<= 1;
            ++lz;
        }
        return -127 - lz;  // p - 149 = (22 - lz) - 149
    }
    return e;
}

// -----------------------------------------------------------------------------
// 前导零计数: 设备侧是单条指令; 主机侧只是参考实现, 用可移植的二分写法.
// (次正规规格化原先靠 while 循环找最高有效位, 循环次数随数据变化, 在一个 warp
//  内必然发散; 改成 __clz 后是固定代价, 且没有分支.)
// -----------------------------------------------------------------------------
LPQ_HD uint32_t clz32(uint32_t x) {
#if LPQ_DEVICE_PASS
    return (uint32_t)__clz(x);
#else
    if (x == 0u)
        return 32u;
    uint32_t n = 0;
    if (x < 0x00010000u) {
        n += 16;
        x <<= 16;
    }
    if (x < 0x01000000u) {
        n += 8;
        x <<= 8;
    }
    if (x < 0x10000000u) {
        n += 4;
        x <<= 4;
    }
    if (x < 0x40000000u) {
        n += 2;
        x <<= 2;
    }
    if (x < 0x80000000u)
        n += 1;
    return n;
#endif
}

// -----------------------------------------------------------------------------
// 核心:|x| 的位模式 -> 目标格式的幅值编码(不含符号位)
//
//   abits : fabsf(x) 的 IEEE754 位模式
//   rnd   : 随机舍入使用的 64bit 随机数(RNE 时可传 0)
//   stoch : 是否随机舍入(调用点都是编译期常量, 分支会被折叠掉)
//
//   算法(统一处理规格化与次正规):
//     1. 取出 24bit 有效数字 sig(含隐含 1)与指数 e;
//     2. 令 drop = 23 - mbits,若 e < emin 再补 (emin - e) 位,
//        即把 sig 截断到目标格式在该量级上的量化格 2^drop;
//     3. 对被截掉的低位做 RNE 或 SR 得到 q;
//     4. q 溢出尾数字段则指数进位;最后做饱和处理.
//
//   实现上刻意做到「无数据相关分支」: 每个分支在 SASS 里都要配一对 BSSY/BSYNC
//   (重汇聚点), 而这里每个元素都要走一遍, 4 元素展开后就是十几条纯开销. 量化
//   kernel 是访存瓶颈, 这些控制指令直接吃掉有效带宽. 具体地:
//     * Inf/NaN/±0 不再提前 return, 改成末尾一次 SEL;
//     * fp32 次正规的规格化用 __clz, 消掉数据相关的 while 循环;
//     * drop 恒 > 0(mbits <= 10 => 23 - mbits >= 13), 左移那条路根本走不到;
//     * drop >= 64 的下溢分支改成把 drop 钳到 63, 此时 q 必为 0 且余数远小于
//       半个 ulp, 结果与直接 return 0 逐位相同;
//     * RNE 用「加 half-1+最低位再截断」一次完成, 省掉两次 64bit 比较.
// -----------------------------------------------------------------------------
template <class F>
LPQ_HD uint32_t f32_bits_to_fmt_mag(uint32_t abits, uint64_t rnd, bool stoch) {
    constexpr int kE = F::ebits;
    constexpr int kM = F::mbits;
    // drop = 23 - mbits + max(0, emin - e) 必须恒 > 0(mbits <= 10), 右移路径才唯一
    static_assert(kM < 23, "drop must stay positive: right-shift path is the only path");

    // --- 特殊值: ±0 与 Inf / NaN --------------------------------------------
    const uint32_t frac = abits & 0x7FFFFFu;
    const uint32_t special = (abits == 0u)
                                 ? 0u
                                 : ((frac != 0u) ? (F::has_nan ? F::qnan_code : F::max_code)
                                                 : (F::has_inf ? F::inf_code : F::max_code));

    // --- 规格化: 统一成 24bit 有效数字 sig, 值 = sig * 2^(e-23) -------------
    const bool is_sub = abits < 0x00800000u;
    const uint32_t sh = clz32(abits) - 8u;  // abits == 0 时结果为 0, 已被 special 覆盖
    const uint32_t sig = is_sub ? (abits << sh) : (0x800000u | frac);
    const int32_t e = is_sub ? ((int32_t)(23u - sh) - 149) : ((int32_t)(abits >> 23) - 127);

    // --- 量化格宽度 ---------------------------------------------------------
    const int32_t extra = F::emin - e;
    int32_t drop = (23 - kM) + (extra > 0 ? extra : 0);
    if (drop > 63)
        drop = 63;

    const uint64_t sig64 = (uint64_t)sig;
    const uint32_t q0 = (uint32_t)(sig64 >> drop);
    uint32_t q;
    if (stoch) {
        // q -> q+1 的概率 = rem / 2^drop
        const uint64_t rem = sig64 & (((uint64_t)1 << drop) - 1u);
        const uint64_t r = rnd & (((uint64_t)1 << drop) - 1u);
        q = q0 + (uint32_t)(r < rem);
    } else {
        // RNE: 给 sig 加上 (half - 1 + q0 的最低位) 再截断. 进位由加法自然完成,
        // 等价于 rem > half, 或 rem == half 且 q0 为奇数(向偶舍入).
        const uint64_t half = (uint64_t)1 << (drop - 1);
        q = (uint32_t)((sig64 + (half - 1u) + (q0 & 1u)) >> drop);
    }

    // --- 组装编码 -----------------------------------------------------------
    // 尾数进位到指数: q 达到 2^(kM+1) 说明有效数字跨过了量化格
    const uint32_t carry = (q >> kM) >= 2u;
    const int32_t eo = e + (int32_t)carry;
    uint32_t mag = (uint32_t)(eo + F::bias) << kM;
    mag |= carry ? 0u : (q & ((1u << kM) - 1u));
    if (eo > F::emax)
        mag = F::max_code;  // 上溢饱和
    // 次正规区: 指数域为 0; q 进位到 2^mbits 时恰好是最小规格化数
    const uint32_t mag_sub = (q < (1u << kM)) ? q : (1u << kM);
    mag = (e >= F::emin) ? mag : mag_sub;
    // 饱和:E4M3 的 0x7F 等保留编码不能由有限输入产生
    if (mag > F::max_code)
        mag = F::max_code;
    // abits == 0 或 Inf/NaN 才走 special: (abits - 1) 的无符号比较一下就覆盖了两端
    return ((abits - 1u) >= 0x7F7FFFFFu) ? special : mag;
}

#ifndef LPQ_HW_FP8X2
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 890)
#define LPQ_HW_FP8X2 1
#else
#define LPQ_HW_FP8X2 0
#endif
#endif

#if LPQ_HW_FP8X2
// 一条指令把两个 f32 打包成两个 e4m3/e5m2:
// 结果的低字节 = convert(lo), 高字节 = convert(hi)
// (与 CUDA 头文件 cuda_fp8.hpp 里 __nv_cvt_float2_to_fp8x2 的约定一致)
LPQ_DEV uint32_t hw_cvt_e4m3x2(float lo, float hi) {
    unsigned short r;
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;" : "=h"(r) : "f"(lo), "f"(hi));
    return (uint32_t)r;
}
LPQ_DEV uint32_t hw_cvt_e5m2x2(float lo, float hi) {
    unsigned short r;
    asm("cvt.rn.satfinite.e5m2x2.f32 %0, %2, %1;" : "=h"(r) : "f"(lo), "f"(hi));
    return (uint32_t)r;
}
#endif

// 反向: 2 个 f8(打包在 16bit, 低字节 = lo) -> half2.
// 已验证 256 个编码(含 NaN code)与下面的软件译码逐位一致.
LPQ_DEV uint32_t hw_cvt_e4m3x2_to_f16x2(uint32_t pair) {
    unsigned int r;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(r) : "h"((unsigned short)pair));
    return r;
}
LPQ_DEV uint32_t hw_cvt_e5m2x2_to_f16x2(uint32_t pair) {
    unsigned int r;
    asm("cvt.rn.f16x2.e5m2x2 %0, %1;" : "=r"(r) : "h"((unsigned short)pair));
    return r;
}
// half -> float(单个半字; f8 字段在 f16 里完全可表示, 这一步是无损的)
LPQ_DEV float hw_cvt_f16_to_f32(uint32_t h) {
    float f;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"((unsigned short)(h & 0xFFFFu)));
    return f;
}

// -----------------------------------------------------------------------------
//  E2M1 专用的 RNE 编码器(|x| 的位模式 -> 4bit 幅值编码)
//
//  通用路径要维护 64bit 有效数字、可变移位、进位与三段饱和; 而 e2m1 只有 16 个编码,
//  量化格的结构简单到可以直接在位模式上做:
//
//    * |x| >= 1: 加 (half-1 + 最低位) 再截断就是 RNE, 而且进位自然会从尾数字段
//      进到指数域 -- 这正好等价于"尾数溢出则指数 +1", 于是
//        code = (biased_exp - 126) * 2 + 保留下的尾数位
//      一次 32bit 加法就同时完成了舍入、进位与指数重算;
//    * |x| < 1: e2m1 在 [0.25, 1) 上只有 0.5 这一个格点, 阈值比较即可
//      (0.25 与 0.75, 两个阈值各自 <=> 编码 1 与 2 的偶数舍入方向);
//    * Inf/NaN 不需要单独分支: 指数域被抬到 0xFF 后 code 已经 > 7, 被同一条饱和截断;
//    * |x| == 0 同理落到负数一侧, 被 clamp 到 0.
//
//  与通用实现逐位等价, 已在全部 2^32 个 fp32 位模式上对拍(见 build/_e2m1.cu 与
//  tests/test_formats.cu 的中点用例)。
// -----------------------------------------------------------------------------
LPQ_HD uint32_t e2m1_mag_rne(uint32_t a) {
    const uint32_t b = a + 0x1FFFFFu + ((a >> 22) & 1u);
    int code = (int)(b >> 23) * 2 - 252 + (int)((b >> 22) & 1u);
    code = (code > 7) ? 7 : code;
    const uint32_t sub = (uint32_t)(a > 0x3E800000u) + (uint32_t)(a >= 0x3F400000u);
    return (a < 0x3F800000u) ? sub : (uint32_t)code;
}

template <class F>
LPQ_HD uint32_t f32_bits_to_fmt_mag_rne(uint32_t abits) {
#if LPQ_HW_FP8X2
    if constexpr ((F::ebits == 4 && F::mbits == 3) || (F::ebits == 5 && F::mbits == 2)) {
        if (abits < 0x7F800000u) {
            const float x = bits_to_f32(abits);
            return (F::ebits == 4 ? hw_cvt_e4m3x2(x, 0.0f) : hw_cvt_e5m2x2(x, 0.0f)) & 0xFFu;
        }
    }
#endif
    if constexpr (F::ebits == 2 && F::mbits == 1)
        return e2m1_mag_rne(abits);
    return f32_bits_to_fmt_mag<F>(abits, 0ull, false);
}

// float -> 目标格式编码(含符号位)
template <class F>
LPQ_HD uint32_t f32_to_fmt(float x, uint64_t rnd = 0, bool stoch = false) {
    uint32_t b = f32_to_bits(x);
    uint32_t sign = b >> 31;
    uint32_t mag = stoch ? f32_bits_to_fmt_mag<F>(b & 0x7FFFFFFFu, rnd, true)
                         : f32_bits_to_fmt_mag_rne<F>(b & 0x7FFFFFFFu);
    return (sign << (F::ebits + F::mbits)) | mag;
}

// 目标格式编码 -> float
//
// 同样是"无数据相关分支"版本: 原来的 mag==0 / inf / nan / 次正规 四条分支在反量化
// kernel 与融合统计里都是每元素走一遍, 换成一两个 SEL 之后控制开销直接消失.
//
// 有限值统一写成 mant * 2^exp:
//   * ef == 0(次正规): 隐含位为 0, 指数取 emin;
//   * ef != 0(规格化): 隐含位为 1, 指数取 ef - bias;
// 于是只有一次整数乘 + 一次 2 的幂乘法, 且两者都精确(乘积的有效位数 <= kM+1 <= 11),
// 与原来的 (1 + mf*2^-kM) * 2^(ef-bias) 逐位相同.
template <class F>
LPQ_HD float fmt_to_f32(uint32_t code) {
#if LPQ_HW_FP8X2
    // 硬件译码: cvt.rn.f16x2.{e4m3,e5m2}x2 + cvt.f32.f16 两条指令, 取代下面
    // 十几条整数拆分 + exp2i 的软件路径. e4m3/e5m2 的全部有限值都能被 fp16 精确
    // 表示, 因此两条路逐位等价(已对全部 256 个编码核对, 见 build/_cvt2.cu).
    if constexpr ((F::ebits == 4 && F::mbits == 3) || (F::ebits == 5 && F::mbits == 2)) {
        const uint32_t pair = (F::ebits == 4) ? hw_cvt_e4m3x2_to_f16x2(code & 0xFFu)
                                              : hw_cvt_e5m2x2_to_f16x2(code & 0xFFu);
        return hw_cvt_f16_to_f32(pair);
    }
#endif
    constexpr int kE = F::ebits, kM = F::mbits;
    const uint32_t sign = (code >> (kE + kM)) & 1u;
    const uint32_t mag = code & ((1u << (kE + kM)) - 1u);
    const uint32_t ef = mag >> kM;
    const uint32_t mf = mag & ((1u << kM) - 1u);
    const uint32_t mant = mf | (ef ? (1u << kM) : 0u);
    const int exp = (ef ? ((int)ef - F::bias) : F::emin) - kM;
    float v = (float)mant * exp2i(exp);
    if (F::has_inf) {  // 编译期常量条件, 格式没有 Inf 时整段消失
        v = (mag == F::inf_code) ? bits_to_f32(0x7F800000u) : v;
    }
    if (F::has_nan) {
        v = (mag >= F::nan_code) ? bits_to_f32(0x7FC00000u) : v;
    }
    return sign ? -v : v;
}

// -----------------------------------------------------------------------------
//  BF16 译码: 与上面的通用模板逐位等价, 但改用纯移位而不是 mant * 2^exp.
//
//  bf16 的次正规数(指数域为 0)对应的 fp32 仍是次正规数, 用乘法算会被设备端的
//  FTZ(flush-to-zero)刷成 0: 摩尔线程 MUSA(mp_22, MTT S4000 实测)的 fp32 乘加
//  一律把次正规结果归零, 而且 -fno-gpu-flush-denormals-to-zero 也关不掉.
//  但 bf16 -> fp32 本来就只是左移 16 位, 有限值(含次正规与 +-0)逐位精确, 全程
//  不经过任何浮点运算, 因此天然不受 FTZ 影响. Inf/NaN 沿用通用模板的同一套掩码
//  (含 0x7FC00000 规范化与符号位的施加方式), 保证两边输出逐位一致.
// -----------------------------------------------------------------------------
template <>
LPQ_HD float fmt_to_f32<FmtBF16>(uint32_t code) {
    const uint32_t sign = (code >> 15) & 1u;
    const uint32_t mag = code & 0x7FFFu;
    uint32_t bits = code << 16;  // 有限值(含次正规)已经正确, 符号位也一并带上
    if (mag == FmtBF16::inf_code)
        bits = 0x7F800000u;
    if (mag >= FmtBF16::nan_code)
        bits = 0x7FC00000u;
    return bits_to_f32(sign ? (bits | 0x80000000u) : bits);
}

// -----------------------------------------------------------------------------
//  E2M1 译码专用版(只有 16 个编码)
//
//  通用实现要算出 mant / exp 再用 exp2i(带次正规修正)拼出 fp32,
//  而 e2m1 的有效数字只有 {0,0.5,1,1.5,2,3,4,6}. 这里直接按位拼
//  fp32:  幅值 m -> 指数域 126 + (m>>1), 尾数 0x400000 只在 m>=3 且 m 为奇时置位.
//  (m=1 是 e2m1 的次正规 0.5, 不能用 1+mf/2 的公式。全部 16 个编码
//   与通用实现逐位对过, 见 tests/test_formats.cu 的穷举对比。)
// -----------------------------------------------------------------------------
template <>
LPQ_HD float fmt_to_f32<FmtE2M1>(uint32_t code) {
    const uint32_t sign = code >> 3;
    const uint32_t m = code & 7u;
    const uint32_t mant = ((m & 1u) & (uint32_t)(m > 1u)) << 22;
    const uint32_t bits = ((126u + (m >> 1)) << 23) | mant;
    float v = bits_to_f32(bits);
    v = (m == 0u) ? 0.0f : v;
    return sign ? -v : v;
}

// 4 个 f8 编码(每字节一个, 打包在 uint32 里) -> 4 个 float.
// 逐位等价于 fmt_to_f32<F>((p >> (8k)) & 0xFF).
template <class F>
LPQ_DEV void fmt_x4_to_f32(uint32_t p, float* v) {
#if LPQ_HW_FP8X2
    if constexpr ((F::ebits == 4 && F::mbits == 3) || (F::ebits == 5 && F::mbits == 2)) {
        const uint32_t lo2 = (F::ebits == 4) ? hw_cvt_e4m3x2_to_f16x2(p & 0xFFFFu)
                                             : hw_cvt_e5m2x2_to_f16x2(p & 0xFFFFu);
        const uint32_t hi2 =
            (F::ebits == 4) ? hw_cvt_e4m3x2_to_f16x2(p >> 16) : hw_cvt_e5m2x2_to_f16x2(p >> 16);
        v[0] = hw_cvt_f16_to_f32(lo2);
        v[1] = hw_cvt_f16_to_f32(lo2 >> 16);
        v[2] = hw_cvt_f16_to_f32(hi2);
        v[3] = hw_cvt_f16_to_f32(hi2 >> 16);
        return;
    }
#endif
    for (int k = 0; k < 4; ++k)
        v[k] = fmt_to_f32<F>((p >> (8 * k)) & 0xFFu);
}

// 元素编码的位宽: e2m1 为 4bit, e4m3 / e5m2 为 8bit.
// 判据写成 ebits + mbits <= 4 而不是枚举格式名, 将来再加 4bit 元素(e2m3 等)时自动成立.
template <class F>
struct ElemTraits {
    static constexpr int bits = (F::ebits + F::mbits <= 4) ? 4 : 8;
};

// 4 个 4bit 编码(每字节 2 个元素, 下标偶数的在低半字节, 打包在 uint16 里) -> 4 个
// float. 逐位等价于 fmt_to_f32<F>((p >> (4k)) & 0xF), 与 k_dequant_nvfp4 的半字节
// 约定完全一致; e2m1 没有硬件译码指令, 因此这里就是逐元素的软件查表.
template <class F>
LPQ_HD void fmt_x4_4bit_to_f32(uint16_t p, float* v) {
    for (int k = 0; k < 4; ++k)
        v[k] = fmt_to_f32<F>((uint32_t)((p >> (4 * k)) & 0xFu));
}

// -----------------------------------------------------------------------------
// E8M0:MX 的共享缩放因子,只有 8bit 指数,值 = 2^(code-127),255 = NaN
// -----------------------------------------------------------------------------
LPQ_HD float e8m0_to_f32(uint8_t code) {
    if (code == 0xFFu)
        return bits_to_f32(0x7FC00000u);
    return exp2i((int)code - 127);
}

// 1 / e8m0_to_f32(code) = 2^(127 - code), 按 2 的幂直接构造, 不做除法.
//
// 为什么不写成 1.0f / e8m0_to_f32(sc): 摩尔线程 MUSA(mp_22, MTT S4000 实测)的
// fp32 除法把次正规的操作数当 0, 而 sc == 0 时 e8m0_to_f32(0) = 2^-127 恰好是
// 次正规数, 于是 1/2^-127 会算成 +Inf, 紧接着的 0.0f * Inf 变成 NaN —— 全零块就
// 被量化成 NaN 编码(e4m3/e5m2 的 0x7F)而不是 0. 改成 exp2i(127 - code) 后两个
// 操作数都是 2 的整数次幂, 逐位可表示, 任何平台上都精确一致; 顺带把内层循环里的
// 一次除法换成了几条整数指令.
LPQ_HD float e8m0_rcp(uint8_t code) {
    if (code == 0xFFu)
        return bits_to_f32(0x7FC00000u);  // 与 e8m0_to_f32 同样把 255 视为 NaN
    return exp2i(127 - (int)code);
}

// 共享指数 -> E8M0 编码:code = shared_exp + 127,饱和到 [0,254](255 保留为 NaN)
LPQ_HD uint8_t e8m0_encode(int shared_exp) {
    int c = shared_exp + 127;
    if (c < 0)
        c = 0;
    if (c > 254)
        c = 254;
    return (uint8_t)c;
}

// -----------------------------------------------------------------------------
// MX 共享指数:shared_exp = floor(log2(amax)) - emax_elem,并夹到 [-127,127]
//   (与 OCP MX v1.0 / microxcaling 参考实现一致)
//   amax == 0 时返回 -127(最小缩放),此时所有元素都会量化为 0.
// -----------------------------------------------------------------------------
LPQ_HD int mx_shared_exp(float amax, int emax_elem) {
    if (!(amax > 0.0f))
        return -127;  // 全零块
    int e = ilog2_floor(amax) - emax_elem;
    if (e < -127)
        e = -127;
    if (e > 127)
        e = 127;
    return e;
}

// 编译期版本(元素格式已知时使用)
template <class ElemFmt>
LPQ_HD int mx_shared_exp(float amax) {
    return mx_shared_exp(amax, ElemFmt::emax);
}

// -----------------------------------------------------------------------------
// NVFP4 两级缩放常量
//   全局缩放(per-tensor, FP32)  : global_scale = max_q * max_s / amax_tensor
//   元素局部缩放(per-16, E4M3)  : block_scale  = e4m3(amax_block * global_scale / max_q)
//   量化                          : q = e2m1(x * global_scale / block_scale)
//   反量化                        : x = q * block_scale / global_scale
//   两处除法都走 f32_div_rn: MUSA 的 fp32 除法不是正确舍入的, 直接写 / 会让设备
//   侧与主机参考在低位分叉(见 common.hpp 里 f32_div_rn 的实测数据).
//
//   标准 NVFP4 取 max_q = 6 (E2M1 最大值), max_s = 448 (E4M3 最大值).
// -----------------------------------------------------------------------------
constexpr float kMaxE2M1 = FmtE2M1::max_norm;  // 6.0   : E2M1 的最大规格化值
constexpr float kMaxE4M3 = FmtE4M3::max_norm;  // 448.0 : E4M3 的最大规格化值

// -----------------------------------------------------------------------------
// 4/6 自适应块缩放的两个常量 (Four Over Six, arXiv:2512.02010)
//
// 两级缩放的分工: 全局缩放把整张量的 amax 映射到 max_q * max_s, 块缩放再把每块的
// amax 映射到 max_q. 标准 NVFP4 取 max_q = 6, max_s = 448.
//
// 4/6 的「4」候选把块缩放因子乘 1.5, 让块内 amax 落到 4.0 而不是 6.0.
// 动因是浮点格点非均匀: E2M1 在 [4, 6] 之间的格距是 2, 而在 [2, 4] 之间只有 1,
// 越靠近最大值量化误差越大. 把最大值压到 4.0, 整个块的可表示值分布就更均匀.
// 代价是 1.5 倍扩张后的块缩放因子可能溢出 E4M3, 因此动态规则要把 max_s 从 448
// 降到 256 -- 256 * 1.5 = 384 <= 448, 正好留出扩张所需的余量.
// (与参考实现 mit-han-lab/fouroversix 中 ScaleType.nv 的 "448 if is_static else 256"
//  取值一致.)
// -----------------------------------------------------------------------------
constexpr float kFourOverSixScaleMax = 256.0f;  // 动态规则的 max_s
constexpr float kFourOverSixExpansion = 1.5f;   // 「4」候选相对「6」候选的扩张倍数

LPQ_HD float nvfp4_global_scale(float tensor_amax, float max_q = kMaxE2M1, float max_s = kMaxE4M3) {
    if (!(tensor_amax > 0.0f))
        return 1.0f;  // 全零张量:避免除零
    return f32_div_rn(max_q * max_s, tensor_amax);
}

// 块缩放因子(尚未舍入到 E4M3;调用方负责再编码到 E4M3)
LPQ_HD float nvfp4_block_scale_f32(float block_amax, float global_scale, float max_q = kMaxE2M1) {
    return f32_div_rn(block_amax * global_scale, max_q);
}

}  // namespace lpq