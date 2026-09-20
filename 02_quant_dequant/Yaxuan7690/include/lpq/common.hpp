// =============================================================================
//  common.hpp -- 基础工具:编译期平台宏,位运算辅助,CUDA 错误检查,计时器
//
//  本文件被主机侧(g++ / cl.exe)与设备侧(nvcc)共同包含,因此所有需要在两侧
//  共享的函数都必须用 LPQ_HD(__host__ __device__)标注.
// =============================================================================
#pragma once

#include "lpq/platform.hpp"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

// -----------------------------------------------------------------------------
// 编译期宏
//   LPQ_CUDA / LPQ_DEVICE_PASS : 平台与编译遍探测
//   LPQ_HD   : 主机/设备双侧可见
//   LPQ_DEV  : 仅设备侧
//   以上均由 platform.hpp 给出(国产平台的适配点集中在那一个文件里)
// -----------------------------------------------------------------------------

namespace lpq {

// -----------------------------------------------------------------------------
// 位模式 <-> 浮点 互转(主机与设备共用)
// -----------------------------------------------------------------------------
LPQ_HD uint32_t f32_to_bits(float x) {
#if LPQ_DEVICE_PASS
    return __float_as_uint(x);
#else
    uint32_t u;
    std::memcpy(&u, &x, sizeof(u));
    return u;
#endif
}

LPQ_HD float bits_to_f32(uint32_t u) {
#if LPQ_DEVICE_PASS
    return __uint_as_float(u);
#else
    float f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
#endif
}

// -----------------------------------------------------------------------------
// 正确舍入的 fp32 除法
//
// 为什么不直接写 a / b: 摩尔线程 MUSA(mp_22, MTT S4000 实测)的 fp32 除法**不是**
// IEEE 正确舍入的. 在 2^22 个随机 fp32 上实测:
//     a / b         3.9% 差 2 ulp 以上 (22.8% 差 1 ulp)
//     1.0f / x      12.4% 差 1 ulp   (MUSA 的 __frcp_rn 与 1.0f / x 逐位相同, 同样是近似的)
//     __fdiv_rn     4194304 / 4194304 逐位与 x86 主机一致
// 而 x86 主机侧的 / 是正确舍入的, 于是设备 kernel 与主机参考实现会在低位分叉 ——
// NVFP4 的两级缩放正好把全局缩放的倒数带进反量化结果(实测 1ulp 就足以让逐位对拍失败).
//
// __fdiv_rn 在英伟达上与 / 逐位相同, 在 MUSA 上是唯一的正确舍入除法, 因此设备侧统一
// 走它, 主机侧退回 /. 注意 MUSA 上不能用 __frcp_rn 代替 1.0f / x, 它和 / 一样是近似的.
// -----------------------------------------------------------------------------
LPQ_HD float f32_div_rn(float a, float b) {
#if LPQ_DEVICE_PASS
    return __fdiv_rn(a, b);
#else
    return a / b;
#endif
}

// -----------------------------------------------------------------------------
//  精确的 fp32 乘法 / 乘加: 用 __fmul_rn / __fmaf_rn 明确禁止编译器把乘法收缩
//  (fuse)进相邻的加法.
//
//  这个坑踩过两次, 所以把不变量写进源码而不是指望优化器恰好放过这几行:
//
//   (1) 融合统计里喂给累加器的 test 值, 必须与「先落盘、再读回」的那份逐位相同:
//      反量化 kernel 里的乘积一旦写进显存, 舍入就被固定下来; 而量化 kernel 里同一个
//      乘积是**就地**交给 d = test - ref 的, 编译器很乐意把这两步融合成一条 FFMA
//      (少做一次舍入), 于是与独立 metrics kernel 差出 1ulp -- 实测 1e5 个元素里约
//      150 个, 相对误差 ~3e-8, 足以让 1e-13 的判据报警.
//
//   (2) AccumFast 里的 dh * dh / ref * ref 会被收缩进 two_sum_accum 的 hi + x.
//      收缩后 hi 累加的是不舍入的乘积, TwoSum 的残差里就自动带上了乘积自己的舍入
//      误差, 而代码又显式把这份残差补进 lo, 修正项被重复计入一次.
//      实测 mcc 把 (hi + dh * dh) 在 1048576/1048576 个样本上都算成了
//      fma(dh, dh, hi), 而 nvcc 一个都没有.
//
//  主机侧同样会收缩(gcc 默认 -ffp-contract=fast 允许跨语句融合), 因此乘法用
//  volatile 挡住, 乘加走 std::fma -- 它与 __fmaf_rn 都是正确舍入的, 逐位一致.
// -----------------------------------------------------------------------------
LPQ_HD float mul_rn(float a, float b) {
#if LPQ_DEVICE_PASS
    return __fmul_rn(a, b);
#else
    volatile float r = a * b;  // volatile 阻止主机编译器把乘法融进相邻的加法
    return r;
#endif
}

LPQ_HD float fma_rn(float a, float b, float c) {
#if LPQ_DEVICE_PASS
    return __fmaf_rn(a, b, c);
#else
    return std::fma(a, b, c);
#endif
}

// 2 的 n 次幂(n 可为负,保证 2^-149 精确;范围外按 0 / +Inf 饱和)
//
// 无分支版本: 规格化与次正规两条路各算一次位模式再用 SEL 选. 每条路的移位量都
// 先钳进合法区间(变量移位越界在 C++ 里是 UB, 顺手也把"n 超出 fp32 指数范围"的
// 情况收进同一个 SEL). 这个函数在反量化 kernel 里每个元素都要调用一次, 原先的
// 三条分支换算成 SASS 就是三对 BSSY/BSYNC.
LPQ_HD float exp2i(int n) {
    const int hi = (n > 127) ? 127 : n;
    const int norm_e = (hi < -126) ? -126 : hi;
    const int sub_e = (n < -149) ? -149 : ((n > -126) ? -126 : n);
    const uint32_t norm = (uint32_t)(norm_e + 127) << 23;
    const uint32_t sub = (n < -149) ? 0u : (1u << (sub_e + 149));
    const float v = bits_to_f32((n >= -126) ? norm : sub);
    return (n > 127) ? bits_to_f32(0x7F800000u) : v;
}

// -----------------------------------------------------------------------------
// 随机数: 快速 32 位雪崩哈希(用于随机舍入 stochastic rounding).
//
// 原实现是 splitmix64: 每次要两次 64bit 乘法 + 多次 64bit 移位/异或, 在访存受限
// 的量化 kernel 里直接吃掉约 17% 的有效带宽. 但翻到 f32_bits_to_fmt_mag 可以看到
// 真正被使用的只有 rnd 的低 24 位:
//     rem = sig & ((1<<drop)-1) 恒 < 2^24, r = rnd & ((1<<drop)-1),
//     而判定式是 (r < rem), 所以结果只取决于 rnd 的低 24 位.
// 于是换成 murmur3 fmix32 风格的 32 位混合(3 次 32bit 乘法), 低位雪崩质量足够,
// 指令数约为原来的 1/3.
//
// 用法不变: seed + 元素下标 -> 哈希, 与线程调度无关, 可复现.
// 下标高位也参与混合, 因此 numel >= 2^32 时不会与低位下标混叠.
// 注意: CPU 参考实现(src/reference.cu 的 table_quantize)从低 32 位构造 [0,1)
//       均匀分布, 与这里的 32 位输出配套.
// -----------------------------------------------------------------------------
LPQ_HD uint64_t random_at(uint64_t seed, uint64_t idx) {
    uint32_t h = (uint32_t)seed + (uint32_t)idx * 0x9E3779B1u + (uint32_t)(idx >> 32) * 0x85EBCA6Bu;
    h ^= h >> 16;
    h *= 0x85EBCA6Bu;
    h ^= h >> 13;
    h *= 0xC2B2AE35u;
    h ^= h >> 16;
    return (uint64_t)h;
}

// -----------------------------------------------------------------------------
// IEEE half / bfloat16 位模式 -> float(主机侧读取低精度数据时使用,纯位运算)
// -----------------------------------------------------------------------------
LPQ_HD float f32_from_half_bits(uint16_t h) {
    const uint32_t sign = (uint32_t)(h >> 15) << 31;
    const uint32_t ef = (h >> 10) & 0x1Fu;
    const uint32_t mf = h & 0x3FFu;
    uint32_t out;
    if (ef == 0u) {
        if (mf == 0u) {
            out = sign;
        } else {
            // fp16 次正规: value = mf * 2^-24. 把最高有效位归一到 bit10, 得到
            // value = (1 + frac) * 2^(p - 24), fp32 指数域 = 127 + p - 24 = 113 - s
            // (s 为左移次数). 循环从 e = 1 开始递减 s 次, 因此 112 + e = 113 - s.
            int e = 1;
            uint32_t m = mf;
            while (!(m & 0x400u)) {
                m <<= 1;
                --e;
            }
            out = sign | ((uint32_t)(112 + e) << 23) | ((m & 0x3FFu) << 13);
        }
    } else if (ef == 31u) {
        out = sign | 0x7F800000u | (mf << 13);
    } else {
        out = sign | ((ef + 112u) << 23) | (mf << 13);
    }
    return bits_to_f32(out);
}

LPQ_HD float f32_from_bf16_bits(uint16_t b) {
    return bits_to_f32((uint32_t)b << 16);
}

}  // namespace lpq
// =============================================================================
//  主机侧工具(异常,CUDA 运行时检查,计时)
//    注意: 这些声明不能放进 #if !defined(__CUDA_ARCH__) -- nvcc 的设备遍也会解析
//    整份翻译单元, 若此时看不到 Error/LPQ_CHECK 会报 identifier undefined.
// =============================================================================
namespace lpq {

// 运行期错误:携带文件/行号,便于定位
class Error : public std::runtime_error {
public:
    Error(const std::string& msg, const char* file, int line)
        : std::runtime_error(std::string(file) + ":" + std::to_string(line) + ": " + msg) {}
};

#define LPQ_CHECK(cond, msg)                               \
    do {                                                   \
        if (!(cond))                                       \
            throw ::lpq::Error((msg), __FILE__, __LINE__); \
    } while (0)

// 墙钟计时器(host 侧:文件 IO,主机内存准备等)
// 若 v 是 2 的幂则返回 log2(v), 否则返回 -1 (主机侧, 用于避免 64bit 除法)
inline int pow2_shift(int64_t v) {
    if (v <= 0)
        return -1;
    if ((v & (v - 1)) != 0)
        return -1;
    int s = 0;
    while ((int64_t(1) << s) < v)
        ++s;
    return s;
}

class WallTimer {
public:
    void start() { t0_ = std::chrono::steady_clock::now(); }
    double stop_ms() const {  // 返回毫秒
        auto t1 = std::chrono::steady_clock::now();
        return std::chrono::duration<double, std::milli>(t1 - t0_).count();
    }

private:
    std::chrono::steady_clock::time_point t0_{};
};

}  // namespace lpq

#if LPQ_CUDA

namespace lpq {

// CUDA 运行时错误检查
#define LPQ_CUDA_CHECK(call)                                                                     \
    do {                                                                                         \
        cudaError_t err__ = (call);                                                              \
        if (err__ != cudaSuccess) {                                                              \
            throw ::lpq::Error(                                                                  \
                std::string("CUDA error: ") + cudaGetErrorString(err__) + " @ " #call, __FILE__, \
                __LINE__);                                                                       \
        }                                                                                        \
    } while (0)

// 上一次 kernel 启动错误检查
#define LPQ_CUDA_CHECK_KERNEL()                                                                  \
    do {                                                                                         \
        cudaError_t err__ = cudaGetLastError();                                                  \
        if (err__ != cudaSuccess) {                                                              \
            throw ::lpq::Error(std::string("kernel launch error: ") + cudaGetErrorString(err__), \
                __FILE__, __LINE__);                                                             \
        }                                                                                        \
    } while (0)

// 设备端事件计时器(测量 GPU 上的 kernel 区间耗时,不受主机调度抖动影响)
class DeviceTimer {
public:
    DeviceTimer() {
        LPQ_CUDA_CHECK(cudaEventCreate(&begin_));
        LPQ_CUDA_CHECK(cudaEventCreate(&end_));
    }
    ~DeviceTimer() {
        cudaEventDestroy(begin_);
        cudaEventDestroy(end_);
    }
    void start() { LPQ_CUDA_CHECK(cudaEventRecord(begin_)); }
    double stop_ms() {
        LPQ_CUDA_CHECK(cudaEventRecord(end_));
        LPQ_CUDA_CHECK(cudaEventSynchronize(end_));
        float ms = 0.0f;
        LPQ_CUDA_CHECK(cudaEventElapsedTime(&ms, begin_, end_));
        return (double)ms;
    }

private:
    cudaEvent_t begin_{}, end_{};
};

}  // namespace lpq

#endif  // LPQ_CUDA