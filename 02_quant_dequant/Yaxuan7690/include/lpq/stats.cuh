// =============================================================================
//  stats.cuh -- 误差统计的累加器与归约「汇点」
//
//  误差统计需要 4 个全局和 (sum|d| / sum d^2 / sum ref^2 / sum d) 加一个 max|d|,
//  其中 d = test - ref. 在 6.7e7 量级的元素上要保持 ~1e-15 的相对精度(测试判据
//  1e-12), 而消费级 GPU 的 fp64 ALU 只有 fp32 的 1/64 吞吐 -- 逐元素转 fp64 会把
//  kernel 从访存瓶颈推到算力瓶颈. 因此从逐元素累加、warp 内归约到 block 内归约,
//  主项全程走 fp32 的双分量 (hi, lo), 只在最后落进 fp64 的 atomicAdd.
//
//  把累加器抽到这个头文件, 是为了让两条路径共用**同一份**累加代码:
//    * metrics.cu 里的独立 k_error_stats kernel;
//    * quantize.cu 里量化 kernel 顺带完成的误差累加(省掉一次 268MB 的输入重读).
//  两者对同一组输入给出逐位相同的结果(见 tests/test_pipeline.cu 的 fused-stats).
// =============================================================================
#pragma once

#include "lpq/common.hpp"
#include "lpq/vec_io.cuh"

namespace lpq {

// 归约树与共享内存大小都依赖它, 必须与 kernel 启动配置一致
constexpr int kStatsThreads = 256;

// -----------------------------------------------------------------------------
//  (hi, lo) 补偿累加: hi + lo 用两倍单精度位数表示"和", 相对误差 ~eps^2 ~ 1e-14.
//  Knuth TwoSum -- 无分支, 对任意 a/b 都给出精确的舍入残差(不像 Fast2Sum 需要
//  |a| >= |b| 这个先验假设, 而对流式累加而言这个假设并不总成立).
// -----------------------------------------------------------------------------
LPQ_DEV void two_sum_accum(float& hi, float& lo, float x) {
    const float t = hi + x;
    const float z = t - hi;
    lo += (hi - (t - z)) + (x - z);
    hi = t;
}

// 精确的 fp32 乘法 mul_rn 与乘加 fma_rn 现在放在 common.hpp —— 误差统计(设备侧)
// 与 4/6 的 CPU 参考实现(主机侧)必须共用同一份语义, 坑的详细说明也一并搬了过去.

// 逐元素累计四个和, 顺带维护 fp32 的 max|d|(两种策略的语义完全一致)
struct AccumPrecise {
    double s[4] = {0.0, 0.0, 0.0, 0.0};
    float m = 0.0f;
    LPQ_DEV void add(float test, float ref) {
        const double r = (double)ref;
        const double d = (double)test - r;
        s[0] += fabs(d);
        s[1] += d * d;
        s[2] += r * r;
        s[3] += d;
        m = fmaxf(m, (float)fabs(d));
    }
    LPQ_DEV double value(int j) const { return s[j]; }
    // 交给 warp 内归约树的两分量表示: 把 double 拆成两个 fp32, 高 24 位之外
    // 余下的部分再取 24 位, 合计 ~48 位, 对 1e-13 的判据绰绰有余.
    LPQ_DEV void two_part(int j, float& a, float& b) const {
        a = (float)s[j];
        b = (float)(s[j] - (double)a);
    }
};

// Fast 累加器. 关键点不是"少算", 而是"每一项本身都算准":
//
//   * d = test - ref 用 Knuth TwoSum 拆成精确的两分量 (dh, dl). 直接做 fp32 减法
//     在 test 与 ref 指数相差较大时(量化误差与数据本身同量级, 例如 ref=3e-3 而
//     dequant=0.5)并不精确, 单这一处就会带来 ~1e-8 的相对误差;
//   * |d| = |dh| + sign(dh)*dl, d^2 = dh^2 + 2*dh*dl + dl^2;
//   * 四个主项 |d| / d^2 / ref^2 / d 全部走 Knuth TwoSum 补偿累加; 两个平方项还要
//     先做"精确拆平方": p = fl(x*x) 只留得下 24 位, pe = fma(x, x, -p) 精确给出被舍掉
//     的那一半. 这一步不能省 -- 若改用 fma(x, x, hi) 把"乘积 + 累加"合成一条, 残差
//     fma(x, x, -t) + hi 会退化成 fl(-hi + e) + hi: 由 |e| <= ulp(t)/2 知 -hi + e 舍入
//     回 float 恰好落回 -hi, 残差恒为 0, 补偿整体失效, 相对误差掉回一阶(实测 mse 从
//     ~1e-15 恶化到 7.8e-10, 直接被 1e-13 的判据拦下);
//   * 量级只有主项 1e-8 的修正项(adl / cross / dl)直接补进 lo -- lo 本身就是 1e-7
//     量级的低分量, 再叠 1e-8 的修正不会改变它的量级, 却省下 4 个常驻寄存器.
//
// 每元素约 56 条 fp32 指令(约合 0.9 条 fp64 指令), 因此 kernel 重新变成访存瓶颈.
struct AccumFast {
    float hi[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float lo[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float m = 0.0f;
    LPQ_DEV void add(float test, float ref) {
        const float dh = test - ref;
        const float zz = dh - test;
        const float dl = (test - (dh - zz)) + ((-ref) - zz);
        const float adh = fabsf(dh);
        const float dsum = dh + dl;  // = fl32(d), 用于正确舍入的 max|d| 与符号判定
        // 精确拆平方: pe 是 fma 内部不舍入的乘积与舍入结果之差, 即被舍掉的那一半
        const float p1 = mul_rn(dh, dh);
        const float p1e = fmaf(dh, dh, -p1);
        const float p2 = mul_rn(ref, ref);
        const float p2e = fmaf(ref, ref, -p2);
        two_sum_accum(hi[0], lo[0], adh);
        two_sum_accum(hi[1], lo[1], p1);
        two_sum_accum(hi[2], lo[2], p2);
        two_sum_accum(hi[3], lo[3], dh);
        lo[0] += (dsum < 0.0f) ? -dl : dl;  // |d| = |dh| + sign(d)*dl
        lo[1] += p1e + (2.0f * (dh * dl) + dl * dl);
        lo[2] += p2e;
        lo[3] += dl;
        // fl32(dh + dl) 就是正确舍入的 d, 与 Precise 路径的 (float)fabs(双精度 d) 一致
        m = fmaxf(m, fabsf(dsum));
    }
    LPQ_DEV double value(int j) const { return (double)hi[j] + (double)lo[j]; }
    LPQ_DEV void two_part(int j, float& a, float& b) const {
        a = hi[j];
        b = lo[j];
    }
};

// -----------------------------------------------------------------------------
//  结果输出类型: 反量化结果会先被舍入到 output_type 再落到显存, 而误差统计比对的
//  正是"落盘后"的那个值, 所以这里必须补一次同样的往返舍入, 否则统计里会多出(或少
//  掉)一份 fp16/bf16 存储舍入误差. OutT = float 时为恒等, 编译后消失.
// -----------------------------------------------------------------------------
template <class OutT>
LPQ_DEV float round_trip(float v) {
    return OutTraits<OutT>::to_float(OutTraits<OutT>::from_float(v));
}

// -----------------------------------------------------------------------------
//  归约「汇点」: 量化/统计 kernel 把 (ref, test) 喂进来即可, 收尾由 stats_epilogue
//  统一完成(共享内存树形归约 -> 每个 block 一次 fp64 atomicAdd / float atomicMax).
//
//  NullSink 在编译期完全消失: 未开启统计时, kernel 与手工删掉这几行逐指令相同.
// -----------------------------------------------------------------------------
template <class OutT>
struct NullSink {
    using out_type = OutT;  // 供启动器从 tag 反推输出类型
    static constexpr bool kEnabled = false;
    LPQ_DEV void add(float, float) {}
};

template <class OutT>
struct FastSink {
    using out_type = OutT;
    static constexpr bool kEnabled = true;
    AccumFast acc;
    LPQ_DEV void add(float ref, float test) { acc.add(round_trip<OutT>(test), ref); }
};

template <class OutT>
struct StrictSink {
    using out_type = OutT;
    static constexpr bool kEnabled = true;
    AccumPrecise acc;
    LPQ_DEV void add(float ref, float test) { acc.add(round_trip<OutT>(test), ref); }
};

template <class Sink>
LPQ_DEV void stats_epilogue(Sink& sink, double* __restrict__ acc_out, float* __restrict__ max_out) {
    if constexpr (Sink::kEnabled) {
        constexpr int kWarps = kStatsThreads / 32;
        // 第一级在 warp 内用 shuffle 做完, 而且整棵树都在 fp32 的两分量 (hi, lo) 上:
        // 消费级 Blackwell 的 fp64 ALU 只有 fp32 的 1/64 吞吐, 25 次 warp 归约若放在
        // fp64 里, 每个线程要付 ~1300 条 fp32 指令的代价 -- 实测占量化 kernel 的 20%
        // 以上. 换成两分量树, 精度不变(每一级仍是精确的 TwoSum, 误差 ~eps^2), 指令数
        // 降到 1/8; fp64 只剩每个 block 收尾的 4 次转换 + 4 次 atomicAdd.
        //
        // 共享内存也只需要 8 个 warp 的部分和(288B, 而不是 9KB), 屏障 1 次 --
        // 在量化 kernel 里, 收尾的共享内存与屏障会明显压低可以同时驻留的 block 数,
        // 代价比归约本身大得多.
        float h[4];
        float l[4];
#pragma unroll
        for (int j = 0; j < 4; ++j)
            sink.acc.two_part(j, h[j], l[j]);
        float m = sink.acc.m;
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float oh = LPQ_SHFL_XOR(h[j], off, 32);
                const float ol = LPQ_SHFL_XOR(l[j], off, 32);
                two_sum_accum(h[j], l[j], oh);  // (h, l) <- (h + l) + oh, 精确
                l[j] += ol;                     // 邻居的低分量(~1e-7 量级)直接并入
            }
            m = fmaxf(m, LPQ_SHFL_XOR(m, off, 32));
        }
        __shared__ double s_sum[4 * kWarps];
        __shared__ float s_max[kWarps];
        const int warp = threadIdx.x >> 5;
        const int lane = threadIdx.x & 31;
        if (lane == 0) {
#pragma unroll
            for (int j = 0; j < 4; ++j)
                s_sum[warp * 4 + j] = (double)h[j] + (double)l[j];
            s_max[warp] = m;
        }
        __syncthreads();
        if (warp == 0 && lane == 0) {
            double t[4] = {0.0, 0.0, 0.0, 0.0};
            float mm = 0.0f;
            for (int i = 0; i < kWarps; ++i) {
#pragma unroll
                for (int j = 0; j < 4; ++j)
                    t[j] += s_sum[i * 4 + j];
                mm = fmaxf(mm, s_max[i]);
            }
#pragma unroll
            for (int j = 0; j < 4; ++j)
                lpq_atomic_add_f64(&acc_out[j], t[j]);
            lpq_atomic_max_f32(max_out, mm);
        }
    } else {
        (void)sink;
        (void)acc_out;
        (void)max_out;
    }
}

}  // namespace lpq