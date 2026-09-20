// payoff.h - host/device 共享的收益函数与路径递推
//
// 约定（CPU/GPU 一致，报告中详述）：
// - 亚式平均取 t_1..t_n 的算术平均，不含 t_0（与 Kemna-Vorst 几何亚式公式的监控约定一致）
// - 障碍敲出判定：max{S_{t_1..t_n}} >= H 即敲出（要求 H > S0，t_0 不影响判定）
// - 路径递推为精确 GBM 转移 S_{i+1} = S_i·exp((r−σ²/2)Δt + σ√Δt·Z)，非 Euler 离散
//   终值分布在每个监控点精确成立；路径依赖收益仅存在离散监控偏差
#pragma once

#include "types.h"

// host/device 共享宏（与 rng.h 一致；g++ 编译时退化为普通 inline）
#ifndef OP_HD
#ifdef __CUDACC__
#define OP_HD __host__ __device__ inline
#else
#define OP_HD inline
#endif
#endif

// 单步推进：更新 S，并按期权类型维护亚式累计和 / 障碍运行最大值
// logsum 仅亚式使用（几何平均 = exp(logsum/n)，控制变量用）；OPT 为编译期常量，非对应分支被消除
template <OptionType OPT>
OP_HD void advance(float drift_step, float vol_step, float z,
                   float& S, float& acc, float& maxS, float& logsum) {
    S *= expf(drift_step + vol_step * z);
    if (OPT == OptionType::AsianCall) {
        acc += S;
        logsum += logf(S);
    } else if (OPT == OptionType::BarrierCallUpOut) {
        maxS = fmaxf(maxS, S);
    }
}

// 到期收益（未折现）。avg 为亚式算术平均，maxS 为障碍运行最大值（其余类型忽略）
template <OptionType OPT>
OP_HD float payoff(float S_T, float avg, float maxS, const OptionParams& p) {
    if (OPT == OptionType::EuropeanCall) {
        return fmaxf(S_T - p.k, 0.0f);
    } else if (OPT == OptionType::EuropeanPut) {
        return fmaxf(p.k - S_T, 0.0f);
    } else if (OPT == OptionType::AsianCall) {
        return fmaxf(avg - p.k, 0.0f);
    } else { // BarrierCallUpOut：触障即敲出，收益为 0
        return (maxS >= p.barrier) ? 0.0f : fmaxf(S_T - p.k, 0.0f);
    }
}

// 控制变量 c（已折现）。geo_mean 为几何平均（仅亚式使用，其余类型忽略）
// 控制变量按期权类型选择（报告中详述）：
//   欧式   c = e^{-rT}·S_T           E[c] = S0
//   亚式   c = e^{-rT}·(G−K)^+       E[c] = Kemna-Vorst 几何亚式价格
//   障碍   c = e^{-rT}·(S_T−K)^+     E[c] = BS 看涨价格
// 注意：欧式不能用自身收益做控制变量（会退化为解析解）
template <OptionType OPT>
OP_HD float control_variate(float S_T, float geo_mean, const OptionParams& p) {
    if (OPT == OptionType::AsianCall) {
        return p.disc * fmaxf(geo_mean - p.k, 0.0f);
    } else if (OPT == OptionType::BarrierCallUpOut) {
        return p.disc * fmaxf(S_T - p.k, 0.0f);
    } else { // 欧式看涨/看跌共用：折现后的终值
        return p.disc * S_T;
    }
}
