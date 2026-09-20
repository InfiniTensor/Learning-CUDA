// engine_common.h - CPU/GPU 引擎共用准备逻辑（host-only）
#pragma once

#include <cmath>

#include "analytic.h"
#include "types.h"

// 由期权/模拟参数构造 device 参数（含 BGK 修正与欧式有效步数）
inline OptionParams make_params(const OptionSpec& os, const SimSpec& ss,
                                const EngineOptions& opts) {
    OptionParams p;
    p.s0 = static_cast<float>(os.spot);
    p.k = static_cast<float>(os.strike);
    p.r = static_cast<float>(os.risk_free_rate);
    p.sigma = static_cast<float>(os.volatility);
    p.T = static_cast<float>(os.maturity);
    p.num_steps = effective_steps(os, ss, opts);

    double dt = os.maturity / p.num_steps;
    p.drift_step = static_cast<float>((os.risk_free_rate - 0.5 * os.volatility * os.volatility) * dt);
    p.vol_step = static_cast<float>(os.volatility * std::sqrt(dt));
    p.disc = static_cast<float>(std::exp(-os.risk_free_rate * os.maturity));
    p.inv_steps = 1.0f / static_cast<float>(p.num_steps);

    if (os.type == OptionType::BarrierCallUpOut) {
        double H = os.barrier;
        if (opts.bgk) H = bgk_shift(os.barrier, os.volatility, os.maturity / ss.num_steps, true);
        p.barrier = static_cast<float>(H);
    } else {
        p.barrier = 0.0f; // 未使用
    }
    return p;
}

// 独立样本单元数：对偶时 1 单元 = 1 对路径
inline long long units_for(const SimSpec& ss) {
    return ss.vr == VarReduction::Antithetic ? ss.num_paths / 2 : ss.num_paths;
}

// 控制变量期望 E[c]（按期权类型选择，与 payoff.h 的 control_variate 对应）
inline double control_mean(const OptionSpec& os, const SimSpec& ss) {
    switch (os.type) {
        case OptionType::EuropeanCall:
        case OptionType::EuropeanPut:
            return os.spot; // E[e^{-rT} S_T] = S0
        case OptionType::AsianCall:
            return geo_asian_call(os.spot, os.strike, os.risk_free_rate, os.volatility,
                                  os.maturity, ss.num_steps);
        case OptionType::BarrierCallUpOut:
            return bs_price(OptionType::EuropeanCall, os.spot, os.strike, os.risk_free_rate,
                            os.volatility, os.maturity);
        default:
            return 0.0;
    }
}
