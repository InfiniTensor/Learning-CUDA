#pragma once

// 提供 GBM 步进常量和欧式、亚式看涨收益计算。

#include <cstdint>

namespace pricer {

// GBM 一步演化和全期折现共用的预计算常量。
struct GbmStepConstants {
    double dt;
    double drift;
    double diffusion;
    double discount;
};

// 根据总期限与步数生成 CPU/GPU 共用的风险中性 GBM 常量。
GbmStepConstants make_gbm_step_constants(double maturity, double risk_free_rate,
                                         double volatility,
                                         std::uint32_t num_steps);

// 计算一条欧式路径的折现到期收益。
double european_call_payoff(double terminal_spot, double strike,
                            double discount);
// 计算一条亚式路径的折现收益；running_sum 不包含初始价格。
double asian_arithmetic_call_payoff(double running_sum, std::uint32_t num_steps,
                                    double strike, double discount);

} // namespace pricer
