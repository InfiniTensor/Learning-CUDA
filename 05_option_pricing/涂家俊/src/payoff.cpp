// 计算 GBM 演化常量以及两类期权的折现收益。
#include "pricer/payoff.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace pricer {
namespace {

// 所有金融输入先要求为有限值，防止 NaN 在统计阶段扩散。
void require_finite(double value, const char *name) {
    if (!std::isfinite(value)) {
        throw std::domain_error(std::string(name) + " must be finite");
    }
}

void validate_payoff_inputs(double spot_or_sum, double strike,
                            double discount) {
    require_finite(spot_or_sum, "spot or running sum");
    require_finite(strike, "strike");
    require_finite(discount, "discount");
    if (spot_or_sum < 0.0 || strike <= 0.0 || discount <= 0.0) {
        throw std::domain_error("payoff inputs are out of range");
    }
}

double require_finite_result(double value) {
    if (!std::isfinite(value)) {
        throw std::overflow_error("payoff result is not finite");
    }
    return value;
}

} // namespace

// 预计算每一步漂移、扩散项和全期限折现因子。
GbmStepConstants make_gbm_step_constants(double maturity, double risk_free_rate,
                                         double volatility,
                                         std::uint32_t num_steps) {
    require_finite(maturity, "maturity");
    require_finite(risk_free_rate, "risk-free rate");
    require_finite(volatility, "volatility");
    if (maturity <= 0.0 || volatility < 0.0 || num_steps == 0U) {
        throw std::domain_error("GBM step parameters are out of range");
    }

    // 把总期限 T 切成 num_steps 段；CPU 和 GPU 都使用同一组常量，
    // 因而两端模拟的是同一个风险中性 GBM 模型。
    const double dt = maturity / static_cast<double>(num_steps);
    const double drift = (risk_free_rate - 0.5 * volatility * volatility) * dt;
    const double diffusion = volatility * std::sqrt(dt);
    const double discount = std::exp(-risk_free_rate * maturity);
    require_finite_result(dt);
    require_finite_result(drift);
    require_finite_result(diffusion);
    require_finite_result(discount);
    return {dt, drift, diffusion, discount};
}

double european_call_payoff(double terminal_spot, double strike,
                            double discount) {
    validate_payoff_inputs(terminal_spot, strike, discount);
    // 这里只计算一条路径的“今天价值”：到期的 max(S_T-K, 0) 先得到，
    // 再乘 exp(-rT) 折回今天。之后 Monte Carlo 只需对很多条这样的值取平均。
    return require_finite_result(discount *
                                 std::max(terminal_spot - strike, 0.0));
}

double asian_arithmetic_call_payoff(double running_sum, std::uint32_t num_steps,
                                    double strike, double discount) {
    validate_payoff_inputs(running_sum, strike, discount);
    if (num_steps == 0U) {
        throw std::domain_error("Asian arithmetic payoff requires steps");
    }
    // running_sum 只累加演化后的 M 个监控点，刻意不把初始价格 S0 算入平均值。
    const double average = running_sum / static_cast<double>(num_steps);
    return require_finite_result(discount * std::max(average - strike, 0.0));
}

} // namespace pricer
