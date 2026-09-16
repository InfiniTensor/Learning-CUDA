// 计算欧式看涨的解析价格，并处理零波动率边界。
#include "pricer/black_scholes.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace pricer {
namespace {

// 在代入解析公式前排除类型、范围和非有限数错误。
void validate_option(const OptionParams &option) {
    if (option.type != OptionType::EuropeanCall) {
        throw std::domain_error("Black-Scholes requires a European call");
    }
    if (!std::isfinite(option.spot) || !std::isfinite(option.strike) ||
        !std::isfinite(option.risk_free_rate) ||
        !std::isfinite(option.volatility) || !std::isfinite(option.maturity)) {
        throw std::domain_error("Black-Scholes parameters must be finite");
    }
    if (option.spot <= 0.0 || option.strike <= 0.0 || option.maturity <= 0.0 ||
        option.volatility < 0.0) {
        throw std::domain_error("Black-Scholes parameters are out of range");
    }
}

// 用 erfc 表示标准正态 CDF，数值上比手写积分更稳定。
double normal_cdf(double value) {
    return 0.5 * std::erfc(-value / std::sqrt(2.0));
}

void require_finite(double value) {
    if (!std::isfinite(value)) {
        throw std::overflow_error("Black-Scholes result is not finite");
    }
}

} // namespace

// 解析公式只适用于欧式看涨；亚式期权没有同样的闭式解。
double black_scholes_call(const OptionParams &option) {
    validate_option(option);

    const double discount = std::exp(-option.risk_free_rate * option.maturity);
    require_finite(discount);

    // sigma=0 时没有随机性，常规 d1/d2 公式会除以零；直接计算唯一的到期价格。
    if (option.volatility == 0.0) {
        const double terminal_spot =
            option.spot * std::exp(option.risk_free_rate * option.maturity);
        const double price =
            discount * std::max(terminal_spot - option.strike, 0.0);
        require_finite(price);
        return price;
    }

    // Black-Scholes 是欧式看涨的解析“标准答案”。它不参与亚式定价，
    // 而是用来检查蒙特卡洛平均值是否落在合理误差范围内。
    const double volatility_sqrt_time =
        option.volatility * std::sqrt(option.maturity);
    require_finite(volatility_sqrt_time);
    const double d1 =
        (std::log(option.spot / option.strike) +
         (option.risk_free_rate + 0.5 * option.volatility * option.volatility) *
             option.maturity) /
        volatility_sqrt_time;
    const double d2 = d1 - volatility_sqrt_time;
    const double price = option.spot * normal_cdf(d1) -
                         option.strike * discount * normal_cdf(d2);
    require_finite(price);
    return price;
}

} // namespace pricer
