#pragma once

// 提供欧式看涨期权的 Black-Scholes 解析参考值。

#include "pricer/config.hpp"

namespace pricer {

// 返回欧式看涨的解析价格，用于校验蒙特卡洛估计。
double black_scholes_call(const OptionParams &option);

} // namespace pricer
