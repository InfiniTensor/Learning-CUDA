#pragma once

// 声明 CPU 单线程蒙特卡洛基线的运行结果和定价入口。

#include "pricer/config.hpp"
#include "pricer/statistics.hpp"

namespace pricer {

// CPU 一次定价产生的累计矩和纯计算耗时。
struct CpuPricingRun {
    RawMoments moments;
    double compute_runtime_ms;
};

class CpuMonteCarloPricer {
  public:
    // 按固定 seed 顺序生成全部路径，供正确性对照使用。
    static CpuPricingRun price(const OptionParams &option,
                               const SimulationParams &simulation);
};

} // namespace pricer
