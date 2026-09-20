// stats.h - 蒙特卡洛累加器 → 价格/标准误/置信区间/误差指标
#pragma once

#include "types.h"

// 三个累加器（无论是否用方差缩减都齐全）：
//   sum     = Σ y（所用估计量的样本，已折现）
//   sumsq   = Σ y²
//   rawsq   = Σ y_raw²（未缩减时的样本二阶矩，用于估计方差缩减因子）
struct Accumulators {
    double sum = 0.0;
    double sumsq = 0.0;
    double rawsq = 0.0;
};

// 由累加器生成完整定价结果（含 CI/误差/z/收敛指标/方差缩减因子）
PriceResult make_result(const Accumulators& acc, long long samples, double reference,
                        const char* reference_name, VarReduction vr, double beta);

// 一次引擎运行的原始输出（主程序负责组装 PriceResult）
struct RawRun {
    Accumulators acc;
    double beta = 0.0;       // 控制变量系数（未启用为 0）
    Timing timing;
    long long paths_run = 0;
    bool cpu_rng_fallback = false;
};
