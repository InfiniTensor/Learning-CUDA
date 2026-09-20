// stats.cpp - 蒙特卡洛统计量实现
#include "stats.h"

#include <cmath>

#include "analytic.h"

PriceResult make_result(const Accumulators& acc, long long samples, double reference,
                        const char* reference_name, VarReduction vr, double beta) {
    PriceResult r;
    r.samples = samples;
    r.reference = reference;
    r.reference_name = reference_name;
    r.beta = beta;

    double n = static_cast<double>(samples);
    r.mean_payoff = acc.sum / n;
    // 样本方差（无偏）：var = (Σy² - n·ȳ²) / (n-1)，double 精度无消去问题
    double var = (acc.sumsq - n * r.mean_payoff * r.mean_payoff) / (n - 1.0);
    r.var_payoff = var;
    r.price = r.mean_payoff;
    r.std_error = std::sqrt(var / n);

    double q = t_quantile_95(samples);
    r.ci95_low = r.price - q * r.std_error;
    r.ci95_high = r.price + q * r.std_error;

    r.abs_error = r.price - reference; // 保留符号（z 统计量需要）
    r.rel_error = std::abs(r.abs_error) / std::abs(reference);
    r.z_score = r.std_error > 0.0 ? r.abs_error / r.std_error : 0.0;
    r.abs_error = std::abs(r.abs_error);

    // 收敛指标：|error|·√N 在误差衰减 ~1/√N 时近似为常数
    r.error_times_sqrt_n = std::abs(r.price - reference) * std::sqrt(n);

    // 方差缩减因子 = var(朴素估计量)/var(所用估计量)，二者在同一路径数 N 下比较
    // 对偶：朴素用 N=2n 条独立路径（var/N），对偶用 n 对（var_used/n），故额外除以 2
    if (vr != VarReduction::None && var > 0.0) {
        double var_raw = (acc.rawsq - n * r.mean_payoff * r.mean_payoff) / (n - 1.0);
        double var_plain = (vr == VarReduction::Antithetic) ? var_raw / 2.0 : var_raw;
        if (var_plain > 0.0) r.var_reduction_factor = var_plain / var;
    }
    return r;
}
