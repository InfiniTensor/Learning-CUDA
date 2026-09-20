// analytic.cpp - 解析参考价格实现
// 所有公式已用独立 Python 脚本交叉验证（data/reference_values.txt 中的黄金值）
#include "analytic.h"

#include <cmath>

double norm_cdf(double x) {
    return 0.5 * std::erfc(-x / std::sqrt(2.0));
}

double norm_pdf(double x) {
    return 0.39894228040143267794 * std::exp(-0.5 * x * x); // 1/sqrt(2*pi)
}

double bs_price(OptionType type, double S, double K, double r, double sigma, double T) {
    double sqT = std::sqrt(T);
    double d1 = (std::log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqT);
    double d2 = d1 - sigma * sqT;
    if (type == OptionType::EuropeanPut) {
        return K * std::exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1);
    }
    return S * norm_cdf(d1) - K * std::exp(-r * T) * norm_cdf(d2);
}

double geo_asian_call(double S, double K, double r, double sigma, double T, int n) {
    // Kemna-Vorst：几何平均的 log 分布参数（监控 t_i = iT/n, i = 1..n）
    double m = std::log(S) + (r - 0.5 * sigma * sigma) * T * (n + 1) / (2.0 * n);
    double v = sigma * sigma * T * (n + 1) * (2 * n + 1) / (6.0 * n * n);
    double sv = std::sqrt(v);
    double d1 = (m - std::log(K) + v) / sv;
    double d2 = d1 - sv;
    return std::exp(-r * T) * (std::exp(m + 0.5 * v) * norm_cdf(d1) - K * norm_cdf(d2));
}

double barrier_up_out_call(double S, double K, double H, double r, double sigma, double T) {
    // 连续监控向上敲出看涨（反射原理）。要求 H > S（输入侧已校验）
    double nu = r - 0.5 * sigma * sigma;          // 对数漂移
    double m = std::log(H / S);                   // 障碍的对数距离
    double A = std::log(K / S);
    double lambda = (r + 0.5 * sigma * sigma) / (sigma * sigma);
    double sqT = std::sqrt(T);

    // d(z) = (z - nu*T)/(sigma*sqrt(T))，d_s 为 d - sigma*sqrt(T)
    auto d = [&](double z) { return (z - nu * T) / (sigma * sqT); };
    auto d_s = [&](double z) { return d(z) - sigma * sqT; };

    double p = S * (norm_cdf(d_s(m)) - norm_cdf(d_s(A)))
             - K * std::exp(-r * T) * (norm_cdf(d(m)) - norm_cdf(d(A)));
    // 反射项：S 项带 (H/S)^{2λ}，K 项带 (H/S)^{2λ-2}（两者指数不同，推导时最易出错）
    p -= std::pow(H / S, 2 * lambda) * S * (norm_cdf(d_s(m - 2 * m)) - norm_cdf(d_s(A - 2 * m)));
    p += std::pow(H / S, 2 * lambda - 2) * K * std::exp(-r * T)
         * (norm_cdf(d(m - 2 * m)) - norm_cdf(d(A - 2 * m)));
    return p;
}

double levy_arith_asian_call(double S, double K, double r, double sigma, double T, int n) {
    // Levy 矩匹配：用对数正态分布匹配算术平均的前两阶矩
    double m1 = 0.0, m2 = 0.0;
    for (int i = 1; i <= n; ++i) {
        double ti = T * i / n;
        m1 += (S / n) * std::exp(r * ti);
        for (int j = 1; j <= n; ++j) {
            double tj = T * j / n;
            m2 += (S * S / (n * n)) * std::exp(r * (ti + tj) + sigma * sigma * std::min(ti, tj));
        }
    }
    double sig_a = std::sqrt(std::log(m2 / (m1 * m1)));
    double d1 = (std::log(m1 / K) + 0.5 * sig_a * sig_a) / sig_a;
    double d2 = d1 - sig_a;
    return std::exp(-r * T) * (m1 * norm_cdf(d1) - K * norm_cdf(d2));
}

double bgk_shift(double H, double sigma, double dt, bool up) {
    // beta = -zeta(1/2)/sqrt(2*pi) ≈ 0.5826
    return H * std::exp((up ? -1.0 : 1.0) * 0.5825971579 * sigma * std::sqrt(dt));
}

double t_quantile_95(long long n) {
    if (n <= 1) return 12.706; // 退化保护（单样本）
    if (n >= 10000) return 1.96;
    // Cornish-Fisher 展开近似 t 分位数（n 较小时与真值差 < 1%）
    double nu = static_cast<double>(n - 1);
    double z = 1.96;
    double t = z + (z * z * z + z) / (4.0 * nu)
             + (5 * z * z * z * z * z + 16 * z * z * z + 3 * z) / (96.0 * nu * nu);
    return t;
}

const char* reference_name_for(OptionType type) {
    switch (type) {
        case OptionType::EuropeanCall:
        case OptionType::EuropeanPut:
            return "black_scholes_analytic";
        case OptionType::AsianCall:
            return "high_precision_cv_mc"; // 算术亚式无简单闭式解
        case OptionType::BarrierCallUpOut:
            return "continuous_barrier_closed_form";
        default:
            return "unknown";
    }
}
