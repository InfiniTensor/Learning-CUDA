// analytic.h - 解析参考价格（host-only，double 精度）
#pragma once

#include "types.h"

// 标准正态 CDF：Phi(x) = 0.5 * erfc(-x / sqrt(2))
double norm_cdf(double x);

// 标准正态 PDF
double norm_pdf(double x);

// Black-Scholes 解析价（欧式看涨/看跌）
double bs_price(OptionType type, double S, double K, double r, double sigma, double T);

// 几何亚式看涨（Kemna-Vorst，离散监控 t_i = iT/n, i = 1..n）
double geo_asian_call(double S, double K, double r, double sigma, double T, int n);

// 连续监控向上敲出看涨（反射原理闭式解，要求 H > S）
double barrier_up_out_call(double S, double K, double H, double r, double sigma, double T);

// 算术亚式看涨的 Levy 矩匹配近似（~+0.3% 偏差，仅作粗参考）
double levy_arith_asian_call(double S, double K, double r, double sigma, double T, int n);

// Broadie-Glasserman-Kou 障碍平移：离散监控 → 连续监控的边界修正
// 向上敲出取负号（边界向 S0 靠拢），向下敲出取正号
double bgk_shift(double H, double sigma, double dt, bool up);

// 95% 双侧置信区间的 Student-t 分位数（n >= 1e4 时取 1.96）
double t_quantile_95(long long n);

// 各期权类型参考价格的来源名称（写进结果文件）
const char* reference_name_for(OptionType type);
