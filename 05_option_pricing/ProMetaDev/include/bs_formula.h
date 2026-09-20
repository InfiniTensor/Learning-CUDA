// ============================================================
// bs_formula.h - Black-Scholes 欧式解析公式 + 几何亚式 (Kemna-Vorst)
//                + 障碍期权 (Reiner-Rubinstein) + Greeks
// ============================================================
#ifndef CUDA_PRICING_BS_FORMULA_H
#define CUDA_PRICING_BS_FORMULA_H
#include "option_params.h"

double norm_cdf(double x);
double norm_pdf(double x);
double bs_european_call(double S, double K, double r, double sigma, double T);
double bs_european_put(double S, double K, double r, double sigma, double T);

// 几何平均亚式期权解析价格 (Kemna-Vorst 1990): 等价于 BS 调整波动率后的欧式
double bs_geometric_asian_price(const OptionParams& p);

// 障碍期权 Reiner-Rubinstein (1991) 解析定价 (连续监控, 无股息, 无 rebate)
// 支持 8 种类型: up/down x in/out x call/put
double bs_barrier_price(const OptionParams& p);

struct GreeksBS {
    double Delta, Gamma, Vega, Theta, Rho;
};
GreeksBS bs_greeks(double S, double K, double r, double sigma, double T, bool is_call);

// 希腊字母计算 (P4): 欧式用解析, 亚式/障碍用有限差分 MC
GreeksBS compute_greeks(const OptionParams& o, const SimParams& s);

// GPU 信息字符串 (P5: 可观测性)
std::string gpu_info_string();

// 解析参考价格 (按 OptionType 分派)
double bs_reference_price(const OptionParams& p);
#endif
