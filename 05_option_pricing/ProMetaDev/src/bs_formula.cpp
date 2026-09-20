// ============================================================
// bs_formula.cpp - Black-Scholes 解析公式 + 几何亚式 (Kemna-Vorst)
//                 + 障碍期权 (Reiner-Rubinstein 1991) + Greeks
//         + OptionParams 静态方法 (parse_option_type / to_string)
// ============================================================
#include "bs_formula.h"
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <cctype>

// -------- OptionParams 静态方法 --------
static std::string to_upper(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return std::toupper(c); });
    return s;
}

OptionType OptionParams::parse_option_type(const std::string& s) {
    std::string u = to_upper(s);
    if (u == "EUROPEAN_CALL")
        return OptionType::EUROPEAN_CALL;
    if (u == "EUROPEAN_PUT")
        return OptionType::EUROPEAN_PUT;
    if (u == "ASIAN_CALL")
        return OptionType::ASIAN_CALL;
    if (u == "ASIAN_PUT")
        return OptionType::ASIAN_PUT;
    if (u == "BARRIER_CALL")
        return OptionType::BARRIER_CALL;
    if (u == "BARRIER_PUT")
        return OptionType::BARRIER_PUT;
    return OptionType::UNKNOWN;
}

const char* OptionParams::to_string(OptionType t) {
    switch (t) {
    case OptionType::EUROPEAN_CALL:
        return "EUROPEAN_CALL";
    case OptionType::EUROPEAN_PUT:
        return "EUROPEAN_PUT";
    case OptionType::ASIAN_CALL:
        return "ASIAN_CALL";
    case OptionType::ASIAN_PUT:
        return "ASIAN_PUT";
    case OptionType::BARRIER_CALL:
        return "BARRIER_CALL";
    case OptionType::BARRIER_PUT:
        return "BARRIER_PUT";
    default:
        return "UNKNOWN";
    }
}

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// -------- 标准正态 CDF / PDF --------
double norm_cdf(double x) {
    return 0.5 * std::erfc(-x / std::sqrt(2.0));
}
double norm_pdf(double x) {
    return std::exp(-0.5 * x * x) / std::sqrt(2.0 * M_PI);
}

// -------- 欧式 BS 公式 --------
double bs_european_call(double S, double K, double r, double sigma, double T) {
    if (T <= 0.0)
        return std::max(S - K, 0.0);
    double d1 = (std::log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * std::sqrt(T));
    double d2 = d1 - sigma * std::sqrt(T);
    return S * norm_cdf(d1) - K * std::exp(-r * T) * norm_cdf(d2);
}
double bs_european_put(double S, double K, double r, double sigma, double T) {
    if (T <= 0.0)
        return std::max(K - S, 0.0);
    double d1 = (std::log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * std::sqrt(T));
    double d2 = d1 - sigma * std::sqrt(T);
    return K * std::exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1);
}

// -------- 几何平均亚式期权 (Kemna-Vorst 连续监控) --------
// 等价于 Black-Scholes 调整波动率后的欧式期权
//   sigma_g = sigma / sqrt(3)
//   b_g     = r/2 - sigma^2/12      (有效 drift)
//   call = S0 * exp((b_g - r)*T) * N(d1g) - K*exp(-rT)*N(d2g)
double bs_geometric_asian_price(const OptionParams& p) {
    const double S = p.spot, K = p.strike, r = p.risk_free;
    const double sigma = p.volatility, T = p.maturity;
    if (T <= 0.0) {
        double payoff =
            (p.type == OptionType::ASIAN_CALL) ? std::max(S - K, 0.0) : std::max(K - S, 0.0);
        return payoff;
    }
    double sigma_g = sigma / std::sqrt(3.0);
    double b_g = 0.5 * r - sigma * sigma / 12.0;
    double d1 = (std::log(S / K) + (b_g + 0.5 * sigma_g * sigma_g) * T) / (sigma_g * std::sqrt(T));
    double d2 = d1 - sigma_g * std::sqrt(T);
    double disc = std::exp((b_g - r) * T);
    if (p.type == OptionType::ASIAN_CALL)
        return S * disc * norm_cdf(d1) - K * std::exp(-r * T) * norm_cdf(d2);
    else
        return K * std::exp(-r * T) * norm_cdf(-d2) - S * disc * norm_cdf(-d1);
}

// ============================================================
// 障碍期权 Reiner-Rubinstein (1991) 解析定价
// 参考: QuantLib AnalyticBarrierEngine 实现
// 假设: 连续监控, 无股息 (q=0), 无 rebate
// 支持 8 种类型: up/down x in/out x call/put
//
// 核心辅助函数 (符号约定与 QuantLib 一致):
//   mu       = (r - sigma^2/2) / sigma^2
//   stdDev   = sigma * sqrt(T)
//   muSigma  = (1 + mu) * stdDev   (= (r + sigma^2/2) * sqrt(T) / sigma)
//   phi      = +1 (call), -1 (put)
//   eta      = +1 (down barrier), -1 (up barrier)
//
//   A(phi) = 普通欧式 BS 期权 (vanilla)
//   B(phi) = 用 H 替代 K 的 BS 期权
//   C(eta, phi) = 反射项 (y1 参数, 用 K)
//   D(eta, phi) = 反射项 (y2 参数, 用 H)
//
//   HS = H/S
//   powHS0 = HS^(2*mu)            (用于 K 项)
//   powHS1 = HS^(2*mu + 2)        (用于 S 项, = HS^(2*(mu+1)))
//   x1 = log(S/K)/stdDev + muSigma
//   x2 = log(S/H)/stdDev + muSigma
//   y1 = log(H^2/(S*K))/stdDev + muSigma
//   y2 = log(H/S)/stdDev + muSigma
// ============================================================
double bs_barrier_price(const OptionParams& p) {
    const double S = p.spot, K = p.strike, H = p.barrier;
    const double r = p.risk_free, sigma = p.volatility, T = p.maturity;

    // 边界: T <= 0 时, 障碍期权退化为 (条件) payoff
    if (T <= 0.0) {
        bool is_call = (p.type == OptionType::BARRIER_CALL);
        return is_call ? std::max(S - K, 0.0) : std::max(K - S, 0.0);
    }
    if (sigma <= 0.0 || S <= 0.0 || K <= 0.0 || H <= 0.0) {
        throw std::runtime_error("bs_barrier_price: invalid parameters (must be positive)");
    }

    // 检查障碍是否已被触发 (此时 knock-out = 0, knock-in = vanilla)
    bool is_up = (p.barrier_dir == BarrierDir::UP);
    bool is_call = (p.type == OptionType::BARRIER_CALL);
    bool is_knock_in = (p.barrier_t == BarrierType::KNOCK_IN);
    bool triggered = is_up ? (S >= H) : (S <= H);
    if (triggered) {
        // 障碍已触及: knock-out 归零, knock-in 等于 vanilla
        if (is_knock_in) {
            return is_call ? bs_european_call(S, K, r, sigma, T)
                           : bs_european_put(S, K, r, sigma, T);
        } else {
            return 0.0;
        }
    }

    const double q = 0.0; // 无股息
    const double stdDev = sigma * std::sqrt(T);
    const double mu = (r - q) / (sigma * sigma) - 0.5; // = (r - sigma^2/2)/sigma^2
    const double muSigma = (1.0 + mu) * stdDev;
    const double divDisc = std::exp(-q * T); // = 1.0
    const double rfDisc = std::exp(-r * T);

    const double phi = is_call ? 1.0 : -1.0;
    const double eta = is_up ? -1.0 : 1.0; // down=+1, up=-1

    // ---- 辅助函数 A(phi): vanilla BS 期权 ----
    auto A = [&](double phi_) -> double {
        double x1 = std::log(S / K) / stdDev + muSigma;
        double N1 = norm_cdf(phi_ * x1);
        double N2 = norm_cdf(phi_ * (x1 - stdDev));
        return phi_ * (S * divDisc * N1 - K * rfDisc * N2);
    };

    // ---- 辅助函数 B(phi): 用 H 替代 K ----
    auto B = [&](double phi_) -> double {
        double x2 = std::log(S / H) / stdDev + muSigma;
        double N1 = norm_cdf(phi_ * x2);
        double N2 = norm_cdf(phi_ * (x2 - stdDev));
        return phi_ * (S * divDisc * N1 - K * rfDisc * N2);
    };

    // ---- 辅助函数 C(eta, phi): 反射项 1 (y1 参数) ----
    auto C = [&](double eta_, double phi_) -> double {
        double HS = H / S;
        double powHS0 = std::pow(HS, 2.0 * mu);
        double powHS1 = powHS0 * HS * HS;
        double y1 = std::log(H * HS / K) / stdDev + muSigma; // = log(H^2/(S*K))/stdDev + muSigma
        double N1 = norm_cdf(eta_ * y1);
        double N2 = norm_cdf(eta_ * (y1 - stdDev));
        // 处理 0 * inf 极限 (N=0 时 powHS 可能无穷, 极限应为 0)
        double term1 = (N1 == 0.0) ? 0.0 : (powHS1 * N1);
        double term2 = (N2 == 0.0) ? 0.0 : (powHS0 * N2);
        return phi_ * (S * divDisc * term1 - K * rfDisc * term2);
    };

    // ---- 辅助函数 D(eta, phi): 反射项 2 (y2 参数) ----
    auto D = [&](double eta_, double phi_) -> double {
        double HS = H / S;
        double powHS0 = std::pow(HS, 2.0 * mu);
        double powHS1 = powHS0 * HS * HS;
        double y2 = std::log(H / S) / stdDev + muSigma; // = log(HS)/stdDev + muSigma
        double N1 = norm_cdf(eta_ * y2);
        double N2 = norm_cdf(eta_ * (y2 - stdDev));
        double term1 = (N1 == 0.0) ? 0.0 : (powHS1 * N1);
        double term2 = (N2 == 0.0) ? 0.0 : (powHS0 * N2);
        return phi_ * (S * divDisc * term1 - K * rfDisc * term2);
    };

    bool K_ge_H = (K >= H);
    double price = 0.0;

    // ---- 8 种障碍期权公式 (无 rebate, E=F=0) ----
    // 参考: QuantLib AnalyticBarrierEngine::calculate()
    if (is_call) {
        if (is_knock_in) {
            if (!is_up) { // Down-and-in call
                if (K_ge_H)
                    price = C(eta, phi); // C(1,1)
                else
                    price = A(phi) - B(phi) + D(eta, phi); // A(1)-B(1)+D(1,1)
            } else {                                       // Up-and-in call
                if (K_ge_H)
                    price = A(phi); // A(1)
                else
                    price = B(phi) - C(eta, phi) + D(eta, phi); // B(1)-C(-1,1)+D(-1,1)
            }
        } else {          // knock-out call
            if (!is_up) { // Down-and-out call
                if (K_ge_H)
                    price = A(phi) - C(eta, phi); // A(1)-C(1,1)
                else
                    price = B(phi) - D(eta, phi); // B(1)-D(1,1)
            } else {                              // Up-and-out call
                if (K_ge_H)
                    price = 0.0; // 0
                else
                    price =
                        A(phi) - B(phi) + C(eta, phi) - D(eta, phi); // A(1)-B(1)+C(-1,1)-D(-1,1)
            }
        }
    } else { // Put
        if (is_knock_in) {
            if (!is_up) { // Down-and-in put
                if (K_ge_H)
                    price = B(phi) - C(eta, phi) + D(eta, phi); // B(-1)-C(1,-1)+D(1,-1)
                else
                    price = A(phi); // A(-1)
            } else {                // Up-and-in put
                if (K_ge_H)
                    price = A(phi) - B(phi) + D(eta, phi); // A(-1)-B(-1)+D(-1,-1)
                else
                    price = C(eta, phi); // C(-1,-1)
            }
        } else {          // knock-out put
            if (!is_up) { // Down-and-out put
                if (K_ge_H)
                    price =
                        A(phi) - B(phi) + C(eta, phi) - D(eta, phi); // A(-1)-B(-1)+C(1,-1)-D(1,-1)
                else
                    price = 0.0; // 0
            } else {             // Up-and-out put
                if (K_ge_H)
                    price = B(phi) - D(eta, phi); // B(-1)-D(-1,-1)
                else
                    price = A(phi) - C(eta, phi); // A(-1)-C(-1,-1)
            }
        }
    }

    // 非负保护 (数值误差可能导致微小负值)
    if (price < 0.0)
        price = 0.0;
    return price;
}

// -------- BS Greeks (欧式) --------
GreeksBS bs_greeks(double S, double K, double r, double sigma, double T, bool is_call) {
    GreeksBS g{};
    if (T <= 0.0 || sigma <= 0.0) {
        g.Delta = is_call ? (S > K ? 1.0 : 0.0) : (S < K ? -1.0 : 0.0);
        return g;
    }
    double sqrtT = std::sqrt(T);
    double d1 = (std::log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrtT);
    double d2 = d1 - sigma * sqrtT;
    double pdf_d1 = norm_pdf(d1);
    double Nd1 = norm_cdf(d1), Nd2 = norm_cdf(d2);

    g.Delta = is_call ? Nd1 : (Nd1 - 1.0);
    g.Gamma = pdf_d1 / (S * sigma * sqrtT);
    g.Vega = S * pdf_d1 * sqrtT; // 注意: Vega 是 dPrice/dSigma (未除 100)
    if (is_call)
        g.Theta = -(S * pdf_d1 * sigma) / (2.0 * sqrtT) - r * K * std::exp(-r * T) * Nd2;
    else
        g.Theta = -(S * pdf_d1 * sigma) / (2.0 * sqrtT) + r * K * std::exp(-r * T) * norm_cdf(-d2);
    g.Rho = is_call ? K * T * std::exp(-r * T) * Nd2 : -K * T * std::exp(-r * T) * norm_cdf(-d2);
    return g;
}

// -------- 解析参考价格 (按期权类型分派) --------
double bs_reference_price(const OptionParams& p) {
    switch (p.type) {
    case OptionType::EUROPEAN_CALL:
        return bs_european_call(p.spot, p.strike, p.risk_free, p.volatility, p.maturity);
    case OptionType::EUROPEAN_PUT:
        return bs_european_put(p.spot, p.strike, p.risk_free, p.volatility, p.maturity);
    case OptionType::ASIAN_CALL:
    case OptionType::ASIAN_PUT:
        // 几何亚式解析解 (作为控制变量基准, 算术亚式无解析解)
        return bs_geometric_asian_price(p);
    case OptionType::BARRIER_CALL:
    case OptionType::BARRIER_PUT:
        // 障碍期权 Reiner-Rubinstein (1991) 解析解
        return bs_barrier_price(p);
    default:
        throw std::runtime_error("bs_reference_price: unknown option type");
    }
}
