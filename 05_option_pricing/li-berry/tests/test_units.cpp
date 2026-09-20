// test_units.cpp - 纯 CPU 单元测试（解析解、配置解析器、RNG、统计量）
// 不依赖 GPU；黄金值来自 data/reference_values.txt（独立 Python 交叉验证）
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>

#include "analytic.h"
#include "config.h"
#include "rng.h"
#include "stats.h"
#include "types.h"

static int g_failures = 0;
static int g_checks = 0;

#define CHECK(cond)                                                        \
    do {                                                                   \
        ++g_checks;                                                        \
        if (!(cond)) {                                                     \
            ++g_failures;                                                  \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);    \
        }                                                                  \
    } while (0)

#define CHECK_NEAR(a, b, tol)                                              \
    do {                                                                   \
        ++g_checks;                                                        \
        double va_ = (a), vb_ = (b);                                       \
        if (std::fabs(va_ - vb_) > (tol)) {                                \
            ++g_failures;                                                  \
            std::printf("FAIL %s:%d: %s = %.17g vs %s = %.17g (tol %g)\n", \
                        __FILE__, __LINE__, #a, va_, #b, vb_, (double)(tol)); \
        }                                                                  \
    } while (0)

// ---------- 解析解 ----------

static void test_analytic() {
    const double S = 100, K = 100, r = 0.03, sig = 0.2, T = 1.0;
    CHECK_NEAR(bs_price(OptionType::EuropeanCall, S, K, r, sig, T), 9.413403383853016, 1e-9);
    CHECK_NEAR(bs_price(OptionType::EuropeanPut, S, K, r, sig, T), 6.457956738703835, 1e-9);
    // put-call parity
    CHECK_NEAR(bs_price(OptionType::EuropeanCall, S, K, r, sig, T)
                   - bs_price(OptionType::EuropeanPut, S, K, r, sig, T),
               S - K * std::exp(-r * T), 1e-9);
    // Kemna-Vorst 几何亚式（n=256）
    CHECK_NEAR(geo_asian_call(S, K, r, sig, T, 256), 5.102804124274063, 1e-9);
    // 连续监控向上敲出（H=120）
    CHECK_NEAR(barrier_up_out_call(S, K, 120.0, r, sig, T), 1.155369999815111, 1e-9);
    // H -> ∞ 时应收敛到 BS 看涨（反射项消失的极限自检）
    CHECK_NEAR(barrier_up_out_call(S, K, 1e6, r, sig, T),
               bs_price(OptionType::EuropeanCall, S, K, r, sig, T), 1e-6);
    // Levy 近似（+0.3% 偏差，容差 1e-3 相对）
    CHECK_NEAR(levy_arith_asian_call(S, K, r, sig, T, 256), 5.3155370371264326, 5e-3);
    // BGK 平移（常数取精确值 -ζ(1/2)/√(2π) ≈ 0.5825971579）
    CHECK_NEAR(bgk_shift(120.0, 0.2, 1.0 / 256, true), 119.12927861011492, 1e-9);
    // 正态 CDF
    CHECK_NEAR(norm_cdf(0.0), 0.5, 1e-15);
    CHECK_NEAR(norm_cdf(1.959963984540054), 0.975, 1e-9);
    CHECK_NEAR(norm_cdf(-1.959963984540054), 0.025, 1e-9);
    // t 分位数：大样本 → 1.96
    CHECK_NEAR(t_quantile_95(10000000), 1.96, 1e-12);
}

// ---------- 配置解析器 ----------

static void write_temp(const std::string& path, const std::string& content) {
    std::ofstream f(path, std::ios::binary);
    f << content;
}

static void test_config() {
    const std::string tmp = "/tmp/op_test_config.txt";

    // CRLF + BOM + 注释 + 空白 + 行内注释
    write_temp(tmp,
               "\xEF\xBB\xBFoption_type = european_call   # 行内注释\r\n"
               "spot = 100.0\r\n"
               "\r\n"
               "# 纯注释行\r\n"
               "strike=100.0\r\n"
               "risk_free_rate = 0.03\r\n"
               "volatility = 0.2\r\n"
               "maturity = 1.0\r\n");
    OptionSpec os = parse_option_file(tmp);
    CHECK(os.type == OptionType::EuropeanCall);
    CHECK_NEAR(os.spot, 100.0, 0.0);
    CHECK_NEAR(os.volatility, 0.2, 0.0);

    // 未知键 → 抛异常（带行号）
    write_temp(tmp, "option_type = european_call\nspot = 100\nstrike = 100\n"
                    "risk_free_rate = 0.03\nvolatility = 0.2\nmaturity = 1\n"
                    "barbell = 42\n");
    bool threw = false;
    try { parse_option_file(tmp); } catch (const ConfigError& e) {
        threw = true;
        CHECK(e.line == 7); // barbell 在第 7 行
    }
    CHECK(threw);

    // barrier_call 缺 barrier → 抛异常
    write_temp(tmp, "option_type = barrier_call\nspot = 100\nstrike = 100\n"
                    "risk_free_rate = 0.03\nvolatility = 0.2\nmaturity = 1\n");
    threw = false;
    try { parse_option_file(tmp); } catch (const ConfigError&) { threw = true; }
    CHECK(threw);

    // barrier <= spot → 抛异常
    write_temp(tmp, "option_type = barrier_call\nspot = 100\nstrike = 100\n"
                    "risk_free_rate = 0.03\nvolatility = 0.2\nmaturity = 1\nbarrier = 90\n");
    threw = false;
    try { parse_option_file(tmp); } catch (const ConfigError&) { threw = true; }
    CHECK(threw);

    // 对偶 + 奇数路径 → 抛异常；未知 rng / 未知方差缩减 → 抛异常
    write_temp(tmp, "num_paths = 99999\nnum_steps = 256\nseed = 1\n"
                    "rng = curand\nvariance_reduction = antithetic\n");
    threw = false;
    try { parse_sim_file(tmp); } catch (const ConfigError&) { threw = true; }
    CHECK(threw);

    write_temp(tmp, "num_paths = 1000\nnum_steps = 256\nseed = 1\nrng = mt19937\n");
    threw = false;
    try { parse_sim_file(tmp); } catch (const ConfigError&) { threw = true; }
    CHECK(threw);

    write_temp(tmp, "num_paths = 1000\nnum_steps = 256\nseed = 1\n"
                    "variance_reduction = super_control\n");
    threw = false;
    try { parse_sim_file(tmp); } catch (const ConfigError&) { threw = true; }
    CHECK(threw);

    // 缺省键用默认值
    write_temp(tmp, "num_paths = 5000\n");
    SimSpec ss = parse_sim_file(tmp);
    CHECK(ss.num_paths == 5000);
    CHECK(ss.num_steps == 256);
    CHECK(ss.seed == 1234);
    CHECK(ss.rng == RngType::Curand);
    CHECK(ss.vr == VarReduction::None);
    std::remove(tmp.c_str());
}

// ---------- RNG ----------

static void test_rng() {
    // u32_to_uniform 永不返回 0（含 x=0 边界），且落在 (0,1)
    for (unsigned int x : {0u, 1u, 255u, 256u, 0xFFFFFFFFu, 123456789u}) {
        float u = u32_to_uniform(x);
        CHECK(u > 0.0f && u < 1.0f);
    }

    // xorshift128+ 确定性：同 (seed, path) → 相同输出；不同 path → 不同输出
    XS128State a, b;
    xs128_seed(a, 1234, 7);
    xs128_seed(b, 1234, 7);
    bool same = true;
    for (int i = 0; i < 1000; ++i)
        if (xs128_next_u32(a) != xs128_next_u32(b)) same = false;
    CHECK(same);

    xs128_seed(a, 1234, 7);
    xs128_seed(b, 1234, 8);
    bool differ = false;
    for (int i = 0; i < 100; ++i)
        if (xs128_next_u32(a) != xs128_next_u32(b)) differ = true;
    CHECK(differ);

    // Box-Muller 统计量：均值 ≈ 0，方差 ≈ 1（大样本松界）
    XS128State st;
    xs128_seed(st, 42, 0);
    double m = 0, m2 = 0;
    const int N = 100000;
    for (int i = 0; i < N; ++i) {
        float u0 = u32_to_uniform(xs128_next_u32(st));
        float u1 = u32_to_uniform(xs128_next_u32(st));
        float n1, n2;
        box_muller(u0, u1, n1, n2);
        m += n1;
        m2 += n1 * n1;
    }
    m /= N;
    m2 /= N;
    CHECK(std::fabs(m) < 0.02);
    CHECK(std::fabs(m2 - 1.0) < 0.05);
}

// ---------- 统计量 ----------

static void test_stats() {
    // 已知样本：{1, 3} → mean 2, var 2, se 1
    Accumulators acc;
    acc.sum = 4.0;
    acc.sumsq = 10.0;
    acc.rawsq = 10.0;
    PriceResult r = make_result(acc, 2, 2.0, "test", VarReduction::None, 0.0);
    CHECK_NEAR(r.price, 2.0, 1e-15);
    CHECK_NEAR(r.std_error, 1.0, 1e-15);
    CHECK_NEAR(r.z_score, 0.0, 1e-15);
    CHECK(r.samples == 2);
    CHECK_NEAR(r.ci95_high - r.ci95_low, 2.0 * r.std_error * t_quantile_95(2), 1e-12);

    // 方差缩减因子：CV 使方差缩小 4 倍
    Accumulators cv;
    cv.sum = 20.0;   // 10 个样本均值 2
    cv.sumsq = 45.0; // var_used = (45 - 40)/(9) = 5/9
    cv.rawsq = 70.0; // var_plain = (70 - 40)/9 = 30/9 = 6*var_used
    PriceResult r2 = make_result(cv, 10, 2.0, "test", VarReduction::ControlVariate, 1.0);
    CHECK_NEAR(r2.var_reduction_factor, 6.0, 1e-12);
}

int main() {
    test_analytic();
    test_config();
    test_rng();
    test_stats();
    std::printf("%d checks, %d failures\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
