// ============================================================
// main.cpp - 项目入口
//   CLI: cuda_pricing [option_params.txt] [sim_params.txt] [result.csv] [perf.log]
//        [--seed N] [--paths N] [--json] [--no-cpu] [--greeks] [--info] [--repro] [--antithetic]
//        [--cv] [--vr NONE|ANTITHETIC|CONTROL_VARIATE]
// ============================================================
#include "option_params.h"
#include "bs_formula.h"
#include "file_io.h"
#include <iostream>
#include <fstream>
#include <chrono>
#include <cmath>
#include <random>
#include <stdexcept>
#include <string>

// 在 pricing_kernels.cu 中定义
PricingResult run_monte_carlo(const OptionParams& o, const SimParams& s, double& occupancy_pct);

// -------- CPU 参考蒙特卡洛 (单线程, 用于加速比对比) --------
static PricingResult cpu_monte_carlo(const OptionParams& o, const SimParams& s) {
    PricingResult r;
    r.option_type_str = OptionParams::to_string(o.type);
    std::mt19937_64 rng(s.seed);
    std::normal_distribution<double> N01(0.0, 1.0);

    const double S0 = o.spot, K = o.strike, rf = o.risk_free;
    const double sigma = o.volatility, T = o.maturity;
    const int n = s.num_steps;
    const double dt = T / n;
    const double drift = (rf - 0.5 * sigma * sigma) * dt;
    const double sd = sigma * std::sqrt(dt);
    const double disc = std::exp(-rf * T);

    double sx = 0, sx2 = 0;
    int64_t Np = s.num_paths;
    bool is_barrier = (o.type == OptionType::BARRIER_CALL || o.type == OptionType::BARRIER_PUT);
    bool is_asian = (o.type == OptionType::ASIAN_CALL || o.type == OptionType::ASIAN_PUT);

    for (int64_t p = 0; p < Np; ++p) {
        double S = S0, sum_S = 0, sum_log = 0;
        bool hit = false;
        for (int i = 0; i < n; ++i) {
            double z = N01(rng);
            S = S * std::exp(drift + sd * z);
            sum_S += S;
            sum_log += std::log(S);
            if (is_barrier) {
                if (o.barrier_dir == BarrierDir::UP && S >= o.barrier)
                    hit = true;
                if (o.barrier_dir == BarrierDir::DOWN && S <= o.barrier)
                    hit = true;
            }
        }
        double arith = sum_S / n, geom = std::exp(sum_log / n);
        double payoff = 0.0;
        switch (o.type) {
        case OptionType::EUROPEAN_CALL:
            payoff = std::max(S - K, 0.0);
            break;
        case OptionType::EUROPEAN_PUT:
            payoff = std::max(K - S, 0.0);
            break;
        case OptionType::ASIAN_CALL:
            payoff = std::max(arith - K, 0.0);
            break;
        case OptionType::ASIAN_PUT:
            payoff = std::max(K - arith, 0.0);
            break;
        case OptionType::BARRIER_CALL:
        case OptionType::BARRIER_PUT: {
            double ep =
                (o.type == OptionType::BARRIER_CALL) ? std::max(S - K, 0.0) : std::max(K - S, 0.0);
            if (o.barrier_t == BarrierType::KNOCK_IN)
                payoff = hit ? ep : 0.0;
            else
                payoff = hit ? 0.0 : ep;
            break;
        }
        default:
            payoff = 0.0;
            break;
        }
        sx += payoff;
        sx2 += payoff * payoff;
    }
    double mean = sx / Np;
    double var = sx2 / Np - mean * mean;
    if (var < 0)
        var = 0;
    r.price = disc * mean;
    r.std_error = std::sqrt(var / Np) * disc;
    r.ci_half = 1.96 * r.std_error;
    r.ref_price = bs_reference_price(o);
    r.abs_error = std::fabs(r.price - r.ref_price);
    r.method_str = "CPU MC";
    return r;
}

static int run_main(int argc, char** argv) {
    // ---- 默认路径 ----
    std::string opt_path = "params/option_params.txt";
    std::string sim_path = "params/sim_params.txt";
    std::string res_path = "outputs/result.csv";
    std::string perf_path = "outputs/perf.log";
    bool json_out = false, no_cpu = false, want_greeks = false;
    bool info_only = false, repro_check = false;

    // ---- 第一遍: 解析位置参数 + flag (跳过 flag 的值) ----
    int positional = 0;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--json") {
            json_out = true;
            continue;
        }
        if (arg == "--no-cpu") {
            no_cpu = true;
            continue;
        }
        if (arg == "--greeks") {
            want_greeks = true;
            continue;
        }
        if (arg == "--info") {
            info_only = true;
            continue;
        }
        if (arg == "--repro") {
            repro_check = true;
            continue;
        }
        if (arg == "--antithetic") { /* overridden in 2nd pass */
            continue;
        }
        if (arg == "--cv") { /* overridden in 2nd pass */
            continue;
        }
        if (arg == "--seed" && i + 1 < argc) {
            i++;
            continue;
        }
        if (arg == "--vr" && i + 1 < argc) {
            i++;
            continue;
        }
        if (arg == "--paths" && i + 1 < argc) {
            i++;
            continue;
        }
        switch (positional) {
        case 0:
            opt_path = arg;
            break;
        case 1:
            sim_path = arg;
            break;
        case 2:
            res_path = arg;
            break;
        case 3:
            perf_path = arg;
            break;
        }
        positional++;
    }

    // ---- 加载参数 ----
    OptionParams o = parse_option_params(opt_path);
    SimParams s = parse_sim_params(sim_path);

    // ---- 第二遍: 覆盖 seed / paths / variance_reduction ----
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--seed" && i + 1 < argc)
            s.seed = (unsigned int)std::stoul(argv[++i]);
        else if (arg == "--paths" && i + 1 < argc)
            s.num_paths = std::stoll(argv[++i]);
        else if (arg == "--vr" && i + 1 < argc) {
            std::string v2 = argv[++i];
            if (v2 == "NONE")
                s.variance_reduction = VarianceReduction::NONE;
            else if (v2 == "ANTITHETIC")
                s.variance_reduction = VarianceReduction::ANTITHETIC;
            else if (v2 == "CONTROL_VARIATE")
                s.variance_reduction = VarianceReduction::CONTROL_VARIATE;
            else
                std::cerr << "[WARN] invalid --vr value: " << v2
                          << " (use NONE/ANTITHETIC/CONTROL_VARIATE)\n";
        } else if (arg == "--antithetic")
            s.variance_reduction = VarianceReduction::ANTITHETIC;
        else if (arg == "--cv")
            s.variance_reduction = VarianceReduction::CONTROL_VARIATE;
    }

    // ---- --info: 只打印 GPU 信息 ----
    std::string gpu_info = gpu_info_string();
    if (info_only) {
        std::cout << "==== GPU Info ====\n" << gpu_info << "\n";
        return 0;
    }

    std::cout << "==== CUDA Pricing Project ====\n";
    std::cout << gpu_info << "\n";
    std::cout << "Option: " << o.to_string(o.type) << "  S=" << o.spot << "  K=" << o.strike
              << "  r=" << o.risk_free << "  sigma=" << o.volatility << "  T=" << o.maturity
              << "\n";
    std::cout << "Sim:    paths=" << s.num_paths << "  steps=" << s.num_steps << "  seed=" << s.seed
              << "  VR=" << static_cast<int>(s.variance_reduction) << "  block=" << s.block_size
              << "\n";

    // ---- GPU 定价 ----
    double occ = 0.0;
    auto t_gpu0 = std::chrono::high_resolution_clock::now();
    PricingResult r = run_monte_carlo(o, s, occ);
    auto t_gpu1 = std::chrono::high_resolution_clock::now();

    // ---- 可复现性验证 (P5): 同 seed 再跑一次, 价格须 bit-identical ----
    if (repro_check) {
        double occ2 = 0.0;
        PricingResult r2 = run_monte_carlo(o, s, occ2);
        bool price_match = (r.price == r2.price);
        bool se_match = (r.std_error == r2.std_error);
        std::cout << "[Repro] run1 price=" << r.price << "  run2 price=" << r2.price
                  << "  identical=" << (price_match && se_match ? "YES" : "NO") << "\n";
    }

    // ---- 可选 CPU 参考 ----
    if (!no_cpu) {
        auto t_cpu0 = std::chrono::high_resolution_clock::now();
        PricingResult r_cpu = cpu_monte_carlo(o, s);
        auto t_cpu1 = std::chrono::high_resolution_clock::now();
        r.cpu_time_ms = std::chrono::duration<double, std::milli>(t_cpu1 - t_cpu0).count();
        r.speedup = r.cpu_time_ms / (r.gpu_time_ms > 0 ? r.gpu_time_ms : 1e-9);
        std::cout << "[CPU] price=" << r_cpu.price << "  SE=" << r_cpu.std_error
                  << "  time=" << r.cpu_time_ms << " ms\n";
    } else {
        r.cpu_time_ms = 0.0;
        r.speedup = -1.0; // -1 = N/A (--no-cpu mode)
    }

    // ---- 输出 ----
    std::cout << "[GPU] price=" << r.price << "  ref=" << r.ref_price << "  |err|=" << r.abs_error
              << "  SE=" << r.std_error << "  CI95=+-" << r.ci_half << "  method=" << r.method_str
              << "\n";
    std::cout << "      gpu_time=" << r.gpu_time_ms << " ms"
              << "  pps=" << r.paths_per_sec << "  speedup=" << r.speedup << "x"
              << "  occupancy=" << occ << "%\n";

    // ---- 希腊字母 (可选) ----
    GreeksBS greeks{};
    bool have_greeks = false;
    if (want_greeks) {
        auto tg0 = std::chrono::high_resolution_clock::now();
        greeks = compute_greeks(o, s);
        auto tg1 = std::chrono::high_resolution_clock::now();
        double gms = std::chrono::duration<double, std::milli>(tg1 - tg0).count();
        have_greeks = true;
        std::cout << "[Greeks] Delta=" << greeks.Delta << "  Gamma=" << greeks.Gamma
                  << "  Vega=" << greeks.Vega << "  Theta=" << greeks.Theta
                  << "  Rho=" << greeks.Rho << "  (fd time=" << gms << " ms)\n";
    }

    if (json_out) {
        std::string jp = res_path;
        size_t p = jp.rfind(".csv");
        if (p != std::string::npos && p + 4 == jp.size())
            jp.replace(p, 4, ".json");
        write_result_json(jp, o, s, r, occ, have_greeks ? &greeks : nullptr);
        std::cout << "[OUT] " << jp << "\n";
    } else {
        write_result_csv(res_path, o, s, r, occ, have_greeks ? &greeks : nullptr);
        std::cout << "[OUT] " << res_path << "\n";
    }
    // perf 日志: 首行写 GPU 信息
    {
        std::ofstream pf(perf_path, std::ios::app);
        pf << "# " << gpu_info << " | " << r.method_str << "\n";
    }
    write_perf_log(perf_path, o, s, r, occ, have_greeks ? &greeks : nullptr);
    std::cout << "[LOG] " << perf_path << "\n";
    return 0;
}

// main 的薄封装：把运行期异常（参数文件缺失 / 数值非法 / 参数校验失败等）
// 转成清晰的错误信息 + 非零退出码，而不是让 std::terminate 直接 SIGABRT 产生 core dump。
int main(int argc, char** argv) {
    try {
        return run_main(argc, argv);
    } catch (const std::exception& e) {
        std::cerr << "[error] " << e.what() << "\n";
        return 1;
    }
}
