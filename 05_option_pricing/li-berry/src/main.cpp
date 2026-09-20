// main.cpp - 命令行入口：解析参数 → 定价（GPU/CPU）→ 输出结果文件与性能日志
//
// 用法：
//   option_pricer --option <期权参数文件> --sim <模拟参数文件> [选项]
// 选项：
//   --mode cpu|gpu|both   计算引擎（默认 gpu）
//   --out-dir <dir>       输出目录（默认 results）
//   --block N             GPU 线程块大小（默认 256）
//   --grid N              GPU 网格块数（默认 0 = 按占用率自动）
//   --cpu-paths N         CPU 引擎路径数上限（默认 500000）
//   --cpu-threads N       CPU 引擎线程数（默认 1；>1 启用 OpenMP）
//   --euro-steps N        欧式模拟步数（默认 1 = 精确终值公式）
//   --bgk                 障碍期权 BGK 离散→连续边界修正
//   --sweep               N = 1e4..1e7 收敛扫描（stdout 输出 CSV 行 + 拟合斜率）
//   --dump-rng N          转储 RNG 流（每路径前 16 个 u32，用于 CPU/GPU 一致性验证）
//   --quiet               抑制控制台输出
// 退出码：0 成功；1 参数/配置错误；2 CUDA 错误；3 数值自检失败（|z|>3.5，仅欧式）
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

#include "analytic.h"
#include "config.h"
#include "cpu_engine.h"
#include "engine_common.h"
#include "gpu_engine.cuh"
#include "output.h"
#include "rng.h"
#include "stats.h"
#include "types.h"

namespace {

struct CliArgs {
    std::string option_file;
    std::string sim_file;
    Engine mode = Engine::Gpu;
    std::string out_dir = "results";
    EngineOptions opts;
};

void print_usage(FILE* f) {
    std::fprintf(f,
        "用法: option_pricer --option <期权参数文件> --sim <模拟参数文件> [选项]\n"
        "  --mode cpu|gpu|both   计算引擎（默认 gpu）\n"
        "  --out-dir <dir>       输出目录（默认 results）\n"
        "  --block N             GPU 线程块大小（默认 256）\n"
        "  --grid N              GPU 网格块数（默认 0 = 按占用率自动）\n"
        "  --cpu-paths N         CPU 引擎路径数上限（默认 500000）\n"
        "  --cpu-threads N       CPU 引擎线程数（默认 1；>1 启用 OpenMP）\n"
        "  --euro-steps N        欧式模拟步数（默认 1 = 精确终值公式）\n"
        "  --bgk                 障碍期权 BGK 边界修正\n"
        "  --sweep               N = 1e4..1e7 收敛扫描\n"
        "  --dump-rng N          转储 RNG 流（CPU/GPU 一致性验证）\n"
        "  --quiet               抑制控制台输出\n");
}

CliArgs parse_args(int argc, char** argv) {
    CliArgs a;
    auto need_value = [&](int& i, const char* flag) -> const char* {
        if (i + 1 >= argc) throw std::runtime_error(std::string(flag) + " 需要参数");
        return argv[++i];
    };
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--option") a.option_file = need_value(i, arg.c_str());
        else if (arg == "--sim") a.sim_file = need_value(i, arg.c_str());
        else if (arg == "--mode") {
            std::string m = need_value(i, arg.c_str());
            if (m == "cpu") a.mode = Engine::Cpu;
            else if (m == "gpu") a.mode = Engine::Gpu;
            else if (m == "both") a.mode = Engine::Both;
            else throw std::runtime_error("--mode 需为 cpu/gpu/both");
        } else if (arg == "--out-dir") a.out_dir = need_value(i, arg.c_str());
        else if (arg == "--block") a.opts.block = std::stoi(need_value(i, arg.c_str()));
        else if (arg == "--grid") a.opts.grid_blocks = std::stoi(need_value(i, arg.c_str()));
        else if (arg == "--cpu-paths") a.opts.cpu_paths = std::stoll(need_value(i, arg.c_str()));
        else if (arg == "--cpu-threads") a.opts.cpu_threads = std::stoi(need_value(i, arg.c_str()));
        else if (arg == "--euro-steps") a.opts.euro_steps = std::stoi(need_value(i, arg.c_str()));
        else if (arg == "--bgk") a.opts.bgk = true;
        else if (arg == "--sweep") a.opts.sweep = true;
        else if (arg == "--dump-rng") {
            a.opts.dump_rng = true;
            a.opts.dump_rng_count = std::stoi(need_value(i, arg.c_str()));
        } else if (arg == "--quiet") a.opts.quiet = true;
        else if (arg == "-h" || arg == "--help") { print_usage(stdout); std::exit(0); }
        else throw std::runtime_error("未知参数: " + arg);
    }
    if (a.option_file.empty() || a.sim_file.empty()) {
        print_usage(stderr);
        throw std::runtime_error("需要 --option 与 --sim");
    }
    if (a.opts.block < 32 || a.opts.block > 1024 || (a.opts.block & 31) != 0)
        throw std::runtime_error("--block 需为 32 的倍数且在 [32,1024]");
    if (a.opts.dump_rng && (a.opts.dump_rng_count <= 0 || a.opts.dump_rng_count % 16 != 0))
        throw std::runtime_error("--dump-rng N 的 N 需为 16 的正倍数");
    return a;
}

// 与主运行去相关的参考种子（避免同流相关误差掩盖真实误差）
constexpr unsigned long long kRefSeedXor = 0x9E3779B97F4A7C15ull;

// 参考价格与来源名称
struct Reference {
    double value;
    const char* name;
};

// 亚式参考：GPU 高精度 CV-MC（1M 路径、独立种子），与主运行误差去相关
double asian_reference_price(const OptionSpec& os, const SimSpec& ss, const EngineOptions& opts) {
    SimSpec ref_sim = ss;
    ref_sim.num_paths = 1000000;
    ref_sim.seed ^= kRefSeedXor;
    ref_sim.rng = RngType::Curand;
    ref_sim.vr = VarReduction::ControlVariate;
    OptionSpec os_copy = os; // BGK 不影响亚式
    const OptionParams p = make_params(os_copy, ref_sim, opts);
    RawRun run = run_gpu(os_copy, ref_sim, opts, p);
    return run.acc.sum / static_cast<double>(units_for(ref_sim));
}

Reference compute_reference(const OptionSpec& os, const SimSpec& ss, const EngineOptions& opts) {
    switch (os.type) {
        case OptionType::EuropeanCall:
        case OptionType::EuropeanPut:
            return {bs_price(os.type, os.spot, os.strike, os.risk_free_rate, os.volatility,
                             os.maturity),
                    reference_name_for(os.type)};
        case OptionType::BarrierCallUpOut:
            return {barrier_up_out_call(os.spot, os.strike, os.barrier, os.risk_free_rate,
                                        os.volatility, os.maturity),
                    reference_name_for(os.type)};
        case OptionType::AsianCall:
            return {asian_reference_price(os, ss, opts), reference_name_for(os.type)};
    }
    throw std::runtime_error("unknown option type");
}

// 组装 RunReport（结果 + 路径速率）。样本数以实际运行路径数为准（CPU 有上限截断）
RunReport assemble(Engine engine, const RawRun& raw, const SimSpec& ss,
                   const Reference& ref) {
    RunReport r;
    r.engine = engine;
    r.timing = raw.timing;
    r.paths_run = raw.paths_run;
    r.paths_per_sec = raw.timing.total_ms > 0.0
                          ? raw.paths_run / (raw.timing.total_ms / 1000.0)
                          : 0.0;
    r.cpu_rng_fallback = raw.cpu_rng_fallback;
    const long long samples = ss.vr == VarReduction::Antithetic ? raw.paths_run / 2
                                                                : raw.paths_run;
    r.result = make_result(raw.acc, samples, ref.value, ref.name, ss.vr, raw.beta);
    return r;
}

// 控制台摘要
void print_summary(const RunReport& r) {
    std::printf("[%s] price=%.6f se=%.6f ref=%.6f z=%+.2f vr=%.1fx time=%.2fms paths/s=%.3e\n",
                r.engine == Engine::Gpu ? "gpu" : "cpu", r.result.price, r.result.std_error,
                r.result.reference, r.result.z_score, r.result.var_reduction_factor,
                r.timing.total_ms, r.paths_per_sec);
}

// 数值自检：欧式有精确解析解，|z|>3.5 视为实现缺陷（概率 ~5e-4）
int numeric_self_check(const OptionSpec& os, const RunReport& r) {
    bool exact_ref = (os.type == OptionType::EuropeanCall || os.type == OptionType::EuropeanPut);
    if (exact_ref && r.result.z_score > 3.5) {
        std::fprintf(stderr, "[error] numeric self-check failed: |z| = %.2f > 3.5 "
                             "(price %.6f vs reference %.6f, se %.6f)\n",
                     r.result.z_score, r.result.price, r.result.reference, r.result.std_error);
        return 3;
    }
    return 0;
}

int run_sweep(const OptionSpec& os, const SimSpec& ss, const EngineOptions& opts,
              const Reference& ref, const OptionParams& params) {
    std::printf("# N price std_error abs_error z_score error_times_sqrtN se_times_sqrtN\n");
    std::vector<double> logn, loge, logse;
    for (long long N : {10000ll, 100000ll, 1000000ll, 10000000ll}) {
        SimSpec s2 = ss;
        s2.num_paths = N; // params 不变：num_steps/有效步数与 N 无关
        RawRun raw = run_gpu(os, s2, opts, params);
        RunReport r = assemble(Engine::Gpu, raw, s2, ref);
        const double se_sqrtn = r.result.std_error * std::sqrt(static_cast<double>(N));
        std::printf("%lld %.9g %.9g %.9g %.9g %.9g %.9g\n", N, r.result.price,
                    r.result.std_error, r.result.abs_error, r.result.z_score,
                    r.result.error_times_sqrt_n, se_sqrtn);
        logn.push_back(std::log10(static_cast<double>(N)));
        if (r.result.abs_error > 0.0) loge.push_back(std::log10(r.result.abs_error));
        logse.push_back(std::log10(r.result.std_error));
    }
    // 最小二乘拟合 log y ~ a + b·logN
    auto fit_slope = [](const std::vector<double>& x, const std::vector<double>& y) {
        double mx = 0, my = 0;
        for (size_t i = 0; i < x.size(); ++i) { mx += x[i]; my += y[i]; }
        mx /= x.size();
        my /= y.size();
        double sxx = 0, sxy = 0;
        for (size_t i = 0; i < x.size(); ++i) {
            sxx += (x[i] - mx) * (x[i] - mx);
            sxy += (x[i] - mx) * (y[i] - my);
        }
        return sxy / sxx;
    };
    // 理论收敛：SE ∝ 1/sqrt(N)，斜率确定性为 -0.5（误差斜率受单次实现噪声主导，仅参考）
    std::printf("# se_convergence_slope %.3f (expect ≈ -0.5, deterministic)\n",
                fit_slope(logn, logse));
    if (loge.size() >= 2)
        std::printf("# error_convergence_slope %.3f (noise-dominated, reference only)\n",
                    fit_slope(logn, loge));
    return 0;
}

void dump_cpu_rng_stream(const SimSpec& ss, int count) {
    const int paths = count / 16;
    if (ss.rng == RngType::Curand)
        std::fprintf(stderr, "[warn] cpu rng dump always uses custom rng (curand 无 CPU 对应)\n");
    for (int p = 0; p < paths; ++p) {
        XS128State st;
        xs128_seed(st, ss.seed, static_cast<unsigned long long>(p));
        std::printf("path %d", p);
        for (int i = 0; i < 16; ++i)
            std::printf(" %016llx", static_cast<unsigned long long>(xs128_next_u32(st)));
        std::printf("\n");
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        CliArgs args = parse_args(argc, argv);
        const OptionSpec os = parse_option_file(args.option_file);
        const SimSpec ss = parse_sim_file(args.sim_file);
        const OptionParams params = make_params(os, ss, args.opts);

        if (args.opts.dump_rng) {
            if (args.mode == Engine::Cpu) dump_cpu_rng_stream(ss, args.opts.dump_rng_count);
            else dump_gpu_rng_stream(ss, args.opts.dump_rng_count);
            return 0;
        }

        const Reference ref = compute_reference(os, ss, args.opts);

        if (args.opts.sweep) return run_sweep(os, ss, args.opts, ref, params);

        std::vector<RunReport> runs;
        const RunReport* cpu_run = nullptr;
        if (args.mode == Engine::Gpu || args.mode == Engine::Both) {
            RawRun raw = run_gpu(os, ss, args.opts, params);
            RunReport r = assemble(Engine::Gpu, raw, ss, ref);
            if (!args.opts.quiet) print_summary(r);
            runs.push_back(r);
        }
        if (args.mode == Engine::Cpu || args.mode == Engine::Both) {
            RawRun raw = run_cpu(os, ss, args.opts, params);
            RunReport r = assemble(Engine::Cpu, raw, ss, ref);
            if (!args.opts.quiet) print_summary(r);
            runs.push_back(r);
            cpu_run = &runs.back();
        }

        std::filesystem::create_directories(args.out_dir);
        write_pricing_result(args.out_dir + "/pricing_result.txt", os, ss, args.opts, runs);
        write_perf_log(args.out_dir + "/perf_log.txt", os, ss, args.opts, runs, cpu_run);
        if (!args.opts.quiet)
            std::printf("输出: %s/pricing_result.txt, %s/perf_log.txt\n",
                        args.out_dir.c_str(), args.out_dir.c_str());

        // 数值自检（欧式精确解析解）
        for (const RunReport& r : runs) {
            int code = numeric_self_check(os, r);
            if (code != 0) return code;
        }
        return 0;
    } catch (const ConfigError& e) {
        std::fprintf(stderr, "[error] 配置错误: %s\n", e.what());
        return 1;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[error] %s\n", e.what());
        // CUDA 错误信息含 "cuda error"
        return std::strstr(e.what(), "cuda error") != nullptr ? 2 : 1;
    }
}
