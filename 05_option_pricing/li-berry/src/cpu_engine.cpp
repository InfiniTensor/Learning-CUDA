// cpu_engine.cpp - CPU 单线程蒙特卡洛基线引擎
// 与 GPU 核共用 sim_core.h 同一份数学逻辑（同参数、同 custom RNG 流），
// 保证"相同/可比采样条件下结果一致"；默认单线程，--cpu-threads > 1 启用 OpenMP
// 注意：OpenMP 归约顺序随线程数变化，严格一致性对比请用 --cpu-threads 1
#include "cpu_engine.h"

#include <chrono>
#include <cstdio>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "engine_common.h"
#include "sim_core.h"

namespace {

// CPU 端控制变量 pilot：对前 pilot_units 个单元累积 5 个统计量估计 beta
struct PilotStats {
    double sy = 0, sy2 = 0, sc = 0, sc2 = 0, syc = 0;
    long long n = 0;
};

// 对某个 unit 模拟并累加：y 为折现收益，c 为折现控制变量，yleg 为对偶第一条腿收益
template <OptionType OPT, bool ANTITHETIC>
inline void sim_unit_cpu(const OptionParams& p, const SimSpec& ss, long long u, double& y,
                         double& c, double& yleg) {
    float fy, fc, fyleg;
    if (ANTITHETIC) {
        sim_antithetic_pair<OPT, CustomGen4>(p, ss.seed, static_cast<unsigned long long>(u), fy,
                                             fc, fyleg);
    } else {
        sim_one_path<OPT, CustomGen4>(p, ss.seed, static_cast<unsigned long long>(u), fy, fc);
    }
    y = fy;
    c = fc;
    yleg = ANTITHETIC ? static_cast<double>(fyleg) : fy;
}

// pilot pass：估计 beta = Cov(y,c)/Var(c)（仅 control_variate 模式调用）
template <OptionType OPT, bool ANTITHETIC>
PilotStats cpu_pilot_pass(const OptionParams& p, const SimSpec& ss, long long pilot_units) {
    PilotStats out;
    for (long long u = 0; u < pilot_units; ++u) {
        double y, c, yleg;
        sim_unit_cpu<OPT, ANTITHETIC>(p, ss, u, y, c, yleg);
        out.sy += y;
        out.sy2 += y * y;
        out.sc += c;
        out.sc2 += c * c;
        out.syc += y * c;
        out.n += 1;
    }
    return out;
}

// main pass：全部单元。y 经控制变量调整（beta != 0 时），rawsq 存未缩减样本二阶矩
template <OptionType OPT, bool ANTITHETIC>
Accumulators cpu_main_pass(const OptionParams& p, const SimSpec& ss, long long units,
                           double beta, double c_mean, int cpu_threads) {
    Accumulators acc;
    // OpenMP 归约子句不接受结构体成员，用局部变量归约后写回
    double s = 0.0, sq = 0.0, raw = 0.0;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (cpu_threads > 1) num_threads(cpu_threads) \
    reduction(+ : s, sq, raw)
#endif
    for (long long u = 0; u < units; ++u) {
        double y, c, yleg;
        sim_unit_cpu<OPT, ANTITHETIC>(p, ss, u, y, c, yleg);
        if (beta != 0.0) y -= beta * (c - c_mean);
        s += y;
        sq += y * y;
        raw += yleg * yleg;
    }
    acc.sum = s;
    acc.sumsq = sq;
    acc.rawsq = raw;
    return acc;
}

// 模板分发辅助：在 run_cpu 的 switch 中按 <OPT, ANTITHETIC> 实例化
template <OptionType OPT>
Accumulators dispatch_engine(bool anti, const OptionParams& p, const SimSpec& ss,
                             long long units, double beta, double c_mean, int cpu_threads) {
    if (anti) return cpu_main_pass<OPT, true>(p, ss, units, beta, c_mean, cpu_threads);
    return cpu_main_pass<OPT, false>(p, ss, units, beta, c_mean, cpu_threads);
}

template <OptionType OPT>
PilotStats dispatch_pilot(bool anti, const OptionParams& p, const SimSpec& ss,
                          long long pilot_units) {
    if (anti) return cpu_pilot_pass<OPT, true>(p, ss, pilot_units);
    return cpu_pilot_pass<OPT, false>(p, ss, pilot_units);
}

} // namespace

RawRun run_cpu(const OptionSpec& os, const SimSpec& ss, const EngineOptions& opts,
               const OptionParams& params) {
    RawRun run;
    run.paths_run = std::min(ss.num_paths, opts.cpu_paths);
    const bool anti = (ss.vr == VarReduction::Antithetic);
    const long long units = anti ? run.paths_run / 2 : run.paths_run;

    run.cpu_rng_fallback = (ss.rng == RngType::Curand);
    if (run.cpu_rng_fallback && !opts.quiet)
        std::fprintf(stderr, "[warn] cpu engine always uses custom rng; "
                             "rng=curand has no cpu equivalent (results compared statistically)\n");

    auto t0 = std::chrono::steady_clock::now();

    // 控制变量 pilot：估计 beta（100k 单元上限，CPU 单线程 ~1-2s，可接受）
    double beta = 0.0;
    if (ss.vr == VarReduction::ControlVariate) {
        const long long pilot_units = units < 100000 ? units : 100000;
        PilotStats ps;
        switch (os.type) {
            case OptionType::EuropeanCall: ps = dispatch_pilot<OptionType::EuropeanCall>(anti, params, ss, pilot_units); break;
            case OptionType::EuropeanPut: ps = dispatch_pilot<OptionType::EuropeanPut>(anti, params, ss, pilot_units); break;
            case OptionType::AsianCall: ps = dispatch_pilot<OptionType::AsianCall>(anti, params, ss, pilot_units); break;
            case OptionType::BarrierCallUpOut: ps = dispatch_pilot<OptionType::BarrierCallUpOut>(anti, params, ss, pilot_units); break;
        }
        double n = static_cast<double>(ps.n);
        double var_c = (ps.sc2 - ps.sc * ps.sc / n) / (n - 1.0);
        if (var_c > 0.0) beta = (ps.syc - ps.sy * ps.sc / n) / (n - 1.0) / var_c;
    }
    const double c_mean = control_mean(os, ss);

    // main pass
    Accumulators acc;
    switch (os.type) {
        case OptionType::EuropeanCall: acc = dispatch_engine<OptionType::EuropeanCall>(anti, params, ss, units, beta, c_mean, opts.cpu_threads); break;
        case OptionType::EuropeanPut: acc = dispatch_engine<OptionType::EuropeanPut>(anti, params, ss, units, beta, c_mean, opts.cpu_threads); break;
        case OptionType::AsianCall: acc = dispatch_engine<OptionType::AsianCall>(anti, params, ss, units, beta, c_mean, opts.cpu_threads); break;
        case OptionType::BarrierCallUpOut: acc = dispatch_engine<OptionType::BarrierCallUpOut>(anti, params, ss, units, beta, c_mean, opts.cpu_threads); break;
    }

    auto t1 = std::chrono::steady_clock::now();
    run.timing.total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    run.acc = acc;
    run.beta = beta;
    return run;
}
