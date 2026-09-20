// output.cpp - 定价结果文件与性能日志输出
#include "output.h"

#include <cstdio>
#include <stdexcept>

namespace {

const char* option_type_name(OptionType t) {
    switch (t) {
        case OptionType::EuropeanCall: return "european_call";
        case OptionType::EuropeanPut: return "european_put";
        case OptionType::AsianCall: return "asian_call";
        case OptionType::BarrierCallUpOut: return "barrier_call";
        default: return "unknown";
    }
}

const char* rng_name(RngType t) { return t == RngType::Curand ? "curand" : "custom"; }

const char* vr_name(VarReduction v) {
    switch (v) {
        case VarReduction::None: return "none";
        case VarReduction::Antithetic: return "antithetic";
        case VarReduction::ControlVariate: return "control_variate";
        default: return "unknown";
    }
}

const char* engine_name(Engine e) {
    switch (e) {
        case Engine::Cpu: return "cpu";
        case Engine::Gpu: return "gpu";
        default: return "unknown";
    }
}

std::string engine_section(Engine e) {
    return e == Engine::Gpu ? "gpu_" : "cpu_";
}

// 一次运行的结果段（engine=both 时文件内出现 gpu_ 与 cpu_ 两段）
void write_run_section(FILE* f, const SimSpec& ss, const RunReport& run) {
    const PriceResult& r = run.result;
    std::string pre = engine_section(run.engine);
    fprintf(f, "%sengine = %s\n", pre.c_str(), engine_name(run.engine));
    fprintf(f, "%sestimator = discounted_mean_payoff\n", pre.c_str());
    fprintf(f, "%sprice_estimate = %.17g\n", pre.c_str(), r.price);
    fprintf(f, "%smc_standard_error = %.17g\n", pre.c_str(), r.std_error);
    fprintf(f, "%sci95_low = %.17g\n", pre.c_str(), r.ci95_low);
    fprintf(f, "%sci95_high = %.17g\n", pre.c_str(), r.ci95_high);
    fprintf(f, "%sci_level = 0.95\n", pre.c_str());
    fprintf(f, "%sci_quantile_source = student_t\n", pre.c_str());
    fprintf(f, "%ssamples = %lld\n", pre.c_str(), r.samples);
    fprintf(f, "%sreference_name = %s\n", pre.c_str(), r.reference_name);
    fprintf(f, "%sreference_price = %.17g\n", pre.c_str(), r.reference);
    fprintf(f, "%sabs_error = %.17g\n", pre.c_str(), r.abs_error);
    fprintf(f, "%srel_error = %.17g\n", pre.c_str(), r.rel_error);
    fprintf(f, "%sz_score = %.17g\n", pre.c_str(), r.z_score);
    fprintf(f, "%sconvergence_error_times_sqrtN = %.17g\n", pre.c_str(), r.error_times_sqrt_n);
    fprintf(f, "%svariance_reduction_factor = %.17g\n", pre.c_str(), r.var_reduction_factor);
    if (ss.vr == VarReduction::ControlVariate)
        fprintf(f, "%scontrol_variate_beta = %.17g\n", pre.c_str(), r.beta);
    if (run.cpu_rng_fallback)
        fprintf(f, "%scpu_rng_fallback = true\n", pre.c_str());
    fprintf(f, "\n");
}

} // namespace

void write_pricing_result(const std::string& path, const OptionSpec& os, const SimSpec& ss,
                          const EngineOptions& opts, const std::vector<RunReport>& runs) {
    FILE* f = std::fopen(path.c_str(), "w");
    if (!f) throw std::runtime_error("cannot write result file: " + path);
    fprintf(f, "# 金融衍生品定价结果（自动生成）\n");
    fprintf(f, "option_type = %s\n", option_type_name(os.type));
    fprintf(f, "spot = %.17g\n", os.spot);
    fprintf(f, "strike = %.17g\n", os.strike);
    fprintf(f, "risk_free_rate = %.17g\n", os.risk_free_rate);
    fprintf(f, "volatility = %.17g\n", os.volatility);
    fprintf(f, "maturity = %.17g\n", os.maturity);
    if (os.type == OptionType::BarrierCallUpOut)
        fprintf(f, "barrier = %.17g\n", os.barrier);
    fprintf(f, "barrier_bgk_correction = %s\n", opts.bgk ? "on" : "off");
    fprintf(f, "num_paths = %lld\n", ss.num_paths);
    fprintf(f, "num_steps = %d\n", ss.num_steps);
    fprintf(f, "effective_steps = %d\n", effective_steps(os, ss, opts));
    fprintf(f, "seed = %llu\n", ss.seed);
    fprintf(f, "rng = %s\n", rng_name(ss.rng));
    fprintf(f, "variance_reduction = %s\n", vr_name(ss.vr));
    fprintf(f, "\n");
    for (const RunReport& run : runs) write_run_section(f, ss, run);
    std::fclose(f);
}

void write_perf_log(const std::string& path, const OptionSpec& os, const SimSpec& ss,
                    const EngineOptions& opts, const std::vector<RunReport>& runs,
                    const RunReport* cpu_run) {
    FILE* f = std::fopen(path.c_str(), "w");
    if (!f) throw std::runtime_error("cannot write perf log: " + path);
    fprintf(f, "# 性能日志（自动生成）\n");
    fprintf(f, "option_type = %s\n", option_type_name(os.type));
    fprintf(f, "num_paths = %lld\n", ss.num_paths);
    fprintf(f, "num_steps = %d\n", ss.num_steps);
    fprintf(f, "effective_steps = %d\n", effective_steps(os, ss, opts));
    fprintf(f, "rng = %s\n", rng_name(ss.rng));
    fprintf(f, "variance_reduction = %s\n", vr_name(ss.vr));

    for (const RunReport& run : runs) {
        if (run.engine != Engine::Gpu) continue;
        const Timing& t = run.timing;
        std::string pre = engine_section(run.engine);
        fprintf(f, "\n[gpu engine]\n");
        fprintf(f, "%sdevice = %s\n", pre.c_str(), t.device_name);
        fprintf(f, "%scompute_capability = %d.%d\n", pre.c_str(), t.compute_major, t.compute_minor);
        fprintf(f, "%ssm_count = %d\n", pre.c_str(), t.sm_count);
        fprintf(f, "%scuda_runtime_version = %d\n", pre.c_str(), t.runtime_version);
        fprintf(f, "%scuda_driver_version = %d\n", pre.c_str(), t.driver_version);
        fprintf(f, "%sgranularity_block_size = %d\n", pre.c_str(), t.block_size);
        fprintf(f, "%sgranularity_grid_blocks = %d\n", pre.c_str(), t.grid_blocks);
        fprintf(f, "%sgranularity_blocks_per_sm = %d\n", pre.c_str(), t.blocks_per_sm);
        fprintf(f, "%sgranularity_total_threads = %d\n", pre.c_str(), t.total_threads);
        fprintf(f, "%sgranularity_registers_per_thread = %d\n", pre.c_str(), t.registers_per_thread);
        fprintf(f, "%sgranularity_shared_bytes_per_block = %d\n", pre.c_str(), t.shared_bytes_per_block);
        fprintf(f, "%skernel_time_ms = %.6f\n", pre.c_str(), t.kernel_ms);
        fprintf(f, "%sfinalize_time_ms = %.6f\n", pre.c_str(), t.finalize_ms);
        fprintf(f, "%sreduction_time_ms = %.6f\n", pre.c_str(), t.reduction_ms);
        fprintf(f, "%sh2d_ms = %.6f\n", pre.c_str(), t.h2d_ms);
        fprintf(f, "%sd2h_ms = %.6f\n", pre.c_str(), t.d2h_ms);
        fprintf(f, "%shost_setup_ms = %.6f\n", pre.c_str(), t.host_setup_ms);
        fprintf(f, "%stotal_runtime_ms = %.6f\n", pre.c_str(), t.total_ms);
        fprintf(f, "%spaths_per_second = %.3e\n", pre.c_str(), run.paths_per_sec);
        long long steps = run.paths_run * (long long)effective_steps(os, ss, opts);
        fprintf(f, "%ssteps_per_second = %.3e\n", pre.c_str(),
                steps / (t.total_ms / 1000.0));
    }

    if (cpu_run) {
        fprintf(f, "\n[cpu baseline]\n");
        fprintf(f, "cpu_engine = single_thread\n");
        fprintf(f, "cpu_threads = %d\n", opts.cpu_threads);
        fprintf(f, "cpu_paths = %lld\n", cpu_run->paths_run);
        fprintf(f, "cpu_time_ms = %.6f\n", cpu_run->timing.total_ms);
        fprintf(f, "cpu_paths_per_second = %.3e\n", cpu_run->paths_per_sec);
        const RunReport* gpu = nullptr;
        for (const RunReport& run : runs)
            if (run.engine == Engine::Gpu) gpu = &run;
        if (gpu && cpu_run->paths_per_sec > 0.0) {
            fprintf(f, "speedup_vs_cpu_single_thread = %.3f\n",
                    gpu->paths_per_sec / cpu_run->paths_per_sec);
            fprintf(f, "speedup_basis = paths_per_second_ratio_same_option\n");
        }
    }
    std::fclose(f);
}
