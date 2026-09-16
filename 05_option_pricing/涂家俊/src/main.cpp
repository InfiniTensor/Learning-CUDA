// 协调配置、CPU/GPU 定价、统计汇总和结果输出的命令行入口。
#include <charconv>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <vector>

#include <cuda_runtime_api.h>

#include "pricer/black_scholes.hpp"
#include "pricer/config.hpp"
#include "pricer/cpu_pricer.hpp"
#include "pricer/cuda_pricer.hpp"
#include "pricer/measurement.hpp"
#include "pricer/output.hpp"
#include "pricer/statistics.hpp"
#include "pricer/version.hpp"

#ifndef PRICER_BUILD_TYPE
#define PRICER_BUILD_TYPE "unknown"
#endif

#ifndef PRICER_GIT_COMMIT
#define PRICER_GIT_COMMIT "unknown"
#endif

namespace {

class CliError : public std::runtime_error {
  public:
    using std::runtime_error::runtime_error;
};

class NumericResultError : public std::runtime_error {
  public:
    using std::runtime_error::runtime_error;
};

class CudaRunError : public std::runtime_error {
  public:
    using std::runtime_error::runtime_error;
};

struct Arguments {
    std::filesystem::path option_path;
    std::filesystem::path simulation_path;
    std::filesystem::path output_dir = "results";
    std::string backend = "both";
    int device = 0;
    std::uint32_t repetitions = 1U;
    std::uint32_t warmup = 1U;
};

struct BackendRun {
    pricer::PricingResult result;
    pricer::PerformanceData performance;
    pricer::EnvironmentData environment;
    std::uint64_t actual_batch_size;
};

void print_help() {
    std::cout
        << "Usage: pricer_cli --option <path> --simulation <path> [options]\n"
        << "  --output-dir <path>       Output directory (default: results)\n"
        << "  --backend <cpu|gpu|both>  Backend selection (default: both)\n"
        << "  --device <id>             CUDA device (default: 0)\n"
        << "  --repetitions <n>         Formal measurements (default: 1)\n"
        << "  --warmup <n>              Unmeasured GPU warmups (default: 1)\n"
        << "  --help                    Show this help\n"
        << "  --version                 Show version\n";
}

template <typename Integer>
Integer parse_integer(std::string_view text, std::string_view flag,
                      bool allow_zero) {
    std::uint64_t parsed = 0U;
    const auto result =
        std::from_chars(text.data(), text.data() + text.size(), parsed);
    if (text.empty() || text.front() == '-' || result.ec != std::errc() ||
        result.ptr != text.data() + text.size() ||
        (!allow_zero && parsed == 0U) ||
        parsed >
            static_cast<std::uint64_t>(std::numeric_limits<Integer>::max())) {
        throw CliError(std::string(flag) + " has an invalid integer value");
    }
    return static_cast<Integer>(parsed);
}

Arguments parse_arguments(int argc, char **argv) {
    Arguments arguments;
    std::unordered_set<std::string> seen;
    for (int index = 1; index < argc; ++index) {
        const std::string flag = argv[index];
        if (flag.empty() || flag.front() != '-') {
            throw CliError("unexpected positional argument '" + flag + "'");
        }
        if (flag == "--help" || flag == "--version") {
            throw CliError(flag + " must be used alone");
        }
        if (flag != "--option" && flag != "--simulation" &&
            flag != "--output-dir" && flag != "--backend" &&
            flag != "--device" && flag != "--repetitions" &&
            flag != "--warmup") {
            throw CliError("unknown flag '" + flag + "'");
        }
        if (!seen.insert(flag).second) {
            throw CliError("duplicate flag '" + flag + "'");
        }
        if (++index >= argc) {
            throw CliError("missing value for '" + flag + "'");
        }
        const std::string value = argv[index];
        if (flag == "--option") {
            arguments.option_path = value;
        } else if (flag == "--simulation") {
            arguments.simulation_path = value;
        } else if (flag == "--output-dir") {
            arguments.output_dir = value;
        } else if (flag == "--backend") {
            if (value != "cpu" && value != "gpu" && value != "both") {
                throw CliError("--backend must be cpu, gpu, or both");
            }
            arguments.backend = value;
        } else if (flag == "--device") {
            arguments.device = parse_integer<int>(value, flag, true);
        } else if (flag == "--repetitions") {
            arguments.repetitions =
                parse_integer<std::uint32_t>(value, flag, false);
        } else {
            arguments.warmup = parse_integer<std::uint32_t>(value, flag, true);
        }
    }
    if (arguments.option_path.empty()) {
        throw CliError("missing required --option");
    }
    if (arguments.simulation_path.empty()) {
        throw CliError("missing required --simulation");
    }
    if (arguments.output_dir.empty()) {
        throw CliError("--output-dir must not be empty");
    }
    return arguments;
}

std::optional<double> reference_price(const pricer::OptionParams &option) {
    // 只有欧式看涨有这里可用的解析参考价；亚式结果以 CPU/GPU 统计一致性验证。
    if (option.type == pricer::OptionType::EuropeanCall) {
        return pricer::black_scholes_call(option);
    }
    return std::nullopt;
}

void require_finite_nonnegative(double value, const char *name) {
    if (!std::isfinite(value) || value < 0.0) {
        throw NumericResultError(std::string(name) +
                                 " must be finite and non-negative");
    }
}

void validate_result(const pricer::PricingResult &result,
                     const pricer::PerformanceData &performance) {
    require_finite_nonnegative(result.price, "price");
    for (const auto &[value, name] :
         std::vector<std::pair<std::optional<double>, const char *>>{
             {result.sample_stddev, "sample stddev"},
             {result.standard_error, "standard error"},
             {result.reference_price, "reference price"},
             {result.absolute_error, "absolute error"},
             {result.relative_error, "relative error"}}) {
        if (value) {
            require_finite_nonnegative(*value, name);
        }
    }
    if (result.ci_lower && !std::isfinite(*result.ci_lower)) {
        throw NumericResultError("confidence interval lower bound is invalid");
    }
    if (result.ci_upper && !std::isfinite(*result.ci_upper)) {
        throw NumericResultError("confidence interval upper bound is invalid");
    }
    if (result.ci_lower && result.ci_upper &&
        *result.ci_lower > *result.ci_upper) {
        throw NumericResultError("confidence interval is inverted");
    }
    require_finite_nonnegative(performance.total_runtime_ms, "total runtime");
    require_finite_nonnegative(performance.compute_runtime_ms,
                               "compute runtime");
    require_finite_nonnegative(performance.paths_per_second,
                               "paths per second");
    for (const auto &value :
         {performance.rng_ms, performance.reduction_ms,
          performance.cpu_runtime_ms, performance.speedup_vs_cpu}) {
        if (value) {
            require_finite_nonnegative(*value, "optional performance metric");
        }
    }
}

double throughput(std::uint64_t paths, double compute_runtime_ms) {
    if (compute_runtime_ms <= 0.0) {
        throw NumericResultError("compute runtime is zero");
    }
    return static_cast<double>(paths) * 1000.0 / compute_runtime_ms;
}

std::optional<std::string> cuda_runtime_version() {
    int version = 0;
    if (cudaRuntimeGetVersion(&version) != cudaSuccess) {
        return std::nullopt;
    }
    return std::to_string(version / 1000) + "." +
           std::to_string((version % 1000) / 10);
}

pricer::EnvironmentData base_environment() {
    return {std::nullopt, std::nullopt, cuda_runtime_version(),
            std::string(PRICER_BUILD_TYPE), std::string(PRICER_GIT_COMMIT)};
}

BackendRun run_cpu(const pricer::RunConfig &config, std::uint32_t repetitions) {
    // measure_cpu 负责重复运行和取中位数；定价器本身只负责产生 RawMoments。
    const auto measured = pricer::measure_cpu(
        config.option, config.simulation, repetitions,
        reference_price(config.option), pricer::CpuMonteCarloPricer::price,
        pricer::ResultAnalyzer::analyze);
    pricer::PerformanceData performance{
        measured.total_runtime_ms,
        measured.compute_runtime_ms,
        std::nullopt,
        std::nullopt,
        throughput(config.simulation.num_paths, measured.compute_runtime_ms),
        config.simulation.block_size,
        0U,
        repetitions,
        "median",
        measured.total_runtime_ms,
        std::nullopt};
    validate_result(measured.result, performance);
    return {measured.result, std::move(performance), base_environment(),
            config.simulation.batch_size};
}

pricer::CudaPricingRun
price_gpu_once(const pricer::OptionParams &option,
               const pricer::SimulationParams &simulation, int device) {
    try {
        return pricer::CudaMonteCarloPricer::price(option, simulation, device);
    } catch (const pricer::CudaError &) {
        throw;
    } catch (const std::invalid_argument &error) {
        throw CudaRunError(error.what());
    } catch (const std::domain_error &error) {
        if (std::string_view(error.what()) ==
            "CUDA device reports no usable x-grid size") {
            throw CudaRunError(error.what());
        }
        throw;
    } catch (const std::overflow_error &error) {
        const std::string_view message(error.what());
        if (message == "CUDA batch byte count overflow" ||
            message == "CUDA payoff byte count overflow") {
            throw CudaRunError(error.what());
        }
        throw;
    }
}

BackendRun run_gpu(const pricer::RunConfig &config, int device,
                   std::uint32_t warmup, std::uint32_t repetitions,
                   std::optional<double> cpu_total_runtime_ms) {
    // GPU 的 warmup 不计入正式结果，用来减少首次 CUDA 初始化对计时的干扰。
    const auto measured =
        pricer::measure_gpu(config.option, config.simulation, device, warmup,
                            repetitions, reference_price(config.option),
                            price_gpu_once, pricer::ResultAnalyzer::analyze);
    // speedup_vs_cpu is an end-to-end metric: both operands include the
    // backend work required to obtain a complete PricingResult.
    const std::optional<double> speedup =
        cpu_total_runtime_ms && measured.total_runtime_ms > 0.0
            ? std::optional<double>(*cpu_total_runtime_ms /
                                    measured.total_runtime_ms)
            : std::nullopt;
    pricer::PerformanceData performance{
        measured.total_runtime_ms,
        measured.compute_runtime_ms,
        measured.rng_ms,
        measured.reduction_ms,
        throughput(config.simulation.num_paths, measured.compute_runtime_ms),
        config.simulation.block_size,
        measured.grid_size,
        repetitions,
        "median",
        cpu_total_runtime_ms,
        speedup};
    auto environment = base_environment();
    environment.gpu_name = measured.device.name;
    environment.compute_capability =
        std::to_string(measured.device.compute_capability_major) + "." +
        std::to_string(measured.device.compute_capability_minor);
    validate_result(measured.result, performance);
    return {measured.result, std::move(performance), std::move(environment),
            measured.actual_batch_size};
}

std::string option_type_name(pricer::OptionType type) {
    return type == pricer::OptionType::EuropeanCall ? "european_call"
                                                    : "asian_call";
}

std::string absolute_path_string(const std::filesystem::path &path) {
    std::error_code error;
    const auto absolute = std::filesystem::absolute(path, error);
    if (error) {
        throw pricer::OutputError("could not resolve absolute path: " +
                                  error.message());
    }
    return absolute.string();
}

void write_backend_output(const Arguments &arguments,
                          const pricer::RunConfig &config,
                          const std::string &timestamp,
                          const std::string &backend, BackendRun run) {
    auto simulation = config.simulation;
    simulation.batch_size = run.actual_batch_size;
    if (backend == "cpu") {
        simulation.precision = pricer::Precision::Fp64;
        simulation.rng = pricer::RngType::Mt19937_64;
    }
    pricer::OutputRecord record{
        timestamp + "-" + option_type_name(config.option.type) + "-" + backend,
        timestamp,
        backend,
        config.option,
        simulation,
        std::move(run.result),
        std::move(run.performance),
        std::move(run.environment)};
    const auto written =
        pricer::write_output_transaction(arguments.output_dir, record);
    record.run_id = written.run_id;

    const auto &result = record.result;
    const auto &performance = record.performance;
    std::cout << "INFO backend=" << backend << " price=" << result.price;
    if (result.standard_error) {
        std::cout << " standard_error=" << *result.standard_error;
    } else {
        std::cout << " standard_error=null";
    }
    if (result.ci_lower && result.ci_upper) {
        std::cout << " ci95=[" << *result.ci_lower << ',' << *result.ci_upper
                  << ']';
    } else {
        std::cout << " ci95=null";
    }
    std::cout << " total_runtime_ms=" << performance.total_runtime_ms
              << " compute_runtime_ms=" << performance.compute_runtime_ms
              << " paths_per_second=" << performance.paths_per_second;
    if (performance.speedup_vs_cpu) {
        std::cout << " speedup_vs_cpu=" << *performance.speedup_vs_cpu;
    }
    std::cout << '\n'
              << "INFO result_json=" << absolute_path_string(written.json_path)
              << '\n'
              << "INFO performance_csv="
              << absolute_path_string(written.csv_path) << '\n';
}

int run_cli(int argc, char **argv) {
    if (argc == 2 && std::string_view(argv[1]) == "--help") {
        print_help();
        return 0;
    }
    if (argc == 2 && std::string_view(argv[1]) == "--version") {
        std::cout << "pricer_cli " << pricer::version() << '\n';
        return 0;
    }
    // 主流程只编排：解析命令行 -> 读取并校验 INI -> CPU/GPU 定价 -> 写
    // JSON/CSV。
    const Arguments arguments = parse_arguments(argc, argv);
    const bool run_cpu_backend = arguments.backend != "gpu";
    const bool run_gpu_backend = arguments.backend != "cpu";
    const auto config = pricer::load_run_config(
        arguments.option_path, arguments.simulation_path, arguments.output_dir,
        run_cpu_backend, run_gpu_backend);
    std::cout << "INFO option=" << absolute_path_string(arguments.option_path)
              << " simulation="
              << absolute_path_string(arguments.simulation_path)
              << " backend=" << arguments.backend << '\n';

    std::optional<BackendRun> cpu;
    if (run_cpu_backend) {
        cpu = run_cpu(config, arguments.repetitions);
    }
    std::optional<BackendRun> gpu;
    if (run_gpu_backend) {
        const std::optional<double> cpu_total_runtime =
            cpu ? cpu->performance.cpu_runtime_ms : std::nullopt;
        gpu = run_gpu(config, arguments.device, arguments.warmup,
                      arguments.repetitions, cpu_total_runtime);
    }

    const std::string timestamp = pricer::utc_timestamp();
    if (cpu) {
        write_backend_output(arguments, config, timestamp, "cpu",
                             std::move(*cpu));
    }
    if (gpu) {
        write_backend_output(arguments, config, timestamp, "gpu",
                             std::move(*gpu));
    }
    return 0;
}

} // namespace

int main(int argc, char **argv) {
    try {
        return run_cli(argc, argv);
    } catch (const CliError &error) {
        std::cerr << "ERROR [CLI_INVALID] " << error.what() << '\n';
        return 2;
    } catch (const pricer::ConfigError &error) {
        std::cerr << error.render() << '\n';
        return pricer::exit_code_for(error.code());
    } catch (const pricer::OutputError &error) {
        std::cerr << "ERROR [FILE_WRITE] " << error.what() << '\n';
        return 3;
    } catch (const pricer::CudaError &error) {
        std::cerr << "ERROR [CUDA_RUNTIME] " << error.api() << ": "
                  << error.cuda_text() << '\n';
        return 4;
    } catch (const CudaRunError &error) {
        std::cerr << "ERROR [CUDA_RUNTIME] " << error.what() << '\n';
        return 4;
    } catch (const NumericResultError &error) {
        std::cerr << "ERROR [NUMERIC_RESULT] " << error.what() << '\n';
        return 5;
    } catch (const std::domain_error &error) {
        std::cerr << "ERROR [NUMERIC_RESULT] " << error.what() << '\n';
        return 5;
    } catch (const std::overflow_error &error) {
        std::cerr << "ERROR [NUMERIC_RESULT] " << error.what() << '\n';
        return 5;
    } catch (const std::exception &error) {
        std::cerr << "ERROR [INTERNAL] " << error.what() << '\n';
        return 10;
    } catch (...) {
        std::cerr << "ERROR [INTERNAL] unknown failure\n";
        return 10;
    }
}
