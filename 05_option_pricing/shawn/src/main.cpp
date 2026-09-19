#include "pricing/pricing.hpp"

#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>

namespace {

struct Arguments {
  std::string option_path;
  std::string simulation_path;
  std::string output_path = "result.txt";
  std::string performance_path = "performance.log";
  std::string backend = "auto";
  std::uint64_t reference_paths = 200'000;
  std::optional<std::uint64_t> cpu_benchmark_paths;
};

void print_usage(const char* program) {
  std::cout
      << "Usage: " << program
      << " --option FILE --simulation FILE [--backend auto|cpu|cuda]"
         " [--output FILE] [--performance FILE] [--reference-paths N]"
         " [--cpu-benchmark-paths N]\n";
}

Arguments parse_arguments(int argc, char** argv) {
  Arguments arguments;
  for (int index = 1; index < argc; ++index) {
    const std::string flag = argv[index];
    if (flag == "--help" || flag == "-h") {
      print_usage(argv[0]);
      std::exit(0);
    }
    if (index + 1 >= argc) {
      throw std::runtime_error("missing value after " + flag);
    }
    const std::string value = argv[++index];
    if (flag == "--option") {
      arguments.option_path = value;
    } else if (flag == "--simulation") {
      arguments.simulation_path = value;
    } else if (flag == "--output") {
      arguments.output_path = value;
    } else if (flag == "--performance") {
      arguments.performance_path = value;
    } else if (flag == "--backend") {
      arguments.backend = value;
    } else if (flag == "--reference-paths") {
      arguments.reference_paths = std::stoull(value);
    } else if (flag == "--cpu-benchmark-paths") {
      arguments.cpu_benchmark_paths = std::stoull(value);
    } else {
      throw std::runtime_error("unknown argument: " + flag);
    }
  }
  if (arguments.option_path.empty() || arguments.simulation_path.empty()) {
    throw std::runtime_error("--option and --simulation are required");
  }
  if (arguments.backend != "auto" && arguments.backend != "cpu" &&
      arguments.backend != "cuda") {
    throw std::runtime_error("--backend must be auto, cpu or cuda");
  }
  if (arguments.reference_paths == 1) {
    throw std::runtime_error("--reference-paths must be 0 or at least 2");
  }
  if (arguments.cpu_benchmark_paths == 1) {
    throw std::runtime_error(
        "--cpu-benchmark-paths must be 0 or at least 2");
  }
  return arguments;
}

void ensure_parent_directory(const std::string& path) {
  const std::filesystem::path parent = std::filesystem::path(path).parent_path();
  if (!parent.empty()) {
    std::filesystem::create_directories(parent);
  }
}

void write_outputs(const Arguments& arguments,
                   const pricing::OptionParams& option,
                   const pricing::SimulationParams& simulation,
                   const pricing::Estimate& estimate, double reference,
                   const std::string& reference_method,
                   const std::optional<pricing::Estimate>& reference_estimate,
                   std::uint64_t reference_seed,
                   const std::optional<pricing::Estimate>& cpu_benchmark,
                   const std::optional<pricing::GreekEstimates>& greeks,
                   const std::optional<pricing::GreekEstimates>&
                       reference_greeks,
                   double pricing_total_ms) {
  // 分开记录请求的 RNG 和实际 RNG；CPU 不调用 cuRAND。
  const std::string effective_rng =
      simulation.rng == "halton" ||
              estimate.backend.find("halton") != std::string::npos
          ? "randomized_halton_inverse_cdf"
          : (estimate.backend.rfind("cpu-", 0) == 0
                 ? "mt19937_64"
                 : (simulation.rng == "custom" ? "splitmix64_box_muller"
                                                : "curand_philox"));
  ensure_parent_directory(arguments.output_path);
  std::ofstream result(arguments.output_path);
  if (!result) {
    throw std::runtime_error("cannot write " + arguments.output_path);
  }
  const bool american = option.type == pricing::OptionType::kAmericanCall ||
                        option.type == pricing::OptionType::kAmericanPut;
  const double missing = std::numeric_limits<double>::quiet_NaN();
  result << std::setprecision(12)
         << "option_type=" << pricing::to_string(option.type) << '\n'
         << "model=" << pricing::to_string(option.model) << '\n'
         << "method=" << (american ? "longstaff_schwartz_" : "monte_carlo_")
         << pricing::to_string(option.model) << '\n'
         << "variance_reduction="
         << pricing::to_string(simulation.variance_reduction) << '\n'
         << "backend=" << estimate.backend << '\n'
         << "rng=" << effective_rng << '\n'
         << "rng_requested=" << simulation.rng << '\n'
         << "seed=" << simulation.seed << '\n'
         << "num_paths=" << simulation.num_paths << '\n'
         << "num_steps=" << simulation.num_steps << '\n'
         << "qmc_replications=" << simulation.qmc_replications << '\n'
         << "num_gpus=" << simulation.num_gpus << '\n'
         << "price=" << estimate.price << '\n'
         << "reference_price=" << reference << '\n'
         << "reference_method=" << reference_method << '\n'
         << "reference_seed=" << reference_seed << '\n'
         << "reference_standard_error="
         << (reference_estimate
                 ? reference_estimate->standard_error
                 : (std::isfinite(reference)
                        ? 0.0
                        : std::numeric_limits<double>::quiet_NaN()))
         << '\n'
         << "reference_confidence_95_low="
         << (reference_estimate ? reference_estimate->confidence_low : reference)
         << '\n'
         << "reference_confidence_95_high="
         << (reference_estimate ? reference_estimate->confidence_high : reference)
         << '\n'
         << "absolute_error=" << std::abs(estimate.price - reference) << '\n'
         << "standard_error=" << estimate.standard_error << '\n'
         << "confidence_95_low=" << estimate.confidence_low << '\n'
         << "confidence_95_high=" << estimate.confidence_high << '\n';
  result << "greeks_computed=" << (greeks ? "true" : "false") << '\n'
         << "delta=" << (greeks ? greeks->delta : missing) << '\n'
         << "gamma=" << (greeks ? greeks->gamma : missing) << '\n'
         << "vega=" << (greeks ? greeks->vega : missing) << '\n'
         << "greeks_method=" << (greeks ? greeks->method : "not_computed")
         << '\n'
         << "greeks_backend=" << (greeks ? greeks->backend : "not_computed")
         << '\n'
         << "reference_delta="
         << (reference_greeks ? reference_greeks->delta : missing) << '\n'
         << "reference_gamma="
         << (reference_greeks ? reference_greeks->gamma : missing) << '\n'
         << "reference_vega="
         << (reference_greeks ? reference_greeks->vega : missing) << '\n'
         << "reference_greeks_method="
         << (reference_greeks ? reference_greeks->method : "not_available")
         << '\n';

  ensure_parent_directory(arguments.performance_path);
  std::ofstream performance(arguments.performance_path);
  if (!performance) {
    throw std::runtime_error("cannot write " + arguments.performance_path);
  }
  const double paths_per_second =
      estimate.elapsed_ms > 0.0
          ? static_cast<double>(estimate.physical_paths) /
                (estimate.elapsed_ms / 1000.0)
          : 0.0;
  const double end_to_end_paths_per_second =
      pricing_total_ms > 0.0
          ? static_cast<double>(estimate.physical_paths) /
                (pricing_total_ms / 1000.0)
          : 0.0;
  const double cpu_paths_per_second =
      cpu_benchmark && cpu_benchmark->elapsed_ms > 0.0
          ? static_cast<double>(cpu_benchmark->physical_paths) /
                (cpu_benchmark->elapsed_ms / 1000.0)
          : 0.0;
  const double speedup = cpu_paths_per_second > 0.0
                             ? end_to_end_paths_per_second /
                                   cpu_paths_per_second
                             : 0.0;
  performance << std::setprecision(12)
              << "backend=" << estimate.backend << '\n'
              << "rng=" << effective_rng << '\n'
              << "rng_requested=" << simulation.rng << '\n'
              << "seed=" << simulation.seed << '\n'
              << "model=" << pricing::to_string(option.model) << '\n'
              << "num_gpus=" << simulation.num_gpus << '\n'
              << "qmc_replications=" << simulation.qmc_replications << '\n'
              << "total_runtime_ms=" << pricing_total_ms << '\n'
              << "greeks_runtime_ms="
              << (greeks ? greeks->elapsed_ms : 0.0) << '\n'
              << "simulation_kernel_ms=" << estimate.simulation_ms << '\n'
              << "reduction_ms=" << estimate.reduction_ms << '\n'
              << "kernel_and_reduction_ms=" << estimate.elapsed_ms << '\n'
              << "estimator_samples=" << estimate.estimator_samples << '\n'
              << "physical_paths=" << estimate.physical_paths << '\n'
              << "kernel_paths_per_second=" << paths_per_second << '\n'
              << "end_to_end_paths_per_second="
              << end_to_end_paths_per_second << '\n'
              << "num_steps=" << simulation.num_steps << '\n'
              << "parallel_granularity="
              << (estimate.backend.rfind("cuda", 0) == 0
                      ? "one_estimator_sample_per_cuda_thread"
                      : "serial_cpu_samples")
              << '\n'
              << "cuda_block_size="
              << (estimate.backend.rfind("cuda", 0) == 0 ? 256 : 0) << '\n'
              << "reduction="
              << (estimate.backend.rfind("cuda", 0) == 0
                      ? "block_moments_then_second_stage_reduce"
                      : "online_welford_moments")
              << '\n'
              << "timing_scope="
              << (estimate.backend.rfind("cuda", 0) == 0
                      ? "kernel_and_device_reduction"
                      : "cpu_simulation_and_reduction")
              << '\n'
              << "cpu_benchmark_paths="
              << (cpu_benchmark ? cpu_benchmark->physical_paths : 0) << '\n'
              << "cpu_benchmark_ms="
              << (cpu_benchmark ? cpu_benchmark->elapsed_ms : 0.0) << '\n'
              << "cpu_paths_per_second=" << cpu_paths_per_second << '\n'
              << "throughput_speedup_vs_cpu_single_thread=" << speedup << '\n';
}

}  // namespace

int main(int argc, char** argv) {
  try {
    // 文件解析和结果 I/O 不计入定价耗时。
    const Arguments arguments = parse_arguments(argc, argv);
    const pricing::OptionParams option =
        pricing::load_option_params(arguments.option_path);
    const pricing::SimulationParams simulation =
        pricing::load_simulation_params(arguments.simulation_path);

    const bool use_cuda =
        arguments.backend == "cuda" ||
        (arguments.backend == "auto" && pricing::cuda_backend_available());
    if (arguments.backend == "cuda" && !pricing::cuda_backend_available()) {
      throw std::runtime_error("CUDA backend requested but no CUDA device is available");
    }
    const auto pricing_start = std::chrono::steady_clock::now();
    const pricing::Estimate estimate =
        use_cuda ? (simulation.num_gpus > 1
                        ? pricing::price_cuda_multi(option, simulation)
                        : pricing::price_cuda(option, simulation))
                 : pricing::price_cpu(option, simulation);
    const auto pricing_end = std::chrono::steady_clock::now();
    const double pricing_total_ms =
        std::chrono::duration<double, std::milli>(pricing_end - pricing_start)
            .count();

    std::optional<pricing::GreekEstimates> greeks;
    if (simulation.compute_greeks) {
      greeks = use_cuda ? pricing::estimate_greeks_cuda(option, simulation)
                        : pricing::estimate_greeks_cpu(option, simulation);
    }

    const bool is_black_scholes_european =
        option.model == pricing::ModelType::kBlackScholes &&
        (option.type == pricing::OptionType::kEuropeanCall ||
         option.type == pricing::OptionType::kEuropeanPut);
    const bool is_black_scholes_american =
        option.model == pricing::ModelType::kBlackScholes &&
        (option.type == pricing::OptionType::kAmericanCall ||
         option.type == pricing::OptionType::kAmericanPut);
    const bool needs_monte_carlo_reference =
        !is_black_scholes_european && !is_black_scholes_american;
    std::optional<pricing::Estimate> reference_estimate;
    std::uint64_t reference_seed = 0;
    if (needs_monte_carlo_reference && arguments.reference_paths >= 2) {
      // 路径依赖期权以独立 seed 的 CPU 控制变量估计作为数值参考。
      pricing::SimulationParams reference_simulation = simulation;
      reference_simulation.num_paths = arguments.reference_paths;
      reference_simulation.num_gpus = 1;
      reference_simulation.compute_greeks = false;
      reference_simulation.rng = "curand";
      const bool american =
          option.type == pricing::OptionType::kAmericanCall ||
          option.type == pricing::OptionType::kAmericanPut;
      reference_simulation.variance_reduction =
          american ? pricing::VarianceReduction::kNone
                   : pricing::VarianceReduction::kControlVariate;
      reference_seed = simulation.seed ^ 0xd1b54a32d192ed03ULL;
      reference_simulation.seed = reference_seed;
      reference_estimate =
          pricing::price_cpu(option, reference_simulation);
    }

    // 性能基线复用被测模型、步数和方差缩减模式，与独立数值参考分开运行。
    const std::uint64_t cpu_benchmark_paths =
        arguments.cpu_benchmark_paths.value_or(arguments.reference_paths);
    std::optional<pricing::Estimate> cpu_benchmark;
    if (use_cuda && cpu_benchmark_paths >= 2) {
      pricing::SimulationParams cpu_simulation = simulation;
      cpu_simulation.num_paths =
          std::min(simulation.num_paths, cpu_benchmark_paths);
      cpu_simulation.num_gpus = 1;
      cpu_benchmark = pricing::price_cpu(option, cpu_simulation);
    } else if (!use_cuda) {
      cpu_benchmark = estimate;
    }

    double reference = std::numeric_limits<double>::quiet_NaN();
    std::string reference_method = "not_computed";
    if (is_black_scholes_european) {
      reference = pricing::black_scholes_price(option);
      reference_method = "black_scholes_analytic";
    } else if (is_black_scholes_american) {
      reference = pricing::american_binomial_price(option);
      reference_method = "crr_binomial_2000_steps";
    } else if (reference_estimate) {
      reference = reference_estimate->price;
      reference_method =
          "independent_cpu_control_variate_" +
          std::to_string(reference_estimate->estimator_samples) + "_samples";
    }

    std::optional<pricing::GreekEstimates> reference_greeks;
    if (greeks && is_black_scholes_european) {
      reference_greeks = pricing::black_scholes_greeks(option);
    } else if (greeks && is_black_scholes_american) {
      // 美式行权边界不光滑，与估值端一致使用至少 1% 的现货价格扰动。
      reference_greeks = pricing::american_binomial_greeks(
          option, std::max(simulation.spot_bump_relative, 1e-2),
          simulation.volatility_bump_absolute);
    }

    write_outputs(arguments, option, simulation, estimate, reference,
                  reference_method, reference_estimate, reference_seed,
                  cpu_benchmark, greeks, reference_greeks, pricing_total_ms);
    std::cout << std::setprecision(10) << "price=" << estimate.price
              << " standard_error=" << estimate.standard_error
              << " ci95=[" << estimate.confidence_low << ", "
              << estimate.confidence_high << "] backend=" << estimate.backend
              << '\n';
    if (greeks) {
      std::cout << "delta=" << greeks->delta << " gamma=" << greeks->gamma
                << " vega=" << greeks->vega << '\n';
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
}
