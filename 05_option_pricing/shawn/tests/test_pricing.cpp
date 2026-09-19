#include "pricing/pricing.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

void test_black_scholes() {
  // 检查解析看涨价格及看涨看跌平价。
  pricing::OptionParams option;
  option.type = pricing::OptionType::kEuropeanCall;
  const double call = pricing::black_scholes_price(option);
  require(std::abs(call - 9.413403383853016) < 1e-10,
          "Black-Scholes call regression failed");

  option.type = pricing::OptionType::kEuropeanPut;
  const double put = pricing::black_scholes_price(option);
  const double parity = option.spot -
                        option.strike *
                            std::exp(-option.risk_free_rate * option.maturity);
  require(std::abs((call - put) - parity) < 1e-10,
          "put-call parity failed");
}

void test_config_parser() {
  const std::string root = PRICING_SOURCE_DIR;
  const auto option =
      pricing::load_option_params(root + "/configs/european_call.txt");
  const auto simulation =
      pricing::load_simulation_params(root + "/configs/simulation_smoke.txt");
  require(option.type == pricing::OptionType::kEuropeanCall,
          "option parser returned wrong type");
  require(simulation.num_paths == 200000,
          "simulation parser returned wrong path count");
  require(simulation.variance_reduction ==
              pricing::VarianceReduction::kAntithetic,
          "variance reduction parser failed");
  const auto heston =
      pricing::load_option_params(root + "/configs/heston_call.txt");
  const auto qmc =
      pricing::load_simulation_params(root + "/configs/simulation_qmc_smoke.txt");
  const auto american_qmc = pricing::load_simulation_params(
      root + "/configs/simulation_american_qmc_validation.txt");
  require(heston.model == pricing::ModelType::kHeston &&
              heston.heston_rho == -0.7,
          "Heston configuration parser failed");
  require(qmc.rng == "halton" && qmc.qmc_replications == 8 &&
              qmc.compute_greeks,
          "advanced simulation parser failed");
  require(american_qmc.rng == "halton" &&
              american_qmc.num_paths == 131072 &&
              american_qmc.num_steps == 64 &&
              american_qmc.spot_bump_relative == 0.01,
          "American QMC validation configuration is wrong");
}

void test_monte_carlo_convergence() {
  // 随机测试只验证统计收敛；路径公式的逐步正确性由下方确定性测试负责。
  pricing::OptionParams option;
  pricing::SimulationParams simulation;
  simulation.num_paths = 150000;
  simulation.num_steps = 32;
  simulation.seed = 20260824;
  simulation.variance_reduction = pricing::VarianceReduction::kControlVariate;
  const pricing::Estimate estimate = pricing::price_cpu(option, simulation);
  const double reference = pricing::black_scholes_price(option);
  require(std::abs(estimate.price - reference) < 5.0 * estimate.standard_error,
          "Monte Carlo estimate is outside a five-standard-error envelope");
  require(estimate.confidence_low < estimate.price &&
              estimate.price < estimate.confidence_high,
          "confidence interval does not contain estimate");
}

void test_deterministic_path_evaluation() {
  pricing::OptionParams option;
  option.type = pricing::OptionType::kAsianCall;
  option.strike = 90.0;
  pricing::SimulationParams simulation;
  simulation.num_steps = 2;
  simulation.variance_reduction = pricing::VarianceReduction::kNone;

  const double dt = option.maturity / simulation.num_steps;
  const double drift =
      (option.risk_free_rate - 0.5 * option.volatility * option.volatility) * dt;
  const double first_spot = option.spot * std::exp(drift);
  const double second_spot = first_spot * std::exp(drift);
  const double discount = std::exp(-option.risk_free_rate * option.maturity);
  const auto asian =
      pricing::evaluate_path(option, simulation, std::vector<double>{0.0, 0.0});
  const double expected_asian =
      discount * std::max(0.5 * (first_spot + second_spot) - option.strike, 0.0);
  require(std::abs(asian.discounted_payoff - expected_asian) < 1e-12,
          "deterministic Asian payoff is wrong");

  option.type = pricing::OptionType::kBarrierCall;
  option.barrier = first_spot - 1e-12;
  const auto knocked_out =
      pricing::evaluate_path(option, simulation, std::vector<double>{0.0, 0.0});
  require(knocked_out.discounted_payoff == 0.0,
          "barrier touch must knock out the option");

  option.barrier = second_spot + 1.0;
  const auto survives =
      pricing::evaluate_path(option, simulation, std::vector<double>{0.0, 0.0});
  require(std::abs(survives.discounted_payoff -
                   discount * std::max(second_spot - option.strike, 0.0)) <
              1e-12,
          "non-hit barrier payoff is wrong");

  option.type = pricing::OptionType::kEuropeanCall;
  option.strike = 90.0;
  simulation.variance_reduction = pricing::VarianceReduction::kAntithetic;
  const double z = 0.4;
  const double full_drift = option.risk_free_rate -
                            0.5 * option.volatility * option.volatility;
  const double spot_a =
      option.spot * std::exp(full_drift + option.volatility * z);
  const double spot_b =
      option.spot * std::exp(full_drift - option.volatility * z);
  const double expected_pair =
      discount * 0.5 * (std::max(spot_a - option.strike, 0.0) +
                        std::max(spot_b - option.strike, 0.0));
  const auto pair =
      pricing::evaluate_path(option, simulation, std::vector<double>{z});
  require(std::abs(pair.discounted_payoff - expected_pair) < 1e-12 &&
              pair.physical_paths == 2,
          "deterministic antithetic estimator is wrong");

  bool rejected_bad_normal_count = false;
  try {
    static_cast<void>(pricing::evaluate_path(option, simulation, {}));
  } catch (const std::invalid_argument&) {
    rejected_bad_normal_count = true;
  }
  require(rejected_bad_normal_count,
          "deterministic path API accepted a wrong normal count");
}

void test_path_dependent_options() {
  pricing::OptionParams option;
  option.type = pricing::OptionType::kAsianCall;
  pricing::SimulationParams simulation;
  simulation.num_paths = 20000;
  simulation.num_steps = 32;
  simulation.seed = 7;
  simulation.variance_reduction = pricing::VarianceReduction::kAntithetic;
  const auto asian = pricing::price_cpu(option, simulation);
  require(asian.price > 0.0 && std::isfinite(asian.price),
          "Asian option price is invalid");
  require(asian.physical_paths == 2 * simulation.num_paths,
          "antithetic physical path accounting is wrong");

  option.type = pricing::OptionType::kBarrierCall;
  option.barrier = 130.0;
  const auto barrier = pricing::price_cpu(option, simulation);
  require(barrier.price >= 0.0 && barrier.price < asian.price + 10.0,
          "barrier option price is invalid");
}

void test_greeks_and_randomized_halton() {
  // 解析 Greeks 校验公式；随机化 Halton 校验可复现性、误差估计和 CRN 差分。
  pricing::OptionParams option;
  const auto analytic = pricing::black_scholes_greeks(option);
  require(analytic.delta > 0.0 && analytic.delta < 1.0 &&
              analytic.gamma > 0.0 && analytic.vega > 0.0,
          "analytic Greeks are invalid");

  pricing::SimulationParams simulation;
  simulation.num_paths = 32768;
  simulation.num_steps = 32;
  simulation.seed = 20260824;
  simulation.rng = "halton";
  simulation.qmc_replications = 8;
  simulation.variance_reduction = pricing::VarianceReduction::kControlVariate;
  simulation.spot_bump_relative = 0.005;
  simulation.volatility_bump_absolute = 0.002;
  const auto first = pricing::price_cpu(option, simulation);
  const auto second = pricing::price_cpu(option, simulation);
  require(first.price == second.price &&
              first.standard_error == second.standard_error,
          "randomized Halton result is not reproducible");
  const double reference = pricing::black_scholes_price(option);
  require(std::abs(first.price - reference) <
              6.0 * first.standard_error + 0.02,
          "randomized Halton price does not converge to Black-Scholes");

  const auto finite_difference =
      pricing::estimate_greeks_cpu(option, simulation);
  require(std::abs(finite_difference.delta - analytic.delta) < 0.03,
          "finite-difference Delta is inaccurate");
  require(std::abs(finite_difference.gamma - analytic.gamma) < 0.015,
          "finite-difference Gamma is inaccurate");
  require(std::abs(finite_difference.vega - analytic.vega) < 2.0,
          "finite-difference Vega is inaccurate");
}

void test_heston_and_local_volatility() {
  // Heston vol-of-vol=0、Local Vol beta=0 时都退化为常波动率模型，应回到 BS 价格。
  pricing::OptionParams black_scholes;
  const double reference = pricing::black_scholes_price(black_scholes);
  pricing::SimulationParams simulation;
  simulation.num_paths = 50000;
  simulation.num_steps = 32;
  simulation.seed = 99;
  simulation.variance_reduction = pricing::VarianceReduction::kControlVariate;

  auto local = black_scholes;
  local.model = pricing::ModelType::kLocalVolatility;
  local.local_vol_beta = 0.0;
  const auto local_estimate = pricing::price_cpu(local, simulation);
  require(std::abs(local_estimate.price - reference) <
              6.0 * local_estimate.standard_error + 0.04,
          "constant local-vol model does not reduce to Black-Scholes");

  auto heston = black_scholes;
  heston.model = pricing::ModelType::kHeston;
  heston.heston_initial_variance =
      black_scholes.volatility * black_scholes.volatility;
  heston.heston_theta = heston.heston_initial_variance;
  heston.heston_vol_of_vol = 0.0;
  const auto heston_estimate = pricing::price_cpu(heston, simulation);
  require(std::abs(heston_estimate.price - reference) <
              6.0 * heston_estimate.standard_error + 0.04,
          "degenerate Heston model does not reduce to Black-Scholes");
}

void test_american_lsm() {
  // American put 与高步数 CRR 二叉树对比；其价值不应低于同参数 European put。
  pricing::OptionParams american;
  american.type = pricing::OptionType::kAmericanPut;
  pricing::SimulationParams simulation;
  simulation.num_paths = 60000;
  simulation.num_steps = 50;
  simulation.seed = 123;
  simulation.variance_reduction = pricing::VarianceReduction::kNone;
  const auto estimate = pricing::price_cpu(american, simulation);
  const double binomial = pricing::american_binomial_price(american, 2000);
  require(std::abs(estimate.price - binomial) <
              6.0 * estimate.standard_error + 0.25,
          "American LSM price is inconsistent with the CRR reference");

  auto european = american;
  european.type = pricing::OptionType::kEuropeanPut;
  const double european_price = pricing::black_scholes_price(european);
  require(estimate.price + 0.10 >= european_price,
          "American put is materially below the European put");

  simulation.spot_bump_relative = 0.01;
  simulation.volatility_bump_absolute = 0.002;
  const auto greeks = pricing::estimate_greeks_cpu(american, simulation);
  const auto tree_greeks = pricing::american_binomial_greeks(
      american, simulation.spot_bump_relative,
      simulation.volatility_bump_absolute);
  require(tree_greeks.gamma > 0.0 && tree_greeks.vega > 0.0 &&
              tree_greeks.method ==
                  "central_finite_difference_crr_binomial_2000_steps",
          "American CRR Greeks reference is invalid");
  require(std::abs(greeks.delta - tree_greeks.delta) < 0.08 &&
              std::abs(greeks.gamma - tree_greeks.gamma) < 0.04 &&
              std::abs(greeks.vega - tree_greeks.vega) < 6.0,
          "American finite-difference Greeks are inconsistent with CRR");

  simulation.rng = "halton";
  simulation.num_paths = 8192;
  simulation.qmc_replications = 8;
  const auto qmc = pricing::price_cpu(american, simulation);
  require(std::isfinite(qmc.price) && qmc.price > 0.0 &&
              std::abs(qmc.price - binomial) < 0.8,
          "American randomized-Halton LSM result is invalid");
}

#ifdef PRICING_HAS_CUDA
void test_cpu_cuda_advanced_consistency() {
  // CPU/CUDA 对照：三个模型、QMC、Greeks 和美式 LSM。
  if (!pricing::cuda_backend_available()) return;
  pricing::SimulationParams simulation;
  simulation.num_paths = 8192;
  simulation.num_steps = 16;
  simulation.seed = 20260824;
  simulation.rng = "halton";
  simulation.qmc_replications = 8;
  simulation.variance_reduction = pricing::VarianceReduction::kControlVariate;
  for (const auto model : {pricing::ModelType::kBlackScholes,
                           pricing::ModelType::kHeston,
                           pricing::ModelType::kLocalVolatility}) {
    pricing::OptionParams option;
    option.model = model;
    option.local_vol_beta = -0.25;
    const auto cpu = pricing::price_cpu(option, simulation);
    const auto gpu = pricing::price_cuda(option, simulation);
    const double tolerance =
        5.0 * std::hypot(cpu.standard_error, gpu.standard_error) + 0.05;
    require(std::abs(cpu.price - gpu.price) <= tolerance,
            "CPU/CUDA advanced-model price comparison failed");
  }

  pricing::OptionParams european;
  simulation.num_paths = 16384;
  simulation.spot_bump_relative = 0.005;
  simulation.volatility_bump_absolute = 0.002;
  const auto cpu_greeks = pricing::estimate_greeks_cpu(european, simulation);
  const auto gpu_greeks = pricing::estimate_greeks_cuda(european, simulation);
  require(std::abs(cpu_greeks.delta - gpu_greeks.delta) < 0.03 &&
              std::abs(cpu_greeks.gamma - gpu_greeks.gamma) < 0.02 &&
              std::abs(cpu_greeks.vega - gpu_greeks.vega) < 2.0,
          "CPU/CUDA Greeks comparison failed");

  pricing::OptionParams american;
  american.type = pricing::OptionType::kAmericanPut;
  simulation.num_paths = 4096;
  simulation.num_steps = 20;
  simulation.variance_reduction = pricing::VarianceReduction::kNone;
  const auto cpu_american = pricing::price_cpu(american, simulation);
  const auto gpu_american = pricing::price_cuda(american, simulation);
  const double american_tolerance =
      5.0 * std::hypot(cpu_american.standard_error,
                       gpu_american.standard_error) +
      0.15;
  require(std::abs(cpu_american.price - gpu_american.price) <=
              american_tolerance,
          "CPU/CUDA American LSM comparison failed");
}
#endif

}  // namespace

int main() {
  try {
    test_black_scholes();
    test_config_parser();
    test_monte_carlo_convergence();
    test_deterministic_path_evaluation();
    test_path_dependent_options();
    test_greeks_and_randomized_halton();
    test_heston_and_local_volatility();
    test_american_lsm();
#ifdef PRICING_HAS_CUDA
    test_cpu_cuda_advanced_consistency();
#endif
    std::cout << "all pricing tests passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "test failure: " << error.what() << '\n';
    return 1;
  }
}
