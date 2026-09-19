#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace pricing {

enum class OptionType {
  kEuropeanCall,
  kEuropeanPut,
  kAsianCall,
  kBarrierCall,
  kAmericanCall,
  kAmericanPut,
};

enum class ModelType {
  kBlackScholes,
  kHeston,
  kLocalVolatility,
};

enum class VarianceReduction {
  kNone,
  kAntithetic,
  kControlVariate,
};

struct OptionParams {
  OptionType type = OptionType::kEuropeanCall;
  ModelType model = ModelType::kBlackScholes;
  double spot = 100.0;
  double strike = 100.0;
  double risk_free_rate = 0.03;
  double volatility = 0.2;
  double maturity = 1.0;
  double barrier = 130.0;
  // Heston 非负投影 Euler 参数；每步更新后的方差截断为非负值。
  double heston_initial_variance = 0.04;
  double heston_kappa = 2.0;
  double heston_theta = 0.04;
  double heston_vol_of_vol = 0.3;
  double heston_rho = -0.7;
  // Local volatility: sigma(S)=volatility*(S/spot)^local_vol_beta。
  double local_vol_beta = 0.0;
};

struct SimulationParams {
  std::uint64_t num_paths = 10'000'000;
  int num_steps = 256;
  std::uint64_t seed = 1234;
  std::string rng = "curand";
  VarianceReduction variance_reduction = VarianceReduction::kNone;
  int qmc_replications = 8;
  int num_gpus = 1;
  bool compute_greeks = false;
  double spot_bump_relative = 1e-3;
  double volatility_bump_absolute = 1e-3;
};

struct Estimate {
  double price = 0.0;
  double standard_error = 0.0;
  double confidence_low = 0.0;
  double confidence_high = 0.0;
  double elapsed_ms = 0.0;
  double simulation_ms = 0.0;
  double reduction_ms = 0.0;
  std::uint64_t estimator_samples = 0;
  std::uint64_t physical_paths = 0;
  std::string backend;
};

// 固定随机输入下的路径估值；反向变量模式下 payoff 为 Z 与 -Z 两条路径的均值。
struct PathEvaluation {
  double discounted_payoff = 0.0;
  double discounted_terminal = 0.0;
  std::uint64_t physical_paths = 1;
};

struct GreekEstimates {
  double delta = 0.0;
  double gamma = 0.0;
  double vega = 0.0;
  double elapsed_ms = 0.0;
  std::string method;
  std::string backend;
};

OptionParams load_option_params(const std::string& path);
SimulationParams load_simulation_params(const std::string& path);

std::string to_string(OptionType type);
std::string to_string(ModelType model);
std::string to_string(VarianceReduction mode);

double black_scholes_price(const OptionParams& option);
GreekEstimates black_scholes_greeks(const OptionParams& option);
double american_binomial_price(const OptionParams& option, int steps = 2000);
GreekEstimates american_binomial_greeks(
    const OptionParams& option, double spot_bump_relative,
    double volatility_bump_absolute, int steps = 2000);
PathEvaluation evaluate_path(const OptionParams& option,
                             const SimulationParams& simulation,
                             const std::vector<double>& standard_normals);
Estimate price_cpu(const OptionParams& option,
                   const SimulationParams& simulation);
GreekEstimates estimate_greeks_cpu(const OptionParams& option,
                                   const SimulationParams& simulation);

bool cuda_backend_available();
Estimate price_cuda(const OptionParams& option,
                    const SimulationParams& simulation);
Estimate price_cuda_multi(const OptionParams& option,
                          const SimulationParams& simulation);
GreekEstimates estimate_greeks_cuda(const OptionParams& option,
                                    const SimulationParams& simulation);

}  // namespace pricing
