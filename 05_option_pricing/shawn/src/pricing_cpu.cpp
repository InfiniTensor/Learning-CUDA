#include "pricing/pricing.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace pricing {
namespace {

constexpr double kConfidenceZ = 1.959963984540054;
constexpr double kInverseSqrtTwoPi = 0.39894228040143267794;

bool is_american(OptionType type) {
  return type == OptionType::kAmericanCall ||
         type == OptionType::kAmericanPut;
}

bool is_call(OptionType type) {
  return type == OptionType::kEuropeanCall ||
         type == OptionType::kAsianCall ||
         type == OptionType::kBarrierCall ||
         type == OptionType::kAmericanCall;
}

int effective_steps(const OptionParams& option,
                    const SimulationParams& simulation) {
  const bool terminal_only = option.type == OptionType::kEuropeanCall ||
                             option.type == OptionType::kEuropeanPut;
  return terminal_only && option.model == ModelType::kBlackScholes
             ? 1
             : simulation.num_steps;
}

int normal_dimensions(const OptionParams& option,
                      const SimulationParams& simulation) {
  return effective_steps(option, simulation) *
         (option.model == ModelType::kHeston ? 2 : 1);
}

double normal_cdf(double value) {
  return 0.5 * std::erfc(-value / std::sqrt(2.0));
}

double normal_pdf(double value) {
  return kInverseSqrtTwoPi * std::exp(-0.5 * value * value);
}

double intrinsic_value(OptionType type, double spot, double strike) {
  return is_call(type) ? std::max(spot - strike, 0.0)
                       : std::max(strike - spot, 0.0);
}

double terminal_payoff(OptionType type, double terminal, double strike,
                       double average, bool barrier_hit) {
  // 按合约类型计算未贴现收益。
  switch (type) {
    case OptionType::kEuropeanCall:
    case OptionType::kAmericanCall:
      return std::max(terminal - strike, 0.0);
    case OptionType::kEuropeanPut:
    case OptionType::kAmericanPut:
      return std::max(strike - terminal, 0.0);
    case OptionType::kAsianCall:
      return std::max(average - strike, 0.0);
    case OptionType::kBarrierCall:
      return barrier_hit ? 0.0 : std::max(terminal - strike, 0.0);
  }
  throw std::logic_error("unknown option type");
}

struct ModelState {
  double spot = 0.0;
  double variance = 0.0;
};

void evolve_model(ModelState& state, const OptionParams& option, double dt,
                  double z_spot, double z_independent) {
  // 推进一个时间步；离散公式与 CUDA 后端一致。
  if (option.model == ModelType::kBlackScholes) {
    const double variance = option.volatility * option.volatility;
    state.spot *= std::exp((option.risk_free_rate - 0.5 * variance) * dt +
                           option.volatility * std::sqrt(dt) * z_spot);
    return;
  }
  if (option.model == ModelType::kLocalVolatility) {
    const double ratio = std::max(state.spot / option.spot, 1e-12);
    const double local_sigma =
        option.volatility * std::pow(ratio, option.local_vol_beta);
    state.spot *=
        std::exp((option.risk_free_rate - 0.5 * local_sigma * local_sigma) * dt +
                 local_sigma * std::sqrt(dt) * z_spot);
    return;
  }

  // Heston 使用非负投影 Euler：系数使用 max(v,0)，更新后再将方差投影到非负区间。
  const double variance = std::max(state.variance, 0.0);
  const double sqrt_variance_dt = std::sqrt(variance * dt);
  state.spot *=
      std::exp((option.risk_free_rate - 0.5 * variance) * dt +
               sqrt_variance_dt * z_spot);
  const double rho_complement =
      std::sqrt(std::max(1.0 - option.heston_rho * option.heston_rho, 0.0));
  const double z_variance =
      option.heston_rho * z_spot + rho_complement * z_independent;
  state.variance = std::max(
      state.variance + option.heston_kappa *
                           (option.heston_theta - variance) * dt +
          option.heston_vol_of_vol * sqrt_variance_dt * z_variance,
      0.0);
}

struct Sample {
  double discounted_payoff = 0.0;
  double discounted_terminal = 0.0;
};

template <typename NormalSource>
Sample draw_sample_with_normal_source(const OptionParams& option,
                                      const SimulationParams& simulation,
                                      NormalSource&& next_normal) {
  // 反向变量的 Z 与 -Z 路径取平均后计为一个样本。
  const bool antithetic =
      simulation.variance_reduction == VarianceReduction::kAntithetic;
  const int steps = effective_steps(option, simulation);
  const double dt = option.maturity / static_cast<double>(steps);

  ModelState state_a{option.spot, option.heston_initial_variance};
  ModelState state_b = state_a;
  double average_a = 0.0;
  double average_b = 0.0;
  bool barrier_a = false;
  bool barrier_b = false;

  for (int step = 0; step < steps; ++step) {
    const double z_spot = next_normal();
    const double z_independent =
        option.model == ModelType::kHeston ? next_normal() : 0.0;
    evolve_model(state_a, option, dt, z_spot, z_independent);
    average_a += state_a.spot;
    barrier_a = barrier_a || state_a.spot >= option.barrier;
    if (antithetic) {
      evolve_model(state_b, option, dt, -z_spot, -z_independent);
      average_b += state_b.spot;
      barrier_b = barrier_b || state_b.spot >= option.barrier;
    }
  }

  average_a /= static_cast<double>(steps);
  const double discount = std::exp(-option.risk_free_rate * option.maturity);
  double payoff = terminal_payoff(option.type, state_a.spot, option.strike,
                                  average_a, barrier_a);
  double terminal = state_a.spot;
  if (antithetic) {
    average_b /= static_cast<double>(steps);
    payoff = 0.5 *
             (payoff + terminal_payoff(option.type, state_b.spot,
                                       option.strike, average_b, barrier_b));
    terminal = 0.5 * (terminal + state_b.spot);
  }
  return {discount * payoff, discount * terminal};
}

template <typename NormalSource>
void draw_spot_path(const OptionParams& option,
                    const SimulationParams& simulation,
                    NormalSource&& next_normal, double* output) {
  // 保存各行权时刻的价格，供 LSM 反向回归使用。
  const int steps = simulation.num_steps;
  const double dt = option.maturity / static_cast<double>(steps);
  ModelState state{option.spot, option.heston_initial_variance};
  output[0] = state.spot;
  for (int step = 1; step <= steps; ++step) {
    const double z_spot = next_normal();
    const double z_independent =
        option.model == ModelType::kHeston ? next_normal() : 0.0;
    evolve_model(state, option, dt, z_spot, z_independent);
    output[step] = state.spot;
  }
}

struct RunningMoments {
  std::uint64_t count = 0;
  double mean_y = 0.0;
  double mean_x = 0.0;
  double sum_square_y = 0.0;
  double sum_square_x = 0.0;
  double sum_cross = 0.0;

  void add(double y, double x) {
    // Welford 在线更新均值、方差和协方差，避免直接累加平方的数值消减。
    ++count;
    const double delta_y = y - mean_y;
    const double delta_x = x - mean_x;
    mean_y += delta_y / static_cast<double>(count);
    mean_x += delta_x / static_cast<double>(count);
    sum_square_y += delta_y * (y - mean_y);
    sum_square_x += delta_x * (x - mean_x);
    sum_cross += delta_y * (x - mean_x);
  }
};

Estimate finalize_estimate(const OptionParams& option,
                           const SimulationParams& simulation,
                           const RunningMoments& moments, double elapsed_ms,
                           const std::string& backend) {
  // 从在线矩计算价格和标准误，并按需施加控制变量修正。
  if (moments.count < 2) {
    throw std::runtime_error("at least two estimator samples are required");
  }
  const double denominator = static_cast<double>(moments.count - 1);
  const double variance_y = moments.sum_square_y / denominator;
  const double variance_x = moments.sum_square_x / denominator;
  const double covariance = moments.sum_cross / denominator;

  double mean = moments.mean_y;
  double variance = variance_y;
  if (simulation.variance_reduction == VarianceReduction::kControlVariate &&
      variance_x > std::numeric_limits<double>::epsilon()) {
    // 在三个风险中性模型下，X=e^{-rT}S_T 的理论期望均为 S_0。
    const double beta = covariance / variance_x;
    mean -= beta * (moments.mean_x - option.spot);
    variance = variance_y + beta * beta * variance_x - 2.0 * beta * covariance;
  }

  variance = std::max(variance, 0.0);
  Estimate estimate;
  estimate.price = mean;
  estimate.standard_error =
      std::sqrt(variance / static_cast<double>(moments.count));
  estimate.confidence_low = mean - kConfidenceZ * estimate.standard_error;
  estimate.confidence_high = mean + kConfidenceZ * estimate.standard_error;
  estimate.elapsed_ms = elapsed_ms;
  estimate.simulation_ms = elapsed_ms;
  estimate.reduction_ms = 0.0;
  estimate.estimator_samples = moments.count;
  estimate.physical_paths =
      moments.count *
      (simulation.variance_reduction == VarianceReduction::kAntithetic ? 2 : 1);
  estimate.backend = backend;
  return estimate;
}

std::uint64_t splitmix64(std::uint64_t value) {
  value += 0x9e3779b97f4a7c15ULL;
  value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31U);
}

std::vector<int> first_primes(int count) {
  // 按所需 Halton 维度生成质数基。
  std::vector<int> primes;
  for (int candidate = 2; static_cast<int>(primes.size()) < count;
       ++candidate) {
    bool prime = true;
    for (int divisor = 2; divisor * divisor <= candidate; ++divisor) {
      if (candidate % divisor == 0) {
        prime = false;
        break;
      }
    }
    if (prime) primes.push_back(candidate);
  }
  return primes;
}

double radical_inverse(std::uint64_t index, int base) {
  double result = 0.0;
  const double inverse_base = 1.0 / static_cast<double>(base);
  double factor = inverse_base;
  while (index > 0) {
    result += static_cast<double>(index % static_cast<std::uint64_t>(base)) *
              factor;
    index /= static_cast<std::uint64_t>(base);
    factor *= inverse_base;
  }
  return result;
}

double randomized_shift(std::uint64_t seed, int replication, int dimension) {
  const std::uint64_t mixed = splitmix64(
      seed ^ (static_cast<std::uint64_t>(replication + 1) *
              0xd1b54a32d192ed03ULL) ^
      (static_cast<std::uint64_t>(dimension + 1) *
       0x94d049bb133111ebULL));
  return static_cast<double>(mixed >> 11U) * 0x1.0p-53;
}

double inverse_normal_cdf(double probability) {
  // Acklam 分段有理近似；输入先夹紧，防止逆变换产生无穷值。
  constexpr double a1 = -3.969683028665376e+01;
  constexpr double a2 = 2.209460984245205e+02;
  constexpr double a3 = -2.759285104469687e+02;
  constexpr double a4 = 1.383577518672690e+02;
  constexpr double a5 = -3.066479806614716e+01;
  constexpr double a6 = 2.506628277459239e+00;
  constexpr double b1 = -5.447609879822406e+01;
  constexpr double b2 = 1.615858368580409e+02;
  constexpr double b3 = -1.556989798598866e+02;
  constexpr double b4 = 6.680131188771972e+01;
  constexpr double b5 = -1.328068155288572e+01;
  constexpr double c1 = -7.784894002430293e-03;
  constexpr double c2 = -3.223964580411365e-01;
  constexpr double c3 = -2.400758277161838e+00;
  constexpr double c4 = -2.549732539343734e+00;
  constexpr double c5 = 4.374664141464968e+00;
  constexpr double c6 = 2.938163982698783e+00;
  constexpr double d1 = 7.784695709041462e-03;
  constexpr double d2 = 3.224671290700398e-01;
  constexpr double d3 = 2.445134137142996e+00;
  constexpr double d4 = 3.754408661907416e+00;
  constexpr double lower = 0.02425;
  constexpr double upper = 1.0 - lower;
  const double p = std::clamp(probability, 1e-15, 1.0 - 1e-15);
  if (p < lower) {
    const double q = std::sqrt(-2.0 * std::log(p));
    return (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) /
           ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0);
  }
  if (p > upper) {
    const double q = std::sqrt(-2.0 * std::log(1.0 - p));
    return -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) /
           ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0);
  }
  const double q = p - 0.5;
  const double r = q * q;
  return (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q /
         (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0);
}

double halton_normal(std::uint64_t path, int dimension, int replication,
                     std::uint64_t seed, const std::vector<int>& primes) {
  double uniform = radical_inverse(path + 1, primes[dimension]) +
                   randomized_shift(seed, replication, dimension);
  uniform -= std::floor(uniform);
  return inverse_normal_cdf(uniform);
}

Estimate price_terminal_pseudorandom(const OptionParams& option,
                                     const SimulationParams& simulation) {
  const auto start = std::chrono::steady_clock::now();
  // MT19937_64：相同 seed 的扰动估值复用随机流。
  std::mt19937_64 generator(simulation.seed);
  std::normal_distribution<double> normal(0.0, 1.0);
  RunningMoments moments;
  for (std::uint64_t path = 0; path < simulation.num_paths; ++path) {
    const Sample sample = draw_sample_with_normal_source(
        option, simulation, [&]() { return normal(generator); });
    moments.add(sample.discounted_payoff, sample.discounted_terminal);
  }
  const auto end = std::chrono::steady_clock::now();
  return finalize_estimate(
      option, simulation, moments,
      std::chrono::duration<double, std::milli>(end - start).count(),
      "cpu-mt19937_64");
}

Estimate price_terminal_halton(const OptionParams& option,
                               const SimulationParams& simulation) {
  // 标准误按独立随机平移的估计值计算，而非按单条 Halton 序列的路径方差计算。
  const auto start = std::chrono::steady_clock::now();
  const int replications = simulation.qmc_replications;
  const std::uint64_t paths_per_replication =
      simulation.num_paths / static_cast<std::uint64_t>(replications);
  if (paths_per_replication < 2) {
    throw std::invalid_argument("each Halton replication needs at least two paths");
  }
  const std::vector<int> primes =
      first_primes(normal_dimensions(option, simulation));
  RunningMoments replication_prices;
  for (int replication = 0; replication < replications; ++replication) {
    RunningMoments path_moments;
    for (std::uint64_t path = 0; path < paths_per_replication; ++path) {
      int dimension = 0;
      const Sample sample = draw_sample_with_normal_source(
          option, simulation,
          [&]() {
            return halton_normal(path, dimension++, replication,
                                 simulation.seed, primes);
          });
      path_moments.add(sample.discounted_payoff, sample.discounted_terminal);
    }
    const Estimate replication_estimate = finalize_estimate(
        option, simulation, path_moments, 0.0, "cpu-rqmc-halton");
    replication_prices.add(replication_estimate.price, 0.0);
  }
  const auto end = std::chrono::steady_clock::now();
  SimulationParams plain = simulation;
  plain.variance_reduction = VarianceReduction::kNone;
  Estimate estimate = finalize_estimate(
      option, plain, replication_prices,
      std::chrono::duration<double, std::milli>(end - start).count(),
      "cpu-rqmc-halton");
  estimate.estimator_samples = simulation.num_paths;
  estimate.physical_paths =
      simulation.num_paths *
      (simulation.variance_reduction == VarianceReduction::kAntithetic ? 2 : 1);
  return estimate;
}

bool solve_regression_3x3(std::array<std::array<double, 4>, 3>& matrix,
                          std::array<double, 3>& coefficients) {
  // 带主元高斯消元，求解 LSM 二次回归的 3×3 方程组。
  for (int pivot = 0; pivot < 3; ++pivot) {
    int best = pivot;
    for (int row = pivot + 1; row < 3; ++row) {
      if (std::abs(matrix[row][pivot]) > std::abs(matrix[best][pivot])) {
        best = row;
      }
    }
    if (std::abs(matrix[best][pivot]) < 1e-14) return false;
    std::swap(matrix[pivot], matrix[best]);
    const double inverse = 1.0 / matrix[pivot][pivot];
    for (int column = pivot; column < 4; ++column) {
      matrix[pivot][column] *= inverse;
    }
    for (int row = 0; row < 3; ++row) {
      if (row == pivot) continue;
      const double factor = matrix[row][pivot];
      for (int column = pivot; column < 4; ++column) {
        matrix[row][column] -= factor * matrix[pivot][column];
      }
    }
  }
  for (int row = 0; row < 3; ++row) coefficients[row] = matrix[row][3];
  return true;
}

template <typename IndexedNormal>
Estimate price_american_lsm_once(const OptionParams& option,
                                 const SimulationParams& simulation,
                                 std::uint64_t path_count,
                                 IndexedNormal&& normal_at,
                                 const std::string& backend) {
  // Longstaff-Schwartz：保存路径，反向回归 continuation value，再逐路径决定提前行权。
  const int steps = simulation.num_steps;
  const std::size_t row_size = static_cast<std::size_t>(steps + 1);
  if (path_count > std::numeric_limits<std::size_t>::max() / row_size) {
    throw std::overflow_error("American path matrix is too large");
  }
  std::vector<double> paths(static_cast<std::size_t>(path_count) * row_size);
  for (std::uint64_t path = 0; path < path_count; ++path) {
    int dimension = 0;
    draw_spot_path(option, simulation,
                   [&]() { return normal_at(path, dimension++); },
                   paths.data() + static_cast<std::size_t>(path) * row_size);
  }

  std::vector<double> cashflow(static_cast<std::size_t>(path_count));
  std::vector<int> exercise_step(static_cast<std::size_t>(path_count), steps);
  for (std::uint64_t path = 0; path < path_count; ++path) {
    cashflow[static_cast<std::size_t>(path)] = intrinsic_value(
        option.type,
        paths[static_cast<std::size_t>(path) * row_size + steps],
        option.strike);
  }
  const double dt = option.maturity / static_cast<double>(steps);
  for (int step = steps - 1; step >= 1; --step) {
    std::array<std::array<double, 4>, 3> equations{};
    std::uint64_t in_the_money = 0;
    for (std::uint64_t path = 0; path < path_count; ++path) {
      const std::size_t index = static_cast<std::size_t>(path);
      const double spot = paths[index * row_size + step];
      if (intrinsic_value(option.type, spot, option.strike) <= 0.0) continue;
      const double x = spot / option.strike;
      const std::array<double, 3> basis = {1.0, x, x * x};
      const double target = cashflow[index] *
          std::exp(-option.risk_free_rate * dt *
                   static_cast<double>(exercise_step[index] - step));
      for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) {
          equations[row][column] += basis[row] * basis[column];
        }
        equations[row][3] += basis[row] * target;
      }
      ++in_the_money;
    }
    if (in_the_money < 3) continue;
    const double ridge = 1e-12 * std::max(equations[0][0], 1.0);
    for (int diagonal = 0; diagonal < 3; ++diagonal) {
      equations[diagonal][diagonal] += ridge;
    }
    std::array<double, 3> coefficients{};
    if (!solve_regression_3x3(equations, coefficients)) continue;
    for (std::uint64_t path = 0; path < path_count; ++path) {
      const std::size_t index = static_cast<std::size_t>(path);
      const double spot = paths[index * row_size + step];
      const double exercise = intrinsic_value(option.type, spot, option.strike);
      if (exercise <= 0.0) continue;
      const double x = spot / option.strike;
      const double continuation =
          coefficients[0] + coefficients[1] * x + coefficients[2] * x * x;
      if (exercise > continuation) {
        cashflow[index] = exercise;
        exercise_step[index] = step;
      }
    }
  }

  RunningMoments moments;
  for (std::uint64_t path = 0; path < path_count; ++path) {
    const std::size_t index = static_cast<std::size_t>(path);
    const double discounted = cashflow[index] *
        std::exp(-option.risk_free_rate * dt * exercise_step[index]);
    moments.add(discounted, 0.0);
  }
  SimulationParams plain = simulation;
  plain.variance_reduction = VarianceReduction::kNone;
  Estimate estimate = finalize_estimate(option, plain, moments, 0.0, backend);
  const double immediate = intrinsic_value(option.type, option.spot, option.strike);
  if (immediate > estimate.price) {
    estimate.price = immediate;
    estimate.standard_error = 0.0;
    estimate.confidence_low = immediate;
    estimate.confidence_high = immediate;
  }
  return estimate;
}

Estimate price_american_cpu(const OptionParams& option,
                            const SimulationParams& simulation) {
  if (simulation.variance_reduction != VarianceReduction::kNone) {
    throw std::invalid_argument(
        "American LSM currently requires variance_reduction=none");
  }
  const auto start = std::chrono::steady_clock::now();
  Estimate estimate;
  if (simulation.rng == "halton") {
    const int replications = simulation.qmc_replications;
    const std::uint64_t paths_per_replication =
        simulation.num_paths / static_cast<std::uint64_t>(replications);
    const std::vector<int> primes =
        first_primes(normal_dimensions(option, simulation));
    RunningMoments replication_prices;
    for (int replication = 0; replication < replications; ++replication) {
      const Estimate current = price_american_lsm_once(
          option, simulation, paths_per_replication,
          [&](std::uint64_t path, int dimension) {
            return halton_normal(path, dimension, replication, simulation.seed,
                                 primes);
          },
          "cpu-rqmc-halton-lsm");
      replication_prices.add(current.price, 0.0);
    }
    SimulationParams plain = simulation;
    plain.variance_reduction = VarianceReduction::kNone;
    estimate = finalize_estimate(option, plain, replication_prices, 0.0,
                                 "cpu-rqmc-halton-lsm");
    estimate.estimator_samples = simulation.num_paths;
    estimate.physical_paths = simulation.num_paths;
  } else {
    std::mt19937_64 generator(simulation.seed);
    std::normal_distribution<double> normal(0.0, 1.0);
    estimate = price_american_lsm_once(
        option, simulation, simulation.num_paths,
        [&](std::uint64_t, int) { return normal(generator); },
        "cpu-mt19937_64-lsm");
  }
  estimate.elapsed_ms = std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - start)
                            .count();
  estimate.simulation_ms = estimate.elapsed_ms;
  return estimate;
}

}  // namespace

PathEvaluation evaluate_path(
    const OptionParams& option, const SimulationParams& simulation,
    const std::vector<double>& standard_normals) {
  // 用固定 Z 序列检验路径演化；美式提前行权需另做 LSM 测试。
  if (is_american(option.type)) {
    throw std::invalid_argument("single-path evaluation is undefined for LSM");
  }
  if (simulation.num_steps <= 0) {
    throw std::invalid_argument("num_steps must be positive");
  }
  const int expected = normal_dimensions(option, simulation);
  if (standard_normals.size() != static_cast<std::size_t>(expected)) {
    throw std::invalid_argument(
        "standard-normal count must match the model path dimension");
  }
  std::size_t index = 0;
  const Sample sample = draw_sample_with_normal_source(
      option, simulation, [&]() { return standard_normals[index++]; });
  return {sample.discounted_payoff, sample.discounted_terminal,
          simulation.variance_reduction == VarianceReduction::kAntithetic ? 2ULL
                                                                           : 1ULL};
}

double black_scholes_price(const OptionParams& option) {
  // 解析解只适用于 Black-Scholes 下的 European call/put。
  if (option.model != ModelType::kBlackScholes ||
      (option.type != OptionType::kEuropeanCall &&
       option.type != OptionType::kEuropeanPut)) {
    throw std::invalid_argument(
        "Black-Scholes reference requires a European call/put under GBM");
  }
  const double discounted_strike =
      option.strike * std::exp(-option.risk_free_rate * option.maturity);
  if (option.volatility == 0.0) {
    return option.type == OptionType::kEuropeanCall
               ? std::max(option.spot - discounted_strike, 0.0)
               : std::max(discounted_strike - option.spot, 0.0);
  }
  const double sigma_sqrt_t = option.volatility * std::sqrt(option.maturity);
  const double d1 =
      (std::log(option.spot / option.strike) +
       (option.risk_free_rate +
        0.5 * option.volatility * option.volatility) * option.maturity) /
      sigma_sqrt_t;
  const double d2 = d1 - sigma_sqrt_t;
  if (option.type == OptionType::kEuropeanCall) {
    return option.spot * normal_cdf(d1) - discounted_strike * normal_cdf(d2);
  }
  return discounted_strike * normal_cdf(-d2) -
         option.spot * normal_cdf(-d1);
}

GreekEstimates black_scholes_greeks(const OptionParams& option) {
  // Black-Scholes 解析 Greeks，作为差分估计的参考。
  if (option.model != ModelType::kBlackScholes ||
      (option.type != OptionType::kEuropeanCall &&
       option.type != OptionType::kEuropeanPut) ||
      option.volatility <= 0.0) {
    throw std::invalid_argument("analytic Greeks require positive-vol GBM European");
  }
  const double sigma_sqrt_t = option.volatility * std::sqrt(option.maturity);
  const double d1 =
      (std::log(option.spot / option.strike) +
       (option.risk_free_rate +
        0.5 * option.volatility * option.volatility) * option.maturity) /
      sigma_sqrt_t;
  GreekEstimates greeks;
  greeks.delta = option.type == OptionType::kEuropeanCall
                     ? normal_cdf(d1)
                     : normal_cdf(d1) - 1.0;
  greeks.gamma = normal_pdf(d1) /
                 (option.spot * option.volatility * std::sqrt(option.maturity));
  greeks.vega = option.spot * normal_pdf(d1) * std::sqrt(option.maturity);
  greeks.method = "black_scholes_analytic";
  greeks.backend = "analytic";
  return greeks;
}

double american_binomial_price(const OptionParams& option, int steps) {
  // CRR 二叉树为 American LSM 提供与 Monte Carlo 独立的确定性参考。
  if (!is_american(option.type) ||
      option.model != ModelType::kBlackScholes || steps <= 0) {
    throw std::invalid_argument("American binomial reference requires GBM American");
  }
  const double dt = option.maturity / static_cast<double>(steps);
  if (option.volatility == 0.0) {
    double best = intrinsic_value(option.type, option.spot, option.strike);
    for (int step = 1; step <= steps; ++step) {
      const double time = dt * step;
      const double spot = option.spot * std::exp(option.risk_free_rate * time);
      best = std::max(best, std::exp(-option.risk_free_rate * time) *
                                intrinsic_value(option.type, spot, option.strike));
    }
    return best;
  }
  const double up = std::exp(option.volatility * std::sqrt(dt));
  const double down = 1.0 / up;
  const double probability =
      (std::exp(option.risk_free_rate * dt) - down) / (up - down);
  if (!(probability >= 0.0 && probability <= 1.0)) {
    throw std::runtime_error("invalid CRR risk-neutral probability");
  }
  const double discount = std::exp(-option.risk_free_rate * dt);
  std::vector<double> values(static_cast<std::size_t>(steps + 1));
  for (int up_moves = 0; up_moves <= steps; ++up_moves) {
    const double spot = option.spot * std::pow(up, up_moves) *
                        std::pow(down, steps - up_moves);
    values[up_moves] = intrinsic_value(option.type, spot, option.strike);
  }
  for (int step = steps - 1; step >= 0; --step) {
    for (int up_moves = 0; up_moves <= step; ++up_moves) {
      const double continuation =
          discount * (probability * values[up_moves + 1] +
                      (1.0 - probability) * values[up_moves]);
      const double spot = option.spot * std::pow(up, up_moves) *
                          std::pow(down, step - up_moves);
      values[up_moves] = std::max(
          continuation, intrinsic_value(option.type, spot, option.strike));
    }
  }
  return values.front();
}

GreekEstimates american_binomial_greeks(
    const OptionParams& option, double spot_bump_relative,
    double volatility_bump_absolute, int steps) {
  // 在 CRR 树上做同口径的中心差分，作为 Monte Carlo Greeks 的独立参考。
  if (!is_american(option.type) ||
      option.model != ModelType::kBlackScholes || option.volatility <= 0.0 ||
      spot_bump_relative <= 0.0 || volatility_bump_absolute <= 0.0 ||
      steps <= 0) {
    throw std::invalid_argument(
        "American CRR Greeks require positive bumps and GBM volatility");
  }
  const auto start = std::chrono::steady_clock::now();
  const double spot_bump = option.spot * spot_bump_relative;
  if (!(spot_bump > 0.0) || option.spot <= spot_bump) {
    throw std::invalid_argument("invalid spot bump for American CRR Greeks");
  }
  // 缩小扰动以保持下扰动为正，避免将中心差分变成单边差分。
  const double volatility_bump =
      std::min(volatility_bump_absolute, option.volatility * 0.5);

  OptionParams spot_up = option;
  OptionParams spot_down = option;
  spot_up.spot += spot_bump;
  spot_down.spot -= spot_bump;
  OptionParams vol_up = option;
  OptionParams vol_down = option;
  vol_up.volatility += volatility_bump;
  vol_down.volatility -= volatility_bump;

  const double base = american_binomial_price(option, steps);
  const double up = american_binomial_price(spot_up, steps);
  const double down = american_binomial_price(spot_down, steps);
  const double v_up = american_binomial_price(vol_up, steps);
  const double v_down = american_binomial_price(vol_down, steps);

  GreekEstimates greeks;
  greeks.delta = (up - down) / (2.0 * spot_bump);
  greeks.gamma = (up - 2.0 * base + down) / (spot_bump * spot_bump);
  greeks.vega = (v_up - v_down) / (2.0 * volatility_bump);
  greeks.elapsed_ms = std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - start)
                          .count();
  greeks.method = "central_finite_difference_crr_binomial_" +
                  std::to_string(steps) + "_steps";
  greeks.backend = "crr_binomial";
  return greeks;
}

Estimate price_cpu(const OptionParams& option,
                   const SimulationParams& simulation) {
  // 按合约类型和随机数方案分派。
  if (simulation.num_paths < 2 || simulation.num_steps <= 0) {
    throw std::invalid_argument("invalid simulation dimensions");
  }
  if (is_american(option.type)) return price_american_cpu(option, simulation);
  return simulation.rng == "halton"
             ? price_terminal_halton(option, simulation)
             : price_terminal_pseudorandom(option, simulation);
}

GreekEstimates estimate_greeks_cpu(const OptionParams& option,
                                   const SimulationParams& simulation) {
  // 各扰动估值复用随机流，以共同随机数降低差分噪声。
  const auto start = std::chrono::steady_clock::now();
  const bool discontinuous_payoff =
      option.type == OptionType::kBarrierCall || is_american(option.type);
  const double spot_bump =
      option.spot *
      std::max(simulation.spot_bump_relative,
               discontinuous_payoff ? 1e-2 : simulation.spot_bump_relative);
  if (!(spot_bump > 0.0) || option.spot <= spot_bump) {
    throw std::invalid_argument("invalid spot bump for Greeks");
  }
  const Estimate base = price_cpu(option, simulation);
  OptionParams spot_up = option;
  OptionParams spot_down = option;
  spot_up.spot += spot_bump;
  spot_down.spot -= spot_bump;
  const Estimate up = price_cpu(spot_up, simulation);
  const Estimate down = price_cpu(spot_down, simulation);

  OptionParams vol_up = option;
  OptionParams vol_down = option;
  double volatility_bump = simulation.volatility_bump_absolute;
  if (option.model == ModelType::kHeston) {
    const double initial_volatility =
        std::sqrt(std::max(option.heston_initial_variance, 0.0));
    volatility_bump =
        std::min(volatility_bump, std::max(initial_volatility * 0.5, 1e-6));
    vol_up.heston_initial_variance =
        (initial_volatility + volatility_bump) *
        (initial_volatility + volatility_bump);
    const double down_volatility =
        std::max(initial_volatility - volatility_bump, 0.0);
    vol_down.heston_initial_variance = down_volatility * down_volatility;
  } else {
    volatility_bump =
        std::min(volatility_bump, std::max(option.volatility * 0.5, 1e-6));
    vol_up.volatility += volatility_bump;
    vol_down.volatility = std::max(option.volatility - volatility_bump, 0.0);
  }
  const Estimate v_up = price_cpu(vol_up, simulation);
  const Estimate v_down = price_cpu(vol_down, simulation);

  GreekEstimates greeks;
  greeks.delta = (up.price - down.price) / (2.0 * spot_bump);
  greeks.gamma =
      (up.price - 2.0 * base.price + down.price) / (spot_bump * spot_bump);
  greeks.vega = (v_up.price - v_down.price) / (2.0 * volatility_bump);
  greeks.elapsed_ms = std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - start)
                          .count();
  greeks.method = "central_finite_difference_common_random_numbers";
  greeks.backend = base.backend;
  return greeks;
}

}  // namespace pricing
