#include "pricing/pricing.hpp"

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cmath>
#include <cstdint>
#include <future>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace pricing {
namespace {

constexpr int kThreads = 256;
constexpr int kMomentCount = 5;
constexpr int kLsmRegressionMomentCount = 8;
constexpr double kConfidenceZ = 1.959963984540054;

void check_cuda(cudaError_t status, const char* expression) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(expression) + ": " +
                             cudaGetErrorString(status));
  }
}

#define PRICING_CUDA_CHECK(expression) check_cuda((expression), #expression)

__device__ std::uint64_t splitmix64(std::uint64_t& state) {
  std::uint64_t value = (state += 0x9e3779b97f4a7c15ULL);
  value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31U);
}

__device__ double custom_uniform(std::uint64_t& state) {
  // 取高 53 位缩放为 FP64 均匀数。
  return (static_cast<double>(splitmix64(state) >> 11U) + 0.5) * 0x1.0p-53;
}

__device__ double custom_normal(std::uint64_t& state) {
  // SplitMix64 均匀数经 Box-Muller 变换生成标准正态数。
  constexpr double kTwoPi = 6.2831853071795864769;
  const double u1 = custom_uniform(state);
  const double u2 = custom_uniform(state);
  return sqrt(-2.0 * log(u1)) * cos(kTwoPi * u2);
}

__device__ double radical_inverse(std::uint64_t index, int base) {
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

__device__ double inverse_normal_cdf(double probability) {
  // 与 CPU 共用 Acklam 逆正态近似。
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
  const double p = fmin(fmax(probability, 1e-15), 1.0 - 1e-15);
  if (p < 0.02425) {
    const double q = sqrt(-2.0 * log(p));
    return (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) /
           ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0);
  }
  if (p > 0.97575) {
    const double q = sqrt(-2.0 * log(1.0 - p));
    return -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) /
           ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0);
  }
  const double q = p - 0.5;
  const double r = q * q;
  return (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q /
         (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0);
}

__device__ double halton_normal(std::uint64_t path, int dimension,
                                int replication, std::uint64_t seed,
                                const int* primes) {
  std::uint64_t state =
      seed ^ (static_cast<std::uint64_t>(replication + 1) *
              0xd1b54a32d192ed03ULL) ^
      (static_cast<std::uint64_t>(dimension + 1) *
       0x94d049bb133111ebULL);
  const double shift =
      static_cast<double>(splitmix64(state) >> 11U) * 0x1.0p-53;
  double uniform = radical_inverse(path + 1, primes[dimension]) + shift;
  uniform -= floor(uniform);
  return inverse_normal_cdf(uniform);
}

__device__ double next_normal(int rng_mode,
                              curandStatePhilox4_32_10_t& state,
                              std::uint64_t& custom_state,
                              std::uint64_t path, int dimension,
                              int qmc_replication, std::uint64_t seed,
                              const int* primes) {
  if (rng_mode == 0) return curand_normal_double(&state);
  if (rng_mode == 1) return custom_normal(custom_state);
  return halton_normal(path, dimension, qmc_replication, seed, primes);
}

__device__ double option_payoff(int option_type, double terminal, double strike,
                                double average, bool barrier_hit) {
  if (option_type == static_cast<int>(OptionType::kEuropeanCall) ||
      option_type == static_cast<int>(OptionType::kAmericanCall)) {
    return fmax(terminal - strike, 0.0);
  }
  if (option_type == static_cast<int>(OptionType::kEuropeanPut) ||
      option_type == static_cast<int>(OptionType::kAmericanPut)) {
    return fmax(strike - terminal, 0.0);
  }
  if (option_type == static_cast<int>(OptionType::kAsianCall)) {
    return fmax(average - strike, 0.0);
  }
  return barrier_hit ? 0.0 : fmax(terminal - strike, 0.0);
}

struct DeviceModelState {
  double spot;
  double variance;
};

__device__ void evolve_model(
    DeviceModelState& state, int model, double initial_spot,
    double risk_free_rate, double volatility, double local_vol_beta,
    double heston_kappa, double heston_theta, double heston_vol_of_vol,
    double heston_rho, double dt, double z_spot, double z_independent) {
  // CPU/GPU 使用同一非负投影 Heston Euler 与 Local Vol 离散公式。
  if (model == static_cast<int>(ModelType::kBlackScholes)) {
    const double variance = volatility * volatility;
    state.spot *= exp((risk_free_rate - 0.5 * variance) * dt +
                      volatility * sqrt(dt) * z_spot);
    return;
  }
  if (model == static_cast<int>(ModelType::kLocalVolatility)) {
    const double ratio = fmax(state.spot / initial_spot, 1e-12);
    const double local_sigma = volatility * pow(ratio, local_vol_beta);
    state.spot *= exp((risk_free_rate - 0.5 * local_sigma * local_sigma) * dt +
                      local_sigma * sqrt(dt) * z_spot);
    return;
  }
  const double variance = fmax(state.variance, 0.0);
  const double sqrt_variance_dt = sqrt(variance * dt);
  state.spot *= exp((risk_free_rate - 0.5 * variance) * dt +
                    sqrt_variance_dt * z_spot);
  const double z_variance =
      heston_rho * z_spot +
      sqrt(fmax(1.0 - heston_rho * heston_rho, 0.0)) * z_independent;
  state.variance =
      fmax(state.variance + heston_kappa * (heston_theta - variance) * dt +
               heston_vol_of_vol * sqrt_variance_dt * z_variance,
           0.0);
}

__global__ void monte_carlo_kernel(
    std::uint64_t num_paths, int num_steps, std::uint64_t seed, int rng_mode,
    int variance_mode, int option_type, int model, double initial_spot,
    double strike, double risk_free_rate, double volatility, double maturity,
    double barrier, double heston_initial_variance, double heston_kappa,
    double heston_theta, double heston_vol_of_vol, double heston_rho,
    double local_vol_beta, int qmc_replication, const int* primes,
    double* block_moments) {
  // Philox 子序列由路径编号确定，不依赖线程调度。
  double local[kMomentCount] = {0.0, 0.0, 0.0, 0.0, 0.0};
  const std::uint64_t thread_id =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::uint64_t stride =
      static_cast<std::uint64_t>(gridDim.x) * blockDim.x;

  for (std::uint64_t path = thread_id; path < num_paths; path += stride) {
    curandStatePhilox4_32_10_t curand_state;
    if (rng_mode == 0) {
      curand_init(seed, path, 0, &curand_state);
    }
    std::uint64_t custom_state =
        seed ^ ((path + 1ULL) * 0xd1342543de82ef95ULL);

    const bool antithetic =
        variance_mode == static_cast<int>(VarianceReduction::kAntithetic);
    const int steps =
        (option_type == static_cast<int>(OptionType::kEuropeanCall) ||
         option_type == static_cast<int>(OptionType::kEuropeanPut)) &&
        model == static_cast<int>(ModelType::kBlackScholes)
            ? 1
            : num_steps;
    const double dt = maturity / static_cast<double>(steps);

    DeviceModelState state_a{initial_spot, heston_initial_variance};
    DeviceModelState state_b = state_a;
    double average_a = 0.0;
    double average_b = 0.0;
    bool barrier_a = false;
    bool barrier_b = false;
    int dimension = 0;
    for (int step = 0; step < steps; ++step) {
      const double z_spot = next_normal(
          rng_mode, curand_state, custom_state, path, dimension++,
          qmc_replication, seed, primes);
      const double z_independent =
          model == static_cast<int>(ModelType::kHeston)
              ? next_normal(rng_mode, curand_state, custom_state, path,
                            dimension++, qmc_replication, seed, primes)
              : 0.0;
      evolve_model(state_a, model, initial_spot, risk_free_rate, volatility,
                   local_vol_beta, heston_kappa, heston_theta,
                   heston_vol_of_vol, heston_rho, dt, z_spot, z_independent);
      average_a += state_a.spot;
      barrier_a = barrier_a || state_a.spot >= barrier;
      if (antithetic) {
        evolve_model(state_b, model, initial_spot, risk_free_rate, volatility,
                     local_vol_beta, heston_kappa, heston_theta,
                     heston_vol_of_vol, heston_rho, dt, -z_spot,
                     -z_independent);
        average_b += state_b.spot;
        barrier_b = barrier_b || state_b.spot >= barrier;
      }
    }

    average_a /= static_cast<double>(steps);
    double payoff = option_payoff(option_type, state_a.spot, strike, average_a,
                                  barrier_a);
    double terminal = state_a.spot;
    if (antithetic) {
      average_b /= static_cast<double>(steps);
      payoff = 0.5 *
               (payoff + option_payoff(option_type, state_b.spot, strike, average_b,
                                       barrier_b));
      terminal = 0.5 * (terminal + state_b.spot);
    }
    const double discount = exp(-risk_free_rate * maturity);
    const double y = discount * payoff;
    const double x = discount * terminal;
    local[0] += y;
    local[1] += y * y;
    local[2] += x;
    local[3] += x * x;
    local[4] += x * y;
  }

  extern __shared__ double shared[];
  // 块内归约五个原始矩，只写回块级结果。
  for (int metric = 0; metric < kMomentCount; ++metric) {
    shared[metric * blockDim.x + threadIdx.x] = local[metric];
  }
  __syncthreads();
  for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (threadIdx.x < offset) {
      for (int metric = 0; metric < kMomentCount; ++metric) {
        shared[metric * blockDim.x + threadIdx.x] +=
            shared[metric * blockDim.x + threadIdx.x + offset];
      }
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    for (int metric = 0; metric < kMomentCount; ++metric) {
      block_moments[static_cast<std::size_t>(blockIdx.x) * kMomentCount +
                    metric] = shared[metric * blockDim.x];
    }
  }
}

__global__ void reduce_moments_kernel(const double* block_moments,
                                      int num_blocks, double* moments) {
  // 每块合并一个统计量，得到五个全局原始矩。
  const int metric = blockIdx.x;
  double local = 0.0;
  for (int block = threadIdx.x; block < num_blocks; block += blockDim.x) {
    local += block_moments[static_cast<std::size_t>(block) * kMomentCount +
                           metric];
  }
  extern __shared__ double shared_reduction[];
  shared_reduction[threadIdx.x] = local;
  __syncthreads();
  for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (threadIdx.x < offset) {
      shared_reduction[threadIdx.x] +=
          shared_reduction[threadIdx.x + offset];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) moments[metric] = shared_reduction[0];
}

__global__ void simulate_american_paths_kernel(
    std::uint64_t num_paths, int num_steps, std::uint64_t seed, int rng_mode,
    int model, double initial_spot, double risk_free_rate, double volatility,
    double maturity, double heston_initial_variance, double heston_kappa,
    double heston_theta, double heston_vol_of_vol, double heston_rho,
    double local_vol_beta, int qmc_replication, const int* primes,
    double* paths) {
  // 每个逻辑路径保存一行，列对应各行权时刻。
  const std::uint64_t thread_id =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::uint64_t stride =
      static_cast<std::uint64_t>(gridDim.x) * blockDim.x;
  const double dt = maturity / static_cast<double>(num_steps);
  const std::size_t row_size = static_cast<std::size_t>(num_steps + 1);
  for (std::uint64_t path = thread_id; path < num_paths; path += stride) {
    curandStatePhilox4_32_10_t curand_state;
    if (rng_mode == 0) curand_init(seed, path, 0, &curand_state);
    std::uint64_t custom_state =
        seed ^ ((path + 1ULL) * 0xd1342543de82ef95ULL);
    DeviceModelState state{initial_spot, heston_initial_variance};
    const std::size_t row = static_cast<std::size_t>(path) * row_size;
    paths[row] = initial_spot;
    int dimension = 0;
    for (int step = 1; step <= num_steps; ++step) {
      const double z_spot = next_normal(
          rng_mode, curand_state, custom_state, path, dimension++,
          qmc_replication, seed, primes);
      const double z_independent =
          model == static_cast<int>(ModelType::kHeston)
              ? next_normal(rng_mode, curand_state, custom_state, path,
                            dimension++, qmc_replication, seed, primes)
              : 0.0;
      evolve_model(state, model, initial_spot, risk_free_rate, volatility,
                   local_vol_beta, heston_kappa, heston_theta,
                   heston_vol_of_vol, heston_rho, dt, z_spot, z_independent);
      paths[row + step] = state.spot;
    }
  }
}

__global__ void initialize_american_cashflow_kernel(
    const double* paths, std::uint64_t num_paths, int num_steps,
    int option_type, double strike, double* cashflow, int* exercise_step) {
  const std::uint64_t thread_id =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::uint64_t stride =
      static_cast<std::uint64_t>(gridDim.x) * blockDim.x;
  const std::size_t row_size = static_cast<std::size_t>(num_steps + 1);
  for (std::uint64_t path = thread_id; path < num_paths; path += stride) {
    const double terminal =
        paths[static_cast<std::size_t>(path) * row_size + num_steps];
    cashflow[path] = option_payoff(option_type, terminal, strike, 0.0, false);
    exercise_step[path] = num_steps;
  }
}

__global__ void lsm_regression_moments_kernel(
    const double* paths, const double* cashflow, const int* exercise_step,
    std::uint64_t num_paths, int num_steps, int current_step, int option_type,
    double strike, double risk_free_rate, double dt, double* block_moments) {
  // 输出块级 [1,x,x²,x³,x⁴,y,xy,x²y] 之和，供主机合并并求解回归。
  double local[kLsmRegressionMomentCount] = {};
  const std::uint64_t thread_id =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::uint64_t stride =
      static_cast<std::uint64_t>(gridDim.x) * blockDim.x;
  const std::size_t row_size = static_cast<std::size_t>(num_steps + 1);
  for (std::uint64_t path = thread_id; path < num_paths; path += stride) {
    const double spot =
        paths[static_cast<std::size_t>(path) * row_size + current_step];
    const double exercise =
        option_payoff(option_type, spot, strike, 0.0, false);
    if (exercise <= 0.0) continue;
    const double x = spot / strike;
    const double x2 = x * x;
    const double target =
        cashflow[path] *
        exp(-risk_free_rate * dt *
            static_cast<double>(exercise_step[path] - current_step));
    local[0] += 1.0;
    local[1] += x;
    local[2] += x2;
    local[3] += x2 * x;
    local[4] += x2 * x2;
    local[5] += target;
    local[6] += x * target;
    local[7] += x2 * target;
  }
  extern __shared__ double shared[];
  for (int metric = 0; metric < kLsmRegressionMomentCount; ++metric) {
    shared[metric * blockDim.x + threadIdx.x] = local[metric];
  }
  __syncthreads();
  for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (threadIdx.x < offset) {
      for (int metric = 0; metric < kLsmRegressionMomentCount; ++metric) {
        shared[metric * blockDim.x + threadIdx.x] +=
            shared[metric * blockDim.x + threadIdx.x + offset];
      }
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    for (int metric = 0; metric < kLsmRegressionMomentCount; ++metric) {
      block_moments[static_cast<std::size_t>(blockIdx.x) *
                        kLsmRegressionMomentCount +
                    metric] = shared[metric * blockDim.x];
    }
  }
}

__global__ void apply_american_exercise_kernel(
    const double* paths, std::uint64_t num_paths, int num_steps,
    int current_step, int option_type, double strike, double coefficient0,
    double coefficient1, double coefficient2, double* cashflow,
    int* exercise_step) {
  // 根据回归系数更新逐路径的行权时刻与现金流。
  const std::uint64_t thread_id =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::uint64_t stride =
      static_cast<std::uint64_t>(gridDim.x) * blockDim.x;
  const std::size_t row_size = static_cast<std::size_t>(num_steps + 1);
  for (std::uint64_t path = thread_id; path < num_paths; path += stride) {
    const double spot =
        paths[static_cast<std::size_t>(path) * row_size + current_step];
    const double exercise =
        option_payoff(option_type, spot, strike, 0.0, false);
    if (exercise <= 0.0) continue;
    const double x = spot / strike;
    const double continuation =
        coefficient0 + coefficient1 * x + coefficient2 * x * x;
    if (exercise > continuation) {
      cashflow[path] = exercise;
      exercise_step[path] = current_step;
    }
  }
}

__global__ void american_cashflow_moments_kernel(
    const double* cashflow, const int* exercise_step, std::uint64_t num_paths,
    double risk_free_rate, double dt, double* block_moments) {
  double sum = 0.0;
  double sum_square = 0.0;
  const std::uint64_t thread_id =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::uint64_t stride =
      static_cast<std::uint64_t>(gridDim.x) * blockDim.x;
  for (std::uint64_t path = thread_id; path < num_paths; path += stride) {
    const double discounted =
        cashflow[path] * exp(-risk_free_rate * dt * exercise_step[path]);
    sum += discounted;
    sum_square += discounted * discounted;
  }
  extern __shared__ double shared[];
  shared[threadIdx.x] = sum;
  shared[blockDim.x + threadIdx.x] = sum_square;
  __syncthreads();
  for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (threadIdx.x < offset) {
      shared[threadIdx.x] += shared[threadIdx.x + offset];
      shared[blockDim.x + threadIdx.x] +=
          shared[blockDim.x + threadIdx.x + offset];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    block_moments[static_cast<std::size_t>(blockIdx.x) * 2] = shared[0];
    block_moments[static_cast<std::size_t>(blockIdx.x) * 2 + 1] =
        shared[blockDim.x];
  }
}

Estimate finalize(const OptionParams& option,
                  const SimulationParams& simulation, const double* moments,
                  float simulation_ms, float reduction_ms) {
  // 从原始矩计算样本方差、控制变量估计和正态近似置信区间。
  const double count = static_cast<double>(simulation.num_paths);
  const double mean_y = moments[0] / count;
  const double mean_x = moments[2] / count;
  const double denominator = count - 1.0;
  const double variance_y =
      std::max((moments[1] - count * mean_y * mean_y) / denominator, 0.0);
  const double variance_x =
      std::max((moments[3] - count * mean_x * mean_x) / denominator, 0.0);
  const double covariance =
      (moments[4] - count * mean_x * mean_y) / denominator;

  double price = mean_y;
  double variance = variance_y;
  if (simulation.variance_reduction == VarianceReduction::kControlVariate &&
      variance_x > std::numeric_limits<double>::epsilon()) {
    const double beta = covariance / variance_x;
    price -= beta * (mean_x - option.spot);
    variance = variance_y + beta * beta * variance_x - 2.0 * beta * covariance;
  }
  variance = std::max(variance, 0.0);

  Estimate estimate;
  estimate.price = price;
  estimate.standard_error = std::sqrt(variance / count);
  estimate.confidence_low = price - kConfidenceZ * estimate.standard_error;
  estimate.confidence_high = price + kConfidenceZ * estimate.standard_error;
  estimate.simulation_ms = simulation_ms;
  estimate.reduction_ms = reduction_ms;
  estimate.elapsed_ms = simulation_ms + reduction_ms;
  estimate.estimator_samples = simulation.num_paths;
  estimate.physical_paths =
      simulation.num_paths *
      (simulation.variance_reduction == VarianceReduction::kAntithetic ? 2 : 1);
  estimate.backend = simulation.rng == "custom"
                         ? "cuda-custom-splitmix64"
                         : (simulation.rng == "halton" ? "cuda-rqmc-halton"
                                                        : "cuda-curand-philox");
  return estimate;
}

std::vector<int> first_primes_host(int count) {
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

int normal_dimensions_host(const OptionParams& option,
                           const SimulationParams& simulation) {
  const bool terminal_only = option.type == OptionType::kEuropeanCall ||
                             option.type == OptionType::kEuropeanPut;
  const int steps = terminal_only && option.model == ModelType::kBlackScholes
                        ? 1
                        : simulation.num_steps;
  return steps * (option.model == ModelType::kHeston ? 2 : 1);
}

bool solve_lsm_regression(std::array<std::array<double, 4>, 3>& matrix,
                          std::array<double, 3>& coefficients) {
  // 与 CPU LSM 相同的带主元 3×3 高斯消元。
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

Estimate run_american_cuda_once(const OptionParams& option,
                                const SimulationParams& simulation,
                                int qmc_replication) {
  // GPU 生成路径、归约块矩并更新行权；主机合并块矩、求解回归和统计最终现金流。
  int device = 0;
  PRICING_CUDA_CHECK(cudaGetDevice(&device));
  cudaDeviceProp properties{};
  PRICING_CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
  const std::uint64_t required_blocks =
      (simulation.num_paths + kThreads - 1) / kThreads;
  const int blocks = static_cast<int>(std::min<std::uint64_t>(
      required_blocks, std::max(properties.multiProcessorCount * 32, 1)));
  const std::size_t row_size =
      static_cast<std::size_t>(simulation.num_steps + 1);
  if (simulation.num_paths >
      std::numeric_limits<std::size_t>::max() / row_size) {
    throw std::overflow_error("CUDA American path matrix overflows size_t");
  }
  const std::size_t path_elements =
      static_cast<std::size_t>(simulation.num_paths) * row_size;
  std::size_t free_bytes = 0;
  std::size_t total_bytes = 0;
  PRICING_CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
  const std::size_t required_bytes =
      path_elements * sizeof(double) +
      static_cast<std::size_t>(simulation.num_paths) *
          (sizeof(double) + sizeof(int)) +
      static_cast<std::size_t>(blocks) * kLsmRegressionMomentCount *
          sizeof(double);
  if (required_bytes > free_bytes * 9 / 10) {
    throw std::runtime_error(
        "American LSM path matrix exceeds 90% of available device memory");
  }

  double* device_paths = nullptr;
  double* device_cashflow = nullptr;
  int* device_exercise_step = nullptr;
  double* device_block_moments = nullptr;
  int* device_primes = nullptr;
  PRICING_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_paths),
                                path_elements * sizeof(double)));
  PRICING_CUDA_CHECK(cudaMalloc(
      reinterpret_cast<void**>(&device_cashflow),
      static_cast<std::size_t>(simulation.num_paths) * sizeof(double)));
  PRICING_CUDA_CHECK(cudaMalloc(
      reinterpret_cast<void**>(&device_exercise_step),
      static_cast<std::size_t>(simulation.num_paths) * sizeof(int)));
  PRICING_CUDA_CHECK(cudaMalloc(
      reinterpret_cast<void**>(&device_block_moments),
      static_cast<std::size_t>(blocks) * kLsmRegressionMomentCount *
          sizeof(double)));

  std::vector<int> primes;
  if (simulation.rng == "halton") {
    primes = first_primes_host(normal_dimensions_host(option, simulation));
    PRICING_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_primes),
                                  primes.size() * sizeof(int)));
    PRICING_CUDA_CHECK(cudaMemcpy(device_primes, primes.data(),
                                 primes.size() * sizeof(int),
                                 cudaMemcpyHostToDevice));
  }

  const auto start = std::chrono::steady_clock::now();
  const int rng_mode = simulation.rng == "custom"
                           ? 1
                           : (simulation.rng == "halton" ? 2 : 0);
  simulate_american_paths_kernel<<<blocks, kThreads>>>(
      simulation.num_paths, simulation.num_steps, simulation.seed, rng_mode,
      static_cast<int>(option.model), option.spot, option.risk_free_rate,
      option.volatility, option.maturity, option.heston_initial_variance,
      option.heston_kappa, option.heston_theta, option.heston_vol_of_vol,
      option.heston_rho, option.local_vol_beta, qmc_replication, device_primes,
      device_paths);
  PRICING_CUDA_CHECK(cudaGetLastError());
  initialize_american_cashflow_kernel<<<blocks, kThreads>>>(
      device_paths, simulation.num_paths, simulation.num_steps,
      static_cast<int>(option.type), option.strike, device_cashflow,
      device_exercise_step);
  PRICING_CUDA_CHECK(cudaGetLastError());

  const double dt = option.maturity / simulation.num_steps;
  std::vector<double> host_block_moments(
      static_cast<std::size_t>(blocks) * kLsmRegressionMomentCount);
  for (int step = simulation.num_steps - 1; step >= 1; --step) {
    lsm_regression_moments_kernel<<<
        blocks, kThreads,
        kLsmRegressionMomentCount * kThreads * sizeof(double)>>>(
        device_paths, device_cashflow, device_exercise_step,
        simulation.num_paths, simulation.num_steps, step,
        static_cast<int>(option.type), option.strike, option.risk_free_rate,
        dt, device_block_moments);
    PRICING_CUDA_CHECK(cudaGetLastError());
    PRICING_CUDA_CHECK(cudaMemcpy(
        host_block_moments.data(), device_block_moments,
        host_block_moments.size() * sizeof(double), cudaMemcpyDeviceToHost));
    std::array<double, kLsmRegressionMomentCount> moments{};
    for (int block = 0; block < blocks; ++block) {
      for (int metric = 0; metric < kLsmRegressionMomentCount; ++metric) {
        moments[metric] +=
            host_block_moments[static_cast<std::size_t>(block) *
                                   kLsmRegressionMomentCount +
                               metric];
      }
    }
    if (moments[0] < 3.0) continue;
    std::array<std::array<double, 4>, 3> equations = {{
        {{moments[0], moments[1], moments[2], moments[5]}},
        {{moments[1], moments[2], moments[3], moments[6]}},
        {{moments[2], moments[3], moments[4], moments[7]}},
    }};
    const double ridge = 1e-12 * std::max(moments[0], 1.0);
    for (int diagonal = 0; diagonal < 3; ++diagonal) {
      equations[diagonal][diagonal] += ridge;
    }
    std::array<double, 3> coefficients{};
    if (!solve_lsm_regression(equations, coefficients)) continue;
    apply_american_exercise_kernel<<<blocks, kThreads>>>(
        device_paths, simulation.num_paths, simulation.num_steps, step,
        static_cast<int>(option.type), option.strike, coefficients[0],
        coefficients[1], coefficients[2], device_cashflow,
        device_exercise_step);
    PRICING_CUDA_CHECK(cudaGetLastError());
  }

  american_cashflow_moments_kernel<<<blocks, kThreads,
                                     2 * kThreads * sizeof(double)>>>(
      device_cashflow, device_exercise_step, simulation.num_paths,
      option.risk_free_rate, dt, device_block_moments);
  PRICING_CUDA_CHECK(cudaGetLastError());
  std::vector<double> host_cashflow_moments(static_cast<std::size_t>(blocks) * 2);
  PRICING_CUDA_CHECK(cudaMemcpy(
      host_cashflow_moments.data(), device_block_moments,
      host_cashflow_moments.size() * sizeof(double), cudaMemcpyDeviceToHost));
  double sum = 0.0;
  double sum_square = 0.0;
  for (int block = 0; block < blocks; ++block) {
    sum += host_cashflow_moments[static_cast<std::size_t>(block) * 2];
    sum_square +=
        host_cashflow_moments[static_cast<std::size_t>(block) * 2 + 1];
  }
  const auto end = std::chrono::steady_clock::now();

  if (device_primes != nullptr) PRICING_CUDA_CHECK(cudaFree(device_primes));
  PRICING_CUDA_CHECK(cudaFree(device_block_moments));
  PRICING_CUDA_CHECK(cudaFree(device_exercise_step));
  PRICING_CUDA_CHECK(cudaFree(device_cashflow));
  PRICING_CUDA_CHECK(cudaFree(device_paths));

  const double count = static_cast<double>(simulation.num_paths);
  const double price = sum / count;
  const double variance =
      std::max((sum_square - count * price * price) / (count - 1.0), 0.0);
  Estimate estimate;
  estimate.price = price;
  estimate.standard_error = std::sqrt(variance / count);
  estimate.confidence_low = price - kConfidenceZ * estimate.standard_error;
  estimate.confidence_high = price + kConfidenceZ * estimate.standard_error;
  const double immediate = option.type == OptionType::kAmericanCall
                               ? std::max(option.spot - option.strike, 0.0)
                               : std::max(option.strike - option.spot, 0.0);
  if (immediate > estimate.price) {
    estimate.price = immediate;
    estimate.standard_error = 0.0;
    estimate.confidence_low = immediate;
    estimate.confidence_high = immediate;
  }
  estimate.elapsed_ms =
      std::chrono::duration<double, std::milli>(end - start).count();
  estimate.simulation_ms = estimate.elapsed_ms;
  estimate.estimator_samples = simulation.num_paths;
  estimate.physical_paths = simulation.num_paths;
  estimate.backend = simulation.rng == "halton"
                         ? "cuda-rqmc-halton-lsm"
                         : (simulation.rng == "custom"
                                ? "cuda-custom-splitmix64-lsm"
                                : "cuda-curand-philox-lsm");
  return estimate;
}

Estimate price_american_cuda_impl(const OptionParams& option,
                                  const SimulationParams& simulation) {
  if (simulation.variance_reduction != VarianceReduction::kNone) {
    throw std::invalid_argument(
        "CUDA American LSM requires variance_reduction=none");
  }
  if (simulation.rng != "halton") {
    return run_american_cuda_once(option, simulation, 0);
  }
  const int replications = simulation.qmc_replications;
  SimulationParams current = simulation;
  current.num_paths =
      simulation.num_paths / static_cast<std::uint64_t>(replications);
  std::vector<Estimate> estimates;
  estimates.reserve(static_cast<std::size_t>(replications));
  double price_sum = 0.0;
  for (int replication = 0; replication < replications; ++replication) {
    estimates.push_back(run_american_cuda_once(option, current, replication));
    price_sum += estimates.back().price;
  }
  const double price = price_sum / replications;
  double square_sum = 0.0;
  double elapsed_ms = 0.0;
  for (const auto& estimate : estimates) {
    const double difference = estimate.price - price;
    square_sum += difference * difference;
    elapsed_ms += estimate.elapsed_ms;
  }
  Estimate combined;
  combined.price = price;
  combined.standard_error = std::sqrt(
      square_sum /
      (static_cast<double>(replications - 1) * replications));
  combined.confidence_low = price - kConfidenceZ * combined.standard_error;
  combined.confidence_high = price + kConfidenceZ * combined.standard_error;
  combined.elapsed_ms = elapsed_ms;
  combined.simulation_ms = elapsed_ms;
  combined.estimator_samples = simulation.num_paths;
  combined.physical_paths = simulation.num_paths;
  combined.backend = "cuda-rqmc-halton-lsm";
  return combined;
}

}  // namespace

bool cuda_backend_available() {
  int count = 0;
  const cudaError_t status = cudaGetDeviceCount(&count);
  if (status != cudaSuccess) {
    cudaGetLastError();
    return false;
  }
  return count > 0;
}

Estimate price_cuda(const OptionParams& option,
                    const SimulationParams& simulation) {
  // 分配设备资源、启动内核，并用 CUDA Event 记录两个阶段的耗时。
  if (!cuda_backend_available()) {
    throw std::runtime_error("no CUDA device is available");
  }
  if (simulation.num_paths < 2 || simulation.num_steps <= 0) {
    throw std::invalid_argument("invalid simulation dimensions");
  }
  if (option.type == OptionType::kAmericanCall ||
      option.type == OptionType::kAmericanPut) {
    return price_american_cuda_impl(option, simulation);
  }

  const auto run_once = [&](const SimulationParams& current,
                            int qmc_replication) {
    int device = 0;
    PRICING_CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp properties{};
    PRICING_CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
    const std::uint64_t required_blocks =
        (current.num_paths + kThreads - 1) / kThreads;
    const int resident_work_blocks =
        std::max(properties.multiProcessorCount * 32, 1);
    const int blocks = static_cast<int>(
        std::min<std::uint64_t>(required_blocks, resident_work_blocks));

    double* device_block_moments = nullptr;
    double* device_moments = nullptr;
    int* device_primes = nullptr;
    PRICING_CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&device_block_moments),
        static_cast<std::size_t>(blocks) * kMomentCount * sizeof(double)));
    PRICING_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_moments),
                                  kMomentCount * sizeof(double)));
    std::vector<int> primes;
    if (current.rng == "halton") {
      primes = first_primes_host(normal_dimensions_host(option, current));
      PRICING_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_primes),
                                    primes.size() * sizeof(int)));
      PRICING_CUDA_CHECK(cudaMemcpy(device_primes, primes.data(),
                                   primes.size() * sizeof(int),
                                   cudaMemcpyHostToDevice));
    }

    cudaEvent_t start = nullptr;
    cudaEvent_t simulation_stop = nullptr;
    cudaEvent_t stop = nullptr;
    PRICING_CUDA_CHECK(cudaEventCreate(&start));
    PRICING_CUDA_CHECK(cudaEventCreate(&simulation_stop));
    PRICING_CUDA_CHECK(cudaEventCreate(&stop));
    PRICING_CUDA_CHECK(cudaEventRecord(start));
    const int rng_mode = current.rng == "custom"
                             ? 1
                             : (current.rng == "halton" ? 2 : 0);
    monte_carlo_kernel<<<blocks, kThreads,
                         kMomentCount * kThreads * sizeof(double)>>>(
        current.num_paths, current.num_steps, current.seed, rng_mode,
        static_cast<int>(current.variance_reduction),
        static_cast<int>(option.type), static_cast<int>(option.model),
        option.spot, option.strike, option.risk_free_rate, option.volatility,
        option.maturity, option.barrier, option.heston_initial_variance,
        option.heston_kappa, option.heston_theta, option.heston_vol_of_vol,
        option.heston_rho, option.local_vol_beta, qmc_replication,
        device_primes, device_block_moments);
    PRICING_CUDA_CHECK(cudaGetLastError());
    PRICING_CUDA_CHECK(cudaEventRecord(simulation_stop));
    reduce_moments_kernel<<<kMomentCount, kThreads,
                            kThreads * sizeof(double)>>>(
        device_block_moments, blocks, device_moments);
    PRICING_CUDA_CHECK(cudaGetLastError());
    PRICING_CUDA_CHECK(cudaEventRecord(stop));
    PRICING_CUDA_CHECK(cudaEventSynchronize(stop));
    float simulation_ms = 0.0f;
    float reduction_ms = 0.0f;
    PRICING_CUDA_CHECK(
        cudaEventElapsedTime(&simulation_ms, start, simulation_stop));
    PRICING_CUDA_CHECK(
        cudaEventElapsedTime(&reduction_ms, simulation_stop, stop));

    double moments[kMomentCount] = {};
    PRICING_CUDA_CHECK(cudaMemcpy(moments, device_moments,
                                 kMomentCount * sizeof(double),
                                 cudaMemcpyDeviceToHost));
    PRICING_CUDA_CHECK(cudaEventDestroy(stop));
    PRICING_CUDA_CHECK(cudaEventDestroy(simulation_stop));
    PRICING_CUDA_CHECK(cudaEventDestroy(start));
    if (device_primes != nullptr) PRICING_CUDA_CHECK(cudaFree(device_primes));
    PRICING_CUDA_CHECK(cudaFree(device_moments));
    PRICING_CUDA_CHECK(cudaFree(device_block_moments));
    return finalize(option, current, moments, simulation_ms, reduction_ms);
  };

  if (simulation.rng != "halton") return run_once(simulation, 0);

  const int replications = simulation.qmc_replications;
  SimulationParams current = simulation;
  current.num_paths =
      simulation.num_paths / static_cast<std::uint64_t>(replications);
  if (current.num_paths < 2) {
    throw std::invalid_argument("each CUDA Halton replication needs two paths");
  }
  std::vector<Estimate> estimates;
  estimates.reserve(static_cast<std::size_t>(replications));
  double price_sum = 0.0;
  for (int replication = 0; replication < replications; ++replication) {
    estimates.push_back(run_once(current, replication));
    price_sum += estimates.back().price;
  }
  const double price = price_sum / static_cast<double>(replications);
  double square_sum = 0.0;
  double simulation_ms = 0.0;
  double reduction_ms = 0.0;
  for (const auto& estimate : estimates) {
    const double difference = estimate.price - price;
    square_sum += difference * difference;
    simulation_ms += estimate.simulation_ms;
    reduction_ms += estimate.reduction_ms;
  }
  Estimate combined;
  combined.price = price;
  combined.standard_error = std::sqrt(
      square_sum /
      (static_cast<double>(replications - 1) * replications));
  combined.confidence_low = price - kConfidenceZ * combined.standard_error;
  combined.confidence_high = price + kConfidenceZ * combined.standard_error;
  combined.simulation_ms = simulation_ms;
  combined.reduction_ms = reduction_ms;
  combined.elapsed_ms = simulation_ms + reduction_ms;
  combined.estimator_samples = simulation.num_paths;
  combined.physical_paths =
      simulation.num_paths *
      (simulation.variance_reduction == VarianceReduction::kAntithetic ? 2 : 1);
  combined.backend = "cuda-rqmc-halton";
  return combined;
}

Estimate price_cuda_multi(const OptionParams& option,
                          const SimulationParams& simulation) {
  // 各卡派生不同 seed；价格按样本数加权，方差按分区独立假设合并。尚未多卡实测。
  int available_devices = 0;
  PRICING_CUDA_CHECK(cudaGetDeviceCount(&available_devices));
  if (simulation.num_gpus <= 0 || simulation.num_gpus > available_devices) {
    throw std::invalid_argument("requested num_gpus exceeds visible CUDA devices");
  }
  if (simulation.num_gpus == 1) return price_cuda(option, simulation);
  if (simulation.num_paths <
      static_cast<std::uint64_t>(2 * simulation.num_gpus)) {
    throw std::invalid_argument("multi-GPU partition needs two paths per device");
  }

  const auto start = std::chrono::steady_clock::now();
  std::vector<std::future<Estimate>> futures;
  futures.reserve(static_cast<std::size_t>(simulation.num_gpus));
  const std::uint64_t quotient = simulation.num_paths / simulation.num_gpus;
  const std::uint64_t remainder = simulation.num_paths % simulation.num_gpus;
  for (int device = 0; device < simulation.num_gpus; ++device) {
    SimulationParams partition = simulation;
    partition.num_gpus = 1;
    partition.num_paths = quotient +
                          (static_cast<std::uint64_t>(device) < remainder ? 1 : 0);
    if (partition.rng == "halton" &&
        partition.num_paths %
                static_cast<std::uint64_t>(partition.qmc_replications) !=
            0) {
      throw std::invalid_argument(
          "each multi-GPU Halton partition must divide qmc_replications");
    }
    std::uint64_t state =
        simulation.seed ^
        (static_cast<std::uint64_t>(device + 1) * 0xd1b54a32d192ed03ULL);
    state += 0x9e3779b97f4a7c15ULL;
    state = (state ^ (state >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    state = (state ^ (state >> 27U)) * 0x94d049bb133111ebULL;
    partition.seed = state ^ (state >> 31U);
    futures.emplace_back(std::async(
        std::launch::async, [option, partition, device]() {
          PRICING_CUDA_CHECK(cudaSetDevice(device));
          return price_cuda(option, partition);
        }));
  }

  std::vector<Estimate> partitions;
  partitions.reserve(futures.size());
  for (auto& future : futures) partitions.push_back(future.get());
  const double total_samples = static_cast<double>(simulation.num_paths);
  Estimate combined;
  double variance_of_mean = 0.0;
  for (const auto& partition : partitions) {
    const double weight =
        static_cast<double>(partition.estimator_samples) / total_samples;
    combined.price += weight * partition.price;
    variance_of_mean += weight * weight * partition.standard_error *
                        partition.standard_error;
    combined.simulation_ms =
        std::max(combined.simulation_ms, partition.simulation_ms);
    combined.reduction_ms =
        std::max(combined.reduction_ms, partition.reduction_ms);
    combined.physical_paths += partition.physical_paths;
  }
  combined.standard_error = std::sqrt(variance_of_mean);
  combined.confidence_low =
      combined.price - kConfidenceZ * combined.standard_error;
  combined.confidence_high =
      combined.price + kConfidenceZ * combined.standard_error;
  combined.elapsed_ms = std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - start)
                            .count();
  combined.estimator_samples = simulation.num_paths;
  combined.backend = "cuda-multi-" + std::to_string(simulation.num_gpus);
  return combined;
}

GreekEstimates estimate_greeks_cuda(const OptionParams& option,
                                    const SimulationParams& simulation) {
  // 沿用 CPU 的中心差分步长；各次估值复用 seed。
  const auto start = std::chrono::steady_clock::now();
  const auto price = [&](const OptionParams& current) {
    return simulation.num_gpus > 1 ? price_cuda_multi(current, simulation)
                                   : price_cuda(current, simulation);
  };
  const bool discontinuous_payoff =
      option.type == OptionType::kBarrierCall ||
      option.type == OptionType::kAmericanCall ||
      option.type == OptionType::kAmericanPut;
  const double spot_bump =
      option.spot *
      std::max(simulation.spot_bump_relative,
               discontinuous_payoff ? 1e-2 : simulation.spot_bump_relative);
  if (!(spot_bump > 0.0) || option.spot <= spot_bump) {
    throw std::invalid_argument("invalid CUDA spot bump for Greeks");
  }
  const Estimate base = price(option);
  OptionParams spot_up = option;
  OptionParams spot_down = option;
  spot_up.spot += spot_bump;
  spot_down.spot -= spot_bump;
  const Estimate up = price(spot_up);
  const Estimate down = price(spot_down);

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
  const Estimate v_up = price(vol_up);
  const Estimate v_down = price(vol_down);

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
