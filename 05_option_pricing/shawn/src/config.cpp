#include "pricing/pricing.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace pricing {
namespace {

std::string trim(std::string value) {
  const auto is_space = [](unsigned char c) { return std::isspace(c) != 0; };
  value.erase(value.begin(),
              std::find_if(value.begin(), value.end(),
                           [&](unsigned char c) { return !is_space(c); }));
  value.erase(std::find_if(value.rbegin(), value.rend(),
                           [&](unsigned char c) { return !is_space(c); })
                  .base(),
              value.end());
  return value;
}

std::string unquote(std::string value) {
  value = trim(std::move(value));
  if (value.size() >= 2 &&
      ((value.front() == '"' && value.back() == '"') ||
       (value.front() == '\'' && value.back() == '\''))) {
    return value.substr(1, value.size() - 2);
  }
  return value;
}

std::unordered_map<std::string, std::string> load_key_values(
    const std::string& path) {
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("cannot open configuration file: " + path);
  }

  std::unordered_map<std::string, std::string> values;
  std::string line;
  std::size_t line_number = 0;
  while (std::getline(input, line)) {
    ++line_number;
    const auto comment = line.find('#');
    if (comment != std::string::npos) {
      line.resize(comment);
    }
    line = trim(std::move(line));
    if (line.empty()) {
      continue;
    }
    const auto equals = line.find('=');
    if (equals == std::string::npos) {
      throw std::runtime_error("expected key = value at " + path + ":" +
                               std::to_string(line_number));
    }
    const std::string key = trim(line.substr(0, equals));
    const std::string value = unquote(line.substr(equals + 1));
    if (key.empty() || value.empty()) {
      throw std::runtime_error("empty key or value at " + path + ":" +
                               std::to_string(line_number));
    }
    if (!values.emplace(key, value).second) {
      throw std::runtime_error("duplicate key '" + key + "' in " + path);
    }
  }
  return values;
}

const std::string& required(
    const std::unordered_map<std::string, std::string>& values,
    const std::string& key, const std::string& path) {
  const auto it = values.find(key);
  if (it == values.end()) {
    throw std::runtime_error("missing required key '" + key + "' in " + path);
  }
  return it->second;
}

double parse_double(const std::string& value, const std::string& key) {
  std::size_t consumed = 0;
  const double parsed = std::stod(value, &consumed);
  if (consumed != value.size()) {
    throw std::runtime_error("invalid floating-point value for '" + key + "'");
  }
  return parsed;
}

std::uint64_t parse_uint64(const std::string& value, const std::string& key) {
  std::size_t consumed = 0;
  const auto parsed = std::stoull(value, &consumed);
  if (consumed != value.size()) {
    throw std::runtime_error("invalid unsigned integer for '" + key + "'");
  }
  return parsed;
}

int parse_int(const std::string& value, const std::string& key) {
  std::size_t consumed = 0;
  const int parsed = std::stoi(value, &consumed);
  if (consumed != value.size()) {
    throw std::runtime_error("invalid integer for '" + key + "'");
  }
  return parsed;
}

bool parse_bool(const std::string& value, const std::string& key) {
  if (value == "true" || value == "1") return true;
  if (value == "false" || value == "0") return false;
  throw std::runtime_error("invalid boolean for '" + key + "'");
}

OptionType parse_option_type(const std::string& value) {
  if (value == "european_call") return OptionType::kEuropeanCall;
  if (value == "european_put") return OptionType::kEuropeanPut;
  if (value == "asian_call") return OptionType::kAsianCall;
  if (value == "barrier_call") return OptionType::kBarrierCall;
  if (value == "american_call") return OptionType::kAmericanCall;
  if (value == "american_put") return OptionType::kAmericanPut;
  throw std::runtime_error("unsupported option_type: " + value);
}

ModelType parse_model_type(const std::string& value) {
  if (value == "black_scholes" || value == "gbm") {
    return ModelType::kBlackScholes;
  }
  if (value == "heston") return ModelType::kHeston;
  if (value == "local_volatility" || value == "local_vol") {
    return ModelType::kLocalVolatility;
  }
  throw std::runtime_error("unsupported model: " + value);
}

VarianceReduction parse_variance_reduction(const std::string& value) {
  if (value == "none") return VarianceReduction::kNone;
  if (value == "antithetic") return VarianceReduction::kAntithetic;
  if (value == "control_variate") return VarianceReduction::kControlVariate;
  throw std::runtime_error("unsupported variance_reduction: " + value);
}

void validate(const OptionParams& option) {
  if (!(option.spot > 0.0) || !(option.strike > 0.0) ||
      !(option.maturity > 0.0) || option.volatility < 0.0) {
    throw std::runtime_error(
        "spot, strike and maturity must be positive; volatility must be non-negative");
  }
  if (option.type == OptionType::kBarrierCall &&
      !(option.barrier > option.spot)) {
    throw std::runtime_error(
        "barrier_call is defined as up-and-out and requires barrier > spot");
  }
  if (option.heston_initial_variance < 0.0 || option.heston_kappa < 0.0 ||
      option.heston_theta < 0.0 || option.heston_vol_of_vol < 0.0 ||
      option.heston_rho < -1.0 || option.heston_rho > 1.0) {
    throw std::runtime_error("invalid Heston parameters");
  }
  if (!std::isfinite(option.local_vol_beta)) {
    throw std::runtime_error("local_vol_beta must be finite");
  }
}

void validate(const SimulationParams& simulation) {
  if (simulation.num_paths < 2) {
    throw std::runtime_error("num_paths must be at least 2");
  }
  if (simulation.num_steps <= 0) {
    throw std::runtime_error("num_steps must be positive");
  }
  if (simulation.rng != "curand" && simulation.rng != "custom" &&
      simulation.rng != "halton") {
    throw std::runtime_error("rng must be 'curand', 'custom' or 'halton'");
  }
  if (simulation.qmc_replications < 2 || simulation.num_gpus <= 0 ||
      !(simulation.spot_bump_relative > 0.0) ||
      !(simulation.volatility_bump_absolute > 0.0)) {
    throw std::runtime_error("invalid QMC, GPU or Greek configuration");
  }
  if (simulation.rng == "halton" &&
      simulation.num_paths %
              static_cast<std::uint64_t>(simulation.qmc_replications) !=
          0) {
    throw std::runtime_error(
        "Halton num_paths must be divisible by qmc_replications");
  }
}

}  // namespace

OptionParams load_option_params(const std::string& path) {
  // 解析合约参数并检查取值范围。
  const auto values = load_key_values(path);
  OptionParams option;
  option.type = parse_option_type(required(values, "option_type", path));
  if (const auto it = values.find("model"); it != values.end()) {
    option.model = parse_model_type(it->second);
  }
  option.spot = parse_double(required(values, "spot", path), "spot");
  option.strike = parse_double(required(values, "strike", path), "strike");
  option.risk_free_rate =
      parse_double(required(values, "risk_free_rate", path), "risk_free_rate");
  option.volatility =
      parse_double(required(values, "volatility", path), "volatility");
  option.maturity =
      parse_double(required(values, "maturity", path), "maturity");
  if (const auto it = values.find("barrier"); it != values.end()) {
    option.barrier = parse_double(it->second, "barrier");
  }
  if (const auto it = values.find("heston_initial_variance");
      it != values.end()) {
    option.heston_initial_variance =
        parse_double(it->second, "heston_initial_variance");
  }
  if (const auto it = values.find("heston_kappa"); it != values.end()) {
    option.heston_kappa = parse_double(it->second, "heston_kappa");
  }
  if (const auto it = values.find("heston_theta"); it != values.end()) {
    option.heston_theta = parse_double(it->second, "heston_theta");
  }
  if (const auto it = values.find("heston_vol_of_vol"); it != values.end()) {
    option.heston_vol_of_vol =
        parse_double(it->second, "heston_vol_of_vol");
  }
  if (const auto it = values.find("heston_rho"); it != values.end()) {
    option.heston_rho = parse_double(it->second, "heston_rho");
  }
  if (const auto it = values.find("local_vol_beta"); it != values.end()) {
    option.local_vol_beta = parse_double(it->second, "local_vol_beta");
  }
  validate(option);
  return option;
}

SimulationParams load_simulation_params(const std::string& path) {
  // 解析并校验仿真参数。
  const auto values = load_key_values(path);
  SimulationParams simulation;
  simulation.num_paths =
      parse_uint64(required(values, "num_paths", path), "num_paths");
  simulation.num_steps =
      parse_int(required(values, "num_steps", path), "num_steps");
  simulation.seed = parse_uint64(required(values, "seed", path), "seed");
  simulation.rng = required(values, "rng", path);
  simulation.variance_reduction = parse_variance_reduction(
      required(values, "variance_reduction", path));
  if (const auto it = values.find("qmc_replications"); it != values.end()) {
    simulation.qmc_replications =
        parse_int(it->second, "qmc_replications");
  }
  if (const auto it = values.find("num_gpus"); it != values.end()) {
    simulation.num_gpus = parse_int(it->second, "num_gpus");
  }
  if (const auto it = values.find("compute_greeks"); it != values.end()) {
    simulation.compute_greeks = parse_bool(it->second, "compute_greeks");
  }
  if (const auto it = values.find("spot_bump_relative"); it != values.end()) {
    simulation.spot_bump_relative =
        parse_double(it->second, "spot_bump_relative");
  }
  if (const auto it = values.find("volatility_bump_absolute");
      it != values.end()) {
    simulation.volatility_bump_absolute =
        parse_double(it->second, "volatility_bump_absolute");
  }
  validate(simulation);
  return simulation;
}

std::string to_string(OptionType type) {
  switch (type) {
    case OptionType::kEuropeanCall:
      return "european_call";
    case OptionType::kEuropeanPut:
      return "european_put";
    case OptionType::kAsianCall:
      return "asian_call";
    case OptionType::kBarrierCall:
      return "barrier_call";
    case OptionType::kAmericanCall:
      return "american_call";
    case OptionType::kAmericanPut:
      return "american_put";
  }
  throw std::logic_error("unknown option type");
}

std::string to_string(ModelType model) {
  switch (model) {
    case ModelType::kBlackScholes:
      return "black_scholes";
    case ModelType::kHeston:
      return "heston";
    case ModelType::kLocalVolatility:
      return "local_volatility";
  }
  throw std::logic_error("unknown model");
}

std::string to_string(VarianceReduction mode) {
  switch (mode) {
    case VarianceReduction::kNone:
      return "none";
    case VarianceReduction::kAntithetic:
      return "antithetic";
    case VarianceReduction::kControlVariate:
      return "control_variate";
  }
  throw std::logic_error("unknown variance reduction mode");
}

}  // namespace pricing
