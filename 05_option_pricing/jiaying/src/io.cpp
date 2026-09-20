#include "engine.hpp"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>

namespace pricing {
namespace {
using Config = std::map<std::string, std::string>;
std::string trim(std::string text) {
  const auto first = text.find_first_not_of(" \r\n\t");
  if (first == std::string::npos)
    return "";
  return text.substr(first, text.find_last_not_of(" \r\n\t") - first + 1);
}
Config read_config(const std::string &path,
                   const std::set<std::string> &allowed) {
  std::ifstream in(path);
  if (!in)
    throw std::runtime_error("Cannot read " + path);
  Config values;
  std::string line;
  int number = 0;
  while (std::getline(in, line)) {
    ++number;
    if (number == 1 && line.substr(0, 3) == "\xef\xbb\xbf")
      line.erase(0, 3);
    line = trim(line.substr(0, line.find('#')));
    if (line.empty())
      continue;
    const auto equal = line.find('=');
    if (equal == std::string::npos)
      throw std::runtime_error(path + ": expected key=value at line " +
                               std::to_string(number));
    const auto key = trim(line.substr(0, equal));
    auto value = trim(line.substr(equal + 1));
    if (value.size() >= 2 && ((value.front() == '"' && value.back() == '"') ||
                              (value.front() == '\'' && value.back() == '\'')))
      value = value.substr(1, value.size() - 2);
    if (!allowed.count(key))
      throw std::runtime_error("Unknown parameter: " + key);
    if (value.empty() || !values.emplace(key, value).second)
      throw std::runtime_error("Empty or duplicate parameter: " + key);
  }
  return values;
}
double real(const Config &c, const std::string &key, double fallback) {
  const auto it = c.find(key);
  if (it == c.end())
    return fallback;
  size_t used = 0;
  const double v = std::stod(it->second, &used);
  if (used != it->second.size() || !std::isfinite(v))
    throw std::runtime_error("Invalid number: " + key);
  return v;
}
uint64_t integer(const Config &c, const std::string &key, uint64_t fallback) {
  const auto it = c.find(key);
  if (it == c.end())
    return fallback;
  if (it->second.find_first_not_of("0123456789") != std::string::npos)
    throw std::runtime_error("Invalid unsigned integer: " + key);
  return std::stoull(it->second);
}
std::string word(const Config &c, const std::string &key,
                 const std::string &fallback) {
  const auto it = c.find(key);
  return it == c.end() ? fallback : it->second;
}
double cdf(double x) { return 0.5 * std::erfc(-x / std::sqrt(2.0)); }
std::string json_string(const std::string &value) {
  std::ostringstream s;
  s << '"';
  for (char c : value) {
    if (c == '"' || c == '\\')
      s << '\\';
    s << c;
  }
  s << '"';
  return s.str();
}
void output_estimate(std::ostream &out, const Estimate &e, const Run &r,
                     bool vega) {
  out << "{\n    \"price\": " << e.price
      << ",\n    \"standard_error\": " << e.standard_error
      << ",\n    \"ci95\": [" << e.low << ", " << e.high
      << "],\n    \"reference_price\": ";
  if (e.has_reference)
    out << e.reference;
  else
    out << "null";
  out << ",\n    \"absolute_error\": ";
  if (e.has_reference)
    out << e.absolute_error;
  else
    out << "null";
  out << ",\n    \"independent_samples\": " << r.samples.n
      << ",\n    \"control_beta\": " << e.beta
      << ",\n    \"raw_price\": " << r.samples.mean[0]
      << ",\n    \"raw_to_adjusted_variance_ratio\": " << e.variance_ratio
      << ",\n    \"delta\": " << e.greek[0]
      << ",\n    \"delta_standard_error\": " << e.greek_se[0]
      << ",\n    \"gamma\": " << e.greek[1]
      << ",\n    \"gamma_standard_error\": " << e.greek_se[1]
      << ",\n    \"vega\": ";
  if (vega)
    out << e.greek[2];
  else
    out << "null";
  out << ",\n    \"vega_standard_error\": ";
  if (vega)
    out << e.greek_se[2];
  else
    out << "null";
  out << "\n  }";
}
void output_run(std::ostream &out, const Run &r, const Simulation &s) {
  const uint64_t work =
      s.paths + (s.reduction == Reduction::ControlVariate ? s.pilot_paths : 0);
  out << "{\"device\": " << json_string(r.device)
      << ", \"total_ms\": " << r.total_ms
      << ", \"context_setup_ms\": " << r.setup_ms
      << ", \"simulation_ms\": " << r.simulation_ms
      << ", \"reduction_ms\": " << r.reduction_ms
      << ", \"paths_per_second\": " << s.paths * 1000.0 / r.total_ms
      << ", \"physical_paths_including_pilot\": " << work
      << ", \"block_size\": " << r.block_size << ", \"blocks\": " << r.blocks
      << ", \"workspace_bytes\": " << r.workspace_bytes << "}";
}
} // namespace

Option read_option(const std::string &path) {
  const auto c =
      read_config(path, {"option_type", "spot", "strike", "risk_free_rate",
                         "volatility", "maturity", "barrier"});
  Option o;
  const auto kind = word(c, "option_type", "european_call");
  if (kind == "european_call")
    o.kind = Kind::EuropeanCall;
  else if (kind == "european_put")
    o.kind = Kind::EuropeanPut;
  else if (kind == "asian_call")
    o.kind = Kind::AsianCall;
  else if (kind == "barrier_call")
    o.kind = Kind::BarrierCall;
  else
    throw std::runtime_error("Unsupported option_type: " + kind);
  o.spot = real(c, "spot", o.spot);
  o.strike = real(c, "strike", o.strike);
  o.rate = real(c, "risk_free_rate", o.rate);
  o.volatility = real(c, "volatility", o.volatility);
  o.maturity = real(c, "maturity", o.maturity);
  o.barrier = real(c, "barrier", o.barrier);
  if (o.spot <= 0 || o.strike <= 0 || o.volatility < 0 || o.maturity < 0 ||
      o.barrier <= 0)
    throw std::runtime_error("spot/strike/barrier must be positive; "
                             "volatility/maturity must be nonnegative");
  for (double v :
       {o.spot, o.strike, o.rate, o.volatility, o.maturity, o.barrier})
    if (!std::isfinite(static_cast<float>(v)) ||
        (v != 0 && static_cast<float>(v) == 0))
      throw std::runtime_error("Parameter outside supported FP32 path range");
  return o;
}
Simulation read_simulation(const std::string &path) {
  const auto c = read_config(path, {"num_paths", "num_steps", "seed", "rng",
                                    "variance_reduction", "pilot_paths",
                                    "block_size", "blocks", "spot_bump"});
  Simulation s;
  s.paths = integer(c, "num_paths", s.paths);
  s.seed = integer(c, "seed", s.seed);
  s.pilot_paths = integer(c, "pilot_paths", s.pilot_paths);
  const uint64_t steps = integer(c, "num_steps", s.steps);
  const uint64_t block = integer(c, "block_size", s.block_size);
  const uint64_t blocks = integer(c, "blocks", s.blocks);
  if (s.paths < 4 || s.paths >= (uint64_t(1) << 63) || s.pilot_paths < 4 ||
      s.pilot_paths >= (uint64_t(1) << 63) || !steps || steps > 1000000 ||
      block < 32 || block > 512 || block % 32 || blocks > 65535)
    throw std::runtime_error(
        "Invalid simulation size (paths >=4, steps 1..1000000, block_size "
        "multiple of 32 in [32,512], blocks 0..65535)");
  s.steps = static_cast<int>(steps);
  s.block_size = static_cast<int>(block);
  s.blocks = static_cast<int>(blocks);
  const auto rng = word(c, "rng", "curand");
  if (rng != "curand" && rng != "custom" && rng != "philox")
    throw std::runtime_error("rng must be curand, custom or philox");
  s.curand = rng == "curand";
  const auto reduction = word(c, "variance_reduction", "none");
  if (reduction == "none")
    s.reduction = Reduction::None;
  else if (reduction == "antithetic")
    s.reduction = Reduction::Antithetic;
  else if (reduction == "control_variate")
    s.reduction = Reduction::ControlVariate;
  else
    throw std::runtime_error("Unknown variance_reduction");
  if (s.reduction == Reduction::Antithetic && s.paths % 2)
    throw std::runtime_error("antithetic requires an even num_paths");
  s.spot_bump = real(c, "spot_bump", s.spot_bump);
  if (s.spot_bump < 0.0001 || s.spot_bump > 0.1)
    throw std::runtime_error("spot_bump must be in [0.0001, 0.1]");
  return s;
}
std::string option_name(Kind k) {
  switch (k) {
  case Kind::EuropeanCall:
    return "european_call";
  case Kind::EuropeanPut:
    return "european_put";
  case Kind::AsianCall:
    return "asian_call";
  case Kind::BarrierCall:
    return "barrier_call";
  }
  throw std::runtime_error("Invalid option kind");
}
std::string reduction_name(Reduction r) {
  switch (r) {
  case Reduction::None:
    return "none";
  case Reduction::Antithetic:
    return "antithetic";
  case Reduction::ControlVariate:
    return "control_variate";
  }
  throw std::runtime_error("Invalid reduction");
}
double black_scholes(const Option &o) {
  const double sign = o.kind == Kind::EuropeanPut ? -1 : 1;
  if (o.maturity == 0)
    return std::max(sign * (o.spot - o.strike), 0.0);
  const double discounted_strike = o.strike * std::exp(-o.rate * o.maturity);
  if (o.volatility == 0)
    return std::max(sign * (o.spot - discounted_strike), 0.0);
  const double st = o.volatility * std::sqrt(o.maturity);
  const double d1 =
      (std::log(o.spot / o.strike) +
       (o.rate + 0.5 * o.volatility * o.volatility) * o.maturity) /
      st;
  const double d2 = d1 - st;
  return sign * (o.spot * cdf(sign * d1) - discounted_strike * cdf(sign * d2));
}
double control_mean(const Option &o, const Simulation &s) {
  if (o.kind != Kind::AsianCall)
    return o.spot;
  // Discrete geometric mean at t_i=i*T/m, excludes S_0.
  const double m = s.steps;
  const double mean_time = o.maturity * (m + 1) / (2 * m);
  const double mean_log =
      std::log(o.spot) +
      (o.rate - 0.5 * o.volatility * o.volatility) * mean_time;
  const double variance = o.volatility * o.volatility * o.maturity * (m + 1) *
                          (2 * m + 1) / (6 * m * m);
  const double discount = std::exp(-o.rate * o.maturity);
  if (variance == 0)
    return discount * std::max(std::exp(mean_log) - o.strike, 0.0);
  const double root = std::sqrt(variance);
  const double d2 = (mean_log - std::log(o.strike)) / root;
  return discount * (std::exp(mean_log + variance / 2) * cdf(d2 + root) -
                     o.strike * cdf(d2));
}
Estimate estimate(const Option &o, const Simulation &s, const Run &run) {
  const auto &m = run.samples;
  if (m.n < 2)
    throw std::runtime_error("Not enough independent observations");
  for (int k = 0; k < 5; ++k)
    if (!std::isfinite(m.mean[k]) || !std::isfinite(m.m2[k]))
      throw std::runtime_error(
          "Non-finite simulation: reduce parameter magnitude");
  Estimate e;
  if (s.reduction == Reduction::ControlVariate) {
    if (!std::isfinite(run.pilot.cross) || !std::isfinite(run.pilot.m2[1]))
      throw std::runtime_error("Non-finite control pilot");
    e.beta = run.pilot.m2[1] > 1e-20 ? run.pilot.cross / run.pilot.m2[1] : 0;
  }
  e.price = m.mean[0] - e.beta * (m.mean[1] - control_mean(o, s));
  const double residual =
      std::max(0.0, m.m2[0] + e.beta * e.beta * m.m2[1] - 2 * e.beta * m.cross);
  const double denominator =
      static_cast<double>(m.n) * static_cast<double>(m.n - 1);
  e.standard_error = std::sqrt(residual / denominator);
  e.low = e.price - 1.959963984540054 * e.standard_error;
  e.high = e.price + 1.959963984540054 * e.standard_error;
  e.variance_ratio = residual > 0 ? m.m2[0] / residual : 1;
  e.has_reference = o.kind == Kind::EuropeanCall || o.kind == Kind::EuropeanPut;
  if (e.has_reference) {
    e.reference = black_scholes(o);
    e.absolute_error = std::abs(e.price - e.reference);
  }
  for (int k = 0; k < 3; ++k) {
    e.greek[k] = m.mean[k + 2];
    e.greek_se[k] = std::sqrt(std::max(0.0, m.m2[k + 2]) / denominator);
  }
  for (double v : {e.price, e.standard_error, e.low, e.high, e.beta,
                   e.reference, e.variance_ratio})
    if (!std::isfinite(v))
      throw std::runtime_error("Non-finite estimate");
  return e;
}
void write_outputs(const std::string &prefix, const Option &o,
                   const Simulation &s, const Run *cpu, const Run *gpu,
                   double io_ms, double process_ms) {
  const auto parent = std::filesystem::path(prefix).parent_path();
  if (!parent.empty())
    std::filesystem::create_directories(parent);
  std::ofstream result(prefix + ".json"),
      performance(prefix + "_performance.json");
  if (!result || !performance)
    throw std::runtime_error("Cannot create output files for " + prefix);
  result << std::setprecision(17);
  result << "{\n  \"option_type\": " << json_string(option_name(o.kind))
         << ",\n  \"method\": \"exact_gbm_monte_carlo\",\n  "
            "\"variance_reduction\": "
         << json_string(reduction_name(s.reduction)) << ",\n  \"rng\": "
         << json_string(s.curand ? "curand_philox4x32_10"
                                 : "custom_philox4x32_10")
         << ",\n  \"normal_transform\": "
            "\"shared_box_muller_fp32_23bit_uniform\",\n  \"num_paths\": "
         << s.paths << ",\n  \"num_steps\": " << s.steps
         << ",\n  \"effective_steps\": "
         << ((o.kind == Kind::EuropeanCall || o.kind == Kind::EuropeanPut)
                 ? 1
                 : s.steps)
         << ",\n  \"seed\": " << s.seed << ",\n  \"pilot_paths\": "
         << (s.reduction == Reduction::ControlVariate ? s.pilot_paths : 0)
         << ",\n  \"spot\": " << o.spot << ",\n  \"strike\": " << o.strike
         << ",\n  \"risk_free_rate\": " << o.rate
         << ",\n  \"volatility\": " << o.volatility
         << ",\n  \"maturity\": " << o.maturity
         << ",\n  \"barrier\": " << o.barrier
         << ",\n  \"spot_bump_relative\": " << s.spot_bump
         << ",\n  \"monitoring\": \"asian: i*T/m (i=1..m); barrier: S0 plus "
            "i*T/m, upper knock-out, no rebate\""
         << ",\n  \"confidence_method\": \"asymptotic normal 95%, sampling "
            "error only\""
         << ",\n  \"greek_method\": \"pathwise delta/vega (european/asian), "
            "CRN central gamma, CRN barrier delta; raw estimates\"";
  if (cpu) {
    result << ",\n  \"cpu\": ";
    output_estimate(result, estimate(o, s, *cpu), *cpu,
                    o.kind != Kind::BarrierCall);
  }
  if (gpu) {
    result << ",\n  \"gpu\": ";
    output_estimate(result, estimate(o, s, *gpu), *gpu,
                    o.kind != Kind::BarrierCall);
  }
  result << "\n}\n";
  performance << std::setprecision(17)
              << "{\n  \"timing_scope\": \"total_ms: allocation, RNG, pilot, "
                 "paths, all reductions, D2H, cleanup; excludes one-time "
                 "context setup and file IO. simulation_ms includes "
                 "intra-block reduction.\",\n  \"input_io_ms\": "
              << io_ms << ",\n  \"process_before_output_ms\": " << process_ms;
  if (cpu) {
    performance << ",\n  \"cpu\": ";
    output_run(performance, *cpu, s);
  }
  if (gpu) {
    performance << ",\n  \"gpu\": ";
    output_run(performance, *gpu, s);
  }
  if (cpu && gpu)
    performance << ",\n  \"speedup\": " << cpu->total_ms / gpu->total_ms
                << ",\n  \"cold_speedup\": "
                << cpu->total_ms / (gpu->total_ms + gpu->setup_ms)
                << ",\n  \"cpu_gpu_price_difference\": "
                << std::abs(estimate(o, s, *cpu).price -
                            estimate(o, s, *gpu).price);
  performance << "\n}\n";
  result.close();
  performance.close();
  if (!result || !performance)
    throw std::runtime_error("Failed writing output files");
}
} // namespace pricing
