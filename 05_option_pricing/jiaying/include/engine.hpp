#pragma once
#include <array>
#include <cstdint>
#include <string>

#ifdef __CUDACC__
#define OP_HD __host__ __device__
#else
#define OP_HD
#endif

namespace pricing {
enum class Kind { EuropeanCall, EuropeanPut, AsianCall, BarrierCall };
enum class Reduction { None, Antithetic, ControlVariate };
struct Option {
  Kind kind = Kind::EuropeanCall;
  double spot = 100, strike = 100, rate = 0.03, volatility = 0.2, maturity = 1;
  double barrier = 130;
};
struct Simulation {
  uint64_t paths = 1000000, seed = 1234, pilot_paths = 8192;
  int steps = 256, block_size = 256, blocks = 0;
  bool curand = true;
  Reduction reduction = Reduction::None;
  double spot_bump = 0.01;
};

// Welford/Chan statistics prevent catastrophic cancellation near zero variance.
// Fields: price, control, delta, gamma, vega. Covariance is price/control only.
struct Moments {
  uint64_t n = 0;
  double mean[5] = {}, m2[5] = {}, cross = 0;
  OP_HD void add(const double *v) {
    ++n;
    double d[5];
    for (int k = 0; k < 5; ++k) {
      d[k] = v[k] - mean[k];
      mean[k] += d[k] / static_cast<double>(n);
      m2[k] += d[k] * (v[k] - mean[k]);
    }
    cross += d[0] * (v[1] - mean[1]);
  }
  OP_HD void merge(const Moments &b) {
    if (!b.n)
      return;
    if (!n) {
      *this = b;
      return;
    }
    const uint64_t total = n + b.n;
    const double w = static_cast<double>(b.n) / static_cast<double>(total);
    const double factor = static_cast<double>(n) * w;
    const double dx = b.mean[0] - mean[0], dy = b.mean[1] - mean[1];
    for (int k = 0; k < 5; ++k) {
      const double d = b.mean[k] - mean[k];
      mean[k] += d * w;
      m2[k] += b.m2[k] + d * d * factor;
    }
    cross += b.cross + dx * dy * factor;
    n = total;
  }
};
struct Run {
  Moments samples, pilot;
  double total_ms = 0, simulation_ms = 0, reduction_ms = 0, setup_ms = 0;
  int blocks = 0, block_size = 1;
  uint64_t workspace_bytes = 0;
  std::string device;
};
struct Estimate {
  double price = 0, standard_error = 0, low = 0, high = 0, beta = 0;
  double reference = 0, absolute_error = 0, variance_ratio = 1;
  bool has_reference = false;
  std::array<double, 3> greek{}, greek_se{};
};
Option read_option(const std::string &path);
Simulation read_simulation(const std::string &path);
std::string option_name(Kind k);
std::string reduction_name(Reduction r);
double black_scholes(const Option &o);
double control_mean(const Option &o, const Simulation &s);
Estimate estimate(const Option &o, const Simulation &s, const Run &run);
Run run_cpu(const Option &o, const Simulation &s);
Run run_gpu(const Option &o, const Simulation &s);
void write_outputs(const std::string &prefix, const Option &o,
                   const Simulation &s, const Run *cpu, const Run *gpu,
                   double io_ms, double process_ms);
} // namespace pricing
