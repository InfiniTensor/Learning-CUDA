#pragma once
#include "random.hpp"
#include <cmath>

namespace pricing {
struct Parameters {
  Kind kind;
  int steps;
  float spot, strike, sigma, barrier, maturity, dt, root_dt, drift, bump;
  double discount;
};
inline Parameters parameters(const Option &o, const Simulation &s) {
  const int steps =
      (o.kind == Kind::EuropeanCall || o.kind == Kind::EuropeanPut) ? 1
                                                                    : s.steps;
  const double dt = o.maturity / steps;
  return {o.kind,
          steps,
          float(o.spot),
          float(o.strike),
          float(o.volatility),
          float(o.barrier),
          float(o.maturity),
          float(dt),
          float(std::sqrt(dt)),
          float(o.rate - 0.5 * o.volatility * o.volatility),
          float(s.spot_bump),
          std::exp(-o.rate * o.maturity)};
}
OP_HD inline double payoff(const Parameters &p, float value, float maximum,
                           float scale) {
  if (p.kind == Kind::BarrierCall && maximum * scale >= p.barrier)
    return 0;
  const float diff = p.kind == Kind::EuropeanPut ? p.strike - value * scale
                                                 : value * scale - p.strike;
  return p.discount * static_cast<double>(fmaxf(diff, 0.0f));
}
struct Path {
  float brownian = 0, sum = 0, log_sum = 0, derivative_sum = 0, terminal,
        maximum;
  OP_HD explicit Path(float spot) : terminal(spot), maximum(spot) {}
  OP_HD void step(const Parameters &p, int i, float z, bool control) {
    const float t = (i + 1) * p.dt;
    brownian += p.root_dt * z;
    const float log_ratio = p.drift * t + p.sigma * brownian;
    terminal = p.spot * expf(log_ratio);
    maximum = fmaxf(maximum, terminal);
    if (p.kind == Kind::AsianCall) {
      sum += terminal;
      derivative_sum += terminal * (brownian - p.sigma * t);
      if (control)
        log_sum += log_ratio;
    }
  }
  OP_HD void finish(const Parameters &p, bool control, double *v) const {
    const float value = p.kind == Kind::AsianCall ? sum / p.steps : terminal;
    v[0] = payoff(p, value, maximum, 1.0f);
    v[1] = 0;
    if (control) {
      v[1] = p.kind == Kind::AsianCall
                 ? p.discount *
                       fmaxf(p.spot * expf(log_sum / p.steps) - p.strike, 0.0f)
                 : p.discount * terminal;
    }
    const double h = static_cast<double>(p.spot) * p.bump;
    const double plus = payoff(p, value, maximum, 1.0f + p.bump);
    const double minus = payoff(p, value, maximum, 1.0f - p.bump);
    v[3] = (plus - 2 * v[0] + minus) / (h * h);
    if (p.kind == Kind::BarrierCall) {
      v[2] = (plus - minus) / (2 * h);
      v[4] = 0; // Barrier vega is deliberately not estimated by a pathwise
                // formula.
    } else {
      const double sign = p.kind == Kind::EuropeanPut ? -1 : 1;
      const bool itm = sign * (value - p.strike) > 0;
      v[2] = itm ? sign * p.discount * value / p.spot : 0;
      const float derivative =
          p.kind == Kind::AsianCall
              ? derivative_sum / p.steps
              : terminal * (brownian - p.sigma * p.maturity);
      v[4] = itm ? sign * p.discount * derivative : 0;
    }
  }
};

// One independent observation is one path, or the mean of a (+Z,-Z) pair.
// Never count the two antithetic legs as independent in a confidence interval.
template <class Bits, bool Antithetic, bool Control>
OP_HD inline void observation(const Parameters &p, uint64_t seed, uint64_t id,
                              double *v) {
  NormalGenerator<Bits> rng(seed, id);
  Path a(p.spot), b(p.spot);
  for (int i = 0; i < p.steps; ++i) {
    const float z = (p.sigma == 0 || p.maturity == 0) ? 0.0f : rng.next();
    a.step(p, i, z, Control);
    if (Antithetic)
      b.step(p, i, -z, Control);
  }
  a.finish(p, Control, v);
  if (Antithetic) {
    double other[5];
    b.finish(p, Control, other);
    for (int k = 0; k < 5; ++k)
      v[k] = (v[k] + other[k]) * 0.5;
  }
}
} // namespace pricing
