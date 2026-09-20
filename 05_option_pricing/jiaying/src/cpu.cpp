#include "model.hpp"
#include <chrono>

namespace pricing {
template <bool Anti, bool Control>
Moments sample_cpu(const Parameters &p, uint64_t seed, uint64_t n,
                   uint64_t offset) {
  Moments result;
  for (uint64_t i = 0; i < n; ++i) {
    double v[5];
    observation<CustomBits, Anti, Control>(p, seed, i + offset, v);
    result.add(v);
  }
  return result;
}
Run run_cpu(const Option &o, const Simulation &s) {
  Run r;
  r.device = "single CPU thread / Philox4x32-10";
  const auto start = std::chrono::steady_clock::now();
  const auto p = parameters(o, s);
  if (s.reduction == Reduction::ControlVariate) {
    r.pilot =
        sample_cpu<false, true>(p, s.seed, s.pilot_paths, uint64_t(1) << 63);
    r.samples = sample_cpu<false, true>(p, s.seed, s.paths, 0);
  } else if (s.reduction == Reduction::Antithetic) {
    r.samples = sample_cpu<true, false>(p, s.seed, s.paths / 2, 0);
  } else {
    r.samples = sample_cpu<false, false>(p, s.seed, s.paths, 0);
  }
  r.total_ms = std::chrono::duration<double, std::milli>(
                   std::chrono::steady_clock::now() - start)
                   .count();
  r.simulation_ms = r.total_ms;
  return r;
}
} // namespace pricing
