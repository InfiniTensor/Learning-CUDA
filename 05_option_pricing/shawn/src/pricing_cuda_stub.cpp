#include "pricing/pricing.hpp"

#include <stdexcept>

namespace pricing {

bool cuda_backend_available() { return false; }

Estimate price_cuda(const OptionParams&, const SimulationParams&) {
  throw std::runtime_error(
      "CUDA backend was not built; configure on a machine with nvcc and the CUDA Toolkit");
}

Estimate price_cuda_multi(const OptionParams&, const SimulationParams&) {
  throw std::runtime_error(
      "multi-GPU backend was not built; configure on a machine with nvcc");
}

GreekEstimates estimate_greeks_cuda(const OptionParams&,
                                    const SimulationParams&) {
  throw std::runtime_error(
      "CUDA Greeks backend was not built; configure on a machine with nvcc");
}

}  // namespace pricing
