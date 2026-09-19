#pragma once

#include <vector>

#include "nbody/config.hpp"
#include "nbody/cpu_solver.hpp"
#include "nbody/types.hpp"

namespace nbody {

// FP32 all-pairs gravity on a MetaX MACA device. The result layout and
// integration semantics match the CUDA backends.
CpuRunResult simulate_maca_naive(const std::vector<Particle>& initial_particles,
                                 const Config& config,
                                 int block_size = 128,
                                 const RunOptions& options = {});

CpuRunResult simulate_maca_tiled(const std::vector<Particle>& initial_particles,
                                 const Config& config,
                                 int block_size = 128,
                                 const RunOptions& options = {});

}  // namespace nbody
