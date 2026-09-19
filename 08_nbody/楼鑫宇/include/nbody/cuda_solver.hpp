#pragma once

#include <vector>

#include "nbody/config.hpp"
#include "nbody/cpu_solver.hpp"
#include "nbody/types.hpp"

namespace nbody {

CpuRunResult simulate_cuda_naive(const std::vector<Particle>& initial_particles,
                                 const Config& config,
                                 int block_size = 128,
                                 const RunOptions& options = {});
CpuRunResult simulate_cuda_tiled(const std::vector<Particle>& initial_particles,
                                 const Config& config,
                                 int block_size = 128,
                                 const RunOptions& options = {});
CpuRunResult simulate_cuda_mixed(const std::vector<Particle>& initial_particles,
                                 const Config& config,
                                 int block_size = 128,
                                 const RunOptions& options = {});
CpuRunResult simulate_cuda_bh(const std::vector<Particle>& initial_particles,
                              const Config& config,
                              int block_size = 128,
                              double theta = 0.5,
                              const RunOptions& options = {});

}  // namespace nbody
