#pragma once

#include <vector>

#include "nbody/config.hpp"
#include "nbody/types.hpp"

namespace nbody {

struct RunOptions {
    bool record_enabled = true;
    bool diagnostics_enabled = true;
};

struct CpuRunResult {
    std::vector<Particle> particles;
    std::vector<int> recorded_steps;
    std::vector<float> trajectory;
    Diagnostics initial_diagnostics;
    Diagnostics final_diagnostics;
    double force_ms = 0.0;
    double simulation_ms = 0.0;
    int force_evaluations = 0;
    bool record_enabled = true;
    bool diagnostics_enabled = true;
};

std::vector<Vec3> compute_accelerations(const std::vector<Particle>& particles,
                                        double gravitational_constant,
                                        double softening);
Diagnostics compute_diagnostics(const std::vector<Particle>& particles,
                                double gravitational_constant,
                                double softening);
std::vector<int> make_recorded_steps(int num_steps, int record_interval);
CpuRunResult simulate_cpu(const std::vector<Particle>& initial_particles,
                          const Config& config,
                          const RunOptions& options = {});

}  // namespace nbody
