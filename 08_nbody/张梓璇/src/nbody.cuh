#pragma once

#include "io.h"

#include <string>
#include <vector>

struct GpuSimulationResult {
    std::vector<Particle> particles;
    PerformanceData performance;
};

GpuSimulationResult simulate_gpu(const std::vector<Particle>& initial_particles,
                                 const SimConfig& config,
                                 const std::string& trajectory_path);
