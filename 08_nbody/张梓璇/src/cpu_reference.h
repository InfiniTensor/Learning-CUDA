#pragma once

#include "io.h"

#include <vector>

void simulate_cpu(std::vector<Particle>& particles, const SimConfig& config);
void state_error(const std::vector<Particle>& reference,
                 const std::vector<Particle>& test,
                 double& max_position_error,
                 double& rms_position_error,
                 double& max_velocity_error,
                 double& rms_velocity_error);
