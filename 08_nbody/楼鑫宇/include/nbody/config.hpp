#pragma once

#include <filesystem>
#include <string>

namespace nbody {

enum class Integrator { Euler, Leapfrog };

struct Config {
    double dt = 0.0;
    int num_steps = 0;
    int record_interval = 0;
    double gravitational_constant = 0.0;
    double softening = 0.0;
    Integrator integrator = Integrator::Leapfrog;
};

Config read_config(const std::filesystem::path& path);
std::string to_string(Integrator integrator);

}  // namespace nbody

