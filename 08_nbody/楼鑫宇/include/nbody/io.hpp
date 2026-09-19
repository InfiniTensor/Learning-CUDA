#pragma once

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include "nbody/config.hpp"
#include "nbody/cpu_solver.hpp"
#include "nbody/types.hpp"

namespace nbody {

std::vector<Particle> read_particles(const std::filesystem::path& path);
void write_run_outputs(const std::filesystem::path& output_dir,
                       const std::filesystem::path& input_path,
                       const Config& config,
                       const CpuRunResult& result,
                       double pre_output_wall_ms,
                       const std::string& backend = "cpu",
                       const std::string& precision = "fp64",
                       std::optional<double> theta = std::nullopt);

}  // namespace nbody
