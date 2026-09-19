#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "nbody/cpu_solver.hpp"
#include "nbody/maca_solver.hpp"

namespace {

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

double component_error(const nbody::Vec3& lhs, const nbody::Vec3& rhs) {
    return std::max({std::abs(lhs.x - rhs.x), std::abs(lhs.y - rhs.y),
                     std::abs(lhs.z - rhs.z)});
}

void check_finite(const nbody::CpuRunResult& result) {
    for (const auto& particle : result.particles) {
        require(std::isfinite(particle.position.x) &&
                    std::isfinite(particle.position.y) &&
                    std::isfinite(particle.position.z) &&
                    std::isfinite(particle.velocity.x) &&
                    std::isfinite(particle.velocity.y) &&
                    std::isfinite(particle.velocity.z) &&
                    std::isfinite(particle.mass),
                "MACA 末态出现非有限粒子数据");
    }
    for (const float coordinate : result.trajectory) {
        require(std::isfinite(coordinate), "MACA 轨迹出现非有限数据");
    }
}

struct Comparison {
    double position = 0.0;
    double velocity = 0.0;
    double trajectory = 0.0;
};

Comparison compare(const nbody::CpuRunResult& cpu,
                   const nbody::CpuRunResult& maca) {
    require(cpu.particles.size() == maca.particles.size(),
            "CPU/MACA 粒子数不一致");
    require(cpu.record_enabled == maca.record_enabled &&
                cpu.diagnostics_enabled == maca.diagnostics_enabled,
            "MACA 运行选项元数据不一致");
    require(cpu.recorded_steps == maca.recorded_steps,
            "CPU/MACA 记录步不一致");
    require(cpu.trajectory.size() == maca.trajectory.size(),
            "CPU/MACA 轨迹大小不一致");
    require(cpu.force_evaluations == maca.force_evaluations,
            "CPU/MACA 力计算次数不一致");
    require(maca.trajectory.size() ==
                maca.particles.size() * maca.recorded_steps.size() * 3,
            "MACA 轨迹布局大小不正确");
    check_finite(maca);

    Comparison comparison;
    for (std::size_t i = 0; i < cpu.particles.size(); ++i) {
        comparison.position = std::max(
            comparison.position,
            component_error(cpu.particles[i].position,
                            maca.particles[i].position));
        comparison.velocity = std::max(
            comparison.velocity,
            component_error(cpu.particles[i].velocity,
                            maca.particles[i].velocity));
    }
    for (std::size_t i = 0; i < cpu.trajectory.size(); ++i) {
        comparison.trajectory = std::max(
            comparison.trajectory,
            std::abs(static_cast<double>(cpu.trajectory[i]) -
                     static_cast<double>(maca.trajectory[i])));
    }
    return comparison;
}

void check_tolerance(const Comparison& comparison,
                     const char* backend,
                     nbody::Integrator integrator,
                     int block_size) {
    if (!(comparison.position < 2.0e-5 &&
          comparison.velocity < 2.0e-5 &&
          comparison.trajectory < 2.0e-5)) {
        throw std::runtime_error(
            std::string(backend) + " 与 CPU FP64 偏差过大 (" +
            (integrator == nbody::Integrator::Euler ? "Euler" : "Leapfrog") +
            ", block=" + std::to_string(block_size) + ")");
    }
}

}  // namespace

int main() {
    try {
        // 129 is deliberately not a multiple of any tested thread block size.
        constexpr int particle_count = 129;
        std::vector<nbody::Particle> particles;
        particles.reserve(particle_count);
        for (int i = 0; i < particle_count; ++i) {
            const double value = static_cast<double>(i);
            particles.push_back({
                {0.35 * std::sin(0.73 * value) + 0.002 * value,
                 0.40 * std::cos(0.51 * value) - 0.001 * value,
                 0.30 * std::sin(0.37 * value + 0.2)},
                {0.01 * std::cos(0.19 * value),
                 0.01 * std::sin(0.23 * value),
                 0.005 * std::cos(0.31 * value)},
                0.5 + 0.01 * static_cast<double>(i % 11),
            });
        }

        nbody::Config config;
        config.dt = 1.0e-4;
        config.num_steps = 5;
        config.record_interval = 2;
        config.gravitational_constant = 1.0;
        config.softening = 0.05;

        for (const auto integrator : {nbody::Integrator::Euler,
                                      nbody::Integrator::Leapfrog}) {
            config.integrator = integrator;
            const auto cpu = nbody::simulate_cpu(particles, config);
            require(cpu.recorded_steps == std::vector<int>({0, 2, 4, 5}),
                    "测试所需的记录步与预期不一致");

            for (const int block_size : {64, 128, 256}) {
                const auto naive =
                    nbody::simulate_maca_naive(particles, config, block_size);
                check_tolerance(compare(cpu, naive), "maca-naive", integrator,
                                block_size);

                const auto tiled =
                    nbody::simulate_maca_tiled(particles, config, block_size);
                check_tolerance(compare(cpu, tiled), "maca-tiled", integrator,
                                block_size);
            }
        }

        // Compute-only runs must not allocate or report recorded frames.
        nbody::RunOptions compute_only;
        compute_only.record_enabled = false;
        compute_only.diagnostics_enabled = false;
        config.integrator = nbody::Integrator::Leapfrog;
        const auto cpu_compute_only =
            nbody::simulate_cpu(particles, config, compute_only);
        const auto naive_compute_only = nbody::simulate_maca_naive(
            particles, config, 128, compute_only);
        const auto tiled_compute_only = nbody::simulate_maca_tiled(
            particles, config, 128, compute_only);
        check_tolerance(compare(cpu_compute_only, naive_compute_only),
                        "maca-naive record=off", config.integrator, 128);
        check_tolerance(compare(cpu_compute_only, tiled_compute_only),
                        "maca-tiled record=off", config.integrator, 128);

        std::cout << "MACA backends vs CPU FP64 passed: N=" << particle_count
                  << " integrators=Euler,Leapfrog blocks=64,128,256\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "MACA comparison test failed: " << error.what() << '\n';
        return 1;
    }
}
