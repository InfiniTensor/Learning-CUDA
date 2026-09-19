#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "nbody/cpu_solver.hpp"
#include "nbody/cuda_solver.hpp"

namespace {

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

double component_error(const nbody::Vec3& lhs, const nbody::Vec3& rhs) {
    return std::max({std::abs(lhs.x - rhs.x), std::abs(lhs.y - rhs.y),
                     std::abs(lhs.z - rhs.z)});
}

struct Comparison {
    double position = 0.0;
    double velocity = 0.0;
    double trajectory = 0.0;
};

Comparison compare(const nbody::CpuRunResult& cpu,
                   const nbody::CpuRunResult& gpu) {
    require(cpu.particles.size() == gpu.particles.size(),
            "CPU/GPU 粒子数不一致");
    require(cpu.recorded_steps == gpu.recorded_steps,
            "CPU/GPU 记录步不一致");
    require(cpu.trajectory.size() == gpu.trajectory.size(),
            "CPU/GPU 轨迹大小不一致");
    require(cpu.force_evaluations == gpu.force_evaluations,
            "CPU/GPU 力计算次数不一致");

    Comparison comparison;
    for (std::size_t i = 0; i < cpu.particles.size(); ++i) {
        comparison.position = std::max(
            comparison.position,
            component_error(cpu.particles[i].position,
                            gpu.particles[i].position));
        comparison.velocity = std::max(
            comparison.velocity,
            component_error(cpu.particles[i].velocity,
                            gpu.particles[i].velocity));
    }
    for (std::size_t i = 0; i < cpu.trajectory.size(); ++i) {
        comparison.trajectory = std::max(
            comparison.trajectory,
            std::abs(static_cast<double>(cpu.trajectory[i]) -
                     static_cast<double>(gpu.trajectory[i])));
    }
    return comparison;
}

void check_tolerance(const Comparison& comparison, const char* backend) {
    if (!(comparison.position < 2.0e-5 &&
          comparison.velocity < 2.0e-5 &&
          comparison.trajectory < 2.0e-5)) {
        throw std::runtime_error(std::string(backend) +
                                 " 与 CPU FP64 偏差过大");
    }
}

}  // namespace

int main() {
    try {
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
        config.integrator = nbody::Integrator::Leapfrog;

        const auto cpu = nbody::simulate_cpu(particles, config);
        const auto naive = nbody::simulate_cuda_naive(particles, config, 128);
        const Comparison naive_comparison = compare(cpu, naive);
        check_tolerance(naive_comparison, "cuda-naive");

        Comparison tiled_comparison;
        for (const int block_size : {64, 128, 256, 512}) {
            const auto tiled =
                nbody::simulate_cuda_tiled(particles, config, block_size);
            tiled_comparison = compare(cpu, tiled);
            check_tolerance(tiled_comparison, "cuda-tiled");
        }

        std::cout << "CUDA backends vs CPU FP64 passed: N=" << particle_count
                  << " naive_position_error=" << naive_comparison.position
                  << " tiled_position_error=" << tiled_comparison.position
                  << " tiled_velocity_error=" << tiled_comparison.velocity
                  << " tiled_trajectory_error=" << tiled_comparison.trajectory
                  << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "CUDA comparison test failed: " << error.what() << '\n';
        return 1;
    }
}
