#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "nbody/cpu_solver.hpp"
#include "nbody/cuda_solver.hpp"

namespace {

double relative_l2(const nbody::CpuRunResult& reference,
                   const nbody::CpuRunResult& candidate,
                   bool position) {
    double error_squared = 0.0;
    double reference_squared = 0.0;
    for (std::size_t i = 0; i < reference.particles.size(); ++i) {
        const nbody::Vec3& a = position ? reference.particles[i].position
                                       : reference.particles[i].velocity;
        const nbody::Vec3& b = position ? candidate.particles[i].position
                                       : candidate.particles[i].velocity;
        const double dx = b.x - a.x;
        const double dy = b.y - a.y;
        const double dz = b.z - a.z;
        error_squared += dx * dx + dy * dy + dz * dz;
        reference_squared += a.x * a.x + a.y * a.y + a.z * a.z;
    }
    return std::sqrt(error_squared / reference_squared);
}

}  // namespace

int main() {
    try {
        constexpr int particle_count = 4096;
        std::vector<nbody::Particle> particles;
        particles.reserve(particle_count);
        for (int i = 0; i < particle_count; ++i) {
            const int x = i % 16;
            const int y = (i / 16) % 16;
            const int z = i / 256;
            particles.push_back({
                {(x - 7.5) / 8.0, (y - 7.5) / 8.0, (z - 7.5) / 8.0},
                {0.001 * (y - 7.5), -0.001 * (x - 7.5), 0.0},
                1.0 / particle_count,
            });
        }

        nbody::Config config;
        config.dt = 0.001;
        config.num_steps = 8;
        config.record_interval = 4;
        config.gravitational_constant = 1.0;
        config.softening = 0.01;
        config.integrator = nbody::Integrator::Leapfrog;

        nbody::RunOptions options;
        options.diagnostics_enabled = false;
        const auto cpu = nbody::simulate_cpu(particles, config, options);
        const auto mixed =
            nbody::simulate_cuda_mixed(particles, config, 64, options);
        if (mixed.particles.size() != cpu.particles.size() ||
            mixed.recorded_steps != cpu.recorded_steps ||
            mixed.trajectory.size() != cpu.trajectory.size() ||
            mixed.force_evaluations != cpu.force_evaluations) {
            throw std::runtime_error("CPU / mixed 输出形状或力计算次数不一致");
        }

        const double position_error = relative_l2(cpu, mixed, true);
        const double velocity_error = relative_l2(cpu, mixed, false);
        if (!std::isfinite(position_error) || !std::isfinite(velocity_error) ||
            position_error >= 0.001 || velocity_error >= 0.001) {
            throw std::runtime_error("4096 粒子 mixed / CPU FP64 偏差过大");
        }
        for (float value : mixed.trajectory) {
            if (!std::isfinite(value)) {
                throw std::runtime_error("mixed 轨迹包含非有限数值");
            }
        }

        std::cout << "cuda-mixed 4096 vs CPU FP64 passed: position_rel_l2="
                  << position_error << " velocity_rel_l2=" << velocity_error
                  << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "CUDA mixed 4096 test failed: " << error.what() << '\n';
        return 1;
    }
}
