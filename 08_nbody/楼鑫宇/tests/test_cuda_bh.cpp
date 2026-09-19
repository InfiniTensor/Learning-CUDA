#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

#include "nbody/cuda_solver.hpp"

namespace {

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

double relative_l2_displacement(
    const std::vector<nbody::Particle>& initial,
    const nbody::CpuRunResult& reference,
    const nbody::CpuRunResult& candidate) {
    double error_squared = 0.0;
    double reference_squared = 0.0;
    for (std::size_t i = 0; i < initial.size(); ++i) {
        const auto a = reference.particles[i].position - initial[i].position;
        const auto b = candidate.particles[i].position - initial[i].position;
        const auto delta = b - a;
        error_squared += nbody::dot(delta, delta);
        reference_squared += nbody::dot(a, a);
    }
    require(reference_squared > 0.0, "reference displacement is zero");
    return std::sqrt(error_squared / reference_squared);
}

double relative_l2_velocity(const nbody::CpuRunResult& reference,
                            const nbody::CpuRunResult& candidate) {
    double error_squared = 0.0;
    double reference_squared = 0.0;
    for (std::size_t i = 0; i < reference.particles.size(); ++i) {
        const auto delta = candidate.particles[i].velocity -
                           reference.particles[i].velocity;
        error_squared += nbody::dot(delta, delta);
        reference_squared += nbody::dot(reference.particles[i].velocity,
                                       reference.particles[i].velocity);
    }
    require(reference_squared > 0.0, "reference velocity is zero");
    return std::sqrt(error_squared / reference_squared);
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
                "Barnes-Hut result contains non-finite particle data");
    }
}

}  // namespace

int main() {
    try {
        constexpr int particle_count = 65536;
        constexpr double theta = 0.5;
        constexpr double error_limit = 0.01;
        std::mt19937_64 generator(42);
        std::uniform_real_distribution<double> coordinate(-1.0, 1.0);
        std::vector<nbody::Particle> particles;
        particles.reserve(particle_count);
        while (particles.size() < particle_count) {
            const double x = coordinate(generator);
            const double y = coordinate(generator);
            const double z = coordinate(generator);
            if (x * x + y * y + z * z > 1.0) continue;
            particles.push_back({{x, y, z}, {0.0, 0.0, 0.0},
                                 1.0 / particle_count});
        }

        nbody::Config config;
        config.dt = 0.001;
        config.num_steps = 100;
        config.record_interval = 10;
        config.gravitational_constant = 1.0;
        config.softening = 0.01;
        config.integrator = nbody::Integrator::Leapfrog;

        nbody::RunOptions options;
        options.record_enabled = false;
        options.diagnostics_enabled = false;
        const auto direct =
            nbody::simulate_cuda_naive(particles, config, 128, options);
        const auto bh =
            nbody::simulate_cuda_bh(particles, config, 128, theta, options);

        require(direct.particles.size() == particle_count &&
                    bh.particles.size() == particle_count,
                "particle count changed");
        require(direct.force_evaluations == 101 && bh.force_evaluations == 101,
                "100 Leapfrog steps must evaluate force 101 times");
        require(direct.recorded_steps.empty() && bh.recorded_steps.empty() &&
                    direct.trajectory.empty() && bh.trajectory.empty(),
                "recording should be disabled");
        check_finite(bh);

        // With zero initial velocity, accumulated displacement and final
        // velocity are independent checks of the approximate force path.
        const double displacement_error =
            relative_l2_displacement(particles, direct, bh);
        const double velocity_error = relative_l2_velocity(direct, bh);
        require(std::isfinite(displacement_error) &&
                    std::isfinite(velocity_error) &&
                    displacement_error < error_limit &&
                    velocity_error < error_limit,
                "65536-particle Barnes-Hut 100-step relative L2 error exceeds 1%");

        // A lone particle has no self-force, independent of the tree layout.
        const std::vector<nbody::Particle> singleton = {
            {{0.25, -0.5, 0.75}, {0.0, 0.0, 0.0}, 1.0}};
        const auto self =
            nbody::simulate_cuda_bh(singleton, config, 128, theta, options);
        require(self.particles.size() == 1 && self.force_evaluations == 101,
                "single-particle output shape is wrong");
        check_finite(self);
        const auto& particle = self.particles.front();
        require(std::abs(particle.position.x - singleton[0].position.x) < 1e-7 &&
                    std::abs(particle.position.y - singleton[0].position.y) < 1e-7 &&
                    std::abs(particle.position.z - singleton[0].position.z) < 1e-7 &&
                    nbody::norm(particle.velocity) < 1e-7,
                "single particle experienced self-force");

        std::cout << "cuda-bh 65536 100-step vs cuda-naive passed: "
                  << "theta=" << theta
                  << " displacement_rel_l2=" << displacement_error
                  << " velocity_rel_l2=" << velocity_error << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "CUDA Barnes-Hut test failed: " << error.what() << '\n';
        return 1;
    }
}
