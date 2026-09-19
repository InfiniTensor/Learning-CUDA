#include "nbody/cpu_solver.hpp"

#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace nbody {
namespace {

using Clock = std::chrono::steady_clock;

double milliseconds(Clock::time_point begin, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - begin).count();
}

void check_finite(const std::vector<Particle>& particles, int step) {
    for (std::size_t i = 0; i < particles.size(); ++i) {
        const auto& p = particles[i];
        if (!std::isfinite(p.position.x) || !std::isfinite(p.position.y) ||
            !std::isfinite(p.position.z) || !std::isfinite(p.velocity.x) ||
            !std::isfinite(p.velocity.y) || !std::isfinite(p.velocity.z)) {
            throw std::runtime_error("第 " + std::to_string(step) + " 步粒子 " +
                                     std::to_string(i) + " 出现非有限状态");
        }
    }
}

void record_snapshot(const std::vector<Particle>& particles,
                     std::size_t record_index,
                     std::size_t record_count,
                     std::vector<float>& trajectory) {
    for (std::size_t p = 0; p < particles.size(); ++p) {
        const std::size_t offset = (p * record_count + record_index) * 3;
        const float x = static_cast<float>(particles[p].position.x);
        const float y = static_cast<float>(particles[p].position.y);
        const float z = static_cast<float>(particles[p].position.z);
        if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) {
            throw std::runtime_error("轨迹坐标无法用 float32 表示");
        }
        trajectory[offset] = x;
        trajectory[offset + 1] = y;
        trajectory[offset + 2] = z;
    }
}

}  // namespace

std::vector<Vec3> compute_accelerations(const std::vector<Particle>& particles,
                                        double gravitational_constant,
                                        double softening) {
    const std::size_t n = particles.size();
    std::vector<Vec3> acceleration(n);
    const double eps2 = softening * softening;
    for (std::size_t i = 0; i < n; ++i) {
        Vec3 sum{};
        for (std::size_t j = 0; j < n; ++j) {
            if (j == i) continue;
            const Vec3 displacement = particles[j].position - particles[i].position;
            const double distance2 = dot(displacement, displacement) + eps2;
            const double inverse_distance = 1.0 / std::sqrt(distance2);
            const double scale = gravitational_constant * particles[j].mass *
                                 inverse_distance * inverse_distance * inverse_distance;
            sum += displacement * scale;
        }
        acceleration[i] = sum;
    }
    return acceleration;
}

Diagnostics compute_diagnostics(const std::vector<Particle>& particles,
                                double gravitational_constant,
                                double softening) {
    Diagnostics diagnostics;
    double total_mass = 0.0;
    for (const auto& particle : particles) {
        diagnostics.kinetic += 0.5 * particle.mass * dot(particle.velocity, particle.velocity);
        diagnostics.momentum += particle.velocity * particle.mass;
        diagnostics.angular_momentum += cross(particle.position, particle.velocity) * particle.mass;
        diagnostics.center_of_mass += particle.position * particle.mass;
        total_mass += particle.mass;
    }
    const double eps2 = softening * softening;
    for (std::size_t i = 0; i < particles.size(); ++i) {
        for (std::size_t j = i + 1; j < particles.size(); ++j) {
            const Vec3 displacement = particles[j].position - particles[i].position;
            diagnostics.potential -= gravitational_constant * particles[i].mass *
                                     particles[j].mass /
                                     std::sqrt(dot(displacement, displacement) + eps2);
        }
    }
    diagnostics.total_energy = diagnostics.kinetic + diagnostics.potential;
    diagnostics.center_of_mass = diagnostics.center_of_mass * (1.0 / total_mass);
    return diagnostics;
}

std::vector<int> make_recorded_steps(int num_steps, int record_interval) {
    if (num_steps < 0 || record_interval < 1) {
        throw std::invalid_argument("无效的记录参数");
    }
    std::vector<int> steps;
    for (int step = 0; step <= num_steps; step += record_interval) {
        steps.push_back(step);
        if (step > num_steps - record_interval) break;
    }
    if (steps.back() != num_steps) steps.push_back(num_steps);
    return steps;
}

CpuRunResult simulate_cpu(const std::vector<Particle>& initial_particles,
                          const Config& config,
                          const RunOptions& options) {
    if (initial_particles.empty()) throw std::invalid_argument("至少需要一个粒子");
    CpuRunResult result;
    result.record_enabled = options.record_enabled;
    result.diagnostics_enabled = options.diagnostics_enabled;
    result.particles = initial_particles;
    if (options.record_enabled) {
        result.recorded_steps =
            make_recorded_steps(config.num_steps, config.record_interval);
    }
    const std::size_t record_count = result.recorded_steps.size();
    const std::size_t n = result.particles.size();
    if (options.record_enabled &&
        record_count > std::numeric_limits<std::size_t>::max() / n / 3) {
        throw std::overflow_error("轨迹大小溢出");
    }
    if (options.record_enabled) {
        result.trajectory.resize(n * record_count * 3);
        record_snapshot(result.particles, 0, record_count, result.trajectory);
    }
    if (options.diagnostics_enabled) {
        result.initial_diagnostics = compute_diagnostics(
            result.particles, config.gravitational_constant, config.softening);
    }

    const auto simulation_begin = Clock::now();
    auto force_begin = Clock::now();
    std::vector<Vec3> acceleration = compute_accelerations(
        result.particles, config.gravitational_constant, config.softening);
    result.force_ms += milliseconds(force_begin, Clock::now());
    result.force_evaluations = 1;
    std::size_t next_record = 1;

    for (int step = 1; step <= config.num_steps; ++step) {
        if (config.integrator == Integrator::Euler) {
            for (std::size_t i = 0; i < n; ++i) {
                const Vec3 old_velocity = result.particles[i].velocity;
                result.particles[i].position += old_velocity * config.dt;
                result.particles[i].velocity += acceleration[i] * config.dt;
            }
            if (step < config.num_steps) {
                force_begin = Clock::now();
                acceleration = compute_accelerations(
                    result.particles, config.gravitational_constant, config.softening);
                result.force_ms += milliseconds(force_begin, Clock::now());
                ++result.force_evaluations;
            }
        } else {
            for (std::size_t i = 0; i < n; ++i) {
                result.particles[i].velocity += acceleration[i] * (0.5 * config.dt);
                result.particles[i].position += result.particles[i].velocity * config.dt;
            }
            force_begin = Clock::now();
            acceleration = compute_accelerations(
                result.particles, config.gravitational_constant, config.softening);
            result.force_ms += milliseconds(force_begin, Clock::now());
            ++result.force_evaluations;
            for (std::size_t i = 0; i < n; ++i) {
                result.particles[i].velocity += acceleration[i] * (0.5 * config.dt);
            }
        }

        check_finite(result.particles, step);
        if (options.record_enabled && next_record < record_count &&
            result.recorded_steps[next_record] == step) {
            record_snapshot(result.particles, next_record, record_count, result.trajectory);
            ++next_record;
        }
    }
    result.simulation_ms = milliseconds(simulation_begin, Clock::now());
    if (options.diagnostics_enabled) {
        result.final_diagnostics = compute_diagnostics(
            result.particles, config.gravitational_constant, config.softening);
    }
    return result;
}

}  // namespace nbody
