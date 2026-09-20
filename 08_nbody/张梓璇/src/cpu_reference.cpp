#include "cpu_reference.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace {

void compute_acceleration(const std::vector<Particle>& particles,
                          float G,
                          float softening,
                          std::vector<float>& ax,
                          std::vector<float>& ay,
                          std::vector<float>& az) {
    const int n = static_cast<int>(particles.size());
    const float softening2 = softening * softening;
    std::fill(ax.begin(), ax.end(), 0.0f);
    std::fill(ay.begin(), ay.end(), 0.0f);
    std::fill(az.begin(), az.end(), 0.0f);

    // CPU 版故意保持最朴素的 O(N^2) 写法。
    // 它的任务是让人容易检查公式，并给小规模 GPU 结果做参照。
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i == j) continue;
            const float dx = particles[j].x - particles[i].x;
            const float dy = particles[j].y - particles[i].y;
            const float dz = particles[j].z - particles[i].z;
            const float dist2 = dx * dx + dy * dy + dz * dz + softening2;
            const float inv_dist = 1.0f / std::sqrt(dist2);
            const float inv_dist3 = inv_dist * inv_dist * inv_dist;
            const float factor = G * particles[j].mass * inv_dist3;
            ax[i] += factor * dx;
            ay[i] += factor * dy;
            az[i] += factor * dz;
        }
    }
}

}  // namespace

void simulate_cpu(std::vector<Particle>& particles, const SimConfig& config) {
    const int n = static_cast<int>(particles.size());
    std::vector<float> ax(n), ay(n), az(n);

    if (config.integrator == "euler") {
        for (int step = 0; step < config.num_steps; ++step) {
            compute_acceleration(particles, config.G, config.softening, ax, ay, az);
            for (int i = 0; i < n; ++i) {
                // 这里明确使用旧速度更新位置，是显式 Euler。
                particles[i].x += particles[i].vx * config.dt;
                particles[i].y += particles[i].vy * config.dt;
                particles[i].z += particles[i].vz * config.dt;
                particles[i].vx += ax[i] * config.dt;
                particles[i].vy += ay[i] * config.dt;
                particles[i].vz += az[i] * config.dt;
            }
        }
        return;
    }

    // Leapfrog 的第一次加速度要在循环外算好。
    // 每一步严格按 kick-drift-重新算加速度-kick 的顺序走。
    compute_acceleration(particles, config.G, config.softening, ax, ay, az);
    for (int step = 0; step < config.num_steps; ++step) {
        for (int i = 0; i < n; ++i) {
            particles[i].vx += 0.5f * ax[i] * config.dt;
            particles[i].vy += 0.5f * ay[i] * config.dt;
            particles[i].vz += 0.5f * az[i] * config.dt;
            particles[i].x += particles[i].vx * config.dt;
            particles[i].y += particles[i].vy * config.dt;
            particles[i].z += particles[i].vz * config.dt;
        }
        compute_acceleration(particles, config.G, config.softening, ax, ay, az);
        for (int i = 0; i < n; ++i) {
            particles[i].vx += 0.5f * ax[i] * config.dt;
            particles[i].vy += 0.5f * ay[i] * config.dt;
            particles[i].vz += 0.5f * az[i] * config.dt;
        }
    }
}

void state_error(const std::vector<Particle>& reference,
                 const std::vector<Particle>& test,
                 double& max_position_error,
                 double& rms_position_error,
                 double& max_velocity_error,
                 double& rms_velocity_error) {
    if (reference.size() != test.size() || reference.empty()) {
        throw std::runtime_error("计算位置误差时粒子数组长度不一致或为空");
    }
    double position_sum_squared = 0.0;
    double velocity_sum_squared = 0.0;
    max_position_error = 0.0;
    max_velocity_error = 0.0;
    for (std::size_t i = 0; i < reference.size(); ++i) {
        const double dx = static_cast<double>(test[i].x) - reference[i].x;
        const double dy = static_cast<double>(test[i].y) - reference[i].y;
        const double dz = static_cast<double>(test[i].z) - reference[i].z;
        const double position_value = std::sqrt(dx * dx + dy * dy + dz * dz);
        max_position_error = std::max(max_position_error, position_value);
        position_sum_squared += position_value * position_value;

        const double dvx = static_cast<double>(test[i].vx) - reference[i].vx;
        const double dvy = static_cast<double>(test[i].vy) - reference[i].vy;
        const double dvz = static_cast<double>(test[i].vz) - reference[i].vz;
        const double velocity_value = std::sqrt(dvx * dvx + dvy * dvy + dvz * dvz);
        max_velocity_error = std::max(max_velocity_error, velocity_value);
        velocity_sum_squared += velocity_value * velocity_value;
    }
    rms_position_error = std::sqrt(position_sum_squared / reference.size());
    rms_velocity_error = std::sqrt(velocity_sum_squared / reference.size());
}
