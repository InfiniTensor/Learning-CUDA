#include "nbody.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace nbody {
namespace {
using Clock = std::chrono::steady_clock;
void acceleration(const std::vector<Vec4> &position, std::vector<Vec4> &accel, const Config &c) {
    const auto n = static_cast<int>(position.size());
    if (c.G == 0) {
        std::fill(accel.begin(), accel.end(), Vec4{});
        return;
    }
    const float eps2 = c.softening * c.softening;
    // Same ordered all-pairs sum as CUDA; GPU uses rsqrtf and FMA, so agreement is a tolerance.
#pragma omp parallel for schedule(static) if (n >= 512)
    for (int i = 0; i < n; ++i) {
        const auto p = position[i];
        float ax = 0, ay = 0, az = 0;
        for (int j = 0; j < n; ++j) {
            if (i == j || position[j].w == 0)
                continue;
            const auto q = position[j];
            const float dx = q.x - p.x, dy = q.y - p.y, dz = q.z - p.z;
            const float r2 = dx * dx + dy * dy + dz * dz + eps2;
            const float inv = 1.0f / std::sqrt(r2);
            const float factor = c.G * q.w * inv * inv * inv;
            ax += dx * factor;
            ay += dy * factor;
            az += dz * factor;
        }
        accel[i] = {ax, ay, az, 0};
    }
}
} // namespace

Timing run_cpu(State &s, const Config &c, const Options &o, const Recorder &record) {
    const auto wall_start = Clock::now();
    Timing timing;
#ifdef _OPENMP
    if (o.cpu_threads > 0)
        omp_set_num_threads(o.cpu_threads);
    timing.cpu_threads = omp_get_max_threads();
#else
    if (o.cpu_threads > 1)
        throw std::runtime_error("This build has no OpenMP support");
#endif
    std::vector<Vec4> accel(s.position.size());
    const auto n = static_cast<int>(s.position.size());
    const bool zero_gravity = c.G == 0;
    auto segment_start = Clock::now();
    if (c.integrator == "leapfrog" && !zero_gravity)
        acceleration(s.position, accel, c);
    for (int step = 1; step <= c.num_steps; ++step) {
        if (c.integrator == "euler") {
            if (!zero_gravity)
                acceleration(s.position, accel, c);
            // Explicit Euler: drift with OLD velocity, then kick.
#pragma omp parallel for schedule(static) if (n >= 512)
            for (int i = 0; i < n; ++i) {
                auto &p = s.position[i];
                auto &v = s.velocity[i];
                p.x += c.dt * v.x;
                p.y += c.dt * v.y;
                p.z += c.dt * v.z;
                v.x += c.dt * accel[i].x;
                v.y += c.dt * accel[i].y;
                v.z += c.dt * accel[i].z;
            }
        } else {
            // Kick-drift-kick leapfrog / velocity Verlet, velocities at integer times.
#pragma omp parallel for schedule(static) if (n >= 512)
            for (int i = 0; i < n; ++i) {
                auto &p = s.position[i];
                auto &v = s.velocity[i];
                v.x += 0.5f * c.dt * accel[i].x;
                v.y += 0.5f * c.dt * accel[i].y;
                v.z += 0.5f * c.dt * accel[i].z;
                p.x += c.dt * v.x;
                p.y += c.dt * v.y;
                p.z += c.dt * v.z;
            }
            if (!zero_gravity)
                acceleration(s.position, accel, c);
#pragma omp parallel for schedule(static) if (n >= 512)
            for (int i = 0; i < n; ++i) {
                s.velocity[i].x += 0.5f * c.dt * accel[i].x;
                s.velocity[i].y += 0.5f * c.dt * accel[i].y;
                s.velocity[i].z += 0.5f * c.dt * accel[i].z;
            }
        }
        if (record && (step % c.record_interval == 0 || step == c.num_steps)) {
            timing.compute_seconds +=
                std::chrono::duration<double>(Clock::now() - segment_start).count();
            record(step, s.position);
            segment_start = Clock::now();
        }
    }
    timing.compute_seconds += std::chrono::duration<double>(Clock::now() - segment_start).count();
    timing.host_compute_seconds = timing.compute_seconds;
    timing.backend_wall_seconds = std::chrono::duration<double>(Clock::now() - wall_start).count();
    return timing;
}

Diagnostics diagnose(const State &s, const Config &c, bool energy) {
    Diagnostics d;
    for (std::size_t i = 0; i < s.position.size(); ++i) {
        const auto &p = s.position[i];
        const auto &v = s.velocity[i];
        if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z) ||
            !std::isfinite(v.x) || !std::isfinite(v.y) || !std::isfinite(v.z))
            throw std::runtime_error("Non-finite state: reduce dt or rescale the input units");
        const double m = p.w;
        const double speed2 = double(v.x) * v.x + double(v.y) * v.y + double(v.z) * v.z;
        d.mass += m;
        d.kinetic += 0.5 * m * speed2;
        d.momentum_scale += m * std::sqrt(speed2);
        d.momentum[0] += m * v.x;
        d.momentum[1] += m * v.y;
        d.momentum[2] += m * v.z;
        d.com[0] += m * p.x;
        d.com[1] += m * p.y;
        d.com[2] += m * p.z;
        d.angular[0] += m * (double(p.y) * v.z - double(p.z) * v.y);
        d.angular[1] += m * (double(p.z) * v.x - double(p.x) * v.z);
        d.angular[2] += m * (double(p.x) * v.y - double(p.y) * v.x);
        if (energy && c.G != 0) {
            for (std::size_t j = i + 1; j < s.position.size(); ++j) {
                const auto &q = s.position[j];
                const double dx = double(q.x) - p.x, dy = double(q.y) - p.y, dz = double(q.z) - p.z;
                d.potential -=
                    double(c.G) * m * q.w /
                    std::sqrt(dx * dx + dy * dy + dz * dz + double(c.softening) * c.softening);
            }
        }
    }
    for (auto &x : d.com)
        x /= d.mass;
    d.energy = d.kinetic + d.potential;
    return d;
}
} // namespace nbody
