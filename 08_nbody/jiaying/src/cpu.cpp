#include "nbody.hpp"
#include <chrono>
#include <cmath>
#include <omp.h>

namespace nb {
namespace {
using Clock = std::chrono::steady_clock;
double elapsed(Clock::time_point t) {
  return std::chrono::duration<double, std::milli>(Clock::now() - t).count();
}
void force(const std::vector<Vec4> &p, std::vector<Vec4> &a, const Config &c,
           int threads) {
  const int n = static_cast<int>(p.size());
  const float eps2 = c.softening * c.softening;
#pragma omp parallel for schedule(static) num_threads(threads) if (threads > 1)
  for (int i = 0; i < n; ++i) {
    float ax = 0, ay = 0, az = 0;
    // Independent destinations; single-thread baseline does not use Newton
    // symmetry.
    for (int j = 0; j < n; ++j) {
      if (i == j)
        continue;
      float dx = p[j].x - p[i].x, dy = p[j].y - p[i].y, dz = p[j].z - p[i].z;
      float inv = 1.0f / std::sqrt(dx * dx + dy * dy + dz * dz + eps2);
      float scale = c.G * p[j].w * inv * inv * inv;
      ax += dx * scale;
      ay += dy * scale;
      az += dz * scale;
    }
    a[i] = {ax, ay, az, 0};
  }
}
} // namespace
Timing run_cpu(State &s, const Config &c, const Options &options,
               const Record &record) {
  std::vector<Vec4> a(s.position.size());
  Timing timing;
  const auto begin = Clock::now();
  if (record)
    record(0, s.position.data());
  auto t = Clock::now();
  if (c.steps > 0 && c.integrator == "leapfrog")
    force(s.position, a, c, options.cpu_threads);
  timing.compute_ms += elapsed(t);
  for (int step = 1; step <= c.steps; ++step) {
    t = Clock::now();
    if (c.integrator == "euler")
      force(s.position, a, c, options.cpu_threads);
    for (std::size_t i = 0; i < a.size(); ++i) {
      auto &p = s.position[i];
      auto &v = s.velocity[i];
      if (c.integrator == "leapfrog") {
        v.x += 0.5f * c.dt * a[i].x;
        v.y += 0.5f * c.dt * a[i].y;
        v.z += 0.5f * c.dt * a[i].z;
      }
      p.x += c.dt * v.x;
      p.y += c.dt * v.y;
      p.z += c.dt * v.z;
      if (c.integrator == "euler") {
        v.x += c.dt * a[i].x;
        v.y += c.dt * a[i].y;
        v.z += c.dt * a[i].z;
      }
    }
    if (c.integrator == "leapfrog") {
      force(s.position, a, c, options.cpu_threads);
      for (std::size_t i = 0; i < a.size(); ++i) {
        auto &v = s.velocity[i];
        v.x += 0.5f * c.dt * a[i].x;
        v.y += 0.5f * c.dt * a[i].y;
        v.z += 0.5f * c.dt * a[i].z;
      }
    }
    timing.compute_ms += elapsed(t);
    if (record && (step % c.record_interval == 0 || step == c.steps))
      record(step, s.position.data());
  }
  timing.wall_ms = elapsed(begin);
  return timing;
}
Diagnostics diagnostics(const State &s, const Config &c) {
  Diagnostics d;
  const double eps2 = double(c.softening) * c.softening;
  for (std::size_t i = 0; i < s.position.size(); ++i) {
    auto p = s.position[i], v = s.velocity[i];
    double mass = p.w;
    d.mass += mass;
    d.kinetic += 0.5 * mass *
                 (double(v.x) * v.x + double(v.y) * v.y + double(v.z) * v.z);
    d.momentum[0] += mass * v.x;
    d.momentum[1] += mass * v.y;
    d.momentum[2] += mass * v.z;
    d.momentum_scale += mass * std::sqrt(double(v.x) * v.x + double(v.y) * v.y +
                                         double(v.z) * v.z);
    d.com[0] += mass * p.x;
    d.com[1] += mass * p.y;
    d.com[2] += mass * p.z;
    d.angular[0] += mass * (double(p.y) * v.z - double(p.z) * v.y);
    d.angular[1] += mass * (double(p.z) * v.x - double(p.x) * v.z);
    d.angular[2] += mass * (double(p.x) * v.y - double(p.y) * v.x);
    for (std::size_t j = 0; j < i; ++j) {
      double dx = double(s.position[j].x) - p.x,
             dy = double(s.position[j].y) - p.y,
             dz = double(s.position[j].z) - p.z;
      d.potential -= double(c.G) * mass * s.position[j].w /
                     std::sqrt(dx * dx + dy * dy + dz * dz + eps2);
    }
  }
  for (double &v : d.com)
    v /= d.mass;
  return d;
}
} // namespace nb
