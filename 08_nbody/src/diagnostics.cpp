// diagnostics.cpp —— 能量 / 动量 / 角动量与二体轨道要素
#include "diagnostics.h"

#include <cmath>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace nbody {

Diagnostics ComputeDiagnostics(const HostBodies &bodies, real G,
                               real softening) {
  const int n = static_cast<int>(bodies.size());
  Diagnostics d;

  const double eps2 =
      static_cast<double>(softening) * static_cast<double>(softening);
  const double g = static_cast<double>(G);

  for (int i = 0; i < n; ++i) {
    const double m = bodies.mass[i];
    const double vx = bodies.vel_x[i], vy = bodies.vel_y[i],
                 vz = bodies.vel_z[i];
    const double x = bodies.pos_x[i], y = bodies.pos_y[i], z = bodies.pos_z[i];

    d.kinetic += 0.5 * m * (vx * vx + vy * vy + vz * vz);

    d.momentum[0] += m * vx;
    d.momentum[1] += m * vy;
    d.momentum[2] += m * vz;

    // L = r × (m v)
    d.angular_momentum[0] += m * (y * vz - z * vy);
    d.angular_momentum[1] += m * (z * vx - x * vz);
    d.angular_momentum[2] += m * (x * vy - y * vx);
  }

  // ------------------------------------------------------------------
  // 势能与最大加速度：O(N^2)，与力计算同样昂贵。
  // 用 OpenMP 并行（N=65536 时串行要半分钟，会显著拖长每次基准运行）。
  // 累加一律用 double，与 real 的选择无关：N=65536 时势能有 ~2e9 次累加，
  // 若用 FP32 累加，舍入误差会淹没真实的物理误差。
  //
  // 势能只统计每对一次（i<j），且与力计算使用同一个软化核——
  // 否则能量守恒的检验对象与实际积分的系统不是同一个系统。
  // ------------------------------------------------------------------
  double potential = 0.0;
  // 每个 i 的加速度模长平方存下来，循环结束后串行取最大值。
  // 不用 reduction(max:) —— 那是 OpenMP 3.1 的特性，而 MSVC 只支持 2.0，
  // 用了会在 Windows 上编译失败。串行取 max 是 O(N)，相比 O(N^2) 可忽略。
  std::vector<double> acc2_per_particle(static_cast<std::size_t>(n), 0.0);
#ifdef _OPENMP
#pragma omp parallel for reduction(+ : potential) schedule(static)
#endif
  for (int i = 0; i < n; ++i) {
    double ax = 0.0, ay = 0.0, az = 0.0;
    double pot_i = 0.0;
    const double xi = bodies.pos_x[i], yi = bodies.pos_y[i],
                 zi = bodies.pos_z[i];
    const double mi = bodies.mass[i];
    for (int j = 0; j < n; ++j) {
      if (j == i)
        continue;
      const double dx = static_cast<double>(bodies.pos_x[j]) - xi;
      const double dy = static_cast<double>(bodies.pos_y[j]) - yi;
      const double dz = static_cast<double>(bodies.pos_z[j]) - zi;
      const double r2 = dx * dx + dy * dy + dz * dz + eps2;
      const double inv_r = 1.0 / std::sqrt(r2);
      const double mj = bodies.mass[j];
      if (j > i)
        pot_i -= g * mi * mj * inv_r;
      const double s = g * mj * inv_r * inv_r * inv_r;
      ax += dx * s;
      ay += dy * s;
      az += dz * s;
    }
    potential += pot_i;
    acc2_per_particle[static_cast<std::size_t>(i)] =
        ax * ax + ay * ay + az * az;
  }

  double acc_max2 = 0.0;
  for (int i = 0; i < n; ++i) {
    if (acc2_per_particle[static_cast<std::size_t>(i)] > acc_max2) {
      acc_max2 = acc2_per_particle[static_cast<std::size_t>(i)];
    }
  }

  d.potential = potential;
  d.max_acc_magnitude = std::sqrt(acc_max2);
  d.total_energy = d.kinetic + d.potential;
  d.momentum_magnitude =
      std::sqrt(d.momentum[0] * d.momentum[0] + d.momentum[1] * d.momentum[1] +
                d.momentum[2] * d.momentum[2]);
  d.angular_momentum_magnitude =
      std::sqrt(d.angular_momentum[0] * d.angular_momentum[0] +
                d.angular_momentum[1] * d.angular_momentum[1] +
                d.angular_momentum[2] * d.angular_momentum[2]);
  return d;
}

OrbitalElements ComputeTwoBodyElements(const HostBodies &bodies, real G) {
  OrbitalElements e;
  if (bodies.size() != 2)
    return e;

  const double m1 = bodies.mass[0], m2 = bodies.mass[1];
  const double mu = static_cast<double>(G) * (m1 + m2); // 标准引力参数

  // 相对坐标与相对速度：把二体问题化为等效单体问题。
  const double rx = static_cast<double>(bodies.pos_x[1]) - bodies.pos_x[0];
  const double ry = static_cast<double>(bodies.pos_y[1]) - bodies.pos_y[0];
  const double rz = static_cast<double>(bodies.pos_z[1]) - bodies.pos_z[0];
  const double vx = static_cast<double>(bodies.vel_x[1]) - bodies.vel_x[0];
  const double vy = static_cast<double>(bodies.vel_y[1]) - bodies.vel_y[0];
  const double vz = static_cast<double>(bodies.vel_z[1]) - bodies.vel_z[0];

  const double r = std::sqrt(rx * rx + ry * ry + rz * rz);
  const double v2 = vx * vx + vy * vy + vz * vz;
  e.separation = r;

  // 比轨道能量 → 半长轴（vis-viva）
  const double energy = 0.5 * v2 - mu / r;
  if (energy < 0.0) {
    e.semi_major_axis = -mu / (2.0 * energy);
    // 开普勒第三定律
    e.period = 2.0 * 3.14159265358979323846 *
               std::sqrt(e.semi_major_axis * e.semi_major_axis *
                         e.semi_major_axis / mu);
  }

  // 比角动量 → 偏心率
  const double hx = ry * vz - rz * vy;
  const double hy = rz * vx - rx * vz;
  const double hz = rx * vy - ry * vx;
  const double h2 = hx * hx + hy * hy + hz * hz;
  const double ecc2 = 1.0 + 2.0 * energy * h2 / (mu * mu);
  e.eccentricity = ecc2 > 0.0 ? std::sqrt(ecc2) : 0.0;
  return e;
}

} // namespace nbody
