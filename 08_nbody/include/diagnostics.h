// diagnostics.h —— 能量 / 动量 / 角动量诊断量
#pragma once

#include "nbody_types.h"

namespace nbody {

// 所有累加一律用 double，与 real 的选择无关。
// 理由：N=65536 时势能项有 ~2e9 次累加，若用 FP32 累加，
// 舍入误差会淹没真实的物理误差，导致"能量守恒得多好"这个结论毫无意义。
struct Diagnostics {
  double kinetic = 0.0;
  double potential = 0.0;
  double total_energy = 0.0;
  double momentum[3] = {0.0, 0.0, 0.0};
  double momentum_magnitude = 0.0;
  double angular_momentum[3] = {0.0, 0.0, 0.0};
  double angular_momentum_magnitude = 0.0;
  // 最大加速度模长，用于监控近距离交会导致的数值爆炸。
  double max_acc_magnitude = 0.0;
};

// 在 CPU 上计算全部诊断量。势能是 O(N^2) 的，大 N 时不要每步都调用。
// 势能只统计每对一次（i<j），符号为负。
Diagnostics ComputeDiagnostics(const HostBodies &bodies, real G,
                               real softening);

// 二体系统的轨道要素，用于圆/椭圆轨道验证。
// 用相对坐标 + 归约质量把二体问题化为等效单体问题。
struct OrbitalElements {
  double semi_major_axis = 0.0;
  double eccentricity = 0.0;
  double separation = 0.0;
  double period = 0.0; // 由 a 与总质量按开普勒第三定律推出
};

OrbitalElements ComputeTwoBodyElements(const HostBodies &bodies, real G);

} // namespace nbody
