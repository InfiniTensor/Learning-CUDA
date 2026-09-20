// ============================================================
// analysis.h —— 正确性验证辅助函数 (纯 host, O(N^2) CPU 端)
//   - calc_total_momentum: 总动量 (封闭系统守恒, 用作 drift 指标)
//   - calc_total_energy:   动能 + 引力势能 (Leapfrog 应近似守恒)
//   - cpu_accelerations_naive: O(N^2) 朴素 CPU 参考加速度 (用于 CUDA kernel 数值比对 + 加速比基准)
// ============================================================
#ifndef NBODY_ANALYSIS_H
#define NBODY_ANALYSIS_H

#include "nbody_types.h"

// 总动量 (3 分量) → 返回 {px, py, pz} = Σ m_i v_i
struct Vec3 {
    double x, y, z;
};
Vec3 calc_total_momentum(const ParticleSet& host_set);

// 总能量 E = 0.5 Σ m_i v_i²  -  Σ_{i<j} G m_i m_j / sqrt(|r_ij|² + eps²)
//   单位化后 drift = |(E_end - E_begin) / E_begin| 应在 Leapfrog 下 <<1%
double calc_total_energy(const ParticleSet& host_set, double G, double eps);

// CPU 朴素 O(N^2) 加速度 (输出 ax,ay,az 必须已分配 N 长度并初始化为 0)
//   用于:
//     1) 与 CUDA kernel 数值结果 bit-close 验证 (正确性单元测试)
//     2) 作为加速比基准 (CPU vs GPU 的 timing 对比)
void cpu_accelerations_naive(const ParticleSet& host_set, double G, double eps, double* ax_out,
                             double* ay_out, double* az_out);

// CPU 朴素一步积分 (与 CUDA 的 nbody_step_once 保持同样的算法, 用于 ground truth)
void cpu_step_once_naive(ParticleSet& host_set, const SimParams& sp);

#endif // NBODY_ANALYSIS_H
