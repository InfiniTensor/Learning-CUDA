// ============================================================
// analysis.cpp —— 纯 host CPU: 动量 / 能量 / 朴素 O(N^2) 加速度 / CPU 参考一步积分
// ============================================================
#include "analysis.h"
#include <cmath>
#include <cstring>

// ============================================================
// 总动量
// ============================================================
Vec3 calc_total_momentum(const ParticleSet& h) {
    Vec3 p = {0.0, 0.0, 0.0};
    for (int i = 0; i < h.N; ++i) {
        double m = (double)h.mass[i];
        p.x += m * (double)h.vx[i];
        p.y += m * (double)h.vy[i];
        p.z += m * (double)h.vz[i];
    }
    return p;
}

// ============================================================
// 总能量 (KE + PE), 双精度
// ============================================================
double calc_total_energy(const ParticleSet& h, double G, double eps) {
    double KE = 0.0;
    for (int i = 0; i < h.N; ++i) {
        double m = (double)h.mass[i];
        double vx = h.vx[i], vy = h.vy[i], vz = h.vz[i];
        KE += 0.5 * m * (vx * vx + vy * vy + vz * vz);
    }
    double PE = 0.0;
    double eps2 = eps * eps;
    for (int i = 0; i < h.N; ++i) {
        double xi = h.x[i], yi = h.y[i], zi = h.z[i], mi = h.mass[i];
        for (int j = i + 1; j < h.N; ++j) {
            double dx = (double)h.x[j] - xi;
            double dy = (double)h.y[j] - yi;
            double dz = (double)h.z[j] - zi;
            double r2 = dx * dx + dy * dy + dz * dz + eps2;
            double inv_r = 1.0 / std::sqrt(r2);
            PE -= G * mi * (double)h.mass[j] * inv_r;
        }
    }
    return KE + PE;
}

// ============================================================
// CPU 朴素 O(N^2) 加速度 (双精度累加, 用于正确性基准)
// ============================================================
void cpu_accelerations_naive(const ParticleSet& h, double G, double eps, double* ax_out,
                             double* ay_out, double* az_out) {
    int N = h.N;
    for (int i = 0; i < N; ++i) {
        ax_out[i] = 0.0;
        ay_out[i] = 0.0;
        az_out[i] = 0.0;
    }
    double eps2 = eps * eps;
    for (int i = 0; i < N; ++i) {
        double xi = h.x[i], yi = h.y[i], zi = h.z[i];
        double axi = 0.0, ayi = 0.0, azi = 0.0;
        for (int j = 0; j < N; ++j) {
            if (i == j)
                continue;
            double dx = (double)h.x[j] - xi;
            double dy = (double)h.y[j] - yi;
            double dz = (double)h.z[j] - zi;
            double mj = (double)h.mass[j];
            double r2 = dx * dx + dy * dy + dz * dz + eps2;
            double inv_r3 = 1.0 / (r2 * std::sqrt(r2));
            double f = G * mj * inv_r3;
            axi += f * dx;
            ayi += f * dy;
            azi += f * dz;
        }
        ax_out[i] = axi;
        ay_out[i] = ayi;
        az_out[i] = azi;
    }
}

// ============================================================
// 辅助: 单步 Euler / Leapfrog host 参考
//   Leapfrog 按 "半推速度 半步 v(t-0.5dt)" 约定:
//     入口前需先调用 integrator_init: v_half = v0 + 0.5*a0*dt
//     然后每步:  x_new  = x + v_half * dt;  a_new;
//                v_half = v_half + a_new * dt;
//     注意: 存贮在 ParticleSet.vx 中的是 "半速" v(t ± 0.5dt).
// ============================================================
static void cpu_compute_accel(const ParticleSet& h, double G, double eps) {
    int N = h.N;
    double eps2 = eps * eps;
    for (int i = 0; i < N; ++i)
        h.ax[i] = h.ay[i] = h.az[i] = 0.0f;
    for (int i = 0; i < N; ++i) {
        double xi = h.x[i], yi = h.y[i], zi = h.z[i];
        double axi = 0, ayi = 0, azi = 0;
        for (int j = 0; j < N; ++j) {
            if (i == j)
                continue;
            double dx = (double)h.x[j] - xi;
            double dy = (double)h.y[j] - yi;
            double dz = (double)h.z[j] - zi;
            double mj = (double)h.mass[j];
            double r2 = dx * dx + dy * dy + dz * dz + eps2;
            double inv_r3 = 1.0 / (r2 * std::sqrt(r2));
            double f = G * mj * inv_r3;
            axi += f * dx;
            ayi += f * dy;
            azi += f * dz;
        }
        h.ax[i] = (float)axi;
        h.ay[i] = (float)ayi;
        h.az[i] = (float)azi;
    }
}
void cpu_step_once_naive(ParticleSet& h, const SimParams& sp) {
    int N = h.N;
    if (sp.integrator == Integrator::EULER) {
        cpu_compute_accel(h, sp.G, sp.softening);
        float dt = (float)sp.dt;
        for (int i = 0; i < N; ++i) {
            h.vx[i] += h.ax[i] * dt;
            h.vy[i] += h.ay[i] * dt;
            h.vz[i] += h.az[i] * dt;
            h.x[i] += h.vx[i] * dt;
            h.y[i] += h.vy[i] * dt;
            h.z[i] += h.vz[i] * dt;
        }
    } else { // LEAPFROG
        // ParticleSet.v* 存的是 v(t-0.5dt)
        // 先算 a(t), 推半速至 v(t+0.5dt), 再推位置 x(t+dt)
        cpu_compute_accel(h, sp.G, sp.softening);
        float dt = (float)sp.dt;
        for (int i = 0; i < N; ++i) {
            h.vx[i] += h.ax[i] * dt;
            h.vy[i] += h.ay[i] * dt;
            h.vz[i] += h.az[i] * dt;
            h.x[i] += h.vx[i] * dt;
            h.y[i] += h.vy[i] * dt;
            h.z[i] += h.vz[i] * dt;
        }
    }
}
