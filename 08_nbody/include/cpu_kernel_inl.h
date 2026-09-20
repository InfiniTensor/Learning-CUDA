// cpu_kernel_inl.h —— CPU 侧单粒子力累加的共享实现
//
// 抽成头文件是为了让三档基线共用同一段标量逻辑：
// 差别只在编译选项（向量化、多线程），而非算法。
// 若三档各写一份，比较出来的就不是"并行与向量化的收益"，而是不同的实现。
#pragma once

#include <cmath>

#include "nbody_types.h"

namespace nbody {

inline void AccumulateOne(const HostBodies &b, int i, real G, real eps2,
                          real *out_ax, real *out_ay, real *out_az) {
  const int n = static_cast<int>(b.size());
  const real xi = b.pos_x[i], yi = b.pos_y[i], zi = b.pos_z[i];
  real ax = 0, ay = 0, az = 0;
  for (int j = 0; j < n; ++j) {
    const real dx = b.pos_x[j] - xi;
    const real dy = b.pos_y[j] - yi;
    const real dz = b.pos_z[j] - zi;
    const real r2 = dx * dx + dy * dy + dz * dz + eps2;
    // 无分支跳过自身：j==i 时 dx=dy=dz=0，分子为零向量，贡献恰好为 0。
    // 这与 GPU kernel 的做法一致，保证 CPU/GPU 交叉验证比较的是
    // 同一个数学表达式（而非"CPU 跳过了自身、GPU 没跳过"这种差异）。
    const real inv_r = real(1) / std::sqrt(r2);
    const real s = G * b.mass[j] * inv_r * inv_r * inv_r;
    ax += dx * s;
    ay += dy * s;
    az += dz * s;
  }
  *out_ax = ax;
  *out_ay = ay;
  *out_az = az;
}

} // namespace nbody
