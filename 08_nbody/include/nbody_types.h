// nbody_types.h —— 全局精度开关与粒子数据布局
#pragma once

#include <cstddef>
#include <vector>

// 精度名称字符串，用于日志。宏不受命名空间约束，故放在 namespace 之外。
#ifdef NBODY_FP64
#define NBODY_PRECISION_NAME "fp64"
#else
#define NBODY_PRECISION_NAME "fp32"
#endif

#ifdef __CUDACC__
#include <vector_types.h>
#endif

namespace nbody {

// ---------------------------------------------------------------------------
// 精度策略
//   默认 FP32：力计算是本项目的热点，而目标卡（RTX 4090D, sm_89）的 FP64
//   吞吐只有 FP32 的 1/64，用 FP64 跑主线等于自废武功。
//   -DNBODY_FP64=ON 时切换为 FP64，仅用于小 N 验证，
//   目的是区分"算法误差"（积分器阶数）与"浮点误差"（FP32 累加）。
//   注意：能量/动量等诊断量的求和一律用 double 累加（见 diagnostics.h），
//   与这里的 real 无关，否则 FP32 的求和误差会被误读成物理误差。
// ---------------------------------------------------------------------------
#ifdef NBODY_FP64
using real = double;
#else
using real = float;
#endif

// 主机侧粒子集合。采用 SoA（Structure of Arrays）而非 AoS：
// 同一 warp 的线程访问 pos_x[i]、pos_x[i+1]... 时地址连续，满足内存合并条件。
struct HostBodies {
  std::vector<real> pos_x, pos_y, pos_z;
  std::vector<real> vel_x, vel_y, vel_z;
  std::vector<real> mass;

  std::size_t size() const { return mass.size(); }

  void resize(std::size_t n) {
    pos_x.resize(n);
    pos_y.resize(n);
    pos_z.resize(n);
    vel_x.resize(n);
    vel_y.resize(n);
    vel_z.resize(n);
    mass.resize(n);
  }
};

#ifdef __CUDACC__

// ---------------------------------------------------------------------------
// real4：位置 + 质量的打包类型（V2 档引入）。
//
// 为什么打包成 16 字节而不是四个独立标量数组：
//   力计算的内层循环每次交互都要 (x_j, y_j, z_j, m_j) 这四个值。
//   标量 SoA 下这是四次独立访存（四条 LDS/LDG 指令、四个地址计算）；
//   打包后是一条 16 字节对齐的向量访存指令。
//   16B 恰好是 GPU 一次访存事务能覆盖的最大宽度，也是 shared memory
//   的天然对齐边界，所以这一步既减少指令数也减少事务数。
// ---------------------------------------------------------------------------
#ifdef NBODY_FP64
using real4 = double4;
// make_float4 / make_double4 是 CUDA 提供的，但名字随类型变。
// 包一层同名构造，kernel 里就不必到处写 #ifdef。
__host__ __device__ __forceinline__ real4 make_real4(real x, real y, real z,
                                                     real w) {
  return make_double4(x, y, z, w);
}
#else
using real4 = float4;
__host__ __device__ __forceinline__ real4 make_real4(real x, real y, real z,
                                                     real w) {
  return make_float4(x, y, z, w);
}
#endif

// ---------------------------------------------------------------------------
// 设备侧粒子集合。
//
// 布局策略：**标量 SoA 始终是唯一的权威副本**，pos_mass 只是它的派生缓存。
//
// 为什么这样而不是"按 kernel 版本二选一"：
//   位置每步都被积分器改写。若 V0/V1 用 SoA、V2+ 用 real4 各自为政，
//   积分器就要分两条路径写不同的布局，两者的数值行为不再逐位可比，
//   优化阶梯的对照实验就不干净了（差异里混进了积分路径的不同）。
//   现在的做法是：积分器永远只更新 SoA，V2+ 在每次力评估前用一个 O(N) 的
//   打包 kernel 把 SoA 刷进 pos_mass。代价是每步一次 O(N) 拷贝，
//   相对 O(N^2) 的力计算可忽略（N=65536 时 <0.1%）；
//   换来的是"阶梯各档之间只有力计算 kernel 不同"这个干净的对照前提。
//
// 因此 pos_mass 是**只读缓存**：任何时候它的内容都由 LaunchPackPosMass
// 从 pos_* 重新生成，绝不反向写回。
//
// 速度与加速度始终是标量 SoA：它们只在 O(N) 的 elementwise kernel 里
// 被顺序访问，打包没有收益（收益来自内层 O(N^2) 循环的广播访问）。
// ---------------------------------------------------------------------------
struct DeviceBodies {
  // 权威副本。积分器只改这一份。
  real *pos_x = nullptr, *pos_y = nullptr, *pos_z = nullptr;
  real *vel_x = nullptr, *vel_y = nullptr, *vel_z = nullptr;
  real *acc_x = nullptr, *acc_y = nullptr, *acc_z = nullptr;
  real *mass = nullptr;

  // V2..V4 的派生缓存：.xyz = 位置，.w = 质量。每次力评估前重新打包。
  real4 *pos_mass = nullptr;

  int n = 0;
};

#endif // __CUDACC__

// FLOP 计数约定 —— 报告里必须写明，否则 GFLOP/s 无法与文献比较。
// 每次两体交互：3 减（dx,dy,dz）+ 5（dx*dx+dy*dy+dz*dz 的 3 乘 2 加）
// + 1 加（+eps^2）+ 1 sqrt/rsqrt + 2 乘（求 inv^3）+ 1 乘（*m_j）
// + 6（3 乘 3 加累加到 a）= 19，取整为 20。
// 与 GPU Gems 3 ch.31 的经典计数一致。
constexpr double kFlopsPerInteraction = 20.0;

} // namespace nbody
