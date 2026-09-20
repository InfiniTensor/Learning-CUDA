// cpu_reference.cpp —— 三档 CPU 基线
//
// 三档共用 cpu_kernel_inl.h 的标量内核，差别只在编译选项
// （向量化开关、多线程）。这是"三档对比反映的是并行与向量化的收益，
// 而不是三份不同实现"的前提。
#include "cpu_reference.h"

#include <chrono>
#include <cmath>
#include <vector>

#include "cpu_kernel_inl.h"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace nbody {

const char *CpuVariantName(CpuVariant v) {
  switch (v) {
  case CpuVariant::kNaive:
    return "cpu_naive";
  case CpuVariant::kScalarO3:
    return "cpu_scalar_o3";
  case CpuVariant::kOpenMP:
    return "cpu_openmp";
  }
  return "cpu_unknown";
}

int OpenMPMaxThreads() {
#ifdef _OPENMP
  return omp_get_max_threads();
#else
  return 1;
#endif
}

namespace {

inline int ClampTargets(double target_interactions, int n) {
  long long m = static_cast<long long>(target_interactions / double(n));
  if (m < 1)
    m = 1;
  if (m > n)
    m = n;
  return static_cast<int>(m);
}

} // namespace

void ComputeAccCpu(const HostBodies &bodies, real G, real softening,
                   real *acc_x, real *acc_y, real *acc_z, CpuVariant variant) {
  const int n = static_cast<int>(bodies.size());
  const real eps2 = softening * softening;

  switch (variant) {
  case CpuVariant::kNaive:
    // 转发到独立 TU，那里关掉了向量化。
    ComputeAccCpuNaive(bodies, G, softening, acc_x, acc_y, acc_z, 0, n);
    break;
  case CpuVariant::kScalarO3:
    for (int i = 0; i < n; ++i) {
      AccumulateOne(bodies, i, G, eps2, &acc_x[i], &acc_y[i], &acc_z[i]);
    }
    break;
  case CpuVariant::kOpenMP:
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < n; ++i) {
      AccumulateOne(bodies, i, G, eps2, &acc_x[i], &acc_y[i], &acc_z[i]);
    }
    break;
  }
}

double BenchmarkCpuForceEval(const HostBodies &bodies, real G, real softening,
                             CpuVariant variant, double target_interactions,
                             int *out_measured_targets) {
  const int n = static_cast<int>(bodies.size());
  const real eps2 = softening * softening;

  const int m = ClampTargets(target_interactions, n);
  if (out_measured_targets)
    *out_measured_targets = m;

  std::vector<real> ax(m), ay(m), az(m);

  // 预热一次：首次访问会有 page fault 和 cache 冷启动，
  // 把这些算进计时会高估 CPU 耗时，从而虚高 GPU 加速比。
  {
    real wx, wy, wz;
    AccumulateOne(bodies, 0, G, eps2, &wx, &wy, &wz);
    ax[0] = wx + wy + wz; // 防止整个调用被优化掉
  }

  const auto t0 = std::chrono::steady_clock::now();
  switch (variant) {
  case CpuVariant::kNaive:
    // 走独立 TU（关闭了向量化），否则这一档和 kScalarO3 会编成同一份代码。
    ComputeAccCpuNaive(bodies, G, softening, ax.data(), ay.data(), az.data(), 0,
                       m);
    break;
  case CpuVariant::kScalarO3:
    for (int i = 0; i < m; ++i) {
      AccumulateOne(bodies, i, G, eps2, &ax[i], &ay[i], &az[i]);
    }
    break;
  case CpuVariant::kOpenMP:
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < m; ++i) {
      AccumulateOne(bodies, i, G, eps2, &ax[i], &ay[i], &az[i]);
    }
    break;
  }
  const auto t1 = std::chrono::steady_clock::now();

  const double measured_ms =
      std::chrono::duration<double, std::milli>(t1 - t0).count();
  // 外推到全部 n 个目标粒子。每个目标的工作量相同且独立，故线性外推成立。
  return measured_ms * (double(n) / double(m));
}

double RunCpuSimulation(HostBodies bodies, const SimParams &params,
                        CpuVariant variant) {
  const int n = static_cast<int>(bodies.size());
  std::vector<real> ax(n), ay(n), az(n);

  const auto t0 = std::chrono::steady_clock::now();

  if (params.integrator == Integrator::kLeapfrog) {
    // KDK：初始需要一次力评估，之后每步复用上一步末算好的 a。
    ComputeAccCpu(bodies, params.G, params.softening, ax.data(), ay.data(),
                  az.data(), variant);
    const real half_dt = params.dt * real(0.5);
    for (int step = 0; step < params.num_steps; ++step) {
      for (int i = 0; i < n; ++i) {
        bodies.vel_x[i] += half_dt * ax[i];
        bodies.vel_y[i] += half_dt * ay[i];
        bodies.vel_z[i] += half_dt * az[i];
        bodies.pos_x[i] += params.dt * bodies.vel_x[i];
        bodies.pos_y[i] += params.dt * bodies.vel_y[i];
        bodies.pos_z[i] += params.dt * bodies.vel_z[i];
      }
      ComputeAccCpu(bodies, params.G, params.softening, ax.data(), ay.data(),
                    az.data(), variant);
      for (int i = 0; i < n; ++i) {
        bodies.vel_x[i] += half_dt * ax[i];
        bodies.vel_y[i] += half_dt * ay[i];
        bodies.vel_z[i] += half_dt * az[i];
      }
    }
  } else {
    for (int step = 0; step < params.num_steps; ++step) {
      ComputeAccCpu(bodies, params.G, params.softening, ax.data(), ay.data(),
                    az.data(), variant);
      for (int i = 0; i < n; ++i) {
        bodies.pos_x[i] += params.dt * bodies.vel_x[i];
        bodies.pos_y[i] += params.dt * bodies.vel_y[i];
        bodies.pos_z[i] += params.dt * bodies.vel_z[i];
        bodies.vel_x[i] += params.dt * ax[i];
        bodies.vel_y[i] += params.dt * ay[i];
        bodies.vel_z[i] += params.dt * az[i];
      }
    }
  }

  const auto t1 = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

} // namespace nbody
