// cpu_naive.cpp —— CPU 基线第 1 档：单线程标量，不向量化
//
// 单独一个编译单元的原因：这一档必须真的关掉自动向量化才有意义。
// 若和其他两档放在同一个 TU 里，编译选项是整个文件级的，
// "朴素"与"向量化"就会编译成同一份代码，
// 报告里"向量化带来 Nx 提升"的说法便无从谈起。
//
// CMakeLists 把这个文件编成独立的 nbody_cpu_naive 静态库，
// 并对它施加 -O2 -fno-tree-vectorize -fno-tree-slp-vectorize（gcc）
// 或 /O2 /Qvec-（MSVC）。
#include "cpu_kernel_inl.h"
#include "cpu_reference.h"

namespace nbody {

void ComputeAccCpuNaive(const HostBodies &bodies, real G, real softening,
                        real *acc_x, real *acc_y, real *acc_z, int begin,
                        int end) {
  const real eps2 = softening * softening;
  for (int i = begin; i < end; ++i) {
    AccumulateOne(bodies, i, G, eps2, &acc_x[i], &acc_y[i], &acc_z[i]);
  }
}

} // namespace nbody
