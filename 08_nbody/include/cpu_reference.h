// cpu_reference.h —— CPU 基线实现（用于正确性交叉验证与加速比分母）
#pragma once

#include "nbody_types.h"
#include "sim_params.h"

namespace nbody {

// 三档 CPU 基线。作业要求"超越同级别实现"，
// 所以报告里用来算加速比的分母必须是 kOpenMP 这一档，
// 只报对 kNaive 的加速比会被认为是稻草人对比。
enum class CpuVariant {
  kNaive = 0, // 单线程标量，无优化提示
  kScalarO3, // 单线程，交给编译器自动向量化（-O3 -march=native）
  kOpenMP,   // 多线程 + 向量化 ← 加速比的分母
};

const char *CpuVariantName(CpuVariant v);

// 第 1 档（kNaive）实现在独立编译单元 cpu_naive.cpp 里，
// 以便对它单独施加"关闭向量化"的编译选项。
// 若与其他两档同处一个 TU，编译选项是文件级的，三档会编成同一份代码，
// "向量化带来多少提升"就无从测量。
void ComputeAccCpuNaive(const HostBodies &bodies, real G, real softening,
                        real *acc_x, real *acc_y, real *acc_z, int begin,
                        int end);

// 计算全体加速度，结果写入 acc_*（长度 n 的数组，由调用方分配）。
// 注意 kOpenMP 这一档同时被 main.cu 的 CrossCheckAcc 当作 GPU 加速度的
// 比对基准 —— 它是"标准答案"，所以编译时刻意不加 fast-math
// （见 CMakeLists.txt 里 nbody_apply_native_flags 的注释）。
void ComputeAccCpu(const HostBodies &bodies, real G, real softening,
                   real *acc_x, real *acc_y, real *acc_z, CpuVariant variant);

// ---------------------------------------------------------------------------
// CPU 力评估的计时。
//
// 不能直接跑满 num_steps：N=262144 时单线程朴素版一次力评估要 ~10 分钟，
// 跑 1000 步是不可能的。但也不能因此换一个更快的"简化版"来计时——
// 那就不是同一个实现了，加速比会失去意义。
//
// 做法：只对前 M 个目标粒子算加速度，再按 N/M 外推。
// 每个目标粒子的工作量完全相同且相互独立（内层都要遍历全部 N 个源粒子），
// 所以这个外推在算术上是精确的，只有 cache 行为上的细微差别。
// 报告里要写明用了外推以及外推的依据。
//
// target_interactions 控制测量规模：M = clamp(target_interactions / N, 1, N)。
// 返回一次**完整**力评估（全部 N 个目标）的外推耗时，单位毫秒。
// out_measured_targets 回传实际测了多少个目标粒子，供报告标注。
// ---------------------------------------------------------------------------
double BenchmarkCpuForceEval(const HostBodies &bodies, real G, real softening,
                             CpuVariant variant, double target_interactions,
                             int *out_measured_targets);

// 完整跑 num_steps 步，用于小 N 的端到端计时与正确性对照。
// 不记录轨迹（避免 IO 干扰计时）。返回总耗时（毫秒）。
double RunCpuSimulation(HostBodies bodies, const SimParams &params,
                        CpuVariant variant);

int OpenMPMaxThreads();

} // namespace nbody
