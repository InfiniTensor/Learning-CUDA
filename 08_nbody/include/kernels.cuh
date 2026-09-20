// kernels.cuh —— GPU 力计算 kernel 与设备内存管理
//
// ===========================================================================
// 本文件是最终提交版：力计算只保留优化阶梯的**终点档 V3**。
//
// 为什么是 V3 而不是 V4：4090D 上的 192 组合扫参（见 n_bodyversion.md §3.5、
// §3.1）显示 V3 在 6 个测试规模里的 4 个上最快、2 个上持平，
// 只在最大规模 N=262144 时被 V4/coarsen=2 以 +1.6% 微弱反超。
// V3 的代码量不到 V4 的一半（没有模板实例化、没有跨步索引算术），
// 换来的是 1.6% —— 在提交版的取舍里，可读性与可验证性优先。
//
// 完整优化阶梯（V0 朴素 / V1 tiling / V2 float4 / V3 rsqrt / V4 粗化）
// 保留在开发版 n_version5 中，n_bodyversion.md §3.1 的逐档对比数据出自那里。
// 本文件在注释里保留了每一档的设计理由，以便读者理解 V3 为什么长这样。
// ===========================================================================
#pragma once

#include <cuda_runtime.h>

#include "nbody_types.h"

namespace nbody {

// 本版采用的 kernel 名。只用于日志与 JSON，没有派发逻辑。
constexpr const char *kKernelName = "v3_rsqrt";

// ---------------------------------------------------------------------------
// V3 = V2(float4 打包) + rsqrtf，而 V2 = V1(shared memory 分块) + float4。
// 三档叠在一起构成 V3，逐层理由如下。
//
// --- V1 层：shared memory 分块（GPU Gems 3 ch.31 式 tile）---
// 把源粒子按 blockDim.x 大小切成若干 tile。每个 tile，block 内每个线程
// 协作载入一个源粒子到 shared memory，同步，然后所有线程从 smem 读这
// blockDim.x 个源粒子做交互。于是每个源粒子只被 block 从 global memory
// 读一次，而被复用 blockDim.x 次。
//   全局访存量：O(N^2) → O(N^2 / blockDim.x)
//
// 但注意：本项目实测发现 V0 从一开始就**不是** DRAM-bound（见 n_bodyversion.md §3.6）。
// N=262144 时整个数据集仅约 4.2 MB，而 4090D 的 L2 有 72 MB —— 数据常驻
// L2，加上同一 warp 的 32 个线程在同一时刻读同一个源粒子（广播访问），
// V0 吃到的已是接近 L2 带宽而非 DRAM 带宽。所以 tiling 的实际收益远低于
// 教科书预期（实测对 V0 仅 +12%~+22%，而非常见的数倍）。
// tiling 真正拿到的是**减少 L2 访问延迟、提升寄存器/L1 复用率**。
//
// --- V2 层：位置与质量打包成 real4 ---
// 收益来自两处：
//   1. global memory：一个粒子的 4 个分量在一次 16B 对齐事务里取全，
//      而标量 SoA 需要 4 次独立的 4B 访问（分属 4 个数组，彼此相距 N*4
//      字节，几乎必然落在不同的 cache line 与不同的 DRAM page）。
//   2. shared memory：一段 real4 数组替代 4 段 real 数组。
//      bank conflict 分析：real4 = 16B，warp 内 32 个线程读同一个 k
//      （广播），广播不产生 conflict；协作载入时线程 t 写 s[t]，
//      跨 4 个 bank 步进，也无 conflict。
//
// --- V3 层：rsqrtf ---
// 唯一的改动就是把 `1/sqrt(r2)` 换成 `rsqrtf(r2)`，并加 `#pragma unroll 4`。
//
// 这是**全部五档里收益最大的一档**（实测对 V0 约 2.3–2.4×，而访存优化
// 只有 1.1–1.2×），原因：
//   - `1/sqrt(x)`：一条 IEEE 正确舍入的 sqrt（多周期迭代）+ 一条除法，
//     两者都远慢于 FMA。
//   - `rsqrtf(x)`：一条 SFU 指令，直接给出 1/sqrt(x)。SFU 吞吐低于 FMA
//     但远高于"sqrt + 除法"的组合。
// 精度代价是真实的（rsqrtf 约 2^-22 相对误差，FP32 尾数本身是 2^-24，
// 即差约 4 倍 ULP），所以必须与 V0 做能量误差对比后才能采纳，
// 不能只看快了多少 —— 实测能量误差仍在 1e-5 量级，代价可接受（见 §13.2）。
//
// `unroll 4` 的作用：rsqrtf 走 SFU，而 SFU 吞吐远低于 FMA，展开能让多条
// 独立的 rsqrt 重叠发射，掩盖 SFU 延迟。展开因子取 4 而非全展开：
// blockDim.x 是运行时量无法全展开，而 4 已足够填满 SFU 流水，
// 再大只增加寄存器压力。
// ---------------------------------------------------------------------------

// 每个 block 需要的动态 shared memory 字节数（一段 real4 数组，长度
// blockDim.x）。 bench 与 occupancy 分析都要用，故公开。
size_t SharedBytesPerBlock(int block_size);

void AllocateDevice(DeviceBodies *d, int n);
void FreeDevice(DeviceBodies *d);
void UploadBodies(DeviceBodies *d, const HostBodies &h);
void DownloadBodies(HostBodies *h, const DeviceBodies &d);
// 只取位置，用于轨迹记录（避免每次记录都把速度也拷回来）。
void DownloadPositionsInterleaved(float *dst_xyz, const DeviceBodies &d);

// 把 pos_{x,y,z} 打包成 xyz 交错写入 dst（设备内存，长度 >= 3*n），
// 在指定流上执行、不做任何同步。传输管线需要自己控制流与目标缓冲，
// 所以这一步必须与"拷回主机"解耦——DownloadPositionsInterleaved 把两件事
// 焊在一起且写死了默认流，无法用于重叠。
void LaunchPackPositions(const DeviceBodies &d, float *dst,
                         cudaStream_t stream);

// 计算全体粒子的加速度，结果写入 d.acc_*。
// 内部会先把 SoA 位置打包进 d.pos_mass（见 DeviceBodies 的注释）。
void LaunchComputeAcc(const DeviceBodies &d, real G, real softening,
                      int block_size, cudaStream_t stream = 0);

// 积分器的两个半步。拆成独立 kernel 而非融合进力计算：
// leapfrog 的 KDK 结构要求在 drift 之后重新评估力，无法融合。
// 这两个 kernel 是纯 elementwise 的，访存受限但耗时相对 O(N^2) 的力计算可忽略。
// 它们只更新标量 SoA —— 权威副本只有一份，见 DeviceBodies 的注释。
void LaunchKickDrift(const DeviceBodies &d, real dt, int block_size,
                     cudaStream_t stream = 0);
void LaunchKick(const DeviceBodies &d, real half_dt, int block_size,
                cudaStream_t stream = 0);
void LaunchEulerUpdate(const DeviceBodies &d, real dt, int block_size,
                       cudaStream_t stream = 0);

} // namespace nbody
