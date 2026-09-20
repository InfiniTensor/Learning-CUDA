// snapshot_stream.cuh —— 轨迹快照的 D2H 传输管线
#pragma once

#include <cuda_runtime.h>

#include <deque>
#include <functional>

#include "nbody_types.h"

namespace nbody {

// ---------------------------------------------------------------------------
// 为什么需要这个类：朴素记录路径是**三重串行**
//
//   打包 kernel → cudaMemcpy（同步、pageable 目标）→ memcpy 进 recorder
//
// 三个问题，逐个说：
//
//   1. **目标是 pageable 内存**。DMA 引擎只能对锁页（pinned）内存做直接传输，
//      所以驱动实际走的是"设备 → 内部 pinned 暂存区 → 用户 pageable 内存"
//      两跳，多一次 CPU 侧拷贝，而且无法与计算重叠。
//   2. **同步拷贝**。cudaMemcpy 会阻塞主机直到传输完成，此间 GPU 空闲——
//      下一步的力计算本可以立刻开始。
//   3. **拷贝与计算在同一个流上**，即便改成异步也不会重叠。
//
// 本类的做法：pinned 主机缓冲 + 独立的非阻塞流 + 事件排序 + 双缓冲。
//
//   计算流:  ...积分 → 打包到 dev_stage[s] → record(pack_done[s]) → 继续下一步
//   拷贝流:  wait(pack_done[s]) → memcpyAsync 到 host_stage[s] →
//   record(copy_done[s])
//
// 于是记录点 k 的传输与"记录点 k 到 k+1 之间的全部计算"重叠。
//
// ---------------------------------------------------------------------------
// 实测结论：**在当前负载下收益为零**（n_bodyversion.md §1.4 有说明）
//
//   扫 record_interval ∈ {1,2,5,10,50,200}，async vs sync 的 per-step 差异
//   全部在 ±2.4% 内，正负均有 —— 测量噪声。
//   原因是结构性的：拷贝量 O(N) 而计算量 O(N^2)。N=16384、
//   record_interval=1（最激进）时拷贝占比也仅 0.74%。
//
//   代码保留且默认开启（正确性已验证、无额外成本），--sync-copy 回到
//   同步路径用于 A/B 对比。"为什么没用"的量化分析本身就是结论。
//
// ---------------------------------------------------------------------------
// 两个必须做对、做错就静默出错的细节
// ---------------------------------------------------------------------------
//
// **A. 拷贝流必须用 cudaStreamNonBlocking 创建。**
//   计算仍跑在 legacy 默认流（stream 0）上。默认流与"阻塞流"之间有隐式同步：
//   若拷贝流用 cudaStreamCreate（默认即阻塞流）创建，默认流上的每个操作都会
//   与拷贝流串行化——代码看着是异步的，实测却完全没有重叠，
//   而且**不会报任何错**。只有 cudaStreamCreateWithFlags(..., NonBlocking)
//   才能让两条流真正独立，再靠事件显式建立我们需要的那一条依赖。
//
// **B. 设备侧暂存区也必须双缓冲，不只是主机侧。**
//   若只有一份 dev_stage，记录点 k+1 的打包 kernel 会在记录点 k 的拷贝
//   还在飞行时覆写它——拷回主机的就是新旧混合的数据。
//   症状是动画里偶发的粒子位置跳变，取决于时序，**不复现、不报错**。
//   所以 dev_stage 与 host_stage 都开 kSlots 份，且槽位复用前必须等它的
//   copy_done 事件。
//
// 同步模式（async=false）保留原始行为，用于 A/B 对比：
// 没有基线就无法证明这条管线有没有用。
// ---------------------------------------------------------------------------
class SnapshotStreamer {
public:
  // 数据就绪时的回调，参数是 xyz 交错、长度 3*n 的主机缓冲。
  // 用回调而非直接持有 TrajectoryRecorder：这个头文件要被 nvcc 编译，
  // 不想把纯主机侧的 IO 类型拖进 CUDA 编译单元。
  using Sink = std::function<void(const float *)>;

  SnapshotStreamer(int n, bool async, Sink sink);
  ~SnapshotStreamer();

  SnapshotStreamer(const SnapshotStreamer &) = delete;
  SnapshotStreamer &operator=(const SnapshotStreamer &) = delete;

  // 提交一次快照。async 模式下不等待传输完成即返回。
  // 内部会在复用槽位前收割该槽位上一次的结果（此时传输早已完成）。
  void Submit(const DeviceBodies &d, cudaStream_t compute_stream = 0);

  // 收割全部在飞行中的快照。主循环结束后必须调用，否则最后一两个记录点会丢。
  // **必须在计时停止之前调用**，否则 async 模式白拿一段没算钱的工作。
  void Drain();

  bool async() const { return async_; }

  // 累计的 pinned 主机内存字节数，写进性能日志用。
  size_t pinned_bytes() const;

private:
  void Harvest(int slot);

  static constexpr int kSlots = 2; // 一份在传输，一份在被打包

  int n_ = 0;
  bool async_ = false;
  Sink sink_;

  float *dev_stage_[kSlots] = {nullptr, nullptr};
  float *host_stage_[kSlots] = {nullptr, nullptr};
  cudaEvent_t pack_done_[kSlots] = {};
  cudaEvent_t copy_done_[kSlots] = {};
  cudaStream_t copy_stream_ = nullptr;

  // 待收割槽位的提交顺序。必须按提交顺序收割，否则记录点会错序——
  // 而错序的轨迹文件大小、P、R 全都对得上，只有动画会乱。
  std::deque<int> pending_;
  int next_slot_ = 0;
};

} // namespace nbody
