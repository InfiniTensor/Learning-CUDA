// snapshot_stream.cu —— 轨迹快照 D2H 管线的实现
#include "snapshot_stream.cuh"

#include <cstdlib>
#include <stdexcept>

#include "cuda_utils.cuh"
#include "kernels.cuh"

namespace nbody {

SnapshotStreamer::SnapshotStreamer(int n, bool async, Sink sink)
    : n_(n), async_(async), sink_(std::move(sink)) {
  const size_t bytes = static_cast<size_t>(n) * 3 * sizeof(float);

  if (!async_) {
    // ------------------------------------------------------------------
    // 同步基线：**刻意**复刻朴素记录路径的行为——pageable 主机内存 +
    // 阻塞式 cudaMemcpy。
    //
    // 这里不用 pinned 内存不是偷懒，而是为了让 A/B 对比有意义：
    // 这条管线的收益由"pinned + 异步 + 独立流"三者共同产生，
    // 若基线也用 pinned，测出的就只是"异步流的收益"，
    // 而报告要回答的是"整条记录路径改造值不值得"。
    // ------------------------------------------------------------------
    CUDA_CHECK(cudaMalloc(&dev_stage_[0], bytes));
    host_stage_[0] = static_cast<float *>(std::malloc(bytes));
    if (!host_stage_[0])
      throw std::runtime_error("SnapshotStreamer: out of host memory");
    return;
  }

  // 拷贝流必须是 non-blocking：计算跑在 legacy 默认流上，
  // 而默认流与普通（阻塞）流之间存在隐式同步，会让"异步"完全失效且不报错。
  CUDA_CHECK(cudaStreamCreateWithFlags(&copy_stream_, cudaStreamNonBlocking));

  for (int s = 0; s < kSlots; ++s) {
    CUDA_CHECK(cudaMalloc(&dev_stage_[s], bytes));
    // cudaHostAllocDefault 即可；我们不需要 mapped/portable。
    CUDA_CHECK(cudaHostAlloc(&host_stage_[s], bytes, cudaHostAllocDefault));
    // DisableTiming：这些事件只用于排序，不用于计时。
    // 带计时的事件会强制额外的同步开销。
    CUDA_CHECK(
        cudaEventCreateWithFlags(&pack_done_[s], cudaEventDisableTiming));
    CUDA_CHECK(
        cudaEventCreateWithFlags(&copy_done_[s], cudaEventDisableTiming));
  }
}

SnapshotStreamer::~SnapshotStreamer() {
  // 不在析构里调 sink_：回调可能抛异常，而析构中抛异常会直接 terminate。
  // 主循环结束后必须显式调用 Drain()——漏掉会丢最后一两个记录点。
  if (copy_stream_) {
    cudaStreamSynchronize(copy_stream_);
    cudaStreamDestroy(copy_stream_);
  }
  for (int s = 0; s < kSlots; ++s) {
    if (dev_stage_[s])
      cudaFree(dev_stage_[s]);
    if (host_stage_[s]) {
      if (async_) {
        cudaFreeHost(host_stage_[s]);
      } else {
        std::free(host_stage_[s]);
      }
    }
    if (pack_done_[s])
      cudaEventDestroy(pack_done_[s]);
    if (copy_done_[s])
      cudaEventDestroy(copy_done_[s]);
  }
}

size_t SnapshotStreamer::pinned_bytes() const {
  if (!async_)
    return 0;
  return static_cast<size_t>(n_) * 3 * sizeof(float) * kSlots;
}

void SnapshotStreamer::Harvest(int slot) {
  // 等这个槽位的拷贝真正落地，再把数据交给 sink。
  CUDA_CHECK(cudaEventSynchronize(copy_done_[slot]));
  sink_(host_stage_[slot]);
}

void SnapshotStreamer::Submit(const DeviceBodies &d,
                              cudaStream_t compute_stream) {
  const size_t bytes = static_cast<size_t>(n_) * 3 * sizeof(float);

  if (!async_) {
    // 同步基线路径：打包 → 阻塞拷贝 → 立刻交付。
    LaunchPackPositions(d, dev_stage_[0], compute_stream);
    CUDA_CHECK(cudaMemcpy(host_stage_[0], dev_stage_[0], bytes,
                          cudaMemcpyDeviceToHost));
    sink_(host_stage_[0]);
    return;
  }

  const int slot = next_slot_;

  // 复用槽位前，必须先把它上一轮的数据收走。
  // 按**提交顺序**收割：pending_ 是队列，front 是最早提交的那个。
  // 若乱序收割，轨迹的记录点顺序就错了——而错序的文件大小、P、R 全对，
  // 程序不报错，只有动画会乱。
  while (!pending_.empty()) {
    const int front = pending_.front();
    pending_.pop_front();
    Harvest(front);
    if (front == slot)
      break;
  }

  // 计算流：打包到该槽位的设备暂存区，并记录"打包完成"。
  LaunchPackPositions(d, dev_stage_[slot], compute_stream);
  CUDA_CHECK(cudaEventRecord(pack_done_[slot], compute_stream));

  // 拷贝流：等打包完成 → 异步 D2H → 记录"拷贝完成"。
  // 这条 wait 是两条流之间**唯一**的依赖，也正是我们想要的那一条：
  // 拷贝只需等自己那一份数据打包好，不需要等后续的力计算。
  CUDA_CHECK(cudaStreamWaitEvent(copy_stream_, pack_done_[slot], 0));
  CUDA_CHECK(cudaMemcpyAsync(host_stage_[slot], dev_stage_[slot], bytes,
                             cudaMemcpyDeviceToHost, copy_stream_));
  CUDA_CHECK(cudaEventRecord(copy_done_[slot], copy_stream_));

  pending_.push_back(slot);
  next_slot_ = (next_slot_ + 1) % kSlots;
}

void SnapshotStreamer::Drain() {
  while (!pending_.empty()) {
    const int slot = pending_.front();
    pending_.pop_front();
    Harvest(slot);
  }
}

} // namespace nbody
