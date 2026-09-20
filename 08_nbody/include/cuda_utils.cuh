// cuda_utils.cuh —— CUDA 错误检查、事件计时、设备信息
#pragma once

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace nbody {

// 所有 CUDA API 调用都要过这个宏。CUDA 的错误是异步且粘性的：
// 一个错误不检查，后续所有调用都会返回同一个错误，定位现场会丢失。
#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err_ = (call);                                                 \
    if (err_ != cudaSuccess) {                                                 \
      std::fprintf(stderr, "CUDA error %s:%d: %s (%s)\n", __FILE__, __LINE__,  \
                   cudaGetErrorString(err_), #call);                           \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (0)

// kernel launch 之后调用：launch 本身是异步的，错误要靠这个同步点暴露。
// Release 构建下每步都同步会拖慢性能，所以只在验证路径和计时边界使用。
#define CUDA_CHECK_LAST() CUDA_CHECK(cudaGetLastError())

// 基于 cudaEvent 的计时器。用事件而非 CPU 侧 chrono：
// kernel 是异步提交的，CPU 侧计时会把 launch 开销和实际执行混在一起。
class CudaTimer {
public:
  CudaTimer() {
    CUDA_CHECK(cudaEventCreate(&start_));
    CUDA_CHECK(cudaEventCreate(&stop_));
  }
  ~CudaTimer() {
    cudaEventDestroy(start_);
    cudaEventDestroy(stop_);
  }
  CudaTimer(const CudaTimer &) = delete;
  CudaTimer &operator=(const CudaTimer &) = delete;

  void Start(cudaStream_t stream = 0) {
    CUDA_CHECK(cudaEventRecord(start_, stream));
  }
  void Stop(cudaStream_t stream = 0) {
    CUDA_CHECK(cudaEventRecord(stop_, stream));
  }

  // 返回毫秒。会同步等待 stop_ 事件完成。
  float ElapsedMs() {
    CUDA_CHECK(cudaEventSynchronize(stop_));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
    return ms;
  }

private:
  cudaEvent_t start_{}, stop_{};
};

struct DeviceInfo {
  std::string name;
  int major = 0, minor = 0;
  int sm_count = 0;
  size_t total_global_mem = 0;
  int max_threads_per_block = 0;
  int shared_mem_per_block = 0;
  double clock_ghz = 0.0;
  double mem_bandwidth_gb_s = 0.0;
};

DeviceInfo QueryDevice(int device = 0);
void PrintDeviceInfo(const DeviceInfo &info);

// 当前进程的显存占用（已用 / 总量），用于性能日志里的"显存占用"一项。
void QueryMemoryUsage(size_t *used_bytes, size_t *total_bytes);

} // namespace nbody
