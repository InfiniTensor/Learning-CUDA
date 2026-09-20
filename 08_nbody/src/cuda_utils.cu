// cuda_utils.cu —— 设备查询实现
#include "cuda_utils.cuh"

namespace nbody {

DeviceInfo QueryDevice(int device) {
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  DeviceInfo info;
  info.name = prop.name;
  info.major = prop.major;
  info.minor = prop.minor;
  info.sm_count = prop.multiProcessorCount;
  info.total_global_mem = prop.totalGlobalMem;
  info.max_threads_per_block = prop.maxThreadsPerBlock;
  info.shared_mem_per_block = static_cast<int>(prop.sharedMemPerBlock);
  info.clock_ghz = prop.clockRate / 1.0e6; // clockRate 单位是 kHz
  // 理论带宽 = 显存位宽/8 * 显存时钟 * 2（DDR 双沿传输）
  info.mem_bandwidth_gb_s =
      prop.memoryBusWidth / 8.0 * prop.memoryClockRate * 2.0 / 1.0e6;
  return info;
}

void PrintDeviceInfo(const DeviceInfo &info) {
  std::printf("device        : %s (sm_%d%d, %d SMs)\n", info.name.c_str(),
              info.major, info.minor, info.sm_count);
  std::printf("global memory : %.2f GiB\n",
              info.total_global_mem / 1024.0 / 1024.0 / 1024.0);
  std::printf("clock / bw    : %.2f GHz / %.1f GB/s (theoretical)\n",
              info.clock_ghz, info.mem_bandwidth_gb_s);
}

void QueryMemoryUsage(size_t *used_bytes, size_t *total_bytes) {
  size_t free_bytes = 0, total = 0;
  CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total));
  *used_bytes = total - free_bytes;
  *total_bytes = total;
}

} // namespace nbody
