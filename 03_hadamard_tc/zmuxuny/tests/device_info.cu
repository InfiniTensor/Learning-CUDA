#include <cstdio>
#include <cuda_runtime.h>
int main() {
  cudaDeviceProp p{};
  auto e = cudaGetDeviceProperties(&p, 0);
  if (e != cudaSuccess)
    return 1;
  int driver = 0, runtime = 0;
  cudaDriverGetVersion(&driver);
  cudaRuntimeGetVersion(&runtime);
  printf("{\"name\":\"%s\",\"global_memory_bytes\":%zu,\"multiprocessors\":%d,\"warp_"
         "size\":%d,\"max_threads_per_block\":%d,\"shared_memory_per_block\":%zu,"
         "\"driver_api_version\":%d,\"runtime_api_version\":%d}\n",
         p.name, p.totalGlobalMem, p.multiProcessorCount, p.warpSize,
         p.maxThreadsPerBlock, p.sharedMemPerBlock, driver, runtime);
}
