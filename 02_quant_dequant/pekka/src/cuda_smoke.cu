#include <cuda_runtime.h>

#include <cstdio>

namespace {

bool check_cuda(cudaError_t status, const char *operation) {
  if (status == cudaSuccess) {
    return true;
  }
  std::fprintf(stderr, "%s failed: %s\n", operation,
               cudaGetErrorString(status));
  return false;
}

__global__ void smoke_kernel(int *result) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    *result = 42;
  }
}

} // namespace

int main() {
  int device_count = 0;
  const cudaError_t device_status = cudaGetDeviceCount(&device_count);
  if (device_status != cudaSuccess || device_count == 0) {
    std::fprintf(stderr, "CUDA smoke test skipped: %s\n",
                 device_status == cudaSuccess
                     ? "no CUDA device is available"
                     : cudaGetErrorString(device_status));
    return 77;
  }

  cudaDeviceProp properties{};
  if (!check_cuda(cudaGetDeviceProperties(&properties, 0),
                  "cudaGetDeviceProperties")) {
    return 1;
  }

  int *device_result = nullptr;
  if (!check_cuda(cudaMalloc(&device_result, sizeof(int)), "cudaMalloc")) {
    return 1;
  }

  smoke_kernel<<<1, 1>>>(device_result);
  int host_result = 0;
  const bool copied =
      check_cuda(cudaGetLastError(), "smoke_kernel launch") &&
      check_cuda(cudaMemcpy(&host_result, device_result, sizeof(int),
                            cudaMemcpyDeviceToHost),
                 "cudaMemcpy");
  const bool freed = check_cuda(cudaFree(device_result), "cudaFree");
  if (!copied || !freed || host_result != 42) {
    return 1;
  }

  std::printf("CUDA smoke test passed on %s (compute capability %d.%d).\n",
              properties.name, properties.major, properties.minor);
  return 0;
}
