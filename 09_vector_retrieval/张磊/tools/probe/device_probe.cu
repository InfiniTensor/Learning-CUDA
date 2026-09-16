// 设备能力探测程序：报告 §3.3 / §7.2 中平台差异结论的最小复现用例。
//
// 检查两项会直接影响向量检索正确性的设备能力：
//   1) 32 位与 64 位 atomicAdd 是否真正生效（倒排表计数/游标依赖）；
//   2) device 端 double 与 float 的长求和精度（距离累计依赖）。
//
// 编译（按平台选择其一）：
//   NVIDIA : nvcc device_probe.cu -o probe -lcublas
//   天数智芯: /usr/local/corex/bin/clang++ -x ivcore --cuda-path=/usr/local/corex \
//                -I/usr/local/corex/include device_probe.cu \
//                -L/usr/local/corex/lib64 -lcudart -lcublas -o probe
//   沐曦   : /opt/maca/tools/cu-bridge/bin/cucc device_probe.cu \
//                -I/opt/maca/tools/cu-bridge/include -I/opt/maca/include \
//                -L/opt/maca/lib -lmcblas -o probe
//
// 运行前需按平台设置运行时库路径，例如沐曦：
//   export LD_LIBRARY_PATH=/opt/maca/lib:/opt/maca/tools/cu-bridge/lib:/opt/maca/lib64

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>

__global__ void atomic32Kernel(unsigned int* out) { atomicAdd(out, 1u); }

__global__ void atomic64Kernel(unsigned long long* out) {
  atomicAdd(out, 1ull);
}

__global__ void sumKernel(const float* q, int dim, double* outD, float* outF) {
  double accd = 0.0;
  float accf = 0.0f;
  for (int d = 0; d < dim; ++d) {
    accd += static_cast<double>(q[d]);
    accf += q[d];
  }
  outD[0] = accd;
  outF[0] = accf;
}

int main() {
  const int threads = 256;
  const int dim = 128;

  unsigned int* d32 = nullptr;
  unsigned long long* d64 = nullptr;
  cudaMalloc(&d32, sizeof(unsigned int));
  cudaMalloc(&d64, sizeof(unsigned long long));
  cudaMemset(d32, 0, sizeof(unsigned int));
  cudaMemset(d64, 0, sizeof(unsigned long long));

  atomic32Kernel<<<1, threads>>>(d32);
  atomic64Kernel<<<1, threads>>>(d64);
  cudaError_t err = cudaDeviceSynchronize();

  unsigned int h32 = 0;
  unsigned long long h64 = 0;
  cudaMemcpy(&h32, d32, sizeof(h32), cudaMemcpyDeviceToHost);
  cudaMemcpy(&h64, d64, sizeof(h64), cudaMemcpyDeviceToHost);
  printf("atomicAdd uint  : %s (%u, expect %d)\n",
         h32 == static_cast<unsigned int>(threads) ? "OK" : "FAIL", h32,
         threads);
  printf("atomicAdd u64   : %s (%llu, expect %d)\n",
         h64 == static_cast<unsigned long long>(threads) ? "OK" : "FAIL", h64,
         threads);
  printf("kernel error    : %s\n", cudaGetErrorString(err));

  float host[dim];
  for (int d = 0; d < dim; ++d) host[d] = (d % 13) * 0.07f + 0.2f;
  float* dq = nullptr;
  double* dOutD = nullptr;
  float* dOutF = nullptr;
  cudaMalloc(&dq, sizeof(host));
  cudaMalloc(&dOutD, sizeof(double));
  cudaMalloc(&dOutF, sizeof(float));
  cudaMemcpy(dq, host, sizeof(host), cudaMemcpyHostToDevice);
  sumKernel<<<1, 1>>>(dq, dim, dOutD, dOutF);
  cudaDeviceSynchronize();

  double gpuDouble = 0.0;
  float gpuFloat = 0.0f;
  cudaMemcpy(&gpuDouble, dOutD, sizeof(gpuDouble), cudaMemcpyDeviceToHost);
  cudaMemcpy(&gpuFloat, dOutF, sizeof(gpuFloat), cudaMemcpyDeviceToHost);
  double cpuRef = 0.0;
  for (int d = 0; d < dim; ++d) cpuRef += static_cast<double>(host[d]);

  printf("double sum      : gpu=%.9f cpu=%.9f err=%.3e\n", gpuDouble, cpuRef,
         std::fabs(gpuDouble - cpuRef));
  printf("float  sum      : gpu=%.9f err=%.3e\n",
         static_cast<double>(gpuFloat),
         std::fabs(static_cast<double>(gpuFloat) - cpuRef));
  return 0;
}
