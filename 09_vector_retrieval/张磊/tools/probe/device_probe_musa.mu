// 设备能力探测（MUSA 原生 API 版本），报告 §3.3 / §7.2 结论的最小复现用例。
//
// 与 device_probe.cu 检查内容一致，但使用 MUSA 原生名称：
//   1) 32 位与 64 位 atomicAdd 是否生效；
//   2) device 端 double / float 长求和精度。
//
// 编译与运行：
//   /usr/local/musa/bin/mcc -x musa --musa-path=/usr/local/musa \
//       device_probe_musa.mu -I/usr/local/musa/include \
//       -L/usr/local/musa/lib -lmusart -lmublas -o probe_musa
//   export LD_LIBRARY_PATH=/usr/local/musa/lib:/usr/local/musa/lib64
//   ./probe_musa

#include <musa_runtime.h>

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
  musaMalloc(&d32, sizeof(unsigned int));
  musaMalloc(&d64, sizeof(unsigned long long));
  musaMemset(d32, 0, sizeof(unsigned int));
  musaMemset(d64, 0, sizeof(unsigned long long));
  atomic32Kernel<<<1, threads>>>(d32);
  atomic64Kernel<<<1, threads>>>(d64);
  musaError_t err = musaDeviceSynchronize();

  unsigned int h32 = 0;
  unsigned long long h64 = 0;
  musaMemcpy(&h32, d32, sizeof(h32), musaMemcpyDeviceToHost);
  musaMemcpy(&h64, d64, sizeof(h64), musaMemcpyDeviceToHost);
  printf("atomicAdd uint  : %s (%u, expect %d)\n",
         h32 == (unsigned)threads ? "OK" : "FAIL", h32, threads);
  printf("atomicAdd u64   : %s (%llu, expect %d)\n",
         h64 == (unsigned long long)threads ? "OK" : "FAIL", h64, threads);
  printf("kernel error    : %s\n", musaGetErrorString(err));

  float host[dim];
  for (int d = 0; d < dim; ++d) host[d] = (d % 13) * 0.07f + 0.2f;
  float* dq = nullptr;
  double* dOutD = nullptr;
  float* dOutF = nullptr;
  musaMalloc(&dq, sizeof(host));
  musaMalloc(&dOutD, sizeof(double));
  musaMalloc(&dOutF, sizeof(float));
  musaMemcpy(dq, host, sizeof(host), musaMemcpyHostToDevice);
  sumKernel<<<1, 1>>>(dq, dim, dOutD, dOutF);
  musaDeviceSynchronize();

  double gpuDouble = 0.0;
  float gpuFloat = 0.0f;
  musaMemcpy(&gpuDouble, dOutD, sizeof(gpuDouble), musaMemcpyDeviceToHost);
  musaMemcpy(&gpuFloat, dOutF, sizeof(gpuFloat), musaMemcpyDeviceToHost);
  double cpuRef = 0.0;
  for (int d = 0; d < dim; ++d) cpuRef += static_cast<double>(host[d]);

  printf("double sum      : gpu=%.9f cpu=%.9f err=%.3e\n", gpuDouble, cpuRef,
         std::fabs(gpuDouble - cpuRef));
  printf("float  sum      : gpu=%.9f err=%.3e\n",
         static_cast<double>(gpuFloat),
         std::fabs(static_cast<double>(gpuFloat) - cpuRef));
  return 0;
}
