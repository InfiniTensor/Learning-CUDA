// 独立确认设备属性及次正规数运算，避免将 SDK 数值差异误判为量化索引错误。
#include <musa_runtime.h>
#include <cstdio>
#include <cmath>
__global__ void probe(float* x) {
  volatile float a = 1e-38f;
  volatile float b = __int_as_float(0x00400000);
  x[0] = a; x[1] = b;
  x[2] = scalbnf(1.0f, -127);
  x[3] = a / b;
  x[4] = __fdiv_rn(a, b);
  x[5] = a / 2688.0f;
  x[6] = __fdiv_rn(a, 2688.0f);
  x[7] = __double2float_rn(static_cast<double>(b) * 2.0);
}
int main() {
  musaDeviceProp p{};
  if (musaGetDeviceProperties(&p, 0) != musaSuccess) return 1;
  std::printf("device=%s warp=%d architecture=%d.%d\n", p.name, p.warpSize, p.major, p.minor);
  float* device = nullptr;
  if (musaMalloc(&device, 8 * sizeof(float)) != musaSuccess) return 1;
  probe<<<1, 128>>>(device);
  float x[8];
  const auto status = musaMemcpy(x, device, sizeof(x), musaMemcpyDeviceToHost);
  musaFree(device);
  if (status != musaSuccess) { std::puts(musaGetErrorString(status)); return 1; }
  for (int i = 0; i < 8; ++i) std::printf("probe[%d]=%a\n", i, double(x[i]));
}
