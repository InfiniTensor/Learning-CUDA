// 隔离数值兼容路径的吞吐；与完整 pipeline 基准分开，不作为正式性能。
#include "common/device_math.cuh"
#include "runtime/cuda_utils.cuh"
#include <algorithm>
#include <iostream>
#include <vector>

template <unsigned Mode>
__global__ void arithmetic(const float* x, const float* s, float* y, unsigned n) {
  const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  if (Mode == 0) y[i] = x[i];
  if (Mode == 1) y[i] = __fdiv_rn(x[i], s[i]);
  if (Mode == 2) y[i] = x[i] / s[i];
  if (Mode == 3) y[i] = pipeline::multiply_rn(x[i], s[i]);
  if (Mode == 4) y[i] = __fmul_rn(x[i], s[i]);
  if (Mode == 5) y[i] = __double2float_rn(static_cast<double>(x[i]) / static_cast<double>(s[i]));
}

template <unsigned Mode>
void measure(float* x, float* s, float* y, unsigned n) {
  pipeline::Events events;
  auto run = [&] { arithmetic<Mode><<<(n + 255) / 256, 256>>>(x, s, y, n); };
  for (int j = 0; j < 3; ++j) events.measure(run);
  std::vector<float> times;
  for (int j = 0; j < 20; ++j) times.push_back(events.measure(run));
  std::sort(times.begin(), times.end());
  std::cout << "mode=" << Mode << " median_ms=" << (times[9] + times[10]) / 2 << '\n';
}
int main() {
  constexpr unsigned n = 4194304;
  std::vector<float> a(n), b(n);
  for (unsigned i = 0; i < n; ++i) { a[i] = float(i % 197 - 98.0f) / 17; b[i] = .01f + (i % 31) / 32.0f; }
  pipeline::DeviceBuffer<float> x(n), s(n), y(n);
  CUDA_CHECK(cudaMemcpy(x.ptr, a.data(), n * 4, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(s.ptr, b.data(), n * 4, cudaMemcpyHostToDevice));
  measure<0>(x.ptr, s.ptr, y.ptr, n);
  measure<1>(x.ptr, s.ptr, y.ptr, n);
  measure<2>(x.ptr, s.ptr, y.ptr, n);
  measure<3>(x.ptr, s.ptr, y.ptr, n);
  measure<4>(x.ptr, s.ptr, y.ptr, n);
  measure<5>(x.ptr, s.ptr, y.ptr, n);
}
