// 设备 event 分阶段计时；不依赖 MUPTI 硬件计数器，不代表指令/带宽利用率分析。
#include "kernels/reduce.cuh"
#include "backends/musa/quantize.cuh"
#include "kernels/dequantize.cuh"
#include "runtime/cuda_utils.cuh"
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

template <typename F>
void measure(const char* name, F run) {
  pipeline::Events events;
  for (int j = 0; j < 3; ++j) events.measure(run);
  std::vector<float> times;
  for (int j = 0; j < 20; ++j) times.push_back(events.measure(run));
  std::sort(times.begin(), times.end());
  std::cout << "{\"stage\":\"" << name << "\",\"median_ms\":" << (times[9] + times[10]) / 2
            << ",\"p95_ms\":" << times[18] << "}\n";
}
int main() {
  using namespace pipeline;
  try {
    constexpr unsigned n = 4194304, partials = 1024;
    DeviceBuffer<float> input(n), scratch(partials), state(2), output(n);
    DeviceBuffer<std::uint8_t> data(data_size(n)), scales(n / kBlockSize);
    std::vector<float> x(n);
    std::mt19937 rng(20260909);
    std::normal_distribution<float> normal;
    for (float& v : x) v = normal(rng);
    CUDA_CHECK(cudaMemcpy(input.ptr, x.data(), n * 4, cudaMemcpyHostToDevice));
    std::cout << std::setprecision(10);
    if (nvfp4) {
      measure("maximum", [&] { maximum<<<partials, 256>>>(input.ptr, scratch.ptr, n); });
      measure("finalize_max", [&] { finalize_max<<<1, 256>>>(scratch.ptr, state.ptr, partials); });
    }
    measure("quantize_vectorized", [&] {
      musa_quantize_vectorized_kernel<<<n / 1024, 256>>>(input.ptr, data.ptr, scales.ptr, state.ptr, n);
    });
    measure("dequantize_fp32", [&] {
      dequantize_kernel<<<n / 256, 256>>>(data.ptr, scales.ptr, state.ptr, output.ptr, n, kBlockSize);
    });
  } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
