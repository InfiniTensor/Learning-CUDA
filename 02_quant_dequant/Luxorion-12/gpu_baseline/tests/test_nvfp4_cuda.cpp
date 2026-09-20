// 逐层比较NVFP4 CPU与CUDA：global、局部scale、packed data和恢复结果。
#include "nvfp4_cuda.h"

#include <cuda_runtime_api.h>

#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
unsigned checks = 0;

void check(bool condition, const std::string& message) {
  ++checks;
  if (!condition) throw std::runtime_error(message);
}

template <typename Exception, typename Function>
void expect_error(Function function, const std::string& message) {
  bool rejected = false;
  try { function(); } catch (const Exception&) { rejected = true; }
  check(rejected, message);
}

void compare_cpu_cuda(const std::vector<float>& input) {
  const auto cpu_q = luxorion::nvfp4_quantize_cpu(input);
  const auto gpu_q = luxorion::nvfp4_quantize_cuda(input);
  check(cpu_q.element_count == gpu_q.element_count, "element count differs");
  check(cpu_q.global_scale == gpu_q.global_scale, "global scale differs");
  check(cpu_q.block_scales == gpu_q.block_scales, "block scale codes differ");
  check(cpu_q.packed_data == gpu_q.packed_data, "packed data differs");

  const auto cpu = luxorion::nvfp4_dequantize_cpu(cpu_q);
  const auto gpu = luxorion::nvfp4_dequantize_cuda(gpu_q);
  check(cpu.size() == gpu.size(), "restored lengths differ");
  for (std::size_t i = 0; i < cpu.size(); ++i) {
    check(cpu[i] == gpu[i], "restored value differs at " + std::to_string(i));
    check(std::signbit(cpu[i]) == std::signbit(gpu[i]),
          "restored zero sign differs at " + std::to_string(i));
  }
}

void test_manual_and_boundaries() {
  std::vector<float> manual(32, 0.0F);
  manual[0] = 2688.0F;
  manual[16] = 6.0F;
  manual[17] = -3.0F;
  manual[18] = 2.5F;
  manual[19] = 3.5F;
  compare_cpu_cuda(manual);
  manual.resize(17);
  compare_cpu_cuda(manual);
  compare_cpu_cuda({0.0F, -0.0F});
}

void test_lengths_and_distributions() {
  std::mt19937 engine(1234);
  std::uniform_real_distribution<float> uniform(-6000.0F, 6000.0F);
  std::normal_distribution<float> normal(0.0F, 20.0F);
  for (std::size_t n : {1U, 2U, 15U, 16U, 17U, 31U, 32U, 33U,
                        255U, 256U, 257U, 1057U}) {
    for (unsigned distribution = 0; distribution < 3; ++distribution) {
      std::vector<float> input(n);
      for (float& value : input) {
        value = distribution == 0 ? 0.0F
              : distribution == 1 ? uniform(engine) : normal(engine);
      }
      if (n > 1) input[1] = -0.0F;
      if (distribution == 2) input[n / 2] = 6000.0F;
      compare_cpu_cuda(input);
    }
  }
}

void test_invalid_input() {
  expect_error<std::invalid_argument>(
      [] { luxorion::nvfp4_quantize_cuda({}); }, "empty input");
  for (float value : {std::numeric_limits<float>::quiet_NaN(),
                      std::numeric_limits<float>::infinity(),
                      -std::numeric_limits<float>::infinity()}) {
    expect_error<std::invalid_argument>(
        [value] { luxorion::nvfp4_quantize_cuda({1.0F, value}); },
        "nonfinite input");
  }
  expect_error<std::underflow_error>([] {
    luxorion::nvfp4_quantize_cuda({std::numeric_limits<float>::denorm_min()});
  }, "global scale underflow");
}
}  // namespace

int main() {
  int device_count = 0;
  const cudaError_t status = cudaGetDeviceCount(&device_count);
  if (status == cudaErrorNoDevice || device_count == 0) {
    std::cout << "跳过 NVFP4 CUDA 测试：没有可用的 CUDA GPU。\n";
    return 77;
  }
  if (status != cudaSuccess) {
    std::cerr << "CUDA 初始化失败：" << cudaGetErrorString(status) << '\n';
    return 1;
  }
  try {
    test_manual_and_boundaries();
    test_lengths_and_distributions();
    test_invalid_input();
    std::cout << "NVFP4 CUDA 量化/反量化：" << checks << " 项检查通过\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "NVFP4 CUDA 测试失败：" << error.what() << '\n';
    return 1;
  }
}
