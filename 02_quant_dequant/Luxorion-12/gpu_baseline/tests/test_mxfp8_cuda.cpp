// 测试模块用途：逐层比较MXFP8 CPU和CUDA量化/反量化。
// 依次检查scale编码、元素编码和恢复结果，定位错误所在阶段。
#include "mxfp8_cuda.h"

#include <cuda_runtime_api.h>

#include <cmath>
#include <cstdint>
#include <iostream>
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

void compare_cpu_cuda(const std::vector<float>& input) {
  const auto cpu_quantized = luxorion::mxfp8_quantize_cpu(input);
  const auto gpu_quantized = luxorion::mxfp8_quantize_cuda(input);
  check(cpu_quantized.scales == gpu_quantized.scales,
        "CPU/GPU scale encodings differ");
  check(cpu_quantized.data == gpu_quantized.data,
        "CPU/GPU element encodings differ");

  const auto cpu = luxorion::mxfp8_dequantize_cpu(cpu_quantized);
  const auto gpu = luxorion::mxfp8_dequantize_cuda(gpu_quantized);
  check(cpu.size() == gpu.size(), "CPU/GPU output lengths differ");
  for (std::size_t i = 0; i < cpu.size(); ++i) {
    check(cpu[i] == gpu[i], "CPU/GPU value differs at element " + std::to_string(i));
    check(std::signbit(cpu[i]) == std::signbit(gpu[i]),
          "CPU/GPU zero sign differs at element " + std::to_string(i));
  }
}

template <typename Exception, typename Function>
void expect_error(Function function, const std::string& message) {
  bool rejected = false;
  try { function(); } catch (const Exception&) { rejected = true; }
  check(rejected, message);
}

void test_manual_and_boundaries() {
  compare_cpu_cuda({1.0F, -1.1F, 15.5F, 260.0F, -0.0F});
  compare_cpu_cuda({600.0F});
  compare_cpu_cuda({2.0e-40F, -2.0e-40F});

  std::vector<float> cross_block(33, 1.0F);
  cross_block[0] = -448.0F;
  cross_block[31] = 256.0F;
  cross_block[32] = 600.0F;
  compare_cpu_cuda(cross_block);
}

void test_lengths_and_random_values() {
  std::mt19937 engine(1234);
  std::uniform_real_distribution<float> distribution(-6000.0F, 6000.0F);
  for (std::size_t n : {1U, 31U, 32U, 33U, 65U, 255U, 256U, 257U, 1057U}) {
    std::vector<float> input(n);
    for (float& value : input) value = distribution(engine);
    if (n > 1) input[1] = -0.0F;
    compare_cpu_cuda(input);
  }
}

void test_invalid_input() {
  expect_error<std::invalid_argument>(
      [] { luxorion::mxfp8_quantize_cuda({}); }, "empty CUDA quantization input");
  for (float value : {NAN, INFINITY, -INFINITY}) {
    expect_error<std::invalid_argument>(
        [value] { luxorion::mxfp8_quantize_cuda({1.0F, value}); },
        "nonfinite CUDA quantization input");
  }
}
}  // namespace

int main() {
  int device_count = 0;
  const cudaError_t status = cudaGetDeviceCount(&device_count);
  if (status == cudaErrorNoDevice || device_count == 0) {
    std::cout << "跳过 MXFP8 CUDA 测试：没有可用的 CUDA GPU。\n";
    return 77;
  }
  if (status != cudaSuccess) {
    std::cerr << "CUDA 初始化失败：" << cudaGetErrorString(status) << '\n';
    return 1;
  }

  try {
    test_manual_and_boundaries();
    test_lengths_and_random_values();
    test_invalid_input();
    std::cout << "MXFP8 CUDA 量化/反量化：" << checks << " 项检查通过\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "MXFP8 CUDA 测试失败：" << error.what() << '\n';
    return 1;
  }
}
