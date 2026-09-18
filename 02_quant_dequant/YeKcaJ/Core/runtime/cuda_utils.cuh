#pragma once

// CUDA 错误检查、显存 RAII 和 event 计时；不依赖具体算子。
#include <cuda_runtime.h>
#include <cstddef>
#include <stdexcept>
#include <string>

namespace pipeline {

// 将 CUDA API 错误转成带源码位置的异常，供 main 统一处理。
inline void check_cuda(cudaError_t status, const char* expression, const char* file, int line) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(expression) + " failed at " + file + ":" +
                             std::to_string(line) + ": " + cudaGetErrorString(status));
  }
}

#define CUDA_CHECK(expression) ::pipeline::check_cuda((expression), #expression, __FILE__, __LINE__)

// 对象销毁时释放显存；禁用复制，避免同一指针被重复释放。
template <typename T>
struct DeviceBuffer {
  T* ptr = nullptr;

  explicit DeviceBuffer(std::size_t n) {
    if (n) CUDA_CHECK(cudaMalloc(&ptr, n * sizeof(T)));
  }

  ~DeviceBuffer() {
    if (ptr) cudaFree(ptr);
  }

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

struct Events {
  cudaEvent_t start{}, stop{};

  Events() {
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
  }

  ~Events() {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
  }

  // 统计 f() 中提交的 GPU 工作；量化计时包含整个 kernel 序列。
  template <typename F>
  float measure(F&& f) {
    CUDA_CHECK(cudaEventRecord(start));
    f();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    return ms;
  }
};

}  // namespace pipeline
