// cuda_check.cuh — CUDA 错误检查封装
//
// 项目约定：所有 CUDA Runtime API 调用必须用 CUDA_CHECK 包裹，
// 任何错误立即报告文件名/行号/错误信息，绝不吞掉错误。
#pragma once

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err__ = (call);                                                \
    if (err__ != cudaSuccess) {                                                \
      std::fprintf(stderr, "[CUDA_CHECK] %s:%d %s -> %s\n", __FILE__, __LINE__, \
                   #call, cudaGetErrorString(err__));                          \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (0)

// 在同步点检查"粘性"错误（kernel 启动配置错误、异步越界等）
#define CUDA_SYNC_CHECK()                                                      \
  do {                                                                         \
    cudaError_t err__ = cudaDeviceSynchronize();                               \
    if (err__ != cudaSuccess) {                                                \
      std::fprintf(stderr, "[CUDA_SYNC_CHECK] %s:%d -> %s\n", __FILE__,        \
                   __LINE__, cudaGetErrorString(err__));                       \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (0)
