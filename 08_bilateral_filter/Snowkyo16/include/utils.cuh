#pragma once

#include <iostream>
#include <cstdlib>

// 多平台运行时 API 适配
// 根据 Makefile 中 PLATFORM_DEFINE 宏选择对应平台的头文件和类型

#if defined(PLATFORM_NVIDIA) || defined(PLATFORM_ILUVATAR)
#include <cuda_runtime.h>
#define RUNTIME_ERR_TYPE cudaError_t
#define RUNTIME_SUCCESS_CODE cudaSuccess
#define RUNTIME_GET_ERROR_STR cudaGetErrorString

#elif defined(PLATFORM_MOORE)
#include <musa_runtime.h>
#define RUNTIME_ERR_TYPE musaError_t
#define RUNTIME_SUCCESS_CODE musaSuccess
#define RUNTIME_GET_ERROR_STR musaGetErrorString

#elif defined(PLATFORM_METAX)
#include <mcr/mc_runtime.h>
#define RUNTIME_ERR_TYPE mcError_t
#define RUNTIME_SUCCESS_CODE mcSuccess
#define RUNTIME_GET_ERROR_STR mcGetErrorString

#else
#error "Unknown PLATFORM for RUNTIME_CHECK"
#endif


// RUNTIME_CHECK 宏：检查 GPU API 调用是否成功
// 用法：RUNTIME_CHECK(cudaMalloc(&ptr, size));
// 如果调用失败，打印错误信息并终止程序
#define RUNTIME_CHECK(call)                                                    \
  do {                                                                         \
    RUNTIME_ERR_TYPE err = call;                                               \
    if (err != RUNTIME_SUCCESS_CODE) {                                         \
      std::cerr << "Runtime error at " << __FILE__ << ":" << __LINE__ << " - " \
                << RUNTIME_GET_ERROR_STR(err) << "\n";                         \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)
