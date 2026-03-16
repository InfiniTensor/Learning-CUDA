#pragma once

#include <iostream>

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

#define RUNTIME_CHECK(call)                                                    \
  do {                                                                         \
    RUNTIME_ERR_TYPE err = call;                                               \
    if (err != RUNTIME_SUCCESS_CODE) {                                         \
      std::cerr << "Runtime error at " << __FILE__ << ":" << __LINE__ << " - " \
                << RUNTIME_GET_ERROR_STR(err) << "\n";                         \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)
