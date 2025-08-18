#pragma once

#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                       \
  {                                                                            \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - "    \
                << cudaGetErrorString(err) << "\n";                            \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  }
