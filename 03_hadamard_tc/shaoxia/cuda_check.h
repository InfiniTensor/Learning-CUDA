#pragma once

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(err)                                                   \
    do {                                                                  \
        cudaError_t e_ = (err);                                           \
        if (e_ != cudaSuccess) {                                          \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__,   \
                         __LINE__, cudaGetErrorString(e_));               \
            std::abort();                                                 \
        }                                                                 \
    } while (0)

#define CUDA_LAUNCH_CHECK()                                               \
    do {                                                                  \
        cudaError_t e_ = cudaGetLastError();                              \
        if (e_ != cudaSuccess) {                                          \
            std::fprintf(stderr, "CUDA launch error at %s:%d: %s\n",      \
                         __FILE__, __LINE__, cudaGetErrorString(e_));     \
            std::abort();                                                 \
        }                                                                 \
    } while (0)
