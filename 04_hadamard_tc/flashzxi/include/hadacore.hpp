#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <cute/tensor.hpp>
#include <cute/layout.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/atom/copy_atom.hpp>
#include <cute/algorithm/copy.hpp>
#include <cute/algorithm/clear.hpp>
#include <cute/algorithm/gemm.hpp>

#include <vector>
#include <cstdio>

#define CUDA_CHECK(call)                                                            \
    do {                                                                            \
        cudaError_t err = call;                                                     \
        if (err != cudaSuccess) {                                                   \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,        \
                    cudaGetErrorString(err));                                       \
            exit(1);                                                                \
        }                                                                           \
    } while(0)

namespace hadacore
{

void test_small();
void test_large();

} // namespace hadacore