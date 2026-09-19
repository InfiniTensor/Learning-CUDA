#pragma once

#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cstdint>

enum class FuseQuantType : int {
    Int8 = 0,
    Fp8E4M3 = 1,
    Int4 = 2,
};

enum class FuseQuantScheme : int {
    Symmetric = 0,
    Asymmetric = 1,
    ScaleOnly = 2,
};

enum class FuseKernelPolicy : int {
    Auto = 0,
    Vec = 1,
    MultiWarp = 2,
};

// Generic entry point. INT8/INT4 accept symmetric or asymmetric; FP8 accepts
// scale-only. zero_points is required for asymmetric integer quantization and
// may be null for symmetric/scale-only modes.
template <typename T>
void fused_hadamard_per_row_quantize(
    const T* input, void* output, float* output_scale,
    uint8_t* zero_points, int rows, int cols,
    FuseQuantType quant, FuseQuantScheme scheme,
    cudaStream_t stream = 0);

// Tuning entry point. Vec/MultiWarp force the corresponding implementation
// for cols in [256, 8192]; other dimensions keep their normal implementation.
template <typename T>
void fused_hadamard_per_row_quantize_with_policy(
    const T* input, void* output, float* output_scale,
    uint8_t* zero_points, int rows, int cols,
    FuseQuantType quant, FuseQuantScheme scheme,
    FuseKernelPolicy policy, cudaStream_t stream = 0);

template <typename T>
void fused_hadamard_per_row_quant(
    const T* input, int8_t* output, float* output_scale,
    int rows, int cols, cudaStream_t stream = 0);

template <typename T>
void fused_hadamard_per_row_quant_fp8(
    const T* input, __nv_fp8_storage_t* output, float* output_scale,
    int rows, int cols, cudaStream_t stream = 0);
