#pragma once

#include "fuse_per_row_quant.cuh"

// Fused Hadamard transform followed by dynamic per-group quantization.
// Groups are contiguous along the last dimension. Therefore scales and,
// for asymmetric integer quantization, zero_points have shape
// [rows, cols / group_size].
//
// cols and group_size must be powers of two, and group_size must divide cols.
// INT8/INT4 support symmetric and asymmetric quantization. FP8 E4M3 supports
// scale-only quantization.
template <typename T>
void fused_hadamard_per_group_quantize(
    const T* input, void* output, float* scales,
    uint8_t* zero_points, int rows, int cols, int group_size,
    FuseQuantType quant, FuseQuantScheme scheme,
    cudaStream_t stream = 0);

