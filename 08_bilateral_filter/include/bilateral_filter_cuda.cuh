#ifndef BILATERAL_FILTER_CUDA_CUH_
#define BILATERAL_FILTER_CUDA_CUH_

#include <cuda_runtime.h>
#include <cstdint>

// Filter modes for benchmarking different implementations
// 0 = STANDARD (shared memory + LUT, runtime radius)
// 1 = TEMPLATE (compile-time radius, full unroll)
// 2 = SEPARABLE (horizontal + vertical passes, O(r) complexity)
void set_bilateral_filter_mode(int mode);

void bilateral_filter_cuda(const float* d_input,
                           float* d_output,
                           int width,
                           int height,
                           int channels,
                           int radius,
                           float sigma_spatial,
                           float sigma_color,
                           cudaStream_t stream = 0);

void apply_bilateral_filter_cuda(const uint8_t* h_input,
                                 uint8_t* h_output,
                                 int width,
                                 int height,
                                 int channels,
                                 int radius,
                                 float sigma_spatial,
                                 float sigma_color);

#endif  // BILATERAL_FILTER_CUDA_CUH_
