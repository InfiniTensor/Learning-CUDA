#pragma once
#include "reference.h"  // BilateralParams

// Phase 1：朴素 kernel，每线程处理一个像素，直接读取 Global Memory
void launch_bilateral_naive(
    const float* d_input,
    float*       d_output,
    int width, int height, int channels,
    const BilateralParams& params
);

// Phase 2：Shared Memory 优化 kernel
void launch_bilateral_shared(
    const float* d_input,
    float*       d_output,
    int width, int height, int channels,
    const BilateralParams& params
);

// Phase 3：自适应半径 kernel
void launch_bilateral_adaptive(
    const float* d_input,
    float*       d_output,
    int width, int height, int channels,
    const BilateralParams& params
);
