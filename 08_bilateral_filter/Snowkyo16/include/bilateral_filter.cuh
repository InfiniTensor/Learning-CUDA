#pragma once

#include "image_io.h"
#include "params.h"


// V0: CPU 基线实现
Image bilateral_filter_cpu_v0(const Image& input, const FilterParams& params);

// V1: GPU Naive 实现，一个线程处理一个像素，全局内存访问
Image bilateral_filter_gpu_v1(const Image& input, const FilterParams& params);

// V2: GPU Shared Memory 优化，Tiling + Halo协作加载
Image bilateral_filter_gpu_v2(const Image& input, const FilterParams& params);