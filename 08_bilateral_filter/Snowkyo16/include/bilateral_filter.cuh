#pragma once

#include "image_io.h"
#include "params.h"


// V0: CPU 基线实现
Image bilateral_filter_cpu_v0(const Image& input, const FilterParams& params);

// V1: GPU Naive 实现，一个线程处理一个像素，全局内存访问
Image bilateral_filter_gpu_v1(const Image& input, const FilterParams& params);

// V2: GPU Shared Memory 优化，Tiling + Halo协作加载
Image bilateral_filter_gpu_v2(const Image& input, const FilterParams& params);

// V3: GPU 常量内存LUT + __expf快速数学 + 循环展开
void bilateral_filter_gpu_v3_init(const FilterParams& params);  // LUT上传
Image bilateral_filter_gpu_v3(const Image& input, const FilterParams& params);

// V4: Pinned Memory + CUDA Streams 流水线，预分配 Device Buffer
void bilateral_filter_gpu_v4_init(const Image& input, const FilterParams& params);  // LUT + buffer + pinned + streams 初始化
Image bilateral_filter_gpu_v4(const Image& input, const FilterParams& params);
void bilateral_filter_gpu_v4_cleanup();  // 释放 buffer、pinned memory、销毁 streams

// V5: Metax C500(MACA) 平台适配，Pinned Memory + Streams 流水线
void bilateral_filter_gpu_v5_init(const Image& input, const FilterParams& params);  // LUT + buffer + pinned + stream初始化
Image bilateral_filter_gpu_v5(const Image& input, const FilterParams& params);
void bilateral_filter_gpu_v5_cleanup();  // 释放buffer、pinned memory、销毁 streams

// V6: Moore Threads S5000(MUSA) 平台适配，Pinned Memory + Streams 流水线
void bilateral_filter_gpu_v6_init(const Image& input, const FilterParams& params);  // LUT + buffer + pinned + streams 初始化
Image bilateral_filter_gpu_v6(const Image& input, const FilterParams& params);
void bilateral_filter_gpu_v6_cleanup();  // 释放 buffer、pinned memory、销毁 streams
