#pragma once
#include <vector>
#include <cstdint>

struct BilateralParams {
    int   radius;
    float sigma_spatial;
    float sigma_color;
};

// CPU 参考实现，调用 OpenCV cv::bilateralFilter
// 输入/输出均为 float 数组（0-255），行主序，交错 RGB 或灰度
void bilateral_filter_cpu(
    const float* input,
    float*       output,
    int width, int height, int channels,
    const BilateralParams& params
);
