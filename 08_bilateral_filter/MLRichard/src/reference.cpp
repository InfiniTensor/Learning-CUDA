#include "reference.h"
#include <opencv2/opencv.hpp>
#include <stdexcept>
#include <algorithm>

void bilateral_filter_cpu(
    const float* input, float* output,
    int w, int h, int c,
    const BilateralParams& p)
{
    // 转换为 OpenCV Mat（uint8）
    cv::Mat src(h, w, c == 3 ? CV_8UC3 : CV_8UC1);
    for (int i = 0; i < h * w * c; i++)
        src.data[i] = static_cast<uint8_t>(std::min(std::max(input[i], 0.f), 255.f));

    // OpenCV bilateralFilter 要求 BGR，我们存的是 RGB，需要转换
    if (c == 3) cv::cvtColor(src, src, cv::COLOR_RGB2BGR);

    cv::Mat dst;
    // d = 2*radius+1，sigma_color 和 sigma_space 与我们的参数对应
    cv::bilateralFilter(src, dst, 2 * p.radius + 1, p.sigma_color, p.sigma_spatial);

    if (c == 3) cv::cvtColor(dst, dst, cv::COLOR_BGR2RGB);

    for (int i = 0; i < h * w * c; i++)
        output[i] = static_cast<float>(dst.data[i]);
}
