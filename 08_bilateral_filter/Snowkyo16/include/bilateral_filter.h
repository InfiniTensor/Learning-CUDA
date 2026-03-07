#pragma once

#include "image_io.h"
#include "params.h"


// CPU 双边滤波
// 输入：原始图像像素（uint8_t, 0-255）
// 输出：滤波后图像像素（uint8_t, 0-255）
// 内部计算使用 float，最后四舍五入回 uint8_t
Image bilateral_filter_cpu(const Image& input, const FilterParams& params);
