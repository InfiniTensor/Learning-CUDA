#pragma once

#include <string>
#include <vector>
#include <cstdint>

// 图像数据结构
// 存储图像的像素数据和基本信息
struct Image {
    int width;                      // 图像宽度（像素）
    int height;                     // 图像高度（像素）
    int channels;                   // 通道数（1=灰度，3=RGB）
    std::vector<uint8_t> data;      // 像素数据，按行主序存储（RGBRGB... 或灰度）
};

// 从 PNG/JPG 文件加载图像
// 返回的 Image.data 中像素值范围为 0-255
Image load_image(const std::string& filepath);

// 将图像保存为 PNG 文件
// data 中像素值范围应为 0-255
void save_image(const std::string& filepath, const Image& img);
