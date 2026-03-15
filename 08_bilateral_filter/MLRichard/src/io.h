#pragma once
#include <string>
#include <vector>
#include <cstdint>

struct Image {
    uint32_t width;
    uint32_t height;
    uint32_t channels;  // 1 (灰度) 或 3 (RGB)
    std::vector<uint8_t> data;  // 行主序，交错存储
};

// 读取 raw 格式文件
Image read_raw(const std::string& path);

// 写入 raw 格式文件
void write_raw(const std::string& path, const Image& img);

// 从 PNG/JPG 转换为 raw（借助 OpenCV）
Image load_image(const std::string& path);

// 将 float 数组（0-255）转换回 uint8 Image
Image float_to_image(const float* data, uint32_t w, uint32_t h, uint32_t c);

// 将 Image 转换为 float 数组（0-255）
std::vector<float> image_to_float(const Image& img);
