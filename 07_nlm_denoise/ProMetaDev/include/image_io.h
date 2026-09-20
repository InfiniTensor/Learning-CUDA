// image_io.h - 图像读写（基于 OpenCV）与质量指标
#pragma once
#include "types.h"
#include <string>

namespace nlm {

// 读图：支持 PNG / JPG / BMP 等 OpenCV 支持的格式。
// 彩色图保持 3 通道，单通道图保持 1 通道（不会被自动转成 3 通道）。
Image load_image(const std::string& path);

// 写图：按扩展名推断格式（与输入同格式即可）
void save_image(const std::string& path, const Image& img);

// 从参数文件读取 NLM 参数（key = value）
NLMParams load_params(const std::string& path);

// 质量指标：img 与 ref 逐像素比较（要求尺寸与通道一致）
Quality compare(const Image& img, const Image& ref);

} // namespace nlm
