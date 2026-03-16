#pragma once

#include <string>

// 双边滤波参数结构体
// 对应 params.txt 中的三个参数
struct FilterParams {
    int radius;            // 窗口半径，实际窗口大小为 (2*radius+1)^2
    float sigma_spatial;   // 空间域标准差（控制距离权重衰减速度）
    float sigma_color;     // 颜色域标准差（控制颜色差异权重衰减速度）
};

// 从文件中解析滤波参数
// 文件格式：每行一个 "key = value"，支持 # 注释
FilterParams load_params(const std::string& filepath);
