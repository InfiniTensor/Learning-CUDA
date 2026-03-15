#pragma once

// 计算平均绝对误差 MAE（输入为 float 数组，值域 0-255）
float compute_mae(const float* a, const float* b, int n);

// 计算峰值信噪比 PSNR（最大值 255）
float compute_psnr(const float* a, const float* b, int n);
