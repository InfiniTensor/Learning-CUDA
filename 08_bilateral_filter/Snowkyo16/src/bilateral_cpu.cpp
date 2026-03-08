#include "bilateral_filter.cuh"
#include <cmath>
#include <algorithm>

// ============================================================
// CPU 双边滤波实现
// 参考：Tomasi & Manduchi 1998 - Section 2
// https://users.soe.ucsc.edu/~manduchi/Papers/ICCV98.pdf
// ============================================================
Image bilateral_filter_cpu_v0(const Image& input, const FilterParams& params) {
    int w = input.width;
    int h = input.height;
    int c = input.channels;
    int r = params.radius;

    // 预计算分母常量，避免在内层循环中重复除法
    // 空间高斯分母：2 * σ_s²
    float spatial_denom = 2.0f * params.sigma_spatial * params.sigma_spatial;
    // 颜色高斯分母：2 * σ_c²
    float color_denom = 2.0f * params.sigma_color * params.sigma_color;

    // 输出图像，与输入同尺寸同通道数
    Image output;
    output.width = w;
    output.height = h;
    output.channels = c;
    output.data.resize(w * h * c);

    // 遍历每个像素 p(x, y)
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {

            float sum_value[3] = {0.0f, 0.0f, 0.0f};  // 每个通道的加权像素值累加器
            float sum_weight = 0.0f;  // 权重累加器（所有通道共享同一个权重）

            // 获取中心像素 p 的颜色值
            int p_idx = (y * w + x) * c;
            float p_color[3];
            for (int ch = 0; ch < c; ch++) {
                p_color[ch] = (float)input.data[p_idx + ch];
            }

            // 遍历邻域窗口
            for (int dy = -r; dy <= r; dy++) {
                for (int dx = -r; dx <= r; dx++) {
                    int qx = x + dx;
                    int qy = y + dy;

                    // 边界处理：跳过越界像素（clamp 策略）
                    if (qx < 0 || qx >= w || qy < 0 || qy >= h) {
                        continue;
                    }

                    // 获取邻域像素 q 的颜色值
                    int q_idx = (qy * w + qx) * c;
                    float q_color[3];
                    for (int ch = 0; ch < c; ch++) {
                        q_color[ch] = (float)input.data[q_idx + ch];
                    }

                    // 计算空间距离的平方
                    float spatial_dist_sq = (float)(dx * dx + dy * dy);

                    // 计算颜色差异的平方
                    // 对于多通道图像，取各通道差的平方和
                    float color_diff_sq = 0.0f;
                    for (int ch = 0; ch < c; ch++) {
                        float diff = p_color[ch] - q_color[ch];
                        color_diff_sq += diff * diff;
                    }

                    // 计算空间权重
                    float w_spatial = expf(-spatial_dist_sq / spatial_denom);

                    // 计算颜色权重
                    float w_color = expf(-color_diff_sq / color_denom);

                    // 综合权重 = 空间权重 × 颜色权重
                    float weight = w_spatial * w_color;

                    // 累加权重和加权像素值
                    sum_weight += weight;
                    for (int ch = 0; ch < c; ch++) {
                        sum_value[ch] += weight * q_color[ch];
                    }
                }
            }

            // 归一化：输出 = 加权总和 / 权重总和
            // 四舍五入并裁剪到 [0, 255] 范围
            int out_idx = (y * w + x) * c;
            for (int ch = 0; ch < c; ch++) {
                float val = sum_value[ch] / sum_weight;
                // roundf 四舍五入，clamp 到 [0, 255]
                output.data[out_idx + ch] = (uint8_t)std::min(
                    255.0f, std::max(0.0f, roundf(val)));
            }
        }
    }

    return output;
}
