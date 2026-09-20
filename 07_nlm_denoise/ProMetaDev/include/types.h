// types.h - NLM 降噪的基础类型
#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace nlm {

// 图像：0-255 的 8 位整数，尺寸 H×W、通道数 C（1 = 灰度，3 = RGB），按行交错存储
struct Image {
    int width = 0;
    int height = 0;
    int channels = 0;
    std::vector<uint8_t> data; // 长度 = height * width * channels

    bool empty() const {
        return data.empty();
    }
    size_t pixels() const {
        return (size_t)width * (size_t)height;
    }
};

// NLM 参数（对应题目给定的参数文件字段）
struct NLMParams {
    int patch_radius = 3;   // patch 半径 rp，patch 边长为 (2*rp+1)
    int search_radius = 10; // 搜索窗口半径 rs，窗口边长为 (2*rs+1)
    double h = 10.0;        // 滤波强度
    double sigma = 25.0;    // 噪声标准差估计（8 位图像，范围 0-255）

    // ---- 以下为可选的加速 / 近似开关（默认关闭，即精确算法） ----
    int search_step = 1;  // 搜索窗口采样步长；>1 表示跳过部分候选（近似）
    int patch_step = 1;   // patch 采样步长；>1 表示稀疏采样 patch（近似）
    bool use_lut = false; // 用查找表近似 exp()
    int lut_bins = 4096;  // 查找表分档数

    int patch_edge() const {
        return 2 * patch_radius + 1;
    }
    int search_edge() const {
        return 2 * search_radius + 1;
    }
    bool is_exact() const {
        return search_step == 1 && patch_step == 1 && !use_lut;
    }
};

// 性能统计
struct PerfStats {
    double load_ms = 0.0;            // 读图
    double preprocess_ms = 0.0;      // 上传 GPU / 预处理
    double denoise_ms = 0.0;         // GPU 降噪
    double download_ms = 0.0;        // 回传
    double total_ms = 0.0;           // 端到端
    double megapixels_per_sec = 0.0; // 吞吐（百万像素/秒）
    double cpu_ms = 0.0;             // CPU 参考耗时
    double opencv_ms = 0.0;          // OpenCV 参考耗时
    double speedup_vs_cpu = 0.0;
    double speedup_vs_opencv = 0.0;
    int blocks = 0;
    int threads = 0;
    double kernel_ms = 0.0;
};

// 图像质量指标
struct Quality {
    double mae = 0.0;  // 平均绝对误差
    double mse = 0.0;  // 均方误差
    double psnr = 0.0; // 峰值信噪比 (dB)
    double max_abs = 0.0;
};

} // namespace nlm
