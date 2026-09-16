#ifndef NLM_BENCHMARK_H
#define NLM_BENCHMARK_H

#include <cmath>
#include <cstdint>
#include <string>

#include "image_io.h"

// ============================================================================
// 质量指标、基准与验证共享接口
// ============================================================================

// 平均绝对误差（要求两图同尺寸同通道）
inline double ComputeMAE(const ImageU8& a, const ImageU8& b) {
    if (a.data.size() != b.data.size() || a.data.empty()) return -1.0;
    double acc = 0.0;
    for (size_t i = 0; i < a.data.size(); ++i)
        acc += std::fabs((double)a.data[i] - (double)b.data[i]);
    return acc / (double)a.data.size();
}

// 峰值信噪比（dB）；两图完全一致时返回 +inf
inline double ComputePSNR(const ImageU8& a, const ImageU8& b) {
    if (a.data.size() != b.data.size() || a.data.empty()) return -1.0;
    double acc = 0.0;
    for (size_t i = 0; i < a.data.size(); ++i) {
        const double d = (double)a.data[i] - (double)b.data[i];
        acc += d * d;
    }
    const double mse = acc / (double)a.data.size();
    if (mse == 0.0) return HUGE_VAL;
    return 20.0 * std::log10(255.0 / std::sqrt(mse));
}

// 合成测试图：平滑渐变背景 + LCG 均匀噪声（用于无外部图像时的端到端测试与基准）
inline ImageU8 MakeSyntheticImage(int w, int h, int c, float sigma, uint32_t seed) {
    ImageU8 img;
    img.width = w;
    img.height = h;
    img.channels = c;
    img.data.resize((size_t)w * h * c);
    uint32_t state = seed ? seed : 1u;
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            for (int ch = 0; ch < c; ++ch) {
                // 背景：x/y 方向渐变 + 通道偏移，制造纹理差异
                float base = (float)((x * 255 / (w > 1 ? w - 1 : 1) +
                                      y * 255 / (h > 1 ? h - 1 : 1)) / 2);
                base = base * 0.8f + 20.0f + 15.0f * ch;
                // LCG -> [0,1) -> 均匀噪声 [-sigma, +sigma]
                state = state * 1664525u + 1013904223u;
                const float u = (float)(state >> 8) / 16777216.0f;
                float v = base + (u - 0.5f) * 2.0f * sigma;
                int q = (int)(v + 0.5f);
                q = q < 0 ? 0 : (q > 255 ? 255 : q);
                img.data[((size_t)y * w + x) * c + ch] = (uint8_t)q;
            }
        }
    }
    return img;
}

// ---------------- 基准与验证子命令入口（main.cpp 调度）----------------
int RunValidate(const std::string& input, const std::string& output,
                const std::string& params_path);
int RunBenchmark(const std::string& sizes_spec, const std::string& channels_spec,
                 const std::string& param_sets_spec, const std::string& csv_path,
                 int warmup, int repeat, bool with_cpu, int cpu_repeat);
int RunMetrics(const std::string& reference_path, const std::string& test_path,
               const std::string& label, const std::string& csv_path);
int RunUnitTests();

#endif // NLM_BENCHMARK_H
