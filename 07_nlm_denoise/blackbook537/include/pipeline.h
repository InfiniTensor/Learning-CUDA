#ifndef NLM_PIPELINE_H
#define NLM_PIPELINE_H

#include <chrono>
#include <string>

#include "image_io.h"
#include "params.h"

// Host 侧计时器（chrono，供端到端/H2D/D2H 计时）
struct HostTimer {
    std::chrono::steady_clock::time_point t0;
    void Start() { t0 = std::chrono::steady_clock::now(); }
    double Ms() const {
        return std::chrono::duration<double, std::milli>(
                   std::chrono::steady_clock::now() - t0).count();
    }
};

// 性能报告（计时口径见架构设计文档 §5.3）
struct NlmPerfReport {
    float kernel_ms;          // NLM kernel 纯计算时间（GPU event）
    float e2e_ms;             // 端到端 GPU 时间（含 H2D/D2H 与转换 kernel，Host chrono）
    float h2d_ms;
    float d2h_ms;
    float throughput_mpx_s;   // 按 kernel_ms 计算的吞吐量（百万像素/秒）
    float cpu_ref_ms;         // CPU 参考耗时（未测量为 -1）
    float speedup_vs_cpu;     // cpu_ref_ms / e2e_ms（未测量为 -1）

    NlmPerfReport()
        : kernel_ms(0), e2e_ms(0), h2d_ms(0), d2h_ms(0),
          throughput_mpx_s(0), cpu_ref_ms(-1.0f), speedup_vs_cpu(-1.0f) {}
};

// GPU 降噪全流程：u8 interleaved -> planar f32 -> NLM -> 量化回 u8。
// 无状态纯函数接口（外层可直接包帧循环用于交互式场景）。
// kernel_version: 0=naive, 1=smem, 2=smem+模板展开。
bool NlmDenoiseGpu(const ImageU8& src, ImageU8* dst,
                   const NlmParams& params, int kernel_version,
                   NlmPerfReport* perf, std::string* err);

#endif // NLM_PIPELINE_H
