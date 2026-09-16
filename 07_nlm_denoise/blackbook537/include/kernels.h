#ifndef NLM_KERNELS_H
#define NLM_KERNELS_H

#include <cstddef>
#include <cstdint>
#include <string>

#include "params.h"

// ============================================================================
// 设备侧参数（Host 预计算，避免 kernel 内重复计算）
// 权重公式：w(p,q) = exp(-max(dist(P_p,P_q) - 2*sigma^2*N_patch, 0) / h^2)
//   dist 为两 patch 的 L2 距离之和（RGB 时跨通道求和，三通道共享权重）
//   N_patch = (2*pr+1)^2 * channels
// ============================================================================
struct NlmParamsDev {
    int   pr;             // patch 半径
    int   sr;             // 搜索窗半径
    float inv_h2;         // 1 / h^2
    float two_sigma2_n;   // 2 * sigma^2 * N_patch
};

inline NlmParamsDev MakeDevParams(const NlmParams& p, int channels) {
    NlmParamsDev d;
    d.pr = p.patch_radius;
    d.sr = p.search_radius;
    d.inv_h2 = 1.0f / (p.h * p.h);
    const float n_patch =
        (float)((2 * p.patch_radius + 1) * (2 * p.patch_radius + 1) * channels);
    d.two_sigma2_n = 2.0f * p.sigma * p.sigma * n_patch;
    return d;
}

// ---------------- 设备资源管理（pipeline 经此持有设备内存，无需包含 GPU 头文件）----------------
bool GpuAlloc(void** ptr, size_t bytes, std::string* err);
bool GpuFree(void* ptr, std::string* err);
bool GpuCopyH2D(void* dst, const void* src, size_t bytes, std::string* err);
bool GpuCopyD2H(void* dst, const void* src, size_t bytes, std::string* err);

// ---------------- 设备端 kernel 调度 ----------------
// 数据布局约定：planar f32，第 ch 通道位于 d_ptr + ch*w*h
bool GpuCvtU8ToF32Planar(const uint8_t* d_in, float* d_out, int w, int h, int c,
                         std::string* err);
bool GpuCvtF32ToU8Interleaved(const float* d_in, uint8_t* d_out, int w, int h, int c,
                              std::string* err);

// NLM 主体。version: 0=naive, 1=smem, 2=smem+模板展开（3 为预留进阶版本）。
// kernel_ms 输出 kernel 纯计算时间（GPU event 计时）。
// smem 需求超出设备上限时自动回退 naive 路径（保证功能正确）。
bool GpuNlm(const float* d_src, float* d_dst, int w, int h, int c,
            const NlmParamsDev& p, int version, float* kernel_ms, std::string* err);

#endif // NLM_KERNELS_H
