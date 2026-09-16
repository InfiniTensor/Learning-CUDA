#include "pipeline.h"

#include "kernels.h"

bool NlmDenoiseGpu(const ImageU8& src, ImageU8* dst,
                   const NlmParams& params, int kernel_version,
                   NlmPerfReport* perf, std::string* err) {
    // ---- 输入校验（系统边界防御）----
    if (!dst || !err) return false;
    if (src.width <= 0 || src.height <= 0 ||
        (src.channels != 1 && src.channels != 3) ||
        src.data.size() != (size_t)src.width * src.height * src.channels) {
        *err = "NlmDenoiseGpu: 非法输入图像";
        return false;
    }

    const int w = src.width, h = src.height, c = src.channels;
    const size_t n_px = (size_t)w * h;
    const size_t u8_bytes = n_px * c * sizeof(uint8_t);
    const size_t f32_bytes = n_px * c * sizeof(float);

    HostTimer t_total, t_stage;
    t_total.Start();

    // ---- 设备资源申请 ----
    uint8_t* d_src_u8 = NULL;
    uint8_t* d_dst_u8 = NULL;
    float* d_src_f32 = NULL;
    float* d_dst_f32 = NULL;
    bool ok = false;

    // 统一资源释放（失败回滚 + 正常退出共用）
    struct Cleanup {
        uint8_t** a; uint8_t** b; float** x; float** y;
        ~Cleanup() {
            std::string ignore;
            if (a && *a) GpuFree(*a, &ignore);
            if (b && *b) GpuFree(*b, &ignore);
            if (x && *x) GpuFree(*x, &ignore);
            if (y && *y) GpuFree(*y, &ignore);
        }
    } cleanup{&d_src_u8, &d_dst_u8, &d_src_f32, &d_dst_f32};

    if (!GpuAlloc((void**)&d_src_u8, u8_bytes, err)) return false;
    if (!GpuAlloc((void**)&d_dst_u8, u8_bytes, err)) return false;
    if (!GpuAlloc((void**)&d_src_f32, f32_bytes, err)) return false;
    if (!GpuAlloc((void**)&d_dst_f32, f32_bytes, err)) return false;

    // ---- H2D + 转换为 planar f32 ----
    t_stage.Start();
    if (!GpuCopyH2D(d_src_u8, src.data.data(), u8_bytes, err)) return false;
    if (perf) perf->h2d_ms = (float)t_stage.Ms();

    if (!GpuCvtU8ToF32Planar(d_src_u8, d_src_f32, w, h, c, err)) return false;

    // ---- NLM 主计算（kernel 纯计算时间由 GPU event 测量）----
    const NlmParamsDev pdev = MakeDevParams(params, c);
    float kernel_ms = 0.0f;
    if (!GpuNlm(d_src_f32, d_dst_f32, w, h, c, pdev, kernel_version,
                &kernel_ms, err)) return false;

    // ---- 量化回 u8 + D2H ----
    if (!GpuCvtF32ToU8Interleaved(d_dst_f32, d_dst_u8, w, h, c, err)) return false;

    t_stage.Start();
    dst->width = w;
    dst->height = h;
    dst->channels = c;
    dst->data.resize(n_px * c);
    if (!GpuCopyD2H(dst->data.data(), d_dst_u8, u8_bytes, err)) return false;
    if (perf) perf->d2h_ms = (float)t_stage.Ms();

    ok = true;
    if (perf) {
        perf->kernel_ms = kernel_ms;
        perf->e2e_ms = (float)t_total.Ms();
        perf->throughput_mpx_s =
            kernel_ms > 0.0f ? (float)((double)n_px / ((double)kernel_ms * 1e3)) : 0.0f;
    }
    (void)ok;
    return true;
}
