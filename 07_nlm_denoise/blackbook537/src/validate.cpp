// ============================================================================
// validate 子命令：正确性验证
//   主基线：自研 CPU 参考实现（语义与 GPU kernel 严格一致），逐版本对比 MAE/PSNR
//   辅基线（可选，-DWITH_OPENCV）：OpenCV fastNlMeansDenoising(Colored) 交叉验证
// 通过阈值（架构设计 §10.1）：vs CPU 参考 V0/V1 MAE<=0.5，V2 MAE<=1.0，PSNR>=30dB
// ============================================================================
#include <cstdio>
#include <string>

#include "nlm_cpu_ref.h"
#include "pipeline.h"
#include "benchmark.h"

#ifdef WITH_OPENCV
#include <opencv2/opencv.hpp>
#include <opencv2/photo.hpp>
#endif

int RunValidate(const std::string& input, const std::string& output,
                const std::string& params_path) {
    std::string err;
    ImageU8 src;
    if (!LoadImage(input.c_str(), &src, &err)) {
        std::fprintf(stderr, "[validate] %s\n", err.c_str());
        return 2;
    }
    NlmParams params;
    if (!ParseParams(params_path.c_str(), &params, &err)) {
        std::fprintf(stderr, "[validate] %s\n", err.c_str());
        return 2;
    }

    std::printf("[validate] 图像 %dx%dx%d，参数 pr=%d sr=%d h=%.1f sigma=%.1f\n",
                src.width, src.height, src.channels,
                params.patch_radius, params.search_radius, params.h, params.sigma);

    // ---- CPU 参考（主基线）----
    ImageU8 cpu;
    HostTimer t;
    t.Start();
    NlmDenoiseCpuRef(src, &cpu, params);
    const double cpu_ms = t.Ms();
    std::printf("[validate] CPU 参考耗时: %.1f ms\n", cpu_ms);

    // ---- 各 GPU kernel 版本对比 ----
    std::printf("%-6s %-12s %-12s %-12s %-10s %s\n",
                "ver", "kernel(ms)", "e2e(ms)", "MAE(vsCPU)", "PSNR(dB)", "判定");
    bool pass = true;
    ImageU8 last_gpu;
    for (int v = 0; v <= 2; ++v) {
        ImageU8 gpu;
        NlmPerfReport perf;
        if (!NlmDenoiseGpu(src, &gpu, params, v, &perf, &err)) {
            std::fprintf(stderr, "[validate] kernel v%d 运行失败: %s\n", v, err.c_str());
            return 2;
        }
        const double mae = ComputeMAE(gpu, cpu);
        const double psnr = ComputePSNR(gpu, cpu);
        const double mae_thr = (v == 2) ? 1.0 : 0.5;
        const bool ok = (mae <= mae_thr) && (psnr >= 30.0);
        std::printf("%-6d %-12.3f %-12.3f %-12.4f %-10.2f %s\n",
                    v, perf.kernel_ms, perf.e2e_ms, mae, psnr, ok ? "PASS" : "FAIL");
        pass = pass && ok;
        if (v == 2) last_gpu = gpu;
    }

#ifdef WITH_OPENCV
    // ---- OpenCV 参考（辅基线，信息性输出，不作为 PASS/FAIL 门槛）----
    {
        cv::Mat cv_src(src.height, src.width,
                       src.channels == 1 ? CV_8UC1 : CV_8UC3, src.data.data());
        cv::Mat cv_dst;
        if (src.channels == 1) {
            cv::fastNlMeansDenoising(cv_src, cv_dst, params.h,
                                     2 * params.patch_radius + 1,
                                     2 * params.search_radius + 1);
        } else {
            cv::fastNlMeansDenoisingColored(cv_src, cv_dst, params.h, params.h,
                                            2 * params.patch_radius + 1,
                                            2 * params.search_radius + 1);
        }
        ImageU8 ocv;
        ocv.width = src.width;
        ocv.height = src.height;
        ocv.channels = src.channels;
        ocv.data.assign(cv_dst.data, cv_dst.data + src.data.size());
        std::printf("[validate] OpenCV 参考对比（信息性）: GPU(v2) vs OpenCV  MAE=%.4f  PSNR=%.2f dB\n",
                    ComputeMAE(last_gpu, ocv), ComputePSNR(last_gpu, ocv));
        std::printf("[validate] 注：OpenCV 的 h 标定与任务公式存在实现差异，"
                    "仅作交叉验证，验收以 CPU 参考为准。\n");
    }
#endif

    if (!output.empty()) {
        if (!SaveImage(output.c_str(), last_gpu, &err)) {
            std::fprintf(stderr, "[validate] %s\n", err.c_str());
            return 2;
        }
        std::printf("[validate] 已保存 GPU(v2) 输出: %s\n", output.c_str());
    }

    std::printf("[validate] 总体判定: %s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
