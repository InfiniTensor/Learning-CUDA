// main.cpp - NLM 降噪 CLI
#include "image_io.h"
#include "nlm.h"
#include "types.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace nlm;

namespace {

void usage(const char* prog) {
    std::cerr << "用法:\n"
              << "  " << prog << " denoise <in.png> <params.txt> <out.png> [选项]\n"
              << "  " << prog << " bench   <in.png> <params.txt> [选项]\n"
              << "  " << prog << " selftest\n"
              << "选项:\n"
              << "  --cpu             额外跑 CPU 参考实现（耗时可能很长）\n"
              << "  --opencv          额外跑 OpenCV fastNlMeansDenoising* 作为外部对照\n"
              << "  --ref <clean.png> 提供无噪真值图，用于计算降噪后的 PSNR/MAE\n"
              << "  --perf <log>      性能日志输出路径（追加写）\n"
              << "  --compare <img>   与另一张图逐像素比较（输出 MAE/PSNR）\n";
}

struct Options {
    bool use_cpu = false;
    bool use_opencv = false;
    std::string ref_path;
    std::string perf_path;
    std::string compare_path;
};

Options parse(int argc, char** argv, int start) {
    Options o;
    for (int i = start; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--cpu")
            o.use_cpu = true;
        else if (a == "--opencv")
            o.use_opencv = true;
        else if (a == "--ref" && i + 1 < argc)
            o.ref_path = argv[++i];
        else if (a == "--perf" && i + 1 < argc)
            o.perf_path = argv[++i];
        else if (a == "--compare" && i + 1 < argc)
            o.compare_path = argv[++i];
        else
            throw std::runtime_error("未知选项: " + a);
    }
    return o;
}

void print_quality(const char* tag, const Quality& q) {
    std::printf("  %-14s MAE=%7.4f  PSNR=%6.2f dB  max|diff|=%3.0f\n", tag, q.mae, q.psnr,
                q.max_abs);
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 2) {
            usage(argv[0]);
            return 1;
        }
        const std::string cmd = argv[1];

        if (cmd == "selftest") {
            // 构造 64x64 灰度图：左半 100、右半 200，加少量噪声，验证平滑后仍保持两段结构
            Image im;
            im.width = 64;
            im.height = 64;
            im.channels = 1;
            im.data.resize(64 * 64);
            for (int y = 0; y < 64; ++y)
                for (int x = 0; x < 64; ++x)
                    im.data[(size_t)y * 64 + x] =
                        (uint8_t)((x < 32 ? 100 : 200) + ((x * 7 + y * 13) % 11) - 5);

            NLMParams p;
            p.patch_radius = 3;
            p.search_radius = 8;
            p.h = 12.0;
            p.sigma = 5.0;

            GpuNLM gpu;
            Image g = gpu.denoise(im, p);
            Image c = cpu_nlm(im, p);
            const Quality q = compare(g, c);
            std::printf("[selftest] GPU vs CPU 参考: MAE=%.4f PSNR=%.2f dB max=%.0f\n", q.mae,
                        q.psnr, q.max_abs);

            // 检查左右两段的均值是否被明显拉近（降噪应平滑段内噪声但保留跳变）
            double lm = 0, rm = 0;
            for (int y = 0; y < 64; ++y)
                for (int x = 0; x < 64; ++x) {
                    const double v = g.data[(size_t)y * 64 + x];
                    if (x < 28)
                        lm += v;
                    else if (x >= 36)
                        rm += v;
                }
            lm /= (64.0 * 28.0);
            rm /= (64.0 * 28.0);
            std::printf("[selftest] 左段均值=%.2f 右段均值=%.2f（应分别接近 100 / 200）\n", lm, rm);

            const bool ok =
                (q.mae < 1.0) && (std::fabs(lm - 100.0) < 5.0) && (std::fabs(rm - 200.0) < 5.0);
            std::printf("[selftest] %s\n", ok ? "PASS" : "FAIL");
            return ok ? 0 : 1;
        }

        if (cmd != "denoise" && cmd != "bench") {
            usage(argv[0]);
            return 1;
        }
        if (argc < 4) {
            usage(argv[0]);
            return 1;
        }
        const std::string in_path = argv[2];
        const std::string param_path = argv[3];
        const std::string out_path = (cmd == "denoise" && argc >= 5) ? argv[4] : "";
        const int opt_start = (cmd == "denoise") ? 5 : 4;
        const Options opt = parse(argc, argv, opt_start);

        const double t0 = std::chrono::duration<double, std::milli>(
                              std::chrono::high_resolution_clock::now().time_since_epoch())
                              .count();
        Image in = load_image(in_path);
        const NLMParams p = load_params(param_path);
        const double t1 = std::chrono::duration<double, std::milli>(
                              std::chrono::high_resolution_clock::now().time_since_epoch())
                              .count();

        std::printf("[nlm] 输入: %dx%d, %d 通道, %.2f MPix\n", in.width, in.height, in.channels,
                    (double)in.pixels() / 1e6);
        std::printf("[nlm] 参数: patch_radius=%d, search_radius=%d, h=%.2f, sigma=%.2f%s\n",
                    p.patch_radius, p.search_radius, p.h, p.sigma,
                    p.is_exact() ? "  (精确)" : "  (近似: step/LUT)");
        std::printf("[nlm] 读图耗时: %.2f ms\n", t1 - t0);

        // ---- GPU ----
        PerfStats st;
        GpuNLM gpu;
        Image out = gpu.denoise(in, p, &st);
        std::printf("[gpu] 预处理 %.2f ms | 核函数 %.2f ms | 回传 %.2f ms | 合计 %.2f ms\n",
                    st.preprocess_ms, st.denoise_ms, st.download_ms, st.total_ms);
        std::printf("[gpu] 吞吐 %.1f MPix/s  (grid=%d blocks × %d threads)\n",
                    st.megapixels_per_sec, st.blocks, st.threads);

        // ---- 与无噪真值比较 ----
        Quality q_ref{};
        bool has_ref = false;
        if (!opt.ref_path.empty()) {
            Image clean = load_image(opt.ref_path);
            q_ref = compare(out, clean);
            has_ref = true;
            print_quality("vs 无噪真值", q_ref);
        }
        if (!opt.compare_path.empty()) {
            Image other = load_image(opt.compare_path);
            print_quality("vs 指定图", compare(out, other));
        }

        // ---- CPU 参考 ----
        Quality q_cpu{};
        bool has_cpu = false;
        if (opt.use_cpu) {
            double cpu_ms = 0;
            Image c = cpu_nlm(in, p, &cpu_ms);
            st.cpu_ms = cpu_ms;
            st.speedup_vs_cpu = st.total_ms > 0 ? cpu_ms / st.total_ms : 0.0;
            std::printf("[cpu] 参考耗时 %.2f ms  (加速比 %.1fx)\n", cpu_ms, st.speedup_vs_cpu);
            q_cpu = compare(out, c);
            has_cpu = true;
            print_quality("vs CPU 参考", q_cpu);
            if (!opt.ref_path.empty()) {
                Image clean = load_image(opt.ref_path);
                print_quality("CPU vs 真值", compare(c, clean));
            }
        }

        // ---- OpenCV 参考 ----
        Quality q_ocv{};
        bool has_ocv = false;
        if (opt.use_opencv && p.is_exact()) {
            double ocv_ms = 0;
            Image o = opencv_nlm(in, p, &ocv_ms);
            st.opencv_ms = ocv_ms;
            st.speedup_vs_opencv = st.total_ms > 0 ? ocv_ms / st.total_ms : 0.0;
            std::printf("[opencv] 参考耗时 %.2f ms  (加速比 %.1fx)\n", ocv_ms,
                        st.speedup_vs_opencv);
            q_ocv = compare(out, o);
            has_ocv = true;
            print_quality("vs OpenCV", q_ocv);
            if (!opt.ref_path.empty()) {
                Image clean = load_image(opt.ref_path);
                print_quality("OpenCV vs 真值", compare(o, clean));
            }
        }

        if (cmd == "denoise" && !out_path.empty()) {
            save_image(out_path, out);
            std::printf("[nlm] 已写出: %s\n", out_path.c_str());
        }

        // ---- 性能日志 ----
        if (!opt.perf_path.empty()) {
            std::ofstream pf(opt.perf_path, std::ios::app);
            if (!pf)
                throw std::runtime_error("无法写入性能日志: " + opt.perf_path);
            pf << "[NLM] " << in.width << "x" << in.height << "x" << in.channels
               << " rp=" << p.patch_radius << " rs=" << p.search_radius << " h=" << p.h
               << " sigma=" << p.sigma << " sstep=" << p.search_step << " pstep=" << p.patch_step
               << " lut=" << (p.use_lut ? 1 : 0) << " | gpu=" << st.total_ms << "ms"
               << " kernel=" << st.denoise_ms << "ms"
               << " throughput=" << st.megapixels_per_sec << "MPix/s";
            if (st.cpu_ms > 0)
                pf << " cpu=" << st.cpu_ms << "ms speedup_cpu=" << st.speedup_vs_cpu;
            if (st.opencv_ms > 0)
                pf << " opencv=" << st.opencv_ms << "ms speedup_opencv=" << st.speedup_vs_opencv;
            if (has_ref)
                pf << " psnr_vs_clean=" << q_ref.psnr << " mae_vs_clean=" << q_ref.mae;
            if (has_cpu)
                pf << " mae_vs_cpu=" << q_cpu.mae;
            if (has_ocv)
                pf << " mae_vs_opencv=" << q_ocv.mae;
            pf << "\n";
            std::printf("[nlm] 性能日志已写入: %s\n", opt.perf_path.c_str());
        }
        (void)has_ref;
        (void)has_cpu;
        (void)has_ocv;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[error] " << e.what() << "\n";
        return 1;
    }
}
