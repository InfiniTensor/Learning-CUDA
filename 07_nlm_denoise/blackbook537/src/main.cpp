// ============================================================================
// nlm_denoise —— 实时图像非局部均值降噪（CUDA）CLI 入口
//
// 子命令：
//   run      -i input.png -o output.png -p params.txt [--kernel v] [--log f]
//   validate -i input.png [-o output.png] -p params.txt
//   bench    [--sizes 1920x1080,3840x2160] [--channels 1,3]
//            [--param-sets all|small,base,strong-h,large] [--log results.csv]
//            [--warmup 3] [--repeat 10] [--with-cpu] [--cpu-repeat 1]
//   metrics  --reference clean.png --test output.png [--label name] [--log quality.csv]
//   gen      -o noisy.png [--size 1920x1080] [--channels 3] [--sigma 25] [--seed 1]
//   units    （Host 侧单元测试，无需 GPU）
// ============================================================================
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <string>

#include "image_io.h"
#include "nlm_cpu_ref.h"
#include "params.h"
#include "pipeline.h"
#include "benchmark.h"

namespace {

void PrintUsage() {
    std::printf(
        "用法: nlm_denoise <子命令> [选项]\n"
        "\n"
        "子命令:\n"
        "  run       对单张图像执行 GPU NLM 降噪并输出性能日志\n"
        "  validate  与 CPU 参考（及可选 OpenCV）对比，输出 MAE/PSNR 校验\n"
        "  bench     分辨率 x 通道 x 参数组合 x kernel 版本性能扫描（CSV）\n"
        "  metrics   计算测试图相对参考图的 MAE/PSNR，可追加写入 CSV\n"
        "  gen       生成合成含噪测试图\n"
        "  units     运行 Host 侧单元测试（无需 GPU）\n"
        "\n"
        "run 选项:\n"
        "  -i <file>     输入图像（PNG/JPG，灰度或 RGB）\n"
        "  -o <file>     输出图像（建议 output/images/，目录不存在时自动创建）\n"
        "  -p <file>     参数文件（默认 params.txt）\n"
        "  --kernel <v>  kernel 版本 0=naive 1=smem 2=smem+unroll（默认 2）\n"
        "  --log <file>  性能日志文件（默认 output/logs/nlm_perf.log，追加写）\n"
        "  --with-cpu    同时运行 CPU 参考以计算加速比\n"
        "\n"
        "bench 选项:\n"
        "  --sizes <list>      尺寸列表（默认 1920x1080）\n"
        "  --channels <list>   1,3（默认 1,3）\n"
        "  --cpu-repeat <n> CPU 重复次数（默认 1）\n"
        "  --param-sets <list> all 或 small,base,strong-h,large（默认 all）\n"
        "  --with-cpu          CPU 统计（仅支持不超过 1080p）\n"
        "\n"
        "metrics 选项:\n"
        "  --reference <file>  干净参考图（必填）\n"
        "  --test <file>       待评估图（必填）\n"
        "  --label <name>      CSV 行标签（默认 unnamed）\n"
        "  --log <file>        可选 CSV 输出\n"
        "\n"
        "gen 选项:\n"
        "  -o <file>     输出图像（缺省按规范自动命名：output/generated/noisy_<尺寸>_<c>ch_sigma<σ>.png）\n");
}

// 解析 "-i xx / --key value / --flag" 到 map（flag 型参数值为 "1"）
std::map<std::string, std::string> ParseArgs(int argc, char** argv, int start) {
    std::map<std::string, std::string> m;
    for (int i = start; i < argc; ++i) {
        std::string a = argv[i];
        if (a.size() > 0 && a[0] == '-') {
            if (i + 1 < argc && argv[i + 1][0] != '-') {
                m[a] = argv[++i];
            } else {
                m[a] = "1";  // flag
            }
        }
    }
    return m;
}

std::string GetArg(const std::map<std::string, std::string>& m,
                   const char* key, const char* dflt) {
    std::map<std::string, std::string>::const_iterator it = m.find(key);
    return it == m.end() ? std::string(dflt) : it->second;
}

int CmdRun(const std::map<std::string, std::string>& args) {
    const std::string input = GetArg(args, "-i", "");
    const std::string output = GetArg(args, "-o", "");
    const std::string params_path = GetArg(args, "-p", "params.txt");
    const std::string log_path = GetArg(args, "--log", "output/logs/nlm_perf.log");
    const int kernel_ver = std::atoi(GetArg(args, "--kernel", "2").c_str());
    const bool with_cpu = args.count("--with-cpu") > 0;

    if (input.empty() || output.empty()) {
        std::fprintf(stderr, "[run] 必须指定 -i 与 -o\n");
        return 2;
    }

    std::string err;
    ImageU8 src;
    if (!LoadImage(input.c_str(), &src, &err)) {
        std::fprintf(stderr, "[run] %s\n", err.c_str());
        return 2;
    }
    NlmParams params;
    if (!ParseParams(params_path.c_str(), &params, &err)) {
        std::fprintf(stderr, "[run] %s\n", err.c_str());
        return 2;
    }

    ImageU8 dst;
    NlmPerfReport perf;
    if (!NlmDenoiseGpu(src, &dst, params, kernel_ver, &perf, &err)) {
        std::fprintf(stderr, "[run] %s\n", err.c_str());
        return 2;
    }
    if (!SaveImage(output.c_str(), dst, &err)) {
        std::fprintf(stderr, "[run] %s\n", err.c_str());
        return 2;
    }

    if (with_cpu) {
        ImageU8 cpu;
        HostTimer t;
        t.Start();
        NlmDenoiseCpuRef(src, &cpu, params);
        perf.cpu_ref_ms = (float)t.Ms();
        perf.speedup_vs_cpu = perf.cpu_ref_ms / perf.e2e_ms;
        std::printf("[run] 对 CPU 参考: MAE=%.4f  PSNR=%.2f dB\n",
                    ComputeMAE(dst, cpu), ComputePSNR(dst, cpu));
    }

    std::printf("[run] %dx%dx%d  kernel v%d  pr=%d sr=%d h=%.1f sigma=%.1f\n",
                src.width, src.height, src.channels, kernel_ver,
                params.patch_radius, params.search_radius, params.h, params.sigma);
    std::printf("[run] kernel=%.3f ms  e2e=%.3f ms (H2D %.3f / D2H %.3f)  %.2f Mpx/s\n",
                perf.kernel_ms, perf.e2e_ms, perf.h2d_ms, perf.d2h_ms,
                perf.throughput_mpx_s);
    if (perf.cpu_ref_ms > 0.0f)
        std::printf("[run] CPU=%.1f ms  加速比=%.1fx\n", perf.cpu_ref_ms, perf.speedup_vs_cpu);

    // 性能日志（CSV 行，追加写）；目录不存在时自动创建（数据集中管理）
    EnsureParentDir(log_path.c_str());
    std::ofstream log(log_path.c_str(), std::ios::app);
    if (log) {
        log << src.width << "x" << src.height << ',' << src.channels << ','
            << params.patch_radius << ',' << params.search_radius << ','
            << params.h << ',' << params.sigma << ','
            << kernel_ver << ',' << perf.kernel_ms << ',' << perf.e2e_ms << ','
            << perf.throughput_mpx_s << ','
            << perf.cpu_ref_ms << ',' << perf.speedup_vs_cpu << '\n';
    }
    return 0;
}

int CmdGen(const std::map<std::string, std::string>& args) {
    const std::string size = GetArg(args, "--size", "1920x1080");
    const int channels = std::atoi(GetArg(args, "--channels", "3").c_str());
    const float sigma = (float)std::atof(GetArg(args, "--sigma", "25").c_str());
    const unsigned seed = (unsigned)std::atoi(GetArg(args, "--seed", "1").c_str());
    int w = 0, h = 0;
    if (std::sscanf(size.c_str(), "%dx%d", &w, &h) != 2 || w <= 0 || h <= 0 ||
        (channels != 1 && channels != 3)) {
        std::fprintf(stderr, "[gen] 非法 --size/--channels\n");
        return 2;
    }
    // 缺省输出：按命名规范自动生成 output/generated/noisy_<尺寸>_<c>ch_sigma<σ>.png
    char auto_name[256];
    std::snprintf(auto_name, sizeof(auto_name),
                  "output/generated/noisy_%dx%d_%dch_sigma%d.png", w, h, channels,
                  (int)(sigma + 0.5f));
    const std::string output = GetArg(args, "-o", auto_name);
    ImageU8 img = MakeSyntheticImage(w, h, channels, sigma, seed);
    std::string err;
    if (!SaveImage(output.c_str(), img, &err)) {
        std::fprintf(stderr, "[gen] %s\n", err.c_str());
        return 2;
    }
    std::printf("[gen] 已生成 %dx%dx%d sigma=%.1f -> %s\n", w, h, channels, sigma, output.c_str());
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        PrintUsage();
        return 2;
    }
    const std::string cmd = argv[1];
    const std::map<std::string, std::string> args = ParseArgs(argc, argv, 2);

    if (cmd == "run") {
        return CmdRun(args);
    } else if (cmd == "validate") {
        const std::string input = GetArg(args, "-i", "");
        if (input.empty()) {
            std::fprintf(stderr, "[validate] 必须指定 -i\n");
            return 2;
        }
        return RunValidate(input, GetArg(args, "-o", ""),
                           GetArg(args, "-p", "params.txt"));
    } else if (cmd == "bench") {
        return RunBenchmark(GetArg(args, "--sizes", "1920x1080"),
                            GetArg(args, "--channels", "1,3"),
                            GetArg(args, "--param-sets", "all"),
                            GetArg(args, "--log", "output/results/benchmark.csv"),
                            std::atoi(GetArg(args, "--warmup", "3").c_str()),
                            std::atoi(GetArg(args, "--repeat", "10").c_str()),
                            args.count("--with-cpu") > 0,
                            std::atoi(GetArg(args, "--cpu-repeat", "1").c_str()));
    } else if (cmd == "metrics") {
        const std::string reference = GetArg(args, "--reference", "");
        const std::string test = GetArg(args, "--test", "");
        if (reference.empty() || test.empty()) {
            std::fprintf(stderr, "[metrics] 必须指定 --reference 与 --test\n");
            return 2;
        }
        return RunMetrics(reference, test, GetArg(args, "--label", "unnamed"),
                          GetArg(args, "--log", ""));
    } else if (cmd == "gen") {
        return CmdGen(args);
    } else if (cmd == "units") {
        return RunUnitTests();
    }

    std::fprintf(stderr, "未知子命令: %s\n\n", cmd.c_str());
    PrintUsage();
    return 2;
}
