// ============================================================================
// bench 子命令：性能基准测试
//   扫描 分辨率 x 通道数 x 参数组合 x kernel 版本，输出可复核的统计 CSV。
//   GPU 每配置执行 warmup 次预热与 repeat 次采样，报告 mean/min/stddev；
//   可选 CPU 基线对同一输入、同一参数独立采样，避免旧版跨参数错配。
// ============================================================================
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>

#include "nlm_cpu_ref.h"
#include "pipeline.h"
#include "benchmark.h"

namespace {

struct SizeSpec {
    int w;
    int h;
};

struct ParamSet {
    std::string name;
    NlmParams p;
};

struct Stats {
    double mean;
    double min;
    double stddev;
};

Stats Summarize(const std::vector<double>& samples) {
    Stats out = {-1.0, -1.0, -1.0};
    if (samples.empty()) return out;
    out.min = *std::min_element(samples.begin(), samples.end());
    double sum = 0.0;
    for (size_t i = 0; i < samples.size(); ++i) sum += samples[i];
    out.mean = sum / static_cast<double>(samples.size());
    double squared = 0.0;
    for (size_t i = 0; i < samples.size(); ++i) {
        const double delta = samples[i] - out.mean;
        squared += delta * delta;
    }
    out.stddev = std::sqrt(squared / static_cast<double>(samples.size()));
    return out;
}

std::vector<SizeSpec> ParseSizes(const std::string& spec) {
    std::vector<SizeSpec> out;
    size_t pos = 0;
    while (pos < spec.size()) {
        const size_t comma = spec.find(',', pos);
        const std::string tok = spec.substr(
            pos, comma == std::string::npos ? comma : comma - pos);
        int w = 0;
        int h = 0;
        if (std::sscanf(tok.c_str(), "%dx%d", &w, &h) == 2 && w > 0 && h > 0) {
            out.push_back(SizeSpec{w, h});
        }
        pos = comma == std::string::npos ? spec.size() : comma + 1;
    }
    return out;
}

std::vector<int> ParseChannels(const std::string& spec) {
    std::vector<int> out;
    size_t pos = 0;
    while (pos < spec.size()) {
        const size_t comma = spec.find(',', pos);
        const int c = std::atoi(spec.substr(
            pos, comma == std::string::npos ? comma : comma - pos).c_str());
        if (c == 1 || c == 3) out.push_back(c);
        pos = comma == std::string::npos ? spec.size() : comma + 1;
    }
    return out;
}

bool ListContains(const std::string& spec, const std::string& name) {
    if (spec.empty() || spec == "all") return true;
    size_t pos = 0;
    while (pos < spec.size()) {
        const size_t comma = spec.find(',', pos);
        const std::string token = spec.substr(
            pos, comma == std::string::npos ? comma : comma - pos);
        if (token == name) return true;
        pos = comma == std::string::npos ? spec.size() : comma + 1;
    }
    return false;
}

std::vector<ParamSet> SelectParamSets(const std::string& spec) {
    std::vector<ParamSet> all;

    NlmParams small;
    small.patch_radius = 2;
    small.search_radius = 7;
    small.h = 10.0f;
    small.sigma = 25.0f;
    all.push_back(ParamSet{"small", small});

    NlmParams base;
    all.push_back(ParamSet{"base", base});

    NlmParams strong_h;
    strong_h.h = 15.0f;
    all.push_back(ParamSet{"strong-h", strong_h});

    NlmParams large;
    large.patch_radius = 4;
    large.search_radius = 14;
    large.h = 10.0f;
    large.sigma = 25.0f;
    all.push_back(ParamSet{"large", large});

    std::vector<ParamSet> selected;
    for (size_t i = 0; i < all.size(); ++i) {
        if (ListContains(spec, all[i].name)) selected.push_back(all[i]);
    }
    return selected;
}

}  // namespace

int RunBenchmark(const std::string& sizes_spec, const std::string& channels_spec,
                 const std::string& param_sets_spec, const std::string& csv_path,
                 int warmup, int repeat, bool with_cpu, int cpu_repeat) {
    const std::vector<SizeSpec> sizes = ParseSizes(sizes_spec);
    const std::vector<int> channels = ParseChannels(channels_spec);
    const std::vector<ParamSet> sets = SelectParamSets(param_sets_spec);
    if (sizes.empty() || channels.empty() || sets.empty()) {
        std::fprintf(stderr, "[bench] 无效的 --sizes、--channels 或 --param-sets\n");
        return 2;
    }
    if (warmup < 0) warmup = 0;
    if (repeat < 1) repeat = 1;
    if (cpu_repeat < 1) cpu_repeat = 1;

    EnsureParentDir(csv_path.c_str());
    std::ofstream csv(csv_path.c_str());
    if (!csv) {
        std::fprintf(stderr, "[bench] 无法写入 CSV: %s\n", csv_path.c_str());
        return 2;
    }
    csv << "config,size,channels,pr,sr,h,sigma,kernel_ver,warmup,repeat,"
           "kernel_mean_ms,kernel_min_ms,kernel_stddev_ms,"
           "e2e_mean_ms,e2e_min_ms,e2e_stddev_ms,mpx_s,"
           "cpu_repeat,cpu_mean_ms,cpu_min_ms,cpu_stddev_ms,speedup_e2e\n";
    csv << std::fixed << std::setprecision(6);

    std::printf("[bench] param_sets=%s warmup=%d repeat=%d cpu_repeat=%d -> %s\n",
                param_sets_spec.c_str(), warmup, repeat, cpu_repeat, csv_path.c_str());
    const int kernel_versions[] = {0, 1, 2};

    for (size_t si = 0; si < sizes.size(); ++si) {
        const int w = sizes[si].w;
        const int h = sizes[si].h;
        for (size_t ci = 0; ci < channels.size(); ++ci) {
            const int c = channels[ci];
            for (size_t pi = 0; pi < sets.size(); ++pi) {
                const ParamSet& set = sets[pi];
                const NlmParams& prm = set.p;
                const unsigned seed = 777u + static_cast<unsigned>(c);
                const ImageU8 src = MakeSyntheticImage(w, h, c, prm.sigma, seed);

                Stats cpu_stats = {-1.0, -1.0, -1.0};
                int measured_cpu_repeat = 0;
                if (with_cpu && static_cast<size_t>(w) * h <= 1920u * 1080u) {
                    std::vector<double> cpu_samples;
                    for (int it = 0; it < cpu_repeat; ++it) {
                        ImageU8 dst_cpu;
                        HostTimer timer;
                        timer.Start();
                        NlmDenoiseCpuRef(src, &dst_cpu, prm);
                        cpu_samples.push_back(timer.Ms());
                    }
                    cpu_stats = Summarize(cpu_samples);
                    measured_cpu_repeat = cpu_repeat;
                    std::printf("[bench] CPU %dx%dx%d %-8s: mean=%.1f min=%.1f std=%.1f ms\n",
                                w, h, c, set.name.c_str(), cpu_stats.mean,
                                cpu_stats.min, cpu_stats.stddev);
                } else if (with_cpu) {
                    std::printf("[bench] CPU %dx%dx%d %-8s: 跳过（超过 1080p 安全上限）\n",
                                w, h, c, set.name.c_str());
                }

                for (size_t ki = 0;
                     ki < sizeof(kernel_versions) / sizeof(kernel_versions[0]); ++ki) {
                    const int ver = kernel_versions[ki];
                    std::vector<double> kernel_samples;
                    std::vector<double> e2e_samples;
                    std::string err;
                    for (int it = 0; it < warmup + repeat; ++it) {
                        ImageU8 dst;
                        NlmPerfReport perf;
                        if (!NlmDenoiseGpu(src, &dst, prm, ver, &perf, &err)) {
                            std::fprintf(stderr, "[bench] v%d 失败: %s\n", ver, err.c_str());
                            return 2;
                        }
                        if (it >= warmup) {
                            kernel_samples.push_back(perf.kernel_ms);
                            e2e_samples.push_back(perf.e2e_ms);
                        }
                    }
                    const Stats kernel_stats = Summarize(kernel_samples);
                    const Stats e2e_stats = Summarize(e2e_samples);
                    const double mpx_s = kernel_stats.mean > 0.0
                                             ? static_cast<double>(w) * h /
                                                   (kernel_stats.mean * 1e3)
                                             : -1.0;
                    const double speedup = cpu_stats.mean > 0.0
                                               ? cpu_stats.mean / e2e_stats.mean
                                               : -1.0;

                    csv << set.name << ',' << w << 'x' << h << ',' << c << ','
                        << prm.patch_radius << ',' << prm.search_radius << ',' << prm.h << ','
                        << prm.sigma << ',' << ver << ',' << warmup << ',' << repeat << ','
                        << kernel_stats.mean << ',' << kernel_stats.min << ','
                        << kernel_stats.stddev << ',' << e2e_stats.mean << ','
                        << e2e_stats.min << ',' << e2e_stats.stddev << ',' << mpx_s << ','
                        << measured_cpu_repeat << ',' << cpu_stats.mean << ',' << cpu_stats.min
                        << ',' << cpu_stats.stddev << ',' << speedup << '\n';

                    std::printf(
                        "[bench] %dx%dx%d %-8s v%d: kernel %.3f±%.3f ms "
                        "e2e %.3f±%.3f ms %.1f Mpx/s%s\n",
                        w, h, c, set.name.c_str(), ver, kernel_stats.mean,
                        kernel_stats.stddev, e2e_stats.mean, e2e_stats.stddev, mpx_s,
                        speedup > 0.0
                            ? (std::string(" speedup=") + std::to_string(speedup) + "x").c_str()
                            : "");
                }
            }
        }
    }
    std::printf("[bench] 完成，CSV 已写入 %s\n", csv_path.c_str());
    return 0;
}
