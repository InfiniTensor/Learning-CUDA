// =============================================================================
//  main.cu -- 低精度格式软件模拟与反量化 主程序
//
//  用法:
//    lpq_quant --input <tensor_file> --params <params_file> [options]
//  选项:
//    --out-prefix <prefix>  输出文件前缀(默认取输入文件名去掉扩展名)
//    --out-dir <dir>        输出目录(默认与输入同目录)
//    --seed <n>             随机舍入种子(覆盖参数文件)
//    --repeat <n>           重复测量次数(默认 20)
//    --warmup <n>           预热次数(默认 5)
//    --device <n>           指定 GPU
//    --cpu-check            额外运行独立的 CPU 参考实现并做逐位比对
//    --metrics-precise      误差统计用逐元素 fp64 累加(默认用补偿 fp32 累加)
//    --no-fuse-stats        不把误差统计融进量化 kernel(改跑独立的 metrics kernel,
//                           便于对比两条路径的耗时; 默认是融合的)
//    --quiet                不打印摘要到 stdout
//
//  输出:
//    <prefix>_lowp.bin          低精度权重文件(packed data + scales + header)
//    <prefix>_dequant.bin       反量化结果(纯二进制,行主序,类型由 output_type 决定)
//    <prefix>_log.txt           误差与性能日志
//    <prefix>_cpu_dequant.bin   仅 --cpu-check 时生成:CPU 参考实现的反量化结果
// =============================================================================
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "lpq/common.hpp"
#include "lpq/config.hpp"
#include "lpq/lowp_tensor.hpp"
#include "lpq/pipeline.hpp"
#include "lpq/reference.hpp"

using namespace lpq;

namespace {

struct Args {
    std::string input;
    std::string params;
    std::string out_prefix;
    std::string out_dir;
    long long seed = -1;
    int repeat = 20;
    int warmup = 5;
    int device = 0;
    bool cpu_check = false;
    bool metrics_precise = false;
    bool fuse_stats = true;
    bool quiet = false;
};

void print_usage() {
    std::printf(
        "usage: lpq_quant --input <tensor_file> --params <params_file> [options]\n"
        "  --out-prefix <p>  output file prefix\n"
        "  --out-dir <d>     output directory\n"
        "  --seed <n>        random seed for stochastic rounding\n"
        "  --repeat <n>      timed repetitions (default 20)\n"
        "  --warmup <n>      warmup iterations (default 5)\n"
        "  --device <n>      select the GPU device\n"
        "  --cpu-check       run the independent CPU reference and compare bitwise\n"
        "  --metrics-precise accumulate error statistics in per-element fp64\n"
        "  --no-fuse-stats   run error statistics as a separate kernel\n"
        "  --quiet           quiet mode\n");
}

Args parse_args(int argc, char** argv) {
    Args a;
    auto need = [&](int& i, const char* name) {
        if (i + 1 >= argc)
            throw Error(std::string("missing value for option: ") + name, __FILE__, __LINE__);
        return std::string(argv[++i]);
    };
    for (int i = 1; i < argc; ++i) {
        const std::string s = argv[i];
        if (s == "--input" || s == "-i") {
            a.input = need(i, "--input");
        } else if (s == "--params" || s == "-p") {
            a.params = need(i, "--params");
        } else if (s == "--out-prefix") {
            a.out_prefix = need(i, "--out-prefix");
        } else if (s == "--out-dir") {
            a.out_dir = need(i, "--out-dir");
        } else if (s == "--seed") {
            a.seed = std::stoll(need(i, "--seed"));
        } else if (s == "--repeat") {
            a.repeat = std::stoi(need(i, "--repeat"));
        } else if (s == "--warmup") {
            a.warmup = std::stoi(need(i, "--warmup"));
        } else if (s == "--device") {
            a.device = std::stoi(need(i, "--device"));
        } else if (s == "--cpu-check") {
            a.cpu_check = true;
        } else if (s == "--metrics-precise") {
            a.metrics_precise = true;
        } else if (s == "--no-fuse-stats") {
            a.fuse_stats = false;
        } else if (s == "--quiet" || s == "-q") {
            a.quiet = true;
        } else if (s == "--help" || s == "-h") {
            print_usage();
            std::exit(0);
        } else {
            throw Error("unknown option: " + s, __FILE__, __LINE__);
        }
    }
    if (a.input.empty() || a.params.empty()) {
        print_usage();
        throw Error("both --input and --params are required", __FILE__, __LINE__);
    }
    if (a.repeat < 1)
        a.repeat = 1;
    if (a.warmup < 0)
        a.warmup = 0;
    return a;
}

std::string basename_noext(const std::string& path) {
    size_t slash = path.find_last_of("/\\");
    std::string base = (slash == std::string::npos) ? path : path.substr(slash + 1);
    size_t dot = base.find_last_of('.');
    if (dot != std::string::npos && dot > 0)
        base = base.substr(0, dot);
    return base;
}

std::string parent_dir(const std::string& path) {
    size_t slash = path.find_last_of("/\\");
    return (slash == std::string::npos) ? std::string(".") : path.substr(0, slash);
}

std::string join_path(const std::string& dir, const std::string& name) {
    if (dir.empty())
        return name;
    const char last = dir.back();
    if (last == '/' || last == '\\')
        return dir + name;
    return dir + "/" + name;
}

struct TimingStats {
    double min = 1e30;
    double sum = 0.0;
    int n = 0;
    void add(double v) {
        min = std::min(min, v);
        sum += v;
        ++n;
    }
    double mean() const { return n ? sum / n : 0.0; }
};

const char* gpu_name() {
    static std::string name = [] {
        cudaDeviceProp prop{};
        if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess)
            return std::string("unknown");
        return std::string(prop.name);
    }();
    return name.c_str();
}

}  // namespace

int main(int argc, char** argv) {
    try {
        WallTimer startup;
        startup.start();
        const Args args = parse_args(argc, argv);
        const QuantConfig cfg0 = parse_quant_config(args.params);
        QuantConfig cfg = cfg0;
        if (args.seed >= 0)
            cfg.seed = (uint64_t)args.seed;

        // 一次性驱动初始化: 首次碰 CUDA 时运行时才会建上下文(本机实测 ~74ms).
        // 它不属于「预处理」(读文件 + H2D), 但也不该被藏起来: 这里从进程启动
        // 开始计时, 到 cudaFree(nullptr) 把运行时唤醒为止, 在日志里单列 cuda_context_init_ms.
        LPQ_CUDA_CHECK(cudaSetDevice(args.device));
        LPQ_CUDA_CHECK(cudaFree(nullptr));
        const double cuda_context_ms = startup.stop_ms();

        // ---------------- 预处理:读文件 + H2D 拷贝 ----------------
        WallTimer wall;
        wall.start();
        const HostTensor host = read_tensor_file(args.input);
        const int64_t numel = host.numel();
        LPQ_CHECK(numel > 0, "input tensor has 0 elements");

        const size_t in_elem = host.elem_size();
        DeviceBuffer d_input(host.data.size());
        LPQ_CUDA_CHECK(
            cudaMemcpy(d_input.get(), host.data.data(), host.data.size(), cudaMemcpyHostToDevice));
        DeviceBuffer d_output((size_t)numel * ((cfg.output_type == OutputType::FP32) ? 4u : 2u));
        const double preprocess_ms = wall.stop_ms();

        // ---------------- 重复测量 ----------------
        TimingStats t_amax, t_scale, t_quant, t_dequant, t_metrics, t_pipeline;
        QuantizedDevice q;
        ErrorStats stats;
        bool stats_fused = false;
        const AccumMode accum = args.metrics_precise ? AccumMode::Precise : AccumMode::Fast;
        for (int it = 0; it < args.warmup + args.repeat; ++it) {
            const bool record = (it >= args.warmup);
            wall.start();
            if (args.fuse_stats) {
                // 误差统计在量化 kernel 里顺带完成: 那一遍 268MB 的 fp32 输入重读
                // 与 134MB 的反量化结果重读(a.k.a. metrics kernel)整体消失.
                q = quantize_on_device(
                    d_input.get(), host.dtype, numel, cfg, nullptr, &stats, accum);
            } else {
                q = quantize_on_device(d_input.get(), host.dtype, numel, cfg);
            }
            const double dq_ms = dequantize_on_device(q, cfg, d_output.get());
            const double pipeline_ms = wall.stop_ms();
            double metrics_ms = 0.0;
            if (!q.stats_fused) {
                // 未融合(或显式关闭)时才需要独立的 metrics kernel
                WallTimer mt;
                mt.start();
                stats = compute_error_stats(d_input.get(), host.dtype, d_output.get(),
                    cfg.output_type, numel, nullptr, accum);
                metrics_ms = mt.stop_ms();
            }
            stats_fused = q.stats_fused;
            if (record) {
                t_amax.add(q.timings.amax_ms);
                t_scale.add(q.timings.scale_ms);
                t_quant.add(q.timings.quant_ms);
                t_dequant.add(dq_ms);
                t_metrics.add(metrics_ms);
                t_pipeline.add(pipeline_ms);
            }
        }

        // ---------------- 写低精度权重文件 ----------------
        LowpFile lf;
        std::memcpy(lf.header.magic, "LPQLOWP", 8);
        lf.header.version = 1;
        // 文件头里的 format 编码与 QuantFormat 的枚举值一致: 0 = mxfp8, 1 = nvfp4,
        // 2 = mxfp4(见 include/lpq/lowp_tensor.hpp 的 LowpHeader)
        lf.header.format = (cfg.format == QuantFormat::MxFP8)   ? 0u
                           : (cfg.format == QuantFormat::MxFP4) ? 2u
                                                                : 1u;
        lf.header.elem_bits = (uint32_t)cfg.element_bits();
        lf.header.elem_kind = (uint32_t)elem_kind_of(cfg);
        lf.header.scale_kind = cfg.is_mx() ? 0u : 1u;
        lf.header.scale_mode = (cfg.scale_mode == ScaleMode::Tensor) ? 0u : 1u;
        lf.header.rounding = (cfg.rounding == Rounding::Nearest) ? 0u : 1u;
        lf.header.has_global_scale = cfg.is_mx() ? 0u : 1u;
        lf.header.num_rows = host.rows;
        lf.header.num_cols = host.cols;
        lf.header.block_size = q.block_size;
        lf.header.num_blocks = q.num_blocks;
        lf.header.packed_bytes = (int64_t)q.packed.size();
        lf.header.scale_bytes = (int64_t)q.scales.size();
        lf.header.global_scale = q.global_scale_value;
        lf.header.global_scale_inv = q.global_scale_inv;
        lf.header.seed = cfg.seed;
        lf.packed.resize((size_t)lf.header.packed_bytes);
        lf.scales.resize((size_t)lf.header.scale_bytes);
        if (!lf.packed.empty()) {
            LPQ_CUDA_CHECK(cudaMemcpy(
                lf.packed.data(), q.packed.get(), lf.packed.size(), cudaMemcpyDeviceToHost));
        }
        if (!lf.scales.empty()) {
            LPQ_CUDA_CHECK(cudaMemcpy(
                lf.scales.data(), q.scales.get(), lf.scales.size(), cudaMemcpyDeviceToHost));
        }

        const std::string out_dir = !args.out_dir.empty() ? args.out_dir : parent_dir(args.input);
        const std::string prefix =
            !args.out_prefix.empty() ? args.out_prefix : basename_noext(args.input);
        const std::string lowp_path = join_path(out_dir, prefix + "_lowp.bin");
        const std::string deq_path = join_path(out_dir, prefix + "_dequant.bin");
        const std::string log_path = join_path(out_dir, prefix + "_log.txt");

        write_lowp_file(lowp_path, lf);

        // ---------------- 写反量化结果 ----------------
        std::vector<uint8_t> deq_host(d_output.size());
        if (!deq_host.empty()) {
            LPQ_CUDA_CHECK(cudaMemcpy(
                deq_host.data(), d_output.get(), deq_host.size(), cudaMemcpyDeviceToHost));
        }
        write_raw_file(deq_path, deq_host.data(), deq_host.size());

        // ---------------- 可选:独立 CPU 参考实现比对 ----------------
        bool cpu_ok = false, cpu_ran = false;
        int64_t cpu_dequant_mismatch = 0;
        double cpu_max_abs_diff = 0.0;
        double cpu_scale_match = 0.0;
        double cpu_quant_ms = 0.0, cpu_dequant_ms = 0.0;
        std::vector<float> cpu_deq_f32;
        if (args.cpu_check) {
            cpu_ran = true;
            std::vector<float> xf((size_t)numel);
            if (host.dtype == InputType::FP32) {
                std::memcpy(xf.data(), host.data.data(), (size_t)numel * 4);
            } else {
                const uint16_t* h = reinterpret_cast<const uint16_t*>(host.data.data());
                for (int64_t i = 0; i < numel; ++i)
                    xf[(size_t)i] = f32_from_half_bits(h[i]);
            }
            WallTimer cpu_wall;
            cpu_wall.start();
            const ref::CpuQuantized cpuq = ref::cpu_quantize(xf.data(), numel, cfg);
            cpu_quant_ms = cpu_wall.stop_ms();
            cpu_ok = (cpuq.packed.size() == lf.packed.size()) &&
                     (std::memcmp(cpuq.packed.data(), lf.packed.data(), lf.packed.size()) == 0);
            cpu_scale_match =
                (cpuq.scales.size() == lf.scales.size()) &&
                        (std::memcmp(cpuq.scales.data(), lf.scales.data(), lf.scales.size()) == 0)
                    ? 1.0
                    : 0.0;
            cpu_wall.start();
            cpu_deq_f32 = ref::cpu_dequantize(cpuq, numel, cfg);
            cpu_dequant_ms = cpu_wall.stop_ms();
            // Compare with the GPU dequantized result: round the CPU fp32 value with the same
            // converter into the output type and compare bitwise (implementation agreement,
            const uint16_t* g16 = reinterpret_cast<const uint16_t*>(deq_host.data());
            const uint32_t* g32 = reinterpret_cast<const uint32_t*>(deq_host.data());
            for (int64_t i = 0; i < numel; ++i) {
                const float cv = cpu_deq_f32[(size_t)i];
                uint32_t cb = 0, gb = 0;
                if (cfg.output_type == OutputType::FP32) {
                    cb = f32_to_bits(cv);
                    gb = g32[i];
                } else if (cfg.output_type == OutputType::FP16) {
                    cb = f32_to_fmt<FmtF16>(cv);
                    gb = g16[i];
                } else {
                    cb = f32_to_fmt<FmtBF16>(cv);
                    gb = g16[i];
                }
                if (cb != gb) {
                    ++cpu_dequant_mismatch;
                    const float gv = (cfg.output_type == OutputType::FP32)
                                         ? bits_to_f32(gb)
                                         : ((cfg.output_type == OutputType::FP16)
                                                   ? f32_from_half_bits((uint16_t)gb)
                                                   : f32_from_bf16_bits((uint16_t)gb));
                    cpu_max_abs_diff = std::max(cpu_max_abs_diff, (double)std::fabs(gv - cv));
                }
            }
        }

        // ---------------- 写日志 ----------------
        const double in_bytes = (double)numel * (double)in_elem;
        const double lowp_bytes =
            (double)(lf.header.packed_bytes + lf.header.scale_bytes) + 4.0;  // 含 FP32 全局缩放
        const double q_bytes = in_bytes + lowp_bytes;
        const double dq_bytes =
            lowp_bytes + (double)numel * ((cfg.output_type == OutputType::FP32) ? 4.0 : 2.0);
        std::ostringstream os;
        os.setf(std::ios::fixed);
        os << "# Low-precision format software simulation and dequantization -- error/perf log\n";
        os << "# generator: lpq_quant (CUDA)\n";
        os << "# note: kernel time is the min/mean over " << args.repeat << " measurements;\n";
        os << "#       timing uses CUDA events and covers GPU kernels only, excluding file IO and "
              "H2D/D2H copies;\n";
        os << "#       preprocess_ms includes file read and H2D copy; pipeline_ms is wall-clock "
              "time.\n";
        os << "#       cuda_context_init_ms covers process start through the one-time CUDA runtime "
              "bring-up.\n";
        os << "\n[config]\n" << quant_config_to_string(cfg);
        os << "device = " << gpu_name() << "\n";
        os << "\n[input]\n";
        os << "file = " << args.input << "\n";
        os << "num_rows = " << host.rows << "\n";
        os << "num_cols = " << host.cols << "\n";
        os << "dtype = " << to_string(host.dtype) << "\n";
        os << "numel = " << numel << "\n";
        os << "input_bytes = " << (long long)in_bytes << "\n";
        os << "\n[lowp]\n";
        os << "file = " << lowp_path << "\n";
        os << "element_format = " << elem_kind_name(lf.header.elem_kind) << "\n";
        os << "scale_format = " << scale_kind_name(lf.header.scale_kind) << "\n";
        os << "packed_bytes = " << lf.header.packed_bytes << "\n";
        os << "scale_bytes = " << lf.header.scale_bytes << "\n";
        os << "global_scale = " << lf.header.global_scale << "\n";
        os << "total_lowp_bytes = " << (long long)lowp_bytes << "\n";
        os << "compression_ratio = " << (lowp_bytes > 0 ? in_bytes / lowp_bytes : 0.0)
           << "  (original bytes / lowp bytes)\n";
        os << "bits_per_element = " << (8.0 * lowp_bytes / (double)numel) << "\n";
        os << "\n[error]\n";
        os << "max_abs_error = " << stats.max_abs << "\n";
        os << "mae = " << stats.mae << "\n";
        os << "mse = " << stats.mse << "\n";
        os << "rmse = " << stats.rmse << "\n";
        os << "relative_mse = " << stats.rel_mse << "\n";
        os << "mean_signed_error = " << stats.mean_signed_error << "\n";
        os << "signal_rms = " << std::sqrt(stats.rel_mse > 0 ? stats.mse / stats.rel_mse : 0.0)
           << "\n";
        os << "\n[performance]\n";
        os << "cuda_context_init_ms = " << cuda_context_ms << "\n";
        os << "preprocess_ms = " << preprocess_ms << "\n";
        os << "amax_kernel_ms = " << t_amax.min << "\n";
        os << "scale_kernel_ms = " << t_scale.min << "\n";
        os << "quantize_kernel_ms = " << t_quant.min << "\n";
        os << "quantize_total_ms = " << (t_amax.min + t_scale.min + t_quant.min) << "\n";
        os << "dequantize_kernel_ms = " << t_dequant.min << "\n";
        os << "metrics_kernel_ms = " << t_metrics.min << "\n";
        os << "metrics_accum = "
           << (args.metrics_precise ? "precise(per-element fp64)" : "fast(compensated fp32)")
           << "\n";
        os << "error_stats_fused_into_quantize = " << (stats_fused ? "yes" : "no") << "\n";
        if (stats_fused) {
            os << "error_stats_fused_note = accumulated inside the quantize kernel; "
                  "zero extra DRAM traffic\n";
        }
        os << "pipeline_wall_ms = " << t_pipeline.min << "\n";
        os << "quantize_kernel_ms_mean = " << t_quant.mean() << "\n";
        os << "dequantize_kernel_ms_mean = " << t_dequant.mean() << "\n";
        os << "quantize_input_passes = " << q.input_passes << "\n";
        os << "effective_bandwidth_quantize_GBps = "
           << (q_bytes / (t_amax.min + t_scale.min + t_quant.min) / 1.0e6)
           << "  (input counted once)\n";
        // 真实显存流量: 输入按实际读取遍数计, 再加上写出的低精度数据
        os << "dram_bandwidth_quantize_GBps = "
           << (((double)q.input_passes * in_bytes + lowp_bytes) /
                  (t_amax.min + t_scale.min + t_quant.min) / 1.0e6)
           << "\n";
        os << "effective_bandwidth_dequantize_GBps = " << (dq_bytes / t_dequant.min / 1.0e6)
           << "\n";
        if (cpu_ran) {
            os << "\n[accuracy_check_cpu_reference]\n";
            os << "cpu_reference = independent table-based CPU reference (src/reference.cu)\n";
            os << "packed_bits_identical = " << (cpu_ok ? "yes" : "no") << "\n";
            os << "scales_identical = " << (cpu_scale_match > 0.5 ? "yes" : "no") << "\n";
            os << "dequant_bit_mismatch_count = " << cpu_dequant_mismatch << "\n";
            os << "dequant_max_abs_diff_gpu_vs_cpu = " << cpu_max_abs_diff << "\n";
            // GPU 与 CPU 同级别实现的耗时对比(项目代码要求: GPU 需承担显著的部分,
            // 且总体性能需超越 CPU 实现). CPU 侧为单线程串行标量实现.
            os << "cpu_reference_single_thread = yes\n";
            os << "cpu_quantize_ms = " << cpu_quant_ms << "\n";
            os << "cpu_dequantize_ms = " << cpu_dequant_ms << "\n";
            const double gpu_q_ms = t_amax.min + t_scale.min + t_quant.min;
            os << "speedup_quantize_gpu_vs_cpu = " << (gpu_q_ms > 0 ? cpu_quant_ms / gpu_q_ms : 0.0)
               << "\n";
            os << "speedup_dequantize_gpu_vs_cpu = "
               << (t_dequant.min > 0 ? cpu_dequant_ms / t_dequant.min : 0.0) << "\n";
        }
        std::ofstream ofs(log_path);
        if (!ofs)
            throw Error("cannot write log file: " + log_path, __FILE__, __LINE__);
        ofs << os.str();
        ofs.close();

        if (!args.quiet) {
            std::printf("%s\n", os.str().c_str());
            std::printf("lowp file : %s\ndequant   : %s\nlog       : %s\n", lowp_path.c_str(),
                deq_path.c_str(), log_path.c_str());
        }
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}