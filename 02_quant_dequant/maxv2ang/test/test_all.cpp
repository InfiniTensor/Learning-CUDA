// ============================================================================
// 选题二「MXFP8 / NVFP4 低精度模拟与反量化」—— 测试主程序（C++）
// ============================================================================
// 在有 GPU 的机器上，工程根目录一行命令跑完全套：
//   make test
//
// 被测对象是 include/ + src/ 构成的库（src/quant_gpu.cu 里的 kernel 与驱动），
// 本程序只是它的调用方 —— 不存在"测试另写一套实现"。
//
// 流程：环境 → 正确性自测 → 生成输入 → 两种格式 × 三种矩阵 → 边界尺寸 → 汇总
//
// 产出（默认写在 test/out/，已被 .gitignore 忽略）：
//   summary.txt                环境信息 + 汇总表 + 逐组合完整指标
//   <格式>/<输入>/metrics.json  误差 / 压缩率 / kernel 时间 / 带宽 / 对拍结果
//   <格式>/<输入>/weights.bin   低精度权重（LPW1）
//   <格式>/<输入>/dequant.bin   反量化输出
//
// 正确性优先：自测或任何组合的对拍失败，立即中止（先对，再快）。
// 性能分析工具（nsys / ncu）不在本程序内 —— 那是 test/run.sh 的职责。
// ============================================================================

#include "quant_gpu.h"

#include "anchors.h"
#include "gen_input.h"

#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace {

const char *kOutDefault = "test/out";
const int kRows = 1024;    // 性能档规模（与 CPU 基线同规模）
const int kCols = 1024;
const int kCovSize = 256;  // 覆盖档规模：够跑全维度，又不至于写爆磁盘

void hr(const std::string &title) {
    std::cout << "\n\033[1m==== " << title << " ====\033[0m\n";
}

// 生成输入矩阵（已存在则跳过，便于重复运行）
int gen_input(const std::string &path, const std::string &dist, int rows, int cols,
              const std::string &dtype = "fp32") {
    if (std::ifstream(path).good())
        return 0;
    gen_input_file(path, dist, rows, cols, dtype);
    return 0;
}

// ---- GPU 自测：手算向量 + 随机数据，全格式 × scale_mode 走一遍 GPU 全流程 ----

static int gpu_self_test() {
    std::cout << "\n== GPU 自测：手算向量 + 随机数据，全格式 × scale_mode ==\n";
    struct Combo {
        const char *format, *smode, *elem;
        int bs;
    };
    std::mt19937_64 rng(42);
    std::vector<float> rnd(4096), nor(4096);
    std::uniform_real_distribution<double> ud(0.0, 1.0);
    std::normal_distribution<double> nd(0.0, 1.0);
    for (auto &x : rnd)
        x = (float)ud(rng);
    for (auto &x : nor)
        x = (float)nd(rng);

    int fails = 0;
    for (Combo c : {Combo{"mxfp8", "block", "e4m3", 32}, Combo{"mxfp8", "tensor", "e4m3", 32},
                    Combo{"mxfp8", "block", "e5m2", 32}, Combo{"mxfp8", "tensor", "e5m2", 32},
                    Combo{"nvfp4", "block", "e2m1", 16}, Combo{"nvfp4", "tensor", "e2m1", 16}}) {
        for (int data = 0; data < 3; data++) {
            std::vector<float> w;
            const char *tag;
            if (data == 0) { // 手算向量（参考实现已验证可精确还原）
                tag = "手算向量";
                w.assign(128, 0.0f);
                if (c.format == std::string("mxfp8")) {
                    // 每 32 一块: 8,4,2,1,0.5,... 重复（全 2 的幂 → E8M0=127, 精确还原）
                    for (int r = 0; r < 4; r++)
                        for (int k = 0; k < 8; k++)
                            w[(size_t)(r * 32 + k * 4)] = (float)ldexp(8.0, -k);
                } else {
                    // 每 16 一块: 448×{0,0.5,1,1.5,2,3,4,6} 重复（gs=1, s_block=448, 精确还原）
                    const double grid[8] = {0, 0.5, 1, 1.5, 2, 3, 4, 6};
                    for (int r = 0; r < 8; r++)
                        for (int k = 0; k < 8; k++)
                            w[(size_t)(r * 16 + k * 2)] = (float)(448.0 * grid[k]);
                }
            } else if (data == 1) {
                w = rnd;
                tag = "均匀随机";
            } else {
                w = nor;
                tag = "正态";
            }
            Config cfg;
            cfg.format = c.format;
            cfg.scale_mode = c.smode;
            cfg.block_size = c.bs;
            cfg.elem_format = c.elem;
            GpuQuantResult gq = quantize_gpu(w, cfg, 1, true);
            GpuDequantResult gd = dequant_gpu(gq.q, OUT_FP32, 1, true);
            bool ok = gq.verify_match && gd.verify_match;
            if (!ok)
                fails++;
            qc_check(std::string(c.format) + "/" + c.smode + "/" + c.elem + " " + tag +
                         " (量化+反量化 GPU==CPU)",
                     ok);
        }
    }
    return fails;
}


// ---- 汇总：性能表 / 覆盖表 / 完整指标清单 ----

const char *vmark(const RunMetrics &r) {
    if (r.verify_disabled)
        return "(off)";
    return (r.verify_quant && r.verify_dequant) ? "OK" : "FAIL";
}

// 性能表：1024x1024 FP32→FP16，供报告的误差/压缩率/性能章节
void print_perf_table(std::ostream &os, const std::vector<RunMetrics> &recs) {
    os << "\n" << std::string(104, '=') << "\n[性能档] " << kRows << "x" << kCols
       << " 输入 fp32 → 输出 fp16，repeat 取最小\n"
       << std::string(104, '=') << "\n";
    os << std::left << std::setw(8) << "格式" << std::setw(11) << "矩阵" << std::setw(14)
       << "缩放/输出" << std::right << std::setw(11) << "量化ms" << std::setw(11) << "反量化ms"
       << std::setw(11) << "带宽GB/s" << std::setw(12) << "maxErr" << std::setw(12) << "MAE"
       << std::setw(10) << "压缩率" << std::setw(8) << "verify" << "\n";
    os << std::string(104, '-') << "\n";
    for (const RunMetrics &r : recs) {
        os << std::left << std::setw(8) << r.format << std::setw(11) << r.matrix << std::setw(14)
           << (r.scale_mode + "/" + r.output_type) << std::right << std::fixed << std::setprecision(4)
           << std::setw(11) << r.quantize_ms << std::setw(11) << r.dequant_ms << std::setw(11)
           << r.dequant_gbps << std::setprecision(5) << std::setw(12) << r.max_abs << std::setw(12)
           << r.mae << std::setprecision(3) << std::setw(10) << r.compression_ratio << std::setw(8)
           << vmark(r) << "\n";
    }
    os << std::string(104, '=') << "\n";
}

// 覆盖表：题目要求的「输入 fp32/fp16 × 输出 fp16/bf16/fp32」逐格跑通
void print_cover_table(std::ostream &os, const std::vector<RunMetrics> &recs) {
    os << "\n" << std::string(104, '=') << "\n[覆盖档] " << kCovSize << "x" << kCovSize
       << " 输入 dtype × 输出类型 × 舍入 × 格式 × 分布 全遍历（每组都与主机端逐字节对拍）\n"
       << std::string(104, '=') << "\n";
    os << std::left << std::setw(8) << "格式" << std::setw(8) << "输入" << std::setw(8) << "输出"
       << std::setw(12) << "舍入" << std::setw(11) << "分布" << std::right << std::setw(12) << "maxErr"
       << std::setw(12) << "MAE" << std::setw(12) << "RMSE" << std::setw(10) << "压缩率"
       << std::setw(8) << "verify" << "\n";
    os << std::string(104, '-') << "\n";
    size_t nfail = 0;
    for (const RunMetrics &r : recs) {
        if (*vmark(r) == 'F')
            nfail++;
        os << std::left << std::setw(8) << r.format << std::setw(8) << r.in_dtype << std::setw(8)
           << r.output_type << std::setw(12) << r.rounding << std::setw(11) << r.matrix << std::right
           << std::fixed << std::setprecision(5) << std::setw(12) << r.max_abs << std::setw(12)
           << r.mae << std::setw(12) << r.rmse << std::setprecision(3) << std::setw(10)
           << r.compression_ratio << std::setw(8) << vmark(r) << "\n";
    }
    os << std::string(104, '=') << "\n";
    os << "覆盖档组合数: " << recs.size() << "，对拍失败: " << nfail << "\n";
}

// 完整指标（机器可读，便于直接取证/复核）
void print_detail(std::ostream &os, const std::vector<RunMetrics> &recs) {
    os << "\n---- 逐组合完整指标 ----\n";
    for (const RunMetrics &r : recs) {
        os << "[" << r.format << "/" << r.in_dtype << "->" << r.output_type << "/" << r.rounding
           << "/" << r.matrix << "]" << " num_elements=" << r.num_elems << " (" << r.rows << "x"
           << r.cols << ")" << " max_abs_err=" << r.max_abs << " mae=" << r.mae << " mse=" << r.mse
           << " rmse=" << r.rmse << " dst_bits_per_elem=" << r.dst_bits
           << " compression_ratio=" << r.compression_ratio << " quantize_ms=" << r.quantize_ms
           << " quantize_gbps=" << r.quant_gbps << " dequantize_ms=" << r.dequant_ms
           << " dequantize_gbps=" << r.dequant_gbps << " verify=(quant="
           << (r.verify_quant ? "true" : "false") << ",dequant="
           << (r.verify_dequant ? "true" : "false") << ")"
           << (r.verify_disabled ? " [disabled]" : "") << " repeat=" << r.repeat << " timing=min\n";
    }
}

} // namespace

int main(int argc, char **argv) {
    std::string outdir = kOutDefault;
    int repeat = 10;
    bool perf_only = false;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--outdir" && i + 1 < argc)
            outdir = argv[++i];
        else if (a == "--repeat" && i + 1 < argc)
            repeat = std::stoi(argv[++i]);
        else if (a == "--perf-only")
            perf_only = true; // 只跑性能档：供 nsys / ncu profile，避免把 48 组跑好多遍
        else {
            std::cerr << "用法: test_all [--outdir DIR] [--repeat N] [--perf-only]\n";
            return 1;
        }
    }

    std::ostringstream env; // 环境信息，最后随汇总一起落盘
    env << "================ 选题二 测试结果 ================\n";

    // ---------------------------------------------------------------- 0. 环境
    hr("0. 环境");
    gpu_init(0);
    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));

    std::cout << "GPU      : " << prop.name << " (sm_" << prop.major << prop.minor << ", "
              << prop.multiProcessorCount << " SM)\n";
    std::cout << "矩阵规模 : 性能档 " << kRows << "x" << kCols << " / 覆盖档 " << kCovSize << "x"
              << kCovSize << "\n";
    std::cout << "repeat   : " << repeat << " 次取最小\n";
    std::cout << "输出目录 : " << outdir << "\n";
    env << "GPU         : " << prop.name << " (sm_" << prop.major << prop.minor << ", "
        << prop.multiProcessorCount << " SM)\n"
        << "矩阵规模    : 性能档 " << kRows << "x" << kCols << " fp32 / 覆盖档 " << kCovSize << "x"
        << kCovSize << "\n"
        << "repeat      : " << repeat << " 次取最小\n"
        << "时间口径    : cudaEvent 夹住 kernel 本体，取 " << repeat << " 次最小值\n";

    qc_mkdir(outdir);

    // ---------------------------------------------------------------- 1. 自测
    hr("1. 正确性自测（格式锚点 + GPU 全流程逐字节对拍）");
    run_anchor_tests();
    int fails = gpu_self_test();
    if (fails) {
        std::cerr << "\n\033[31m✘ 自测未通过（" << fails << " 项）——先修正确性，再看性能\033[0m\n";
        return 1;
    }
    std::cout << "\n\033[32m✔ 自测全部通过\033[0m\n";

    // ---------------------------------------------------------------- 2. 输入
    hr("2. 生成输入矩阵");
    const std::string dists[3] = {"random", "normal", "outlier"};

    // 性能档输入：1024x1024 FP32（与 CPU 基线同规模）
    std::vector<std::string> perf_inputs;
    for (const std::string &d : dists) {
        std::string p = outdir + "/in_" + d + ".bin";
        if (gen_input(p, d, kRows, kCols, "fp32") != 0) {
            std::cerr << "\n\033[31m✘ 生成 " << d << " 输入失败\033[0m\n";
            return 1;
        }
        perf_inputs.push_back(p);
    }
    std::cout << "  性能档: " << kRows << "x" << kCols << " fp32 × " << 3 << " 分布\n";

    // 覆盖档输入：256x256，fp32 与 fp16 各三种分布
    const std::string cover_dtypes[2] = {"fp32", "fp16"};
    for (const std::string &dt : cover_dtypes)
        for (const std::string &d : dists) {
            std::string p = outdir + "/cov_" + dt + "_" + d + ".bin";
            if (gen_input(p, d, kCovSize, kCovSize, dt) != 0) {
                std::cerr << "\n\033[31m✘ 生成 " << dt << "/" << d << " 输入失败\033[0m\n";
                return 1;
            }
        }
    std::cout << "  覆盖档: " << kCovSize << "x" << kCovSize << " × {fp32,fp16} × 3 分布\n";

    std::vector<RunMetrics> perf_recs, cover_recs;

    // 按 (格式, 输入 dtype, 输出类型, 舍入) 构造配置；block 缩放、seed 固定
    auto make_cfg = [](const std::string &fmt, const std::string &out_tp, const std::string &rnd) {
        Config cfg;
        cfg.format = fmt;
        cfg.block_size = (fmt == "mxfp8") ? 32 : 16;
        cfg.scale_mode = "block";
        cfg.elem_format = "e4m3";
        cfg.output_type = out_tp;
        cfg.rounding = rnd;
        cfg.seed = 42;
        return cfg;
    };

    // ------------------------------------------------ 3. 性能档（1024²，用配置文件）
    // 走 read_config，顺带覆盖「量化参数文件」这条题目要求
    hr("3. 性能档（1024×1024 fp32 → fp16，出误差/压缩率/性能表）");
    for (const std::string &fmt : {std::string("mxfp8"), std::string("nvfp4")}) {
        Config cfg = read_config(fmt == "mxfp8" ? "config_mxfp8.txt" : "config_nvfp4.txt");
        std::cout << "  " << fmt << " 配置: 块" << cfg.block_size << " " << cfg.scale_mode
                  << "-scale → " << cfg.output_type << " (" << cfg.rounding << ")\n";
        for (const std::string &in : perf_inputs) {
            RunMetrics row;
            if (run_one(in, cfg, outdir + "/perf/" + fmt, repeat, true, &row) != 0) {
                std::cerr << "\n\033[31m✘ 性能档 " << fmt << " / " << row.matrix
                          << " 对拍不一致\033[0m\n";
                return 1;
            }
            perf_recs.push_back(row);
            std::cout << "    " << row.matrix << ": 量化 " << std::fixed << std::setprecision(4)
                      << row.quantize_ms << " ms, 反量化 " << row.dequant_ms << " ms, 带宽 "
                      << std::setprecision(1) << row.dequant_gbps << " GB/s, MAE "
                      << std::setprecision(5) << row.mae << "  ✔\n";
        }
    }

    int cover_runs = 0;
    if (!perf_only) {
        // ------------------------------ 4. 覆盖档（256²，题目要求的输入/输出维度全遍历）
        hr("4. 覆盖档（输入 dtype × 输出类型 × 舍入 × 格式 × 分布）");
        // 4a：nearest —— 输入 {fp32,fp16} × 输出 {fp16,bf16,fp32}
        for (const std::string &fmt : {std::string("mxfp8"), std::string("nvfp4")})
            for (const std::string &in_dt : cover_dtypes)
                for (const char *out_tp : {"fp16", "bf16", "fp32"}) {
                    Config cfg = make_cfg(fmt, out_tp, "nearest");
                    for (const std::string &d : dists) {
                        std::string tag = "cov/" + fmt + "_" + in_dt + "_" + out_tp + "_nearest";
                        RunMetrics row;
                        if (run_one(outdir + "/cov_" + in_dt + "_" + d + ".bin", cfg, outdir + "/" + tag,
                                    repeat, true, &row) != 0) {
                            std::cerr << "\n\033[31m✘ 覆盖档 " << tag << "/" << d
                                      << " 对拍不一致\033[0m\n";
                            return 1;
                        }
                        cover_recs.push_back(row);
                        cover_runs++;
                    }
                    std::cout << "  " << fmt << " " << in_dt << "→" << out_tp << " nearest  3 分布 ✔\n";
                }
        // 4b：stochastic —— 走 GPU 全流程，验证随机舍入下设备端与主机端仍逐字节一致
        for (const std::string &fmt : {std::string("mxfp8"), std::string("nvfp4")}) {
            Config cfg = make_cfg(fmt, "fp32", "stochastic");
            for (const std::string &d : dists) {
                std::string tag = "cov/" + fmt + "_fp32_fp32_stochastic";
                RunMetrics row;
                if (run_one(outdir + "/cov_fp32_" + d + ".bin", cfg, outdir + "/" + tag, repeat, true,
                            &row) != 0) {
                    std::cerr << "\n\033[31m✘ 覆盖档 stochastic " << fmt << "/" << d
                              << " 对拍不一致\033[0m\n";
                    return 1;
                }
                cover_recs.push_back(row);
                cover_runs++;
            }
            std::cout << "  " << fmt << " fp32→fp32 stochastic  3 分布 ✔\n";
        }

        // ---------------------------------------------------------------- 5. 边界
        hr("5. 边界尺寸（非整块 257x131，覆盖尾部块路径）");
        {
            std::string odd = outdir + "/cov_odd.bin";
            if (gen_input(odd, "normal", 257, 131, "fp32") != 0) {
                std::cerr << "\n\033[31m✘ 生成边界尺寸输入失败\033[0m\n";
                return 1;
            }
            RunMetrics row;
            if (run_one(odd, make_cfg("nvfp4", "fp32", "nearest"), outdir + "/odd", 3, true, &row) != 0) {
                std::cerr << "\n\033[31m✘ 边界尺寸 257x131 失败\033[0m\n";
                return 1;
            }
            cover_recs.push_back(row);
            std::cout << "\n\033[32m✔ 257x131 通过\033[0m\n";
        }
    }

    // ---------------------------------------------------------------- 6. 汇总
    hr("6. 汇总");
    std::ostringstream sum;
    sum << env.str() << "\n";
    print_perf_table(sum, perf_recs);
    if (!perf_only)
        print_cover_table(sum, cover_recs);
    print_detail(sum, perf_recs);
    if (!perf_only)
        print_detail(sum, cover_recs);
    std::cout << sum.str();

    std::ofstream fo(outdir + "/summary.txt");
    if (fo)
        fo << sum.str();
    else
        std::cerr << "警告: 无法写入 " << outdir << "/summary.txt\n";

    hr("完成");
    std::cout << "性能档 " << perf_recs.size() << " 组"
              << (perf_only ? "（--perf-only，已跳过覆盖档与边界尺寸）"
                            : "，覆盖档 " + std::to_string(cover_recs.size()) + " 组（" +
                                  std::to_string(cover_runs) + " 次遍历）")
              << "\n"
              << "汇总已写入 " << outdir << "/summary.txt\n"
              << "原始数据：" << outdir << "/**/metrics.json\n"
              << "把 " << outdir << " 目录发回即可用于完善总结报告。\n";
    return 0;
}
