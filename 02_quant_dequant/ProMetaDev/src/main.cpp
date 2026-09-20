// main.cpp - 命令行入口
//
//   lowprec gen   <kind> <rows> <cols> <out_path> [--fp16] [--seed N]
//   lowprec quant <tensor> <config> <out_prefix> [--no-cpu]
//   lowprec dequant <quant_file> <out_bin>
//
// kind = random | normal | outlier
#include "io.h"
#include "metrics.h"
#include "quant.h"

#include <cstdio>
#include <cstring>
#include <ctime>
#include <string>
#include <vector>

using namespace lowp;

static void usage() {
    std::printf("用法:\n"
                "  lowprec gen   <kind> <rows> <cols> <out_path> [--fp16] [--seed N]\n"
                "  lowprec quant <tensor> <config> <out_prefix> [--no-cpu]\n"
                "  lowprec dequant <quant_file> <out_bin>\n"
                "  lowprec selftest\n"
                "kind = random | normal | outlier | zeros\n");
}

static std::string now_string() {
    std::time_t t = std::time(nullptr);
    char buf[64];
    std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", std::localtime(&t));
    return buf;
}

// ======================= gen =======================
static int cmd_gen(int argc, char** argv) {
    if (argc < 6) {
        usage();
        return 1;
    }
    std::string kind = argv[2];
    int64_t rows = std::strtoll(argv[3], nullptr, 10);
    int64_t cols = std::strtoll(argv[4], nullptr, 10);
    std::string out = argv[5];
    bool fp16 = false;
    uint32_t seed = 1234;
    for (int i = 6; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--fp16")
            fp16 = true;
        else if (a == "--seed" && i + 1 < argc)
            seed = (uint32_t)std::strtoul(argv[++i], nullptr, 10);
    }
    Tensor t = gen_tensor(kind, rows, cols, seed, fp16);
    save_tensor(out, t);
    std::printf("[gen] 已生成 %s 张量 %lld x %lld -> %s\n", kind.c_str(), (long long)rows,
                (long long)cols, out.c_str());
    return 0;
}

// ======================= quant =======================
static int cmd_quant(int argc, char** argv) {
    if (argc < 5) {
        usage();
        return 1;
    }
    std::string tensor_path = argv[2];
    std::string config_path = argv[3];
    std::string prefix = argv[4];

    Config cfg = load_config(config_path);
    for (int i = 5; i < argc; i++)
        if (std::string(argv[i]) == "--no-cpu")
            cfg.no_cpu_ref = true;

    Tensor t = load_tensor(tensor_path);
    if (t.data.empty()) {
        std::fprintf(stderr, "[quant] 输入张量读取失败\n");
        return 2;
    }
    const int64_t n = t.numel();

    std::printf("[quant] 输入: %s  (%lld x %lld, %s)\n", tensor_path.c_str(), (long long)t.rows,
                (long long)t.cols, t.src_fp16 ? "fp16" : "fp32");
    std::printf("[quant] 格式: %s, block=%d, scale_mode=%s, out=%s, round=%s\n",
                format_name(cfg.format), cfg.block_size, scale_mode_name(cfg.scale_mode),
                out_dtype_name(cfg.out_dtype), round_mode_name(cfg.round));

    // ---- GPU 流程 ----
    QuantResult gpu;
    std::string err;
    if (!run_gpu(t.data, n, cfg, gpu, err)) {
        std::fprintf(stderr, "[quant] GPU 执行失败: %s\n", err.c_str());
        return 3;
    }

    // ---- CPU 参考 ----
    const bool have_cpu = !cfg.no_cpu_ref;
    MatchResult mr{};
    QuantResult cpu;
    if (have_cpu) {
        cpu = run_cpu(t.data, n, cfg);
        mr = compare_exact(cpu.dequant, gpu.dequant);
    }
    const double gpu_kernel_ms = gpu.quant_ms + gpu.dequant_ms;
    const double speedup = (have_cpu && gpu_kernel_ms > 0.0) ? cpu.cpu_ms / gpu_kernel_ms : 0.0;

    // ---- 误差 ----
    // 纯量化误差：在 fp32 域比较，不含输出类型舍入与量程限制
    const std::vector<float> deq_fp32 =
        host_dequant_fp32(gpu.packed, gpu.scales, gpu.global_scale, n, gpu.block_size,
                          cfg.format == LowFormat::NVFP4, cfg.elem);
    const ErrMetrics em_quant = compute_error(t.data, deq_fp32);
    // 端到端误差：含 fp16/bf16 输出舍入与量程限制（超出量程会得到 inf）
    const ErrMetrics em_e2e = compute_error(t.data, gpu.dequant);

    // ---- 性能与压缩率 ----
    const double in_bytes = (double)n * (t.src_fp16 ? 2.0 : 4.0);
    const double comp_bytes = (double)gpu.packed.size() + (double)gpu.scales.size() + 4.0;
    const double deq_bytes = (double)n * ((cfg.out_dtype == OutDtype::FP32) ? 4.0 : 2.0);
    const double comp_ratio = comp_bytes > 0 ? in_bytes / comp_bytes : 0.0;
    const double quant_bw = effective_bw_gbps(
        (double)n * 4.0, (double)gpu.packed.size() + (double)gpu.scales.size(), gpu.quant_ms);
    const double dequant_bw = effective_bw_gbps(
        (double)gpu.packed.size() + (double)gpu.scales.size(), deq_bytes, gpu.dequant_ms);

    // ---- 保存 ----
    QuantHeader h{};
    std::memcpy(h.magic, "LPQ1", 4);
    h.format = (int32_t)cfg.format;
    h.elem = (int32_t)(cfg.elem == ElemFmt::E4M3 ? 0 : 1);
    h.rows = t.rows;
    h.cols = t.cols;
    h.block_size = gpu.block_size;
    h.scale_mode = (int32_t)cfg.scale_mode;
    h.out_dtype = (int32_t)cfg.out_dtype;
    h.round_mode = (int32_t)cfg.round;
    h.num_blocks = gpu.num_blocks;
    h.global_scale = gpu.global_scale;
    h.packed_bytes = (int64_t)gpu.packed.size();
    h.scale_bytes = (int64_t)gpu.scales.size();

    std::string qpath = prefix + ".quant";
    std::string dpath = prefix + ".dequant.bin";
    std::string lpath = prefix + ".log";
    save_quant_file(qpath, h, gpu.packed.data(), gpu.scales.data());
    save_dequant_tensor(dpath, gpu.dequant, cfg.out_dtype);

    // 校验：从权重文件重新反量化，结果应与内存中的一致
    bool reload_ok = false;
    double reload_ms = 0.0;
    {
        QuantHeader rh{};
        std::vector<uint8_t> rp, rs;
        if (load_quant_file(qpath, rh, rp, rs)) {
            std::vector<float> re;
            std::string e2;
            if (run_gpu_dequant(rp, rs, rh.global_scale, n, rh.block_size, rh.format == 1, rh.elem,
                                (OutDtype)rh.out_dtype, re, &reload_ms, e2)) {
                reload_ok = (compare_exact(re, gpu.dequant).mismatch == 0);
            }
        }
    }

    // ---- 日志 ----
    std::string log;
    auto add = [&](const std::string& s) { log += s; };
    char buf[512];
    add("=== MXFP8 / NVFP4 低精度模拟与反量化 日志 ===\n");
    std::snprintf(buf, sizeof(buf), "时间: %s\n输入张量: %s\n形状: %lld x %lld\n输入类型: %s\n",
                  now_string().c_str(), tensor_path.c_str(), (long long)t.rows, (long long)t.cols,
                  t.src_fp16 ? "fp16" : "fp32");
    add(buf);
    std::snprintf(
        buf, sizeof(buf),
        "格式: %s%s\nblock_size: %d\nscale_mode: %s\n输出类型: %s\n舍入: %s\ntarget_gpu: %s\n",
        format_name(cfg.format),
        cfg.format == LowFormat::MXFP8 ? (cfg.elem == ElemFmt::E4M3 ? " (e4m3)" : " (e5m2)") : "",
        gpu.block_size, scale_mode_name(cfg.scale_mode), out_dtype_name(cfg.out_dtype),
        round_mode_name(cfg.round), cfg.target_gpu.c_str());
    add(buf);
    std::snprintf(buf, sizeof(buf), "num_blocks: %lld\nglobal_scale: %.9g\n\n",
                  (long long)gpu.num_blocks, gpu.global_scale);
    add(buf);

    add("[误差指标] (反量化结果 vs 原始输入)\n");
    std::snprintf(buf, sizeof(buf),
                  "纯量化误差 (fp32 域):\n"
                  "  max_abs_error: %.9g\n  MAE: %.9g\n  MSE: %.9g\n  RMSE: %.9g\n"
                  "  NMAE: %.9g\n  SQNR: %.4f dB\n",
                  em_quant.max_abs, em_quant.mae, em_quant.mse, em_quant.rmse, em_quant.nmae,
                  em_quant.sqnr_db);
    add(buf);
    std::snprintf(buf, sizeof(buf),
                  "端到端误差 (含 %s 输出舍入/量程):\n"
                  "  max_abs_error: %.9g\n  MAE: %.9g\n  MSE: %.9g\n  RMSE: %.9g\n"
                  "  NMAE: %.9g\n  SQNR: %.4f dB\n\n",
                  out_dtype_name(cfg.out_dtype), em_e2e.max_abs, em_e2e.mae, em_e2e.mse,
                  em_e2e.rmse, em_e2e.nmae, em_e2e.sqnr_db);
    add(buf);

    add("[压缩率]\n");
    std::snprintf(buf, sizeof(buf),
                  "原始数据字节: %.0f\n压缩后字节: %.0f  (packed=%lld, scales=%lld, "
                  "global=4)\n压缩率: %.4fx\n\n",
                  in_bytes, comp_bytes, (long long)h.packed_bytes, (long long)h.scale_bytes,
                  comp_ratio);
    add(buf);

    add("[性能]\n");
    std::snprintf(buf, sizeof(buf),
                  "量化 kernel 时间 (ms): %.4f\n反量化 kernel 时间 (ms): %.4f\n"
                  "量化有效带宽 (GB/s): %.2f\n反量化有效带宽 (GB/s): %.2f\n",
                  gpu.quant_ms, gpu.dequant_ms, quant_bw, dequant_bw);
    add(buf);
    if (have_cpu) {
        std::snprintf(buf, sizeof(buf),
                      "CPU 单线程参考总时间 (ms): %.3f\nGPU 加速比 (CPU / GPU kernel): %.1fx\n\n",
                      cpu.cpu_ms, speedup);
        add(buf);
    } else {
        add("CPU 参考: 已跳过 (--no-cpu)\n\n");
    }

    add("[正确性验证]\n");
    if (cfg.format == LowFormat::NVFP4) {
        int64_t expect = (n + 1) / 2;
        std::snprintf(buf, sizeof(buf),
                      "4bit 打包: packed_bytes=%lld == ceil(n/2)=%lld -> %s (每字节 2 个元素)\n",
                      (long long)h.packed_bytes, (long long)expect,
                      h.packed_bytes == expect ? "符合" : "不符合");
        add(buf);
    }
    if (have_cpu) {
        std::snprintf(buf, sizeof(buf), "CPU 参考 vs GPU: %s (mismatch=%lld, max_diff=%.3g)\n",
                      mr.mismatch == 0 ? "逐位一致" : "存在差异", (long long)mr.mismatch,
                      mr.max_diff);
        add(buf);
    } else {
        add("CPU 参考: 已跳过 (--no-cpu)\n");
    }
    std::snprintf(buf, sizeof(buf), "权重文件重载反量化: %s (重载反量化时间 %.4f ms)\n",
                  reload_ok ? "一致" : "不一致", reload_ms);
    add(buf);
    add("\n[输出文件]\n");
    add(std::string("低精度权重: ") + qpath + "\n");
    add(std::string("反量化张量: ") + dpath + "\n");

    std::printf("\n%s", log.c_str());
    FILE* fp = std::fopen(lpath.c_str(), "w");
    if (fp) {
        std::fputs(log.c_str(), fp);
        std::fclose(fp);
    }

    bool ok = (!have_cpu || mr.mismatch == 0) && reload_ok;
    std::printf("[quant] %s\n", ok ? "通过：CPU/GPU 一致且权重文件可重载" : "警告：存在不一致");
    return ok ? 0 : 4;
}

// ======================= dequant =======================
static int cmd_dequant(int argc, char** argv) {
    if (argc < 4) {
        usage();
        return 1;
    }
    std::string qpath = argv[2];
    std::string opath = argv[3];

    QuantHeader h{};
    std::vector<uint8_t> packed, scales;
    if (!load_quant_file(qpath, h, packed, scales))
        return 2;

    const int64_t n = h.rows * h.cols;
    std::vector<float> out;
    double ms = 0.0;
    std::string err;
    if (!run_gpu_dequant(packed, scales, h.global_scale, n, h.block_size, h.format == 1, h.elem,
                         (OutDtype)h.out_dtype, out, &ms, err)) {
        std::fprintf(stderr, "[dequant] GPU 执行失败: %s\n", err.c_str());
        return 3;
    }
    save_dequant_tensor(opath, out, (OutDtype)h.out_dtype);
    const double bytes = (double)packed.size() + (double)scales.size() +
                         (double)n * ((h.out_dtype == 2) ? 4.0 : 2.0);
    std::printf("[dequant] %s -> %s\n", qpath.c_str(), opath.c_str());
    std::printf("[dequant] 元素 %lld, kernel 时间 %.4f ms, 有效带宽 %.2f GB/s\n", (long long)n, ms,
                effective_bw_gbps((double)packed.size() + (double)scales.size(),
                                  (double)n * ((h.out_dtype == 2) ? 4.0 : 2.0), ms));
    (void)bytes;
    return 0;
}

// ======================= selftest =======================
static int cmd_selftest() {
    int fails = 0;
    auto check = [&](bool ok, const char* what) {
        if (!ok) {
            std::printf("  [FAIL] %s\n", what);
            fails++;
        } else {
            std::printf("  [ ok ] %s\n", what);
        }
    };
    std::printf("== 低精度格式编解码自检 ==\n");

    // ---- E4M3 ----
    int e4_bad = 0;
    for (int c = 0; c < 256; c++) {
        if ((c & 0x7F) == 0x7F)
            continue; // 0x7F / 0xFF 为 NaN
        if (f32_to_e4m3(e4m3_to_f32((uint8_t)c), 0, 0.0f) != (uint8_t)c)
            e4_bad++;
    }
    check(e4_bad == 0, "E4M3 全编码 解码->重编码 往返一致");
    check(e4m3_to_f32(0x7E) == 448.0f, "E4M3 最大可表示值 = 448");
    check(f32_to_e4m3(448.0f, 0, 0.0f) == 0x7E, "E4M3 encode(448) = 0x7E");
    check(f32_to_e4m3(1e30f, 0, 0.0f) == 0x7E, "E4M3 上溢饱和到 448");
    check(f32_to_e4m3(1.0f, 0, 0.0f) == 0x38, "E4M3 encode(1.0) = 0x38");
    check(e4m3_to_f32(0x38) == 1.0f, "E4M3 decode(0x38) = 1.0");

    // ---- E5M2 ----
    int e5_bad = 0;
    for (int c = 0; c < 256; c++) {
        if (((c >> 2) & 0x1F) == 0x1F)
            continue; // inf / NaN
        if (f32_to_e5m2(e5m2_to_f32((uint8_t)c), 0, 0.0f) != (uint8_t)c)
            e5_bad++;
    }
    check(e5_bad == 0, "E5M2 编解码往返一致");
    check(e5m2_to_f32(0x7B) == 57344.0f, "E5M2 最大可表示值 = 57344");

    // ---- E2M1 (FP4) ----
    const float expect2[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    int e2_bad = 0;
    for (int c = 0; c < 8; c++) {
        if (e2m1_mag_to_f32(c) != expect2[c])
            e2_bad++;
        if (e2m1_encode_mag(expect2[c], 0, 0.0f) != (uint32_t)c)
            e2_bad++;
    }
    check(e2_bad == 0, "E2M1 幅值表 {0,.5,1,1.5,2,3,4,6} 与往返");
    check(e2m1_encode_mag(6.5f, 0, 0.0f) == 7, "E2M1 上溢饱和到 6");
    check(e2m1_to_f32(0x0Fu) == -6.0f, "E2M1 符号位解码 (-6)");
    check(f32_to_e2m1(-6.0f, 0, 0.0f) == 0x0Fu, "E2M1 encode(-6) = 0x0F (sign=1, mag=7)");
    check(e2m1_encode_mag(0.25f, 0, 0.0f) == 0, "E2M1 中点 0.25 取偶数编码 -> 0");

    // ---- E8M0 ----
    int e8_bad = 0;
    for (int e = -127; e <= 127; e++)
        if (e8m0_to_f32(e8m0_from_exp(e)) != ldexpf(1.0f, e))
            e8_bad++;
    check(e8_bad == 0, "E8M0 指数往返 (2^e)");
    check(e8m0_to_f32(127) == 1.0f, "E8M0 decode(127) = 1.0");

    std::printf("== 自检%s：%d 项失败 ==\n", fails == 0 ? "通过" : "失败", fails);
    return fails == 0 ? 0 : 1;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        usage();
        return 1;
    }
    std::string mode = argv[1];
    if (mode == "gen")
        return cmd_gen(argc, argv);
    if (mode == "quant")
        return cmd_quant(argc, argv);
    if (mode == "dequant")
        return cmd_dequant(argc, argv);
    if (mode == "selftest")
        return cmd_selftest();
    usage();
    return 1;
}
