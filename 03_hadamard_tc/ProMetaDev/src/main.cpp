// main.cpp - 命令行入口
//
//   hadamard gen     <kind> <rows> <cols> <out> [--dtype fp32|fp16|bf16] [--seed S]
//   hadamard run     <data> <config> <out_prefix> [--ref <ref_file>] [--torch-ms T]
//   hadamard verify  <ref_file> <got_file>
//   hadamard selftest
//
// kind = uniform | normal | outlier
#include "hadamard.h"
#include "fp8.h"
#include "io.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <string>
#include <vector>

using namespace hd;

static void usage() {
    std::printf(
        "用法:\n"
        "  hadamard gen     <kind> <rows> <cols> <out> [--dtype fp32|fp16|bf16] [--seed S]\n"
        "  hadamard run     <data> <config> <out_prefix> [--ref <ref_file>] [--torch-ms T]\n"
        "  hadamard quant   <data> <config> <out_prefix>\n"
        "  hadamard verify  <ref_file> <got_file>\n"
        "  hadamard selftest\n"
        "kind = uniform | normal | outlier\n");
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
    Dtype dt = Dtype::FP32;
    uint32_t seed = 1234;
    for (int i = 6; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--seed" && i + 1 < argc)
            seed = (uint32_t)std::strtoul(argv[++i], nullptr, 10);
        else if (a == "--dtype" && i + 1 < argc) {
            std::string d = argv[++i];
            dt = (d == "fp16") ? Dtype::FP16 : (d == "bf16" ? Dtype::BF16 : Dtype::FP32);
        }
    }
    Tensor t = gen_tensor(rows, cols, kind, seed, dt);
    save_tensor(out, t, dt);
    std::printf("[gen] %s 数据 %lld x %lld (%s) -> %s\n", kind.c_str(), (long long)rows,
                (long long)cols, dtype_name(dt), out.c_str());
    return 0;
}

// ======================= verify =======================
static int cmd_verify(int argc, char** argv) {
    if (argc < 4) {
        usage();
        return 1;
    }
    Tensor ref = load_tensor(argv[2]);
    Tensor got = load_tensor(argv[3]);
    if (ref.data.empty() || got.data.empty() || ref.numel() != got.numel()) {
        std::fprintf(stderr, "[verify] 读取失败或形状不一致\n");
        return 2;
    }
    ErrMetrics m = compare_error(ref.data, got.data);
    std::printf("[verify] max_abs_error=%.6e  MAE=%.6e  MSE=%.6e  rel_L2=%.6e\n", m.max_abs, m.mae,
                m.mse, m.rel_l2);
    bool ok = m.max_abs <= 1e-3;
    std::printf("[verify] %s (阈值 1e-3)\n", ok ? "通过" : "超阈值");
    return ok ? 0 : 4;
}

// ======================= run =======================
static int cmd_run(int argc, char** argv) {
    if (argc < 5) {
        usage();
        return 1;
    }
    std::string data_path = argv[2], cfg_path = argv[3], prefix = argv[4];
    std::string ref_path;
    double torch_ms = 0.0;
    for (int i = 5; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--ref" && i + 1 < argc)
            ref_path = argv[++i];
        else if (a == "--torch-ms" && i + 1 < argc)
            torch_ms = std::atof(argv[++i]);
    }

    Config cfg = load_config(cfg_path);
    Tensor in = load_tensor(data_path);
    if (in.data.empty()) {
        std::fprintf(stderr, "[run] 数据读取失败\n");
        return 2;
    }
    if (in.cols != cfg.hadamard_size) {
        std::fprintf(stderr, "[run] 列数 %d != hadamard_size %d\n", in.cols, cfg.hadamard_size);
        return 2;
    }

    const int64_t total = in.numel();
    const double io_bytes_fht = (double)total * 4.0 * 2.0; // 读 + 写 fp32
    const double io_bytes_quant =
        (double)total * 4.0 + (double)total + (double)(total / cfg.block_size);
    std::printf("[run] 输入 %lld x %lld (%s)，hadamard_size=%d，scale=%.6g，量化=%s\n",
                (long long)in.rows, (long long)in.cols, dtype_name(in.dtype), cfg.hadamard_size,
                (double)cfg.scale, cfg.quantize_enable ? quant_name(cfg.quant_format) : "关闭");

    std::string log;
    char buf[512];
    auto add = [&](const char* fmt, ...) {
        va_list ap;
        va_start(ap, fmt);
        std::vsnprintf(buf, sizeof(buf), fmt, ap);
        va_end(ap);
        log += buf;
    };

    add("=== Hadamard 变换加速 日志 ===\n时间: %s\n", now_string().c_str());
    add("数据文件: %s\n参数文件: %s\n", data_path.c_str(), cfg_path.c_str());
    add("规模: rows=%lld  cols=%d  dtype=%s\n", (long long)in.rows, in.cols, dtype_name(in.dtype));
    add("hadamard_size: %d   scale: %.9g\n", cfg.hadamard_size, (double)cfg.scale);
    add("量化: %s   block_size: %d\n\n",
        cfg.quantize_enable ? quant_name(cfg.quant_format) : "关闭", cfg.block_size);

    // ---- CPU 参考：验证 FWHT 与直接矩阵乘一致 ----
    Tensor cpu_mm, cpu_fht_v;
    ErrMetrics e_cpu{};
    const bool do_cpu_mm = (in.rows <= 4096); // O(n^2) 参考只在小规模上跑
    if (do_cpu_mm) {
        cpu_matmul_ref(in, cpu_mm, cfg.scale);
        cpu_fht(in, cpu_fht_v, cfg.scale);
        e_cpu = compare_error(cpu_mm.data, cpu_fht_v.data);
        add("[CPU 参考交叉验证] FWHT vs 直接构造 H_n 矩阵乘\n");
        add("  max_abs_error = %.6e   (H[i][j] = (-1)^popcount(i&j)，O(n^2) 定义式)\n\n",
            e_cpu.max_abs);
    }

    // ---- GPU FHT ----
    Tensor gpu_out;
    PerfStats perf;
    std::string err;
    if (!gpu_fht(in, cfg, gpu_out, &perf.fht_ms, err)) {
        std::fprintf(stderr, "[run] GPU FHT 失败: %s\n", err.c_str());
        return 3;
    }
    perf.fht_bw = bandwidth_gbps(io_bytes_fht, perf.fht_ms);

    // ---- 「手动」矩阵乘基线（O(n^2)）：性能对照 + 大数据下的正确性参考 ----
    Tensor mm_out;
    double mm_ms = 0.0;
    const bool have_mm = gpu_matmul_baseline(in, cfg.scale, mm_out, &mm_ms, err);

    // ---- 与参考对比 ----
    ErrMetrics e_mm{}, e_ref{};
    if (do_cpu_mm)
        e_mm = compare_error(cpu_mm.data, gpu_out.data);
    else if (have_mm)
        e_mm = compare_error(mm_out.data, gpu_out.data);
    if (!ref_path.empty()) {
        Tensor ref = load_tensor(ref_path);
        if (ref.numel() == gpu_out.numel())
            e_ref = compare_error(ref.data, gpu_out.data);
    }

    // ---- 量化：分离式 vs 融合式 ----
    QuantResult q_sep, q_fused;
    bool have_quant = cfg.quantize_enable;
    if (have_quant) {
        // 分离式：先 FHT（已计时），再对结果单独量化
        if (!gpu_quantize(gpu_out, cfg, q_sep, &perf.quant_ms, err)) {
            std::fprintf(stderr, "[run] 量化 kernel 失败: %s\n", err.c_str());
            return 3;
        }
        perf.quant_bw = bandwidth_gbps(io_bytes_quant, perf.quant_ms);

        // 融合式：FHT + 量化一次完成
        if (cfg.block_size == 32) {
            if (!gpu_fht_quant_fused(in, cfg, q_fused, &perf.fused_ms, err)) {
                std::fprintf(stderr, "[run] 融合 kernel 失败: %s\n", err.c_str());
                return 3;
            }
            perf.fused_bw = bandwidth_gbps(io_bytes_quant, perf.fused_ms);
            perf.fusion_speedup =
                (perf.fused_ms > 0.0) ? (perf.fht_ms + perf.quant_ms) / perf.fused_ms : 0.0;
        }

        // 量化误差：反量化 vs 变换后数据
        Tensor deq;
        cpu_dequant(q_sep, in.rows, in.cols, cfg.block_size, cfg.quant_format, deq);
        ErrMetrics e_q = compare_error(gpu_out.data, deq.data);
        add("[量化误差] (反量化结果 vs 变换后数据)\n");
        add("  max_abs_error = %.6e\n  MAE = %.6e\n  MSE = %.6e\n  rel_L2 = %.6e\n\n", e_q.max_abs,
            e_q.mae, e_q.mse, e_q.rel_l2);

        // 融合与分离的一致性
        if (!q_fused.qdata.empty()) {
            size_t diff = 0;
            for (size_t i = 0; i < q_sep.qdata.size() && i < q_fused.qdata.size(); i++)
                if (q_sep.qdata[i] != q_fused.qdata[i])
                    diff++;
            size_t sdiff = 0;
            for (size_t i = 0; i < q_sep.scales.size() && i < q_fused.scales.size(); i++)
                if (q_sep.scales[i] != q_fused.scales[i])
                    sdiff++;
            add("[融合一致性] 融合 vs 分离: FP8 码流不一致 %zu / %zu，缩放因子不一致 %zu / %zu\n\n",
                diff, q_sep.qdata.size(), sdiff, q_sep.scales.size());
        }
    }

    // ---- 性能 ----
    add("[性能]\n");
    add("FHT kernel 时间 (ms):          %.6f\n", perf.fht_ms);
    add("FHT 有效带宽 (GB/s):           %.2f\n", perf.fht_bw);
    if (have_quant) {
        add("量化 kernel 时间 (ms):         %.6f\n", perf.quant_ms);
        add("量化有效带宽 (GB/s):           %.2f\n", perf.quant_bw);
        if (perf.fused_ms > 0) {
            add("融合 kernel 时间 (ms):         %.6f\n", perf.fused_ms);
            add("融合有效带宽 (GB/s):           %.2f\n", perf.fused_bw);
            add("融合加速比 (FHT+量化)/融合:    %.3fx\n", perf.fusion_speedup);
            add("融合相对 FHT 单独执行的时间节省: %.2f%%\n",
                100.0 * (1.0 - perf.fused_ms / std::max(perf.fht_ms, 1e-12)));
        }
    }
    if (have_mm) {
        add("GPU 朴素矩阵乘基线时间 (ms):   %.6f\n", mm_ms);
        add("FHT 相对矩阵乘基线加速比:      %.2fx\n",
            perf.fht_ms > 0.0 ? mm_ms / perf.fht_ms : 0.0);
        add("  (两者 FLOP 比约 n/log2(n) = %.1fx，FHT 的算法收益)\n",
            (double)cfg.hadamard_size / std::log2((double)cfg.hadamard_size));
    }
    if (torch_ms > 0.0) {
        perf.torch_ms = torch_ms;
        perf.vs_torch_speedup = (perf.fht_ms > 0.0) ? torch_ms / perf.fht_ms : 0.0;
        add("PyTorch 基线时间 (ms):         %.6f\n", torch_ms);
        add("相对 PyTorch 加速比:           %.2fx\n", perf.vs_torch_speedup);
    }
    add("\n");

    // ---- 正确性 ----
    add("[正确性验证]\n");
    if (do_cpu_mm)
        add("GPU FHT vs CPU 矩阵乘参考: max_abs_error = %.6e  -> %s\n", e_mm.max_abs,
            e_mm.max_abs <= 1e-3 ? "通过 (<=1e-3)" : "超阈值");
    else if (have_mm)
        add("GPU FHT vs GPU 矩阵乘基线(O(n^2)): max_abs_error = %.6e  -> %s\n", e_mm.max_abs,
            e_mm.max_abs <= 1e-3 ? "通过 (<=1e-3)" : "超阈值");
    else
        add("矩阵乘参考: 不可用\n");
    if (!ref_path.empty())
        add("GPU FHT vs 外部参考(%s): max_abs_error = %.6e  -> %s\n", ref_path.c_str(),
            e_ref.max_abs, e_ref.max_abs <= 1e-3 ? "通过 (<=1e-3)" : "超阈值");
    add("\n");

    // ---- 输出 ----
    std::string fht_path = prefix + ".fht.bin";
    std::string quant_path = prefix + ".quant.bin";
    save_tensor(fht_path, gpu_out, Dtype::FP32);
    add("[输出]\nFHT 结果: %s\n", fht_path.c_str());
    if (have_quant) {
        QuantHeader h{};
        std::memcpy(h.magic, "HQ01", 4);
        h.rows = in.rows;
        h.cols = in.cols;
        h.block_size = cfg.block_size;
        h.quant_format = (int32_t)((cfg.quant_format == QuantFormat::FP8_E5M2) ? 1 : 0);
        h.scale = cfg.scale;
        h.data_bytes = (int64_t)q_sep.qdata.size();
        h.scale_bytes = (int64_t)q_sep.scales.size();
        save_quant_file(quant_path, h, q_sep);
        add("量化结果: %s\n", quant_path.c_str());
    }

    std::printf("\n%s", log.c_str());
    FILE* fp = std::fopen((prefix + ".log").c_str(), "w");
    if (fp) {
        std::fputs(log.c_str(), fp);
        std::fclose(fp);
    }

    bool ok = true;
    if (do_cpu_mm && e_mm.max_abs > 1e-3)
        ok = false;
    if (!ref_path.empty() && e_ref.max_abs > 1e-3)
        ok = false;
    std::printf("[run] %s\n", ok ? "通过：误差在阈值内 (<=1e-3)" : "警告：误差超阈值");
    return ok ? 0 : 4;
}

// ======================= quant（不做变换，直接量化）=======================
// 用于与「先旋转再量化」对比，验证 Hadamard 旋转能否降低量化误差
static int cmd_quant(int argc, char** argv) {
    if (argc < 5) {
        usage();
        return 1;
    }
    std::string data_path = argv[2], cfg_path = argv[3], prefix = argv[4];
    Config cfg = load_config(cfg_path);
    Tensor in = load_tensor(data_path);
    if (in.data.empty()) {
        std::fprintf(stderr, "[quant] 数据读取失败\n");
        return 2;
    }

    QuantResult qr;
    double ms = 0.0;
    std::string err;
    if (!gpu_quantize(in, cfg, qr, &ms, err)) {
        std::fprintf(stderr, "[quant] 失败: %s\n", err.c_str());
        return 3;
    }
    Tensor deq;
    cpu_dequant(qr, in.rows, in.cols, cfg.block_size, cfg.quant_format, deq);
    ErrMetrics e = compare_error(in.data, deq.data);

    const int64_t total = in.numel();
    const double bytes =
        (double)total * 4.0 + (double)total + (double)(total / std::max(cfg.block_size, 1));
    std::printf("=== 直接量化（无旋转）===\n");
    std::printf("规模: rows=%d cols=%d  量化: %s  block=%d\n", in.rows, in.cols,
                quant_name(cfg.quant_format), cfg.block_size);
    std::printf("量化 kernel 时间 (ms): %.6f   带宽 (GB/s): %.2f\n", ms, bandwidth_gbps(bytes, ms));
    std::printf("量化误差: max_abs=%.6e  MAE=%.6e  rel_L2=%.6e\n", e.max_abs, e.mae, e.rel_l2);

    QuantHeader h{};
    std::memcpy(h.magic, "HQ01", 4);
    h.rows = in.rows;
    h.cols = in.cols;
    h.block_size = cfg.block_size;
    h.quant_format = (int32_t)((cfg.quant_format == QuantFormat::FP8_E5M2) ? 1 : 0);
    h.scale = 1.0f;
    h.data_bytes = (int64_t)qr.qdata.size();
    h.scale_bytes = (int64_t)qr.scales.size();
    save_quant_file(prefix + ".quant.bin", h, qr);
    FILE* fp = std::fopen((prefix + ".log").c_str(), "w");
    if (fp) {
        std::fprintf(fp,
                     "=== 直接量化（无旋转）===\nrows=%d cols=%d quant=%s block=%d\n"
                     "quant_ms=%.6f bw_GBps=%.2f\nmax_abs=%.9e MAE=%.9e rel_L2=%.9e\n",
                     in.rows, in.cols, quant_name(cfg.quant_format), cfg.block_size, ms,
                     bandwidth_gbps(bytes, ms), e.max_abs, e.mae, e.rel_l2);
        std::fclose(fp);
    }
    return 0;
}

// ======================= selftest =======================
static int cmd_selftest() {
    int fails = 0;
    auto check = [&](bool ok, const char* what) {
        std::printf("  [%s] %s\n", ok ? " ok " : "FAIL", what);
        if (!ok)
            fails++;
    };
    std::printf("== Hadamard 变换引擎自检 ==\n");

    // 1) FP8 编解码往返
    {
        int bad = 0;
        for (int c = 0; c < 256; c++) {
            if ((c & 0x7F) == 0x7F)
                continue;
            if (f32_to_e4m3(e4m3_to_f32((uint8_t)c)) != (uint8_t)c)
                bad++;
        }
        check(bad == 0, "E4M3 全编码往返一致");
    }
    // 2) FWHT vs 直接矩阵乘（多个 n）
    for (int n : {64, 128, 256}) {
        Tensor in = gen_tensor(16, n, "normal", 7, Dtype::FP32);
        Tensor mm, fw;
        cpu_matmul_ref(in, mm, 1.0f);
        cpu_fht(in, fw, 1.0f);
        ErrMetrics e = compare_error(mm.data, fw.data);
        char msg[128];
        std::snprintf(msg, sizeof(msg), "CPU FWHT vs 矩阵乘定义式 n=%d (max_err=%.2e)", n,
                      e.max_abs);
        check(e.max_abs < 1e-3, msg);
    }
    // 3) GPU FHT vs CPU 矩阵乘，n ∈ {64,128,256}，scale=1/sqrt(n)
    for (int n : {64, 128, 256}) {
        Tensor in = gen_tensor(64, n, "normal", 11, Dtype::FP32);
        Config cfg;
        cfg.hadamard_size = n;
        cfg.scale = 1.0f / std::sqrt((float)n);
        Tensor mm;
        cpu_matmul_ref(in, mm, cfg.scale);
        Tensor g;
        double ms = 0;
        std::string err;
        bool ok = gpu_fht(in, cfg, g, &ms, err);
        if (!ok) {
            check(false, ("GPU FHT 执行失败: " + err).c_str());
            continue;
        }
        ErrMetrics e = compare_error(mm.data, g.data);
        char msg[160];
        std::snprintf(msg, sizeof(msg), "GPU FHT vs CPU 参考 n=%d scale=1/sqrt(n) (max_err=%.2e)",
                      n, e.max_abs);
        check(e.max_abs <= 1e-3, msg);
    }
    // 4) 融合与分离量化结果一致
    {
        Tensor in = gen_tensor(128, 128, "outlier", 5, Dtype::FP32);
        Config cfg;
        cfg.hadamard_size = 128;
        cfg.scale = 1.0f / std::sqrt(128.0f);
        cfg.quantize_enable = true;
        cfg.block_size = 32;
        Tensor fht;
        double ms = 0;
        std::string err;
        if (gpu_fht(in, cfg, fht, &ms, err)) {
            QuantResult qs, qf;
            double t1 = 0, t2 = 0;
            bool ok1 = gpu_quantize(fht, cfg, qs, &t1, err);
            bool ok2 = gpu_fht_quant_fused(in, cfg, qf, &t2, err);
            bool same = ok1 && ok2 && (qs.qdata == qf.qdata) && (qs.scales == qf.scales);
            check(same, "融合 FHT+量化 与 分离式结果逐字节一致");
        } else {
            check(false, "融合一致性测试的前置 FHT 失败");
        }
    }

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
    if (mode == "run")
        return cmd_run(argc, argv);
    if (mode == "quant")
        return cmd_quant(argc, argv);
    if (mode == "verify")
        return cmd_verify(argc, argv);
    if (mode == "selftest")
        return cmd_selftest();
    usage();
    return 1;
}
