// main.cpp - 命令行入口
//
//   vs gen     <kind> <n> <dim> <out> [--nlist L] [--seed S]
//   vs genq    <base> <nq> <out> [--seed S]
//   vs exact   <base> <queries> <config> <out_prefix> [--no-cpu]
//   vs build   <base> <config> <index_file>
//   vs search  <base> <queries> <index> <config> <out_prefix> [--gt <gt_result>]
//   vs bench   <base> <queries> <index> <config> <out_prefix> --gt <gt_result>
//   vs selftest
//
// kind = uniform | normal | clustered
#include "io.h"
#include "metrics.h"
#include "search.h"

#include <algorithm>
#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <string>
#include <vector>

using namespace vs;

static void usage() {
    std::printf("用法:\n"
                "  vs gen     <kind> <n> <dim> <out> [--nlist L] [--seed S]\n"
                "  vs genq    <base> <nq> <out> [--seed S]\n"
                "  vs exact   <base> <queries> <config> <out_prefix> [--no-cpu]\n"
                "  vs build   <base> <config> <index_file>\n"
                "  vs search  <base> <queries> <index> <config> <out_prefix> [--gt <gt_result>]\n"
                "  vs bench   <base> <queries> <index> <config> <out_prefix> --gt <gt_result>\n"
                "  vs selftest\n"
                "kind = uniform | normal | clustered\n");
}

static std::string now_string() {
    std::time_t t = std::time(nullptr);
    char buf[64];
    std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", std::localtime(&t));
    return buf;
}

static void append(std::string& s, const char* fmt, ...) {
    char buf[1024];
    va_list ap;
    va_start(ap, fmt);
    std::vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    s += buf;
}

static double run_cpu_exact_timed(const VectorSet& base, const VectorSet& queries, int k, Metric m,
                                  SearchResult& out) {
    auto t0 = std::chrono::high_resolution_clock::now();
    out = cpu_exact_search(base, queries, k, m);
    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

// ======================= gen / genq =======================
static int cmd_gen(int argc, char** argv) {
    if (argc < 6) {
        usage();
        return 1;
    }
    std::string kind = argv[2];
    int64_t n = std::strtoll(argv[3], nullptr, 10);
    int32_t dim = (int32_t)std::atoi(argv[4]);
    std::string out = argv[5];
    int nlist = 0;
    uint32_t seed = 1234;
    Metric metric = Metric::L2;
    for (int i = 6; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--nlist" && i + 1 < argc)
            nlist = std::atoi(argv[++i]);
        else if (a == "--seed" && i + 1 < argc)
            seed = (uint32_t)std::strtoul(argv[++i], nullptr, 10);
        else if (a == "--metric" && i + 1 < argc) {
            std::string m = argv[++i];
            metric = (m == "inner_product") ? Metric::INNER_PRODUCT
                     : (m == "cosine")      ? Metric::COSINE
                                            : Metric::L2;
        }
    }
    VectorSet s = gen_vectors(n, dim, kind, seed, nlist);
    s.metric = metric;
    save_vectors(out, s);
    std::printf("[gen] %s 向量库 %lld x %d -> %s\n", kind.c_str(), (long long)n, dim, out.c_str());
    return 0;
}

static int cmd_genq(int argc, char** argv) {
    if (argc < 5) {
        usage();
        return 1;
    }
    std::string base_path = argv[2];
    int64_t nq = std::strtoll(argv[3], nullptr, 10);
    std::string out = argv[4];
    uint32_t seed = 999;
    std::string kind = "normal";
    for (int i = 5; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--seed" && i + 1 < argc)
            seed = (uint32_t)std::strtoul(argv[++i], nullptr, 10);
        else if (a == "--kind" && i + 1 < argc)
            kind = argv[++i];
    }
    VectorSet base = load_vectors(base_path);
    if (base.data.empty()) {
        std::fprintf(stderr, "[genq] 无法读取向量库\n");
        return 2;
    }
    VectorSet q = gen_queries(nq, base, kind, seed);
    q.metric = base.metric;
    save_vectors(out, q);
    std::printf("[genq] 查询集 %lld x %d (kind=%s) -> %s\n", (long long)nq, base.dim, kind.c_str(),
                out.c_str());
    return 0;
}

// ======================= exact =======================
static int cmd_exact(int argc, char** argv) {
    if (argc < 6) {
        usage();
        return 1;
    }
    std::string base_path = argv[2], q_path = argv[3], cfg_path = argv[4], prefix = argv[5];
    Config cfg = load_config(cfg_path);
    for (int i = 6; i < argc; i++)
        if (std::string(argv[i]) == "--no-cpu")
            cfg.no_cpu = true;

    VectorSet base = load_vectors(base_path);
    VectorSet queries = load_vectors(q_path);
    if (base.data.empty() || queries.data.empty()) {
        std::fprintf(stderr, "[exact] 输入读取失败\n");
        return 2;
    }

    double cpu_ms = 0.0;
    SearchResult cpu;
    CompareResult cmp;
    if (!cfg.no_cpu) {
        cpu_ms = run_cpu_exact_timed(base, queries, cfg.top_k, base.metric, cpu);
    }

    SearchResult gpu;
    PerfStats perf;
    std::string err;
    if (!gpu_exact_search(base, queries, cfg, gpu, perf, cpu_ms > 0 ? &cpu_ms : nullptr, err)) {
        std::fprintf(stderr, "[exact] GPU 执行失败: %s\n", err.c_str());
        return 3;
    }
    if (!cfg.no_cpu)
        cmp = compare_results(gpu, cpu);

    save_result(prefix + ".result", gpu);

    std::string log;
    append(log, "=== GPU 向量检索引擎 · 精确检索日志 ===\n");
    append(log, "时间: %s\n", now_string().c_str());
    append(log, "向量库: %s\n查询集: %s\n", base_path.c_str(), q_path.c_str());
    append(log, "度量: %s   搜索模式: exact   top_k: %d\n", metric_name(base.metric), cfg.top_k);
    append(log, "规模: n=%lld  dim=%d  nq=%lld\n\n", (long long)base.n, base.dim,
           (long long)queries.n);

    append(log, "[性能]\n");
    append(log, "查询总时间 (ms): %.4f\n", perf.search_ms);
    append(log, "QPS: %.1f\n", perf.qps);
    append(log, "P50 延迟 (ms): %.6f\n", percentile(gpu.latency_ms, 0.50));
    append(log, "P99 延迟 (ms): %.6f\n", percentile(gpu.latency_ms, 0.99));
    append(log, "显存占用 (MB): %.1f\n", perf.gpu_mem_mb);
    append(log, "扫描比例: %.4f%%\n", perf.scanned_ratio * 100.0);
    if (!cfg.no_cpu)
        append(log, "CPU 参考总时间 (ms): %.2f\nGPU 加速比: %.2fx\n", cpu_ms, perf.speedup);
    else
        append(log, "CPU 参考: 已跳过 (--no-cpu)\n");

    append(log, "\n[正确性验证] (GPU 精确 vs CPU 参考)\n");
    if (!cfg.no_cpu) {
        append(log, "id 集合不一致的 query 数: %lld / %lld\n", (long long)cmp.set_mismatch,
               (long long)gpu.nq);
        append(log, "逐名次 id 不一致: %lld / %lld\n", (long long)cmp.rank_mismatch,
               (long long)cmp.total_pairs);
        append(log, "最大相对距离差: %.3e\n", cmp.max_rel_dist_diff);
        append(log, "说明: 判定标准为“每个 query 的 top-K id 集合完全一致”。\n"
                    "      由于 GPU 与 CPU 的浮点累加顺序/FMA 收缩可能不同，\n"
                    "      分数几乎相等时名次顺序可能有个别差异。\n");
    } else {
        append(log, "已跳过\n");
    }
    append(log, "\n[输出]\n结果文件: %s.result\n", prefix.c_str());

    std::printf("\n%s", log.c_str());
    FILE* fp = std::fopen((prefix + ".log").c_str(), "w");
    if (fp) {
        std::fputs(log.c_str(), fp);
        std::fclose(fp);
    }

    bool ok = cfg.no_cpu || (cmp.set_mismatch == 0);
    std::printf("[exact] %s\n",
                ok ? "通过：GPU 精确检索的 top-K 集合与 CPU 参考一致" : "警告：集合不一致");
    return ok ? 0 : 4;
}

// ======================= build =======================
static int cmd_build(int argc, char** argv) {
    if (argc < 5) {
        usage();
        return 1;
    }
    std::string base_path = argv[2], cfg_path = argv[3], index_path = argv[4];
    Config cfg = load_config(cfg_path);
    VectorSet base = load_vectors(base_path);
    if (base.data.empty()) {
        std::fprintf(stderr, "[build] 读取向量库失败\n");
        return 2;
    }

    IvfIndex idx;
    PerfStats perf;
    std::string err;
    std::printf("[build] 开始建索引: n=%lld dim=%d nlist=%d iters=%d\n", (long long)base.n,
                base.dim, cfg.nlist, cfg.kmeans_iters);
    if (!gpu_build_ivf(base, cfg, idx, perf, err)) {
        std::fprintf(stderr, "[build] 失败: %s\n", err.c_str());
        return 3;
    }
    save_index(index_path, idx);
    std::printf("[build] 完成: 建索引 %.2f ms, 聚类中心 %d, 倒排表 %zu 条 -> %s\n", perf.build_ms,
                idx.nlist, idx.list_ids.size(), index_path.c_str());

    // ---- 倒排表合法性校验：list_ids 必须是 0..n-1 的一个排列 ----
    {
        const int64_t n = (int64_t)idx.list_ids.size();
        bool struct_ok = (idx.list_start.size() == (size_t)(idx.nlist + 1)) &&
                         (idx.list_start.front() == 0) && (idx.list_start.back() == n);
        for (int i = 0; i < idx.nlist && struct_ok; i++)
            if (idx.list_start[i + 1] < idx.list_start[i])
                struct_ok = false;
        std::vector<uint8_t> seen((size_t)std::max<int64_t>(n, 1), 0);
        int64_t dup = 0, oob = 0;
        for (int32_t id : idx.list_ids) {
            if (id < 0 || id >= n) {
                oob++;
                continue;
            }
            if (seen[(size_t)id])
                dup++;
            seen[(size_t)id] = 1;
        }
        int64_t missing = 0;
        for (int64_t i = 0; i < n; i++)
            if (!seen[(size_t)i])
                missing++;
        std::printf("[build] 倒排表校验: 结构%s, 越界=%lld, 重复=%lld, 缺失=%lld\n",
                    struct_ok ? "正常" : "异常", (long long)oob, (long long)dup,
                    (long long)missing);
        if (!struct_ok || oob || dup || missing) {
            std::fprintf(stderr, "[build] 倒排表不合法！\n");
            return 5;
        }
    }
    return 0;
}

// ======================= search =======================
struct SearchOutcome {
    SearchResult res;
    PerfStats perf;
    double recall = -1.0;
    double dist_err = -1.0;
};

static bool do_search(const VectorSet& base, const VectorSet& queries, const IvfIndex& idx,
                      const Config& cfg, SearchOutcome& oc, std::string& err) {
    if (!gpu_ivf_search(base, queries, idx, cfg, oc.res, oc.perf, err))
        return false;
    return true;
}

static int cmd_search(int argc, char** argv) {
    if (argc < 7) {
        usage();
        return 1;
    }
    std::string base_path = argv[2], q_path = argv[3], index_path = argv[4], cfg_path = argv[5];
    std::string prefix = argv[6];
    std::string gt_path;
    for (int i = 7; i < argc; i++)
        if (std::string(argv[i]) == "--gt" && i + 1 < argc)
            gt_path = argv[++i];

    Config cfg = load_config(cfg_path);
    VectorSet base = load_vectors(base_path);
    VectorSet queries = load_vectors(q_path);
    IvfIndex idx;
    if (base.data.empty() || queries.data.empty() || !load_index(index_path, idx)) {
        std::fprintf(stderr, "[search] 输入或索引读取失败\n");
        return 2;
    }

    SearchOutcome oc;
    std::string err;
    if (!do_search(base, queries, idx, cfg, oc, err)) {
        std::fprintf(stderr, "[search] 失败: %s\n", err.c_str());
        return 3;
    }
    if (!gt_path.empty()) {
        SearchResult gt = load_result(gt_path, queries.n, cfg.top_k);
        oc.recall = recall_at_k(oc.res, gt);
        oc.dist_err = mean_dist_error(oc.res, gt);
    }
    save_result(prefix + ".result", oc.res);

    std::string log;
    append(log, "=== GPU 向量检索引擎 · IVF 查询日志 ===\n");
    append(log, "时间: %s\n", now_string().c_str());
    append(log, "向量库: %s\n查询集: %s\n索引: %s\n", base_path.c_str(), q_path.c_str(),
           index_path.c_str());
    append(log, "度量: %s   搜索模式: %s\n", metric_name(base.metric),
           idx.pq_m > 0 ? "ivf_pq" : "ivf_flat");
    append(log, "top_k: %d   nlist: %d   nprobe: %d   batch_size: %d\n", cfg.top_k, idx.nlist,
           cfg.nprobe, cfg.batch_size);
    append(log, "规模: n=%lld  dim=%d  nq=%lld\n\n", (long long)base.n, base.dim,
           (long long)queries.n);

    append(log, "[性能]\n");
    append(log, "查询总时间 (ms): %.4f\n", oc.perf.search_ms);
    append(log, "QPS: %.1f\n", oc.perf.qps);
    append(log, "P50 延迟 (ms): %.6f\n", percentile(oc.res.latency_ms, 0.50));
    append(log, "P99 延迟 (ms): %.6f\n", percentile(oc.res.latency_ms, 0.99));
    append(log, "显存占用 (MB): %.1f\n", oc.perf.gpu_mem_mb);
    append(log, "实际扫描向量比例: %.4f%%\n", oc.perf.scanned_ratio * 100.0);

    append(log, "\n[质量]\n");
    if (oc.recall >= 0)
        append(log, "recall@%d: %.6f\n平均距离误差: %.6e\n", cfg.top_k, oc.recall, oc.dist_err);
    else
        append(log, "未提供 ground truth (--gt)，跳过召回率\n");
    append(log, "\n[输出]\n结果文件: %s.result\n", prefix.c_str());

    std::printf("\n%s", log.c_str());
    FILE* fp = std::fopen((prefix + ".log").c_str(), "w");
    if (fp) {
        std::fputs(log.c_str(), fp);
        std::fclose(fp);
    }
    return 0;
}

// ======================= bench =======================
static int cmd_bench(int argc, char** argv) {
    if (argc < 7) {
        usage();
        return 1;
    }
    std::string base_path = argv[2], q_path = argv[3], index_path = argv[4], cfg_path = argv[5];
    std::string prefix = argv[6];
    std::string gt_path;
    for (int i = 7; i < argc; i++)
        if (std::string(argv[i]) == "--gt" && i + 1 < argc)
            gt_path = argv[++i];

    Config cfg = load_config(cfg_path);
    VectorSet base = load_vectors(base_path);
    VectorSet queries = load_vectors(q_path);
    IvfIndex idx;
    if (base.data.empty() || queries.data.empty() || !load_index(index_path, idx)) {
        std::fprintf(stderr, "[bench] 输入或索引读取失败\n");
        return 2;
    }
    SearchResult gt;
    if (!gt_path.empty())
        gt = load_result(gt_path, queries.n, cfg.top_k);

    std::string log;
    append(log, "=== IVF 参数扫描 (nprobe / batch_size) ===\n");
    append(log, "时间: %s\n", now_string().c_str());
    append(log, "度量: %s   top_k: %d   nlist: %d   n: %lld   nq: %lld\n\n",
           metric_name(base.metric), cfg.top_k, idx.nlist, (long long)base.n, (long long)queries.n);

    append(log, "[nprobe 扫描]  (batch_size=%d)\n", cfg.batch_size);
    append(log, "%-8s %-14s %-12s %-12s %-12s %-14s\n", "nprobe", "recall@K", "QPS", "P50(ms)",
           "P99(ms)", "扫描比例(%)");
    printf("%s", log.c_str());

    std::vector<int> probes = {1, 2, 4, 8, 16, 32, 64, 128};
    for (int np : probes) {
        if (np > idx.nlist)
            break;
        Config c = cfg;
        c.nprobe = np;
        SearchOutcome oc;
        std::string err;
        if (!do_search(base, queries, idx, c, oc, err)) {
            std::fprintf(stderr, "[bench] nprobe=%d 失败: %s\n", np, err.c_str());
            continue;
        }
        double rec = gt.ids.empty() ? -1.0 : recall_at_k(oc.res, gt);
        char line[256];
        std::snprintf(line, sizeof(line), "%-8d %-14.6f %-12.1f %-12.6f %-12.6f %-14.4f\n", np, rec,
                      oc.perf.qps, percentile(oc.res.latency_ms, 0.50),
                      percentile(oc.res.latency_ms, 0.99), oc.perf.scanned_ratio * 100.0);
        log += line;
        std::printf("%s", line);
    }

    append(log, "\n[batch_size 扫描]  (nprobe=%d)\n", cfg.nprobe);
    append(log, "%-12s %-14s %-12s %-12s\n", "batch_size", "recall@K", "QPS", "P50(ms)");
    std::printf("\n%-12s %-14s %-12s %-12s\n", "batch_size", "recall@K", "QPS", "P50(ms)");
    for (int bs : {1, 16, 64, 128, 256, 512}) {
        Config c = cfg;
        c.batch_size = bs;
        SearchOutcome oc;
        std::string err;
        if (!do_search(base, queries, idx, c, oc, err))
            continue;
        double rec = gt.ids.empty() ? -1.0 : recall_at_k(oc.res, gt);
        char line[256];
        std::snprintf(line, sizeof(line), "%-12d %-14.6f %-12.1f %-12.6f\n", bs, rec, oc.perf.qps,
                      percentile(oc.res.latency_ms, 0.50));
        log += line;
        std::printf("%s", line);
    }

    FILE* fp = std::fopen((prefix + ".bench.log").c_str(), "w");
    if (fp) {
        std::fputs(log.c_str(), fp);
        std::fclose(fp);
    }
    std::printf("\n[bench] 日志已写入 %s.bench.log\n", prefix.c_str());
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
    std::printf("== 向量检索引擎自检 ==\n");

    // 小规模数据：GPU 精确检索应与 CPU 参考一致
    VectorSet base = gen_vectors(3000, 64, "normal", 7, 0);
    base.metric = Metric::L2;
    VectorSet queries = gen_queries(20, base, "normal", 11);
    queries.metric = Metric::L2;

    Config cfg;
    cfg.top_k = 10;
    cfg.batch_size = 8;
    SearchResult cpu = cpu_exact_search(base, queries, cfg.top_k, Metric::L2);
    SearchResult gpu;
    PerfStats perf;
    std::string err;
    bool ok = gpu_exact_search(base, queries, cfg, gpu, perf, nullptr, err);
    check(ok, "精确检索 kernel 执行");
    if (ok) {
        CompareResult cmp = compare_results(gpu, cpu);
        check(cmp.set_mismatch == 0, "精确检索 top-K 集合与 CPU 参考一致");
        if (cmp.rank_mismatch)
            std::printf("        (提示: 名次顺序差异 %lld 项，源于浮点累加差异)\n",
                        (long long)cmp.rank_mismatch);
    }

    // IVF：大 nprobe 下召回率应接近 1
    cfg.nlist = 16;
    cfg.nprobe = 16;
    cfg.kmeans_iters = 5;
    IvfIndex idx;
    if (gpu_build_ivf(base, cfg, idx, perf, err)) {
        SearchResult ivf;
        if (gpu_ivf_search(base, queries, idx, cfg, ivf, perf, err)) {
            double rec = recall_at_k(ivf, cpu);
            std::printf("  IVF recall@%d (nprobe=nlist) = %.4f\n", cfg.top_k, rec);
            check(rec > 0.99, "nprobe=nlist 时 IVF 召回率 ≈ 1");
        } else {
            check(false, std::string("IVF 查询失败: " + err).c_str());
        }
    } else {
        check(false, std::string("IVF 建索引失败: " + err).c_str());
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
    if (mode == "genq")
        return cmd_genq(argc, argv);
    if (mode == "exact")
        return cmd_exact(argc, argv);
    if (mode == "build")
        return cmd_build(argc, argv);
    if (mode == "search")
        return cmd_search(argc, argv);
    if (mode == "bench")
        return cmd_bench(argc, argv);
    if (mode == "selftest")
        return cmd_selftest();
    usage();
    return 1;
}
