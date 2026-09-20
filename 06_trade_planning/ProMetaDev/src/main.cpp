// main.cpp - GPU 最大流 CLI
#include "io.h"
#include "maxflow.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

using namespace mf;

static void usage(const char* prog) {
    std::cerr << "用法:\n"
              << "  " << prog
              << " run <graph.csr> <queries.txt> <result_out> [perf_log] [--cpu|--cpu-ref]\n"
              << "  " << prog << " selftest\n"
              << "模式:\n"
              << "  (默认)     GPU 求解\n"
              << "  --cpu      仅 CPU Edmonds-Karp\n"
              << "  --cpu-ref  GPU 与 CPU Edmonds-Karp 逐查询比对，不一致则退出非零\n";
}

// 高精度计时（毫秒）
static double now_ms() {
    using clock = std::chrono::high_resolution_clock;
    return std::chrono::duration<double, std::milli>(clock::now().time_since_epoch()).count();
}

int main(int argc, char** argv) {
    if (argc < 2) {
        usage(argv[0]);
        return 1;
    }
    std::string cmd = argv[1];

    if (cmd == "selftest") {
        // 内置小图正确性测试
        // 图: 0->1(3), 0->2(2), 1->2(1), 1->3(2), 2->3(3)
        // 最大流 0->3 = 5
        GraphCSR g;
        g.num_nodes = 4;
        g.num_edges = 5;
        g.row_ptr = {0, 2, 4, 5, 5};
        g.col_idx = {1, 2, 2, 3, 3};
        g.cap = {3, 2, 1, 2, 3};

        ResidualGraph rg = build_residual_graph(g);

        int64_t cpu = cpu_edmonds_karp_maxflow(rg, 0, 3);
        std::cout << "[selftest] CPU Edmonds-Karp maxflow(0->3) = " << cpu << " (期望 5)\n";

        GpuMaxFlow gpu;
        gpu.upload(rg);
        int64_t gpu_flow = gpu.solve(0, 3);
        std::cout << "[selftest] GPU Push-Relabel maxflow(0->3) = " << gpu_flow << " (期望 5)\n";

        bool ok = (cpu == 5 && gpu_flow == 5);
        std::cout << "[selftest] " << (ok ? "PASS" : "FAIL") << "\n";
        return ok ? 0 : 1;
    }

    if (cmd == "run") {
        if (argc < 5) {
            usage(argv[0]);
            return 1;
        }
        std::string graph_path = argv[2];
        std::string query_path = argv[3];
        std::string result_path = argv[4];
        std::string perf_path;
        bool use_cpu = false;
        bool cpu_ref = false;
        for (int i = 5; i < argc; ++i) {
            std::string a = argv[i];
            if (a == "--cpu")
                use_cpu = true;
            else if (a == "--cpu-ref")
                cpu_ref = true;
            else
                perf_path = a;
        }

        // ===== T_preprocess: 读图 + 建残量图 + 上传 GPU =====
        double t0 = now_ms();
        GraphCSR g = load_graph(graph_path);
        ResidualGraph rg = build_residual_graph(g);
        double t1 = now_ms();

        std::vector<Query> queries = load_queries(query_path);
        std::vector<int64_t> results(queries.size());

        PerfStats stats;
        stats.num_queries = (int)queries.size();

        if (use_cpu) {
            // CPU 模式：每个查询用 Edmonds-Karp
            double t_start = now_ms();
            for (size_t i = 0; i < queries.size(); ++i) {
                results[i] = cpu_edmonds_karp_maxflow(rg, queries[i].source, queries[i].target);
                if (i == 0)
                    stats.ttfq_ms = now_ms() - t_start;
            }
            double t_end = now_ms();
            stats.total_ms = t_end - t_start;
        } else {
            // GPU 模式
            GpuMaxFlow gpu;
            gpu.upload(rg);
            double t_upload = now_ms();
            stats.preprocess_ms = (t1 - t0) + (t_upload - t1);

            double t_start = now_ms();
            for (size_t i = 0; i < queries.size(); ++i) {
                int phases = 0;
                results[i] = gpu.solve(queries[i].source, queries[i].target, &phases);
                if (i == 0)
                    stats.ttfq_ms = now_ms() - t_start;
                if (i == queries.size() - 1)
                    stats.num_phases = phases;
            }
            double t_end = now_ms();
            stats.total_ms = t_end - t_start;
        }

        if (queries.size() > 0) {
            stats.tpq_ms = stats.total_ms / queries.size();
        }

        // --cpu-ref: 用 CPU Edmonds-Karp 作为参考，逐查询比对
        int mismatch = 0;
        if (cpu_ref) {
            std::cout << "[cpu-ref] 逐查询比对 GPU vs CPU Edmonds-Karp ...\n";
            for (size_t i = 0; i < queries.size(); ++i) {
                int64_t cpu_f = cpu_edmonds_karp_maxflow(rg, queries[i].source, queries[i].target);
                if (cpu_f != results[i]) {
                    std::cout << "[cpu-ref] 不一致: (" << queries[i].source << " -> "
                              << queries[i].target << ") GPU=" << results[i] << " CPU=" << cpu_f
                              << "\n";
                    mismatch++;
                }
            }
            if (mismatch == 0) {
                std::cout << "[cpu-ref] 全部 " << queries.size() << " 个查询一致 ✓\n";
            } else {
                std::cout << "[cpu-ref] " << mismatch << "/" << queries.size()
                          << " 个查询不一致 ✗\n";
            }
        }

        write_results(result_path, queries, results);
        if (!perf_path.empty()) {
            write_perf_log(perf_path, stats);
        }

        // 控制台摘要
        std::cout << "[run] 图: N=" << g.num_nodes << " M=" << g.num_edges
                  << "  查询数=" << queries.size() << "\n";
        std::cout << "[run] T_preprocess = " << stats.preprocess_ms << " ms\n";
        std::cout << "[run] TTFQ         = " << stats.ttfq_ms << " ms\n";
        std::cout << "[run] T_total      = " << stats.total_ms << " ms\n";
        std::cout << "[run] TPQ          = " << stats.tpq_ms << " ms\n";
        if (!use_cpu) {
            std::cout << "[run] 最后查询阶段数 = " << stats.num_phases << "\n";
        }
        std::cout << "[run] 结果已写入: " << result_path << "\n";
        return (cpu_ref && mismatch > 0) ? 1 : 0;
    }

    usage(argv[0]);
    return 1;
}
