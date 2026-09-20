// main.cpp - 序列比对 CLI
#include "align.h"
#include "io.h"
#include "types.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace sa;

namespace {

double now_ms() {
    using clk = std::chrono::high_resolution_clock;
    return std::chrono::duration<double, std::milli>(clk::now().time_since_epoch()).count();
}

void usage(const char* prog) {
    std::cerr << "用法:\n"
              << "  " << prog << " cpu <ref.fa> <reads.fq> <out.txt> [options]\n"
              << "  " << prog << " cpuseed <ref.fa> <reads.fq> <out.txt> [options]\n"
              << "  " << prog << " gpu <ref.fa> <reads.fq> <out.txt> [options]\n"
              << "  " << prog << " selftest\n"
              << "模式:\n"
              << "  cpu       CPU 穷举（每个锚点都验证，仅适合小参考，作为正确性基准）\n"
              << "  cpuseed   CPU 版 seed-and-extend（与 GPU 同算法，作为性能基准）\n"
              << "  gpu       GPU seed-and-extend\n"
              << "选项:\n"
              << "  --k <int>             种子长度 (默认 15)\n"
              << "  --band <int>          带状 DP 半宽 (默认 16)\n"
              << "  --min-ratio <float>   判定阈值比例 (默认 0.4)\n"
              << "  --max-cand <int>      每条 read 最大候选数 (默认 64)\n"
              << "  --seed-step <int>     read 取种子步长 (默认 4)\n"
              << "  --index-bits <int>    索引哈希桶位数 (0=自动, 默认 0)\n"
              << "  --truth <file>        真值文件，用于统计准确率\n"
              << "  --perf <file>         性能日志输出路径\n";
}

struct Options {
    Config cfg;
    std::string truth_path;
    std::string perf_path;
};

Options parse_options(int argc, char** argv, int start) {
    Options opt;
    for (int i = start; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) {
                std::cerr << "缺少参数: " << a << "\n";
                std::exit(2);
            }
            return argv[++i];
        };
        if (a == "--k")
            opt.cfg.k = std::stoi(next());
        else if (a == "--band")
            opt.cfg.band = std::stoi(next());
        else if (a == "--min-ratio")
            opt.cfg.min_score_ratio = std::stod(next());
        else if (a == "--max-cand")
            opt.cfg.max_candidates = std::stoi(next());
        else if (a == "--seed-step")
            opt.cfg.seed_step = std::stoi(next());
        else if (a == "--index-bits")
            opt.cfg.index_bits = std::stoi(next());
        else if (a == "--truth")
            opt.truth_path = next();
        else if (a == "--perf")
            opt.perf_path = next();
        else {
            std::cerr << "未知选项: " << a << "\n";
            std::exit(2);
        }
    }
    return opt;
}

// 统计比对结果与真值的吻合度
void report_accuracy(const std::vector<Hit>& hits, const std::vector<Hit>& truth) {
    if (truth.empty())
        return;
    if (truth.size() != hits.size()) {
        std::printf("[check] 真值条数 (%zu) 与结果条数 (%zu) 不一致，跳过校验\n", truth.size(),
                    hits.size());
        return;
    }
    int64_t src_total = 0, src_aligned = 0, pos_exact = 0, score_exact = 0, score_cmp = 0;
    int64_t rnd_total = 0, rnd_unknown = 0;
    for (size_t i = 0; i < hits.size(); ++i) {
        const Hit& t = truth[i];
        const Hit& h = hits[i];
        if (t.known()) {
            ++src_total;
            if (h.known()) {
                ++src_aligned;
                if (h.seq_id == t.seq_id && h.pos == t.pos)
                    ++pos_exact;
                if (t.score >= 0) {
                    ++score_cmp;
                    if (h.score == t.score)
                        ++score_exact;
                }
            }
        } else {
            ++rnd_total;
            if (!h.known())
                ++rnd_unknown;
        }
    }
    auto pct = [](int64_t a, int64_t b) { return b > 0 ? 100.0 * (double)a / (double)b : 0.0; };
    std::printf("[check] 有来源 read: %lld/%lld 被比对 (%.2f%%)  "
                "起点完全一致 %lld (%.2f%%)  得分完全一致 %lld/%lld (%.2f%%)\n",
                (long long)src_aligned, (long long)src_total, pct(src_aligned, src_total),
                (long long)pos_exact, pct(pos_exact, src_total), (long long)score_exact,
                (long long)score_cmp, pct(score_exact, score_cmp));
    std::printf("[check] 随机 read: %lld/%lld 正确判为 unknown_origin (%.2f%%)\n",
                (long long)rnd_unknown, (long long)rnd_total, pct(rnd_unknown, rnd_total));
}

// 简易确定性伪随机 DNA 生成（仅用于自检）
std::string rand_dna(uint64_t& s, int n) {
    static const char B[4] = {'A', 'C', 'G', 'T'};
    std::string r;
    r.reserve((size_t)n);
    for (int i = 0; i < n; ++i) {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        r.push_back(B[(s >> 33) & 3u]);
    }
    return r;
}

int cmd_selftest() {
    Config cfg;
    cfg.k = 11;
    cfg.band = 8;
    cfg.seed_step = 4;
    cfg.min_score_ratio = 0.7; // 自检关注"最优解是否一致"，同时启用判定阈值

    // 两条 200bp 的随机参考序列
    uint64_t seed = 12345;
    const std::string s1 = rand_dna(seed, 200);
    const std::string s2 = rand_dna(seed, 200);
    RefGenome ref = make_reference({"chr1", "chr2"}, {s1, s2}, cfg.k);

    // 构造 reads：精确片段 / 带替换突变片段 / 纯随机片段
    std::vector<Read> reads;
    auto add = [&](const std::string& name, const std::string& seq) {
        Read r;
        r.name = name;
        r.bases = seq;
        reads.push_back(r);
    };
    add("exact1", s1.substr(50, 40));
    add("exact2", s2.substr(120, 40));
    {
        std::string m = s1.substr(10, 40);
        m[3] = (m[3] == 'A') ? 'C' : 'A';
        m[25] = (m[25] == 'G') ? 'T' : 'G';
        add("mut2", m);
    }
    add("random1", rand_dna(seed, 40));
    add("random2", rand_dna(seed, 40));

    // CPU 穷举
    std::vector<Hit> cpu_hits = cpu_align_all(ref, reads, cfg);

    // GPU
    GpuAligner gpu;
    gpu.build_index(ref, cfg);
    std::vector<Hit> gpu_hits = gpu.align(reads, cfg);

    int fail = 0;
    for (size_t i = 0; i < reads.size(); ++i) {
        const Hit& a = cpu_hits[i];
        const Hit& b = gpu_hits[i];
        const bool same = (a.seq_id == b.seq_id) && (a.pos == b.pos) && (a.score == b.score);
        std::printf(
            "[selftest] %-8s CPU(seq=%d pos=%lld score=%d)  GPU(seq=%d pos=%lld score=%d)  %s\n",
            reads[i].name.c_str(), a.seq_id, (long long)a.pos, a.score, b.seq_id, (long long)b.pos,
            b.score, same ? "PASS" : "FAIL");
        if (!same)
            ++fail;
    }

    // 精确片段应当得到满分 2L、起点正确
    const int L = 40;
    if (!(cpu_hits[0].seq_id == 0 && cpu_hits[0].pos == 50 && cpu_hits[0].score == 2 * L)) {
        std::printf("[selftest] exact1 期望 seq=0 pos=50 score=%d\n", 2 * L);
        ++fail;
    }
    if (!(cpu_hits[1].seq_id == 1 && cpu_hits[1].pos == 120 && cpu_hits[1].score == 2 * L)) {
        std::printf("[selftest] exact2 期望 seq=1 pos=120 score=%d\n", 2 * L);
        ++fail;
    }

    std::printf("[selftest] %s\n", fail == 0 ? "PASS" : "FAIL");
    return fail == 0 ? 0 : 1;
}

} // namespace

// GPU 预热：用一段极小的合成分片跑完整流水线。
// 本程序以 sm_90 为目标编译，在更新架构（如 sm_120）上由驱动 JIT 执行 PTX；
// JIT 发生在每个内核的首次启动，且 WSL2 下不跨进程复用编译缓存。
// 若不预热，这部分一次性开销会被计入 T_index / T_align，使性能数据失真。
void warmup_gpu(const Config& cfg) {
    RefGenome wr =
        make_reference({"warmup"}, {"ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT"}, cfg.k);
    std::vector<Read> wreads(2);
    wreads[0].name = "w0";
    wreads[0].bases = "ACGTACGTACGTACGTACGTA";
    wreads[1].name = "w1";
    wreads[1].bases = "TTTTGGGGCCCCAAAATTTTGGGGCCCC";
    GpuAligner w;
    w.build_index(wr, cfg);
    w.align(wreads, cfg);
    w.release();
}

int main(int argc, char** argv) {
    if (argc < 2) {
        usage(argv[0]);
        return 1;
    }
    const std::string cmd = argv[1];

    if (cmd == "selftest")
        return cmd_selftest();

    if (cmd == "cpu" || cmd == "cpuseed" || cmd == "gpu") {
        if (argc < 5) {
            usage(argv[0]);
            return 1;
        }
        const std::string ref_path = argv[2];
        const std::string read_path = argv[3];
        const std::string out_path = argv[4];
        const Options opt = parse_options(argc, argv, 5);
        const Config& cfg = opt.cfg;

        if (cmd == "gpu")
            warmup_gpu(cfg); // 触发 PTX JIT，排除一次性开销

        const double t0 = now_ms();
        RefGenome ref = load_reference(ref_path, cfg.k);
        std::vector<Read> reads = load_reads(read_path);
        const double t1 = now_ms();

        std::vector<Hit> hits;
        AlignStats st;
        double cpu_index_ms = 0.0;

        if (cmd == "cpu") {
            hits = cpu_align_all(ref, reads, cfg);
        } else if (cmd == "cpuseed") {
            hits = cpu_seeded_align_all(ref, reads, cfg, &cpu_index_ms);
        } else {
            GpuAligner aligner;
            aligner.build_index(ref, cfg);
            hits = aligner.align(reads, cfg);
            st = aligner.stats();
        }
        const double t2 = now_ms();

        write_hits(out_path, ref, reads, hits);

        std::printf("[%s] 参考: %d 条序列 / %lld bp（拼接后含填充/分隔符 %lld bp）\n", cmd.c_str(),
                    ref.num_seqs(), (long long)ref.real_bases(), (long long)ref.total_bases());
        std::printf("[%s] reads: %zu 条 × %d bp\n", cmd.c_str(), reads.size(),
                    reads.empty() ? 0 : (int)reads[0].bases.size());
        std::printf("[%s] T_load  = %.2f ms\n", cmd.c_str(), t1 - t0);
        if (cmd == "gpu") {
            std::printf("[gpu] 索引条目 = %lld  种子 = %lld  候选对 = %lld\n",
                        (long long)st.index_entries, (long long)st.num_seeds,
                        (long long)st.num_candidates);
            std::printf("[gpu] T_index  = %.2f ms\n", st.build_index_ms);
            std::printf("[gpu] T_upload = %.2f ms\n", st.upload_ms);
            std::printf("[gpu] T_seed   = %.2f ms\n", st.seed_ms);
            std::printf("[gpu] T_verify = %.2f ms\n", st.verify_ms);
            std::printf("[gpu] T_reduce = %.2f ms\n", st.reduce_ms);
            std::printf("[gpu] T_align  = %.2f ms\n", st.total_ms);
        } else if (cmd == "cpuseed") {
            std::printf("[cpuseed] T_index = %.2f ms\n", cpu_index_ms);
            std::printf("[cpuseed] T_align = %.2f ms\n", t2 - t1 - cpu_index_ms);
        } else {
            std::printf("[cpu] T_align = %.2f ms\n", t2 - t1);
        }
        std::printf("[%s] T_total  = %.2f ms\n", cmd.c_str(), t2 - t0);
        std::printf("[%s] 结果已写入: %s\n", cmd.c_str(), out_path.c_str());

        if (!opt.truth_path.empty()) {
            std::vector<Hit> truth = load_truth(opt.truth_path);
            report_accuracy(hits, truth);
        }

        if (!opt.perf_path.empty()) {
            std::ofstream pf(opt.perf_path);
            if (!pf)
                throw std::runtime_error("无法写入性能日志: " + opt.perf_path);
            pf << "mode = " << cmd << "\n";
            pf << "ref_seqs = " << ref.num_seqs() << "\n";
            pf << "ref_bases = " << ref.real_bases() << "\n";
            pf << "num_reads = " << reads.size() << "\n";
            pf << "k = " << cfg.k << "\n";
            pf << "band = " << cfg.band << "\n";
            pf << "seed_step = " << cfg.seed_step << "\n";
            pf << "T_load_ms = " << (t1 - t0) << "\n";
            if (cmd == "gpu") {
                pf << "index_entries = " << st.index_entries << "\n";
                pf << "num_seeds = " << st.num_seeds << "\n";
                pf << "num_candidates = " << st.num_candidates << "\n";
                pf << "T_index_ms = " << st.build_index_ms << "\n";
                pf << "T_upload_ms = " << st.upload_ms << "\n";
                pf << "T_seed_ms = " << st.seed_ms << "\n";
                pf << "T_verify_ms = " << st.verify_ms << "\n";
                pf << "T_reduce_ms = " << st.reduce_ms << "\n";
                pf << "T_align_ms = " << st.total_ms << "\n";
            } else if (cmd == "cpuseed") {
                pf << "T_index_ms = " << cpu_index_ms << "\n";
                pf << "T_align_ms = " << (t2 - t1 - cpu_index_ms) << "\n";
            } else {
                pf << "T_align_ms = " << (t2 - t1) << "\n";
            }
            pf << "T_total_ms = " << (t2 - t0) << "\n";
            std::printf("[%s] 性能日志已写入: %s\n", cmd.c_str(), opt.perf_path.c_str());
        }
        return 0;
    }

    usage(argv[0]);
    return 1;
}
