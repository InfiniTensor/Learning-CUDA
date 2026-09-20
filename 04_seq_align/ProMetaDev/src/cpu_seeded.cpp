// cpu_seeded.cpp - CPU 上运行与 GPU 完全相同的 seed-and-extend 流水线
//
// 目的：给出"同一算法、同一参数"下的 CPU 性能基准，使 GPU 加速比可比。
// 流程与 GPU 端一致：
//   ① 主机端构建同样的 k-mer 散列索引（桶内按 code 过滤散列碰撞）
//   ② 每条 read 按 step 取种子，查索引得到候选锚点
//   ③ 候选排序去重
//   ④ 对每个候选做窗口内自由起始的带状 DP
//   ⑤ 每条 read 归约取最优，并按阈值判定 unknown_origin
// 用 OpenMP 在 read 维度并行（默认使用全部可用核心）。
#include "align.h"
#include "dna.h"

#include <algorithm>
#include <chrono>
#include <climits>
#include <cstdint>
#include <vector>

namespace sa {

namespace {

constexpr int32_t NEG = INT32_MIN / 4;
constexpr int SCAN_CAP = 512; // 每个种子最多扫描的桶内条目数（与 GPU 端一致）

struct HostIndex {
    int64_t ref_len = 0;
    uint32_t mask = 0;
    std::vector<unsigned long long> bucket_off; // nb + 1
    std::vector<uint32_t> entry_code;
    std::vector<int64_t> entry_pos;
};

void build_host_index(HostIndex& ix, const RefGenome& ref, const Config& cfg) {
    const int64_t n = ref.total_bases();
    const int k = cfg.k;
    ix.ref_len = n;

    int bits = cfg.index_bits;
    if (bits <= 0) {
        bits = 16;
        while (bits < 28 && (1LL << bits) < (n / 16))
            ++bits;
    }
    const uint32_t nb = 1u << bits;
    ix.mask = nb - 1;

    std::vector<uint8_t> codes((size_t)n);
    for (int64_t i = 0; i < n; ++i) {
        const int c = base_code(ref.bases[(size_t)i]);
        codes[(size_t)i] = (uint8_t)(c > 3 ? 4 : c);
    }

    std::vector<unsigned long long> cnt((size_t)nb, 0);
    auto encode = [&](int64_t p, uint32_t* out) -> bool {
        uint32_t c = 0;
        for (int j = 0; j < k; ++j) {
            const uint8_t v = codes[(size_t)(p + j)];
            if (v >= 4)
                return false;
            c = (c << 2) | v;
        }
        *out = c;
        return true;
    };

    for (int64_t p = 0; p + k <= n; ++p) {
        uint32_t c;
        if (encode(p, &c))
            ++cnt[kmer_hash(c, ix.mask)];
    }
    ix.bucket_off.assign((size_t)nb + 1, 0);
    for (uint32_t b = 0; b < nb; ++b)
        ix.bucket_off[b + 1] = ix.bucket_off[b] + cnt[b];
    const int64_t total = (int64_t)ix.bucket_off[nb];
    ix.entry_code.resize((size_t)total);
    ix.entry_pos.resize((size_t)total);

    std::vector<unsigned long long> cur(ix.bucket_off.begin(), ix.bucket_off.end() - 1);
    for (int64_t p = 0; p + k <= n; ++p) {
        uint32_t c;
        if (!encode(p, &c))
            continue;
        const uint32_t b = kmer_hash(c, ix.mask);
        const unsigned long long slot = cur[b]++;
        ix.entry_code[(size_t)slot] = c;
        ix.entry_pos[(size_t)slot] = p;
    }
}

} // namespace

std::vector<Hit> cpu_seeded_align_all(const RefGenome& ref, const std::vector<Read>& reads,
                                      const Config& cfg, double* out_index_ms) {
    const auto t0 = std::chrono::high_resolution_clock::now();
    HostIndex ix;
    build_host_index(ix, ref, cfg);
    const auto t1 = std::chrono::high_resolution_clock::now();
    if (out_index_ms) {
        *out_index_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    }

    const int R = (int)reads.size();
    std::vector<Hit> hits((size_t)R);
    const int k = cfg.k;
    const int step = std::max(1, cfg.seed_step);
    const int B = cfg.band;

#pragma omp parallel for schedule(dynamic, 8)
    for (int ri = 0; ri < R; ++ri) {
        const std::string& seq = reads[(size_t)ri].bases;
        const int L = (int)seq.size();
        const int64_t ref_len = ix.ref_len;

        // ① 种子 → 候选
        std::vector<int64_t> cand;
        cand.reserve(64);
        for (int off = 0; off + k <= L; off += step) {
            uint32_t c = 0;
            bool ok = true;
            for (int j = 0; j < k; ++j) {
                const int v = base_code(seq[(size_t)(off + j)]);
                if (v > 3) {
                    ok = false;
                    break;
                }
                c = (c << 2) | (uint32_t)v;
            }
            if (!ok)
                continue;
            const uint32_t b = kmer_hash(c, ix.mask);
            const int64_t e0 = (int64_t)ix.bucket_off[b];
            const int64_t e1 = (int64_t)ix.bucket_off[b + 1];
            const int64_t lim = std::min<int64_t>(SCAN_CAP, e1 - e0);
            for (int64_t e = e0; e < e0 + lim; ++e) {
                if (ix.entry_code[(size_t)e] != c)
                    continue;
                const int64_t q = ix.entry_pos[(size_t)e] - off;
                if (q < B || q + L + B >= ref_len)
                    continue;
                cand.push_back(q);
            }
        }

        // ② 排序 + 去重
        std::sort(cand.begin(), cand.end());
        cand.erase(std::unique(cand.begin(), cand.end()), cand.end());

        // ③ 验证 + 归约
        int32_t best = INT32_MIN;
        int64_t best_q = -1;
        for (int64_t q : cand) {
            int64_t st = -1;
            const int32_t s = window_best_score(ref, seq.data(), L, q, cfg, &st);
            if (s == INT32_MIN)
                continue;
            if (s > best || (s == best && (best_q < 0 || st < best_q))) {
                best = s;
                best_q = st;
            }
        }

        const int32_t threshold = (int32_t)(cfg.min_score_ratio * cfg.match * L);
        if (best < threshold)
            continue; // unknown_origin
        int32_t sid = -1;
        int64_t off_in_seq = -1;
        global_to_local(ref, best_q, sid, off_in_seq);
        if (sid < 0)
            continue;
        hits[(size_t)ri].seq_id = sid;
        hits[(size_t)ri].pos = off_in_seq;
        hits[(size_t)ri].score = best;
    }
    return hits;
}

} // namespace sa
