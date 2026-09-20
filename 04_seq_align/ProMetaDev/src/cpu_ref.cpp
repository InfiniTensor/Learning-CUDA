// cpu_ref.cpp - CPU 穷举参考实现（窗口内自由起始的带状 fitting 比对）
//
// 打分与 DP 定义（与 GPU 端 verify_kernel 完全一致）：
//   给定锚点 q，窗口为 ref[q-B, q+L+B)（长度 W = L + 2B + 1）。
//   整条 read 必须被完整比对到窗口内的某一段上，比对可以在窗口内任意位置起始。
//   用 e = j - i - B 作列下标（i 为已消耗的 read 碱基数，j 为已消耗的参考碱基数），
//   e ∈ [-B, B]，即 j - i ∈ [0, 2B]。
//
// 由于 mismatch(-1) 与 gap(-1) 同分，DP 会用空位"挑选"更匹配的参考位置，
// 因此必须同时记录比对起始点才能给出确定结果。这里把 (得分, 起始点排名)
// 打包进同一个 int32，既省一次 DP 数组，也让"同分取最左"的规则自然成立：
//   packed = (score + L) * WID + (2B - t),  t = 起始列的编号 ∈ [0, 2B]
// 取 packed 最大 ⇒ 得分最大；得分相同时 t 最小 ⇒ 起始位置最靠左。
#include "align.h"
#include "dna.h"
#include <algorithm>
#include <climits>
#include <vector>

namespace sa {

namespace {

constexpr int32_t NEG = INT32_MIN / 4; // 安全负无穷

struct FitScratch {
    std::vector<int32_t> prev, cur;
    std::vector<uint8_t> rcode; // read 的碱基编码（复用缓冲）
};

// 返回打包值；out_start 输出全局起始坐标
int32_t window_best_scratch(const RefGenome& ref, const char* read, int L, int64_t q,
                            const Config& cfg, FitScratch& sc, int64_t* out_start) {
    const int64_t n = ref.total_bases();
    const int B = cfg.band;
    const int WID = 2 * B + 1;
    const int64_t qb = q - B;                 // 窗口起点
    const int64_t W = (int64_t)L + 2 * B + 1; // 窗口长度
    if (qb < 0 || qb + W > n)
        return NEG;

    if ((int)sc.prev.size() != WID) {
        sc.prev.assign((size_t)WID, NEG);
        sc.cur.assign((size_t)WID, NEG);
    }

    const int32_t w_match = cfg.match * WID;
    const int32_t w_mismatch = cfg.mismatch * WID;
    const int32_t w_gap = cfg.gap * WID;
    const int32_t base0 = L * WID; // score = 0 时的打包基准

    // i = 0：j ∈ [0, 2B]（e ∈ [-B, B]）任意列都可作为起始
    for (int e = -B; e <= B; ++e) {
        const int t = B + e; // 起始列 j0 = t
        sc.prev[(size_t)(e + B)] = base0 + (2 * B - t);
    }

    // read 碱基编码（每个锚点复用，避免重复转换）
    if ((int)sc.rcode.size() != L)
        sc.rcode.resize((size_t)L);
    for (int i = 0; i < L; ++i) {
        const int c = base_code(read[i]);
        sc.rcode[(size_t)i] = (uint8_t)(c > 3 ? 4 : c);
    }
    const uint8_t* rcode = sc.rcode.data();
    const char* refbase = ref.bases.data();

    for (int i = 1; i <= L; ++i) {
        std::fill(sc.cur.begin(), sc.cur.end(), NEG);
        const int elo = -B;
        const int ehi = B;
        const int ra = rcode[(size_t)(i - 1)];

        for (int e = elo; e <= ehi; ++e) {
            const int64_t j = (int64_t)i + B + e;
            if (j > W)
                break;
            int32_t best = NEG;

            if (j >= 1) { // 对角
                const int32_t d = sc.prev[(size_t)(e + B)];
                if (d > NEG) {
                    const int rc = base_code(refbase[qb + j - 1]);
                    best = d + ((ra < 4 && ra == rc) ? w_match : w_mismatch);
                }
            }
            if (e + 1 <= B) { // 参考中插入空位
                const int32_t u = sc.prev[(size_t)(e + 1 + B)];
                if (u > NEG)
                    best = std::max(best, u + w_gap);
            }
            if (e - 1 >= -B) { // read 中插入空位
                const int32_t l = sc.cur[(size_t)(e - 1 + B)];
                if (l > NEG)
                    best = std::max(best, l + w_gap);
            }
            sc.cur[(size_t)(e + B)] = best;
        }
        sc.prev.swap(sc.cur);
    }

    int32_t best = NEG;
    for (int e = -B; e <= B; ++e)
        best = std::max(best, sc.prev[(size_t)(e + B)]);
    if (best <= NEG)
        return NEG;

    if (out_start) {
        const int t = 2 * B - (best % WID);
        *out_start = qb + t;
    }
    return best;
}

inline int32_t unpack_score(int32_t packed, int L, int B) {
    return packed / (2 * B + 1) - L;
}

} // namespace

int32_t window_best_score(const RefGenome& ref, const char* read, int L, int64_t q,
                          const Config& cfg, int64_t* out_start) {
    // thread_local：OpenMP 并行时每线程复用同一组缓冲，避免每个候选都 malloc
    static thread_local FitScratch sc;
    const int32_t pv = window_best_scratch(ref, read, L, q, cfg, sc, out_start);
    if (pv <= NEG) {
        if (out_start)
            *out_start = -1;
        return INT32_MIN;
    }
    return unpack_score(pv, L, cfg.band);
}

Hit cpu_align_one(const RefGenome& ref, const std::string& read, const Config& cfg) {
    const int L = (int)read.size();
    const int B = cfg.band;
    const int32_t threshold = (int32_t)std::max(0.0, cfg.min_score_ratio * cfg.match * L);

    FitScratch sc;
    int32_t best_packed = NEG;
    int64_t best_start = -1;

    // 穷举所有锚点：每个锚点覆盖 [q-B, q+B] 内的全部起始位置
    for (int64_t q = 0; q < ref.total_bases(); ++q) {
        int64_t start = -1;
        const int32_t pv = window_best_scratch(ref, read.data(), L, q, cfg, sc, &start);
        if (pv == NEG)
            continue;
        if (pv > best_packed) { // 打包值已内含"同分取最左"
            best_packed = pv;
            best_start = start;
        }
    }

    Hit h;
    if (best_packed <= NEG)
        return h;
    const int32_t score = unpack_score(best_packed, L, B);
    if (score < threshold)
        return h; // seq_id = -1 → unknown_origin

    int32_t sid = -1;
    int64_t off = -1;
    global_to_local(ref, best_start, sid, off);
    if (sid < 0)
        return h;
    h.seq_id = sid;
    h.pos = off;
    h.score = score;
    return h;
}

std::vector<Hit> cpu_align_all(const RefGenome& ref, const std::vector<Read>& reads,
                               const Config& cfg) {
    std::vector<Hit> hits(reads.size());
#pragma omp parallel for schedule(dynamic)
    for (int64_t i = 0; i < (int64_t)reads.size(); ++i) {
        hits[(size_t)i] = cpu_align_one(ref, reads[(size_t)i].bases, cfg);
    }
    return hits;
}

void global_to_local(const RefGenome& ref, int64_t gpos, int32_t& seq_id, int64_t& off) {
    seq_id = -1;
    off = -1;
    const int S = ref.num_seqs();
    int lo = 0, hi = S - 1, k = -1;
    while (lo <= hi) { // 最后一个 seq_begin <= gpos
        const int mid = (lo + hi) / 2;
        if (ref.seq_begin[(size_t)mid] <= gpos) {
            k = mid;
            lo = mid + 1;
        } else
            hi = mid - 1;
    }
    if (k < 0)
        return;
    const int64_t rel = gpos - ref.seq_begin[(size_t)k];
    if (rel < 0 || rel >= ref.lens[(size_t)k])
        return; // 落在填充/分隔区
    seq_id = k;
    off = rel;
}

} // namespace sa
