// align_gpu.cu - GPU 读段比对流水线（seed-and-extend）
//   1) 构建 k-mer 索引（计数排序 + 散列分桶）
//   2) 种子枚举：read 上的 k-mer 查索引 → 候选锚点（定长容量 + 原子追加）
//   3) 候选去重：每 read 一个线程做插入排序 + 相邻去重
//   4) 带状 DP 验证：一线程一个候选，窗口内自由起始的 fitting 比对
//   5) 每 read 归约取最优
#include "align.h"
#include "dna.h"

#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace sa {

namespace {

constexpr int32_t NEG = INT32_MIN / 4;

#define CUDA_CHECK(call)                                                                           \
    do {                                                                                           \
        cudaError_t err_ = (call);                                                                 \
        if (err_ != cudaSuccess) {                                                                 \
            throw std::runtime_error(std::string("CUDA 错误: ") + cudaGetErrorString(err_) +       \
                                     " @ " + #call);                                               \
        }                                                                                          \
    } while (0)

// ======================= k-mer 编码 =======================
// 把从 p 开始的 k 个碱基编码为 2k 位整数；含非法碱基（N）则返回 false
__device__ __forceinline__ bool encode_at(const uint8_t* b, int64_t p, int k, uint32_t* out) {
    uint32_t c = 0;
#pragma unroll 4
    for (int j = 0; j < k; ++j) {
        const uint8_t v = b[p + j];
        if (v >= 4)
            return false;
        c = (c << 2) | v;
    }
    *out = c;
    return true;
}

// ======================= 索引：计数 =======================
__global__ void kmer_count_kernel(const uint8_t* __restrict__ bases, int64_t n, int k,
                                  uint32_t mask, unsigned long long* __restrict__ count) {
    const int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for (int64_t p = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; p + k <= n; p += stride) {
        uint32_t code;
        if (!encode_at(bases, p, k, &code))
            continue;
        atomicAdd(&count[kmer_hash(code, mask)], 1ULL);
    }
}

// ======================= 索引：散列 =======================
__global__ void kmer_scatter_kernel(const uint8_t* __restrict__ bases, int64_t n, int k,
                                    uint32_t mask, unsigned long long* __restrict__ cursor,
                                    uint32_t* __restrict__ entry_code,
                                    int64_t* __restrict__ entry_pos) {
    const int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for (int64_t p = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; p + k <= n; p += stride) {
        uint32_t code;
        if (!encode_at(bases, p, k, &code))
            continue;
        const uint32_t b = kmer_hash(code, mask);
        const unsigned long long slot = atomicAdd(&cursor[b], 1ULL);
        entry_code[slot] = code;
        entry_pos[slot] = p;
    }
}

// ======================= 种子枚举 =======================
// 网格 (num_reads, max_seeds)；一线程一个 (read, seed)
__global__ void emit_candidates_kernel(
    const uint8_t* __restrict__ read_bases, const int32_t* __restrict__ read_off,
    const int32_t* __restrict__ read_len, int k, int step, int band, int64_t ref_len, uint32_t mask,
    const unsigned long long* __restrict__ bucket_off, const uint32_t* __restrict__ entry_code,
    const int64_t* __restrict__ entry_pos, int scan_cap, int max_cand, int64_t* __restrict__ cand_q,
    int32_t* __restrict__ cand_cnt) {
    const int ri = blockIdx.x;
    const int si = blockIdx.y * blockDim.x + threadIdx.x;
    const int L = read_len[ri];
    const int off = si * step;
    if (off + k > L)
        return;

    uint32_t code;
    if (!encode_at(read_bases, read_off[ri] + off, k, &code))
        return;

    const uint32_t b = kmer_hash(code, mask);
    const int64_t e0 = (int64_t)bucket_off[b];
    const int64_t e1 = (int64_t)bucket_off[b + 1];
    const int64_t lim = (int64_t)scan_cap < (e1 - e0) ? (int64_t)scan_cap : (e1 - e0);

    for (int64_t e = e0; e < e0 + lim; ++e) {
        if (entry_code[e] != code)
            continue;                         // 过滤散列碰撞
        const int64_t q = entry_pos[e] - off; // 候选锚点
        // 锚点必须让整窗口落在参考内
        if (q < band || q + L + band >= ref_len)
            continue;
        const int32_t idx = atomicAdd(&cand_cnt[ri], 1);
        if (idx < max_cand)
            cand_q[(int64_t)ri * max_cand + idx] = q;
    }
}

// ======================= 候选排序 + 去重 =======================
__global__ void dedup_kernel(int num_reads, int max_cand, int64_t* __restrict__ cand_q,
                             int32_t* __restrict__ cand_cnt) {
    const int ri = blockIdx.x * blockDim.x + threadIdx.x;
    if (ri >= num_reads)
        return;
    int32_t n = cand_cnt[ri];
    if (n > max_cand)
        n = max_cand;
    if (n <= 0) {
        cand_cnt[ri] = 0;
        return;
    }

    int64_t* a = cand_q + (int64_t)ri * max_cand;
    for (int i = 1; i < n; ++i) { // 插入排序
        const int64_t key = a[i];
        int j = i - 1;
        while (j >= 0 && a[j] > key) {
            a[j + 1] = a[j];
            --j;
        }
        a[j + 1] = key;
    }
    int m = 1;
    for (int i = 1; i < n; ++i) { // 相邻去重
        if (a[i] != a[m - 1])
            a[m++] = a[i];
    }
    cand_cnt[ri] = m;
}

// ======================= 归集到 CSR =======================
__global__ void gather_candidates_kernel(int num_reads, int max_cand,
                                         const int64_t* __restrict__ cand_q,
                                         const int32_t* __restrict__ cand_cnt,
                                         const int32_t* __restrict__ cand_off,
                                         int64_t* __restrict__ out_q) {
    const int ri = blockIdx.x * blockDim.x + threadIdx.x;
    if (ri >= num_reads)
        return;
    const int32_t n = cand_cnt[ri];
    const int32_t base = cand_off[ri];
    const int64_t* a = cand_q + (int64_t)ri * max_cand;
    for (int i = 0; i < n; ++i)
        out_q[base + i] = a[i];
}

// ======================= 带状 DP 验证 =======================
// 窗口 ref[q-B, q+L+B)，长度 W = L+2B+1；比对可在窗口内任意列起始。
// 列下标 e = j - i - B ∈ [-B, B]。把 (得分, 起始列排名) 打包进 int32：
//   packed = (score + L) * (2B+1) + (2B - t)，t 为起始列编号 ∈ [0, 2B]
// 取最大 ⇒ 得分最大；同分 ⇒ t 最小 ⇒ 起始位置最靠左（与 CPU 参考一致）。
template <int B>
__global__ void
verify_kernel(int32_t num_cands, const int64_t* __restrict__ cand_q,
              const int64_t* __restrict__ cand_read, const uint8_t* __restrict__ read_bases,
              const int32_t* __restrict__ read_off, const int32_t* __restrict__ read_len,
              const uint8_t* __restrict__ ref_bases, int64_t ref_len, int match, int mismatch,
              int gap, int32_t* __restrict__ out_score, int64_t* __restrict__ out_start) {
    const int32_t c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= num_cands)
        return;

    constexpr int WID = 2 * B + 1;
    const int ri = (int)cand_read[c];
    const int L = read_len[ri];
    const int64_t q = cand_q[c];
    const int64_t qb = q - B;
    const int64_t W = (int64_t)L + 2 * B + 1;

    if (qb < 0 || qb + W > ref_len) {
        out_score[c] = INT32_MIN;
        out_start[c] = -1;
        return;
    }

    int32_t prev[WID], cur[WID];
    const int32_t base0 = L * WID;
#pragma unroll
    for (int e = -B; e <= B; ++e) {
        const int t = B + e;
        prev[e + B] = base0 + (2 * B - t);
    }

    const int32_t wm = match * WID;
    const int32_t wmm = mismatch * WID;
    const int32_t wg = gap * WID;

    const uint8_t* rd = read_bases + read_off[ri];
    const uint8_t* rw = ref_bases + qb;

    for (int i = 1; i <= L; ++i) {
#pragma unroll
        for (int k = 0; k < WID; ++k)
            cur[k] = NEG;
        const int ra = rd[i - 1];
#pragma unroll
        for (int e = -B; e <= B; ++e) {
            const int j = i + B + e;
            if (j > W)
                break;
            int32_t best = NEG;
            if (j >= 1) { // 对角
                const int32_t d = prev[e + B];
                if (d > NEG) {
                    const int rc = rw[j - 1];
                    best = d + ((ra < 4 && ra == rc) ? wm : wmm);
                }
            }
            if (e + 1 <= B) { // 参考中插入空位
                const int32_t u = prev[e + 1 + B];
                if (u > NEG)
                    best = max(best, u + wg);
            }
            if (e - 1 >= -B) { // read 中插入空位
                const int32_t l = cur[e - 1 + B];
                if (l > NEG)
                    best = max(best, l + wg);
            }
            cur[e + B] = best;
        }
#pragma unroll
        for (int k = 0; k < WID; ++k)
            prev[k] = cur[k];
    }

    int32_t best = NEG;
#pragma unroll
    for (int k = 0; k < WID; ++k)
        best = max(best, prev[k]);

    if (best <= NEG) {
        out_score[c] = INT32_MIN;
        out_start[c] = -1;
    } else {
        out_score[c] = best / WID - L;
        const int t = 2 * B - (best % WID);
        out_start[c] = qb + t;
    }
}

// ======================= 每 read 归约 =======================
__global__ void reduce_kernel(int32_t num_reads, const int32_t* __restrict__ cand_off,
                              const int32_t* __restrict__ cand_cnt,
                              const int64_t* __restrict__ cand_start,
                              const int32_t* __restrict__ cand_score,
                              const int32_t* __restrict__ read_len, int match, double min_ratio,
                              int32_t* __restrict__ best_score, int64_t* __restrict__ best_start) {
    const int32_t ri = blockIdx.x * blockDim.x + threadIdx.x;
    if (ri >= num_reads)
        return;

    const int32_t base = cand_off[ri];
    const int32_t n = cand_cnt[ri];
    int32_t bs = INT32_MIN;
    int64_t bq = -1;
    for (int32_t i = 0; i < n; ++i) {
        const int32_t s = cand_score[base + i];
        if (s == INT32_MIN)
            continue;
        const int64_t st = cand_start[base + i];
        if (s > bs || (s == bs && (bq < 0 || st < bq))) { // 同分取最左
            bs = s;
            bq = st;
        }
    }
    const int32_t threshold = (int32_t)(min_ratio * match * read_len[ri]);
    if (bs < threshold) {
        bs = INT32_MIN;
        bq = -1;
    }
    best_score[ri] = bs;
    best_start[ri] = bq;
}

// ======================= 设备端索引 =======================
struct IndexDev {
    int64_t ref_len = 0;
    int32_t k = 0;
    uint32_t num_buckets = 0;
    uint32_t mask = 0;
    int64_t num_entries = 0;
    uint8_t* d_bases = nullptr;                 // 参考碱基编码
    unsigned long long* d_bucket_off = nullptr; // num_buckets + 1
    uint32_t* d_entry_code = nullptr;           // num_entries
    int64_t* d_entry_pos = nullptr;             // num_entries
    unsigned long long* d_cursor = nullptr;     // 散列阶段用
};

void free_index(IndexDev& ix) {
    if (ix.d_bases)
        cudaFree(ix.d_bases);
    if (ix.d_bucket_off)
        cudaFree(ix.d_bucket_off);
    if (ix.d_entry_code)
        cudaFree(ix.d_entry_code);
    if (ix.d_entry_pos)
        cudaFree(ix.d_entry_pos);
    if (ix.d_cursor)
        cudaFree(ix.d_cursor);
    ix = IndexDev{};
}

double ms_since(const std::chrono::high_resolution_clock::time_point& t0) {
    using clk = std::chrono::high_resolution_clock;
    return std::chrono::duration<double, std::milli>(clk::now() - t0).count();
}

} // namespace

// ======================= GpuAligner::Impl =======================
struct GpuAligner::Impl {
    IndexDev ix;
    RefGenome href;
    AlignStats st;

    uint8_t* d_read_bases = nullptr;
    int32_t* d_read_off = nullptr;
    int32_t* d_read_len = nullptr;

    int64_t* d_cand_q = nullptr;    // [R * max_cand]
    int32_t* d_cand_cnt = nullptr;  // [R]
    int64_t* d_cand2_q = nullptr;   // 去重后 CSR（锚点）
    int32_t* d_cand2_off = nullptr; // [R+1]
    int64_t* d_cand_read = nullptr; // [候选数]
    int32_t* d_cand_score = nullptr;
    int64_t* d_cand_start = nullptr;

    int32_t* d_best_score = nullptr;
    int64_t* d_best_start = nullptr;

    void build_index(const RefGenome& ref, const Config& cfg) {
        const auto t0 = std::chrono::high_resolution_clock::now();
        href = ref;
        const int64_t n = ref.total_bases();
        const int k = cfg.k;
        if (k < 1 || k > 16)
            throw std::runtime_error("k 必须在 [1,16] 范围内");

        ix.ref_len = n;
        ix.k = k;
        // 桶数按"平均负载 16 个条目/桶"自动选取：
        // 桶越少，主机端前缀和与 H2D/D2H 传输越省；桶越多，种子查表越快。
        // 12~16 的负载下查表扫描仍很短，故取 16。
        int bits = cfg.index_bits;
        if (bits <= 0) {
            bits = 16;
            while (bits < 28 && (1LL << bits) < (n / 16))
                ++bits;
        }
        if (bits < 8 || bits > 30)
            throw std::runtime_error("index_bits 必须在 [8,30] 范围内");
        ix.num_buckets = 1u << bits;
        ix.mask = ix.num_buckets - 1;

        std::vector<uint8_t> codes((size_t)n);
        for (int64_t i = 0; i < n; ++i) {
            const int c = base_code(ref.bases[(size_t)i]);
            codes[(size_t)i] = (uint8_t)(c > 3 ? 4 : c);
        }
        CUDA_CHECK(cudaMalloc(&ix.d_bases, (size_t)n));
        CUDA_CHECK(cudaMemcpy(ix.d_bases, codes.data(), (size_t)n, cudaMemcpyHostToDevice));

        unsigned long long* d_count = nullptr;
        CUDA_CHECK(cudaMalloc(&d_count, sizeof(unsigned long long) * ix.num_buckets));
        CUDA_CHECK(cudaMemset(d_count, 0, sizeof(unsigned long long) * ix.num_buckets));

        const int threads = 256;
        int64_t blocks = (n > 0) ? (n + threads - 1) / threads : 1;
        if (blocks > 65535)
            blocks = 65535;
        if (blocks < 1)
            blocks = 1;

        kmer_count_kernel<<<(int)blocks, threads>>>(ix.d_bases, n, k, ix.mask, d_count);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // 桶前缀和（主机端线性扫描）
        std::vector<unsigned long long> hcount((size_t)ix.num_buckets);
        CUDA_CHECK(cudaMemcpy(hcount.data(), d_count, sizeof(unsigned long long) * ix.num_buckets,
                              cudaMemcpyDeviceToHost));
        cudaFree(d_count);

        std::vector<unsigned long long> hoff((size_t)ix.num_buckets + 1);
        hoff[0] = 0;
        for (size_t i = 0; i < hcount.size(); ++i)
            hoff[i + 1] = hoff[i] + hcount[i];
        ix.num_entries = (int64_t)hoff[ix.num_buckets];

        CUDA_CHECK(cudaMalloc(&ix.d_bucket_off, sizeof(unsigned long long) * (ix.num_buckets + 1)));
        CUDA_CHECK(cudaMemcpy(ix.d_bucket_off, hoff.data(),
                              sizeof(unsigned long long) * (ix.num_buckets + 1),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc(&ix.d_cursor, sizeof(unsigned long long) * ix.num_buckets));
        CUDA_CHECK(cudaMemcpy(ix.d_cursor, hoff.data(), sizeof(unsigned long long) * ix.num_buckets,
                              cudaMemcpyHostToDevice));

        if (ix.num_entries > 0) {
            CUDA_CHECK(cudaMalloc(&ix.d_entry_code, sizeof(uint32_t) * (size_t)ix.num_entries));
            CUDA_CHECK(cudaMalloc(&ix.d_entry_pos, sizeof(int64_t) * (size_t)ix.num_entries));
            kmer_scatter_kernel<<<(int)blocks, threads>>>(ix.d_bases, n, k, ix.mask, ix.d_cursor,
                                                          ix.d_entry_code, ix.d_entry_pos);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
        }
        cudaFree(ix.d_cursor);
        ix.d_cursor = nullptr;

        st.index_entries = ix.num_entries;
        st.build_index_ms = ms_since(t0);
    }

    std::vector<Hit> align(const std::vector<Read>& reads, const Config& cfg) {
        const auto t_all = std::chrono::high_resolution_clock::now();
        const int32_t R = (int32_t)reads.size();
        if (R == 0)
            return {};

        // ---------- 上传 reads ----------
        auto t = std::chrono::high_resolution_clock::now();
        int64_t total_bases = 0;
        int32_t max_len = 0;
        for (const auto& r : reads) {
            total_bases += (int64_t)r.bases.size();
            max_len = std::max(max_len, (int32_t)r.bases.size());
        }
        std::vector<uint8_t> rb((size_t)total_bases);
        std::vector<int32_t> roff((size_t)R), rlen((size_t)R);
        int64_t acc = 0;
        for (int32_t i = 0; i < R; ++i) {
            roff[(size_t)i] = (int32_t)acc;
            rlen[(size_t)i] = (int32_t)reads[(size_t)i].bases.size();
            for (size_t j = 0; j < reads[(size_t)i].bases.size(); ++j) {
                const int c = base_code(reads[(size_t)i].bases[j]);
                rb[(size_t)(acc + (int64_t)j)] = (uint8_t)(c > 3 ? 4 : c);
            }
            acc += (int64_t)reads[(size_t)i].bases.size();
        }
        CUDA_CHECK(cudaMalloc(&d_read_bases, (size_t)std::max<int64_t>(total_bases, 1)));
        CUDA_CHECK(
            cudaMemcpy(d_read_bases, rb.data(), (size_t)total_bases, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc(&d_read_off, sizeof(int32_t) * (size_t)R));
        CUDA_CHECK(cudaMemcpy(d_read_off, roff.data(), sizeof(int32_t) * (size_t)R,
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc(&d_read_len, sizeof(int32_t) * (size_t)R));
        CUDA_CHECK(cudaMemcpy(d_read_len, rlen.data(), sizeof(int32_t) * (size_t)R,
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaDeviceSynchronize());
        st.upload_ms = ms_since(t);

        // ---------- 候选枚举 ----------
        t = std::chrono::high_resolution_clock::now();
        const int k = cfg.k;
        const int step = std::max(1, cfg.seed_step);
        const int max_cand = cfg.max_candidates;
        const int max_seeds = std::max(1, (max_len - k) / step + 1);

        CUDA_CHECK(cudaMalloc(&d_cand_q, sizeof(int64_t) * (size_t)R * (size_t)max_cand));
        CUDA_CHECK(cudaMalloc(&d_cand_cnt, sizeof(int32_t) * (size_t)R));
        CUDA_CHECK(cudaMemset(d_cand_cnt, 0, sizeof(int32_t) * (size_t)R));

        if (ix.num_entries > 0) {
            const dim3 grid((unsigned)R, (unsigned)max_seeds);
            emit_candidates_kernel<<<grid, 64>>>(d_read_bases, d_read_off, d_read_len, k, step,
                                                 cfg.band, ix.ref_len, ix.mask, ix.d_bucket_off,
                                                 ix.d_entry_code, ix.d_entry_pos, 512, max_cand,
                                                 d_cand_q, d_cand_cnt);
            CUDA_CHECK(cudaGetLastError());
        }
        st.num_seeds = (int64_t)R * max_seeds;

        { // 去重
            const int threads = 128;
            const int blocks = (R + threads - 1) / threads;
            dedup_kernel<<<blocks, threads>>>(R, max_cand, d_cand_q, d_cand_cnt);
            CUDA_CHECK(cudaGetLastError());
        }

        // 候选 CSR 前缀和
        std::vector<int32_t> cnt((size_t)R);
        CUDA_CHECK(cudaMemcpy(cnt.data(), d_cand_cnt, sizeof(int32_t) * (size_t)R,
                              cudaMemcpyDeviceToHost));
        std::vector<int32_t> coff((size_t)R + 1);
        coff[0] = 0;
        for (int32_t i = 0; i < R; ++i)
            coff[(size_t)i + 1] = coff[(size_t)i] + cnt[(size_t)i];
        const int64_t total_cand = coff[(size_t)R];
        st.num_candidates = total_cand;

        CUDA_CHECK(cudaMalloc(&d_cand2_off, sizeof(int32_t) * ((size_t)R + 1)));
        CUDA_CHECK(cudaMemcpy(d_cand2_off, coff.data(), sizeof(int32_t) * ((size_t)R + 1),
                              cudaMemcpyHostToDevice));

        std::vector<Hit> hits((size_t)R);

        if (total_cand > 0) {
            CUDA_CHECK(cudaMalloc(&d_cand2_q, sizeof(int64_t) * (size_t)total_cand));
            CUDA_CHECK(cudaMalloc(&d_cand_read, sizeof(int64_t) * (size_t)total_cand));
            CUDA_CHECK(cudaMalloc(&d_cand_score, sizeof(int32_t) * (size_t)total_cand));
            CUDA_CHECK(cudaMalloc(&d_cand_start, sizeof(int64_t) * (size_t)total_cand));

            {
                const int threads = 128;
                const int blocks = (R + threads - 1) / threads;
                gather_candidates_kernel<<<blocks, threads>>>(R, max_cand, d_cand_q, d_cand_cnt,
                                                              d_cand2_off, d_cand2_q);
                CUDA_CHECK(cudaGetLastError());
            }
            { // 候选 → read id
                std::vector<int64_t> crd((size_t)total_cand);
                for (int32_t i = 0; i < R; ++i)
                    for (int32_t j = coff[(size_t)i]; j < coff[(size_t)i + 1]; ++j)
                        crd[(size_t)j] = i;
                CUDA_CHECK(cudaMemcpy(d_cand_read, crd.data(), sizeof(int64_t) * (size_t)total_cand,
                                      cudaMemcpyHostToDevice));
            }
            CUDA_CHECK(cudaDeviceSynchronize());
        }
        st.seed_ms = ms_since(t);

        // ---------- 带状 DP 验证 ----------
        t = std::chrono::high_resolution_clock::now();
        if (total_cand > 0) {
            const int threads = 256;
            const int blocks = (int)((total_cand + threads - 1) / threads);
            switch (cfg.band) {
            case 4:
                verify_kernel<4>
                    <<<blocks, threads>>>((int32_t)total_cand, d_cand2_q, d_cand_read, d_read_bases,
                                          d_read_off, d_read_len, ix.d_bases, ix.ref_len, cfg.match,
                                          cfg.mismatch, cfg.gap, d_cand_score, d_cand_start);
                break;
            case 8:
                verify_kernel<8>
                    <<<blocks, threads>>>((int32_t)total_cand, d_cand2_q, d_cand_read, d_read_bases,
                                          d_read_off, d_read_len, ix.d_bases, ix.ref_len, cfg.match,
                                          cfg.mismatch, cfg.gap, d_cand_score, d_cand_start);
                break;
            case 16:
                verify_kernel<16>
                    <<<blocks, threads>>>((int32_t)total_cand, d_cand2_q, d_cand_read, d_read_bases,
                                          d_read_off, d_read_len, ix.d_bases, ix.ref_len, cfg.match,
                                          cfg.mismatch, cfg.gap, d_cand_score, d_cand_start);
                break;
            default:
                throw std::runtime_error("band 仅支持 4 / 8 / 16");
            }
            CUDA_CHECK(cudaGetLastError());
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        st.verify_ms = ms_since(t);

        // ---------- 每 read 归约 ----------
        t = std::chrono::high_resolution_clock::now();
        CUDA_CHECK(cudaMalloc(&d_best_score, sizeof(int32_t) * (size_t)R));
        CUDA_CHECK(cudaMalloc(&d_best_start, sizeof(int64_t) * (size_t)R));
        {
            const int threads = 256;
            const int blocks = (R + threads - 1) / threads;
            reduce_kernel<<<blocks, threads>>>(R, d_cand2_off, d_cand_cnt, d_cand_start,
                                               d_cand_score, d_read_len, cfg.match,
                                               cfg.min_score_ratio, d_best_score, d_best_start);
            CUDA_CHECK(cudaGetLastError());
        }

        std::vector<int32_t> bs((size_t)R);
        std::vector<int64_t> bq((size_t)R);
        CUDA_CHECK(cudaMemcpy(bs.data(), d_best_score, sizeof(int32_t) * (size_t)R,
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(bq.data(), d_best_start, sizeof(int64_t) * (size_t)R,
                              cudaMemcpyDeviceToHost));
        st.reduce_ms = ms_since(t);

        // ---------- 坐标映射 ----------
        for (int32_t i = 0; i < R; ++i) {
            if (bq[(size_t)i] < 0)
                continue;
            int32_t sid = -1;
            int64_t off = -1;
            global_to_local(href, bq[(size_t)i], sid, off);
            if (sid < 0)
                continue;
            hits[(size_t)i].seq_id = sid;
            hits[(size_t)i].pos = off;
            hits[(size_t)i].score = bs[(size_t)i];
        }
        st.total_ms = ms_since(t_all);
        return hits;
    }

    void release() {
        free_index(ix);
        if (d_read_bases)
            cudaFree(d_read_bases);
        if (d_read_off)
            cudaFree(d_read_off);
        if (d_read_len)
            cudaFree(d_read_len);
        if (d_cand_q)
            cudaFree(d_cand_q);
        if (d_cand_cnt)
            cudaFree(d_cand_cnt);
        if (d_cand2_q)
            cudaFree(d_cand2_q);
        if (d_cand2_off)
            cudaFree(d_cand2_off);
        if (d_cand_read)
            cudaFree(d_cand_read);
        if (d_cand_score)
            cudaFree(d_cand_score);
        if (d_cand_start)
            cudaFree(d_cand_start);
        if (d_best_score)
            cudaFree(d_best_score);
        if (d_best_start)
            cudaFree(d_best_start);
        d_read_bases = nullptr;
        d_read_off = nullptr;
        d_read_len = nullptr;
        d_cand_q = nullptr;
        d_cand_cnt = nullptr;
        d_cand2_q = nullptr;
        d_cand2_off = nullptr;
        d_cand_read = nullptr;
        d_cand_score = nullptr;
        d_cand_start = nullptr;
        d_best_score = nullptr;
        d_best_start = nullptr;
    }
};

// ======================= 接口 =======================
GpuAligner::GpuAligner() : impl_(new Impl()) {}
GpuAligner::~GpuAligner() {
    impl_->release();
    delete impl_;
}

void GpuAligner::build_index(const RefGenome& ref, const Config& cfg) {
    impl_->build_index(ref, cfg);
}

std::vector<Hit> GpuAligner::align(const std::vector<Read>& reads, const Config& cfg) {
    return impl_->align(reads, cfg);
}

const AlignStats& GpuAligner::stats() const {
    return impl_->st;
}

void GpuAligner::release() {
    impl_->release();
}

} // namespace sa
