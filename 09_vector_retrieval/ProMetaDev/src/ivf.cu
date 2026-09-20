// ivf.cu - IVF-Flat 索引构建与查询
//
// 索引构建（k-means + 倒排表）：
//   1) 从向量库抽取子样本训练 k-means（避免在百万级数据上反复做全量分配）
//   2) 每轮：分配（assign）→ 计数 → 前缀和 → 散列成倒排表 → 更新聚类中心
//   3) 用训练好的中心对全量数据做一次分配，得到最终倒排表
//   4) 索引落盘（聚类中心 + list_start/list_ids）
//
// 查询（IVF-Flat）：
//   Phase A：计算 query 到 nlist 个聚类中心的距离，选出最近的 nprobe 个
//   Phase B：扫描这些倒排桶中的全部向量（精确距离），做与精确检索相同的 Top-K
//
// 说明：粗量化（聚类）统一使用 L2 距离；细扫（Phase B）使用配置的度量。
//       cosine 在进入本流程前已把向量归一化，等价于内积。
#include "search.h"
#include "topk_merge.cuh"
#include "normalize.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cfloat>
#include <climits>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

namespace vs {
namespace {

constexpr int kThreads = 256;

// ======================= 分配：向量 → 最近聚类中心 =======================
// 中心按 CT 个一组放入共享内存，向量按 DC 分块读入寄存器，避免重复访存
template <int CT, int DC, int THREADS>
__global__ void k_assign(const float* __restrict__ base, int64_t n, int32_t dim,
                         const float* __restrict__ centroids, int nlist,
                         int32_t* __restrict__ labels) {
    extern __shared__ float s_c[]; // CT * dim
    const int tid = threadIdx.x;
    const int64_t stride = (int64_t)gridDim.x * THREADS;
    // 注意：外层循环条件只依赖 blockIdx，保证块内所有线程的迭代次数一致。
    // 若按线程判断 (v < n)，块内线程迭代次数不同，内层 __syncthreads() 会分歧（UB），
    // 曾因此产生错误的标签、进而使倒排表出现重复与缺失。
    for (int64_t v0 = (int64_t)blockIdx.x * THREADS; v0 < n; v0 += stride) {
        const int64_t v = v0 + tid;
        const bool active = (v < n);
        const float* x = base + (active ? v : 0) * dim;
        float best = FLT_MAX;
        int bestc = 0;
        for (int ct = 0; ct < nlist; ct += CT) {
            // 协同载入 CT 个中心到共享内存
            for (int i = tid; i < CT * dim; i += THREADS) {
                int ci = i / dim, d = i % dim;
                int gc = ct + ci;
                s_c[i] = (gc < nlist) ? centroids[(int64_t)gc * dim + d] : 0.0f;
            }
            __syncthreads();
            if (active) {
                float acc[CT];
#pragma unroll
                for (int i = 0; i < CT; i++)
                    acc[i] = 0.0f;
                for (int32_t d0 = 0; d0 < dim; d0 += DC) {
                    float xv[DC];
                    const int32_t dn = min(DC, dim - d0);
#pragma unroll
                    for (int k = 0; k < DC; k++)
                        xv[k] = (k < dn) ? x[d0 + k] : 0.0f;
#pragma unroll
                    for (int ci = 0; ci < CT; ci++) {
                        const float* c = s_c + ci * dim + d0;
#pragma unroll
                        for (int k = 0; k < DC; k++) {
                            // dim 不是 DC 的整数倍时，尾部不足的通道必须跳过，
                            // 否则会读到相邻中心的数据
                            if (d0 + k < dim) {
                                float t = xv[k] - c[k];
                                acc[ci] += t * t;
                            }
                        }
                    }
                }
                int nc = min(CT, nlist - ct);
                for (int ci = 0; ci < nc; ci++) {
                    if (acc[ci] < best) {
                        best = acc[ci];
                        bestc = ct + ci;
                    }
                }
            }
            __syncthreads();
        }
        if (active)
            labels[v] = bestc;
    }
}

// ======================= 倒排表：计数 + 散列 =======================
__global__ void k_count_labels(const int32_t* __restrict__ labels, int64_t n, int nlist,
                               int32_t* __restrict__ counts) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        atomicAdd(&counts[labels[i]], 1);
}

__global__ void k_scatter(const int32_t* __restrict__ labels, int64_t n,
                          const int32_t* __restrict__ offset, int32_t* __restrict__ cursor,
                          int32_t* __restrict__ list_ids) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    int c = labels[i];
    int32_t pos = offset[c] + atomicAdd(&cursor[c], 1);
    list_ids[pos] = (int32_t)i;
}

// ======================= 更新聚类中心 =======================
__global__ void k_update_centroids(const float* __restrict__ base, int32_t dim,
                                   const int32_t* __restrict__ list_start,
                                   const int32_t* __restrict__ list_ids,
                                   float* __restrict__ centroids) {
    extern __shared__ float s_sum[]; // dim
    const int c = blockIdx.x;
    const int64_t lo = list_start[c], hi = list_start[c + 1];
    const int64_t cnt = hi - lo;
    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
        float s = 0.0f;
        for (int64_t k = lo; k < hi; k++)
            s += base[(int64_t)list_ids[k] * dim + d];
        s_sum[d] = s;
    }
    __syncthreads();
    if (threadIdx.x == 0 && cnt > 0) {
        float inv = 1.0f / (float)cnt;
        float* cen = centroids + (int64_t)c * dim;
        for (int d = 0; d < dim; d++)
            cen[d] = s_sum[d] * inv;
    }
}

// ======================= IVF-Flat 查询 =======================
template <int K, int THREADS>
__global__ void
k_ivf_query(const float* __restrict__ base, int64_t n, int32_t dim,
            const float* __restrict__ queries, int64_t q_offset,
            const float* __restrict__ centroids, int nlist, const int32_t* __restrict__ list_start,
            const int32_t* __restrict__ list_ids, int nprobe, int smaller,
            float* __restrict__ out_d, int32_t* __restrict__ out_i, int32_t* __restrict__ scanned) {
    constexpr int NWARP = THREADS / 32;
    extern __shared__ char smem[];
    float* s_cd = reinterpret_cast<float*>(smem);                 // nlist
    float* sq = s_cd + nlist;                                     // dim
    int32_t* s_probe = reinterpret_cast<int32_t*>(sq + dim);      // nprobe
    float* s_wd = reinterpret_cast<float*>(s_probe + nprobe);     // NWARP*K
    int32_t* s_wi = reinterpret_cast<int32_t*>(s_wd + NWARP * K); // NWARP*K
    __shared__ float s_rv[THREADS];
    __shared__ int32_t s_ri[THREADS];

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int64_t qi = q_offset + blockIdx.x;
    const bool smallerb = (smaller != 0);
    const float WORST = smallerb ? FLT_MAX : -FLT_MAX;

    // query 载入共享内存
    for (int32_t d = tid; d < dim; d += THREADS)
        sq[d] = queries[qi * dim + d];
    __syncthreads();

    // ---- Phase A：选 nprobe 个最近聚类中心（粗量化用 L2）----
    for (int c = tid; c < nlist; c += THREADS) {
        const float* cen = centroids + (int64_t)c * dim;
        float acc = 0.0f;
        for (int32_t d = 0; d < dim; d++) {
            float t = sq[d] - cen[d];
            acc += t * t;
        }
        s_cd[c] = acc;
    }
    __syncthreads();

    int nprob = min(nprobe, nlist);
    for (int r = 0; r < nprob; r++) {
        // 线程局部最小
        float best = FLT_MAX;
        int bidx = INT_MAX;
        for (int c = tid; c < nlist; c += THREADS) {
            float cd = s_cd[c];
            if (cd < best || (cd == best && c < bidx)) {
                best = cd;
                bidx = c;
            }
        }
        // warp 内归约（值 + 下标一起）
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            float ov = __shfl_xor_sync(0xffffffffu, best, off, 32);
            int oid = __shfl_xor_sync(0xffffffffu, bidx, off, 32);
            if (ov < best || (ov == best && oid < bidx)) {
                best = ov;
                bidx = oid;
            }
        }
        if (lane == 0) {
            s_rv[warp] = best;
            s_ri[warp] = bidx;
        }
        __syncthreads();
        if (warp == 0) {
            float bv = (lane < NWARP) ? s_rv[lane] : FLT_MAX;
            int bi = (lane < NWARP) ? s_ri[lane] : INT_MAX;
#pragma unroll
            for (int off = NWARP / 2; off > 0; off >>= 1) {
                float ov = __shfl_xor_sync(0xffffffffu, bv, off, NWARP);
                int oid = __shfl_xor_sync(0xffffffffu, bi, off, NWARP);
                if (ov < bv || (ov == bv && oid < bi)) {
                    bv = ov;
                    bi = oid;
                }
            }
            if (lane == 0 && bi != INT_MAX) {
                s_probe[r] = bi;
                s_cd[bi] = FLT_MAX; // 标记已选
            }
        }
        __syncthreads();
    }

    // ---- Phase B：扫描选中的倒排桶，精确距离 + Top-K ----
    float bd[K];
    int32_t bi_[K];
#pragma unroll
    for (int i = 0; i < K; i++) {
        bd[i] = WORST;
        bi_[i] = INT_MAX;
    }

    int64_t scan_cnt = 0;
    for (int r = 0; r < nprob; r++) {
        const int c = s_probe[r];
        const int64_t lo = list_start[c], hi = list_start[c + 1];
        scan_cnt += (hi - lo);
        for (int64_t k = lo + tid; k < hi; k += THREADS) {
            const int32_t id = list_ids[k];
            const float* b = base + (int64_t)id * dim;
            float acc = 0.0f;
            for (int32_t d = 0; d < dim; d++) {
                if (smallerb) {
                    float t = sq[d] - b[d];
                    acc += t * t;
                } else {
                    acc += sq[d] * b[d];
                }
            }
            if (is_better_d(acc, id, bd[K - 1], bi_[K - 1], smallerb)) {
                int pos = K - 1;
                while (pos > 0 && is_better_d(acc, id, bd[pos - 1], bi_[pos - 1], smallerb)) {
                    bd[pos] = bd[pos - 1];
                    bi_[pos] = bi_[pos - 1];
                    pos--;
                }
                bd[pos] = acc;
                bi_[pos] = id;
            }
        }
    }
    if (tid == 0 && scanned)
        scanned[qi] = (int32_t)scan_cnt;

    block_topk_merge<K, THREADS>(bd, bi_, smallerb, s_wd, s_wi, out_d, out_i, qi * K);
}

// ======================= 主机端辅助 =======================
struct DevBuf {
    void* p = nullptr;
    ~DevBuf() {
        if (p)
            cudaFree(p);
    }
    bool alloc(size_t bytes) {
        return cudaMalloc(&p, bytes) == cudaSuccess;
    }
};

// 前缀和（nlist 很小，直接在主机上算）
void prefix_sum_host(const std::vector<int32_t>& counts, std::vector<int32_t>& starts) {
    starts.assign(counts.size() + 1, 0);
    int32_t acc = 0;
    for (size_t i = 0; i < counts.size(); i++) {
        starts[i] = acc;
        acc += counts[i];
    }
    starts[counts.size()] = acc;
}

// 用给定的聚类中心对 data 做一次分配，并（可选）输出倒排表
bool assign_and_build_lists(const float* d_data, int64_t n, int32_t dim, const float* d_centroids,
                            int nlist, int32_t* d_labels, std::vector<int32_t>* out_list_start,
                            std::vector<int32_t>* out_list_ids, std::string& err) {
    DevBuf d_counts, d_cursor, d_start, d_ids;
    if (!d_counts.alloc((size_t)nlist * sizeof(int32_t)) ||
        !d_cursor.alloc((size_t)nlist * sizeof(int32_t)) ||
        !d_start.alloc((size_t)(nlist + 1) * sizeof(int32_t)) ||
        !d_ids.alloc((size_t)std::max<int64_t>(n, 1) * sizeof(int32_t))) {
        err = "显存分配失败(list)";
        return false;
    }
    if (cudaMemset(d_counts.p, 0, (size_t)nlist * sizeof(int32_t)) != cudaSuccess) {
        err = "memset counts 失败";
        return false;
    }

    // 分配
    {
        constexpr int CT = 32, DC = 16;
        size_t smem = (size_t)CT * dim * sizeof(float);
        int blocks = (int)std::min<int64_t>((n + kThreads - 1) / kThreads, 8192);
        k_assign<CT, DC, kThreads>
            <<<blocks, kThreads, smem>>>(d_data, n, dim, d_centroids, nlist, d_labels);
        if (cudaGetLastError() != cudaSuccess) {
            err = std::string("k_assign 失败: ") + cudaGetErrorString(cudaGetLastError());
            return false;
        }
    }
    // 计数
    {
        int blocks = (int)std::min<int64_t>((n + kThreads - 1) / kThreads, 32768);
        k_count_labels<<<blocks, kThreads>>>((const int32_t*)d_labels, n, nlist,
                                             (int32_t*)d_counts.p);
    }
    // 前缀和（主机）→ 回传
    std::vector<int32_t> counts((size_t)nlist);
    if (cudaMemcpy(counts.data(), d_counts.p, (size_t)nlist * sizeof(int32_t),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
        err = "回读 counts 失败";
        return false;
    }
    std::vector<int32_t> starts;
    prefix_sum_host(counts, starts);
    // cursor 必须清零：散列时 pos = offset[c] + atomicAdd(&cursor[c], 1)，
    // 若把 cursor 初始化成前缀和，偏移会被加两次，导致位置重叠（重复+缺失）。
    if (cudaMemcpy(d_start.p, starts.data(), (size_t)(nlist + 1) * sizeof(int32_t),
                   cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemset(d_cursor.p, 0, (size_t)nlist * sizeof(int32_t)) != cudaSuccess) {
        err = "上传前缀和失败";
        return false;
    }
    // 散列
    {
        int blocks = (int)std::min<int64_t>((n + kThreads - 1) / kThreads, 32768);
        k_scatter<<<blocks, kThreads>>>((const int32_t*)d_labels, n, (const int32_t*)d_start.p,
                                        (int32_t*)d_cursor.p, (int32_t*)d_ids.p);
    }
    if (out_list_start)
        *out_list_start = starts;
    if (out_list_ids) {
        out_list_ids->resize((size_t)n);
        if (cudaMemcpy(out_list_ids->data(), d_ids.p, (size_t)n * sizeof(int32_t),
                       cudaMemcpyDeviceToHost) != cudaSuccess) {
            err = "回读 list_ids 失败";
            return false;
        }
    }
    return true;
}

// 更新聚类中心（一次）
bool update_centroids(const float* d_data, int32_t dim, const std::vector<int32_t>& starts,
                      const std::vector<int32_t>& ids, float* d_centroids, int nlist,
                      std::string& err) {
    DevBuf d_start, d_ids;
    if (!d_start.alloc((size_t)(nlist + 1) * sizeof(int32_t)) ||
        !d_ids.alloc((size_t)std::max<size_t>(ids.size(), 1) * sizeof(int32_t))) {
        err = "显存分配失败(update)";
        return false;
    }
    if (cudaMemcpy(d_start.p, starts.data(), (size_t)(nlist + 1) * sizeof(int32_t),
                   cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(d_ids.p, ids.data(), (size_t)ids.size() * sizeof(int32_t),
                   cudaMemcpyHostToDevice) != cudaSuccess) {
        err = "上传倒排表失败";
        return false;
    }
    size_t smem = (size_t)dim * sizeof(float);
    k_update_centroids<<<nlist, kThreads, smem>>>(
        (const float*)d_data, dim, (const int32_t*)d_start.p, (const int32_t*)d_ids.p, d_centroids);
    if (cudaGetLastError() != cudaSuccess) {
        err = std::string("k_update_centroids 失败: ") + cudaGetErrorString(cudaGetLastError());
        return false;
    }
    return true;
}

template <int K>
void launch_ivf(const float* d_base, int64_t n, int32_t dim, const float* d_q, int64_t nq,
                int64_t q_offset, const float* d_cent, int nlist, const int32_t* d_start,
                const int32_t* d_ids, int nprobe, int smaller, float* d_outd, int32_t* d_outi,
                int32_t* d_scan, cudaStream_t stream) {
    constexpr int NWARP = kThreads / 32;
    const size_t smem = (size_t)nlist * sizeof(float) + (size_t)dim * sizeof(float) +
                        (size_t)nprobe * sizeof(int32_t) +
                        (size_t)NWARP * K * (sizeof(float) + sizeof(int32_t));
    k_ivf_query<K, kThreads><<<(unsigned)nq, kThreads, smem, stream>>>(
        d_base, n, dim, d_q, q_offset, d_cent, nlist, d_start, d_ids, nprobe, smaller, d_outd,
        d_outi, d_scan);
}

bool launch_ivf_dispatch(int K, const float* d_base, int64_t n, int32_t dim, const float* d_q,
                         int64_t nq, int64_t q_offset, const float* d_cent, int nlist,
                         const int32_t* d_start, const int32_t* d_ids, int nprobe, int smaller,
                         float* d_outd, int32_t* d_outi, int32_t* d_scan, cudaStream_t stream,
                         std::string& err) {
    switch (K) {
    case 1:
        launch_ivf<1>(d_base, n, dim, d_q, nq, q_offset, d_cent, nlist, d_start, d_ids, nprobe,
                      smaller, d_outd, d_outi, d_scan, stream);
        break;
    case 10:
        launch_ivf<10>(d_base, n, dim, d_q, nq, q_offset, d_cent, nlist, d_start, d_ids, nprobe,
                       smaller, d_outd, d_outi, d_scan, stream);
        break;
    case 50:
        launch_ivf<50>(d_base, n, dim, d_q, nq, q_offset, d_cent, nlist, d_start, d_ids, nprobe,
                       smaller, d_outd, d_outi, d_scan, stream);
        break;
    case 100:
        launch_ivf<100>(d_base, n, dim, d_q, nq, q_offset, d_cent, nlist, d_start, d_ids, nprobe,
                        smaller, d_outd, d_outi, d_scan, stream);
        break;
    default:
        err = "IVF 查询目前仅支持 top_k ∈ {1, 10, 50, 100}，当前为 " + std::to_string(K);
        return false;
    }
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        err = std::string("IVF kernel 启动失败: ") + cudaGetErrorString(e);
        return false;
    }
    return true;
}

} // namespace

// ======================= 建索引 =======================
bool gpu_build_ivf(const VectorSet& base, const Config& cfg, IvfIndex& idx, PerfStats& perf,
                   std::string& err) {
    if (base.n <= 0) {
        err = "空的向量库";
        return false;
    }
    const int64_t n = base.n;
    const int32_t dim = base.dim;
    int nlist = cfg.nlist;
    if (nlist < 1)
        nlist = 1;
    if (nlist > (int)n)
        nlist = (int)n;

    // 共享内存上限检查（nlist 个 float）
    if ((size_t)nlist * 4 > 48 * 1024) {
        err = "nlist 过大：查询 kernel 需要 nlist 个 float 的共享内存（上限约 12288）";
        return false;
    }

    // ---- 子样本训练集（避免在百万级数据上反复全量分配）----
    const int64_t want = std::max<int64_t>((int64_t)nlist * 20, 10000);
    const int64_t sample = std::min(n, want);
    const int64_t sstride = std::max<int64_t>(1, n / sample);
    std::vector<float> sub((size_t)(sample * dim));
    for (int64_t i = 0; i < sample; i++) {
        int64_t src = std::min<int64_t>(i * sstride, n - 1);
        std::copy(base.data.begin() + (long)(src * dim),
                  base.data.begin() + (long)(src * dim + dim), sub.begin() + (long)(i * dim));
    }

    DevBuf d_sub, d_cent, d_labels;
    if (!d_sub.alloc((size_t)(sample * dim) * sizeof(float)) ||
        !d_cent.alloc((size_t)((int64_t)nlist * dim) * sizeof(float)) ||
        !d_labels.alloc((size_t)sample * sizeof(int32_t))) {
        err = "显存分配失败(build)";
        return false;
    }
    if (cudaMemcpy(d_sub.p, sub.data(), (size_t)(sample * dim) * sizeof(float),
                   cudaMemcpyHostToDevice) != cudaSuccess) {
        err = "上传子样本失败";
        return false;
    }

    // 初始化聚类中心：从子样本随机抽样
    {
        std::mt19937 rng(cfg.seed);
        std::uniform_int_distribution<int64_t> pick(0, sample - 1);
        std::vector<float> init((size_t)((int64_t)nlist * dim));
        for (int c = 0; c < nlist; c++) {
            int64_t s = pick(rng);
            std::copy(sub.begin() + (long)(s * dim), sub.begin() + (long)(s * dim + dim),
                      init.begin() + (long)((int64_t)c * dim));
        }
        if (cudaMemcpy(d_cent.p, init.data(), (size_t)((int64_t)nlist * dim) * sizeof(float),
                       cudaMemcpyHostToDevice) != cudaSuccess) {
            err = "上传初始中心失败";
            return false;
        }
    }

    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    cudaEventRecord(e0, 0);

    // ---- k-means 迭代 ----
    std::vector<int32_t> starts, ids;
    for (int it = 0; it < cfg.kmeans_iters; it++) {
        if (!assign_and_build_lists((const float*)d_sub.p, sample, dim, (const float*)d_cent.p,
                                    nlist, (int32_t*)d_labels.p, &starts, &ids, err))
            return false;
        if (!update_centroids((const float*)d_sub.p, dim, starts, ids, (float*)d_cent.p, nlist,
                              err))
            return false;
    }

    // ---- 对全量数据做最终分配，得到真正的倒排表 ----
    DevBuf d_base, d_labels_full;
    if (!d_base.alloc((size_t)(n * dim) * sizeof(float)) ||
        !d_labels_full.alloc((size_t)n * sizeof(int32_t))) {
        err = "显存分配失败(full)";
        return false;
    }
    if (cudaMemcpy(d_base.p, base.data.data(), (size_t)(n * dim) * sizeof(float),
                   cudaMemcpyHostToDevice) != cudaSuccess) {
        err = "上传全量向量失败";
        return false;
    }
    if (!assign_and_build_lists((const float*)d_base.p, n, dim, (const float*)d_cent.p, nlist,
                                (int32_t*)d_labels_full.p, &starts, &ids, err))
        return false;

    cudaEventRecord(e1, 0);
    if (cudaEventSynchronize(e1) != cudaSuccess) {
        err = "建索引同步失败";
        return false;
    }
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, e0, e1);
    perf.build_ms = (double)ms;

    // ---- 组装索引 ----
    idx.nlist = nlist;
    idx.dim = dim;
    idx.metric = base.metric;
    idx.pq_m = 0; // IVF-Flat
    idx.centroids.assign((size_t)((int64_t)nlist * dim), 0.0f);
    if (cudaMemcpy(idx.centroids.data(), d_cent.p, (size_t)((int64_t)nlist * dim) * sizeof(float),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
        err = "回读聚类中心失败";
        return false;
    }
    idx.list_start = starts;
    idx.list_ids = ids;

    perf.gpu_mem_mb = (double)((size_t)(sample * dim) * 4 + (size_t)((int64_t)nlist * dim) * 4 +
                               (size_t)n * 4 + (size_t)n * 4) /
                      (1024.0 * 1024.0);
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    return true;
}

// ======================= IVF 查询 =======================
bool gpu_ivf_search(const VectorSet& base, const VectorSet& queries, const IvfIndex& idx,
                    const Config& cfg, SearchResult& out, PerfStats& perf, std::string& err) {
    if (idx.empty()) {
        err = "索引为空";
        return false;
    }
    if (idx.dim != base.dim || idx.dim != queries.dim) {
        err = "索引维度与向量/查询不一致";
        return false;
    }
    if (idx.metric != base.metric) {
        err = "索引的度量与向量库不一致";
        return false;
    }
    const int K = cfg.top_k;
    const int64_t n = base.n;
    const int64_t nq = queries.n;
    const int32_t dim = base.dim;
    const int nlist = idx.nlist;
    const bool cosine = (base.metric == Metric::COSINE);
    const int smaller = (base.metric == Metric::L2) ? 1 : 0;

    DevBuf d_base, d_q, d_cent, d_start, d_ids, d_outd, d_outi, d_scan;
    if (!d_base.alloc((size_t)(n * dim) * sizeof(float)) ||
        !d_q.alloc((size_t)(nq * dim) * sizeof(float)) ||
        !d_cent.alloc((size_t)((int64_t)nlist * dim) * sizeof(float)) ||
        !d_start.alloc((size_t)(nlist + 1) * sizeof(int32_t)) ||
        !d_ids.alloc((size_t)std::max<size_t>(idx.list_ids.size(), 1) * sizeof(int32_t)) ||
        !d_outd.alloc((size_t)(nq * K) * sizeof(float)) ||
        !d_outi.alloc((size_t)(nq * K) * sizeof(int32_t)) ||
        !d_scan.alloc((size_t)nq * sizeof(int32_t))) {
        err = "显存分配失败(search)";
        return false;
    }
    if (cudaMemcpy(d_base.p, base.data.data(), (size_t)(n * dim) * sizeof(float),
                   cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(d_q.p, queries.data.data(), (size_t)(nq * dim) * sizeof(float),
                   cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(d_cent.p, idx.centroids.data(), (size_t)((int64_t)nlist * dim) * sizeof(float),
                   cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(d_start.p, idx.list_start.data(), (size_t)(nlist + 1) * sizeof(int32_t),
                   cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(d_ids.p, idx.list_ids.data(), idx.list_ids.size() * sizeof(int32_t),
                   cudaMemcpyHostToDevice) != cudaSuccess) {
        err = "H2D 拷贝失败(search)";
        return false;
    }

    cudaStream_t stream = nullptr;
    if (cosine) {
        int blocks = (int)((n + kThreads - 1) / kThreads);
        k_normalize<<<blocks, kThreads, 0, stream>>>((float*)d_base.p, n, dim);
        int qblocks = (int)((nq + kThreads - 1) / kThreads);
        k_normalize<<<qblocks, kThreads, 0, stream>>>((float*)d_q.p, nq, dim);
        if (cudaGetLastError() != cudaSuccess) {
            err = "cosine 归一化失败";
            return false;
        }
    }

    out.nq = nq;
    out.k = K;
    out.ids.assign((size_t)(nq * K), -1);
    out.dists.assign((size_t)(nq * K), 0.0f);

    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    const int64_t batch = std::max<int64_t>(1, cfg.batch_size);
    float total_ms = 0.0f;
    cudaEventRecord(e0, stream);
    for (int64_t off = 0; off < nq; off += batch) {
        const int64_t cnt = std::min(batch, nq - off);
        if (!launch_ivf_dispatch(K, (const float*)d_base.p, n, dim, (const float*)d_q.p, cnt, off,
                                 (const float*)d_cent.p, nlist, (const int32_t*)d_start.p,
                                 (const int32_t*)d_ids.p, cfg.nprobe, smaller, (float*)d_outd.p,
                                 (int32_t*)d_outi.p, (int32_t*)d_scan.p, stream, err))
            return false;
    }
    cudaEventRecord(e1, stream);
    if (cudaEventSynchronize(e1) != cudaSuccess) {
        err = std::string("IVF kernel 执行失败: ") + cudaGetErrorString(cudaGetLastError());
        return false;
    }
    cudaEventElapsedTime(&total_ms, e0, e1);
    perf.search_ms = (double)total_ms;
    perf.qps = (total_ms > 0.0f) ? (double)nq / ((double)total_ms * 1e-3) : 0.0;

    // 逐 query 延迟
    out.latency_ms.assign((size_t)nq, 0.0);
    for (int64_t i = 0; i < nq; i++) {
        cudaEventRecord(e0, stream);
        if (!launch_ivf_dispatch(K, (const float*)d_base.p, n, dim, (const float*)d_q.p, 1, i,
                                 (const float*)d_cent.p, nlist, (const int32_t*)d_start.p,
                                 (const int32_t*)d_ids.p, cfg.nprobe, smaller, (float*)d_outd.p,
                                 (int32_t*)d_outi.p, (int32_t*)d_scan.p, stream, err))
            return false;
        cudaEventRecord(e1, stream);
        if (cudaEventSynchronize(e1) != cudaSuccess) {
            err = "逐 query 计时失败";
            return false;
        }
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, e0, e1);
        out.latency_ms[(size_t)i] = (double)ms;
    }

    if (cudaMemcpy(out.dists.data(), d_outd.p, (size_t)(nq * K) * sizeof(float),
                   cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(out.ids.data(), d_outi.p, (size_t)(nq * K) * sizeof(int32_t),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
        err = "D2H 拷贝失败(search)";
        return false;
    }

    // 实际扫描比例
    std::vector<int32_t> scan((size_t)nq);
    if (cudaMemcpy(scan.data(), d_scan.p, (size_t)nq * sizeof(int32_t), cudaMemcpyDeviceToHost) ==
        cudaSuccess) {
        double sum = 0;
        for (int64_t i = 0; i < nq; i++)
            sum += scan[(size_t)i];
        perf.scanned_ratio = (n > 0 && nq > 0) ? (sum / (double)nq) / (double)n : 0.0;
    }

    perf.gpu_mem_mb =
        (double)((size_t)(n * dim) * 4 + (size_t)(nq * dim) * 4 +
                 (size_t)((int64_t)nlist * dim) * 4 + (size_t)n * 4 + (size_t)(nq * K) * 8) /
        (1024.0 * 1024.0);
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    return true;
}

} // namespace vs
