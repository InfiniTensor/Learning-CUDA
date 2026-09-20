// exact.cu - GPU 精确检索（暴力扫描 + Top-K）
//
// 设计：
//   * 一个线程块处理一个 query（网格 = query 数），块内线程按步长扫描全部向量
//   * 每线程在本地维护一个有序 top-K（早退：先与本地第 K 名比较，仅更优时才插入）
//   * 两级归并：warp 内 32 路归并（K 轮 32 路最小值，用 shuffle 完成）
//              → 跨 warp 归并（warp0 的 NWARP 个 lane）
//   * 归并利用“每个列表本身已有序”这一性质，避免整体排序，代价仅 O(K·log)
//   * 比较采用与 CPU 参考一致的“全序”：分数更优者胜；分数相等时 id 小者胜
//
// 说明：精确检索在数据规模较大时是**显存带宽受限**的（每个 query 都要读一遍
// 全量库）。真正的吞吐提升来自 IVF 近似检索，见 ivf.cu。
#include "search.h"
#include "topk_merge.cuh"
#include "normalize.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cfloat>
#include <climits>
#include <cstdio>
#include <string>
#include <vector>

namespace vs {
namespace {

// 精确检索主 kernel：网格 = 本批 query 数
template <int K, int THREADS>
__global__ void k_exact_topk(const float* __restrict__ base, int64_t n, int32_t dim,
                             const float* __restrict__ queries, int64_t q_offset, int smaller,
                             float* __restrict__ out_d, int32_t* __restrict__ out_i) {
    constexpr int NWARP = THREADS / 32;
    extern __shared__ char smem[];
    float* sq = reinterpret_cast<float*>(smem);                   // dim
    float* s_wd = sq + dim;                                       // NWARP*K
    int32_t* s_wi = reinterpret_cast<int32_t*>(s_wd + NWARP * K); // NWARP*K

    const int tid = threadIdx.x;
    const int64_t qi = q_offset + blockIdx.x;
    const bool smallerb = (smaller != 0);
    const float WORST = smallerb ? FLT_MAX : -FLT_MAX;

    // query 载入共享内存（块内所有线程复用，避免重复访存）
    for (int32_t d = tid; d < dim; d += THREADS)
        sq[d] = queries[qi * dim + d];
    __syncthreads();

    // 每线程局部 top-K（保持有序：优在前）
    float bd[K];
    int32_t bi[K];
#pragma unroll
    for (int i = 0; i < K; i++) {
        bd[i] = WORST;
        bi[i] = INT_MAX; // 哨兵 id：分数相同时真实候选优先
    }

    const bool vec4 = ((dim & 3) == 0);
    for (int64_t v = tid; v < n; v += THREADS) {
        const float* b = base + v * dim;
        float acc = 0.0f;
        int32_t d = 0;
        if (vec4) {
            for (; d + 4 <= dim; d += 4) {
                const float4 bv = *reinterpret_cast<const float4*>(b + d);
                if (smallerb) {
                    float t0 = sq[d] - bv.x, t1 = sq[d + 1] - bv.y;
                    float t2 = sq[d + 2] - bv.z, t3 = sq[d + 3] - bv.w;
                    // 与 CPU 参考相同的累加顺序，避免浮点重排带来的顺序差异
                    acc += t0 * t0;
                    acc += t1 * t1;
                    acc += t2 * t2;
                    acc += t3 * t3;
                } else {
                    acc += sq[d] * bv.x;
                    acc += sq[d + 1] * bv.y;
                    acc += sq[d + 2] * bv.z;
                    acc += sq[d + 3] * bv.w;
                }
            }
        }
        for (; d < dim; d++) {
            float x = sq[d];
            if (smallerb) {
                float t = x - b[d];
                acc += t * t;
            } else {
                acc += x * b[d];
            }
        }
        // 早退：仅当优于本地第 K 名时才插入
        if (is_better_d(acc, (int32_t)v, bd[K - 1], bi[K - 1], smallerb)) {
            int pos = K - 1;
            while (pos > 0 && is_better_d(acc, (int32_t)v, bd[pos - 1], bi[pos - 1], smallerb)) {
                bd[pos] = bd[pos - 1];
                bi[pos] = bi[pos - 1];
                pos--;
            }
            bd[pos] = acc;
            bi[pos] = (int32_t)v;
        }
    }

    // 两级归并：warp 内 32 路 → 跨 warp（公共实现见 topk_merge.cuh）
    block_topk_merge<K, THREADS>(bd, bi, smallerb, s_wd, s_wi, out_d, out_i, qi * K);
}

constexpr int kThreads = 256;

template <int K>
static void launch_exact(const float* d_base, int64_t n, int32_t dim, const float* d_q, int64_t nq,
                         int64_t q_offset, int smaller, float* d_outd, int32_t* d_outi,
                         cudaStream_t stream) {
    constexpr int NWARP = kThreads / 32;
    const size_t smem =
        (size_t)dim * sizeof(float) + (size_t)NWARP * K * (sizeof(float) + sizeof(int32_t));
    k_exact_topk<K, kThreads><<<(unsigned)nq, kThreads, smem, stream>>>(
        d_base, n, dim, d_q, q_offset, smaller, d_outd, d_outi);
}

bool launch_exact_dispatch(int K, const float* d_base, int64_t n, int32_t dim, const float* d_q,
                           int64_t nq, int64_t q_offset, int smaller, float* d_outd,
                           int32_t* d_outi, cudaStream_t stream, std::string& err) {
    switch (K) {
    case 1:
        launch_exact<1>(d_base, n, dim, d_q, nq, q_offset, smaller, d_outd, d_outi, stream);
        break;
    case 10:
        launch_exact<10>(d_base, n, dim, d_q, nq, q_offset, smaller, d_outd, d_outi, stream);
        break;
    case 50:
        launch_exact<50>(d_base, n, dim, d_q, nq, q_offset, smaller, d_outd, d_outi, stream);
        break;
    case 100:
        launch_exact<100>(d_base, n, dim, d_q, nq, q_offset, smaller, d_outd, d_outi, stream);
        break;
    default:
        err = "精确检索目前仅支持 top_k ∈ {1, 10, 50, 100}，当前为 " + std::to_string(K);
        return false;
    }
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        err = std::string("精确检索 kernel 启动失败: ") + cudaGetErrorString(e);
        return false;
    }
    return true;
}

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

} // namespace

bool gpu_exact_search(const VectorSet& base, const VectorSet& queries, const Config& cfg,
                      SearchResult& out, PerfStats& perf, double* cpu_ms_for_speedup,
                      std::string& err) {
    if (base.n <= 0 || queries.n <= 0) {
        err = "空的向量库或查询集";
        return false;
    }
    if (base.dim != queries.dim) {
        err = "向量库与查询的维度不一致";
        return false;
    }
    const int K = cfg.top_k;
    const int64_t n = base.n;
    const int64_t nq = queries.n;
    const int32_t dim = base.dim;

    const bool cosine = (base.metric == Metric::COSINE);
    const int smaller = (base.metric == Metric::L2) ? 1 : 0;

    DevBuf d_base, d_q, d_outd, d_outi;
    if (!d_base.alloc((size_t)n * dim * sizeof(float)) ||
        !d_q.alloc((size_t)nq * dim * sizeof(float)) ||
        !d_outd.alloc((size_t)nq * K * sizeof(float)) ||
        !d_outi.alloc((size_t)nq * K * sizeof(int32_t))) {
        err = "显存分配失败";
        return false;
    }
    if (cudaMemcpy(d_base.p, base.data.data(), (size_t)n * dim * sizeof(float),
                   cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(d_q.p, queries.data.data(), (size_t)nq * dim * sizeof(float),
                   cudaMemcpyHostToDevice) != cudaSuccess) {
        err = "H2D 拷贝失败";
        return false;
    }

    cudaStream_t stream = nullptr;
    if (cosine) {
        int threads = 256;
        int blocks = (int)((n + threads - 1) / threads);
        k_normalize<<<blocks, threads, 0, stream>>>((float*)d_base.p, n, dim);
        k_normalize<<<blocks, threads, 0, stream>>>((float*)d_q.p, nq, dim);
        if (cudaGetLastError() != cudaSuccess) {
            err = "cosine 归一化 kernel 失败";
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

    // ---- 分批吞吐测量 ----
    const int64_t batch = std::max<int64_t>(1, cfg.batch_size);
    float total_ms = 0.0f;
    if (cudaEventRecord(e0, stream) != cudaSuccess) {
        err = "event 记录失败";
        return false;
    }
    for (int64_t off = 0; off < nq; off += batch) {
        const int64_t cnt = std::min(batch, nq - off);
        if (!launch_exact_dispatch(K, (const float*)d_base.p, n, dim, (const float*)d_q.p, cnt, off,
                                   smaller, (float*)d_outd.p, (int32_t*)d_outi.p, stream, err))
            return false;
    }
    cudaEventRecord(e1, stream);
    if (cudaEventSynchronize(e1) != cudaSuccess) {
        err = std::string("kernel 执行失败: ") + cudaGetErrorString(cudaGetLastError());
        return false;
    }
    cudaEventElapsedTime(&total_ms, e0, e1);
    perf.search_ms = (double)total_ms;
    perf.qps = (total_ms > 0.0f) ? (double)nq / ((double)total_ms * 1e-3) : 0.0;

    // ---- 逐 query 延迟测量（用于 P50/P99）----
    out.latency_ms.assign((size_t)nq, 0.0);
    for (int64_t i = 0; i < nq; i++) {
        cudaEventRecord(e0, stream);
        if (!launch_exact_dispatch(K, (const float*)d_base.p, n, dim, (const float*)d_q.p, 1, i,
                                   smaller, (float*)d_outd.p, (int32_t*)d_outi.p, stream, err))
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

    // ---- 回读结果 ----
    if (cudaMemcpy(out.dists.data(), d_outd.p, (size_t)nq * K * sizeof(float),
                   cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(out.ids.data(), d_outi.p, (size_t)nq * K * sizeof(int32_t),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
        err = "D2H 拷贝失败";
        return false;
    }

    perf.gpu_mem_mb = (double)((size_t)n * dim * 4 + (size_t)nq * dim * 4 + (size_t)nq * K * 8) /
                      (1024.0 * 1024.0);
    perf.scanned_ratio = 1.0;
    if (cpu_ms_for_speedup && *cpu_ms_for_speedup > 0.0)
        perf.speedup = *cpu_ms_for_speedup / perf.search_ms;

    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    return true;
}

} // namespace vs
