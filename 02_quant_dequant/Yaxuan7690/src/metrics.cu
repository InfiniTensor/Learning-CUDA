// =============================================================================
//  metrics.cu -- 反量化结果的误差统计(GPU 归约)
//
//  统计量:最大绝对误差 / 平均绝对误差(MAE) / 均方误差(MSE) / RMSE / 相对 MSE /
//          带符号平均误差(bias)
//  误差基准 = 原始输入张量(fp32 或 fp16),比较对象 = 反量化结果.
//
//  RNE always rounds to one side of a non-representable input, so its bias is non-zero;
//  the bias of SR is zero in expectation.
//
//  ---------------------------------------------------------------------------
//  这个 kernel 曾经是整条链路里最慢的一环(占 GPU 时间 55%~70%, 有效带宽只有
//  ~123 GB/s). 瓶颈不在访存而在 ALU: 消费级 Blackwell 的 fp64 吞吐只有 fp32 的
//  1/64, 而"逐元素转 fp64 再累加"相当于每个元素要付约 300 条 fp32 指令的代价.
//  因此保留两种累加策略, 由 AccumMode 选择:
//
//    Precise -- 每个元素转 fp64 再累加, 与 CPU 双精度参考实现逐位可比(参考路径);
//    Fast    -- 线程内用 (hi, lo) 补偿累加主项 + 一个 fp32 寄存器攒修正项,
//               线程间仍在 fp64 归约. 每元素 ~50 条 fp32 指令(约合 0.8 条
//               fp64 指令), 把 kernel 从 fp64 ALU 瓶颈拉回访存瓶颈;
//               相对误差实测 ~1e-15, 远小于测试判据 1e-12(见 docs/measurements.md).
//
//  两种策略都按「每线程一次处理 4 个连续元素」组织: fp32 走 float4(16 B),
//  fp16/bf16 走 2 x 16bit 向量(8 B), 取代原先每线程 1 个元素的标量装载.
// =============================================================================
#include <algorithm>

#include "lpq/common.hpp"
#include "lpq/pipeline.hpp"
#include "lpq/stats.cuh"
#include "lpq/vec_io.cuh"

namespace lpq {
namespace {

constexpr int kThreads = 256;

// two_sum_accum / AccumPrecise / AccumFast 已移入 lpq/stats.cuh:
// 量化 kernel 里融合的误差统计与这里的独立 kernel 用的是同一份累加代码,
// 两条路径才能给出逐位相同的结果.

// -----------------------------------------------------------------------------
//  同一棵归约树服务两种累加策略: 线程内累加 -> 共享内存树形归约(fp64)
//  -> 每个 block 一次 atomicAdd(double). 线程内用什么精度由 Accum 决定.
// -----------------------------------------------------------------------------
template <class Accum, class RefT, class TestT>
__global__ void k_error_stats(const RefT* __restrict__ ref, const TestT* __restrict__ test,
    int64_t numel, double* __restrict__ acc, float* __restrict__ mx) {
    __shared__ double s_sum[4 * kThreads];
    __shared__ float s_max[kThreads];
    const int tid = threadIdx.x;

    // 两组独立的累加器: 补偿求和在 (hi, lo) 上是一条串行依赖链,
    // 一组足以把 FADD 的延迟暴露出来(SASS 里有大量 NOP 就是
    // 调度器在等依赖). 把 8 个元素拆成两半交给两组累加器,
    // ILP 翻倍; 最后在 fp64 里合并, 对结果的影响在 1e-16 量级.
    Accum A0;
    Accum A1;
    const int64_t nvec = numel >> 3;
    const int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for (int64_t v = (int64_t)blockIdx.x * blockDim.x + tid; v < nvec; v += stride) {
        float rv[8];
        float tv[8];
        InTraits<RefT>::load4(ref + (v << 3), rv);
        InTraits<RefT>::load4(ref + (v << 3) + 4, rv + 4);
        InTraits<TestT>::load4(test + (v << 3), tv);
        InTraits<TestT>::load4(test + (v << 3) + 4, tv + 4);
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            A0.add(tv[k], rv[k]);
            A1.add(tv[k + 4], rv[k + 4]);
        }
    }
    // 尾部不足 8 个元素的部分(最多 7 个)由 0 号 block 的 0 号线程补上
    if (tid == 0 && blockIdx.x == 0) {
        for (int64_t i = nvec << 3; i < numel; ++i)
            A0.add(InTraits<TestT>::to_float(test[i]), InTraits<RefT>::to_float(ref[i]));
    }
#pragma unroll
    for (int j = 0; j < 4; ++j)
        s_sum[tid * 4 + j] = A0.value(j) + A1.value(j);
    s_max[tid] = fmaxf(A0.m, A1.m);
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            s_sum[tid * 4 + 0] += s_sum[(tid + s) * 4 + 0];
            s_sum[tid * 4 + 1] += s_sum[(tid + s) * 4 + 1];
            s_sum[tid * 4 + 2] += s_sum[(tid + s) * 4 + 2];
            s_sum[tid * 4 + 3] += s_sum[(tid + s) * 4 + 3];
            s_max[tid] = fmaxf(s_max[tid], s_max[tid + s]);
        }
        __syncthreads();
    }
    if (tid == 0) {
        lpq_atomic_add_f64(&acc[0], s_sum[0]);
        lpq_atomic_add_f64(&acc[1], s_sum[1]);
        lpq_atomic_add_f64(&acc[2], s_sum[2]);
        lpq_atomic_add_f64(&acc[3], s_sum[3]);
        lpq_atomic_max_f32(mx, s_max[0]);
    }
}

template <class RefT, class TestT>
void launch_stats(const void* ref, const void* test, int64_t numel, double* d_acc, float* d_max,
    AccumMode mode, cudaStream_t stream) {
    const int64_t nvec = numel >> 2;
    const int64_t want = (nvec + kThreads - 1) / kThreads;
    const int64_t blocks = std::max<int64_t>(1, std::min<int64_t>(want, 2048));
    const RefT* rp = static_cast<const RefT*>(ref);
    const TestT* tp = static_cast<const TestT*>(test);
    if (mode == AccumMode::Fast) {
        k_error_stats<AccumFast>
            <<<(unsigned int)blocks, kThreads, 0, stream>>>(rp, tp, numel, d_acc, d_max);
    } else {
        k_error_stats<AccumPrecise>
            <<<(unsigned int)blocks, kThreads, 0, stream>>>(rp, tp, numel, d_acc, d_max);
    }
    LPQ_CUDA_CHECK_KERNEL();
}

template <class RefT>
void dispatch_test_type(const void* ref, const void* test, OutputType tt, int64_t numel,
    double* d_acc, float* d_max, AccumMode mode, cudaStream_t stream) {
    switch (tt) {
        case OutputType::FP16:
            launch_stats<RefT, __half>(ref, test, numel, d_acc, d_max, mode, stream);
            break;
        case OutputType::BF16:
            launch_stats<RefT, __nv_bfloat16>(ref, test, numel, d_acc, d_max, mode, stream);
            break;
        default:
            launch_stats<RefT, float>(ref, test, numel, d_acc, d_max, mode, stream);
            break;
    }
}

// 统计量只有 4 个 double + 1 个 float, 但原来每次调用都要 cudaMalloc/cudaFree.
// 在 8192x8192 这种规模下 kernel 本身只有 1 ms, 而两次分配加两次
// 释放就要 ~40 us(约 4%). 改成进程内缓存一小块显存。
struct StatsScratch {
    double* acc = nullptr;
    float* mx = nullptr;
    ~StatsScratch() {
        if (acc)
            cudaFree(acc);
        if (mx)
            cudaFree(mx);
    }
};

StatsScratch& stats_scratch() {
    static StatsScratch s;
    if (s.acc == nullptr) {
        LPQ_CUDA_CHECK(cudaMalloc(&s.acc, 4 * sizeof(double)));
        LPQ_CUDA_CHECK(cudaMalloc(&s.mx, sizeof(float)));
    }
    return s;
}

}  // namespace

StatsReduction stats_reduction_scratch() {
    const StatsScratch& s = stats_scratch();
    return StatsReduction{s.acc, s.mx};
}

ErrorStats finalize_error_stats(const double acc[4], float max_abs, int64_t numel) {
    ErrorStats st;
    st.numel = numel;
    const double n = (double)numel;
    if (n <= 0.0)
        return st;
    st.mae = acc[0] / n;
    st.mse = acc[1] / n;
    st.rmse = std::sqrt(st.mse);
    st.max_abs = (double)max_abs;
    const double ref_ms = acc[2] / n;
    st.ref_amax = std::sqrt(acc[2]);  // 这里给出的是 ref 的 L2 范数,供参考
    st.rel_mse = (ref_ms > 0.0) ? (st.mse / ref_ms) : 0.0;
    st.mean_signed_error = acc[3] / n;
    return st;
}

ErrorStats compute_error_stats(const void* d_ref, InputType ref_type, const void* d_test,
    OutputType test_type, int64_t numel, cudaStream_t stream, AccumMode mode) {
    ErrorStats st;
    st.numel = numel;
    if (numel <= 0)
        return st;

    StatsScratch& sc = stats_scratch();
    double* d_acc = sc.acc;
    float* d_max = sc.mx;
    LPQ_CUDA_CHECK(cudaMemsetAsync(d_acc, 0, 4 * sizeof(double), stream));
    LPQ_CUDA_CHECK(cudaMemsetAsync(d_max, 0, sizeof(float), stream));

    if (ref_type == InputType::FP32) {
        dispatch_test_type<float>(d_ref, d_test, test_type, numel, d_acc, d_max, mode, stream);
    } else {
        dispatch_test_type<__half>(d_ref, d_test, test_type, numel, d_acc, d_max, mode, stream);
    }

    double h_acc[4] = {0.0, 0.0, 0.0, 0.0};
    float h_max = 0.0f;
    LPQ_CUDA_CHECK(cudaMemcpyAsync(h_acc, d_acc, sizeof(h_acc), cudaMemcpyDeviceToHost, stream));
    LPQ_CUDA_CHECK(cudaMemcpyAsync(&h_max, d_max, sizeof(h_max), cudaMemcpyDeviceToHost, stream));
    LPQ_CUDA_CHECK(cudaStreamSynchronize(stream));

    return finalize_error_stats(h_acc, h_max, numel);
}

}  // namespace lpq