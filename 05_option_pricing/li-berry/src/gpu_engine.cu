// gpu_engine.cu - CUDA 蒙特卡洛引擎
//
// 并行化设计（报告中详述）：
// - 粒度：一条路径（对偶时一对路径）一个线程单元，grid-stride 循环覆盖全部路径
// - 内存布局：融合生成+模拟+收益，每线程 O(1) 寄存器状态，无全局 Z 矩阵
//   （10M×256×4B = 10GB 超出 8GB 显存；融合设计全 GPU 内存占用 < 1MB）
// - 随机流：流位置仅由 (seed, path_index) 决定（curand 子序列 / splitmix64 播种），
//   与网格配置无关 → 重跑 bitwise 一致、N 较小运行是 N 较大运行的精确子集
// - 归约：两级确定性归约（warp shuffle → 块内 → finalize 单块核），无原子操作，
//   FP 加法非结合，原子序不确定会破坏可复现性
#include "gpu_engine.cuh"

#include <chrono>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "engine_common.h"
#include "rng.h"
#include "sim_core.h"

namespace {

struct CudaError : std::runtime_error {
    explicit CudaError(const std::string& msg) : std::runtime_error(msg) {}
};

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err_ = (call);                                           \
        if (err_ != cudaSuccess)                                             \
            throw CudaError(std::string("cuda error: ") + cudaGetErrorString(err_) + \
                            " at " + __FILE__ + ":" + std::to_string(__LINE__)); \
    } while (0)

// ---------- 块内归约（warp shuffle + 共享内存，支持任意 blockDim <= 1024）----------
// 累加顺序固定（lane 0..31 逐级 shfl_down），结果可复现

__device__ inline double warp_reduce_sum(double v) {
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
    return v;
}

// smem 需至少 nwarps 个 double（由调用方保证，nwarps <= 32 → 32 个）
__device__ inline double block_reduce_sum(double v, double* smem) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int nwarps = (blockDim.x + 31) >> 5;
    v = warp_reduce_sum(v);
    if (lane == 0) smem[warp] = v;
    __syncthreads();
    v = (threadIdx.x < nwarps) ? smem[threadIdx.x] : 0.0;
    if (warp == 0) v = warp_reduce_sum(v);
    return v;
}

// ---------- 主核：每条路径一个单元 ----------
// block_partials 布局：block_partials[3*blockIdx + {sum, sumsq, rawsq}]

template <OptionType OPT, bool ANTITHETIC>
__global__ void mc_kernel(const OptionParams p, const CVParams cv,
                          unsigned long long seed, int rng_kind,
                          long long units, double* __restrict__ block_partials) {
    const long long gid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long stride = (long long)gridDim.x * blockDim.x;

    double acc_sum = 0.0, acc_sumsq = 0.0, acc_rawsq = 0.0;
    for (long long u = gid; u < units; u += stride) {
        float y = 0.0f, c = 0.0f, yleg = 0.0f;
        if (rng_kind == static_cast<int>(RngType::Curand)) {
            if (ANTITHETIC) {
                sim_antithetic_pair<OPT, CurandGen4>(p, seed, static_cast<unsigned long long>(u),
                                                     y, c, yleg);
            } else {
                sim_one_path<OPT, CurandGen4>(p, seed, static_cast<unsigned long long>(u), y, c);
            }
        } else {
            if (ANTITHETIC) {
                sim_antithetic_pair<OPT, CustomGen4>(p, seed, static_cast<unsigned long long>(u),
                                                     y, c, yleg);
            } else {
                sim_one_path<OPT, CustomGen4>(p, seed, static_cast<unsigned long long>(u), y, c);
            }
        }
        // y_raw：未缩减样本（对偶 = 第一条腿收益；CV = 调整前收益），用于方差缩减因子
        const double y_raw = ANTITHETIC ? static_cast<double>(yleg) : static_cast<double>(y);
        double yd = static_cast<double>(y);
        if (cv.beta != 0.0) yd -= cv.beta * (static_cast<double>(c) - static_cast<double>(cv.c_mean));
        acc_sum += yd;
        acc_sumsq += yd * yd;
        acc_rawsq += y_raw * y_raw;
    }

    // 块内归约：3 个 double 各 32 个共享槽（nwarps <= 32）
    __shared__ double smem_sum[32], smem_sq[32], smem_raw[32];
    acc_sum = block_reduce_sum(acc_sum, smem_sum);
    acc_sumsq = block_reduce_sum(acc_sumsq, smem_sq);
    acc_rawsq = block_reduce_sum(acc_rawsq, smem_raw);
    if (threadIdx.x == 0) {
        block_partials[3 * blockIdx.x + 0] = acc_sum;
        block_partials[3 * blockIdx.x + 1] = acc_sumsq;
        block_partials[3 * blockIdx.x + 2] = acc_rawsq;
    }
}

// ---------- 二级归约：单块固定顺序求和（确定性）----------

__global__ void finalize_reduce(const double* __restrict__ partials, int n_blocks,
                                double* __restrict__ out /* [3] */) {
    double s0 = 0.0, s1 = 0.0, s2 = 0.0;
    for (int i = threadIdx.x; i < n_blocks; i += blockDim.x) {
        s0 += partials[3 * i + 0];
        s1 += partials[3 * i + 1];
        s2 += partials[3 * i + 2];
    }
    __shared__ double smem0[32], smem1[32], smem2[32];
    s0 = block_reduce_sum(s0, smem0);
    s1 = block_reduce_sum(s1, smem1);
    s2 = block_reduce_sum(s2, smem2);
    if (threadIdx.x == 0) {
        out[0] = s0;
        out[1] = s1;
        out[2] = s2;
    }
}

// ---------- 控制变量 pilot 核：估计 beta（5 个累加器，原子序无关紧要）----------

template <OptionType OPT>
__global__ void cv_pilot_kernel(const OptionParams p, unsigned long long seed, int rng_kind,
                                long long pilot_units, double* __restrict__ acc /* [5] */) {
    const long long gid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long stride = (long long)gridDim.x * blockDim.x;
    for (long long u = gid; u < pilot_units; u += stride) {
        float y = 0.0f, c = 0.0f;
        if (rng_kind == static_cast<int>(RngType::Curand)) {
            sim_one_path<OPT, CurandGen4>(p, seed, static_cast<unsigned long long>(u), y, c);
        } else {
            sim_one_path<OPT, CustomGen4>(p, seed, static_cast<unsigned long long>(u), y, c);
        }
        double yd = y, cd = c;
        atomicAdd(&acc[0], yd);
        atomicAdd(&acc[1], yd * yd);
        atomicAdd(&acc[2], cd);
        atomicAdd(&acc[3], cd * cd);
        atomicAdd(&acc[4], yd * cd);
    }
}

// ---------- RNG 流转储核（CPU/GPU 一致性验证用）----------

__global__ void dump_rng_kernel(unsigned long long seed, int rng_kind, int paths,
                                unsigned long long* __restrict__ out /* paths*16 */) {
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= paths) return;
    unsigned long long buf[16];
    if (rng_kind == static_cast<int>(RngType::Custom)) {
        XS128State st;
        xs128_seed(st, seed, static_cast<unsigned long long>(p));
        for (int i = 0; i < 16; ++i) buf[i] = xs128_next_u32(st);
    } else {
        curandStatePhilox4_32_10_t st;
        curand_init(seed, static_cast<unsigned long long>(p), 0, &st);
        for (int i = 0; i < 16; i += 4) {
            float4 u = curand_uniform4(&st);
            buf[i + 0] = __float_as_uint(u.x);
            buf[i + 1] = __float_as_uint(u.y);
            buf[i + 2] = __float_as_uint(u.z);
            buf[i + 3] = __float_as_uint(u.w);
        }
    }
    for (int i = 0; i < 16; ++i) out[static_cast<long long>(p) * 16 + i] = buf[i];
}

// ---------- 启动配置与分发 ----------

struct LaunchInfo {
    int block = 256;
    int grid = 1;
    int blocks_per_sm = 0;
};

template <OptionType OPT, bool ANTITHETIC>
LaunchInfo launch_info_for(long long units, const EngineOptions& opts, int sm_count) {
    LaunchInfo li;
    li.block = opts.block;
    if (opts.grid_blocks > 0) {
        li.grid = opts.grid_blocks;
    } else {
        int bps = 0;
        // 若请求的块大小放不进 SM（如 1024 线程 × 68 寄存器 > 64K/块），
        // 自动降级到能容纳的最大块大小
        while (li.block > 32) {
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bps, mc_kernel<OPT, ANTITHETIC>,
                                                          li.block, 0);
            if (bps > 0) break;
            li.block /= 2;
        }
        li.blocks_per_sm = bps;
        long long need = (units + li.block - 1) / li.block;
        int cap = sm_count * bps * 4; // 最多 4 波，兼顾尾部均衡与启动开销
        li.grid = static_cast<int>(need < cap ? need : cap);
        if (li.grid < 1) li.grid = 1;
    }
    return li;
}

// 分发主核（8 个 <OPT, ANTITHETIC> 实例化）
template <OptionType OPT>
void launch_mc(bool anti, const OptionParams& p, const CVParams& cv, const SimSpec& ss,
               long long units, const EngineOptions& opts, int sm_count,
               double* d_partials, double& kernel_ms, LaunchInfo& li_out) {
    LaunchInfo li = anti ? launch_info_for<OPT, true>(units, opts, sm_count)
                         : launch_info_for<OPT, false>(units, opts, sm_count);
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    cudaEventRecord(e0);
    if (anti) {
        mc_kernel<OPT, true><<<li.grid, li.block>>>(p, cv, ss.seed,
                                                    static_cast<int>(ss.rng), units, d_partials);
    } else {
        mc_kernel<OPT, false><<<li.grid, li.block>>>(p, cv, ss.seed,
                                                     static_cast<int>(ss.rng), units, d_partials);
    }
    cudaEventRecord(e1);
    cudaEventSynchronize(e1);
    CUDA_CHECK(cudaGetLastError());
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, e0, e1);
    kernel_ms = ms;
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    li_out = li;
}

// 分发 pilot 核
template <OptionType OPT>
void launch_pilot(const OptionParams& p, const SimSpec& ss, long long pilot_units,
                  double* d_acc) {
    int block = 256;
    int grid = (pilot_units + block - 1) / block;
    if (grid > 144) grid = 144;
    cv_pilot_kernel<OPT><<<grid, block>>>(p, ss.seed, static_cast<int>(ss.rng), pilot_units,
                                          d_acc);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

// 控制变量 pilot：清零累加器 → 运行 → 读回估计 beta
template <OptionType OPT>
double estimate_beta(const OptionParams& p, const SimSpec& ss, long long units,
                     double* d_acc, double* h_acc) {
    CUDA_CHECK(cudaMemset(d_acc, 0, 5 * sizeof(double)));
    long long pilot_units = units < 100000 ? units : 100000;
    launch_pilot<OPT>(p, ss, pilot_units, d_acc);
    CUDA_CHECK(cudaMemcpy(h_acc, d_acc, 5 * sizeof(double), cudaMemcpyDeviceToHost));
    double n = static_cast<double>(pilot_units);
    double var_c = (h_acc[3] - h_acc[2] * h_acc[2] / n) / (n - 1.0);
    if (var_c <= 0.0) return 0.0;
    return (h_acc[4] - h_acc[0] * h_acc[2] / n) / (n - 1.0) / var_c;
}

// 获取设备信息（填入 Timing）
void fill_device_info(Timing& t) {
    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    t.device_id = dev;
    std::snprintf(t.device_name, sizeof(t.device_name), "%s", prop.name);
    t.compute_major = prop.major;
    t.compute_minor = prop.minor;
    t.sm_count = prop.multiProcessorCount;
    CUDA_CHECK(cudaDriverGetVersion(&t.driver_version));
    CUDA_CHECK(cudaRuntimeGetVersion(&t.runtime_version));
}

// 引擎主体：按 <OPT, ANTITHETIC> 分发
template <OptionType OPT>
RawRun gpu_engine_impl(const OptionSpec& os, const SimSpec& ss, const EngineOptions& opts,
                       const OptionParams& params) {
    RawRun run;
    Timing& t = run.timing;
    fill_device_info(t);
    run.paths_run = ss.num_paths;
    const bool anti = (ss.vr == VarReduction::Antithetic);
    const long long units = anti ? ss.num_paths / 2 : ss.num_paths;

    const auto host_t0 = std::chrono::steady_clock::now();

    // 控制变量 pilot
    double beta = 0.0;
    CVParams cv;
    double *d_acc = nullptr, *h_acc = nullptr;
    if (ss.vr == VarReduction::ControlVariate) {
        CUDA_CHECK(cudaMalloc(&d_acc, 5 * sizeof(double)));
        h_acc = new double[5];
        beta = estimate_beta<OPT>(params, ss, units, d_acc, h_acc);
        cv.c_mean = static_cast<float>(control_mean(os, ss));
    }
    cv.beta = beta;

    // 主归约缓冲（先按与正式启动相同的参数计算网格）
    LaunchInfo li;
    if (anti) li = launch_info_for<OPT, true>(units, opts, t.sm_count);
    else li = launch_info_for<OPT, false>(units, opts, t.sm_count);

    double *d_partials = nullptr, *d_out = nullptr, *h_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_partials, static_cast<size_t>(li.grid) * 3 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_out, 3 * sizeof(double)));
    h_out = new double[3];

    const auto host_t1 = std::chrono::steady_clock::now();
    t.host_setup_ms = std::chrono::duration<double, std::milli>(host_t1 - host_t0).count();

    // 主核 + 二级归约 + 回传（事件计时）
    cudaEvent_t e_start, e_after_kernel, e_after_finalize, e_after_d2h;
    cudaEventCreate(&e_start);
    cudaEventCreate(&e_after_kernel);
    cudaEventCreate(&e_after_finalize);
    cudaEventCreate(&e_after_d2h);
    cudaEventRecord(e_start);

    launch_mc<OPT>(anti, params, cv, ss, units, opts, t.sm_count, d_partials, t.kernel_ms, li);
    cudaEventRecord(e_after_kernel);

    finalize_reduce<<<1, 256>>>(d_partials, li.grid, d_out);
    CUDA_CHECK(cudaGetLastError());
    cudaEventRecord(e_after_finalize);

    CUDA_CHECK(cudaMemcpy(h_out, d_out, 3 * sizeof(double), cudaMemcpyDeviceToHost));
    cudaEventRecord(e_after_d2h);
    cudaEventSynchronize(e_after_d2h);
    CUDA_CHECK(cudaGetLastError());

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, e_after_kernel, e_after_finalize);
    t.finalize_ms = ms;
    cudaEventElapsedTime(&ms, e_after_kernel, e_after_d2h);
    t.reduction_ms = ms;
    cudaEventElapsedTime(&ms, e_after_finalize, e_after_d2h);
    t.d2h_ms = ms;
    cudaEventElapsedTime(&ms, e_start, e_after_d2h);
    t.total_ms = ms;
    cudaEventDestroy(e_start);
    cudaEventDestroy(e_after_kernel);
    cudaEventDestroy(e_after_finalize);
    cudaEventDestroy(e_after_d2h);

    // 粒度信息
    cudaFuncAttributes attr;
    if (anti) {
        cudaFuncGetAttributes(&attr, mc_kernel<OPT, true>);
    } else {
        cudaFuncGetAttributes(&attr, mc_kernel<OPT, false>);
    }
    t.grid_blocks = li.grid;
    t.block_size = li.block;
    t.blocks_per_sm = li.blocks_per_sm;
    t.total_threads = li.grid * li.block;
    t.registers_per_thread = attr.numRegs;
    t.shared_bytes_per_block = static_cast<int>(attr.sharedSizeBytes);

    run.acc.sum = h_out[0];
    run.acc.sumsq = h_out[1];
    run.acc.rawsq = h_out[2];
    run.beta = beta;

    delete[] h_out;
    delete[] h_acc;
    cudaFree(d_partials);
    cudaFree(d_out);
    cudaFree(d_acc);
    return run;
}

} // namespace

RawRun run_gpu(const OptionSpec& os, const SimSpec& ss, const EngineOptions& opts,
               const OptionParams& params) {
    // anti 标志在 gpu_engine_impl 内部由 ss.vr 决定
    switch (os.type) {
        case OptionType::EuropeanCall: return gpu_engine_impl<OptionType::EuropeanCall>(os, ss, opts, params);
        case OptionType::EuropeanPut: return gpu_engine_impl<OptionType::EuropeanPut>(os, ss, opts, params);
        case OptionType::AsianCall: return gpu_engine_impl<OptionType::AsianCall>(os, ss, opts, params);
        case OptionType::BarrierCallUpOut: return gpu_engine_impl<OptionType::BarrierCallUpOut>(os, ss, opts, params);
    }
    throw CudaError("unknown option type");
}

void dump_gpu_rng_stream(const SimSpec& ss, int count) {
    const int paths = count / 16;
    std::vector<unsigned long long> buf(static_cast<size_t>(count));
    unsigned long long* d_buf = nullptr;
    CUDA_CHECK(cudaMalloc(&d_buf, static_cast<size_t>(count) * sizeof(unsigned long long)));
    const int block = 256;
    const int grid = (paths + block - 1) / block;
    dump_rng_kernel<<<grid, block>>>(ss.seed, static_cast<int>(ss.rng), paths, d_buf);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(buf.data(), d_buf, static_cast<size_t>(count) * sizeof(unsigned long long),
                          cudaMemcpyDeviceToHost));
    cudaFree(d_buf);
    for (int p = 0; p < paths; ++p) {
        std::printf("path %d", p);
        for (int i = 0; i < 16; ++i) std::printf(" %016llx", buf[static_cast<size_t>(p) * 16 + i]);
        std::printf("\n");
    }
}
