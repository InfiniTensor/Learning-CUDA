// fht.cu - 快速 Hadamard 变换（FHT）kernel 与 FHT+FP8 量化融合 kernel
//
// 算法设计（warp-per-row）：
//   * 一行 n 个元素由 32 个 lane 共同处理，lane l 持有元素 { l + 32k : k = 0..V-1 }，
//     V = n / 32（n=64 → V=2，n=128 → V=4，n=256 → V=8），全部驻留寄存器。
//   * 元素下标 i = l + 32k，因此 bit0..4 来自 lane、bit5.. 来自 k。
//     于是 log2(n) 个蝶形阶段被拆成：
//       阶段 s = 0..4  —— 跨 lane，用 __shfl_xor_sync 完成；
//       阶段 s = 5..log2(n)-1 —— lane 内，在寄存器数组的不同 slot 间完成。
//   * 载入/写回均按 lane 连续访问（lane 读 base + lane + 32k），访存完全合并。
//
// 融合思路：
//   变换与量化都在寄存器内完成，中间结果不落显存。
//   由于 block_size == 32 恰好等于一个「slot」（元素 [32k, 32k+32)），
//   每个 slot 的 amax 只需一次 warp 归约即可得到，随后立即量化并写出。
//   相比「先写 FHT 结果 -> 再读回来量化」，省掉一整趟往返显存。
#include "hadamard.h"
#include "fp8.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cfloat>
#include <cstdio>
#include <string>
#include <vector>

namespace hd {
namespace {

__device__ __forceinline__ float warp_max32(float v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, off, 32));
    return v;
}

// 寄存器内完成一行长度 n = 1<<LOG_N 的 FHT（含 scale）
//
// 蝶形约定（与定义式 H[i][j] = (-1)^popcount(i&j) 对齐）：
//   设 a 为下标 bit=0 的那个元素、b 为 bit=1 的那个，则
//       bit=0 的元素 -> a + b
//       bit=1 的元素 -> a - b
//   注意：对 bit=1 的元素是「a - b」而不是「b - a」。若写成 v - partner，
//   会在该 bit 上多出一个负号，累积后等价于把第 i 行乘以 (-1)^popcount(i)，
//   结果仍是一个 Hadamard 矩阵、但不再是 H（曾因这个符号约定踩坑）。
template <int LOG_N>
__device__ __forceinline__ void fht_row_regs(float (&v)[(1 << LOG_N) / 32], int lane, float scale) {
    constexpr int V = (1 << LOG_N) / 32;
    // 阶段 0..4：跨 lane（bit0..4 由 lane 决定）
#pragma unroll
    for (int s = 0; s < 5; s++) {
        const int off = 1 << s;
        const bool bit1 = ((lane >> s) & 1) != 0;
#pragma unroll
        for (int k = 0; k < V; k++) {
            const float o = __shfl_xor_sync(0xffffffffu, v[k], off, 32);
            v[k] = bit1 ? (o - v[k]) : (v[k] + o);
        }
    }
    // 阶段 5..LOG_N-1：lane 内不同 slot 之间（bit s 由 k 决定）
#pragma unroll
    for (int s = 5; s < LOG_N; s++) {
        const int kb = 1 << (s - 5);
#pragma unroll
        for (int k = 0; k < V; k++) {
            if ((k & kb) == 0) {
                float a = v[k], b = v[k + kb];
                v[k] = a + b;      // bit=0 侧
                v[k + kb] = a - b; // bit=1 侧
            }
        }
    }
    if (scale != 1.0f) {
#pragma unroll
        for (int k = 0; k < V; k++)
            v[k] *= scale;
    }
}

// ======================= 单独的 FHT kernel =======================
// 每个 warp 以 grid-stride 方式处理多行：单行工作量太小（仅 512B 读写），
// 若一个 warp 只做一行，访存延迟无法被隐藏，实测带宽只有 ~100 GB/s。
// 让 warp 连续处理多行后，编译器可以流水化下一行的载入，带宽显著提升。
template <int LOG_N, int THREADS>
__global__ void k_fht(const float* __restrict__ in, float* __restrict__ out, int64_t rows,
                      float scale) {
    constexpr int V = (1 << LOG_N) / 32;
    constexpr int N = 1 << LOG_N;
    constexpr int NWARP = THREADS / 32;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int64_t nwarp = (int64_t)gridDim.x * NWARP;

    for (int64_t row = (int64_t)blockIdx.x * NWARP + warp; row < rows; row += nwarp) {
        const float* src = in + row * N;
        float* dst = out + row * N;
        float v[V];
#pragma unroll
        for (int k = 0; k < V; k++)
            v[k] = src[lane + 32 * k];

        fht_row_regs<LOG_N>(v, lane, scale);

#pragma unroll
        for (int k = 0; k < V; k++)
            dst[lane + 32 * k] = v[k];
    }
}

// ======================= 单独的量化 kernel（MXFP8 风格，逐 block）=======================
template <int THREADS>
__global__ void k_quantize(const float* __restrict__ in, int64_t n, int block_size, int is_e5m2,
                           uint8_t* __restrict__ qdata, uint8_t* __restrict__ scales) {
    const int lane = threadIdx.x & 31;
    constexpr int NWARP = THREADS / 32;
    const int64_t nblk = (n + block_size - 1) / block_size;
    const int64_t nwarp = (int64_t)gridDim.x * NWARP;
    for (int64_t b = (int64_t)blockIdx.x * NWARP + (threadIdx.x >> 5); b < nblk; b += nwarp) {
        const int64_t base = b * block_size;
        float m = 0.0f;
        for (int k = lane; k < block_size; k += 32) {
            int64_t i = base + k;
            if (i < n)
                m = fmaxf(m, fabsf(in[i]));
        }
        m = warp_max32(m);
        const int e = mx_scale_exp(m, is_e5m2);
        const float sc = ldexpf(1.0f, e);
        if (lane == 0)
            scales[b] = e8m0_from_exp(e);
        const float inv = (sc > 0.0f) ? (1.0f / sc) : 0.0f;
        for (int k = lane; k < block_size; k += 32) {
            int64_t i = base + k;
            if (i < n) {
                float x = in[i] * inv;
                qdata[i] = is_e5m2 ? f32_to_e5m2(x) : f32_to_e4m3(x);
            }
        }
    }
}

// ======================= 融合 kernel：FHT + 量化 =======================
// 仅在 block_size == 32（与 warp slot 对齐）时使用
template <int LOG_N, int THREADS>
__global__ void k_fht_quant(const float* __restrict__ in, int64_t rows, float scale, int is_e5m2,
                            uint8_t* __restrict__ qdata, uint8_t* __restrict__ scales) {
    constexpr int V = (1 << LOG_N) / 32;
    constexpr int N = 1 << LOG_N;
    constexpr int NWARP = THREADS / 32;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int64_t nwarp = (int64_t)gridDim.x * NWARP;

    for (int64_t row = (int64_t)blockIdx.x * NWARP + warp; row < rows; row += nwarp) {
        const float* src = in + row * N;
        float v[V];
#pragma unroll
        for (int k = 0; k < V; k++)
            v[k] = src[lane + 32 * k];

        // 1) 寄存器内完成 FHT
        fht_row_regs<LOG_N>(v, lane, scale);

        // 2) 每个 slot k 恰为量化块 [32k, 32k+32) -> 一次 warp 归约得到该块 amax
        uint8_t* qrow = qdata + row * N;
        uint8_t* srow = scales + row * V;
#pragma unroll
        for (int k = 0; k < V; k++) {
            const float m = warp_max32(fabsf(v[k]));
            const int e = mx_scale_exp(m, is_e5m2);
            const float sc = ldexpf(1.0f, e);
            if (lane == 0)
                srow[k] = e8m0_from_exp(e);
            const float inv = (sc > 0.0f) ? (1.0f / sc) : 0.0f;
            const float x = v[k] * inv;
            qrow[lane + 32 * k] = is_e5m2 ? f32_to_e5m2(x) : f32_to_e4m3(x);
        }
    }
}

constexpr int kThreads = 256;
// 网格上限：让每个 warp 通过 grid-stride 循环处理多行/多块，从而隐藏访存延迟
constexpr int kMaxBlocks = 8192;

template <int LOG_N>
static void launch_fht(const float* in, float* out, int64_t rows, float scale, cudaStream_t st) {
    constexpr int NWARP = kThreads / 32;
    const unsigned blocks = (unsigned)std::min<int64_t>((rows + NWARP - 1) / NWARP, kMaxBlocks);
    k_fht<LOG_N, kThreads><<<blocks, kThreads, 0, st>>>(in, out, rows, scale);
}

template <int LOG_N>
static void launch_fht_quant(const float* in, int64_t rows, float scale, int is_e5m2,
                             uint8_t* qdata, uint8_t* scales, cudaStream_t st) {
    constexpr int NWARP = kThreads / 32;
    const unsigned blocks = (unsigned)std::min<int64_t>((rows + NWARP - 1) / NWARP, kMaxBlocks);
    k_fht_quant<LOG_N, kThreads>
        <<<blocks, kThreads, 0, st>>>(in, rows, scale, is_e5m2, qdata, scales);
}

// 按 n 分派（支持 2 的幂：32 ~ 512，覆盖题目要求的 64/128/256）
bool dispatch_fht(int n, const float* in, float* out, int64_t rows, float scale, cudaStream_t st,
                  std::string& err) {
    switch (n) {
    case 32:
        launch_fht<5>(in, out, rows, scale, st);
        break;
    case 64:
        launch_fht<6>(in, out, rows, scale, st);
        break;
    case 128:
        launch_fht<7>(in, out, rows, scale, st);
        break;
    case 256:
        launch_fht<8>(in, out, rows, scale, st);
        break;
    case 512:
        launch_fht<9>(in, out, rows, scale, st);
        break;
    default:
        err = "hadamard_size 仅支持 2 的幂且 ∈ {32,64,128,256,512}，当前为 " + std::to_string(n);
        return false;
    }
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        err = std::string("FHT kernel 启动失败: ") + cudaGetErrorString(e);
        return false;
    }
    return true;
}

bool dispatch_fht_quant(int n, const float* in, int64_t rows, float scale, int is_e5m2,
                        uint8_t* qdata, uint8_t* scales, cudaStream_t st, std::string& err) {
    switch (n) {
    case 32:
        launch_fht_quant<5>(in, rows, scale, is_e5m2, qdata, scales, st);
        break;
    case 64:
        launch_fht_quant<6>(in, rows, scale, is_e5m2, qdata, scales, st);
        break;
    case 128:
        launch_fht_quant<7>(in, rows, scale, is_e5m2, qdata, scales, st);
        break;
    case 256:
        launch_fht_quant<8>(in, rows, scale, is_e5m2, qdata, scales, st);
        break;
    case 512:
        launch_fht_quant<9>(in, rows, scale, is_e5m2, qdata, scales, st);
        break;
    default:
        err = "融合 kernel 仅支持 hadamard_size ∈ {32,64,128,256,512}";
        return false;
    }
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        err = std::string("融合 kernel 启动失败: ") + cudaGetErrorString(e);
        return false;
    }
    return true;
}

// ======================= 朴素矩阵乘基线（O(n^2)）=======================
// 每个线程计算一个输出元素 y[r][i] = scale * Σ_j H[i][j] * x[r][j]。
// 同一行的 128 个线程共享读 x[r][:]（广播，L1 命中），H 只有 n^2*4 = 64KB（L2 命中）。
// 作为「手动实现」的对照基线：不做任何快速算法，也天然给出大 n 下的正确性参考。
__global__ void k_matmul_baseline(const float* __restrict__ in, const float* __restrict__ H,
                                  float* __restrict__ out, int64_t total, int n, float scale) {
    const int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += stride) {
        const int64_t r = idx / n;
        const int i = (int)(idx - r * n);
        const float* x = in + r * n;
        const float* h = H + (int64_t)i * n;
        float acc = 0.0f;
        for (int j = 0; j < n; j++)
            acc += h[j] * x[j];
        out[idx] = acc * scale;
    }
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

bool gpu_matmul_baseline(const Tensor& in, float scale, Tensor& out, double* ms, std::string& err) {
    const int n = in.cols;
    const int64_t total = in.numel();
    // 构造 H_n：H[i][j] = (-1)^popcount(i & j)
    std::vector<float> H((size_t)n * (size_t)n);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            H[(size_t)i * n + j] = (__builtin_popcount((unsigned)(i & j)) & 1) ? -1.0f : 1.0f;

    out.rows = in.rows;
    out.cols = n;
    out.dtype = in.dtype;
    out.data.assign((size_t)total, 0.0f);

    DevBuf d_in, d_h, d_out;
    if (!d_in.alloc((size_t)total * 4) || !d_h.alloc((size_t)n * (size_t)n * 4) ||
        !d_out.alloc((size_t)total * 4)) {
        err = "显存分配失败(baseline)";
        return false;
    }
    if (cudaMemcpy(d_in.p, in.data.data(), (size_t)total * 4, cudaMemcpyHostToDevice) !=
            cudaSuccess ||
        cudaMemcpy(d_h.p, H.data(), (size_t)n * (size_t)n * 4, cudaMemcpyHostToDevice) !=
            cudaSuccess) {
        err = "H2D 拷贝失败(baseline)";
        return false;
    }
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    cudaEventRecord(e0, 0);
    k_matmul_baseline<<<kMaxBlocks, kThreads>>>((const float*)d_in.p, (const float*)d_h.p,
                                                (float*)d_out.p, total, n, scale);
    cudaEventRecord(e1, 0);
    if (cudaEventSynchronize(e1) != cudaSuccess) {
        err = std::string("矩阵乘基线执行失败: ") + cudaGetErrorString(cudaGetLastError());
        return false;
    }
    float t = 0.0f;
    cudaEventElapsedTime(&t, e0, e1);
    if (ms)
        *ms = (double)t;
    if (cudaMemcpy(out.data.data(), d_out.p, (size_t)total * 4, cudaMemcpyDeviceToHost) !=
        cudaSuccess) {
        err = "D2H 拷贝失败(baseline)";
        return false;
    }
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    return true;
}

// ======================= 主机端接口 =======================
bool gpu_fht(const Tensor& in, const Config& cfg, Tensor& out, double* ms, std::string& err) {
    const int n = in.cols;
    if (n != cfg.hadamard_size) {
        err = "数据的列数(" + std::to_string(n) + ")与 hadamard_size(" +
              std::to_string(cfg.hadamard_size) + ")不一致";
        return false;
    }
    out.rows = in.rows;
    out.cols = in.cols;
    out.dtype = in.dtype;
    out.data.assign(in.data.size(), 0.0f);

    const int64_t total = in.numel();
    DevBuf d_in, d_out;
    if (!d_in.alloc((size_t)total * 4) || !d_out.alloc((size_t)total * 4)) {
        err = "显存分配失败";
        return false;
    }
    if (cudaMemcpy(d_in.p, in.data.data(), (size_t)total * 4, cudaMemcpyHostToDevice) !=
        cudaSuccess) {
        err = "H2D 拷贝失败";
        return false;
    }
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    cudaEventRecord(e0, 0);
    if (!dispatch_fht(n, (const float*)d_in.p, (float*)d_out.p, in.rows, cfg.scale, 0, err))
        return false;
    cudaEventRecord(e1, 0);
    if (cudaEventSynchronize(e1) != cudaSuccess) {
        err = std::string("FHT 执行失败: ") + cudaGetErrorString(cudaGetLastError());
        return false;
    }
    float t = 0.0f;
    cudaEventElapsedTime(&t, e0, e1);
    if (ms)
        *ms = (double)t;
    if (cudaMemcpy(out.data.data(), d_out.p, (size_t)total * 4, cudaMemcpyDeviceToHost) !=
        cudaSuccess) {
        err = "D2H 拷贝失败";
        return false;
    }
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    return true;
}

bool gpu_quantize(const Tensor& in, const Config& cfg, QuantResult& out, double* ms,
                  std::string& err) {
    const int64_t total = in.numel();
    const int bs = cfg.block_size;
    const bool e5m2 = (cfg.quant_format == QuantFormat::FP8_E5M2);
    const int64_t nblk = (total + bs - 1) / bs;
    out.qdata.assign((size_t)total, 0);
    out.scales.assign((size_t)nblk, 0);
    out.num_blocks = nblk;

    DevBuf d_in, d_q, d_s;
    if (!d_in.alloc((size_t)total * 4) || !d_q.alloc((size_t)total) || !d_s.alloc((size_t)nblk)) {
        err = "显存分配失败";
        return false;
    }
    if (cudaMemcpy(d_in.p, in.data.data(), (size_t)total * 4, cudaMemcpyHostToDevice) !=
        cudaSuccess) {
        err = "H2D 拷贝失败";
        return false;
    }
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    cudaEventRecord(e0, 0);
    {
        constexpr int NWARP = kThreads / 32;
        const unsigned blocks = (unsigned)std::min<int64_t>((nblk + NWARP - 1) / NWARP, kMaxBlocks);
        k_quantize<kThreads><<<blocks, kThreads, 0, 0>>>(
            (const float*)d_in.p, total, bs, e5m2 ? 1 : 0, (uint8_t*)d_q.p, (uint8_t*)d_s.p);
    }
    cudaEventRecord(e1, 0);
    if (cudaEventSynchronize(e1) != cudaSuccess) {
        err = std::string("量化 kernel 执行失败: ") + cudaGetErrorString(cudaGetLastError());
        return false;
    }
    float t = 0.0f;
    cudaEventElapsedTime(&t, e0, e1);
    if (ms)
        *ms = (double)t;
    if (cudaMemcpy(out.qdata.data(), d_q.p, (size_t)total, cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(out.scales.data(), d_s.p, (size_t)nblk, cudaMemcpyDeviceToHost) != cudaSuccess) {
        err = "D2H 拷贝失败";
        return false;
    }
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    return true;
}

bool gpu_fht_quant_fused(const Tensor& in, const Config& cfg, QuantResult& out, double* ms,
                         std::string& err) {
    const int n = in.cols;
    if (n != cfg.hadamard_size) {
        err = "数据的列数(" + std::to_string(n) + ")与 hadamard_size(" +
              std::to_string(cfg.hadamard_size) + ")不一致";
        return false;
    }
    if (cfg.block_size != 32) {
        err = "融合 kernel 要求 block_size == 32（与 warp slot 对齐），当前为 " +
              std::to_string(cfg.block_size);
        return false;
    }
    const int64_t total = in.numel();
    const int64_t nblk = in.rows * (n / 32);
    out.qdata.assign((size_t)total, 0);
    out.scales.assign((size_t)nblk, 0);
    out.num_blocks = nblk;
    const bool e5m2 = (cfg.quant_format == QuantFormat::FP8_E5M2);

    DevBuf d_in, d_q, d_s;
    if (!d_in.alloc((size_t)total * 4) || !d_q.alloc((size_t)total) || !d_s.alloc((size_t)nblk)) {
        err = "显存分配失败";
        return false;
    }
    if (cudaMemcpy(d_in.p, in.data.data(), (size_t)total * 4, cudaMemcpyHostToDevice) !=
        cudaSuccess) {
        err = "H2D 拷贝失败";
        return false;
    }
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    cudaEventRecord(e0, 0);
    if (!dispatch_fht_quant(n, (const float*)d_in.p, in.rows, cfg.scale, e5m2 ? 1 : 0,
                            (uint8_t*)d_q.p, (uint8_t*)d_s.p, 0, err))
        return false;
    cudaEventRecord(e1, 0);
    if (cudaEventSynchronize(e1) != cudaSuccess) {
        err = std::string("融合 kernel 执行失败: ") + cudaGetErrorString(cudaGetLastError());
        return false;
    }
    float t = 0.0f;
    cudaEventElapsedTime(&t, e0, e1);
    if (ms)
        *ms = (double)t;
    if (cudaMemcpy(out.qdata.data(), d_q.p, (size_t)total, cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(out.scales.data(), d_s.p, (size_t)nblk, cudaMemcpyDeviceToHost) != cudaSuccess) {
        err = "D2H 拷贝失败";
        return false;
    }
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    return true;
}

} // namespace hd
