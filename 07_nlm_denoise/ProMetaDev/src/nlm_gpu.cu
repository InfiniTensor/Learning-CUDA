// nlm_gpu.cu - GPU 非局部均值降噪
//
// 核心思想：把"每个像素遍历搜索窗口、每次重算整块 patch 距离"的直接做法
// 改写为"按位移分解 + 可分离盒式求和"，从而在不改变数学结果的前提下把
// 计算量从 O(S²·P²) 降到 O(S²·P)：  记 patch 半径 rp、搜索半径 rs，
// P = (2rp+1)²、S = (2rs+1)²。
//
// 对每个位移 s=(dx,dy)：
//   G_s(u) = Σ_c ( I_c(clamp(u)) - I_c(clamp(u+s)) )²          // 逐通道平方差
//   D(p,p+s) = Σ_{k∈patch} G_s(p+k)                            // 盒式求和（可分离）
//   w(p,s)  = exp( -max( D/|patch| - 2σ², 0 ) / h² )
//   num += w·I(clamp(p+s));  den += w
// 由于盒式求和可分离，块内用两次"共享内存 + 前缀式累加"即可完成，
// 无需为每个候选重复读取整块 patch。
#include "nlm.h"

#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

namespace nlm {

namespace {

#define CUDA_CHECK(call)                                                                           \
    do {                                                                                           \
        cudaError_t err_ = (call);                                                                 \
        if (err_ != cudaSuccess) {                                                                 \
            throw std::runtime_error(std::string("CUDA 错误: ") + cudaGetErrorString(err_) +       \
                                     " @ " + #call);                                               \
        }                                                                                          \
    } while (0)

constexpr int TX = 32;       // tile 宽（输出像素数）
constexpr int TY = 8;        // tile 高
constexpr int TPB = TX * TY; // 每块线程数 = 256

// ============================================================
// 主核函数
//   RP: patch 半径（编译期常量，便于展开）
//   C : 通道数（1 或 3）
// ============================================================
template <int RP, int C>
__global__ void __launch_bounds__(TPB)
    nlm_kernel(const float* __restrict__ img, int W, int H, int rs, int sstep, int pstep, int nk,
               float inv_patch, float h2, float two_sigma2, const float* __restrict__ lut,
               int lut_n, float lut_scale, uint8_t* __restrict__ out) {
    constexpr int GX = TX + 2 * RP; // halo 后的 tile 宽
    constexpr int GY = TY + 2 * RP;
    extern __shared__ float smem[];
    float* sG = smem;           // GY × GX 的平方差
    float* sH = smem + GY * GX; // GY × TX 的水平累加结果

    const int tid = threadIdx.y * TX + threadIdx.x;
    const int bx0 = blockIdx.x * TX; // 本 tile 左上角（输出坐标）
    const int by0 = blockIdx.y * TY;
    const int ox = bx0 + threadIdx.x;
    const int oy = by0 + threadIdx.y;
    const bool active = (ox < W && oy < H);

    float num[C];
#pragma unroll
    for (int c = 0; c < C; ++c)
        num[c] = 0.0f;
    float den = 0.0f;

    for (int dy = -rs; dy <= rs; dy += sstep) {
        for (int dx = -rs; dx <= rs; dx += sstep) {
            // ---- ① 铺满 G_s（含 halo），索引越界处按复制边缘处理 ----
            for (int i = tid; i < GY * GX; i += TPB) {
                const int ly = i / GX;
                const int lx = i - ly * GX;
                const int ax = bx0 + lx - RP;
                const int ay = by0 + ly - RP;
                const int px = min(max(ax, 0), W - 1);
                const int py = min(max(ay, 0), H - 1);
                const int cx = min(max(ax + dx, 0), W - 1);
                const int cy = min(max(ay + dy, 0), H - 1);
                const float* A = img + ((size_t)py * W + px) * C;
                const float* B = img + ((size_t)cy * W + cx) * C;
                float g = 0.0f;
#pragma unroll
                for (int c = 0; c < C; ++c) {
                    const float d = A[c] - B[c];
                    g += d * d;
                }
                sG[i] = g;
            }
            __syncthreads();

            // ---- ② 水平方向按 patch 采样点累加：sH[ly][lx] = Σ_j sG[ly][lx + j*pstep] ----
            for (int i = tid; i < GY * TX; i += TPB) {
                const int ly = i / TX;
                const int lx = i - ly * TX;
                const float* row = sG + ly * GX + lx;
                float s = 0.0f;
                for (int j = 0; j < nk; ++j)
                    s += row[j * pstep];
                sH[i] = s;
            }
            __syncthreads();

            // ---- ③ 垂直方向累加 + 权重 + 加权平均 ----
            if (active) {
                const int qx = ox + dx;
                const int qy = oy + dy;
                if (qx >= 0 && qx < W && qy >= 0 && qy < H) { // 搜索窗口裁剪到图像内
                    const float* col = sH + threadIdx.y * TX + threadIdx.x;
                    float D = 0.0f;
                    for (int j = 0; j < nk; ++j)
                        D += col[j * pstep * TX];

                    const float msd = D * inv_patch;
                    float w;
                    if (lut != nullptr) {
                        // LUT 的定义域是指数 x = msd/h²，其中 2σ² 已在建表时折算进去，
                        // 因此这里必须用未减 2σ² 的 msd 查表（用 dd 会导致 2σ² 被减两次）。
                        int idx = (int)(msd * lut_scale);
                        if (idx >= lut_n)
                            idx = lut_n - 1;
                        if (idx < 0)
                            idx = 0;
                        w = lut[idx];
                    } else {
                        // __expf：硬件加速指数，相对误差 ~1e-6，远小于 8 位量化步长
                        w = __expf(-fmaxf(msd - two_sigma2, 0.0f) / h2);
                    }
                    const float* q = img + ((size_t)qy * W + qx) * C;
#pragma unroll
                    for (int c = 0; c < C; ++c)
                        num[c] += w * q[c];
                    den += w;
                }
            }
            __syncthreads(); // 下一轮会覆盖 sG/sH
        }
    }

    if (active) {
        uint8_t* o = out + ((size_t)oy * W + ox) * C;
        const float* self = img + ((size_t)oy * W + ox) * C;
#pragma unroll
        for (int c = 0; c < C; ++c) {
            const float v = (den > 0.0f) ? num[c] / den : self[c];
            const int iv = (int)lrintf(v);
            o[c] = (uint8_t)(iv < 0 ? 0 : (iv > 255 ? 255 : iv));
        }
    }
}

template <int RP>
void launch_c(int ch, const float* d_img, int W, int H, const NLMParams& p, int nk, float inv_patch,
              float h2, float two_sigma2, const float* d_lut, int lut_n, float lut_scale,
              uint8_t* d_out, int* blocks, int* threads) {
    const dim3 grid((W + TX - 1) / TX, (H + TY - 1) / TY);
    const dim3 block(TX, TY);
    const size_t shmem =
        (size_t)((TY + 2 * RP) * (TX + 2 * RP) + (TY + 2 * RP) * TX) * sizeof(float);
    if (ch == 1)
        nlm_kernel<RP, 1><<<grid, block, shmem>>>(d_img, W, H, p.search_radius, p.search_step,
                                                  p.patch_step, nk, inv_patch, h2, two_sigma2,
                                                  d_lut, lut_n, lut_scale, d_out);
    else
        nlm_kernel<RP, 3><<<grid, block, shmem>>>(d_img, W, H, p.search_radius, p.search_step,
                                                  p.patch_step, nk, inv_patch, h2, two_sigma2,
                                                  d_lut, lut_n, lut_scale, d_out);
    CUDA_CHECK(cudaGetLastError());
    if (blocks)
        *blocks = (int)(grid.x * grid.y);
    if (threads)
        *threads = TPB;
}

double now_ms() {
    using clk = std::chrono::high_resolution_clock;
    return std::chrono::duration<double, std::milli>(clk::now().time_since_epoch()).count();
}

} // namespace

// ============================================================
// 权重查找表：w(x) = exp(-max(x - 2σ²/h², 0))，按指数 x = d/h² 在 [0, LUT_XMAX] 等分
// ============================================================
void build_weight_lut(const NLMParams& p, std::vector<float>& lut) {
    const int n = std::max(2, p.lut_bins);
    const double h2 = p.h * p.h;
    const double x2s = 2.0 * p.sigma * p.sigma / h2; // 2σ² 折算到指数域
    lut.resize((size_t)n);
    for (int i = 0; i < n; ++i) {
        const double x = (i + 0.5) * LUT_XMAX / (double)n; // 取档中心
        lut[(size_t)i] = (float)std::exp(-std::max(x - x2s, 0.0));
    }
}

// ============================================================
// GpuNLM::Impl
// ============================================================
struct GpuNLM::Impl {
    float* d_img = nullptr;
    uint8_t* d_out = nullptr;
    float* d_lut = nullptr;
    size_t cap_px = 0; // 已分配像素容量
    size_t cap_lut = 0;
    bool warmed = false;
    PerfStats st;

    // 预热：以 sm_90 为目标编译的 PTX 在本机 sm_120 上由驱动 JIT，
    // 首次启动内核会产生一次性编译开销；先跑一张极小的图把它移出计时区间。
    void warmup() {
        Image tiny;
        tiny.width = 8;
        tiny.height = 8;
        tiny.channels = 1;
        tiny.data.assign(64, 128);
        NLMParams p;
        p.patch_radius = 1;
        p.search_radius = 1;
        p.h = 10.0;
        p.sigma = 10.0;
        denoise(tiny, p, nullptr);
    }

    void ensure(size_t px, size_t lut_n) {
        if (px > cap_px) {
            if (d_img)
                cudaFree(d_img);
            if (d_out)
                cudaFree(d_out);
            CUDA_CHECK(cudaMalloc(&d_img, px * 3 * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_out, px * 3));
            cap_px = px;
        }
        if (lut_n > cap_lut) {
            if (d_lut)
                cudaFree(d_lut);
            if (lut_n > 0)
                CUDA_CHECK(cudaMalloc(&d_lut, lut_n * sizeof(float)));
            cap_lut = lut_n;
        }
    }

    Image denoise(const Image& in, const NLMParams& p, PerfStats* stats) {
        const int W = in.width, H = in.height, C = in.channels;
        if (W <= 0 || H <= 0 || (C != 1 && C != 3))
            throw std::runtime_error("GpuNLM::denoise: 图像尺寸/通道非法");
        if (p.patch_radius < 1 || p.patch_radius > 5)
            throw std::runtime_error("GpuNLM::denoise: patch_radius 支持 1..5");

        if (!warmed) {
            warmed = true; // 先置位，warmup 内部再次进入 denoise 时不再递归
            warmup();
        }

        // ---- 预处理：uint8 → float ----
        // 显存分配必须放在总计时之前：cudaMalloc/cudaFree 在 WSL2 下开销可达数十毫秒，
        // 计进 T_total 会让"同一分辨率首次运行"比"复用缓冲区"慢一倍以上，数字不可比。
        const size_t px = in.pixels();
        const int nk = (2 * p.patch_radius) / p.patch_step + 1;
        std::vector<float> lut;
        if (p.use_lut)
            build_weight_lut(p, lut);
        ensure(px, lut.size());

        const double t_all = now_ms();
        const double t0 = now_ms();
        std::vector<float> f(in.data.size());
        for (size_t i = 0; i < in.data.size(); ++i)
            f[i] = (float)in.data[i];

        CUDA_CHECK(cudaMemcpy(d_img, f.data(), f.size() * sizeof(float), cudaMemcpyHostToDevice));
        if (!lut.empty())
            CUDA_CHECK(
                cudaMemcpy(d_lut, lut.data(), lut.size() * sizeof(float), cudaMemcpyHostToDevice));
        const double t1 = now_ms();

        // ---- 降噪 ----
        const float h2 = (float)(p.h * p.h);
        const float two_sigma2 = (float)(2.0 * p.sigma * p.sigma);
        // 归一化除以"patch 采样点数 × 通道数"（与 CPU 参考保持一致；
        // 漏掉通道数会使彩色图的距离放大 C 倍，导致几乎不降噪）
        const float inv_patch = 1.0f / (float)(nk * nk * C);
        const float* lut_ptr = lut.empty() ? nullptr : d_lut;
        const int lut_n = (int)lut.size();
        // 查表索引 = dd * lut_scale，使 dd/h² ∈ [0, LUT_XMAX] 映射到 [0, lut_n)
        const float lut_scale =
            (lut_n > 0) ? (float)((double)lut_n / (LUT_XMAX * (double)p.h * p.h)) : 0.0f;

        int blocks = 0, threads = 0;
        switch (p.patch_radius) {
        case 1:
            launch_c<1>(C, d_img, W, H, p, nk, inv_patch, h2, two_sigma2, lut_ptr, lut_n, lut_scale,
                        d_out, &blocks, &threads);
            break;
        case 2:
            launch_c<2>(C, d_img, W, H, p, nk, inv_patch, h2, two_sigma2, lut_ptr, lut_n, lut_scale,
                        d_out, &blocks, &threads);
            break;
        case 3:
            launch_c<3>(C, d_img, W, H, p, nk, inv_patch, h2, two_sigma2, lut_ptr, lut_n, lut_scale,
                        d_out, &blocks, &threads);
            break;
        case 4:
            launch_c<4>(C, d_img, W, H, p, nk, inv_patch, h2, two_sigma2, lut_ptr, lut_n, lut_scale,
                        d_out, &blocks, &threads);
            break;
        default:
            launch_c<5>(C, d_img, W, H, p, nk, inv_patch, h2, two_sigma2, lut_ptr, lut_n, lut_scale,
                        d_out, &blocks, &threads);
            break;
        }
        // 内核启动是异步的：必须同步后再取时间，否则耗时会被算到回传阶段
        CUDA_CHECK(cudaDeviceSynchronize());
        const double t2 = now_ms();

        // ---- 回传 ----
        Image out;
        out.width = W;
        out.height = H;
        out.channels = C;
        out.data.resize(in.data.size());
        CUDA_CHECK(cudaMemcpy(out.data.data(), d_out, out.data.size(), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaDeviceSynchronize());
        const double t3 = now_ms();

        st.preprocess_ms = t1 - t0;
        st.denoise_ms = t2 - t1;
        st.kernel_ms = st.denoise_ms;
        st.download_ms = t3 - t2;
        st.total_ms = t3 - t_all;
        st.blocks = blocks;
        st.threads = threads;
        st.megapixels_per_sec = (st.total_ms > 0.0) ? (double)px / st.total_ms / 1000.0 : 0.0;
        if (stats)
            *stats = st;
        return out;
    }

    void release() {
        if (d_img)
            cudaFree(d_img);
        if (d_out)
            cudaFree(d_out);
        if (d_lut)
            cudaFree(d_lut);
        d_img = nullptr;
        d_out = nullptr;
        d_lut = nullptr;
        cap_px = cap_lut = 0;
        st = PerfStats{};
    }
};

GpuNLM::GpuNLM() : impl_(new Impl()) {}
GpuNLM::~GpuNLM() {
    impl_->release();
    delete impl_;
}

Image GpuNLM::denoise(const Image& in, const NLMParams& p, PerfStats* stats) {
    return impl_->denoise(in, p, stats);
}

void GpuNLM::release() {
    impl_->release();
}

} // namespace nlm
