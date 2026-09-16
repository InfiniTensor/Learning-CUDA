// ============================================================================
// NLM 全部设备端实现与 Host 端调度（算法核心，多平台 100% 共享）
// NVIDIA / Iluvatar 直接编译本文件，MACA / MUSA 入口包含本文件，
// 平台差异仅通过 platform_api.h 中的 PAL 宏适配。
// ============================================================================
#include "kernels.h"
#include "platform_api.h"

namespace nlm {

// ---------------------------------------------------------------------------
// 设备端小工具
// ---------------------------------------------------------------------------
__device__ __forceinline__ int ClampI(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

// NLM 权重：w = exp(-max(dist - 2*sigma^2*N_patch, 0) / h^2)
// NLM_FAST_EXP（编译期开关）：使用 __expf 快速指数，误差写入报告误差来源分析
__device__ __forceinline__ float NlmWeight(float dist, float ts2n, float inv_h2) {
    const float d = fmaxf(dist - ts2n, 0.0f);
#ifdef NLM_FAST_EXP
    return __expf(-d * inv_h2);
#else
    return expf(-d * inv_h2);
#endif
}

// ---------------------------------------------------------------------------
// 数据转换 kernel：interleaved u8 <-> planar f32
// ---------------------------------------------------------------------------
__global__ void CvtU8ToF32PlanarKernel(const uint8_t* __restrict__ in,
                                       float* __restrict__ out,
                                       int w, int h, int c) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int n = w * h * c;
    if (idx >= n) return;
    const int ch = idx / (w * h);          // planar 通道
    const int rem = idx - ch * (w * h);    // 通道内像素偏移
    out[(size_t)ch * w * h + rem] = (float)in[(size_t)rem * c + ch];
}

__global__ void CvtF32ToU8InterleavedKernel(const float* __restrict__ in,
                                            uint8_t* __restrict__ out,
                                            int w, int h, int c) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int n = w * h * c;
    if (idx >= n) return;
    const int ch = idx / (w * h);
    const int rem = idx - ch * (w * h);
    const float v = in[(size_t)ch * w * h + rem];
    int q = (int)(v + 0.5f);               // round
    q = q < 0 ? 0 : (q > 255 ? 255 : q);   // clamp [0,255]
    out[(size_t)rem * c + ch] = (uint8_t)q;
}

// ---------------------------------------------------------------------------
// V0 naive kernel：1 thread = 1 output pixel，全部走全局内存
// 边界语义：对每次最终访问坐标独立 clamp（等价 OpenCV BORDER_REPLICATE），
// 与 CPU 参考实现严格一致。
// ---------------------------------------------------------------------------
__global__ void NlmNaiveKernel(const float* __restrict__ src,
                               float* __restrict__ dst,
                               int w, int h, int c, NlmParamsDev p) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    const int pr = p.pr, sr = p.sr;
    const size_t plane = (size_t)w * h;
    float sum_w = 0.0f;
    float sum[3] = {0.0f, 0.0f, 0.0f};

    for (int dy = -sr; dy <= sr; ++dy) {
        for (int dx = -sr; dx <= sr; ++dx) {
            const int qx = x + dx, qy = y + dy;
            // ---- patch L2 距离（跨通道求和，共享权重）----
            float dist = 0.0f;
            for (int py = -pr; py <= pr; ++py) {
                const int ay = ClampI(y + py, 0, h - 1);
                const int by = ClampI(qy + py, 0, h - 1);
                for (int px = -pr; px <= pr; ++px) {
                    const int ax = ClampI(x + px, 0, w - 1);
                    const int bx = ClampI(qx + px, 0, w - 1);
                    for (int ch = 0; ch < c; ++ch) {
                        const float d = src[ch * plane + (size_t)ay * w + ax] -
                                        src[ch * plane + (size_t)by * w + bx];
                        dist += d * d;
                    }
                }
            }
            const float wgt = NlmWeight(dist, p.two_sigma2_n, p.inv_h2);
            sum_w += wgt;
            const int cqx = ClampI(qx, 0, w - 1);
            const int cqy = ClampI(qy, 0, h - 1);
            for (int ch = 0; ch < c; ++ch)
                sum[ch] += wgt * src[ch * plane + (size_t)cqy * w + cqx];
        }
    }

    // sum_w 恒 >= 1（自身权重 w(p,p)=exp(0)=1），无除零风险
    const float inv = 1.0f / sum_w;
    for (int ch = 0; ch < c; ++ch)
        dst[ch * plane + (size_t)y * w + x] = sum[ch] * inv;
}

// ---------------------------------------------------------------------------
// V1 smem kernel：block 计算 16x16 输出 tile，halo 区域（含搜索窗+patch 半径）
// 协同加载进 shared memory，patch 距离全部命中 smem。
//
// 关键性质（正确性论证）：smem[l] = src[clamp(b0 - radius + l)]，
// 而线程中心局部坐标 lc 满足 b0 - radius + lc == 像素全局坐标，
// 因此 smem[lc + 偏移] == src[clamp(全局坐标 + 偏移)]，
// 与 CPU 参考的"最终坐标独立 clamp"语义逐位一致，内层循环无需任何 clamp。
// ---------------------------------------------------------------------------
__global__ void NlmSmemKernel(const float* __restrict__ src,
                              float* __restrict__ dst,
                              int w, int h, int c, NlmParamsDev p) {
    extern __shared__ float smem[];
    const int pr = p.pr, sr = p.sr;
    const int radius = sr + pr;
    const int tileW = blockDim.x + 2 * radius;
    const int tileH = blockDim.y + 2 * radius;
    const int stride = tileW + 1;            // padding 消除 bank conflict
    const int slab = tileH * stride;         // 每通道 smem 元素数
    const int bx0 = blockIdx.x * blockDim.x;
    const int by0 = blockIdx.y * blockDim.y;
    const size_t plane = (size_t)w * h;

    // 协同加载 halo tile（复制边界）
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int nthreads = blockDim.x * blockDim.y;
    for (int ch = 0; ch < c; ++ch) {
        for (int idx = tid; idx < tileW * tileH; idx += nthreads) {
            const int lx = idx % tileW;
            const int ly = idx / tileW;
            const int gx = ClampI(bx0 - radius + lx, 0, w - 1);
            const int gy = ClampI(by0 - radius + ly, 0, h - 1);
            smem[ch * slab + ly * stride + lx] = src[ch * plane + (size_t)gy * w + gx];
        }
    }
    __syncthreads();

    const int x = bx0 + threadIdx.x;
    const int y = by0 + threadIdx.y;
    if (x >= w || y >= h) return;

    const int lcx = threadIdx.x + radius;
    const int lcy = threadIdx.y + radius;
    float sum_w = 0.0f;
    float sum[3] = {0.0f, 0.0f, 0.0f};

    for (int dy = -sr; dy <= sr; ++dy) {
        const int qly = lcy + dy;
        for (int dx = -sr; dx <= sr; ++dx) {
            const int qlx = lcx + dx;
            float dist = 0.0f;
            for (int py = -pr; py <= pr; ++py) {
                const int rowA = (lcy + py) * stride + lcx;
                const int rowB = (qly + py) * stride + qlx;
                for (int px = -pr; px <= pr; ++px) {
                    for (int ch = 0; ch < c; ++ch) {
                        const float d = smem[ch * slab + rowA + px] -
                                        smem[ch * slab + rowB + px];
                        dist += d * d;
                    }
                }
            }
            const float wgt = NlmWeight(dist, p.two_sigma2_n, p.inv_h2);
            sum_w += wgt;
            for (int ch = 0; ch < c; ++ch)
                sum[ch] += wgt * smem[ch * slab + qly * stride + qlx];
        }
    }

    const float inv = 1.0f / sum_w;
    for (int ch = 0; ch < c; ++ch)
        dst[ch * plane + (size_t)y * w + x] = sum[ch] * inv;
}

// ---------------------------------------------------------------------------
// V2 kernel：V1 + patch 半径模板化（#pragma unroll 完全展开内层循环）
// 与 V1 语义逐位一致，仅通过编译期常量消除循环开销、提升指令级并行。
// ---------------------------------------------------------------------------
template <int PR>
__global__ __launch_bounds__(256) void NlmSmemUnrollKernel(
        const float* __restrict__ src, float* __restrict__ dst,
        int w, int h, int c, NlmParamsDev p) {
    extern __shared__ float smem[];
    const int pr = PR;
    const int sr = p.sr;
    const int radius = sr + pr;
    const int tileW = blockDim.x + 2 * radius;
    const int tileH = blockDim.y + 2 * radius;
    const int stride = tileW + 1;
    const int slab = tileH * stride;
    const int bx0 = blockIdx.x * blockDim.x;
    const int by0 = blockIdx.y * blockDim.y;
    const size_t plane = (size_t)w * h;

    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int nthreads = blockDim.x * blockDim.y;
    for (int ch = 0; ch < c; ++ch) {
        for (int idx = tid; idx < tileW * tileH; idx += nthreads) {
            const int lx = idx % tileW;
            const int ly = idx / tileW;
            const int gx = ClampI(bx0 - radius + lx, 0, w - 1);
            const int gy = ClampI(by0 - radius + ly, 0, h - 1);
            smem[ch * slab + ly * stride + lx] = src[ch * plane + (size_t)gy * w + gx];
        }
    }
    __syncthreads();

    const int x = bx0 + threadIdx.x;
    const int y = by0 + threadIdx.y;
    if (x >= w || y >= h) return;

    const int lcx = threadIdx.x + radius;
    const int lcy = threadIdx.y + radius;
    float sum_w = 0.0f;
    float sum[3] = {0.0f, 0.0f, 0.0f};

    for (int dy = -sr; dy <= sr; ++dy) {
        const int qly = lcy + dy;
        for (int dx = -sr; dx <= sr; ++dx) {
            const int qlx = lcx + dx;
            float dist = 0.0f;
#pragma unroll
            for (int py = -PR; py <= PR; ++py) {
                const int rowA = (lcy + py) * stride + lcx;
                const int rowB = (qly + py) * stride + qlx;
#pragma unroll
                for (int px = -PR; px <= PR; ++px) {
                    for (int ch = 0; ch < c; ++ch) {
                        const float d = smem[ch * slab + rowA + px] -
                                        smem[ch * slab + rowB + px];
                        dist += d * d;
                    }
                }
            }
            const float wgt = NlmWeight(dist, p.two_sigma2_n, p.inv_h2);
            sum_w += wgt;
            for (int ch = 0; ch < c; ++ch)
                sum[ch] += wgt * smem[ch * slab + qly * stride + qlx];
        }
    }

    const float inv = 1.0f / sum_w;
    for (int ch = 0; ch < c; ++ch)
        dst[ch * plane + (size_t)y * w + x] = sum[ch] * inv;
}

} // namespace nlm

// ===========================================================================
// Host 端调度接口（对 pipeline 暴露）
// ===========================================================================
bool GpuAlloc(void** ptr, size_t bytes, std::string* err) {
    if (!ptr || bytes == 0) {
        if (err) *err = "GpuAlloc: 非法参数";
        return false;
    }
    GPU_CHECK(GPU_MALLOC(ptr, bytes), err);
    return true;
}

bool GpuFree(void* ptr, std::string* err) {
    if (!ptr) return true;
    GPU_CHECK(GPU_FREE(ptr), err);
    return true;
}

bool GpuCopyH2D(void* dst, const void* src, size_t bytes, std::string* err) {
    GPU_CHECK(GPU_MEMCPY(dst, src, bytes, GPU_MEMCPY_H2D), err);
    return true;
}

bool GpuCopyD2H(void* dst, const void* src, size_t bytes, std::string* err) {
    GPU_CHECK(GPU_MEMCPY(dst, src, bytes, GPU_MEMCPY_D2H), err);
    return true;
}

bool GpuCvtU8ToF32Planar(const uint8_t* d_in, float* d_out, int w, int h, int c,
                         std::string* err) {
    const int n = w * h * c;
    const int block = 256;
    const int grid = (n + block - 1) / block;
    nlm::CvtU8ToF32PlanarKernel<<<grid, block>>>(d_in, d_out, w, h, c);
    GPU_CHECK(GPU_GET_LAST_ERROR(), err);
    return true;
}

bool GpuCvtF32ToU8Interleaved(const float* d_in, uint8_t* d_out, int w, int h, int c,
                              std::string* err) {
    const int n = w * h * c;
    const int block = 256;
    const int grid = (n + block - 1) / block;
    nlm::CvtF32ToU8InterleavedKernel<<<grid, block>>>(d_in, d_out, w, h, c);
    GPU_CHECK(GPU_GET_LAST_ERROR(), err);
    return true;
}

bool GpuNlm(const float* d_src, float* d_dst, int w, int h, int c,
            const NlmParamsDev& p, int version, float* kernel_ms, std::string* err) {
    typedef void (*NlmKernelFn)(const float*, float*, int, int, int, NlmParamsDev);

    const dim3 block(16, 16);
    const dim3 grid((w + (int)block.x - 1) / (int)block.x,
                    (h + (int)block.y - 1) / (int)block.y, 1);

    // smem 需求：c 个通道，每通道 (tileH) x (tileW+1) 个 float
    const int radius = p.sr + p.pr;
    const int tileW = (int)block.x + 2 * radius;
    const int tileH = (int)block.y + 2 * radius;
    const size_t smem_bytes = (size_t)c * tileH * (tileW + 1) * sizeof(float);

    NlmKernelFn fn = NULL;
    bool use_smem = false;

    if (version == 0) {
        fn = nlm::NlmNaiveKernel;
    } else if (version == 1 || version == 2) {
        // 查询设备动态 smem 上限，不足则回退 naive（保证功能正确）
        int dev = 0, smem_optin = 48 * 1024;
        GPU_CHECK(GPU_GET_DEVICE(&dev), err);
        GPU_CHECK(GPU_GET_SMEM_OPTIN(dev, &smem_optin), err);
        if (smem_bytes <= (size_t)smem_optin) {
            use_smem = true;
            if (version == 1) {
                fn = nlm::NlmSmemKernel;
            } else {
                // V2：按 patch 半径模板分派。
                // 实测（RTX 3060, bench.csv）：pr<=3 时完全展开有 5%~30% 收益；
                // pr=4 时 9x9x3 内层全展开导致寄存器溢出、占用率塌陷，
                // 性能反降 ~70%（如 4K RGB large 参数 5540ms -> 9472ms），
                // 因此 pr>=4 一律退回 V1 通用循环路径。
                switch (p.pr) {
                    case 1: fn = nlm::NlmSmemUnrollKernel<1>; break;
                    case 2: fn = nlm::NlmSmemUnrollKernel<2>; break;
                    case 3: fn = nlm::NlmSmemUnrollKernel<3>; break;
                    default: fn = nlm::NlmSmemKernel; break;
                }
            }
            if (smem_bytes > 48 * 1024) {
                GPU_CHECK(GPU_FUNC_SET_MAX_SMEM(fn, smem_bytes), err);
            }
        } else {
            fn = nlm::NlmNaiveKernel;  // smem 不足，回退
        }
    } else {
        if (err) *err = "kernel version=" + std::to_string(version) +
                        " 未实现（V3 为预留进阶版本）";
        return false;
    }

    gpuEvent_t e0, e1;
    GPU_CHECK(GPU_EVENT_CREATE(&e0), err);
    GPU_CHECK(GPU_EVENT_CREATE(&e1), err);

    GPU_CHECK(GPU_EVENT_RECORD(e0), err);
    if (use_smem) {
        fn<<<grid, block, smem_bytes>>>(d_src, d_dst, w, h, c, p);
    } else {
        fn<<<grid, block>>>(d_src, d_dst, w, h, c, p);
    }
    GPU_CHECK(GPU_GET_LAST_ERROR(), err);
    GPU_CHECK(GPU_EVENT_RECORD(e1), err);
    GPU_CHECK(GPU_EVENT_SYNC(e1), err);

    float ms = 0.0f;
    GPU_CHECK(GPU_EVENT_ELAPSED(&ms, e0, e1), err);
    GPU_EVENT_DESTROY(e0);
    GPU_EVENT_DESTROY(e1);
    if (kernel_ms) *kernel_ms = ms;
    return true;
}
