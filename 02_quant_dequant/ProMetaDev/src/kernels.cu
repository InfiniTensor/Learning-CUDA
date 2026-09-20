// kernels.cu - CUDA 量化 / 反量化 kernel 与 GPU 流程编排
//
// 设计要点：
//   * 全部为纯软件位运算实现，不依赖 Hopper/Blackwell/Ada 的新指令
//   * BLOCK 模式：共享内存分块 + warp shuffle 规约，访存合并
//   * TENSOR 模式：先做全张量 amax 规约，再用统一缩放因子逐元素量化
//   * NVFP4：block 局部 E4M3 缩放 + 张量级 FP32 全局缩放，4bit 双元素打包
//   * 结果与 reference.cpp 逐位一致（相同缩放推导、相同舍入、相同打包布局）
#include "quant.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <cstdio>
#include <algorithm>
#include <string>
#include <vector>

namespace lowp {

// ======================= 设备端辅助 =======================
template <int WIDTH> __device__ __forceinline__ float warp_max_w(float v) {
#pragma unroll
    for (int off = WIDTH / 2; off > 0; off >>= 1)
        v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, off, WIDTH));
    return v;
}

__device__ __forceinline__ void atomic_max_float(float* addr, float val) {
    if (val < 0.0f)
        return;
    int* a = reinterpret_cast<int*>(addr);
    int old = *a, assumed;
    do {
        assumed = old;
        float nv = fmaxf(__int_as_float(assumed), val);
        old = atomicCAS(a, assumed, __float_as_int(nv));
    } while (old != assumed);
}

template <typename T> __device__ __forceinline__ T cast_out(float v);
template <> __device__ __forceinline__ float cast_out<float>(float v) {
    return v;
}
template <> __device__ __forceinline__ __half cast_out<__half>(float v) {
    return __float2half(v);
}
template <> __device__ __forceinline__ __nv_bfloat16 cast_out<__nv_bfloat16>(float v) {
    return __float2bfloat16(v);
}

__device__ __forceinline__ ElemFmt to_elem(int e) {
    return e == 0 ? ElemFmt::E4M3 : ElemFmt::E5M2;
}

// 单个 block 的缩放因子推导（GPU 与 CPU 参考完全一致）
__device__ __forceinline__ void block_scale(int is_nvfp4, int elem_fmt, float amax, float gscale,
                                            uint8_t& enc, float& val) {
    if (!is_nvfp4) {
        int e = mxfp8_scale_exp(amax, elem_max_val(to_elem(elem_fmt)));
        enc = e8m0_from_exp(e);
        val = ldexpf(1.0f, e);
    } else {
        float target = amax / NVFP4_ELEM_MAX;
        enc = f32_to_e4m3(target / gscale, 0, 0.0f); // 缩放因子固定最近舍入
        val = e4m3_to_f32(enc) * gscale;
    }
}

__device__ __forceinline__ uint8_t encode_elem(int is_nvfp4, int elem_fmt, float v, int rmode,
                                               float rnd) {
    if (!is_nvfp4)
        return (elem_fmt == 0) ? f32_to_e4m3(v, rmode, rnd) : f32_to_e5m2(v, rmode, rnd);
    return f32_to_e2m1(v, rmode, rnd);
}

// ======================= 全局 amax 规约 =======================
__global__ void k_reduce_amax(const float* __restrict__ in, int64_t n, float* __restrict__ out) {
    float m = 0.0f;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride)
        m = fmaxf(m, fabsf(in[i]));
    __shared__ float sm[256];
    sm[threadIdx.x] = m;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            sm[threadIdx.x] = fmaxf(sm[threadIdx.x], sm[threadIdx.x + s]);
        __syncthreads();
    }
    if (threadIdx.x == 0)
        atomic_max_float(out, sm[0]);
}

// 量化前的缩放因子收尾（单线程）
struct ScaleInfo {
    float gscale;
    float tensor_scale;
    unsigned char tensor_scale_byte;
};

__global__ void k_finalize_scales(const float* __restrict__ gmax, int is_nvfp4, int elem_fmt,
                                  ScaleInfo* __restrict__ si) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        float gm = *gmax;
        ScaleInfo s;
        if (is_nvfp4) {
            s.gscale = nvfp4_global_scale(gm);
            float target = gm / NVFP4_ELEM_MAX;
            uint8_t sc = f32_to_e4m3(target / s.gscale, 0, 0.0f);
            s.tensor_scale = e4m3_to_f32(sc) * s.gscale;
            s.tensor_scale_byte = sc;
        } else {
            s.gscale = 1.0f;
            int e = mxfp8_scale_exp(gm, elem_max_val(to_elem(elem_fmt)));
            s.tensor_scale = ldexpf(1.0f, e);
            s.tensor_scale_byte = e8m0_from_exp(e);
        }
        *si = s;
    }
}

// ======================= BLOCK 模式量化（共享内存分块 + shuffle） =======================
// 一个 thread block 处理 TILE = THREADS * BS 个元素；BS ∈ {16, 32}
template <int BS, int THREADS>
__global__ void k_quant_block(const float* __restrict__ in, int64_t n, int is_nvfp4, int elem_fmt,
                              int rmode, uint32_t seed, const ScaleInfo* __restrict__ si,
                              uint8_t* __restrict__ packed, uint8_t* __restrict__ scales) {
    const float gscale = si->gscale;
    constexpr int TILE = THREADS * BS;
    constexpr int NSUB = TILE / BS; // 本 tile 内 block 数
    constexpr int BPW = 32 / BS;    // 每个 warp 每轮处理 block 数
    constexpr int LANES = (BS >= 32) ? 32 : BS;
    constexpr int NWARPS = THREADS / 32;

    extern __shared__ float smem[];
    float* sdata = smem;         // TILE 个元素
    float* sscale = smem + TILE; // NSUB 个 block 缩放值

    const int64_t base = (int64_t)blockIdx.x * TILE;
    if (base >= n)
        return;
    int64_t nsub = (n - base + BS - 1) / BS;
    if (nsub > NSUB)
        nsub = NSUB;

    // 1) 合并载入 tile（越界补 0，不影响 amax，因为取 max(|x|,0)）
    for (int k = threadIdx.x; k < TILE; k += THREADS) {
        int64_t gi = base + k;
        sdata[k] = (gi < n) ? in[gi] : 0.0f;
    }
    __syncthreads();

    // 2) 计算每个 block 的缩放因子
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    for (int64_t b = warp * BPW; b < nsub; b += NWARPS * BPW) {
        int64_t blk = b + lane / LANES;
        int l = lane % LANES;
        float x = (blk < nsub) ? sdata[blk * BS + l] : 0.0f;
        float amax = warp_max_w<LANES>(fabsf(x));
        if (l == 0 && blk < nsub) {
            uint8_t enc;
            float val;
            block_scale(is_nvfp4, elem_fmt, amax, gscale, enc, val);
            sscale[blk] = val;
            int64_t gblk = base / BS + blk;
            scales[gblk] = enc;
        }
    }
    __syncthreads();

    // 3) 量化并写出
    if (!is_nvfp4) {
        for (int k = threadIdx.x; k < TILE; k += THREADS) {
            int64_t gi = base + k;
            if (gi >= n)
                break;
            float sc = sscale[k / BS];
            float inv = (sc > 0.0f) ? (1.0f / sc) : 0.0f;
            float rnd = (rmode == 1) ? rng_uniform(seed, (uint64_t)gi) : 0.0f;
            packed[gi] = encode_elem(0, elem_fmt, sdata[k] * inv, rmode, rnd);
        }
    } else {
        // 每线程处理一对元素 (2k, 2k+1)，合并写入同一字节，避免写竞争
        for (int k = threadIdx.x * 2; k < TILE; k += THREADS * 2) {
            int64_t gi = base + k;
            if (gi >= n)
                break;
            float sc = sscale[k / BS];
            float inv = (sc > 0.0f) ? (1.0f / sc) : 0.0f;
            float r0 = (rmode == 1) ? rng_uniform(seed, (uint64_t)gi) : 0.0f;
            uint8_t c0 = f32_to_e2m1(sdata[k] * inv, rmode, r0);
            uint8_t c1 = 0;
            if (gi + 1 < n) {
                float r1 = (rmode == 1) ? rng_uniform(seed, (uint64_t)(gi + 1)) : 0.0f;
                c1 = f32_to_e2m1(sdata[k + 1] * inv, rmode, r1);
            }
            packed[gi >> 1] = (uint8_t)((c0 & 0x0Fu) | ((c1 & 0x0Fu) << 4));
        }
    }
}

// ======================= TENSOR 模式量化（统一缩放因子） =======================
__global__ void k_quant_tensor(const float* __restrict__ in, int64_t n, int is_nvfp4, int elem_fmt,
                               int rmode, uint32_t seed, const ScaleInfo* __restrict__ si,
                               uint8_t* __restrict__ packed) {
    const float scale_val = si->tensor_scale;
    float inv = (scale_val > 0.0f) ? (1.0f / scale_val) : 0.0f;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    if (!is_nvfp4) {
        for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
            float rnd = (rmode == 1) ? rng_uniform(seed, (uint64_t)i) : 0.0f;
            packed[i] = encode_elem(0, elem_fmt, in[i] * inv, rmode, rnd);
        }
    } else {
        for (int64_t i = ((int64_t)blockIdx.x * blockDim.x + threadIdx.x) * 2; i < n;
             i += stride * 2) {
            float r0 = (rmode == 1) ? rng_uniform(seed, (uint64_t)i) : 0.0f;
            uint8_t c0 = f32_to_e2m1(in[i] * inv, rmode, r0);
            uint8_t c1 = 0;
            if (i + 1 < n) {
                float r1 = (rmode == 1) ? rng_uniform(seed, (uint64_t)(i + 1)) : 0.0f;
                c1 = f32_to_e2m1(in[i + 1] * inv, rmode, r1);
            }
            packed[i >> 1] = (uint8_t)((c0 & 0x0Fu) | ((c1 & 0x0Fu) << 4));
        }
    }
}

// ======================= 通用 BLOCK 路径（任意 block_size） =======================
// 当 block_size 不是 16/32 时走此路径：先算 block 缩放因子，再逐元素/逐对量化。
// 访存仍然合并，只是少了共享内存分块与 warp 级融合，作为通用兜底实现。
__global__ void k_block_scales_general(const float* __restrict__ in, int64_t n, int64_t bs,
                                       int is_nvfp4, int elem_fmt, const ScaleInfo* __restrict__ si,
                                       float* __restrict__ bscale, uint8_t* __restrict__ scales) {
    const float gscale = si->gscale;
    const int64_t nb = (n + bs - 1) / bs;
    const int lane = threadIdx.x & 31;
    const int64_t nwarps = (int64_t)gridDim.x * (blockDim.x / 32);
    for (int64_t w = (int64_t)blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32); w < nb;
         w += nwarps) {
        const int64_t base = w * bs;
        float m = 0.0f;
        for (int64_t k = lane; k < bs; k += 32) {
            int64_t gi = base + k;
            if (gi < n)
                m = fmaxf(m, fabsf(in[gi]));
        }
        m = warp_max_w<32>(m);
        if (lane == 0) {
            uint8_t enc;
            float val;
            block_scale(is_nvfp4, elem_fmt, m, gscale, enc, val);
            bscale[w] = val;
            scales[w] = enc;
        }
    }
}

// 逐元素量化（MXFP8）
__global__ void k_quant_from_scales_f8(const float* __restrict__ in, int64_t n, int64_t bs,
                                       int elem_fmt, int rmode, uint32_t seed,
                                       const float* __restrict__ bscale,
                                       uint8_t* __restrict__ packed) {
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        float sc = bscale[i / bs];
        float inv = (sc > 0.0f) ? (1.0f / sc) : 0.0f;
        float rnd = (rmode == 1) ? rng_uniform(seed, (uint64_t)i) : 0.0f;
        packed[i] = encode_elem(0, elem_fmt, in[i] * inv, rmode, rnd);
    }
}

// 逐对量化 + 打包（NVFP4，每字节 2 元素）
__global__ void k_quant_from_scales_f4(const float* __restrict__ in, int64_t n, int64_t bs,
                                       int rmode, uint32_t seed, const float* __restrict__ bscale,
                                       uint8_t* __restrict__ packed) {
    const int64_t npair = (n + 1) / 2;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for (int64_t p = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; p < npair; p += stride) {
        const int64_t i0 = p * 2, i1 = i0 + 1;
        float sc0 = bscale[i0 / bs];
        float inv0 = (sc0 > 0.0f) ? (1.0f / sc0) : 0.0f;
        float r0 = (rmode == 1) ? rng_uniform(seed, (uint64_t)i0) : 0.0f;
        uint8_t c0 = f32_to_e2m1(in[i0] * inv0, rmode, r0);
        uint8_t c1 = 0;
        if (i1 < n) {
            float sc1 = bscale[i1 / bs];
            float inv1 = (sc1 > 0.0f) ? (1.0f / sc1) : 0.0f;
            float r1 = (rmode == 1) ? rng_uniform(seed, (uint64_t)i1) : 0.0f;
            c1 = f32_to_e2m1(in[i1] * inv1, rmode, r1);
        }
        packed[p] = (uint8_t)((c0 & 0x0Fu) | ((c1 & 0x0Fu) << 4));
    }
}

// ======================= 反量化 =======================
// 向量化 packed load：每线程用一次 uchar4 读取 4 个打包字节
//   MXFP8：4 字节 -> 4 个元素
//   NVFP4：4 字节 -> 8 个元素（每字节 2 个 nibble）
// 这样每个打包字节只被读取一次，避免逐元素访问造成的 2 倍读放大。
template <typename T>
__global__ void k_dequant_mxfp8(const uint8_t* __restrict__ packed,
                                const uint8_t* __restrict__ scales, int64_t n, int64_t bs,
                                int elem_fmt, T* __restrict__ out) {
    const int64_t nvec = (n + 3) / 4;
    const int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for (int64_t v = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; v < nvec; v += stride) {
        const uchar4 b = reinterpret_cast<const uchar4*>(packed)[v];
        const uint8_t bytes[4] = {b.x, b.y, b.z, b.w};
        const int64_t i0 = v * 4;
#pragma unroll
        for (int k = 0; k < 4; k++) {
            const int64_t i = i0 + k;
            if (i >= n)
                break;
            float sc = e8m0_to_f32(scales[i / bs]);
            float val = (elem_fmt == 0) ? e4m3_to_f32(bytes[k]) : e5m2_to_f32(bytes[k]);
            out[i] = cast_out<T>(val * sc);
        }
    }
}

template <typename T>
__global__ void k_dequant_nvfp4(const uint8_t* __restrict__ packed,
                                const uint8_t* __restrict__ scales, int64_t n, int64_t bs,
                                float gscale, T* __restrict__ out) {
    const int64_t nvec = (n + 7) / 8;
    const int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for (int64_t v = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; v < nvec; v += stride) {
        const uchar4 b = reinterpret_cast<const uchar4*>(packed)[v];
        const uint8_t bytes[4] = {b.x, b.y, b.z, b.w};
        const int64_t i0 = v * 8;
#pragma unroll
        for (int k = 0; k < 8; k++) {
            const int64_t i = i0 + k;
            if (i >= n)
                break;
            const uint8_t byte = bytes[k >> 1];
            const uint8_t code = (k & 1) ? (uint8_t)(byte >> 4) : (uint8_t)(byte & 0x0Fu);
            float sc = e4m3_to_f32(scales[i / bs]) * gscale;
            out[i] = cast_out<T>(e2m1_to_f32(code) * sc);
        }
    }
}

// ======================= 主机端编排 =======================
struct DevBufs {
    float *in = nullptr, *gmax = nullptr;
    ScaleInfo* si = nullptr;
    float* bscale = nullptr;
    uint8_t *packed = nullptr, *scales = nullptr;
    void* out = nullptr;
    ~DevBufs() {
        if (in)
            cudaFree(in);
        if (gmax)
            cudaFree(gmax);
        if (si)
            cudaFree(si);
        if (bscale)
            cudaFree(bscale);
        if (packed)
            cudaFree(packed);
        if (scales)
            cudaFree(scales);
        if (out)
            cudaFree(out);
    }
};

template <typename T>
static void launch_dequant(const uint8_t* dp, const uint8_t* ds, int64_t n, int64_t bs,
                           int is_nvfp4, int elem_fmt, float gscale, T* dout, int blocks,
                           int threads) {
    if (!is_nvfp4)
        k_dequant_mxfp8<T><<<blocks, threads>>>(dp, ds, n, bs, elem_fmt, dout);
    else
        k_dequant_nvfp4<T><<<blocks, threads>>>(dp, ds, n, bs, gscale, dout);
}

bool run_gpu(const std::vector<float>& in, int64_t n, const Config& cfg, QuantResult& out,
             std::string& err) {
    if (n <= 0) {
        err = "空张量";
        return false;
    }

    const bool is_nvfp4 = (cfg.format == LowFormat::NVFP4);
    const bool tensor_mode = (cfg.scale_mode == ScaleMode::TENSOR);
    const int elem_fmt = (cfg.elem == ElemFmt::E4M3) ? 0 : 1;
    const int rmode = (cfg.round == RoundMode::STOCHASTIC) ? 1 : 0;

    // block 粒度：16 / 32 走共享内存优化路径（MX / NV 标准粒度），其余走通用路径
    int64_t bs = tensor_mode ? n : (int64_t)cfg.block_size;
    if (bs <= 0) {
        err = "block_size 必须为正整数";
        return false;
    }
    const bool use_fast_block = (!tensor_mode && (bs == 16 || bs == 32));
    const bool use_gen_block = (!tensor_mode && !use_fast_block);
    int64_t nb = (n + bs - 1) / bs;
    int64_t packed_bytes = is_nvfp4 ? (n + 1) / 2 : n;

    out.block_size = (int)bs;
    out.num_blocks = nb;
    out.packed.assign((size_t)packed_bytes, 0);
    out.scales.assign((size_t)nb, 0);
    out.dequant.assign((size_t)n, 0.0f);

    DevBufs d;
    cudaEvent_t e0 = nullptr, e1 = nullptr, e2 = nullptr, e3 = nullptr;
    auto fail = [&](const char* what) {
        err = std::string("CUDA 失败(") + what + "): " + cudaGetErrorString(cudaGetLastError());
        if (e0)
            cudaEventDestroy(e0);
        if (e1)
            cudaEventDestroy(e1);
        if (e2)
            cudaEventDestroy(e2);
        if (e3)
            cudaEventDestroy(e3);
        return false;
    };

    if (cudaMalloc(&d.in, (size_t)n * 4) != cudaSuccess)
        return fail("malloc in");
    if (cudaMalloc(&d.gmax, sizeof(float)) != cudaSuccess)
        return fail("malloc gmax");
    if (cudaMalloc(&d.si, sizeof(ScaleInfo)) != cudaSuccess)
        return fail("malloc si");
    // 多分配 16 字节：向量化 packed load 一次读 4 字节，允许尾部越界读取
    if (cudaMalloc(&d.packed, (size_t)packed_bytes + 16) != cudaSuccess)
        return fail("malloc packed");
    if (cudaMalloc(&d.scales, (size_t)nb) != cudaSuccess)
        return fail("malloc scales");
    if (use_gen_block && cudaMalloc(&d.bscale, (size_t)nb * sizeof(float)) != cudaSuccess)
        return fail("malloc bscale");
    int osz = (cfg.out_dtype == OutDtype::FP32) ? 4 : 2;
    if (cudaMalloc(&d.out, (size_t)n * osz) != cudaSuccess)
        return fail("malloc out");

    if (cudaMemcpy(d.in, in.data(), (size_t)n * 4, cudaMemcpyHostToDevice) != cudaSuccess)
        return fail("H2D in");
    if (cudaMemset(d.gmax, 0, sizeof(float)) != cudaSuccess)
        return fail("memset gmax");

    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    cudaEventCreate(&e2);
    cudaEventCreate(&e3);

    // ---- 量化 ----
    // MXFP8 的 BLOCK 模式不需要全局缩放，预先写入默认值（该拷贝不计入 kernel 计时）
    if (!is_nvfp4 && !tensor_mode) {
        ScaleInfo def{1.0f, 0.0f, 0};
        if (cudaMemcpy(d.si, &def, sizeof(ScaleInfo), cudaMemcpyHostToDevice) != cudaSuccess)
            return fail("H2D default si");
    }

    cudaEventRecord(e0);
    // 1) 全张量 amax + 缩放收尾（NVFP4 / TENSOR 模式需要；全部在 device 侧，避免主机同步）
    if (is_nvfp4 || tensor_mode) {
        const int threads = 256;
        const int blocks = (int)std::min<int64_t>((n + threads - 1) / threads, 4096);
        k_reduce_amax<<<blocks, threads>>>(d.in, n, d.gmax);
        k_finalize_scales<<<1, 1>>>(d.gmax, is_nvfp4 ? 1 : 0, elem_fmt, d.si);
    }

    // 2) 量化 kernel
    if (!tensor_mode) {
        const int threads = 256;
        if (!use_gen_block) {
            const int64_t TILE = (int64_t)threads * bs;
            const int blocks = (int)std::min<int64_t>((n + TILE - 1) / TILE, 65535);
            const int smem = (int)((TILE + threads) * sizeof(float));
            if (bs == 32)
                k_quant_block<32, 256><<<blocks, threads, smem>>>(
                    d.in, n, is_nvfp4 ? 1 : 0, elem_fmt, rmode, cfg.seed, d.si, d.packed, d.scales);
            else
                k_quant_block<16, 256><<<blocks, threads, smem>>>(
                    d.in, n, is_nvfp4 ? 1 : 0, elem_fmt, rmode, cfg.seed, d.si, d.packed, d.scales);
        } else {
            // 通用路径：先求 block 缩放因子，再逐元素/逐对量化
            const int wblocks = (int)std::min<int64_t>((nb + 7) / 8, 4096);
            k_block_scales_general<<<wblocks, threads>>>(d.in, n, bs, is_nvfp4 ? 1 : 0, elem_fmt,
                                                         d.si, d.bscale, d.scales);
            const int eblocks = (int)std::min<int64_t>((n + threads - 1) / threads, 4096);
            if (!is_nvfp4)
                k_quant_from_scales_f8<<<eblocks, threads>>>(d.in, n, bs, elem_fmt, rmode, cfg.seed,
                                                             d.bscale, d.packed);
            else
                k_quant_from_scales_f4<<<eblocks, threads>>>(d.in, n, bs, rmode, cfg.seed, d.bscale,
                                                             d.packed);
        }
    } else {
        const int threads = 256;
        const int blocks = (int)std::min<int64_t>((n + threads - 1) / threads, 4096);
        k_quant_tensor<<<blocks, threads>>>(d.in, n, is_nvfp4 ? 1 : 0, elem_fmt, rmode, cfg.seed,
                                            d.si, d.packed);
    }
    cudaEventRecord(e1);

    // 3) 取回缩放信息 / 数据
    //    仅 NVFP4 或 TENSOR 模式会写 d.si，其余情况 global_scale 恒为 1
    out.global_scale = 1.0f;
    if (is_nvfp4 || tensor_mode) {
        ScaleInfo hsi{};
        if (cudaMemcpy(&hsi, d.si, sizeof(ScaleInfo), cudaMemcpyDeviceToHost) != cudaSuccess)
            return fail("D2H si final");
        out.global_scale = hsi.gscale;
        if (tensor_mode) {
            // tensor 模式的缩放因子由 finalize kernel 算出，需回写到设备端供反量化使用
            out.scales[0] = hsi.tensor_scale_byte;
            if (cudaMemcpy(d.scales, &out.scales[0], 1, cudaMemcpyHostToDevice) != cudaSuccess)
                return fail("H2D tensor scale");
        }
    }
    if (!tensor_mode) {
        if (cudaMemcpy(out.scales.data(), d.scales, (size_t)nb, cudaMemcpyDeviceToHost) !=
            cudaSuccess)
            return fail("D2H scales");
    }
    if (cudaMemcpy(out.packed.data(), d.packed, (size_t)packed_bytes, cudaMemcpyDeviceToHost) !=
        cudaSuccess)
        return fail("D2H packed");

    const float gscale = out.global_scale; // 反量化内核使用同一缩放（主机侧已回读）

    // ---- 反量化 ----
    cudaEventRecord(e2);
    {
        const int threads = 256;
        const int64_t nvec = is_nvfp4 ? (n + 7) / 8 : (n + 3) / 4;
        const int blocks = (int)std::min<int64_t>((nvec + threads - 1) / threads, 4096);
        if (cfg.out_dtype == OutDtype::FP32)
            launch_dequant<float>(d.packed, d.scales, n, bs, is_nvfp4 ? 1 : 0, elem_fmt, gscale,
                                  reinterpret_cast<float*>(d.out), blocks, threads);
        else if (cfg.out_dtype == OutDtype::FP16)
            launch_dequant<__half>(d.packed, d.scales, n, bs, is_nvfp4 ? 1 : 0, elem_fmt, gscale,
                                   reinterpret_cast<__half*>(d.out), blocks, threads);
        else
            launch_dequant<__nv_bfloat16>(d.packed, d.scales, n, bs, is_nvfp4 ? 1 : 0, elem_fmt,
                                          gscale, reinterpret_cast<__nv_bfloat16*>(d.out), blocks,
                                          threads);
    }
    cudaEventRecord(e3);
    if (cudaEventSynchronize(e3) != cudaSuccess)
        return fail("sync");
    float qms = 0.0f, dms = 0.0f;
    cudaEventElapsedTime(&qms, e0, e1);
    cudaEventElapsedTime(&dms, e2, e3);
    out.quant_ms = (double)qms;
    out.dequant_ms = (double)dms;

    // 4) 取回反量化结果（按输出类型读回并转 fp32）
    if (cfg.out_dtype == OutDtype::FP32) {
        if (cudaMemcpy(out.dequant.data(), d.out, (size_t)n * 4, cudaMemcpyDeviceToHost) !=
            cudaSuccess)
            return fail("D2H out f32");
    } else if (cfg.out_dtype == OutDtype::FP16) {
        std::vector<__half> tmp((size_t)n);
        if (cudaMemcpy(tmp.data(), d.out, (size_t)n * 2, cudaMemcpyDeviceToHost) != cudaSuccess)
            return fail("D2H out f16");
        for (int64_t i = 0; i < n; i++)
            out.dequant[(size_t)i] = __half2float(tmp[(size_t)i]);
    } else {
        std::vector<__nv_bfloat16> tmp((size_t)n);
        if (cudaMemcpy(tmp.data(), d.out, (size_t)n * 2, cudaMemcpyDeviceToHost) != cudaSuccess)
            return fail("D2H out bf16");
        for (int64_t i = 0; i < n; i++)
            out.dequant[(size_t)i] = __bfloat162float(tmp[(size_t)i]);
    }

    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    cudaEventDestroy(e2);
    cudaEventDestroy(e3);
    return true;
}

bool run_gpu_dequant(const std::vector<uint8_t>& packed, const std::vector<uint8_t>& scales,
                     float gscale, int64_t n, int64_t bs, int is_nvfp4, int elem_fmt, OutDtype dt,
                     std::vector<float>& out, double* ms, std::string& err) {
    if (n <= 0) {
        err = "空张量";
        return false;
    }
    DevBufs d;
    cudaEvent_t a = nullptr, b = nullptr;
    auto fail = [&](const char* what) {
        err = std::string("CUDA 失败(") + what + "): " + cudaGetErrorString(cudaGetLastError());
        if (a)
            cudaEventDestroy(a);
        if (b)
            cudaEventDestroy(b);
        return false;
    };
    const int osz = (dt == OutDtype::FP32) ? 4 : 2;
    if (cudaMalloc(&d.packed, packed.size() + 16) != cudaSuccess)
        return fail("malloc packed");
    if (cudaMalloc(&d.scales, scales.size() + 1) != cudaSuccess)
        return fail("malloc scales");
    if (cudaMalloc(&d.out, (size_t)n * osz) != cudaSuccess)
        return fail("malloc out");
    if (cudaMemcpy(d.packed, packed.data(), packed.size(), cudaMemcpyHostToDevice) != cudaSuccess)
        return fail("H2D packed");
    if (cudaMemcpy(d.scales, scales.data(), scales.size(), cudaMemcpyHostToDevice) != cudaSuccess)
        return fail("H2D scales");

    cudaEventCreate(&a);
    cudaEventCreate(&b);
    cudaEventRecord(a);
    const int threads = 256;
    const int64_t nvec = is_nvfp4 ? (n + 7) / 8 : (n + 3) / 4;
    const int blocks = (int)std::min<int64_t>((nvec + threads - 1) / threads, 4096);
    if (dt == OutDtype::FP32)
        launch_dequant<float>(d.packed, d.scales, n, bs, is_nvfp4, elem_fmt, gscale,
                              reinterpret_cast<float*>(d.out), blocks, threads);
    else if (dt == OutDtype::FP16)
        launch_dequant<__half>(d.packed, d.scales, n, bs, is_nvfp4, elem_fmt, gscale,
                               reinterpret_cast<__half*>(d.out), blocks, threads);
    else
        launch_dequant<__nv_bfloat16>(d.packed, d.scales, n, bs, is_nvfp4, elem_fmt, gscale,
                                      reinterpret_cast<__nv_bfloat16*>(d.out), blocks, threads);
    cudaEventRecord(b);
    if (cudaEventSynchronize(b) != cudaSuccess)
        return fail("sync");
    float fms = 0.0f;
    cudaEventElapsedTime(&fms, a, b);
    *ms = (double)fms;

    out.resize((size_t)n);
    if (dt == OutDtype::FP32) {
        if (cudaMemcpy(out.data(), d.out, (size_t)n * 4, cudaMemcpyDeviceToHost) != cudaSuccess)
            return fail("D2H out f32");
    } else if (dt == OutDtype::FP16) {
        std::vector<__half> tmp((size_t)n);
        if (cudaMemcpy(tmp.data(), d.out, (size_t)n * 2, cudaMemcpyDeviceToHost) != cudaSuccess)
            return fail("D2H out f16");
        for (int64_t i = 0; i < n; i++)
            out[(size_t)i] = __half2float(tmp[(size_t)i]);
    } else {
        std::vector<__nv_bfloat16> tmp((size_t)n);
        if (cudaMemcpy(tmp.data(), d.out, (size_t)n * 2, cudaMemcpyDeviceToHost) != cudaSuccess)
            return fail("D2H out bf16");
        for (int64_t i = 0; i < n; i++)
            out[(size_t)i] = __bfloat162float(tmp[(size_t)i]);
    }
    cudaEventDestroy(a);
    cudaEventDestroy(b);
    return true;
}

} // namespace lowp
