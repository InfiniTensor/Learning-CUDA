// =============================================================================
//  quantize.cu -- 低精度量化流水线(GPU)
//
//  阶段 1 (amax)  : 计算每个缩放块 / 整个张量的最大绝对值
//  阶段 2 (scale) : 生成缩放因子
//                     MXFP8 / MXFP4 : 每块一个 E8M0(2 的整数次幂)
//                     NVFP4 : 全张量 FP32 全局缩放 + 每 16 元素一个 E4M3 局部缩放;
//                             scale_rule 为自适应时, 逐块在 4 / 6 两个候选里选
//  阶段 3 (quant) : 元素量化 + 按真实位宽打包(4bit 严格两元素/字节)
//
//  所有 kernel 只使用基础 CUDA,不依赖 Hopper/Blackwell 指令,也不依赖
//  FP8/FP4 Tensor Core 或任何 cuda_fp8/cuda_fp4 头文件,可在任意 sm_60+ GPU 上运行.
// =============================================================================
#include <algorithm>
#include <cstring>
#include <type_traits>
#include <vector>

#include "lpq/common.hpp"
#include "lpq/formats.hpp"
#include "lpq/pipeline.hpp"
#include "lpq/stats.cuh"
#include "lpq/vec_io.cuh"

namespace lpq {

// =============================================================================
//  DeviceBuffer
//
//  cudaMalloc/cudaFree 每次都要向驱动申请 / 归还一段地址空间并建页表, 单次开销就有
//  几百微秒; 而本项目一次量化调用里的 GPU 工作总共只有 1~2ms. 实测「每轮分配 4 个
//  固定大小的缓冲区再释放」在稳态下要 2.8ms -- 比整个 GPU 部分还贵. 因此这里加一层
//  进程内的按大小缓存: 释放掉的块留在池子里, 下一次同尺寸请求直接复用, 稳态下不再
//  触碰驱动. (同样的做法在 PyTorch / Paddle 里就是它们的 caching allocator.)
//
//  池子限制块数与总字节数, 避免非常规尺寸把显存一直占着; 超出上限的块照常 cudaFree
//  还给驱动. 语义上没有变化: 复用前缓冲区的内容同样是未定义的, 所有调用方本来就必须
//  自己初始化(见 run_quantize 里的 memset 与量化 kernel 的写入).
// =============================================================================
namespace {

constexpr size_t kPoolMaxBlocks = 8;
constexpr size_t kPoolMaxBytes = size_t(2) << 30;  // 2 GiB

struct PoolSlot {
    void* ptr;
    size_t bytes;
};

struct DevicePool {
    PoolSlot slots[kPoolMaxBlocks];
    size_t count = 0;
    size_t bytes = 0;
};

DevicePool& device_pool() {
    static DevicePool pool;
    return pool;
}

// 最适配优先: 取满足请求的最小块, 免得好不容易攒下的大块被小请求切碎
void* pool_alloc(size_t bytes, size_t& cap) {
    DevicePool& p = device_pool();
    size_t best = kPoolMaxBlocks;
    for (size_t i = 0; i < p.count; ++i) {
        if (p.slots[i].bytes >= bytes &&
            (best == kPoolMaxBlocks || p.slots[i].bytes < p.slots[best].bytes))
            best = i;
    }
    if (best != kPoolMaxBlocks) {
        void* ptr = p.slots[best].ptr;
        cap = p.slots[best].bytes;
        p.bytes -= p.slots[best].bytes;
        p.slots[best] = p.slots[--p.count];
        return ptr;
    }
    void* ptr = nullptr;
    LPQ_CUDA_CHECK(cudaMalloc(&ptr, bytes));
    cap = bytes;
    return ptr;
}

void pool_free(void* ptr, size_t bytes) {
    if (ptr == nullptr)
        return;
    DevicePool& p = device_pool();
    if (p.count < kPoolMaxBlocks && p.bytes + bytes <= kPoolMaxBytes) {
        p.slots[p.count].ptr = ptr;
        p.slots[p.count].bytes = bytes;
        ++p.count;
        p.bytes += bytes;
        return;
    }
    cudaFree(ptr);
}

}  // namespace

DeviceBuffer::DeviceBuffer(size_t bytes) : bytes_(bytes) {
    if (bytes_ > 0)
        ptr_ = pool_alloc(bytes_, cap_);
}

DeviceBuffer::~DeviceBuffer() {
    reset();
}

void DeviceBuffer::reset() {
    if (ptr_ != nullptr) {
        pool_free(ptr_, cap_);
        ptr_ = nullptr;
    }
    bytes_ = 0;
    cap_ = 0;
}

DeviceBuffer::DeviceBuffer(DeviceBuffer&& o) noexcept
    : ptr_(o.ptr_), bytes_(o.bytes_), cap_(o.cap_) {
    o.ptr_ = nullptr;
    o.bytes_ = 0;
    o.cap_ = 0;
}

DeviceBuffer& DeviceBuffer::operator=(DeviceBuffer&& o) noexcept {
    if (this != &o) {
        reset();
        ptr_ = o.ptr_;
        bytes_ = o.bytes_;
        cap_ = o.cap_;
        o.ptr_ = nullptr;
        o.bytes_ = 0;
        o.cap_ = 0;
    }
    return *this;
}

namespace {

constexpr int kThreads = 256;

// 量化 kernel 的占用率提示「每 SM 至少 N 个 block」.
// 这是寄存器预算与常驻 block 数之间的取舍: 256 线程的 block 在 64K 寄存器文件里,
// 6 个 block/SM 等于每线程只有 40 个寄存器, 带误差统计的热点 kernel 会被 ptxas
// 插进 spill; 放宽到 5 个能拿到 48 个, 4 个能拿到 64 个但占用率降到 2/3.
//
// 两个 kernel 的最优点并不一样, 因为 NVFP4 每个元素还多一份除法与 4bit 打包,
// 寄存器压力明显更大(MXFP8 在 N=5 就能完全不 spill, NVFP4 要放到 N=4). 都用
// 实测挑出来的: 8192^2 下 N=5 让 MXFP8 quant 从 0.953 降到 0.944 ms, N=4 让
// NVFP4 quant 从 0.988 降到 0.962 ms; 反过来配则两者都变差.
// 可用 -DLPQ_QUANT_MIN_BLOCKS / -DLPQ_QUANT_MIN_BLOCKS_NV 覆盖做扫描,
// 0 表示不限制占用率(交给 ptxas 自己选, 通常落在 4 个 block 上).
#ifndef LPQ_QUANT_MIN_BLOCKS
#define LPQ_QUANT_MIN_BLOCKS 5
#endif
#ifndef LPQ_QUANT_MIN_BLOCKS_NV
#define LPQ_QUANT_MIN_BLOCKS_NV 4
#endif
#if LPQ_QUANT_MIN_BLOCKS > 0
#define LPQ_QUANT_LAUNCH_BOUNDS __launch_bounds__(kThreads, LPQ_QUANT_MIN_BLOCKS)
#else
#define LPQ_QUANT_LAUNCH_BOUNDS __launch_bounds__(kThreads)
#endif
#if LPQ_QUANT_MIN_BLOCKS_NV > 0
#define LPQ_QUANT_LAUNCH_BOUNDS_NV __launch_bounds__(kThreads, LPQ_QUANT_MIN_BLOCKS_NV)
#else
#define LPQ_QUANT_LAUNCH_BOUNDS_NV __launch_bounds__(kThreads)
#endif

// -----------------------------------------------------------------------------
// 设备辅助
// -----------------------------------------------------------------------------

// 块内最大值归约(blockDim.x 必须是 2 的幂)
__device__ __forceinline__ void block_reduce_max(float& v, float* smem) {
    const int tid = threadIdx.x;
    smem[tid] = v;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s)
            smem[tid] = fmaxf(smem[tid], smem[tid + s]);
        __syncthreads();
    }
    v = smem[0];
}

// 元素下标 -> 缩放块下标;shift >= 0 表示 block_size 为 2 的幂
__device__ __forceinline__ int64_t block_index(int64_t i, int64_t block_size, int shift) {
    return (shift >= 0) ? (i >> shift) : (i / block_size);
}

// 2 的幂 -> 移位量;非 2 的幂返回 -1
int pow2_shift(int64_t v) {
    if (v <= 0)
        return -1;
    if ((v & (v - 1)) != 0)
        return -1;
    int s = 0;
    while ((int64_t(1) << s) < v)
        ++s;
    return s;
}

// -----------------------------------------------------------------------------
// 阶段 1:amax
// -----------------------------------------------------------------------------

// 单块平铺:整个张量(或整个张量共用一个缩放因子的情形)
template <class InT>
__global__ void k_tensor_amax(const InT* __restrict__ x, float* __restrict__ out, int64_t numel) {
    extern __shared__ float smem[];
    float v = 0.0f;
    const int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += stride) {
        v = fmaxf(v, fabsf(InTraits<InT>::to_float(x[i])));
    }
    block_reduce_max(v, smem);
    if (threadIdx.x == 0)
        lpq_atomic_max_f32(out, v);
}

// 每 warp 处理 32 个连续元素;block_size 为 2 的幂且 <= 32 时使用
template <class InT>
__global__ void k_block_amax_pow2(const InT* __restrict__ x, float* __restrict__ amax,
    float* __restrict__ tensor_amax, int64_t numel, int block_size) {
    const int lane = threadIdx.x & 31;
    const int64_t gid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t base = (gid >> 5) << 5;  // 本 warp 负责的元素区间起点
    const int64_t idx = base + lane;
    float v = (idx < numel) ? fabsf(InTraits<InT>::to_float(x[idx])) : 0.0f;
    for (int off = block_size >> 1; off > 0; off >>= 1) {
        v = fmaxf(v, LPQ_SHFL_XOR(v, off, block_size));
    }
    // 注意: LPQ_SHFL_XOR 用了 width = block_size, 因此每个 block_size 大小的
    // 子组各有一位组长持有本组的归约结果. tensor_amax 必须由**每个**组长贡献,
    // 否则当 block_size < 32 时一个 warp 里的第 2..(32/block_size) 个块会被漏掉,
    // NVFP4 的全张量缩放随之出错.
    if ((lane & (block_size - 1)) == 0) {
        const int64_t b = (base + lane) / block_size;
        if (b * block_size < numel) {
            amax[b] = v;
            if (tensor_amax != nullptr) {
                lpq_atomic_max_f32(tensor_amax, v);
            }
        }
    }
}

// 每线程处理 4 个连续元素(与融合量化 kernel 相同的数据映射).
// 同一个缩放块由 block_size/4 个相邻线程负责, 因此块内归约只需一次 width =
// block_size/4 的 shuffle. 相比上一版(每线程 1 个标量元素)指令数降到 1/4,
// 在 NVFP4 的默认配置下把这一阶段的带宽从 ~250 GB/s 提到接近显存峰值.
// 前置条件: block_size 为 2 的幂且 4 <= block_size <= 32(即 shift 在 [2,5]).
template <class InT>
__global__ void k_block_amax_pow2_vec4(const InT* __restrict__ x, float* __restrict__ amax,
    float* __restrict__ tensor_amax, int64_t numel, int shift, int log_lanes) {
    const int lane = threadIdx.x & 31;
    const int lanes_per_block = 1 << log_lanes;  // = block_size / 4
    const int64_t gid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t i0 = gid << 2;

    float v = 0.0f;
    if (i0 + 3 < numel) {
        float t[4];
        InTraits<InT>::load4(x + i0, t);
        v = fmaxf(fmaxf(fabsf(t[0]), fabsf(t[1])), fmaxf(fabsf(t[2]), fabsf(t[3])));
    } else if (i0 < numel) {  // 尾部不足 4 个元素
        for (int k = 0; k < 4 && i0 + k < numel; ++k)
            v = fmaxf(v, fabsf(InTraits<InT>::to_float(x[i0 + k])));
    }
    // 归约必须在整组上一致执行, 因此越界线程也参与(贡献 0, 对 max 无影响)
    for (int off = lanes_per_block >> 1; off > 0; off >>= 1) {
        v = fmaxf(v, LPQ_SHFL_XOR(v, off, lanes_per_block));
    }
    if ((lane & (lanes_per_block - 1)) == 0) {
        const int64_t b = i0 >> shift;  // 本组对应的块下标
        if ((b << shift) < numel) {
            amax[b] = v;
            if (tensor_amax != nullptr)
                lpq_atomic_max_f32(tensor_amax, v);
        }
    }
}

// 通用:每个 warp 负责一个 block(允许 block_size 任意大)
template <class InT>
__global__ void k_block_amax_generic(const InT* __restrict__ x, float* __restrict__ amax,
    float* __restrict__ tensor_amax, int64_t numel, int64_t block_size) {
    const int lane = threadIdx.x & 31;
    const int64_t warp = ((int64_t)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int64_t base = warp * block_size;
    if (base >= numel)
        return;
    const int64_t end = min(base + block_size, numel);
    float v = 0.0f;
    for (int64_t i = base + lane; i < end; i += 32) {
        v = fmaxf(v, fabsf(InTraits<InT>::to_float(x[i])));
    }
    for (int off = 16; off > 0; off >>= 1) {
        v = fmaxf(v, LPQ_SHFL_XOR(v, off, 32));
    }
    if (lane == 0) {
        amax[warp] = v;
        if (tensor_amax != nullptr)
            lpq_atomic_max_f32(tensor_amax, v);
    }
}

// -----------------------------------------------------------------------------
// 阶段 2:缩放因子
// -----------------------------------------------------------------------------

// MXFP8 / MXFP4:每块一个 E8M0 编码(两者的差别只在 emax_elem)
__global__ void k_mxfp8_scales(
    const float* __restrict__ amax, int64_t nb, int emax_elem, uint8_t* __restrict__ scales) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nb)
        return;
    const int e = mx_shared_exp(amax[i], emax_elem);
    scales[i] = e8m0_encode(e);
}

// NVFP4:由全张量 amax 得到 FP32 全局缩放及其倒数
__global__ void k_nvfp4_global_scale(const float* __restrict__ tensor_amax, float max_q,
    float max_s, float* __restrict__ gscale, float* __restrict__ ginv) {
    if (threadIdx.x != 0 || blockIdx.x != 0)
        return;
    const float gs = nvfp4_global_scale(tensor_amax[0], max_q, max_s);
    gscale[0] = gs;
    ginv[0] = f32_div_rn(1.0f, gs);
}

// NVFP4:每块一个 E4M3 局部缩放(恒用 RNE,与 NVIDIA 参考实现一致)
__global__ void k_nvfp4_block_scales(const float* __restrict__ amax, int64_t nb,
    const float* __restrict__ gscale, float max_q, uint8_t* __restrict__ scales) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nb)
        return;
    const float s = nvfp4_block_scale_f32(amax[i], gscale[0], max_q);
    scales[i] = (uint8_t)f32_to_fmt<FmtE4M3>(s);
}

// -----------------------------------------------------------------------------
// NVFP4 的 4/6 自适应块缩放:逐块在「6 候选」和「4 候选」里挑一个
//
// 一个缩放块交给一个线程:块内元素按下标顺序逐个量化-反量化, 并把误差按配置的准则
// 累加。顺序累加 + 单线程, 使得设备侧与 CPU 参考实现的累加次序完全一致, 于是选择
// 结果(进而 scales)也能逐位对拍。
//
// 「6 候选」把块内 amax 映射到 6.0, 「4 候选」把块缩放因子乘 1.5 从而映射到 4.0.
// 后者让块内最大值落在 E2M1 格距为 1 的 [2, 4] 区间而不是格距为 2 的 [4, 6] 区间,
// 可表示值分布更均匀; 代价是块内较小的值会被压得更小。选哪个由误差准则决定。
//
// 这个 kernel 只写 scales, 量化本身仍由 k_quant_nvfp4 完成。因此它单独存在时 4/6
// 路径要读 3 遍输入(amax / 选择 / 量化); 能融合时选择由 k_quant_nvfp4_46 顺带完成,
// 遍数回到 2, 这里只在 can_fuse_nvfp4_46 判定不满足时作为回落路径使用。
// -----------------------------------------------------------------------------
// 误差准则编码: 0 = abs_max(最大绝对误差), 1 = mae(平均绝对误差), 2 = mse(均方误差)
__device__ __forceinline__ float scale_rule_accumulate(int rule, float acc, float err) {
    if (rule == 0)
        return fmaxf(acc, fabsf(err));
    if (rule == 1)
        return acc + fabsf(err);
    return fma_rn(err, err, acc);
}

template <class InT>
__global__ void k_nvfp4_block_scales_46(const InT* __restrict__ x,
    const float* __restrict__ block_amax, const float* __restrict__ gscale, int64_t numel,
    int64_t block_size, int64_t num_blocks, float max_q, int rule, uint64_t seed, bool stoch,
    uint8_t* __restrict__ scales) {
    const int64_t stride = (int64_t)gridDim.x * blockDim.x;
    const float gs = gscale[0];
    const float gf = gscale[1];  // = 1 / global_scale, 与反量化逐位同一个因子
    for (int64_t b = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; b < num_blocks; b += stride) {
        const int64_t i0 = b * block_size;
        const int64_t n = (i0 + block_size <= numel) ? block_size : (numel - i0);
        // 两个候选的块缩放因子(E4M3 编码后就是反量化时读回显存的那个值)
        const float bs6_raw = nvfp4_block_scale_f32(block_amax[b], gs, max_q);
        const uint8_t code6 = (uint8_t)f32_to_fmt<FmtE4M3>(bs6_raw);
        const uint8_t code4 = (uint8_t)f32_to_fmt<FmtE4M3>(mul_rn(bs6_raw, kFourOverSixExpansion));
        const float bs6 = fmt_to_f32<FmtE4M3>(code6);
        const float bs4 = fmt_to_f32<FmtE4M3>(code4);

        float err6 = 0.0f, err4 = 0.0f;
        for (int64_t k = 0; k < n; ++k) {
            const float xv = InTraits<InT>::to_float(x[i0 + k]);
            const uint64_t r = stoch ? random_at(seed, (uint64_t)(i0 + k)) : 0ull;
            const float q6 = (bs6 > 0.0f) ? f32_div_rn(xv * gs, bs6) : 0.0f;
            const float q4 = (bs4 > 0.0f) ? f32_div_rn(xv * gs, bs4) : 0.0f;
            const float d6 =
                mul_rn(fmt_to_f32<FmtE2M1>((uint8_t)(f32_to_fmt<FmtE2M1>(q6, r, stoch) & 0xFu)),
                    mul_rn(bs6, gf));
            const float d4 =
                mul_rn(fmt_to_f32<FmtE2M1>((uint8_t)(f32_to_fmt<FmtE2M1>(q4, r, stoch) & 0xFu)),
                    mul_rn(bs4, gf));
            err6 = scale_rule_accumulate(rule, err6, d6 - xv);
            err4 = scale_rule_accumulate(rule, err4, d4 - xv);
        }
        // 平手时保留 6 候选(与参考实现的严格小于一致)
        scales[b] = (err4 < err6) ? code4 : code6;
    }
}

// -----------------------------------------------------------------------------
// 阶段 3:量化 + 打包
// -----------------------------------------------------------------------------

// ---------------- MXFP8 ----------------
// ---------------- MXFP8 ----------------
// Sink 是误差统计的「汇点」: 量化 kernel 每算出一个元素, 就把
// (原始值 ref, 反量化值 test) 喂给它. NullSink 在编译期完全消失, 所以未开启统计时
// 这些 kernel 与手工删掉这几行逐指令相同. 详见 include/lpq/stats.cuh.
template <class InT, class EFmt, class Sink, bool STOCH>
LPQ_QUANT_LAUNCH_BOUNDS __global__ void k_quant_mxfp8(const InT* __restrict__ x,
    const uint8_t* __restrict__ scales, int64_t numel, int64_t block_size, int shift,
    int64_t num_blocks, uint64_t seed, uint8_t* __restrict__ out, double* __restrict__ stats_acc,
    float* __restrict__ stats_max) {
    const int64_t t0 = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride = (int64_t)gridDim.x * blockDim.x;
    const int64_t nvec = numel >> 2;
    Sink sink;
    for (int64_t t = t0; t < nvec; t += stride) {
        const int64_t i0 = t << 2;
        float v[4];
        InTraits<InT>::load4(x + i0, v);
        float inv[4];
        float sdec[4];
        // 4 个连续元素是否同块: 只比较首尾两个块号即可(块号单调不减), 而块号本身
        // 走 block_index(2 的幂时就是一条移位). 原先这里调 same_block(), 里面是
        // 两次 i64 整数除法 -- 在访存瓶颈的量化 kernel 里, 那一项比整个累加器还贵.
        const int64_t b0 = (num_blocks == 1) ? 0 : block_index(i0, block_size, shift);
        const int64_t b3 = (num_blocks == 1) ? 0 : block_index(i0 + 3, block_size, shift);
        if (b0 == b3) {
            const uint8_t c0 = scales[b0];
            const float s = e8m0_to_f32(c0);
            const float r = e8m0_rcp(c0);
            inv[0] = inv[1] = inv[2] = inv[3] = r;
            sdec[0] = sdec[1] = sdec[2] = sdec[3] = s;
        } else {
            for (int k = 0; k < 4; ++k) {
                const uint8_t c = scales[block_index(i0 + k, block_size, shift)];
                sdec[k] = e8m0_to_f32(c);
                inv[k] = e8m0_rcp(c);
            }
        }
        if constexpr (ElemTraits<EFmt>::bits == 4) {
            // 4bit 元素: 4 个元素 = 2 字节, 下标偶数的在低半字节
            uint32_t packed = 0;
            for (int k = 0; k < 4; ++k) {
                const uint64_t r = STOCH ? random_at(seed, (uint64_t)(i0 + k)) : 0ull;
                packed |= (f32_to_fmt<EFmt>(v[k] * inv[k], r, STOCH) & 0xFu) << (4 * k);
            }
            *reinterpret_cast<uint16_t*>(out + (i0 >> 1)) = (uint16_t)packed;
            float dec[4];
            fmt_x4_4bit_to_f32<EFmt>((uint16_t)packed, dec);
            for (int k = 0; k < 4; ++k)
                sink.add(v[k], mul_rn(dec[k], sdec[k]));
        } else {
            uint32_t packed = 0;
            for (int k = 0; k < 4; ++k) {
                const uint64_t r = STOCH ? random_at(seed, (uint64_t)(i0 + k)) : 0ull;
                packed |= (f32_to_fmt<EFmt>(v[k] * inv[k], r, STOCH) & 0xFFu) << (8 * k);
                // 与 k_dequant_mxfp8 里逐位同一个表达式: e4m3/e5m2(code) * e8m0(scale)
            }
            *reinterpret_cast<uint32_t*>(out + i0) = packed;
            // 统计的 test 必须与「先落盘、再读回」位同一: 先打包成 packed,
            // 再用与 k_dequant_mxfp8 完全相同的译码(fmt_x4_to_f32). sm_89+ 上走硬件.
            float dec[4];
            fmt_x4_to_f32<EFmt>(packed, dec);
            for (int k = 0; k < 4; ++k)
                sink.add(v[k], mul_rn(dec[k], sdec[k]));
        }
    }
    // 尾部不足 4 个元素的收尾(最多 3 个),由 0 号线程串行处理
    if (t0 == 0) {
        for (int64_t i = nvec * 4; i < numel; ++i) {
            const int64_t b = (num_blocks == 1) ? 0 : block_index(i, block_size, shift);
            const uint8_t scode = scales[b];
            const float s = e8m0_to_f32(scode);
            const float inv = e8m0_rcp(scode);  // 与主循环一致: 乘倒数, 不是除法
            const float xv = InTraits<InT>::to_float(x[i]);
            const uint64_t r = STOCH ? random_at(seed, (uint64_t)i) : 0ull;
            const uint8_t c = (uint8_t)(f32_to_fmt<EFmt>(xv * inv, r, STOCH) &
                                        (uint32_t)((1u << ElemTraits<EFmt>::bits) - 1u));
            if constexpr (ElemTraits<EFmt>::bits == 4) {
                // 偶下标写整个字节(顺带把高半字节清零), 奇下标只填高半字节
                const int64_t byte = i >> 1;
                if ((i & 1) == 0)
                    out[byte] = c;
                else
                    out[byte] = (uint8_t)((out[byte] & 0x0Fu) | (uint8_t)(c << 4));
            } else {
                out[i] = c;
            }
            sink.add(xv, mul_rn(fmt_to_f32<EFmt>(c), s));
        }
    }
    stats_epilogue(sink, stats_acc, stats_max);
}

// ---------------- NVFP4 ----------------
// 反量化公式(与 k_dequant_nvfp4 一致): x = e2m1(code) * (e4m3(block_scale) * 1/global)
// 统计里的 test 必须逐位复刻这一表达式: 结合顺序要一致(先 (bs * gf) 再乘 e2m1),
// 而且两次乘法都不能被后续的 test - ref 吸收成 FFMA -- 所以走 mul_rn, 详见 stats.cuh.
template <class InT, class OutT, class Sink, bool STOCH>
LPQ_QUANT_LAUNCH_BOUNDS_NV __global__ void k_quant_nvfp4(const InT* __restrict__ x,
    const uint8_t* __restrict__ scales, const float* __restrict__ gscale, int64_t numel,
    int64_t block_size, int shift, int64_t num_blocks, uint64_t seed, uint8_t* __restrict__ out,
    double* __restrict__ stats_acc, float* __restrict__ stats_max) {
    const int64_t t0 = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride = (int64_t)gridDim.x * blockDim.x;
    const int64_t nvec = numel >> 2;
    const float gs = gscale[0];
    const float gf = gscale[1];  // = 1 / global_scale, 反量化用的就是它
    Sink sink;
    for (int64_t t = t0; t < nvec; t += stride) {
        const int64_t i0 = t << 2;
        float v[4];
        InTraits<InT>::load4(x + i0, v);
        float bs[4];
        // 同 MXFP8: 首尾块号相等即 4 个元素同块, 避开 same_block() 里的 i64 除法
        const int64_t b0 = (num_blocks == 1) ? 0 : block_index(i0, block_size, shift);
        const int64_t b3 = (num_blocks == 1) ? 0 : block_index(i0 + 3, block_size, shift);
        if (b0 == b3) {
            const float s = fmt_to_f32<FmtE4M3>(scales[b0]);
            bs[0] = bs[1] = bs[2] = bs[3] = s;
        } else {
            for (int k = 0; k < 4; ++k) {
                bs[k] = fmt_to_f32<FmtE4M3>(scales[block_index(i0 + k, block_size, shift)]);
            }
        }
        uint32_t packed = 0;
        for (int k = 0; k < 4; ++k) {
            const float q = (bs[k] > 0.0f) ? f32_div_rn(v[k] * gs, bs[k]) : 0.0f;
            const uint64_t r = STOCH ? random_at(seed, (uint64_t)(i0 + k)) : 0ull;
            const uint32_t c = f32_to_fmt<FmtE2M1>(q, r, STOCH) & 0xFu;
            packed |= c << (4 * k);  // 低半字节在前
            sink.add(v[k], mul_rn(fmt_to_f32<FmtE2M1>((uint8_t)c), mul_rn(bs[k], gf)));
        }
        *reinterpret_cast<uint16_t*>(out + (i0 >> 1)) = (uint16_t)packed;
    }
    if (t0 == 0) {
        for (int64_t i = nvec * 4; i < numel; ++i) {
            const int64_t b = (num_blocks == 1) ? 0 : block_index(i, block_size, shift);
            const float s = fmt_to_f32<FmtE4M3>(scales[b]);
            const float xv = InTraits<InT>::to_float(x[i]);
            const float q = (s > 0.0f) ? f32_div_rn(xv * gs, s) : 0.0f;
            const uint64_t r = STOCH ? random_at(seed, (uint64_t)i) : 0ull;
            const uint8_t c = (uint8_t)(f32_to_fmt<FmtE2M1>(q, r, STOCH) & 0xFu);
            const int64_t byte = i >> 1;
            if ((i & 1) == 0) {
                out[byte] = c;
            } else {
                out[byte] = (uint8_t)((out[byte] & 0x0Fu) | (uint8_t)(c << 4));
            }
            sink.add(xv, mul_rn(fmt_to_f32<FmtE2M1>(c), mul_rn(s, gf)));
        }
    }
    stats_epilogue(sink, stats_acc, stats_max);
}

// ---------------- 融合版 NVFP4 + 4/6(把「选缩放因子」并进量化 kernel) ----------------
// 未融合时 4/6 路径要比标准 NVFP4 多读一遍输入: 独立的 k_nvfp4_block_scales_46 为了
// 逐块比较两个候选, 必须把块内元素再扫一遍。这里把选择并进量化 kernel: 块缩放因子所需的
// 两个输入(块 amax、全张量缩放)在上一遍已经定稿, 不依赖本遍的量化结果。
//
// 线程到数据的映射是「一个线程负责一整个缩放块」: 块内 block_size 个元素连续, 依次读入、
// 量化-反量化、按块内下标顺序累加误差。于是
//   * 不需要 shuffle 归约 -- 块内元素本来就在同一个线程的寄存器里;
//   * 不存在「组长累加、组内其余 lane 空转」-- 每个 lane 都在算自己那个块的判定;
//   * 判定所需的累加次序天然就是块内下标的自然顺序, 与 CPU 参考实现、独立选择 kernel
//     逐位相同(浮点加法与 fma 都不满足结合律, 次序一致是逐位一致的关键)。
// 输出侧也顺带受益: 一个块的低精度数据由本线程一次写成一个 uint32/uint16, 不再是每线程
// 2 字节的碎写。
//
// 块大小(块内元素数)必须是编译期常量才能完全展开、不进 local memory, 因此 LOG_BLOCK
// 是模板参数, 由启动包装按 block_size = 4 / 8 / 16 / 32 特化出四个实例。
// 前置条件由 can_fuse_nvfp4_46 保证: block_size ∈ {4,8,16,32}、块完整
// (numel 能被 block_size 整除)、num_blocks > 1; 不满足时回落原来的三遍路径。
template <class InT, class OutT, class Sink, bool STOCH, int LOG_BLOCK>
LPQ_QUANT_LAUNCH_BOUNDS_NV __global__ void k_quant_nvfp4_46(const InT* __restrict__ x,
    const float* __restrict__ block_amax, const float* __restrict__ gscale, int64_t numel, int rule,
    float max_q, uint64_t seed, uint8_t* __restrict__ scales, uint8_t* __restrict__ out,
    double* __restrict__ stats_acc, float* __restrict__ stats_max) {
    constexpr int kBlock = 1 << LOG_BLOCK;                // 每个缩放块的元素数
    constexpr int kChunk = kBlock >> 2;                   // 本线程读入的 float4 个数
    constexpr int kWord = kBlock >= 8 ? kBlock >> 3 : 1;  // 每个候选占的 uint32 个数
    const int64_t num_blocks = numel >> LOG_BLOCK;
    const int64_t stride = (int64_t)gridDim.x * blockDim.x;
    const float gs = gscale[0];
    const float gf = gscale[1];  // = 1 / global_scale, 反量化用的就是它
    Sink sink;
    for (int64_t b = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; b < num_blocks; b += stride) {
        const int64_t i0 = b << LOG_BLOCK;
        // 两个候选的块缩放因子: E4M3 编码后就是反量化时读回显存的那个值
        const float raw = nvfp4_block_scale_f32(block_amax[b], gs, max_q);
        const uint8_t code6 = (uint8_t)f32_to_fmt<FmtE4M3>(raw);
        const uint8_t code4 = (uint8_t)f32_to_fmt<FmtE4M3>(mul_rn(raw, kFourOverSixExpansion));
        const float bs6 = fmt_to_f32<FmtE4M3>(code6);
        const float bs4 = fmt_to_f32<FmtE4M3>(code4);
        const float d6 = mul_rn(bs6, gf), d4 = mul_rn(bs4, gf);

        // 第一遍: 逐元素算两个候选的量化码与 (d - x), 按块内下标顺序累加
        float err6 = 0.0f, err4 = 0.0f;
        uint32_t pk6[kWord] = {}, pk4[kWord] = {};
#pragma unroll
        for (int c = 0; c < kChunk; ++c) {
            float v[4];
            InTraits<InT>::load4(x + i0 + 4 * c, v);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int k = 4 * c + j;
                const float xg = mul_rn(v[j], gs);
                const uint64_t r = STOCH ? random_at(seed, (uint64_t)(i0 + k)) : 0ull;
                const float q6 = (bs6 > 0.0f) ? f32_div_rn(xg, bs6) : 0.0f;
                const float q4 = (bs4 > 0.0f) ? f32_div_rn(xg, bs4) : 0.0f;
                const uint32_t c6 = f32_to_fmt<FmtE2M1>(q6, r, STOCH) & 0xFu;
                const uint32_t c4 = f32_to_fmt<FmtE2M1>(q4, r, STOCH) & 0xFu;
                pk6[k >> 3] |= c6 << (4 * (k & 7));
                pk4[k >> 3] |= c4 << (4 * (k & 7));
                err6 = scale_rule_accumulate(
                    rule, err6, mul_rn(fmt_to_f32<FmtE2M1>((uint8_t)c6), d6) - v[j]);
                err4 = scale_rule_accumulate(
                    rule, err4, mul_rn(fmt_to_f32<FmtE2M1>((uint8_t)c4), d4) - v[j]);
            }
        }
        // 平手时保留 6 候选(与参考实现的严格小于一致)
        const bool take4 = err4 < err6;
        scales[b] = take4 ? code4 : code6;
        const uint32_t* pk = take4 ? pk4 : pk6;
        const float d = take4 ? d4 : d6;
        const int64_t ob = i0 >> 1;  // 每 2 个元素 1 字节
        if constexpr (kWord == 1) {
            *reinterpret_cast<uint16_t*>(out + ob) = (uint16_t)pk[0];
        } else {
#pragma unroll
            for (int w = 0; w < kWord; ++w)
                *reinterpret_cast<uint32_t*>(out + ob + 4 * w) = pk[w];
        }
        // 第二遍只做误差统计: 它要的是 (原始值, 选中的反量化值), 而原始值刚从这里读过,
        // 二次读命中 L1, 比把整个块的原始值一直挂在寄存器里更省寄存器。未开启统计时
        // NullSink 是空操作, 整个循环会被编译器删掉。
#pragma unroll
        for (int c = 0; c < kChunk; ++c) {
            float v[4];
            InTraits<InT>::load4(x + i0 + 4 * c, v);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int k = 4 * c + j;
                const uint32_t code = (pk[k >> 3] >> (4 * (k & 7))) & 0xFu;
                sink.add(v[j], mul_rn(fmt_to_f32<FmtE2M1>((uint8_t)code), d));
            }
        }
    }
    stats_epilogue(sink, stats_acc, stats_max);
}

// ---------------- 融合版 MXFP8(block 模式, 单遍) ----------------
// 前置条件: block_size 为 2 的幂且 4 <= block_size <= 128.
//
// MXFP8 的缩放因子完全由本块内的 amax 决定, 因此「块内 amax -> E8M0 -> 量化 ->
// 打包」可以在一次访存内完成: 输入 x 只需从显存读一遍, 也不再需要独立的 amax
// kernel 与 scale kernel. 这消除了原三阶段实现中一次完整的输入重读.
//
// 线程到数据的映射沿用「每线程 4 个连续元素」: 于是同一缩放块恰好由
// block_size/4 个相邻线程负责, 块内归约可以用一次 LPQ_SHFL_XOR
// (width = block_size/4) 在寄存器内完成, 不需要共享内存, 也不需要第二次访存.
//
// 注意越界线程只置谓词而不提前 return: 收尾的 stats_epilogue 里有
// __syncthreads(), 一旦有线程提前退出, 整个 block 都会挂死.
template <class InT, class EFmt, class Sink, bool STOCH>
LPQ_QUANT_LAUNCH_BOUNDS __global__ void k_quant_mxfp8_block_fused(const InT* __restrict__ x,
    int64_t numel, int shift, uint64_t seed, uint8_t* __restrict__ scales,
    uint8_t* __restrict__ out, double* __restrict__ stats_acc, float* __restrict__ stats_max) {
    const int lane = threadIdx.x & 31;
    const int lanes_per_block = 1 << (shift - 2);  // = block_size / 4
    const int64_t nvec = (numel + 3) >> 2;
    const int64_t stride = (int64_t)gridDim.x * blockDim.x;
    // 循环边界按 32 对齐, 于是 warp 内所有 lane 的迭代次数完全相同 -- __shfl_xor_sync
    // 要求整个 warp 收敛, 边界不齐会让越界 lane 提前退出而触发未定义行为.
    const int64_t vmax = ((nvec + 31) >> 5) << 5;
    const int64_t warp0 = (int64_t)blockIdx.x * blockDim.x + (threadIdx.x & ~31);

    Sink sink;
    for (int64_t w = warp0; w < vmax; w += stride) {
        const int64_t i0 = (w + lane) << 2;
        const bool live = (i0 < numel);
        float v[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        if (i0 + 3 < numel) {
            InTraits<InT>::load4(x + i0, v);
        } else if (live) {  // 尾部不足 4 个元素
            for (int k = 0; k < 4 && i0 + k < numel; ++k)
                v[k] = InTraits<InT>::to_float(x[i0 + k]);
        }
        float m = fmaxf(fmaxf(fabsf(v[0]), fabsf(v[1])), fmaxf(fabsf(v[2]), fabsf(v[3])));
        // 归约必须在整个 warp 上一致执行, 因此越界线程也参与(贡献 0, 对 max 无影响)
        for (int off = lanes_per_block >> 1; off > 0; off >>= 1) {
            m = fmaxf(m, LPQ_SHFL_XOR(m, off, lanes_per_block));
        }
        if (!live)
            continue;  // 仍然保持在 warp 收敛的循环里

        const uint8_t sc = e8m0_encode(mx_shared_exp(m, EFmt::emax));
        if ((lane & (lanes_per_block - 1)) == 0)
            scales[i0 >> shift] = sc;        // 组长的块索引
        const float sdec = e8m0_to_f32(sc);  // 反量化时读回显存的就是这个值
        const float inv = e8m0_rcp(sc);
        if (i0 + 3 < numel) {
            if constexpr (ElemTraits<EFmt>::bits == 4) {
                uint32_t packed = 0;
                for (int k = 0; k < 4; ++k) {
                    const uint64_t r = STOCH ? random_at(seed, (uint64_t)(i0 + k)) : 0ull;
                    packed |= (f32_to_fmt<EFmt>(v[k] * inv, r, STOCH) & 0xFu) << (4 * k);
                }
                *reinterpret_cast<uint16_t*>(out + (i0 >> 1)) = (uint16_t)packed;
                float dec[4];
                fmt_x4_4bit_to_f32<EFmt>((uint16_t)packed, dec);
                for (int k = 0; k < 4; ++k)
                    sink.add(v[k], mul_rn(dec[k], sdec));
            } else {
                uint32_t packed = 0;
                for (int k = 0; k < 4; ++k) {
                    const uint64_t r = STOCH ? random_at(seed, (uint64_t)(i0 + k)) : 0ull;
                    packed |= (f32_to_fmt<EFmt>(v[k] * inv, r, STOCH) & 0xFFu) << (8 * k);
                }
                *reinterpret_cast<uint32_t*>(out + i0) = packed;
                float dec[4];
                fmt_x4_to_f32<EFmt>(packed, dec);
                for (int k = 0; k < 4; ++k)
                    sink.add(v[k], mul_rn(dec[k], sdec));
            }
        } else {
            for (int k = 0; k < 4 && i0 + k < numel; ++k) {
                const uint64_t r = STOCH ? random_at(seed, (uint64_t)(i0 + k)) : 0ull;
                const uint8_t c = (uint8_t)(f32_to_fmt<EFmt>(v[k] * inv, r, STOCH) &
                                            (uint32_t)((1u << ElemTraits<EFmt>::bits) - 1u));
                if constexpr (ElemTraits<EFmt>::bits == 4) {
                    // 偶下标写整个字节(顺带把高半字节清零), 奇下标只填高半字节
                    const int64_t byte = (i0 + k) >> 1;
                    if (((i0 + k) & 1) == 0)
                        out[byte] = c;
                    else
                        out[byte] = (uint8_t)((out[byte] & 0x0Fu) | (uint8_t)(c << 4));
                } else {
                    out[i0 + k] = c;
                }
                sink.add(v[k], mul_rn(fmt_to_f32<EFmt>(c), sdec));
            }
        }
    }
    stats_epilogue(sink, stats_acc, stats_max);
}

// -----------------------------------------------------------------------------
// 主机侧启动封装
// -----------------------------------------------------------------------------

enum class SinkKind { None, Fast, Strict };

// 把「是否统计 / 用哪种累加器」两个运行期入参映射成编译期汇点类型
unsigned int grid_for(int64_t work, int threads) {
    int64_t g = (work + threads - 1) / threads;
    if (g < 1)
        g = 1;
    if (g > 2147483647LL)
        g = 2147483647LL;
    return (unsigned int)g;
}

// 开启误差统计时, 每个 block 收尾都要做一次树形归约 + 4 次 fp64 atomicAdd.
// 量化 kernel 原本是「一个线程处理 4 个元素」, 8192^2 下就是 65536 个 block --
// 65536 x 4 次 atomicAdd(double) 全部撞在同样几个地址上, 实测光这一项就要 2.3ms,
// 比整个 metrics kernel 还贵. 所以带统计时把块数压到与 metrics kernel 同量级,
// 改用网格跨步循环: 同样的线程数, 块数少了 32 倍, 收尾成本就被摊掉了.
// 上限本身又与每条线程的注册器压力有关, 因此按「哪条量化路径」分开标定。
// 8192^2, 每个配置 50 次取最小(ms), 两轮取 min:
//                      2048     4096     8192
//   MXFP8  RNE        0.870    0.858    0.847   <-- 压力最小, 上限越高越好
//   MXFP8  SR         0.928    0.943    0.977   <-- SR 多一份随机数运算, 反而要压低
//   NVFP4  RNE        0.733    0.741      -     <-- 每元素多一次除法 + 4bit 打包
//   NVFP4  SR         0.978    0.998      -         驻留块数最低, 收尾占比最高
//   (上限过高的后果: 32768 -> 1.02, 65536 -> 1.93ms -- 4*块数次 fp64 atomicAdd
//    全撞在同一组地址上, 被串行化。)
// 因此: MXFP8 的 RNE 路径取 8192, 其余三条都取 2048。
// -DLPQ_STATS_MAX_BLOCKS / *_SR / *_NV 可覆盖重扫。
#ifndef LPQ_STATS_MAX_BLOCKS
#define LPQ_STATS_MAX_BLOCKS 8192
#endif
#ifndef LPQ_STATS_MAX_BLOCKS_SR
#define LPQ_STATS_MAX_BLOCKS_SR 2048
#endif
#ifndef LPQ_STATS_MAX_BLOCKS_NV
#define LPQ_STATS_MAX_BLOCKS_NV 2048
#endif
constexpr int64_t kStatsMaxBlocks = LPQ_STATS_MAX_BLOCKS;
constexpr int64_t kStatsMaxBlocksSr = LPQ_STATS_MAX_BLOCKS_SR;
constexpr int64_t kStatsMaxBlocksNv = LPQ_STATS_MAX_BLOCKS_NV;

unsigned int grid_for_quant(int64_t nvec, SinkKind sk, int64_t stats_cap) {
    int64_t g = (nvec + kThreads - 1) / kThreads;
    if (g < 1)
        g = 1;
    if (sk != SinkKind::None && g > stats_cap)
        g = stats_cap;
    if (g > 2147483647LL)
        g = 2147483647LL;
    return (unsigned int)g;
}

template <class InT>
void launch_amax(const InT* x, int64_t numel, const QuantConfig& cfg, int64_t block_size,
    int64_t num_blocks, float* d_block_amax, float* d_tensor_amax, cudaStream_t stream) {
    if (cfg.scale_mode == ScaleMode::Tensor || num_blocks == 1) {
        const int threads = kThreads;
        const int smem = threads * (int)sizeof(float);
        int64_t want = (numel + threads - 1) / threads;
        unsigned int grid = (unsigned int)std::min<int64_t>(want, 4096);
        k_tensor_amax<InT><<<grid, threads, smem, stream>>>(x, d_block_amax, numel);
        LPQ_CUDA_CHECK_KERNEL();
        return;
    }
    const int shift = pow2_shift(block_size);
    if (shift >= 2 && shift <= 5) {
        // block_size in {4,8,16,32}: 每线程 4 个元素, 每组 block_size/4 个线程
        const int64_t vecs = (numel + 3) / 4;
        k_block_amax_pow2_vec4<InT><<<grid_for(vecs, kThreads), kThreads, 0, stream>>>(
            x, d_block_amax, d_tensor_amax, numel, shift, shift - 2);
    } else if (shift >= 0 && shift <= 5) {
        const int64_t warps = (numel + 31) / 32;
        k_block_amax_pow2<InT><<<grid_for(warps * 32, kThreads), kThreads, 0, stream>>>(
            x, d_block_amax, d_tensor_amax, numel, (int)block_size);
    } else {
        const int64_t warps = num_blocks;
        k_block_amax_generic<InT><<<grid_for(warps * 32, kThreads), kThreads, 0, stream>>>(
            x, d_block_amax, d_tensor_amax, numel, block_size);
    }
    LPQ_CUDA_CHECK_KERNEL();
}

// -----------------------------------------------------------------------------
//  (输出类型, 统计模式) -> 编译期实例
//
//  输出类型决定误差统计里要做哪种往返舍入(反量化结果会先舍入到 output_type 才
//  落盘); 统计模式决定用哪种累加器. 两个选择都是运行期的, 但必须展开成编译期实例
//  才能让 NullSink 真正消失、让舍入不落进内层循环的分支里.
//  泛型 lambda + decltype 让这 3x3 的展开只写一遍, 三个 kernel 共用.
// -----------------------------------------------------------------------------
SinkKind sink_kind_for(bool want_stats, AccumMode mode) {
    if (!want_stats)
        return SinkKind::None;
    return (mode == AccumMode::Fast) ? SinkKind::Fast : SinkKind::Strict;
}

template <class F>
void dispatch_sink_out(OutputType ot, SinkKind sk, F&& f) {
    switch (sk) {
        case SinkKind::None:
            switch (ot) {
                case OutputType::FP16:
                    f(NullSink<__half>{});
                    break;
                case OutputType::BF16:
                    f(NullSink<__nv_bfloat16>{});
                    break;
                default:
                    f(NullSink<float>{});
                    break;
            }
            break;
        case SinkKind::Fast:
            switch (ot) {
                case OutputType::FP16:
                    f(FastSink<__half>{});
                    break;
                case OutputType::BF16:
                    f(FastSink<__nv_bfloat16>{});
                    break;
                default:
                    f(FastSink<float>{});
                    break;
            }
            break;
        default:
            switch (ot) {
                case OutputType::FP16:
                    f(StrictSink<__half>{});
                    break;
                case OutputType::BF16:
                    f(StrictSink<__nv_bfloat16>{});
                    break;
                default:
                    f(StrictSink<float>{});
                    break;
            }
            break;
    }
}

// 随机舍入开关也提到编译期. 运行时分支会让 f32_to_fmt 里那套 SR 逻辑与 RNE 逻辑
// 同时常驻在内层循环: 实测 FastSink 的三个 kernel 因此从"零 spill"变成 48B spill
// (lpq 给 ptxas 的 40 寄存器上限), 量化阶段平白慢 6%. 拆成两个编译期版本后, 热点
// 指令流与寄存器需求都回到只有 RNE 时的水平; 代价是每颗 kernel 多一份实例, 只影响
// 编译时间, 不影响运行.
template <class F>
void dispatch_stoch(bool stoch, F&& f) {
    if (stoch)
        f(std::true_type{});
    else
        f(std::false_type{});
}

template <class InT, class EFmt>
void launch_quant_mxfp8(const InT* x, int64_t numel, const QuantConfig& cfg, int64_t block_size,
    int64_t num_blocks, const uint8_t* d_scales, uint8_t* d_out, cudaStream_t stream,
    double* d_stats_acc, float* d_stats_max, SinkKind sk) {
    const int shift = pow2_shift(block_size);
    const int64_t nvec = (numel + 3) / 4;
    const bool stoch = cfg.rounding == Rounding::Stochastic;
    const unsigned int grid = grid_for_quant(nvec, sk, stoch ? kStatsMaxBlocksSr : kStatsMaxBlocks);
    const uint64_t seed = cfg.seed;
    dispatch_sink_out(cfg.output_type, sk, [&](auto tag) {
        using Sink = decltype(tag);
        dispatch_stoch(stoch, [&](auto sct) {
            constexpr bool kS = decltype(sct)::value;
            k_quant_mxfp8<InT, EFmt, Sink, kS><<<grid, kThreads, 0, stream>>>(x, d_scales, numel,
                block_size, shift, num_blocks, seed, d_out, d_stats_acc, d_stats_max);
        });
    });
    LPQ_CUDA_CHECK_KERNEL();
}

template <class InT>
void launch_quant_nvfp4(const InT* x, int64_t numel, const QuantConfig& cfg, int64_t block_size,
    int64_t num_blocks, const uint8_t* d_scales, const float* d_gscale, uint8_t* d_out,
    cudaStream_t stream, double* d_stats_acc, float* d_stats_max, SinkKind sk) {
    const int shift = pow2_shift(block_size);
    const int64_t nvec = (numel + 3) / 4;
    const unsigned int grid = grid_for_quant(nvec, sk, kStatsMaxBlocksNv);
    const uint64_t seed = cfg.seed;
    const bool stoch = cfg.rounding == Rounding::Stochastic;
    dispatch_sink_out(cfg.output_type, sk, [&](auto tag) {
        using Sink = decltype(tag);
        using OutT = typename Sink::out_type;
        dispatch_stoch(stoch, [&](auto sct) {
            constexpr bool kS = decltype(sct)::value;
            k_quant_nvfp4<InT, OutT, Sink, kS><<<grid, kThreads, 0, stream>>>(x, d_scales, d_gscale,
                numel, block_size, shift, num_blocks, seed, d_out, d_stats_acc, d_stats_max);
        });
    });
    LPQ_CUDA_CHECK_KERNEL();
}

template <class InT, class EFmt>
void launch_quant_mxfp8_fused(const InT* x, int64_t numel, const QuantConfig& cfg, int shift,
    uint8_t* d_scales, uint8_t* d_out, cudaStream_t stream, double* d_stats_acc, float* d_stats_max,
    SinkKind sk) {
    const int64_t nvec = (numel + 3) / 4;
    const bool stoch = cfg.rounding == Rounding::Stochastic;
    const unsigned int grid = grid_for_quant(nvec, sk, stoch ? kStatsMaxBlocksSr : kStatsMaxBlocks);
    const uint64_t seed = cfg.seed;
    dispatch_sink_out(cfg.output_type, sk, [&](auto tag) {
        using Sink = decltype(tag);
        dispatch_stoch(stoch, [&](auto sct) {
            constexpr bool kS = decltype(sct)::value;
            k_quant_mxfp8_block_fused<InT, EFmt, Sink, kS><<<grid, kThreads, 0, stream>>>(
                x, numel, shift, seed, d_scales, d_out, d_stats_acc, d_stats_max);
        });
    });
    LPQ_CUDA_CHECK_KERNEL();
}

template <class InT>
QuantizedDevice run_quantize(const InT* x, int64_t numel, const QuantConfig& cfg,
    cudaStream_t stream, double* d_stats_acc, float* d_stats_max, SinkKind sk);

// NVFP4 两级缩放里的 max_q / max_s, 由 scale_rule 决定(取值理由见 formats.hpp):
//   static_6         : max_q = 6,   max_s = 448  <- 标准 NVFP4
//   static_4         : max_q = 4,   max_s = 448
//   abs_max/mae/mse  : max_q = 6,   max_s = 256  <- 给 1.5 倍扩张留余量
float nvfp4_max_q_of(const QuantConfig& cfg) {
    return (cfg.scale_rule == ScaleRule::Static4) ? 4.0f : kMaxE2M1;
}
float nvfp4_max_s_of(const QuantConfig& cfg) {
    return cfg.scale_rule_adaptive() ? kFourOverSixScaleMax : kMaxE4M3;
}
int scale_rule_code(ScaleRule r) {
    switch (r) {
        case ScaleRule::AbsMax:
            return 0;
        case ScaleRule::Mae:
            return 1;
        default:
            return 2;  // mse
    }
}

// 融合版 4/6 kernel 的前置条件。
//
// 融合版的线程到数据映射是「一个线程负责一整个缩放块」, 块内元素数决定线程的循环长度,
// 而那个长度必须是编译期常量(否则块内元素进 local memory), 所以启动包装按
// block_size = 4 / 8 / 16 / 32 特化出四个实例。要让选择就地完成, 必须:
//   * block_size ∈ {4, 8, 16, 32} -- 有对应特化;
//   * 每个块都是完整的(numel 能被 block_size 整除) -- 否则最后一个块的元素数与
//     特化出的长度对不上, 累加次序也就对不上;
//   * 不是单块平铺(num_blocks > 1) -- 单块时整张量就是一个缩放块, 没有可并的遍,
//     交给通用路径更省事。
// 不满足时回落三遍路径(k_nvfp4_block_scales_46 + k_quant_nvfp4),
// 正确性不受影响。
bool can_fuse_nvfp4_46(
    const QuantConfig& cfg, int64_t numel, int64_t block_size, int64_t num_blocks) {
    if (cfg.format != QuantFormat::NVFP4 || !cfg.scale_rule_adaptive())
        return false;
    if (cfg.scale_mode != ScaleMode::Block || num_blocks <= 1)
        return false;
    const int shift = pow2_shift(block_size);
    if (shift < 2 || shift > 5)  // block_size in {4, 8, 16, 32}
        return false;
    return (numel % block_size) == 0;
}

// 融合版 4/6 的启动包装(输入遍数 3 -> 2, 选择跟量化同一个 kernel)。
// 块内元素数就是每个线程的循环长度, 必须是编译期常量, 所以按 block_size = 4 / 8 / 16 / 32
// 特化出四个 LOG_BLOCK 实例(block_size 是 2 的幂由 can_fuse_nvfp4_46 保证)。
template <class InT>
void launch_quant_nvfp4_46(const InT* x, int64_t numel, const QuantConfig& cfg, int64_t block_size,
    const float* d_block_amax, const float* d_gscale, uint8_t* d_scales, uint8_t* d_out,
    cudaStream_t stream, double* d_stats_acc, float* d_stats_max, SinkKind sk) {
    const int shift = pow2_shift(block_size);
    const int64_t num_blocks = numel >> shift;
    const unsigned int grid = grid_for_quant(num_blocks, sk, kStatsMaxBlocksNv);
    const uint64_t seed = cfg.seed;
    const bool stoch = cfg.rounding == Rounding::Stochastic;
    const int rule = scale_rule_code(cfg.scale_rule);
    const float max_q = nvfp4_max_q_of(cfg);
    dispatch_sink_out(cfg.output_type, sk, [&](auto tag) {
        using Sink = decltype(tag);
        using OutT = typename Sink::out_type;
        dispatch_stoch(stoch, [&](auto sct) {
            constexpr bool kS = decltype(sct)::value;
            switch (shift) {
                case 2:
                    k_quant_nvfp4_46<InT, OutT, Sink, kS, 2><<<grid, kThreads, 0, stream>>>(x,
                        d_block_amax, d_gscale, numel, rule, max_q, seed, d_scales, d_out,
                        d_stats_acc, d_stats_max);
                    break;
                case 3:
                    k_quant_nvfp4_46<InT, OutT, Sink, kS, 3><<<grid, kThreads, 0, stream>>>(x,
                        d_block_amax, d_gscale, numel, rule, max_q, seed, d_scales, d_out,
                        d_stats_acc, d_stats_max);
                    break;
                case 4:
                    k_quant_nvfp4_46<InT, OutT, Sink, kS, 4><<<grid, kThreads, 0, stream>>>(x,
                        d_block_amax, d_gscale, numel, rule, max_q, seed, d_scales, d_out,
                        d_stats_acc, d_stats_max);
                    break;
                default:
                    k_quant_nvfp4_46<InT, OutT, Sink, kS, 5><<<grid, kThreads, 0, stream>>>(x,
                        d_block_amax, d_gscale, numel, rule, max_q, seed, d_scales, d_out,
                        d_stats_acc, d_stats_max);
                    break;
            }
        });
    });
    LPQ_CUDA_CHECK_KERNEL();
}

template <class InT>
void run_scales(const InT* x, const float* d_block_amax, const float* d_tensor_amax, int64_t numel,
    int64_t num_blocks, const QuantConfig& cfg, QuantizedDevice& q, cudaStream_t stream) {
    if (cfg.is_mx()) {
        // MX 家族:每块一个 E8M0. emax_elem 是元素格式的最大无偏指数
        // (e4m3 -> 8, e5m2 -> 15, e2m1 -> 2)
        const int emax_elem = (cfg.format == QuantFormat::MxFP4)
                                  ? FmtE2M1::emax
                                  : (cfg.mxfp8_e5m2 ? FmtE5M2::emax : FmtE4M3::emax);
        k_mxfp8_scales<<<grid_for(num_blocks, kThreads), kThreads, 0, stream>>>(
            d_block_amax, num_blocks, emax_elem, q.scales.as<uint8_t>());
        LPQ_CUDA_CHECK_KERNEL();
        return;
    }

    // NVFP4:先定 per-tensor 的 FP32 全局缩放, 再定 per-block 的 E4M3 块缩放
    const float max_q = nvfp4_max_q_of(cfg);
    const float max_s = nvfp4_max_s_of(cfg);
    k_nvfp4_global_scale<<<1, 1, 0, stream>>>(
        d_tensor_amax, max_q, max_s, q.global_scale.as<float>(), q.global_scale.as<float>() + 1);
    LPQ_CUDA_CHECK_KERNEL();

    if (cfg.scale_rule_adaptive() && !can_fuse_nvfp4_46(cfg, numel, q.block_size, q.num_blocks)) {
        // 4/6 自适应:逐块在 6 / 4 两个候选里挑, 需要把输入再扫一遍。
        // 可融合时选择由量化 kernel 自己完成(k_quant_nvfp4_46), 这里就不再单独起一遍。
        k_nvfp4_block_scales_46<InT><<<grid_for(num_blocks, kThreads), kThreads, 0, stream>>>(x,
            d_block_amax, q.global_scale.as<float>(), numel, q.block_size, num_blocks, max_q,
            scale_rule_code(cfg.scale_rule), cfg.seed, cfg.rounding == Rounding::Stochastic,
            q.scales.as<uint8_t>());
        LPQ_CUDA_CHECK_KERNEL();
        return;
    }
    k_nvfp4_block_scales<<<grid_for(num_blocks, kThreads), kThreads, 0, stream>>>(
        d_block_amax, num_blocks, q.global_scale.as<float>(), max_q, q.scales.as<uint8_t>());
    LPQ_CUDA_CHECK_KERNEL();
}

}  // namespace

// =============================================================================
//  对外接口
// =============================================================================

ElemKind elem_kind_of(const QuantConfig& cfg) {
    if (cfg.format == QuantFormat::NVFP4 || cfg.format == QuantFormat::MxFP4)
        return ElemKind::E2M1;
    return cfg.mxfp8_e5m2 ? ElemKind::E5M2 : ElemKind::E4M3;
}
QuantizedDevice quantize_on_device(const void* d_input, InputType in_type, int64_t numel,
    const QuantConfig& cfg, cudaStream_t stream, ErrorStats* out_stats, AccumMode mode) {
    LPQ_CHECK(numel > 0, "quantization input length must be positive");
    double* d_acc = nullptr;
    float* d_max = nullptr;
    if (out_stats != nullptr) {
        const StatsReduction red = stats_reduction_scratch();
        LPQ_CUDA_CHECK(cudaMemsetAsync(red.acc, 0, 4 * sizeof(double), stream));
        LPQ_CUDA_CHECK(cudaMemsetAsync(red.max, 0, sizeof(float), stream));
        d_acc = red.acc;
        d_max = red.max;
    }
    const SinkKind sk = sink_kind_for(out_stats != nullptr, mode);
    QuantizedDevice q = (in_type == InputType::FP32)
                            ? run_quantize<float>(static_cast<const float*>(d_input), numel, cfg,
                                  stream, d_acc, d_max, sk)
                            : run_quantize<__half>(static_cast<const __half*>(d_input), numel, cfg,
                                  stream, d_acc, d_max, sk);
    // 需要回到主机的东西只有一个 NVFP4 的全局缩放标量(8 字节)和误差统计的 4 个和
    // (36 字节). 这里把它们全部排好队再同步**一次**: 原来 NVFP4 在缩放阶段同步一次、
    // 统计再同步一次, 两次同步之间的流水线气泡是白付的.
    const bool want_gscale = (cfg.format == QuantFormat::NVFP4);
    const bool want_stats = (out_stats != nullptr && q.stats_fused);
    float h_gs[2] = {1.0f, 1.0f};
    double h_acc[4] = {0.0, 0.0, 0.0, 0.0};
    float h_max = 0.0f;
    if (want_gscale)
        LPQ_CUDA_CHECK(cudaMemcpyAsync(
            h_gs, q.global_scale.get(), sizeof(h_gs), cudaMemcpyDeviceToHost, stream));
    if (want_stats) {
        LPQ_CUDA_CHECK(
            cudaMemcpyAsync(h_acc, d_acc, sizeof(h_acc), cudaMemcpyDeviceToHost, stream));
        LPQ_CUDA_CHECK(
            cudaMemcpyAsync(&h_max, d_max, sizeof(h_max), cudaMemcpyDeviceToHost, stream));
    }
    if (want_gscale || want_stats)
        LPQ_CUDA_CHECK(cudaStreamSynchronize(stream));
    if (want_gscale) {
        q.global_scale_value = h_gs[0];
        q.global_scale_inv = h_gs[1];
    }
    if (want_stats)
        *out_stats = finalize_error_stats(h_acc, h_max, numel);
    return q;
}

QuantizedDevice quantize_on_device(const void* d_input, InputType in_type, int64_t numel,
    const QuantConfig& cfg, cudaStream_t stream) {
    return quantize_on_device(d_input, in_type, numel, cfg, stream, nullptr, AccumMode::Fast);
}

QuantizedDevice quantize_on_device(
    const void* d_input, InputType in_type, int64_t numel, const QuantConfig& cfg) {
    return quantize_on_device(d_input, in_type, numel, cfg, nullptr, nullptr, AccumMode::Fast);
}

namespace {

template <class InT>
QuantizedDevice run_quantize(const InT* x, int64_t numel, const QuantConfig& cfg,
    cudaStream_t stream, double* d_stats_acc, float* d_stats_max, SinkKind sk) {
    QuantizedDevice q;
    q.numel = numel;
    q.block_size = (cfg.scale_mode == ScaleMode::Tensor) ? numel : (int64_t)cfg.block_size;
    if (q.block_size <= 0)
        q.block_size = numel;
    q.num_blocks = (numel + q.block_size - 1) / q.block_size;
    if (q.num_blocks < 1)
        q.num_blocks = 1;
    // 标准 NVFP4 要先算出全张量 amax 才能定全局缩放, 因此固定读两遍输入;
    // 4/6 自适应默认还要多一遍只读扫描来逐块挑选缩放因子,
    // 但可融合时这一步由量化 kernel 顺带完成(见 can_fuse_nvfp4_46), 遍数回到 2.
    const bool fuse46 = can_fuse_nvfp4_46(cfg, numel, q.block_size, q.num_blocks);
    q.input_passes = (cfg.scale_rule_adaptive() && !fuse46) ? 3 : 2;

    // 4bit 元素严格两元素/字节, 8bit 元素一元素/字节
    const int64_t packed_bytes = cfg.is_4bit() ? ((numel + 1) / 2) : numel;
    q.packed = DeviceBuffer((size_t)packed_bytes);
    q.scales = DeviceBuffer((size_t)q.num_blocks);
    q.global_scale = DeviceBuffer(2 * sizeof(float));

    DeviceTimer timer;

    // ---- 融合路径:MXFP8 + block 模式 + block_size 为 [4,128] 内的 2 的幂 ----
    // 此时缩放因子只依赖本块 amax, 因此可以在一次访存里完成
    // amax -> E8M0 -> 量化 -> 打包, 省掉独立的 amax/scale kernel 与一次输入重读.
    const int64_t fused_shift = pow2_shift(q.block_size);
    const bool can_fuse =
        cfg.is_mx() && (cfg.scale_mode == ScaleMode::Block) && fused_shift >= 2 && fused_shift <= 7;
    if (can_fuse) {
        const float one[2] = {1.0f, 1.0f};
        LPQ_CUDA_CHECK(cudaMemcpyAsync(
            q.global_scale.get(), one, sizeof(one), cudaMemcpyHostToDevice, stream));
        timer.start();
        if (cfg.format == QuantFormat::MxFP4) {
            launch_quant_mxfp8_fused<InT, FmtE2M1>(x, numel, cfg, (int)fused_shift,
                q.scales.as<uint8_t>(), q.packed.as<uint8_t>(), stream, d_stats_acc, d_stats_max,
                sk);
        } else if (cfg.mxfp8_e5m2) {
            launch_quant_mxfp8_fused<InT, FmtE5M2>(x, numel, cfg, (int)fused_shift,
                q.scales.as<uint8_t>(), q.packed.as<uint8_t>(), stream, d_stats_acc, d_stats_max,
                sk);
        } else {
            launch_quant_mxfp8_fused<InT, FmtE4M3>(x, numel, cfg, (int)fused_shift,
                q.scales.as<uint8_t>(), q.packed.as<uint8_t>(), stream, d_stats_acc, d_stats_max,
                sk);
        }
        q.input_passes = 1;  // 融合 kernel: 输入只读一遍
        q.stats_fused = (d_stats_acc != nullptr);
        q.timings.quant_ms = timer.stop_ms();
        return q;
    }

    // 中间缓冲:每块 amax + 全张量 amax.
    // 只有"单块平铺"那条路径(k_tensor_amax, 用 atomicMax 把全张量 amax 写进块数组
    // 的第 0 个槽)才需要预先把块数组清零; 走 k_block_amax_* 时每个槽位都会被显式
    // 写一遍, 再清 16.7MB(8192^2 下)纯属白费显存带宽.
    const bool amax_single_tile = (cfg.scale_mode == ScaleMode::Tensor) || (q.num_blocks == 1);
    DeviceBuffer block_amax((size_t)q.num_blocks * sizeof(float));
    DeviceBuffer tensor_amax(sizeof(float));
    if (amax_single_tile)
        LPQ_CUDA_CHECK(cudaMemsetAsync(block_amax.get(), 0, block_amax.size(), stream));
    LPQ_CUDA_CHECK(cudaMemsetAsync(tensor_amax.get(), 0, tensor_amax.size(), stream));

    // ---- 阶段 1:amax ----
    timer.start();
    float* d_tensor_amax = (cfg.format == QuantFormat::NVFP4) ? tensor_amax.as<float>() : nullptr;
    launch_amax<InT>(
        x, numel, cfg, q.block_size, q.num_blocks, block_amax.as<float>(), d_tensor_amax, stream);
    q.timings.amax_ms = timer.stop_ms();
    // 单块平铺时全张量 amax 直接被写进 block_amax[0], d_tensor_amax 并没有被写过.
    // 此时必须从 block_amax[0] 取, 否则 NVFP4 的全张量缩放会退化成 1.0.
    const float* eff_tensor_amax = (cfg.format == QuantFormat::NVFP4)
                                       ? (amax_single_tile ? block_amax.as<float>() : d_tensor_amax)
                                       : nullptr;

    // ---- 阶段 2:缩放因子 ----
    timer.start();
    run_scales(x, block_amax.as<float>(), eff_tensor_amax, numel, q.num_blocks, cfg, q, stream);
    if (cfg.is_mx()) {
        const float one[2] = {1.0f, 1.0f};
        LPQ_CUDA_CHECK(cudaMemcpyAsync(
            q.global_scale.get(), one, sizeof(one), cudaMemcpyHostToDevice, stream));
    }
    q.timings.scale_ms = timer.stop_ms();

    // ---- 阶段 3:量化 + 打包 ----
    // 误差统计(启用时)在这里顺带完成: 量化 kernel 手里本来就有原始值 x 和刚算出的
    // 量化码, 顺手反量化一次就能累加, 不需要事后再把 fp32 输入和反量化结果各读一遍.
    timer.start();
    if (cfg.format == QuantFormat::MxFP4) {
        // MXFP4 = e2m1 元素 + E8M0 共享缩放, 与 MXFP8 只差元素格式与打包位宽
        launch_quant_mxfp8<InT, FmtE2M1>(x, numel, cfg, q.block_size, q.num_blocks,
            q.scales.as<uint8_t>(), q.packed.as<uint8_t>(), stream, d_stats_acc, d_stats_max, sk);
    } else if (cfg.format == QuantFormat::MxFP8) {
        if (cfg.mxfp8_e5m2) {
            launch_quant_mxfp8<InT, FmtE5M2>(x, numel, cfg, q.block_size, q.num_blocks,
                q.scales.as<uint8_t>(), q.packed.as<uint8_t>(), stream, d_stats_acc, d_stats_max,
                sk);
        } else {
            launch_quant_mxfp8<InT, FmtE4M3>(x, numel, cfg, q.block_size, q.num_blocks,
                q.scales.as<uint8_t>(), q.packed.as<uint8_t>(), stream, d_stats_acc, d_stats_max,
                sk);
        }
    } else if (fuse46) {
        // 4/6 融合路径:选择与量化合并, 输入只读两遍
        launch_quant_nvfp4_46<InT>(x, numel, cfg, q.block_size, block_amax.as<float>(),
            q.global_scale.as<float>(), q.scales.as<uint8_t>(), q.packed.as<uint8_t>(), stream,
            d_stats_acc, d_stats_max, sk);
    } else {
        launch_quant_nvfp4<InT>(x, numel, cfg, q.block_size, q.num_blocks, q.scales.as<uint8_t>(),
            q.global_scale.as<float>(), q.packed.as<uint8_t>(), stream, d_stats_acc, d_stats_max,
            sk);
    }
    q.stats_fused = (d_stats_acc != nullptr);
    q.timings.quant_ms = timer.stop_ms();
    return q;
}

}  // namespace

}  // namespace lpq