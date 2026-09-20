#include "project.hpp"
#include <algorithm>
#include <chrono>
#include <cuda_runtime.h>
namespace lp {
static void check(cudaError_t e) {
    if (e != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(e));
}
struct Device {
    uint8_t *p = nullptr;
    explicit Device(size_t n) {
        check(cudaMalloc(&p, n));
    }
    ~Device() {
        if (p)
            cudaFree(p);
    }
    Device(const Device &) = delete;
};
struct Events {
    cudaEvent_t e[7]{};
    Events() {
        try {
            for (auto &x : e)
                check(cudaEventCreate(&x));
        } catch (...) {
            for (auto x : e)
                if (x)
                    cudaEventDestroy(x);
            throw;
        }
    }
    ~Events() {
        for (auto x : e)
            cudaEventDestroy(x);
    }
    void record(int i) {
        check(cudaEventRecord(e[i]));
    }
    double elapsed(int a, int b) {
        float t;
        check(cudaEventElapsedTime(&t, e[a], e[b]));
        return t;
    }
};
// 并行全局归约：256 线程的每个块扫描独立网格步长，再 atomicMax 合并非负 float 位模式。
__global__ void reduce_all(const uint8_t *x, uint64_t n, int dtype, unsigned *maximum,
                           int *invalid) {
    __shared__ float shared[256];
    float a = 0;
    for (uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += uint64_t(gridDim.x) * blockDim.x) {
        float v = input_value(x, i, dtype);
        if (!isfinite(v))
            atomicExch(invalid, 1);
        else
            a = fmaxf(a, fabsf(v));
    }
    shared[threadIdx.x] = a;
    __syncthreads();
    for (int s = 128; s; s /= 2) {
        if (threadIdx.x < s)
            shared[threadIdx.x] = fmaxf(shared[threadIdx.x], shared[threadIdx.x + s]);
        __syncthreads();
    }
    if (!threadIdx.x)
        atomicMax(maximum, __float_as_uint(shared[0]));
}
__global__ void make_global(const unsigned *maximum, float *g, int f) {
    if (!threadIdx.x)
        *g = f == 2 ? global_scale(__uint_as_float(*maximum)) : 1;
}
// 一个 warp 对应一个量化块，块沿张量的“行”方向划分。
template <bool Fast>
__global__ void make_scales(const uint8_t *x, uint64_t n, uint64_t cols, int dtype, int f,
                            int block, int mode, const unsigned *maximum, const float *g,
                            uint8_t *scales, uint64_t nb) {
    // b 是 warp 负责的全局块号（沿行方向编号）
    uint64_t b = (uint64_t(blockIdx.x) * blockDim.x + threadIdx.x) / 32;
    //lane 是线程在 warp 内的编号 0~31。
    int lane = threadIdx.x & 31;
    float a = 0;
    if (b < nb) {
        if (mode)
            a = __uint_as_float(*maximum);
        else {
            // 行内块数
            uint64_t bpr = (cols + block - 1) / block;
            // 从全局块号反推行号和行内块号
            //计算行号 r。
            uint64_t r = b / bpr;
            //计算行内块号cb
            uint64_t cb = b % bpr;
            // 该块覆盖的列范围
            uint64_t col = cb * uint64_t(block) + lane;
            uint64_t i = r * cols + col;
            // 边界检查：
            // lane < block：确保当前 lane 在块大小范围内。如果 block < 32，只有前 block 个 lane 有效。
            // col < cols：确保列没有超出实际列数（最后一行的最后一个块可能不满）。
            // i < n：确保全局索引没有超出元素总数。
            if (lane < block && col < cols && i < n)
                a = fabsf(input_value(x, i, dtype));
        }
    }
    for (int s = 16; s; s /= 2)
        //这里的 __shfl_down_sync(0xffffffff, a, s) 本身就带有同步语义，
        //它在 shuffle 之前会自动让 warp 内所有 mask 指定的 lane 到达同一个执行点。
        a = fmaxf(a, __shfl_down_sync(0xffffffff, a, s));

    //只有 lane == 0 且块号有效时，才进行最终的 scale 计算和写入。
    //!lane 等价于 lane == 0。
    //因为归约后最大值在 lane 0 上。
    if (!lane && b < nb) {
        //如果量化格式 f == 2 且模板参数 Fast 为 true，走快速编码路径。
        if (f == 2 && Fast) {
            auto q = encode_fast(double(a) / (6.0 * double(*g)), 0);
            scales[b] = a == 0 ? 56 : q ? q : 1;
        } else
            scales[b] = f == 2 ? nvscale(a, *g) : mxscale(a, f);
    }
}
template <bool Fast>
__global__ void encode_kernel(const uint8_t *x, uint64_t n, uint64_t cols, int dtype, int f,
                              int block, int mode, int rounding, uint64_t seed,
                              const uint8_t *scales, const float *g, uint8_t *packed) {
    //grid=unsigned((p.data.size() + 255) / 256)
    //所以j是输出字节的下标
    uint64_t j = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    //NVFP4（f == 2）：每个元素 4 bit
    //NVFP4 的“4”就是指每个量化值占 4 个 bit（半个字节）。
    //一个字节是 8 bit，所以：8 bit / 4 bit = 2 个元素
    //因此 count = 2，一个线程负责生成一个字节，里面装两个元素：
    int count = f == 2 ? 2 : 1;
    if (j >= (n + count - 1) / count)
        return;
    unsigned byte = 0;
    uint64_t bpr = (cols + block - 1) / block;
    // 字节为写入所有权单位，两个半字节由同一个线程生成，避免竞争。
    //count = 2 时循环两次，分别处理低 4-bit 和高 4-bit。
    //count = 1 时只循环一次
    for (int k = 0; k < count; ++k) {
        //第 j 个字节负责的元素下标是 j*count + k。
        //i：当前线程正在处理的元素编号
        uint64_t i = j * count + k;
        if (i < n) {
            // 按行方向定位元素所属块
            uint64_t blk = mode ? 0 : (i / cols) * bpr + (i % cols) / block;
            //得到该元素最终使用的缩放因子。
            double s = scale_value(scales[blk], f, *g);
            //count=1,k=0,byte |= unsigned(q) << 0;

            //count=2,k=0,byte |= unsigned(q) << 0;  //处理低4-bit
            //        k=1,byte |= unsigned(q) << 4;  //处理高4-bit
            byte |=
                unsigned(
                    Fast ? encode_fast(double(input_value(x, i, dtype)) / s, f, rounding, seed, i)
                         : encode(double(input_value(x, i, dtype)) / s, f, rounding, seed, i))
                << (4 * k);
        }
    }
    // byte 是 unsigned（通常是 32 位），但 packed[j] 是 uint8_t。
    // 赋值时会发生隐式截断，只保留 byte 的低 8 位写入 packed[j]。
    // 所以最终写入内存的确实是一个字节。
    packed[j] = byte;
}
template <bool Fast>
__global__ void decode_kernel(const uint8_t *packed, const uint8_t *scales, const float *g,
                              uint64_t n, uint64_t cols, int f, int block, int mode, int dtype,
                              uint8_t *out) {
    uint64_t j = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    int count = f == 2 ? 2 : 1;
    if (j >= (n + count - 1) / count)
        return;
    unsigned byte = packed[j];
    uint64_t bpr = (cols + block - 1) / block;
    for (int k = 0; k < count; ++k) {
        uint64_t i = j * count + k;
        if (i < n) {
            unsigned q = f == 2 ? (byte >> (4 * k)) & 15 : byte;
            uint64_t blk = mode ? 0 : (i / cols) * bpr + (i % cols) / block;
            //获取缩放因子并还原浮点值
            double v = (Fast ? decode_fast(q, f) : decode(q, f)) *
                       scale_value(scales[blk], f, *g);
            store_value(out, i, dtype, output_bits(v, dtype));
        }
    }
}
Packed quant_gpu(const Tensor &x, const Config &c, const std::string &variant, Timing *timing,
                 Tensor *decoded) {
    validate(c);
    if (variant != "baseline" && variant != "optimized")
        throw std::runtime_error("kernel variant not implemented");
    uint64_t n = checked_size(x.rows, x.cols);
    if (x.dtype < 0 || x.dtype > 1 || x.data.size() != n * (x.dtype ? 2 : 4))
        throw std::runtime_error("invalid input tensor");
    //p 是返回给调用者的 Packed 对象。
    Packed p{x.rows, x.cols, x.dtype, c, 1, {}, {}};
    // 沿行方向分块：每行 bpr 个块，共 rows * bpr 个块
    uint64_t bpr = (x.cols + c.block - 1) / c.block;
    uint64_t nb = c.mode ? 1 : x.rows * bpr;
    p.scales.resize(nb);
    //根据量化格式给输出字节流分配正确的大小。
    p.data.resize(c.format == 2 ? (n + 1) / 2 : n);
    bool need_decode = timing || decoded;
    Tensor y{x.rows, x.cols, c.out, {}};
    if (need_decode)
        y.data.resize(n * (c.out ? 2 : 4));
    auto start = std::chrono::steady_clock::now();
    //dp.p 是 GPU 上的一块设备内存，大小和 p.data 一样。这块内存就是 encode_kernel 的输出目标。
    Device dx(x.data.size()), ds(nb), dp(p.data.size()), dg(4), dm(4), bad(4),
        dy(std::max<size_t>(1, y.data.size()));
    Events ev;
    ev.record(0);
    check(cudaMemcpy(dx.p, x.data.data(), x.data.size(), cudaMemcpyHostToDevice));
    ev.record(1);
    check(cudaMemset(dm.p, 0, 4));
    check(cudaMemset(bad.p, 0, 4));
    reduce_all<<<unsigned(std::min<uint64_t>((n + 255) / 256, 4096)), 256>>>(
        dx.p, n, x.dtype, reinterpret_cast<unsigned *>(dm.p), reinterpret_cast<int *>(bad.p));
    check(cudaGetLastError());
    make_global<<<1, 1>>>(reinterpret_cast<unsigned *>(dm.p), reinterpret_cast<float *>(dg.p),
                          c.format);
    check(cudaGetLastError());
    if (variant == "optimized")
        //一个Block里有256个线程    256 / 32 = 8 个 warp。
        make_scales<true><<<unsigned((nb + 7) / 8), 256>>>(
            dx.p, n, x.cols, x.dtype, c.format, c.block, c.mode,
            reinterpret_cast<unsigned *>(dm.p), reinterpret_cast<float *>(dg.p), ds.p, nb);
    else
        make_scales<false><<<unsigned((nb + 7) / 8), 256>>>(
            dx.p, n, x.cols, x.dtype, c.format, c.block, c.mode,
            reinterpret_cast<unsigned *>(dm.p), reinterpret_cast<float *>(dg.p), ds.p, nb);
    check(cudaGetLastError());
    ev.record(2);
    if (variant == "optimized")
        encode_kernel<true><<<unsigned((p.data.size() + 255) / 256), 256>>>(
            dx.p, n, x.cols, x.dtype, c.format, c.block, c.mode, c.round, c.seed, ds.p,
            reinterpret_cast<float *>(dg.p), dp.p);
    else
        encode_kernel<false><<<unsigned((p.data.size() + 255) / 256), 256>>>(
            dx.p, n, x.cols, x.dtype, c.format, c.block, c.mode, c.round, c.seed, ds.p,
            reinterpret_cast<float *>(dg.p), dp.p);
    check(cudaGetLastError());
    ev.record(3);
    if (need_decode) {
        if (variant == "optimized")
            decode_kernel<true><<<unsigned((p.data.size() + 255) / 256), 256>>>(
                dp.p, ds.p, reinterpret_cast<float *>(dg.p), n, x.cols, c.format, c.block,
                c.mode, c.out, dy.p);
        else
            decode_kernel<false><<<unsigned((p.data.size() + 255) / 256), 256>>>(
                dp.p, ds.p, reinterpret_cast<float *>(dg.p), n, x.cols, c.format, c.block,
                c.mode, c.out, dy.p);
        check(cudaGetLastError());
    }
    ev.record(4);
    check(cudaMemcpy(p.data.data(), dp.p, p.data.size(), cudaMemcpyDeviceToHost));
    check(cudaMemcpy(p.scales.data(), ds.p, nb, cudaMemcpyDeviceToHost));
    check(cudaMemcpy(&p.global, dg.p, 4, cudaMemcpyDeviceToHost));
    if (!y.data.empty())
        check(cudaMemcpy(y.data.data(), dy.p, y.data.size(), cudaMemcpyDeviceToHost));
    int invalid = 0;
    check(cudaMemcpy(&invalid, bad.p, 4, cudaMemcpyDeviceToHost));
    ev.record(5);
    check(cudaEventSynchronize(ev.e[5]));
    if (invalid)
        throw std::runtime_error("nonfinite input");
    if (timing) {
        timing->h2d = ev.elapsed(0, 1);
        timing->scale = ev.elapsed(1, 2);
        timing->encode = ev.elapsed(2, 3);
        timing->quant = ev.elapsed(1, 3);
        timing->dequant = ev.elapsed(3, 4);
        timing->d2h = ev.elapsed(4, 5);
        timing->process =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
                .count();
    }
    if (decoded)
        *decoded = std::move(y);
    return p;
}
Tensor dequant_gpu(const Packed &p, Timing *timing) {
    validate_packed(p);
    Tensor y{p.rows, p.cols, p.c.out, {}};
    y.data.resize(p.size() * (p.c.out ? 2 : 4));
    Device dp(p.data.size()), ds(p.scales.size()), dg(4), dy(y.data.size());
    Events ev;
    ev.record(0);
    check(cudaMemcpy(dp.p, p.data.data(), p.data.size(), cudaMemcpyHostToDevice));
    check(cudaMemcpy(ds.p, p.scales.data(), p.scales.size(), cudaMemcpyHostToDevice));
    check(cudaMemcpy(dg.p, &p.global, 4, cudaMemcpyHostToDevice));
    ev.record(1);
    decode_kernel<true><<<unsigned((p.data.size() + 255) / 256), 256>>>(
        dp.p, ds.p, reinterpret_cast<float *>(dg.p), p.size(), p.cols, p.c.format, p.c.block,
        p.c.mode, p.c.out, dy.p);
    check(cudaGetLastError());
    ev.record(2);
    check(cudaMemcpy(y.data.data(), dy.p, y.data.size(), cudaMemcpyDeviceToHost));
    ev.record(3);
    check(cudaEventSynchronize(ev.e[3]));
    if (timing)
        timing->dequant = ev.elapsed(1, 2);
    return y;
}
} // namespace lp