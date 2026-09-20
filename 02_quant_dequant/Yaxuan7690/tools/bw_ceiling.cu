// =============================================================================
//  bw_ceiling.cu -- 逐阶段显存带宽上限标定
//
//  目的: 回答「这个 kernel 还有多少余地」。每个阶段 kernel 的访存都能抽象成
//  「每处理固定个元素读 R 字节、写 W 字节」的纯流式模式, 于是造一个只做同样字节
//  搬运、不做任何计算的对照 kernel, 量出该 (R, W) 组合下硬件真正能跑到的带宽 --
//  这就是那个阶段的**上限**, 而不是拿 448GB/s 的纸面峰值去比。
//
//  本机(RTX 5060 Ti, 128bit GDDR7 @28Gbps => 纸面 448GB/s)实测: 纯读 ~423GB/s,
//  纯写 ~426GB/s, 各阶段的混合模式都落在 366~431GB/s -- 读写在 DRAM 上的换向
//  开销是硬性的, 与 kernel 写得好不好无关。结论见 总结报告 §6.3。
//
//  百分比的分母不再写死 448: 换成从 cudaMemGetInfo 同源的设备属性算出来的纸面峰值
//  (2 x memoryClockRate x busWidth / 8)。在 RTX 5060 Ti 上它算出来就是 448, 输出
//  与以前逐字相同; 但换到国产卡上写死 448 会让百分比彻底失去意义(MTT S4000 实测
//  纯读 672GB/s, 写死 448 会印出 150%)。
//
//  用法: bw_ceiling [numel=67108864]
// =============================================================================
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CK(x)                                                                                    \
    do {                                                                                         \
        cudaError_t e_ = (x);                                                                    \
        if (e_ != cudaSuccess) {                                                                 \
            std::printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); \
            return 1;                                                                            \
        }                                                                                        \
    } while (0)

// 读 RB 字节 / 写 WB 字节, 全部按「每条线程一条连续的 16B 段」的方式落位,
// 与真实 kernel 的访问模式一致。
template <int RB, int WB>
__global__ void k_mix(const char* __restrict__ src, char* __restrict__ dst, long long n) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    unsigned long long acc = 0;
    if (RB >= 64) {
#pragma unroll
        for (int k = 0; k < RB / 16; ++k) {
            const uint4 v = *reinterpret_cast<const uint4*>(src + i * RB + k * 16);
            acc ^= (unsigned long long)v.x ^ ((unsigned long long)v.y << 9) ^
                   ((unsigned long long)v.z << 19) ^ ((unsigned long long)v.w << 29);
        }
    } else if (RB == 16) {
        const uint4 v = *reinterpret_cast<const uint4*>(src + i * 16);
        acc = (unsigned long long)v.x ^ ((unsigned long long)v.y << 9) ^
              ((unsigned long long)v.z << 19) ^ ((unsigned long long)v.w << 29);
    } else if (RB == 8) {
        const uint2 v = *reinterpret_cast<const uint2*>(src + i * 8);
        acc = (unsigned long long)v.x ^ ((unsigned long long)v.y << 13);
    } else if (RB == 4) {
        acc = *reinterpret_cast<const unsigned int*>(src + i * 4);
    } else if (RB == 2) {
        acc = *reinterpret_cast<const unsigned short*>(src + i * 2);
    }
    if (WB == 8) {
        uint2 o;
        o.x = (unsigned int)acc;
        o.y = (unsigned int)(acc >> 32) + (unsigned int)(acc >> 7);
        *reinterpret_cast<uint2*>(dst + i * 8) = o;
    } else if (WB == 4) {
        *reinterpret_cast<unsigned int*>(dst + i * 4) =
            (unsigned int)acc ^ (unsigned int)(acc >> 19);
    } else if (WB == 2) {
        *reinterpret_cast<unsigned short*>(dst + i * 2) = (unsigned short)(acc ^ (acc >> 11));
    } else if (WB == 1) {
        *reinterpret_cast<unsigned char*>(dst + i) = (unsigned char)(acc ^ (acc >> 23));
    } else {
        // 纯读: 这个条件必须是"可达"的, 否则 nvcc 会把整条读循环当死代码删掉。
        // acc 只由 4 个 32 bit 值的移位异或构成, 最高位到不了 2^61, 因此写成
        // acc == 0xDEADBEEFCAFEBABEull 时 nvcc 能直接证明"恒为假", 于是本内核被折叠成
        // 一条 EXIT, 读数变成 5000+ GB/s 的假值(实测过)。这里把常数限制在 acc 的可达
        // 范围内, 既保住那些 LDG, 又因为落点概率约 2^-60 而不产生任何写流量。
        if (acc == 0x0F0F0F0F0F0F0F0Full)
            *reinterpret_cast<volatile unsigned char*>(dst) = 1;
    }
}

// 每条线程迭代处理的元素数: 决定一侧的字节数, 另一侧按真实 kernel 的比例给
struct Case {
    const char* name;
    int rb, wb;
    int elems;
    double read_per_elem, write_per_elem;
    void (*launch)(const char*, char*, long long, int);
};

template <int RB, int WB>
static void launch_mix(const char* s, char* d, long long n, int threads) {
    k_mix<RB, WB><<<(unsigned)((n + threads - 1) / threads), threads>>>(s, d, n);
}

int main(int argc, char** argv) {
    const long long numel = (argc > 1) ? atoll(argv[1]) : 67108864LL;  // 与 bench 一致
    const int threads = 256;
    const int iters = 30;

    char *dsrc = nullptr, *ddst = nullptr;
    CK(cudaMalloc(&dsrc, (size_t)numel * 4));
    CK(cudaMalloc(&ddst, (size_t)numel * 2));
    CK(cudaMemset(dsrc, 0x5A, (size_t)numel * 4));
    CK(cudaMemset(ddst, 0x00, (size_t)numel * 2));

    const Case cases[] = {
        {"pure-read         ", 16, 0, 4, 4.0, 0.0, launch_mix<16, 0>},
        {"amax   64B/16elem ", 64, 4, 16, 4.0, 0.25, launch_mix<64, 4>},
        {"mxfp8 quant 16->4 ", 16, 4, 4, 4.0, 1.0, launch_mix<16, 4>},
        {"mxfp8 deq    4->8 ", 4, 8, 4, 1.0, 2.0, launch_mix<4, 8>},
        {"nvfp4 quant 16->2 ", 16, 2, 4, 4.0, 0.5, launch_mix<16, 2>},
        {"nvfp4 deq    2->8 ", 2, 8, 4, 0.5, 2.0, launch_mix<2, 8>},
        {"pure-write   0->8 ", 0, 8, 4, 0.0, 2.0, launch_mix<0, 8>},
    };

    // 纸面峰值: 由运行时报告的显存时钟与位宽算出, 换机器自动跟着变
    double paper_peak = 448.0;
    {
        cudaDeviceProp prop = {};
        if (cudaGetDeviceProperties(&prop, 0) == cudaSuccess && prop.memoryClockRate > 0 &&
            prop.memoryBusWidth > 0)
            paper_peak = 2.0 * (double)prop.memoryClockRate * 1e3 *
                         ((double)prop.memoryBusWidth / 8.0) / 1e9;
    }

    double best_mixed = 0.0;
    for (size_t c = 0; c < sizeof(cases) / sizeof(cases[0]); ++c) {
        const Case& cc = cases[c];
        const long long n = numel / cc.elems;
        cudaEvent_t e0, e1;
        CK(cudaEventCreate(&e0));
        CK(cudaEventCreate(&e1));
        for (int i = 0; i < 5; ++i)
            cc.launch(dsrc, ddst, n, threads);
        float best = 1e30f;
        for (int i = 0; i < iters; ++i) {
            CK(cudaEventRecord(e0));
            cc.launch(dsrc, ddst, n, threads);
            CK(cudaEventRecord(e1));
            CK(cudaEventSynchronize(e1));
            float ms = 0.f;
            CK(cudaEventElapsedTime(&ms, e0, e1));
            if (ms < best)
                best = ms;
        }
        const double rb = cc.read_per_elem * (double)numel;
        const double wb = cc.write_per_elem * (double)numel;
        const double gbps = (rb + wb) / 1e9 / ((double)best / 1e3);
        if (rb > 0.0 && wb > 0.0 && gbps > best_mixed)
            best_mixed = gbps;
        std::printf("%s %7.3f ms  read %7.1f MB  write %7.1f MB => %6.1f GB/s (%.1f%% of %.0f)\n",
            cc.name, best, rb / 1e6, wb / 1e6, gbps, 100.0 * gbps / paper_peak, paper_peak);
        cudaEventDestroy(e0);
        cudaEventDestroy(e1);
    }
    std::printf("\nbest mixed read/write rate on this device: %.1f GB/s\n", best_mixed);
    cudaFree(dsrc);
    cudaFree(ddst);
    return 0;
}