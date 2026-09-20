// =============================================================================
//  test_portability.cu -- 国产平台移植探针
//
//  在本项目里, 需要感知平台的**源码级**差异只有六处, 全部集中在
//  include/lpq/platform.hpp. 本程序把这六项能力拆成互相独立的小测试逐个报告,
//  并在失败时直接给出应该添加的命令行开关 —— 这样在按小时计费的租用机器上,
//  不必通过「改一处、重编一次」的方式盲试.
//
//  另有一处**构建级**差异光靠 platform.hpp 修不了: 平台运行时的命名, 见下面
//  排查清单的第 0 条.
//
//  用法 (在构建目录下直接运行):
//      ./test_portability
//
//  如果连编译都过不去, 按下面的顺序排查:
//      0. 报 cuda_runtime.h 找不到, 或者 cudaMalloc / cudaMemcpy 之类未定义
//         -> 该平台的运行时**不叫 cudaXxx**. 这是构建级差异(第七处), 改
//            platform.hpp 没有用. 以摩尔线程 MUSA 为例(运行时叫 musaXxx),
//            官方有三条路, Makefile 会自动挑这台机器上可用的那条:
//              a) mcc -mtgpu -cuda_wrapper ... -lcuda2musa     (首选)
//                 mcc 的"兼容封装模式": 源码直接用 CUDA API 名字即可.
//                 MUSA SDK 3.1.0 与 5.2.0 的手册都写了这条.
//              b) mcc -x musa -fplugin=<libMusaMapping.so>     (SDK >= 5.2.0)
//                 编译期 Clang 插件, 按映射表重写头文件名与标识符.
//              c) musify-text --inplace -- src/*.cu tests/*.cu  (离线迁移)
//                 改完源码后用 make PLATFORM=moore MUSA_MODE=plain build.
//            链接报"未定义引用"时: wrapper 模式先补 -lmusart 试试, 再检查
//            -l 的先后顺序(mcc 默认按 --as-needed 链接, 顺序敏感).
//      1. 报 cuda_fp16.h / cuda_bf16.h 找不到
//         -> 用 -DLPQ_FP16_HEADER=<musa_fp16.h> / -DLPQ_BF16_HEADER=<...> 指定
//            平台自带的低精度类型头; 只想先看宏探测结果可以加
//            -DLPQ_PROBE_SKIP_FP16 跳过低精度类型部分.
//      2. 报 __host__ / __device__ 不认识
//         -> 该编译器未被识别为设备编译器, 加 -DLPQ_CUDA=1.
//      3. 报 __shfl_xor_sync / atomicMax(int) / atomicAdd(double) 不认识
//         -> 见对应小节给出的开关.
//      4. 编译过了, 但一跑就说找不到设备 / 非法指令 / kernel 启不起来
//         -> 目标架构代号选错了. MUSA 用 --offload-arch=mp_XX(mp_21 / mp_22 /
//            mp_31 ...), 直接跑 scripts/run_domestic.sh moore, 它会自动试出
//            这台机器可用的代号, 不用手工猜.
// =============================================================================
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "lpq/common.hpp"
#include "lpq/platform.hpp"

#ifndef LPQ_PROBE_SKIP_FP16
#include "lpq/vec_io.cuh"
#endif

#include "test_common.hpp"

// =============================================================================
//  设备侧: 报告宏探测结果
// =============================================================================
__global__ void k_report_macros(int* out) {
    // out[0] warpSize, out[1] LPQ_CUDA, out[2] LPQ_DEVICE_PASS,
    // out[3] 架构宏位掩码 (1=__CUDA_ARCH__ 2=__MUSA_ARCH__ 4=__MACA_ARCH__),
    // out[4..6] 三个架构宏的取值
    out[0] = (int)warpSize;
    out[1] = LPQ_CUDA;
    out[2] = LPQ_DEVICE_PASS;
    int m = 0;
#if defined(__CUDA_ARCH__)
    m |= 1;
    out[4] = __CUDA_ARCH__;
#endif
#if defined(__MUSA_ARCH__)
    m |= 2;
    out[5] = __MUSA_ARCH__;
#endif
#if defined(__MACA_ARCH__)
    m |= 4;
    out[6] = __MACA_ARCH__;
#endif
    out[3] = m;
}

// =============================================================================
//  [4] warp 内交换 LPQ_SHFL_XOR
//  每个线程取 (lane & (width-1)), 在宽度为 width 的组内做异或归约. 组的成员
//  恰好是连续的 width 个 lane, 因此每一组的结果都相同, 等于 0..width-1 的异或.
// =============================================================================
__global__ void k_test_shfl(int width, unsigned int* out) {
    const int lane = threadIdx.x & 31;
    unsigned int v = (unsigned int)(lane & (width - 1));
    for (int off = width >> 1; off > 0; off >>= 1) {
        v ^= LPQ_SHFL_XOR(v, off, width);
    }
    out[threadIdx.x] = v;
}

// =============================================================================
//  [5] 浮点原子最大: 每个线程先写一个大值, 再写一个 0(不能把结果打回去)
// =============================================================================
__global__ void k_test_atomic_max(float* out) {
    const float gid = (float)((long long)blockIdx.x * blockDim.x + threadIdx.x);
    lpq::lpq_atomic_max_f32(out, gid);
    lpq::lpq_atomic_max_f32(out, 0.0f);
}

// =============================================================================
//  [6] 双精度原子加: 每个线程分别加 1.0 与 0.5
// =============================================================================
__global__ void k_test_atomic_add_f64(double* out) {
    lpq::lpq_atomic_add_f64(&out[0], 1.0);
    lpq::lpq_atomic_add_f64(&out[1], 0.5);
}

// =============================================================================
//  [1][2] 位模式互转(设备遍走 __float_as_uint / __uint_as_float 分支)
// =============================================================================
__global__ void k_test_bits(const float* in, uint32_t* out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        out[i] = lpq::f32_to_bits(in[i]);
}

__global__ void k_test_unbits(const uint32_t* in, float* out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        out[i] = lpq::bits_to_f32(in[i]);
}

#ifndef LPQ_PROBE_SKIP_FP16
// =============================================================================
//  [3a] __half / __nv_bfloat16 / __half2 的取值
//  bits[0] 半精度编码与位运算参考不一致的元素数
//  bits[1] 半精度解码与位运算参考不一致的元素数
//  bits[2] 半精度编码-解码往返不稳定的元素数
//  bits[3] bfloat16 编码不一致
//  bits[4] bfloat16 解码不一致
//  bits[5] __half2 低元素拆分错误
//  bits[6] __half2 高元素拆分错误
// =============================================================================
__global__ void k_test_lowp_scalar(const float* in, uint32_t* bits, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    const float x = in[i];
    {
        const uint32_t want = lpq::f32_to_fmt<lpq::FmtF16>(x);
        const __half h = lpq::OutTraits<__half>::from_float(x);
        if ((uint32_t)__half_as_ushort(h) != want)
            atomicAdd(&bits[0], 1u);
        if (lpq::InTraits<__half>::to_float(h) != lpq::f32_from_half_bits((uint16_t)want))
            atomicAdd(&bits[1], 1u);
        const __half h2 = lpq::OutTraits<__half>::from_float(lpq::InTraits<__half>::to_float(h));
        if ((uint32_t)__half_as_ushort(h2) != want)
            atomicAdd(&bits[2], 1u);
    }
    {
        const uint32_t want = lpq::f32_to_fmt<lpq::FmtBF16>(x);
        const __nv_bfloat16 b = lpq::OutTraits<__nv_bfloat16>::from_float(x);
        if ((uint32_t)__bfloat16_as_ushort(b) != want)
            atomicAdd(&bits[3], 1u);
        if (lpq::OutTraits<__nv_bfloat16>::to_float(b) != lpq::f32_from_bf16_bits((uint16_t)want))
            atomicAdd(&bits[4], 1u);
    }
    {
        const __half lo = lpq::OutTraits<__half>::from_float(x);
        const __half hi = lpq::OutTraits<__half>::from_float(-x);
        const __half2 p = __halves2half2(lo, hi);
        if (__low2float(p) != lpq::InTraits<__half>::to_float(lo))
            atomicAdd(&bits[5], 1u);
        if (__high2float(p) != lpq::InTraits<__half>::to_float(hi))
            atomicAdd(&bits[6], 1u);
    }
}

// =============================================================================
//  [3b] 向量化访存: 每个线程处理 4 个连续元素
// =============================================================================
__global__ void k_vec_fp32(const float* in, float* out, int m) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m)
        return;
    float v[4];
    lpq::InTraits<float>::load4(in + (size_t)i * 4, v);
    lpq::OutTraits<float>::store4(out + (size_t)i * 4, v);
}

__global__ void k_vec_fp16(const float* in, __half* out, int m) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m)
        return;
    float v[4];
    lpq::InTraits<float>::load4(in + (size_t)i * 4, v);
    lpq::OutTraits<__half>::store4(out + (size_t)i * 4, v);
}

__global__ void k_vec_bf16(const float* in, __nv_bfloat16* out, int m) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m)
        return;
    float v[4];
    lpq::InTraits<float>::load4(in + (size_t)i * 4, v);
    lpq::OutTraits<__nv_bfloat16>::store4(out + (size_t)i * 4, v);
}
#endif  // LPQ_PROBE_SKIP_FP16

// =============================================================================
//  主机侧工具
// =============================================================================
static void banner(const char* title) {
    std::printf("\n---- %s ----\n", title);
    std::fflush(stdout);
}

static std::string arch_bitmask_str(int m) {
    std::string s;
    if (m & 1)
        s += "__CUDA_ARCH__ ";
    if (m & 2)
        s += "__MUSA_ARCH__ ";
    if (m & 4)
        s += "__MACA_ARCH__ ";
    if (s.empty())
        s = "(none)";
    return s;
}

int main() {
    std::printf("==================== portability probe ====================\n");

    // ---------------------------- 主机编译遍 ----------------------------
    banner("macros seen by the host compilation pass");
    std::printf("  LPQ_CUDA        = %d   (1 = recognised as a device compiler)\n", (int)LPQ_CUDA);
    std::printf("  LPQ_DEVICE_PASS = %d   (expected 0 on the host pass)\n", (int)LPQ_DEVICE_PASS);
#if defined(__CUDACC__)
    std::printf("  __CUDACC__      defined\n");
#endif
#if defined(__MUSACC__)
    std::printf("  __MUSACC__      defined\n");
#endif
#if defined(__MUSA__)
    std::printf("  __MUSA__        defined\n");
#endif
#if defined(__MACA__)
    std::printf("  __MACA__        defined\n");
#endif
#if defined(__clang__)
    std::printf("  host compiler   clang %d.%d.%d\n", __clang_major__, __clang_minor__,
        __clang_patchlevel__);
#elif defined(_MSC_VER)
    std::printf("  host compiler   MSVC %d\n", (int)_MSC_VER);
#elif defined(__GNUC__)
    std::printf("  host compiler   gcc %d.%d.%d\n", __GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__);
#endif

    // ---------------------------- 设备属性 ----------------------------
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) {
        std::printf("\n!! cudaGetDevice failed: %s\n", cudaGetErrorString(cudaGetLastError()));
        std::printf("RESULT: FAILED\n");
        return 1;
    }
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, dev) != cudaSuccess) {
        std::printf("\n!! cudaGetDeviceProperties failed\n");
        std::printf("RESULT: FAILED\n");
        return 1;
    }
    banner("device");
    std::printf("  name                = %s\n", prop.name);
    std::printf("  compute capability  = %d.%d\n", prop.major, prop.minor);
    std::printf("  SM count            = %d\n", prop.multiProcessorCount);
    std::printf("  global memory       = %.2f GiB\n",
        (double)prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    std::printf("  memory bandwidth    = %.1f GB/s (theoretical peak)\n",
        2.0 * (double)prop.memoryClockRate * 1e3 * ((double)prop.memoryBusWidth / 8.0) / 1e9);

    // ---------------------------- 设备遍宏探测 ----------------------------
    int h_mac[7];
    for (int i = 0; i < 7; ++i)
        h_mac[i] = -1;
    int* d_mac = nullptr;
    LPQ_CUDA_CHECK(cudaMalloc(&d_mac, sizeof(h_mac)));
    k_report_macros<<<1, 1>>>(d_mac);
    LPQ_CUDA_CHECK_KERNEL();
    LPQ_CUDA_CHECK(cudaMemcpy(h_mac, d_mac, sizeof(h_mac), cudaMemcpyDeviceToHost));
    LPQ_CUDA_CHECK(cudaFree(d_mac));

    banner("macros seen by the device compilation pass");
    std::printf("  warpSize        = %d\n", h_mac[0]);
    std::printf("  LPQ_CUDA        = %d   (expected 1 on the device pass)\n", h_mac[1]);
    std::printf("  LPQ_DEVICE_PASS = %d   (expected 1 on the device pass)\n", h_mac[2]);
    std::printf("  arch macros     = %s\n", arch_bitmask_str(h_mac[3]).c_str());
    std::printf("  arch values     = CUDA:%d MUSA:%d MACA:%d\n", h_mac[4], h_mac[5], h_mac[6]);

    testfw::begin_section("probe/macro-detection");
    CHECK_MSG(h_mac[1] == 1, "LPQ_CUDA must be 1 on the device pass; add -DLPQ_CUDA=1");
    CHECK_MSG(
        h_mac[2] == 1, "LPQ_DEVICE_PASS must be 1 on the device pass; add -DLPQ_DEVICE_PASS=1");
    testfw::end_section();

    // ---------------------------- [4] warp 内交换 ----------------------------
    testfw::begin_section("probe/shfl-xor");
    {
        const int kThreads = 256;
        unsigned int h_out[kThreads];
        unsigned int* d_out = nullptr;
        LPQ_CUDA_CHECK(cudaMalloc(&d_out, sizeof(h_out)));
        const int widths[6] = {1, 2, 4, 8, 16, 32};
        for (int wi = 0; wi < 6; ++wi) {
            const int w = widths[wi];
            std::memset(h_out, 0, sizeof(h_out));
            LPQ_CUDA_CHECK(cudaMemcpy(d_out, h_out, sizeof(h_out), cudaMemcpyHostToDevice));
            k_test_shfl<<<1, kThreads>>>(w, d_out);
            LPQ_CUDA_CHECK_KERNEL();
            LPQ_CUDA_CHECK(cudaMemcpy(h_out, d_out, sizeof(h_out), cudaMemcpyDeviceToHost));
            unsigned int expect = 0;
            for (int i = 0; i < w; ++i)
                expect ^= (unsigned int)i;
            long long bad = 0;
            for (int t = 0; t < kThreads; ++t)
                if (h_out[t] != expect)
                    ++bad;
            CHECK_MSG(bad == 0, "width=" + testfw::s_of(w) + ": " + testfw::s_of(bad) +
                                    " of 256 lanes wrong, expected " + testfw::hex32(expect) +
                                    "; if __shfl_xor_sync does not compile, see LPQ_SHFL_XOR in "
                                    "include/lpq/platform.hpp (single point of change)");
        }
        LPQ_CUDA_CHECK(cudaFree(d_out));
    }
    testfw::end_section();

    // ---------------------------- [5] 浮点原子最大 ----------------------------
    testfw::begin_section("probe/atomic-max-f32");
    {
        const long long n = 8 * 256;
        float h_back = -1.0f;
        float* d_out = nullptr;
        LPQ_CUDA_CHECK(cudaMalloc(&d_out, sizeof(float)));
        LPQ_CUDA_CHECK(cudaMemset(d_out, 0, sizeof(float)));
        k_test_atomic_max<<<8, 256>>>(d_out);
        LPQ_CUDA_CHECK_KERNEL();
        LPQ_CUDA_CHECK(cudaMemcpy(&h_back, d_out, sizeof(float), cudaMemcpyDeviceToHost));
        LPQ_CUDA_CHECK(cudaFree(d_out));
        CHECK_MSG(h_back == (float)(n - 1),
            "got " + testfw::f_of(h_back) + ", expected " + testfw::f_of((double)(n - 1)) +
                "; if this does not compile add -DLPQ_NO_ATOMIC_MAX_INT=1");
    }
    testfw::end_section();

    // ---------------------------- [6] 双精度原子加 ----------------------------
    testfw::begin_section("probe/atomic-add-f64");
    {
        const long long n = 8 * 256;
        double h_back[2] = {-1.0, -1.0};
        double* d_out = nullptr;
        LPQ_CUDA_CHECK(cudaMalloc(&d_out, sizeof(h_back)));
        LPQ_CUDA_CHECK(cudaMemset(d_out, 0, sizeof(h_back)));
        k_test_atomic_add_f64<<<8, 256>>>(d_out);
        LPQ_CUDA_CHECK_KERNEL();
        LPQ_CUDA_CHECK(cudaMemcpy(h_back, d_out, sizeof(h_back), cudaMemcpyDeviceToHost));
        LPQ_CUDA_CHECK(cudaFree(d_out));
        CHECK_MSG(
            h_back[0] == (double)n, "summing 1.0 gave " + testfw::f_of(h_back[0]) + ", expected " +
                                        testfw::f_of((double)n) +
                                        "; if this does not compile add -DLPQ_NO_ATOMIC_ADD_F64=1");
        CHECK_MSG(h_back[1] == (double)n * 0.5, "summing 0.5 gave " + testfw::f_of(h_back[1]) +
                                                    ", expected " + testfw::f_of((double)n * 0.5));
    }
    testfw::end_section();

    // ---------------------------- [1][2] 位模式互转 ----------------------------
    testfw::begin_section("probe/bits-roundtrip");
    {
        const int n = 1024;
        std::vector<float> in(n);
        for (int i = 0; i < n; ++i) {
            const uint32_t u = 0x3F800000u + (uint32_t)i * 977u;
            std::memcpy(&in[i], &u, 4);
        }
        in[0] = 0.0f;
        in[1] = -0.0f;
        in[2] = 1.0f;
        in[3] = -1.0f;
        float* d_in = nullptr;
        uint32_t* d_bits = nullptr;
        float* d_out = nullptr;
        LPQ_CUDA_CHECK(cudaMalloc(&d_in, (size_t)n * 4));
        LPQ_CUDA_CHECK(cudaMalloc(&d_bits, (size_t)n * 4));
        LPQ_CUDA_CHECK(cudaMalloc(&d_out, (size_t)n * 4));
        LPQ_CUDA_CHECK(cudaMemcpy(d_in, in.data(), (size_t)n * 4, cudaMemcpyHostToDevice));
        k_test_bits<<<(n + 255) / 256, 256>>>(d_in, d_bits, n);
        LPQ_CUDA_CHECK_KERNEL();
        k_test_unbits<<<(n + 255) / 256, 256>>>(d_bits, d_out, n);
        LPQ_CUDA_CHECK_KERNEL();
        std::vector<float> back(n);
        LPQ_CUDA_CHECK(cudaMemcpy(back.data(), d_out, (size_t)n * 4, cudaMemcpyDeviceToHost));
        LPQ_CUDA_CHECK(cudaFree(d_in));
        LPQ_CUDA_CHECK(cudaFree(d_bits));
        LPQ_CUDA_CHECK(cudaFree(d_out));
        long long bad = 0;
        for (int i = 0; i < n; ++i) {
            uint32_t a, b;
            std::memcpy(&a, &in[i], 4);
            std::memcpy(&b, &back[i], 4);
            if (a != b)
                ++bad;
        }
        CHECK_MSG(bad == 0,
            testfw::s_of(bad) + " of 1024 bit patterns changed on round trip; " +
                "__float_as_uint / __uint_as_float may be unavailable on the device pass");
    }
    testfw::end_section();

#ifndef LPQ_PROBE_SKIP_FP16
    // ---------------------------- [3a] 低精度类型 ----------------------------
    testfw::begin_section("probe/lowp-types");
    {
        const int n = 1024;
        std::vector<float> in(n);
        for (int i = 0; i < n; ++i)
            in[i] = (float)((i % 97) - 48) * 0.137f;
        float* d_in = nullptr;
        uint32_t* d_bits = nullptr;
        uint32_t h_bits[7];
        std::memset(h_bits, 0, sizeof(h_bits));
        LPQ_CUDA_CHECK(cudaMalloc(&d_in, (size_t)n * 4));
        LPQ_CUDA_CHECK(cudaMalloc(&d_bits, sizeof(h_bits)));
        LPQ_CUDA_CHECK(cudaMemcpy(d_in, in.data(), (size_t)n * 4, cudaMemcpyHostToDevice));
        LPQ_CUDA_CHECK(cudaMemcpy(d_bits, h_bits, sizeof(h_bits), cudaMemcpyHostToDevice));
        k_test_lowp_scalar<<<(n + 255) / 256, 256>>>(d_in, d_bits, n);
        LPQ_CUDA_CHECK_KERNEL();
        LPQ_CUDA_CHECK(cudaMemcpy(h_bits, d_bits, sizeof(h_bits), cudaMemcpyDeviceToHost));
        LPQ_CUDA_CHECK(cudaFree(d_in));
        LPQ_CUDA_CHECK(cudaFree(d_bits));
        const char* names[7] = {"fp16 encode matches bit-exact reference",
            "fp16 decode matches bit-exact reference", "fp16 encode-decode is stable",
            "bf16 encode matches bit-exact reference", "bf16 decode matches bit-exact reference",
            "__half2 low element is correct", "__half2 high element is correct"};
        for (int i = 0; i < 7; ++i) {
            CHECK_MSG(h_bits[i] == 0,
                std::string(names[i]) + ": " + testfw::s_of(h_bits[i]) + " / " + testfw::s_of(n) +
                    " elements differ; check -DLPQ_FP16_HEADER / " + "-DLPQ_BF16_HEADER");
        }
    }
    testfw::end_section();

    // ---------------------------- [3b] 向量化访存 ----------------------------
    testfw::begin_section("probe/vec-io");
    {
        const int m = 256;  // 每个线程 4 个元素
        std::vector<float> in((size_t)m * 4);
        for (size_t i = 0; i < in.size(); ++i)
            in[i] = (float)((i % 61) - 30) * 0.371f;

        float* d_in = nullptr;
        float* d_out = nullptr;
        LPQ_CUDA_CHECK(cudaMalloc(&d_in, in.size() * 4));
        LPQ_CUDA_CHECK(cudaMalloc(&d_out, in.size() * 4));
        LPQ_CUDA_CHECK(cudaMemcpy(d_in, in.data(), in.size() * 4, cudaMemcpyHostToDevice));

        // fp32: load4 + store4 往返
        k_vec_fp32<<<(m + 63) / 64, 64>>>(d_in, d_out, m);
        LPQ_CUDA_CHECK_KERNEL();
        std::vector<float> back(in.size(), 0.0f);
        LPQ_CUDA_CHECK(cudaMemcpy(back.data(), d_out, in.size() * 4, cudaMemcpyDeviceToHost));
        long long bad32 = 0;
        for (size_t i = 0; i < in.size(); ++i)
            if (back[i] != in[i])
                ++bad32;
        CHECK_MSG(bad32 == 0,
            "float4 load4/store4 round trip: " + testfw::s_of(bad32) + " elements differ");

        // fp16: load4 + 16bit packed store4, 与位运算参考逐位比较
        std::vector<uint16_t> h16(in.size(), 0xFFFFu);
        uint16_t* d16 = nullptr;
        LPQ_CUDA_CHECK(cudaMalloc(&d16, in.size() * 2));
        LPQ_CUDA_CHECK(cudaMemcpy(d16, h16.data(), in.size() * 2, cudaMemcpyHostToDevice));
        k_vec_fp16<<<(m + 63) / 64, 64>>>(d_in, reinterpret_cast<__half*>(d16), m);
        LPQ_CUDA_CHECK_KERNEL();
        LPQ_CUDA_CHECK(cudaMemcpy(h16.data(), d16, in.size() * 2, cudaMemcpyDeviceToHost));
        long long bad16 = 0;
        for (size_t i = 0; i < in.size(); ++i)
            if ((uint32_t)h16[i] != lpq::f32_to_fmt<lpq::FmtF16>(in[i]))
                ++bad16;
        CHECK_MSG(bad16 == 0, "fp16 packed store: " + testfw::s_of(bad16) +
                                  " elements differ from the bit-exact reference");

        // bf16: 同一路径
        std::memset(h16.data(), 0xFF, in.size() * 2);
        LPQ_CUDA_CHECK(cudaMemcpy(d16, h16.data(), in.size() * 2, cudaMemcpyHostToDevice));
        k_vec_bf16<<<(m + 63) / 64, 64>>>(d_in, reinterpret_cast<__nv_bfloat16*>(d16), m);
        LPQ_CUDA_CHECK_KERNEL();
        LPQ_CUDA_CHECK(cudaMemcpy(h16.data(), d16, in.size() * 2, cudaMemcpyDeviceToHost));
        long long badbf = 0;
        for (size_t i = 0; i < in.size(); ++i)
            if ((uint32_t)h16[i] != lpq::f32_to_fmt<lpq::FmtBF16>(in[i]))
                ++badbf;
        CHECK_MSG(badbf == 0, "bf16 packed store: " + testfw::s_of(badbf) +
                                  " elements differ from the bit-exact reference");

        LPQ_CUDA_CHECK(cudaFree(d16));
        LPQ_CUDA_CHECK(cudaFree(d_in));
        LPQ_CUDA_CHECK(cudaFree(d_out));
    }
    testfw::end_section();
#else
    std::printf("\n[skipped] low-precision type and vector IO tests "\n
        "(LPQ_PROBE_SKIP_FP16 is defined)\n");
#endif

    return testfw::summary("test_portability");
}
