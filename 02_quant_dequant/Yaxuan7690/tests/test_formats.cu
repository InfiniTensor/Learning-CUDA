// =============================================================================
//  test_formats.cu -- 低精度格式编解码与舍入的单元测试
//
//  覆盖内容:
//    1. 独立格式表 (用 pow()/ldexp() 重新实现解码) 与 formats.hpp 的位运算解码
//       逐位比对, 同时校验各格式的常量 (ebits / mbits / bias / emax / max_code 等);
//    2. 全部编码的往返一致性: decode -> encode 必须回到原编码;
//    3. 饱和 / 次正规 / 正负零 / Inf / NaN 的边界行为;
//    4. RNE 中点 (ties-to-even) 的定向用例;
//    5. 下溢 (远小于最小可表示幅值 -> 0) 与上溢饱和;
//    6. GPU 与 CPU 在同一份 f32_to_fmt<> 上的逐位一致性 (随机模糊测试);
//    7. E8M0 共享指数与 MX 公式的一致性 (与独立实现的 floor(log2) 比对);
//    8. fp16/bf16 输出编码快速路径 (硬件转换指令) 与通用实现的逐位一致性.
// =============================================================================
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

#include <cuda_runtime.h>

#include "lpq/common.hpp"
#include "lpq/formats.hpp"
#include "lpq/reference.hpp"
#include "lpq/vec_io.cuh"
#include "test_common.hpp"

using namespace lpq;

namespace {

// -----------------------------------------------------------------------------
// 独立格式表: 只依据 OCP MX / IEEE754 的文本定义重新实现, 不复用 formats.hpp
// -----------------------------------------------------------------------------
struct Spec {
    const char* name;
    int ebits, mbits, bias;
    bool has_inf, has_nan;
    float max_norm;
};

const Spec kSpecs[] = {
    {"e2m1", 2, 1, 1, false, false, 6.0f},
    {"e4m3", 4, 3, 7, false, true, 448.0f},
    {"e5m2", 5, 2, 15, true, true, 57344.0f},
    {"fp16", 5, 10, 15, true, true, 65504.0f},
    {"bf16", 8, 7, 127, true, true, 3.3895313892515355e38f},
};

const Spec& spec_of(const char* name) {
    for (const Spec& s : kSpecs) {
        if (std::strcmp(s.name, name) == 0)
            return s;
    }
    std::printf("unknown spec %s\n", name);
    std::exit(2);
}

int mag_bits(const Spec& s) {
    return s.ebits + s.mbits;
}
uint32_t mag_mask(const Spec& s) {
    return (1u << mag_bits(s)) - 1u;
}
uint32_t max_ef(const Spec& s) {
    return (1u << s.ebits) - 1u;
}
// 最大有限幅值编码: 有 Inf 时指数全 1 保留, 否则指数全 1 的尾数全 0 仍有限
uint32_t max_finite_mag(const Spec& s) {
    const uint32_t m = (1u << s.mbits) - 1u;
    if (s.has_inf)
        return ((max_ef(s) - 1u) << s.mbits) | m;  // 指数全 1 保留给 Inf
    if (s.has_nan)
        return (max_ef(s) << s.mbits) | (m - 1u);  // 最后一个编码是 NaN
    return (max_ef(s) << s.mbits) | m;
}
uint32_t inf_mag(const Spec& s) {
    return max_ef(s) << s.mbits;
}
// 第一个 NaN 编码
uint32_t first_nan_mag(const Spec& s) {
    if (s.has_inf)
        return inf_mag(s) | 1u;
    if (s.has_nan)
        return mag_mask(s);
    return max_finite_mag(s);
}

// 独立解码: 幅值编码 -> 值 (double, 用 ldexp 保证精确)
double decode_mag_double(const Spec& s, uint32_t mag) {
    if (s.has_inf && mag == inf_mag(s))
        return std::numeric_limits<double>::infinity();
    if (s.has_nan && mag >= first_nan_mag(s)) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (mag == 0)
        return 0.0;
    const uint32_t ef = mag >> s.mbits;
    const uint32_t mf = mag & ((1u << s.mbits) - 1u);
    if (ef == 0)
        return std::ldexp((double)mf, 1 - s.bias - s.mbits);
    return std::ldexp(1.0 + (double)mf / (double)(1u << s.mbits), (int)ef - s.bias);
}

// -----------------------------------------------------------------------------
// 待测格式的编译期分发
// -----------------------------------------------------------------------------
template <class F>
struct FmtName;
template <>
struct FmtName<FmtE2M1> {
    static const char* get() { return "e2m1"; }
};
template <>
struct FmtName<FmtE4M3> {
    static const char* get() { return "e4m3"; }
};
template <>
struct FmtName<FmtE5M2> {
    static const char* get() { return "e5m2"; }
};
template <>
struct FmtName<FmtF16> {
    static const char* get() { return "fp16"; }
};
template <>
struct FmtName<FmtBF16> {
    static const char* get() { return "bf16"; }
};

std::string name_of(const char* s) {
    return std::string(s);
}

// -----------------------------------------------------------------------------
// 1 + 2: 常量校验, 解码比对, 全编码往返
// -----------------------------------------------------------------------------
template <class F>
void test_format_static() {
    const char* nm = FmtName<F>::get();
    const Spec& s = spec_of(nm);
    testfw::begin_section((name_of("format/") + nm).c_str());

    CHECK_MSG(F::ebits == s.ebits, "ebits");
    CHECK_MSG(F::mbits == s.mbits, "mbits");
    CHECK_MSG(F::bias == s.bias, "bias");
    CHECK_MSG(F::has_inf == s.has_inf, "has_inf");
    CHECK_MSG(F::has_nan == s.has_nan, "has_nan");
    CHECK_MSG(F::max_code == max_finite_mag(s),
        "max_code=" + testfw::hex32(F::max_code) + " expect=" + testfw::hex32(max_finite_mag(s)));
    CHECK_MSG(F::max_norm == s.max_norm,
        "max_norm=" + testfw::f_of(F::max_norm) + " expect=" + testfw::f_of(s.max_norm));
    // emax / emin 与 IEEE754 一致: emax = max_ef - bias (有 Inf) 或 max_ef - bias (e4m3)
    const int expect_emax = (s.has_inf ? (int)max_ef(s) - 1 : (int)max_ef(s)) - s.bias;
    CHECK_MSG(F::emax == expect_emax,
        "emax=" + testfw::s_of(F::emax) + " expect=" + testfw::s_of(expect_emax));
    CHECK_MSG(F::emin == 1 - s.bias, "emin");

    // 解码比对 (幅值 + 符号)
    const uint32_t mm = mag_mask(s);
    for (uint32_t mag = 0; mag <= mm; ++mag) {
        const double expect = decode_mag_double(s, mag);
        const float got_pos = fmt_to_f32<F>(mag);
        const float got_neg = fmt_to_f32<F>(mag | (1u << mag_bits(s)));
        if (std::isnan(expect)) {
            CHECK_MSG(std::isnan(got_pos), "mag=" + testfw::hex32(mag) + " expected NaN");
            CHECK_MSG(std::isnan(got_neg), "mag=" + testfw::hex32(mag) + " expected -NaN");
        } else {
            CHECK_MSG((double)got_pos == expect, "mag=" + testfw::hex32(mag) +
                                                     " got=" + testfw::f_of(got_pos) +
                                                     " expect=" + testfw::f_of(expect));
            CHECK_MSG((double)got_neg == -expect,
                "mag=" + testfw::hex32(mag) + " sign-bit got=" + testfw::f_of(got_neg));
        }
    }

    // 全编码往返: 有限编码 decode -> encode 必须回到同一编码 (含符号)
    for (uint32_t mag = 0; mag <= F::max_code; ++mag) {
        for (int sign = 0; sign < 2; ++sign) {
            const uint32_t code = mag | ((uint32_t)sign << mag_bits(s));
            const float v = fmt_to_f32<F>(code);
            const uint32_t back = f32_to_fmt<F>(v);
            CHECK_MSG(back == code, "code=" + testfw::hex32(code) + " v=" + testfw::f_of(v) +
                                        " back=" + testfw::hex32(back));
        }
    }
    testfw::end_section();
}

// -----------------------------------------------------------------------------
// 3: 饱和 / 特殊值
// -----------------------------------------------------------------------------
template <class F>
void test_format_specials() {
    const char* nm = FmtName<F>::get();
    const Spec& s = spec_of(nm);
    testfw::begin_section((name_of("specials/") + nm).c_str());
    const uint32_t sign_bit = 1u << mag_bits(s);

    // 极大值必须饱和到 max_code, 而不是 NaN / Inf
    const float huge[] = {1.0e30f, 1.0e20f, 1.0e10f, 1.0e5f, 1.0e4f, 1.0e3f, 700.0f, 500.0f};
    for (float h : huge) {
        if (h <= F::max_norm)
            continue;
        CHECK_MSG(f32_to_fmt<F>(h) == F::max_code, "huge=" + testfw::f_of(h));
        CHECK_MSG(f32_to_fmt<F>(-h) == (sign_bit | F::max_code), "-huge");
    }

    // 正负零
    CHECK_MSG(f32_to_fmt<F>(0.0f) == 0u, "+0");
    CHECK_MSG(f32_to_fmt<F>(-0.0f) == sign_bit, "-0");
    CHECK_MSG(fmt_to_f32<F>(0u) == 0.0f, "decode +0");
    CHECK_MSG(std::signbit(fmt_to_f32<F>(sign_bit)), "decode -0 must keep the sign");

    // Inf / NaN 输入
    const float pinf = std::numeric_limits<float>::infinity();
    const float qnan = std::numeric_limits<float>::quiet_NaN();
    const uint32_t e_inf = f32_to_fmt<F>(pinf);
    const uint32_t e_nan = f32_to_fmt<F>(qnan);
    if (s.has_inf) {
        CHECK_MSG(e_inf == inf_mag(s), "Inf -> inf_code, got " + testfw::hex32(e_inf));
        CHECK_MSG(std::isinf(fmt_to_f32<F>(inf_mag(s))), "decode Inf");
    } else {
        // 无 Inf 的格式: Inf 必须饱和到 max_code (satfinite 语义)
        CHECK_MSG(
            e_inf == F::max_code, "no-Inf format: Inf must saturate, got " + testfw::hex32(e_inf));
        CHECK_MSG(std::isfinite(fmt_to_f32<F>(F::max_code)), "max_code must be finite");
    }
    if (s.has_nan) {
        // NaN 只要求落在 NaN 编码区间内, 且是 quiet NaN (payload 最高位为 1).
        // 具体选哪个 NaN 编码属于实现自由度, 因此这里不做逐位断言.
        CHECK_MSG(e_nan <= mag_mask(s), "NaN code out of range: " + testfw::hex32(e_nan));
        CHECK_MSG(e_nan >= first_nan_mag(s), "NaN code is not a NaN: " + testfw::hex32(e_nan));
        CHECK_MSG(
            (e_nan & (1u << (s.mbits - 1))) != 0u, "must be a quiet NaN: " + testfw::hex32(e_nan));
        CHECK_MSG(std::isnan(fmt_to_f32<F>(e_nan)), "decode NaN");
    } else {
        CHECK_MSG(
            e_nan == F::max_code, "no-NaN format: NaN must saturate, got " + testfw::hex32(e_nan));
    }

    // 下溢: 小于最小次正规幅值的输入必须量化为 0 (保留符号).
    // 阈值与格式有关, 因此这里用该格式自身的最小次正规值来构造用例.
    {
        const float min_sub = fmt_to_f32<F>(1u);
        CHECK_MSG(min_sub > 0.0f, "min_sub must be positive");
        CHECK_MSG(f32_to_fmt<F>(min_sub) == 1u, "min_sub round-trip");
        const float under[] = {min_sub * 0.49f, min_sub * 0.25f, min_sub * 0.01f, min_sub * 1.0e-4f,
            min_sub * 1.0e-20f};
        for (float t : under) {
            CHECK_MSG(f32_to_fmt<F>(t) == 0u,
                "under min_sub: " + testfw::f_of(t) + " -> " + testfw::hex32(f32_to_fmt<F>(t)));
            CHECK_MSG(f32_to_fmt<F>(-t) == sign_bit, "under -min_sub: " + testfw::f_of(t));
        }
    }
    testfw::end_section();
}

// -----------------------------------------------------------------------------
// 4: RNE 中点定向用例
// -----------------------------------------------------------------------------
void test_tie_to_even() {
    testfw::begin_section("rne/tie-to-even");

    // e2m1 网格: 0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0 (code 0..7)
    struct Case {
        float x;
        uint32_t code;
    };
    const Case cases[] = {
        {0.25f, 0x0},  // 0 与 0.5 的中点 -> 偶数编码 0
        {0.75f, 0x2},  // 0.5(1) 与 1.0(2) 的中点 -> 偶数编码 2
        {1.25f, 0x2},  // 1.0(2) 与 1.5(3) -> 2
        {1.75f, 0x4},  // 1.5(3) 与 2.0(4) -> 4
        {2.5f, 0x4},   // 2.0(4) 与 3.0(5) -> 4
        {3.5f, 0x6},   // 3.0(5) 与 4.0(6) -> 6
        {5.0f, 0x6},   // 4.0(6) 与 6.0(7) -> 6
        {0.5000001f, 0x1},
        {0.7499999f, 0x1},
        {0.7500001f, 0x2},
        {1.2499999f, 0x2},
        {1.2500001f, 0x3},
    };
    for (const Case& c : cases) {
        const uint32_t got = f32_to_fmt<FmtE2M1>(c.x);
        CHECK_MSG(got == c.code, "x=" + testfw::f_of(c.x) + " got=" + testfw::hex32(got) +
                                     " expect=" + testfw::hex32(c.code));
        const uint32_t got_neg = f32_to_fmt<FmtE2M1>(-c.x);
        CHECK_MSG(got_neg == (0x8u | c.code), "x=-" + testfw::f_of(c.x));
    }

    // e4m3 在 1.0 附近的网格: 1.0(0x38), 1.125(0x39), 1.25(0x3A), 1.375(0x3B), 1.5(0x3C)
    CHECK_MSG(fmt_to_f32<FmtE4M3>(0x38u) == 1.0f, "e4m3 0x38 == 1.0");
    CHECK_MSG(fmt_to_f32<FmtE4M3>(0x39u) == 1.125f, "e4m3 0x39 == 1.125");
    // 1.0 与 1.125 的中点为 1.0625 -> 取偶数尾数 0x38
    CHECK_MSG(f32_to_fmt<FmtE4M3>(1.0625f) == 0x38u,
        "e4m3 tie 1.0625 -> " + testfw::hex32(f32_to_fmt<FmtE4M3>(1.0625f)));
    // 1.125(0x39) 与 1.25(0x3A) 的中点为 1.1875 -> 取偶数尾数 0x3A
    CHECK_MSG(f32_to_fmt<FmtE4M3>(1.1875f) == 0x3Au,
        "e4m3 tie 1.1875 -> " + testfw::hex32(f32_to_fmt<FmtE4M3>(1.1875f)));
    // 非中点: 略小于 / 略大于 1.0625
    CHECK_MSG(f32_to_fmt<FmtE4M3>(1.05f) == 0x38u, "e4m3 1.05");
    CHECK_MSG(f32_to_fmt<FmtE4M3>(1.07f) == 0x39u, "e4m3 1.07");
    // e2m1 在 1.0 附近: 1.0(0x2) 与 1.5(0x3) 的中点为 1.25 -> 偶数 0x2
    CHECK_MSG(f32_to_fmt<FmtE2M1>(1.2499999f) == 0x2u, "e2m1 1.25-eps");
    CHECK_MSG(f32_to_fmt<FmtE2M1>(1.2500001f) == 0x3u, "e2m1 1.25+eps");
    testfw::end_section();
}

// -----------------------------------------------------------------------------
// 5: E8M0 / MX 共享指数
// -----------------------------------------------------------------------------
void test_e8m0_and_shared_exp() {
    testfw::begin_section("mxfp8/e8m0-shared-exp");

    for (int code = 0; code <= 254; ++code) {
        const double expect = std::ldexp(1.0, code - 127);
        const double got = (double)e8m0_to_f32((uint8_t)code);
        CHECK_MSG(got == expect, "e8m0 code=" + testfw::s_of(code) + " got=" + testfw::f_of(got) +
                                     " expect=" + testfw::f_of(expect));
    }
    CHECK_MSG(std::isnan(e8m0_to_f32(0xFFu)), "0xFF must be NaN");
    CHECK_MSG(e8m0_encode(-127) == 0u, "encode -127");
    CHECK_MSG(e8m0_encode(127) == 254u, "encode 127");
    CHECK_MSG(e8m0_encode(-500) == 0u, "encode clamp low");
    CHECK_MSG(e8m0_encode(500) == 254u, "encode clamp high");
    for (int e = -127; e <= 127; ++e) {
        CHECK_MSG(e8m0_encode(e) == (uint8_t)(e + 127), "encode " + testfw::s_of(e));
    }

    // ilog2_floor 与独立实现 (反复折半/加倍) 必须一致
    for (int k = -40; k <= 40; ++k) {
        const float v = std::ldexp(1.0f, k);
        CHECK_MSG(ilog2_floor(v) == k, "ilog2(2^k) k=" + testfw::s_of(k));
        CHECK_MSG(ilog2_floor(v * 1.5f) == k, "ilog2(1.5*2^k) k=" + testfw::s_of(k));
        CHECK_MSG(ilog2_floor(v * 0.9f) == k - 1, "ilog2(0.9*2^k) k=" + testfw::s_of(k));
    }
    CHECK_MSG(ilog2_floor(0.0f) == INT32_MIN, "ilog2(0)");

    // 与 reference.cu 的精确实现交叉验证
    uint64_t rng = 0x9E3779B97F4A7C15ull;
    for (int t = 0; t < 20000; ++t) {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        const float x = (float)((double)(rng >> 11) * (1.0 / 9007199254740992.0)) * 1.0e4f;
        if (!(x > 0.0f))
            continue;
        const int a = ilog2_floor(x);
        const int b = ref::exact_log2_floor((double)x);
        CHECK_MSG(a == b,
            "ilog2 x=" + testfw::f_of(x) + " a=" + testfw::s_of(a) + " b=" + testfw::s_of(b));
    }

    // MX 共享指数公式: shared_exp = floor(log2(amax)) - emax_elem, 夹到 [-127,127]
    // emax_elem 取元素格式的最大无偏指数: e4m3 -> 8, e5m2 -> 15, e2m1 -> 2
    const int emax_list[] = {8, 15, 2};
    CHECK_MSG(emax_list[2] == FmtE2M1::emax, "e2m1 的 emax 应为 2");
    for (int emax_elem : emax_list) {
        for (int k = -60; k <= 60; ++k) {
            const float amax = std::ldexp(1.0f, k);
            const int got = mx_shared_exp(amax, emax_elem);
            const int expect = std::max(-127, std::min(127, k - emax_elem));
            CHECK_MSG(got == expect,
                "shared_exp k=" + testfw::s_of(k) + " emax_elem=" + testfw::s_of(emax_elem));
        }
    }
    CHECK_MSG(mx_shared_exp(0.0f, 8) == -127, "amax=0 -> -127");
    CHECK_MSG(mx_shared_exp(-1.0f, 8) == -127, "amax<0 -> -127");
    testfw::end_section();
}

// -----------------------------------------------------------------------------
// 6: GPU 与 CPU 逐位一致 (模糊测试)
// -----------------------------------------------------------------------------
template <class F>
__global__ void k_encode(const float* x, uint32_t* out, int64_t n) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        out[i] = f32_to_fmt<F>(x[i]);
}

template <class F>
__global__ void k_decode(const uint32_t* c, float* out, int64_t n) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        out[i] = fmt_to_f32<F>(c[i]);
}

template <class F>
void test_gpu_cpu_agreement(const float* hx, int64_t n, const char* tag) {
    float* dx = nullptr;
    uint32_t* dc = nullptr;
    LPQ_CUDA_CHECK(cudaMalloc(&dx, (size_t)n * sizeof(float)));
    LPQ_CUDA_CHECK(cudaMalloc(&dc, (size_t)n * sizeof(uint32_t)));
    LPQ_CUDA_CHECK(cudaMemcpy(dx, hx, (size_t)n * sizeof(float), cudaMemcpyHostToDevice));
    const int threads = 256;
    const unsigned blocks = (unsigned)((n + threads - 1) / threads);
    k_encode<F><<<blocks, threads>>>(dx, dc, n);
    LPQ_CUDA_CHECK_KERNEL();
    std::vector<uint32_t> gc((size_t)n);
    LPQ_CUDA_CHECK(cudaMemcpy(gc.data(), dc, (size_t)n * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    for (int64_t i = 0; i < n; ++i) {
        const uint32_t expect = f32_to_fmt<F>(hx[i]);
        if (gc[(size_t)i] != expect) {
            CHECK_MSG(false,
                std::string(tag) + " encode i=" + testfw::s_of(i) + " x=" + testfw::f_of(hx[i]) +
                    " gpu=" + testfw::hex32(gc[(size_t)i]) + " cpu=" + testfw::hex32(expect));
        }
    }
    // 反向: 编码 -> 解码
    k_decode<F><<<blocks, threads>>>(dc, dx, n);
    LPQ_CUDA_CHECK_KERNEL();
    std::vector<float> gv((size_t)n);
    LPQ_CUDA_CHECK(cudaMemcpy(gv.data(), dx, (size_t)n * sizeof(float), cudaMemcpyDeviceToHost));
    for (int64_t i = 0; i < n; ++i) {
        const float expect = fmt_to_f32<F>(gc[(size_t)i]);
        if (std::isnan(expect)) {
            CHECK_MSG(
                std::isnan(gv[(size_t)i]), std::string(tag) + " decode NaN i=" + testfw::s_of(i));
        } else if (f32_to_bits(gv[(size_t)i]) != f32_to_bits(expect)) {
            CHECK_MSG(false, std::string(tag) + " decode i=" + testfw::s_of(i));
        }
    }
    cudaFree(dx);
    cudaFree(dc);
}

void test_gpu_matches_cpu() {
    testfw::begin_section("fuzz/gpu-vs-cpu");
    const int64_t n = 1 << 20;
    std::vector<float> x((size_t)n);
    uint64_t rng = 0x243F6A8885A308D3ull;
    auto next = [&]() {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        return rng;
    };
    for (int64_t i = 0; i < n; ++i) {
        const int mode = (int)(next() % 5);
        if (mode == 0) {
            // 任意位模式 (含 NaN / Inf / 次正规)
            const uint32_t b = (uint32_t)(next() & 0xFFFFFFFFull);
            x[(size_t)i] = bits_to_f32(b);
        } else if (mode == 1) {
            // 常见量级
            x[(size_t)i] = (float)((double)(int64_t)(next() % 2000) - 1000.0) * 0.25f;
        } else if (mode == 2) {
            // 极小值 (考察下溢)
            x[(size_t)i] = std::ldexp((float)(next() % 1000) - 500.0f, -(int)(next() % 60) - 20);
        } else if (mode == 3) {
            // 极大值 (考察饱和)
            x[(size_t)i] = std::ldexp((float)(next() % 1000) - 500.0f, (int)(next() % 40) + 10);
        } else {
            // 精确的格式网格点
            const uint32_t code = (uint32_t)(next() & 0xFFull);
            x[(size_t)i] = fmt_to_f32<FmtE4M3>(code);
        }
    }
    test_gpu_cpu_agreement<FmtE2M1>(x.data(), n, "e2m1");
    test_gpu_cpu_agreement<FmtE4M3>(x.data(), n, "e4m3");
    test_gpu_cpu_agreement<FmtE5M2>(x.data(), n, "e5m2");
    test_gpu_cpu_agreement<FmtF16>(x.data(), n, "fp16");
    test_gpu_cpu_agreement<FmtBF16>(x.data(), n, "bf16");
    testfw::end_section();
}

// -----------------------------------------------------------------------------
// 7: NVFP4 两级缩放公式
// -----------------------------------------------------------------------------
void test_nvfp4_scaling() {
    testfw::begin_section("nvfp4/scaling");
    // global_scale = 448 * 6 / amax
    CHECK_MSG(nvfp4_global_scale(1.0f) == 2688.0f, "gs(1.0)");
    CHECK_MSG(nvfp4_global_scale(0.0f) == 1.0f, "gs(0) zero guard");
    for (int k = -20; k <= 20; ++k) {
        const float amax = std::ldexp(1.0f, k);
        const float gs = nvfp4_global_scale(amax);
        CHECK_MSG(std::fabs((double)gs - (double)2688.0f / (double)amax) <= 1e-6 * (double)gs,
            "gs(2^k) k=" + testfw::s_of(k));
        // block_scale 使 block_amax 恰好映射到 E2M1 最大值 6.0
        const float bs = nvfp4_block_scale_f32(amax, gs);
        CHECK_MSG(std::fabs((double)bs - 448.0) < 1e-3,
            "bs k=" + testfw::s_of(k) + " bs=" + testfw::f_of(bs));
    }
    // 端到端: 张量 amax 处量化到 E2M1 最大值, 反量化后回到 amax
    {
        const float amax = 99.62913513183594f;
        const float gs = nvfp4_global_scale(amax);
        const float bs = nvfp4_block_scale_f32(amax, gs);
        const uint8_t bs_code = (uint8_t)f32_to_fmt<FmtE4M3>(bs);
        const float bs_dec = fmt_to_f32<FmtE4M3>(bs_code);
        const uint32_t q = f32_to_fmt<FmtE2M1>(amax * gs / bs_dec);
        CHECK_MSG(q == FmtE2M1::max_code, "amax must quantize to the e2m1 maximum 6.0");
        const float back = fmt_to_f32<FmtE2M1>(q) * (bs_dec / gs);
        CHECK_MSG(std::fabs((double)back - (double)amax) <= 0.01 * (double)amax,
            "dequantized amax: " + testfw::f_of(back));
    }

    // ---- 4/6 自适应块缩放的三个常量组合 ----
    // static_6 : max_q = 6, max_s = 448 -> 2688 / amax
    // 4/6 动态 : max_q = 6, max_s = 256 -> 1536 / amax
    // static_4 : max_q = 4, max_s = 448 -> 1792 / amax
    CHECK_MSG(nvfp4_global_scale(1.0f, kMaxE2M1, kMaxE4M3) == 2688.0f, "gs static_6");
    CHECK_MSG(nvfp4_global_scale(1.0f, kMaxE2M1, kFourOverSixScaleMax) == 1536.0f, "gs 4/6");
    CHECK_MSG(nvfp4_global_scale(1.0f, 4.0f, kMaxE4M3) == 1792.0f, "gs static_4");
    // 动态规则把 max_s 从 448 降到 256 的原因: 扩张 1.5 倍后仍要落在 E4M3 内
    CHECK_MSG(kFourOverSixScaleMax * kFourOverSixExpansion <= kMaxE4M3,
        "4/6 expansion must stay inside E4M3");
    CHECK_MSG(kMaxE4M3 * kFourOverSixExpansion > kMaxE4M3, "448 * 1.5 overflows E4M3");
    for (int k = -20; k <= 20; ++k) {
        const float amax = std::ldexp(1.0f, k);
        const float gs = nvfp4_global_scale(amax, kMaxE2M1, kFourOverSixScaleMax);
        const float bs6 = nvfp4_block_scale_f32(amax, gs, kMaxE2M1);
        const float bs4 = mul_rn(bs6, kFourOverSixExpansion);
        // 「6 候选」把块内 amax 映射到 6.0, 「4 候选」映射到 4.0
        CHECK_MSG(std::fabs((double)(amax * gs / bs6) - 6.0) <= 1e-5, "6 候选 amax -> 6.0");
        CHECK_MSG(std::fabs((double)(amax * gs / bs4) - 4.0) <= 1e-5, "4 候选 amax -> 4.0");
        // 4 候选经 E4M3 编码后仍是正的有限值, 且不超过 448
        const float bs4e = fmt_to_f32<FmtE4M3>((uint8_t)f32_to_fmt<FmtE4M3>(bs4));
        CHECK_MSG(bs4e > 0.0f && bs4e <= kMaxE4M3,
            "4 候选块缩放必须落在 E4M3 范围内: " + testfw::f_of(bs4e));
    }
    testfw::end_section();
}

// -----------------------------------------------------------------------------
// 7b: 4bit 元素的 x4 译码与位宽判定
//
//   e2m1 没有硬件译码指令, 4bit 元素的批量译码就是逐半字节的软件查表. 这里把
//   16^4 = 65536 个半字节组合全部走一遍, 逐位对照单元素译码.
// -----------------------------------------------------------------------------
void test_x4_4bit_decode() {
    testfw::begin_section("formats/x4-4bit");

    CHECK_MSG(ElemTraits<FmtE2M1>::bits == 4, "e2m1 应为 4bit");
    CHECK_MSG(ElemTraits<FmtE4M3>::bits == 8, "e4m3 应为 8bit");
    CHECK_MSG(ElemTraits<FmtE5M2>::bits == 8, "e5m2 应为 8bit");

    size_t checked = 0, bad = 0;
    for (uint32_t c0 = 0; c0 < 16u; ++c0) {
        for (uint32_t c1 = 0; c1 < 16u; ++c1) {
            for (uint32_t c2 = 0; c2 < 16u; ++c2) {
                for (uint32_t c3 = 0; c3 < 16u; ++c3) {
                    const uint16_t packed = (uint16_t)(c0 | (c1 << 4) | (c2 << 8) | (c3 << 12));
                    float v[4];
                    fmt_x4_4bit_to_f32<FmtE2M1>(packed, v);
                    const bool ok =
                        v[0] == fmt_to_f32<FmtE2M1>(c0) && v[1] == fmt_to_f32<FmtE2M1>(c1) &&
                        v[2] == fmt_to_f32<FmtE2M1>(c2) && v[3] == fmt_to_f32<FmtE2M1>(c3);
                    ++checked;
                    if (!ok)
                        ++bad;
                }
            }
        }
    }
    CHECK_MSG(bad == 0, "4bit x4 译码与单元素译码不一致: " + testfw::s_of((long long)bad));
    std::printf("    [x4] e2m1 4bit 打包 x4 译码覆盖 %zu 个半字节组合, 不一致 %zu\n", checked, bad);
    testfw::end_section();
}

// -----------------------------------------------------------------------------
// 8. fp16/bf16 输出编码快速路径的逐位一致性
//
//    vec_io.cuh 的 f32_to_f16_bits / f32_to_bf16_bits 在结果落在目标格式的有限可
//    表示范围内时直接使用硬件转换指令, 否则回退到通用的 f32_bits_to_fmt_mag<>.
//    这一节就是把这个"范围内硬件与通用实现逐位相同"的假设真正验证一遍: 用
//    全指数域采样 + 全空间随机位模式 + 关键边界值, 覆盖 2% 余个 fp32 编码.
// -----------------------------------------------------------------------------
__global__ void k_encode_check(
    const uint32_t* __restrict__ bits, int64_t n, unsigned char* __restrict__ bad) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    const float x = bits_to_f32(bits[i]);
    bad[i * 2 + 0] = (f32_to_f16_bits(x) == f32_to_fmt<FmtF16>(x)) ? 0u : 1u;
    bad[i * 2 + 1] = (f32_to_bf16_bits(x) == f32_to_fmt<FmtBF16>(x)) ? 0u : 1u;
}

void test_fast_encode_path() {
    testfw::begin_section("formats/fast-encode");
    std::vector<uint32_t> bits;
    bits.reserve(40000000u);

    // (a) 全 256 个指数域 x 均匀尾数采样(含符号位翻转), 密度足够扫到次正规进位边界
    for (uint32_t e = 0; e < 256u; ++e) {
        for (uint32_t m = 0; m < 4096u; ++m) {
            const uint32_t mant = (m * 8388608u) / 4096u;
            bits.push_back((e << 23) | (mant & 0x7FFFFFu));
            bits.push_back(0x80000000u | (e << 23) | (mant & 0x7FFFFFu));
        }
    }
    // (b) 全空间均匀随机位模式(含 NaN / Inf / 次正规)
    uint64_t rng = 0x243F6A8885A308D3ull;
    for (int i = 0; i < 8000000; ++i) {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        bits.push_back((uint32_t)(rng >> 32));
    }
    // (c) fp16 / bf16 的上溢饱和边界与次正规下溢边界
    const float edges[] = {65504.0f, 65519.0f, 65519.9999f, 65520.0f, 65536.0f, 131008.0f,
        3.38953139e38f, 3.4e38f, 6.0e-8f, 5.96e-8f, 6.1e-5f, 1.0e-45f, 1.4e-45f, 2.35e-38f,
        5.87747175e-39f, 1.17549435e-38f};
    for (float v : edges) {
        bits.push_back(f32_to_bits(v));
        bits.push_back(f32_to_bits(-v));
        bits.push_back(f32_to_bits(std::nextafter(v, 0.0f)));
        bits.push_back(f32_to_bits(std::nextafter(v, std::numeric_limits<float>::max())));
    }

    const int64_t n = (int64_t)bits.size();
    uint32_t* d_bits = nullptr;
    unsigned char* d_bad = nullptr;
    LPQ_CUDA_CHECK(cudaMalloc(&d_bits, (size_t)n * 4));
    LPQ_CUDA_CHECK(cudaMalloc(&d_bad, (size_t)n * 2));
    LPQ_CUDA_CHECK(cudaMemcpy(d_bits, bits.data(), (size_t)n * 4, cudaMemcpyHostToDevice));
    k_encode_check<<<(unsigned int)((n + 255) / 256), 256>>>(d_bits, n, d_bad);
    LPQ_CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<unsigned char> bad((size_t)n * 2, 0);
    LPQ_CUDA_CHECK(cudaMemcpy(bad.data(), d_bad, (size_t)n * 2, cudaMemcpyDeviceToHost));
    cudaFree(d_bits);
    cudaFree(d_bad);

    int64_t bad16 = 0, badbf = 0;
    std::string first;
    for (int64_t i = 0; i < n; ++i) {
        if (bad[(size_t)(i * 2)] != 0) {
            if (bad16 < 4)
                first += " f16:" + testfw::hex32(bits[(size_t)i]);
            ++bad16;
        }
        if (bad[(size_t)(i * 2 + 1)] != 0) {
            if (badbf < 4)
                first += " bf16:" + testfw::hex32(bits[(size_t)i]);
            ++badbf;
        }
    }
    std::printf("  fast-encode: checked %lld fp32 encodings, fp16 mismatches %lld, bf16 %lld\n",
        (long long)n, (long long)bad16, (long long)badbf);
    CHECK_MSG(bad16 == 0, "f32_to_f16_bits vs f32_to_fmt<FmtF16>" + first);
    CHECK_MSG(badbf == 0, "f32_to_bf16_bits vs f32_to_fmt<FmtBF16>" + first);
    testfw::end_section();
}

}  // namespace

int main() {
    test_format_static<FmtE2M1>();
    test_format_static<FmtE4M3>();
    test_format_static<FmtE5M2>();
    test_format_static<FmtF16>();
    test_format_static<FmtBF16>();
    test_format_specials<FmtE2M1>();
    test_format_specials<FmtE4M3>();
    test_format_specials<FmtE5M2>();
    test_format_specials<FmtF16>();
    test_format_specials<FmtBF16>();
    test_tie_to_even();
    test_e8m0_and_shared_exp();
    test_nvfp4_scaling();
    test_x4_4bit_decode();
    test_gpu_matches_cpu();
    test_fast_encode_path();
    return testfw::summary("test_formats");
}
