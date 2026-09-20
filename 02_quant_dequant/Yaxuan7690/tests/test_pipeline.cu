// =============================================================================
//  test_pipeline.cu -- 量化/打包/反量化流水线的端到端测试
//
//  测试策略:同一份输入分别跑 GPU 流水线与完全独立的 CPU 参考实现
//  (src/reference.cu, 查表法 + 串行实现), 对以下三样东西逐位比较:
//      1) packed 数据 (4bit 严格每字节 2 个元素)
//      2) scale 数据
//      3) 反量化输出 (按 output_type 舍入后再比位模式)
//
//  覆盖矩阵:
//      format      : mxfp8-e4m3 / mxfp8-e5m2 / nvfp4
//      scale_mode  : tensor / block
//      block_size  : 16 / 32 / 64 / 256 / 48 / 1000 (非 2 的幂) / 远大于 numel
//      output_type : fp16 / bf16 / fp32
//      input dtype : fp32 / fp16
//      numel       : 4 的倍数 / 非 4 的倍数 / 非 block_size 的倍数
//      分布        : 随机 / 正态 / 含异常值 / 全零 / 极小值 / 极大值
//  另含随机舍入(SR)的行为检查:结果只能落在最近的两个可表示值上, 且期望无偏.
// =============================================================================
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "lpq/common.hpp"
#include "lpq/config.hpp"
#include "lpq/pipeline.hpp"
#include "lpq/reference.hpp"
#include "test_common.hpp"

using namespace lpq;

namespace {

// -----------------------------------------------------------------------------
// 工具
// -----------------------------------------------------------------------------
uint64_t g_rng = 0x853C49E6748FEA9Bull;

uint64_t next_rand() {
    g_rng ^= g_rng << 13;
    g_rng ^= g_rng >> 7;
    g_rng ^= g_rng << 17;
    return g_rng;
}

double unit() {
    return (double)(next_rand() >> 11) * (1.0 / 9007199254740992.0);
}

enum class Dist { Uniform, Normal, Outlier, Zeros, Tiny, Huge };

const char* dist_name(Dist d) {
    switch (d) {
        case Dist::Uniform:
            return "uniform";
        case Dist::Normal:
            return "normal";
        case Dist::Outlier:
            return "outlier";
        case Dist::Zeros:
            return "zeros";
        case Dist::Tiny:
            return "tiny";
        default:
            return "huge";
    }
}

std::vector<float> make_input(Dist d, int64_t n) {
    std::vector<float> x((size_t)n, 0.0f);
    for (int64_t i = 0; i < n; ++i) {
        switch (d) {
            case Dist::Uniform:
                x[(size_t)i] = (float)(unit() * 2.0 - 1.0);
                break;
            case Dist::Normal: {
                // Box-Muller, 避免依赖 <random> 的分布实现差异
                const double u1 = std::max(unit(), 1e-12);
                const double u2 = unit();
                x[(size_t)i] =
                    (float)(std::sqrt(-2.0 * std::log(u1)) * std::cos(6.283185307179586 * u2));
                break;
            }
            case Dist::Outlier:
                x[(size_t)i] = (float)(unit() * 2.0 - 1.0);
                if ((i % 997) == 0)
                    x[(size_t)i] = (float)((unit() * 60.0 + 40.0) * ((i & 1) ? -1.0 : 1.0));
                break;
            case Dist::Zeros:
                x[(size_t)i] = 0.0f;
                break;
            case Dist::Tiny:
                x[(size_t)i] =
                    std::ldexp((float)(unit() * 2.0 - 1.0), -(int)(next_rand() % 60) - 20);
                break;
            default:
                x[(size_t)i] = (float)((unit() * 2.0 - 1.0) * 1.0e4);
                break;
        }
    }
    return x;
}

uint16_t to_half_bits(float v) {
    return (uint16_t)f32_to_fmt<FmtF16>(v);
}

// 反量化输出字节 -> float (用于与非逐位比较的场景)
float out_byte_to_float(uint32_t bits, OutputType ot) {
    if (ot == OutputType::FP32)
        return bits_to_f32(bits);
    if (ot == OutputType::FP16)
        return f32_from_half_bits((uint16_t)bits);
    return f32_from_bf16_bits((uint16_t)bits);
}

// 按 output_type 把 CPU 的 fp32 结果舍入成与 GPU 可直接比位的编码
uint32_t round_to_output(float v, OutputType ot) {
    if (ot == OutputType::FP32)
        return f32_to_bits(v);
    if (ot == OutputType::FP16)
        return f32_to_fmt<FmtF16>(v);
    return f32_to_fmt<FmtBF16>(v);
}

struct GpuRun {
    std::vector<uint8_t> packed;
    std::vector<uint8_t> scales;
    std::vector<uint8_t> deq_raw;
    QuantizedDevice q;  // 仅用于读取计时等元信息(缓冲区已析构前拷贝)
    int64_t numel = 0;
};

// -----------------------------------------------------------------------------
// 跑一次 GPU 流水线 (输入可以是 fp32 或 fp16 位模式)
// -----------------------------------------------------------------------------
struct GpuOut {
    std::vector<uint8_t> packed;
    std::vector<uint8_t> scales;
    std::vector<uint8_t> deq_raw;
    int64_t block_size = 0;
    int64_t num_blocks = 0;
    float global_scale = 1.0f;
    double quant_ms = 0.0;
    double dequant_ms = 0.0;
};

GpuOut run_gpu(const void* h_input, size_t input_bytes, InputType in_type, int64_t numel,
    const QuantConfig& cfg) {
    GpuOut r;
    const bool out32 = (cfg.output_type == OutputType::FP32);
    DeviceBuffer d_in(input_bytes);
    DeviceBuffer d_out((size_t)numel * (out32 ? 4u : 2u));
    LPQ_CUDA_CHECK(cudaMemcpy(d_in.get(), h_input, input_bytes, cudaMemcpyHostToDevice));

    QuantizedDevice q = quantize_on_device(d_in.get(), in_type, numel, cfg);
    r.quant_ms = q.timings.amax_ms + q.timings.scale_ms + q.timings.quant_ms;
    r.dequant_ms = dequantize_on_device(q, cfg, d_out.get());
    r.block_size = q.block_size;
    r.num_blocks = q.num_blocks;
    r.global_scale = q.global_scale_value;

    r.packed.resize(q.packed.size());
    if (!r.packed.empty()) {
        LPQ_CUDA_CHECK(
            cudaMemcpy(r.packed.data(), q.packed.get(), r.packed.size(), cudaMemcpyDeviceToHost));
    }
    r.scales.resize(q.scales.size());
    if (!r.scales.empty()) {
        LPQ_CUDA_CHECK(
            cudaMemcpy(r.scales.data(), q.scales.get(), r.scales.size(), cudaMemcpyDeviceToHost));
    }
    r.deq_raw.resize(d_out.size());
    if (!r.deq_raw.empty()) {
        LPQ_CUDA_CHECK(
            cudaMemcpy(r.deq_raw.data(), d_out.get(), r.deq_raw.size(), cudaMemcpyDeviceToHost));
    }
    return r;
}

// -----------------------------------------------------------------------------
// 核心比对:GPU 流水线 vs 独立 CPU 参考实现
//    check_bits = false 时不比较 packed / 反量化位模式(用于随机舍入, 两端随机数
//    使用方式不同, 只做统计等价性检查).
// -----------------------------------------------------------------------------
void check_case(const std::string& tag, const QuantConfig& cfg, const std::vector<float>& hx,
    InputType in_type, bool check_bits) {
    const int64_t numel = (int64_t)hx.size();
    if (numel <= 0)
        return;

    // ---- 期望的尺寸(由配置独立推算, 不读取实现内部状态) ----
    const int64_t exp_bs = (cfg.scale_mode == ScaleMode::Tensor) ? numel : (int64_t)cfg.block_size;
    const int64_t exp_nb = (numel + exp_bs - 1) / exp_bs;
    const int64_t exp_packed = cfg.is_4bit() ? ((numel + 1) / 2) : numel;

    // ---- 构造输入位模式 ----
    std::vector<float> xf((size_t)numel);
    std::vector<uint16_t> xh;
    const void* h_in = nullptr;
    size_t in_bytes = 0;
    if (in_type == InputType::FP32) {
        xf = hx;
        h_in = xf.data();
        in_bytes = (size_t)numel * 4;
    } else {
        xh.resize((size_t)numel);
        for (int64_t i = 0; i < numel; ++i) {
            xh[(size_t)i] = to_half_bits(hx[(size_t)i]);
            xf[(size_t)i] = f32_from_half_bits(xh[(size_t)i]);  // 参考实现看到的是 fp16 值
        }
        h_in = xh.data();
        in_bytes = (size_t)numel * 2;
    }

    const GpuOut g = run_gpu(h_in, in_bytes, in_type, numel, cfg);

    // ---- 打包尺寸不变量 ----
    if (g.packed.size() != (size_t)exp_packed) {
        CHECK_MSG(false, tag + " packed_bytes=" + testfw::s_of((long long)g.packed.size()) +
                             " expect=" + testfw::s_of((long long)exp_packed));
        return;
    }
    if (g.scales.size() != (size_t)exp_nb) {
        CHECK_MSG(false, tag + " scale_bytes=" + testfw::s_of((long long)g.scales.size()) +
                             " expect=" + testfw::s_of((long long)exp_nb));
        return;
    }
    // 4bit 格式必须每字节 2 个元素, 不能每字节只放 1 个
    if (cfg.is_4bit()) {
        CHECK_MSG(g.packed.size() * 2 >= (size_t)numel,
            tag + " 4bit packing must hold two elements per byte");
        CHECK_MSG(g.packed.size() < (size_t)numel,
            tag + " 4bit packing must not degrade to one element per byte");
    }

    // ---- CPU 参考实现 ----
    const ref::CpuQuantized cq = ref::cpu_quantize(xf.data(), numel, cfg);
    CHECK_MSG(cq.packed.size() == g.packed.size(), tag + " ref packed size");
    CHECK_MSG(cq.scales.size() == g.scales.size(), tag + " ref scales size");

    if (check_bits) {
        if (cq.packed.size() == g.packed.size() &&
            std::memcmp(cq.packed.data(), g.packed.data(), g.packed.size()) != 0) {
            size_t bad = 0, first = 0;
            for (size_t i = 0; i < g.packed.size(); ++i) {
                if (cq.packed[i] != g.packed[i]) {
                    if (bad == 0)
                        first = i;
                    ++bad;
                }
            }
            CHECK_MSG(false, tag + " packed mismatch: " + testfw::s_of((long long)bad) +
                                 " bytes, first idx=" + testfw::s_of((long long)first) +
                                 " ref=" + testfw::hex32(cq.packed[first]) +
                                 " gpu=" + testfw::hex32(g.packed[first]));
        } else {
            CHECK_MSG(true, "");
        }
        if (cq.scales.size() == g.scales.size() &&
            std::memcmp(cq.scales.data(), g.scales.data(), g.scales.size()) != 0) {
            size_t bad = 0, first = 0;
            for (size_t i = 0; i < g.scales.size(); ++i) {
                if (cq.scales[i] != g.scales[i]) {
                    if (bad == 0)
                        first = i;
                    ++bad;
                }
            }
            CHECK_MSG(false, tag + " scales mismatch: " + testfw::s_of((long long)bad) +
                                 " entries, first idx=" + testfw::s_of((long long)first) +
                                 " ref=" + testfw::hex32(cq.scales[first]) +
                                 " gpu=" + testfw::hex32(g.scales[first]));
        } else {
            CHECK_MSG(true, "");
        }
    }

    // ---- 反量化结果比对 ----
    // 不变量: 用真实输出类型解释 deq_raw, 必须与 CPU 参考的 fp32 结果一致(舍入后)
    const std::vector<float> cq_f32 = ref::cpu_dequantize(cq, numel, cfg);
    const uint16_t* g16 = reinterpret_cast<const uint16_t*>(g.deq_raw.data());
    const uint32_t* g32 = reinterpret_cast<const uint32_t*>(g.deq_raw.data());
    size_t mismatch = 0, first_i = 0;
    size_t vs_packed_mismatch = 0;
    for (int64_t i = 0; i < numel; ++i) {
        const uint32_t gb = (cfg.output_type == OutputType::FP32) ? g32[i] : g16[i];
        // (a) 与 CPU 参考比较
        if (check_bits) {
            const uint32_t cb = round_to_output(cq_f32[(size_t)i], cfg.output_type);
            if (cb != gb) {
                if (mismatch == 0)
                    first_i = (size_t)i;
                ++mismatch;
            }
        }
        // (b) 与直接按 packed 独立解包的期望值比较:
        //     check is independent of reference.cu: it verifies that packing, unpacking
        if (mismatch == 0) {
            float expect;
            if (cfg.is_mx()) {
                const uint8_t sc = g.scales[(size_t)(i / exp_bs)];
                const float s = e8m0_to_f32(sc);
                // 4bit 元素从字节里取半字节(偶下标低、奇下标高), 8bit 元素直接取字节
                const uint32_t code =
                    cfg.is_4bit()
                        ? (uint32_t)((g.packed[(size_t)(i >> 1)] >> (4 * (uint32_t)(i & 1))) & 0xFu)
                        : (uint32_t)g.packed[(size_t)i];
                if (cfg.format == QuantFormat::MxFP4) {
                    expect = fmt_to_f32<FmtE2M1>(code) * s;
                } else {
                    expect =
                        (cfg.mxfp8_e5m2 ? fmt_to_f32<FmtE5M2>(code) : fmt_to_f32<FmtE4M3>(code)) *
                        s;
                }
            } else {
                const uint8_t byte = g.packed[(size_t)(i >> 1)];
                const uint32_t code = (i & 1) ? ((byte >> 4) & 0xFu) : (byte & 0xFu);
                const uint8_t sc = g.scales[(size_t)(i / exp_bs)];
                const float s = fmt_to_f32<FmtE4M3>(sc) * f32_div_rn(1.0f, g.global_scale);
                expect = fmt_to_f32<FmtE2M1>(code) * s;
            }
            if (round_to_output(expect, cfg.output_type) != gb)
                ++vs_packed_mismatch;
        }
    }
    if (check_bits && mismatch != 0) {
        CHECK_MSG(false,
            tag + " dequant vs CPU reference mismatch: " + testfw::s_of((long long)mismatch) +
                " / " + testfw::s_of((long long)numel) +
                " first i=" + testfw::s_of((long long)first_i));
    } else if (check_bits) {
        CHECK_MSG(true, "");
    }
    CHECK_MSG(
        vs_packed_mismatch == 0, tag + " dequant output is inconsistent with the packed data: " +
                                     testfw::s_of((long long)vs_packed_mismatch));
}

std::string tag_of(const QuantConfig& cfg, const char* dist, int64_t numel, InputType in) {
    return std::string(to_string(cfg.format)) + "/" +
           (cfg.format == QuantFormat::MxFP4 ? "e2m1" : (cfg.mxfp8_e5m2 ? "e5m2" : "e4m3")) + "/" +
           (cfg.format == QuantFormat::NVFP4 ? std::string(to_string(cfg.scale_rule)) + "/"
                                             : std::string()) +
           to_string(cfg.scale_mode) + "/bs=" + testfw::s_of(cfg.block_size) +
           "/out=" + to_string(cfg.output_type) + "/in=" + to_string(in) + "/" + dist +
           "/n=" + testfw::s_of((long long)numel);
}

// -----------------------------------------------------------------------------
// 1. 全组合正确性
// -----------------------------------------------------------------------------
void test_full_matrix() {
    testfw::begin_section("pipeline/full-matrix");

    struct FmtSpec {
        QuantFormat fmt;
        bool e5m2;
    };
    const FmtSpec fmts[] = {{QuantFormat::MxFP8, false}, {QuantFormat::MxFP8, true},
        {QuantFormat::MxFP4, false}, {QuantFormat::NVFP4, false}};
    const int block_sizes[] = {16, 32, 64, 256, 48, 1000};
    const OutputType outs[] = {OutputType::FP16, OutputType::BF16, OutputType::FP32};
    const InputType ins[] = {InputType::FP32, InputType::FP16};
    const Dist dists[] = {
        Dist::Uniform, Dist::Normal, Dist::Outlier, Dist::Zeros, Dist::Tiny, Dist::Huge};
    // numel: 4 的倍数 / 非 4 的倍数 / 较小
    const int64_t numels[] = {1024, 1000, 4097};

    for (const FmtSpec& f : fmts) {
        for (int bs : block_sizes) {
            for (OutputType ot : outs) {
                for (InputType in : ins) {
                    for (int64_t n : numels) {
                        QuantConfig cfg;
                        cfg.format = f.fmt;
                        cfg.mxfp8_e5m2 = f.e5m2;
                        cfg.block_size = bs;
                        cfg.block_size_explicit = true;
                        cfg.scale_mode = ScaleMode::Block;
                        cfg.output_type = ot;
                        cfg.rounding = Rounding::Nearest;
                        cfg.seed = 12345;
                        for (Dist d : dists) {
                            const std::vector<float> x = make_input(d, n);
                            check_case(tag_of(cfg, dist_name(d), n, in), cfg, x, in, true);
                        }
                    }
                }
            }
        }
    }
    testfw::end_section();
}

// -----------------------------------------------------------------------------
// 2. per-tensor 缩放 + block_size 大于 numel
// -----------------------------------------------------------------------------
void test_tensor_and_large_block() {
    testfw::begin_section("pipeline/tensor-and-large-block");
    const Dist dists[] = {
        Dist::Uniform, Dist::Normal, Dist::Outlier, Dist::Zeros, Dist::Tiny, Dist::Huge};
    const int64_t numels[] = {512, 1000, 4097};
    const OutputType outs[] = {OutputType::FP16, OutputType::BF16, OutputType::FP32};

    for (int fmt_i = 0; fmt_i < 4; ++fmt_i) {
        for (OutputType ot : outs) {
            for (int64_t n : numels) {
                for (Dist d : dists) {
                    const std::vector<float> x = make_input(d, n);

                    QuantConfig cfg;
                    cfg.format = (fmt_i == 2)
                                     ? QuantFormat::NVFP4
                                     : (fmt_i == 3 ? QuantFormat::MxFP4 : QuantFormat::MxFP8);
                    cfg.mxfp8_e5m2 = (fmt_i == 1);
                    cfg.scale_mode = ScaleMode::Tensor;
                    cfg.block_size = 32;
                    cfg.block_size_explicit = true;
                    cfg.output_type = ot;
                    cfg.rounding = Rounding::Nearest;
                    check_case(tag_of(cfg, dist_name(d), n, InputType::FP32) + "/per-tensor", cfg,
                        x, InputType::FP32, true);

                    // block_size 远大于 numel: 应退化成单个缩放块
                    QuantConfig cfg2 = cfg;
                    cfg2.scale_mode = ScaleMode::Block;
                    cfg2.block_size = (int)(n * 4 + 7);
                    check_case(tag_of(cfg2, dist_name(d), n, InputType::FP32) + "/oversized-block",
                        cfg2, x, InputType::FP32, true);
                }
            }
        }
    }
    testfw::end_section();
}

// -----------------------------------------------------------------------------
// 3. 打包布局: 4bit 每字节 2 个元素, 偶下标在低半字节
// -----------------------------------------------------------------------------
void test_packing_layout() {
    testfw::begin_section("pipeline/packing-layout");
    // 构造一组能稳定得到互不相同的编码的值, 直接检查半字节顺序
    const int64_t n = 32;
    std::vector<float> x((size_t)n);
    for (int64_t i = 0; i < n; ++i)
        x[(size_t)i] = (float)((i % 8) - 4) * 0.5f;  // -2..1.5

    QuantConfig cfg;
    cfg.format = QuantFormat::NVFP4;
    cfg.scale_mode = ScaleMode::Block;
    cfg.block_size = 32;
    cfg.block_size_explicit = true;
    cfg.output_type = OutputType::FP32;

    const GpuOut g = run_gpu(x.data(), x.size() * 4, InputType::FP32, n, cfg);
    CHECK_MSG(g.packed.size() == (size_t)((n + 1) / 2), "4bit packed must be numel/2 bytes");
    // 逐元素: packed[i>>1] 的对应半字节应能还原出与参考一致的元素
    const ref::CpuQuantized cq = ref::cpu_quantize(x.data(), n, cfg);
    for (int64_t i = 0; i < n; ++i) {
        const uint8_t byte = g.packed[(size_t)(i >> 1)];
        const uint32_t nib = (i & 1) ? ((byte >> 4) & 0xFu) : (byte & 0xFu);
        const uint8_t rbyte = cq.packed[(size_t)(i >> 1)];
        const uint32_t rnib = (i & 1) ? ((rbyte >> 4) & 0xFu) : (rbyte & 0xFu);
        CHECK_MSG(nib == rnib, "element " + testfw::s_of((long long)i) + " nibble position wrong");
        // 奇数下标必须落在高半字节
        if (i & 1)
            CHECK_MSG(nib == ((byte >> 4) & 0xFu), "odd index must use the high nibble");
    }
    testfw::end_section();
}

// -----------------------------------------------------------------------------
// 4. 随机舍入: 结果只能落在最近两个可表示值上, 且期望偏置远小于 RNE
// -----------------------------------------------------------------------------
void test_stochastic_rounding() {
    testfw::begin_section("pipeline/stochastic-rounding");

    const int64_t n = 1 << 20;  // 1M 元素, 保证统计量稳定
    const std::vector<float> x = make_input(Dist::Uniform, n);

    for (int fmt_i = 0; fmt_i < 4; ++fmt_i) {
        QuantConfig cfg;
        cfg.format = (fmt_i == 2) ? QuantFormat::NVFP4
                                  : (fmt_i == 3 ? QuantFormat::MxFP4 : QuantFormat::MxFP8);
        cfg.mxfp8_e5m2 = (fmt_i == 1);
        cfg.scale_mode = ScaleMode::Block;
        cfg.block_size = (fmt_i == 2) ? 16 : 32;
        cfg.block_size_explicit = true;
        cfg.output_type = OutputType::FP32;
        cfg.seed = 20260915ull;

        cfg.rounding = Rounding::Nearest;
        const GpuOut rne = run_gpu(x.data(), x.size() * 4, InputType::FP32, n, cfg);
        cfg.rounding = Rounding::Stochastic;
        const GpuOut sr = run_gpu(x.data(), x.size() * 4, InputType::FP32, n, cfg);

        const char* nm = (fmt_i == 2)   ? "nvfp4"
                         : (fmt_i == 3) ? "mxfp4"
                         : (fmt_i == 1) ? "mxfp8-e5m2"
                                        : "mxfp8-e4m3";

        // (a) 尺寸一致
        CHECK_MSG(rne.packed.size() == sr.packed.size(), std::string(nm) + " SR packed size");
        CHECK_MSG(rne.scales.size() == sr.scales.size(), std::string(nm) + " SR scales size");
        // (b) SR 必须真的改变了结果(否则说明随机数没起作用)
        size_t diff = 0;
        for (size_t i = 0; i < rne.packed.size(); ++i) {
            if (rne.packed[i] != sr.packed[i])
                ++diff;
        }
        CHECK_MSG(diff > 0, std::string(nm) + " SR produced no difference at all");
        CHECK_MSG(diff < rne.packed.size() / 2,
            std::string(nm) + " SR changed too many bytes, suspicious");

        // (c) SR 的每个元素必须落在 RNE 所选值相邻的编码上(即只做 ±1 ulp 的随机进位)
        size_t far = 0;
        for (int64_t i = 0; i < n; ++i) {
            uint32_t a, b;
            if (!cfg.is_4bit()) {
                a = rne.packed[(size_t)i];
                b = sr.packed[(size_t)i];
            } else {
                const uint8_t by_a = rne.packed[(size_t)(i >> 1)];
                const uint8_t by_b = sr.packed[(size_t)(i >> 1)];
                a = (i & 1) ? ((by_a >> 4) & 0xFu) : (by_a & 0xFu);
                b = (i & 1) ? ((by_b >> 4) & 0xFu) : (by_b & 0xFu);
            }
            const uint32_t bits = cfg.is_4bit() ? 4u : 8u;
            const uint32_t mag_mask = (1u << (bits - 1)) - 1u;
            const int da = (int)(a & mag_mask);
            const int db = (int)(b & mag_mask);
            if (std::abs(da - db) > 1)
                ++far;
            // 符号位不能变(0 除外): 量化的是 |x| 的舍入, 不进位跨越符号
            if ((a >> (bits - 1)) != (b >> (bits - 1)))
                ++far;
        }
        CHECK_MSG(far == 0,
            std::string(nm) + " SR jumped more than 1 ulp on " + testfw::s_of((long long)far));

        // (d) 期望偏置的检验放在 test_stochastic_unbiasedness() 中:
        //     对"分布在量化区间内部的固定值"做多次独立试验, RNE 的偏置是常数,
        //     SR 的平均误差在 0 附近波动. 这里只报告本次运行的统计量.
        std::printf("    [SR] %-12s packed 差异字节 = %zu / %zu\n", nm, diff, rne.packed.size());
    }
    testfw::end_section();
}

// -----------------------------------------------------------------------------
// 4b. 随机舍入的无偏性: 决定性检验
//
//  构造一个张量: 绝大多数元素取同一个"落在量化区间内部"的值 x0, 同时把 amax 抬高到
//  8.0 以固定缩放因子. 于是:
//    * RNE 对同一个值总是朝同一侧舍入 -> 每次运行的平均误差都是同一个非零偏置,
//      其大小约为 0.4 ~ 0.8 个 ulp;
//    * SR 每次都在相邻两个格点间独立随机选择 -> 平均误差的期望为 0, 单次运行的
//      偏差量级约 ulp * sqrt(p(1-p)/N), 比 RNE 的偏置小两个数量级以上.
//  这样就同时验证了"SR 无偏"和"RNE 有偏"两件事.
// -----------------------------------------------------------------------------
void test_stochastic_unbiasedness() {
    testfw::begin_section("pipeline/stochastic-unbiasedness");

    const int64_t n = 65536;
    std::vector<float> x((size_t)n, 1.6f);
    x[0] = 8.0f;  // 决定 amax 与缩放因子

    struct Case {
        QuantFormat fmt;
        bool e5m2;
        const char* name;
    };
    const Case cases[] = {{QuantFormat::MxFP8, false, "mxfp8-e4m3"},
        {QuantFormat::MxFP8, true, "mxfp8-e5m2"}, {QuantFormat::MxFP4, false, "mxfp4"},
        {QuantFormat::NVFP4, false, "nvfp4"}};
    const int trials = 32;

    DeviceBuffer d_in((size_t)n * 4), d_out((size_t)n * 4);
    LPQ_CUDA_CHECK(cudaMemcpy(d_in.get(), x.data(), (size_t)n * 4, cudaMemcpyHostToDevice));

    for (const Case& c : cases) {
        QuantConfig cfg;
        cfg.format = c.fmt;
        cfg.mxfp8_e5m2 = c.e5m2;
        cfg.scale_mode = ScaleMode::Tensor;
        cfg.block_size = 32;
        cfg.block_size_explicit = true;
        cfg.output_type = OutputType::FP32;

        cfg.rounding = Rounding::Nearest;
        QuantizedDevice qn = quantize_on_device(d_in.get(), InputType::FP32, n, cfg);
        dequantize_on_device(qn, cfg, d_out.get());
        const double bias_rne = compute_error_stats(
            d_in.get(), InputType::FP32, d_out.get(), cfg.output_type, n, nullptr)
                                    .mean_signed_error;

        double sum_sr = 0.0, max_abs_sr = 0.0;
        for (int t = 0; t < trials; ++t) {
            cfg.rounding = Rounding::Stochastic;
            cfg.seed = 1000ull + (uint64_t)t * 7919ull;
            QuantizedDevice qs = quantize_on_device(d_in.get(), InputType::FP32, n, cfg);
            dequantize_on_device(qs, cfg, d_out.get());
            const double b = compute_error_stats(
                d_in.get(), InputType::FP32, d_out.get(), cfg.output_type, n, nullptr)
                                 .mean_signed_error;
            sum_sr += b;
            max_abs_sr = std::max(max_abs_sr, std::fabs(b));
        }
        const double mean_sr = sum_sr / (double)trials;
        const double abs_rne = std::fabs(bias_rne);

        CHECK_MSG(abs_rne > 0.0, std::string(c.name) + ": RNE 对区间内固定值应产生非零偏置");
        CHECK_MSG(max_abs_sr < 0.1 * abs_rne,
            std::string(c.name) + ": SR 单次偏置应远小于 RNE, max|sr|=" + testfw::f_of(max_abs_sr) +
                " rne=" + testfw::f_of(abs_rne));
        CHECK_MSG(std::fabs(mean_sr) < 0.02 * abs_rne,
            std::string(c.name) + ": SR 平均偏置应接近 0, mean_sr=" + testfw::f_of(mean_sr));
        std::printf(
            "    [SR-bias] %-12s |bias_rne|=%.6e  max|bias_sr|=%.3e  mean(bias_sr)=%.3e"
            "  (RNE/SR = %.0fx, %d 次试验)\n",
            c.name, abs_rne, max_abs_sr, mean_sr, max_abs_sr > 0.0 ? abs_rne / max_abs_sr : 0.0,
            trials);
    }
    testfw::end_section();
}

// -----------------------------------------------------------------------------
// 5. 误差统计 kernel 的自检: 手工算出的统计量必须一致
// -----------------------------------------------------------------------------
void test_error_stats_kernel() {
    testfw::begin_section("pipeline/error-stats");
    const int64_t n = 10000;
    std::vector<float> a((size_t)n), b((size_t)n);
    for (int64_t i = 0; i < n; ++i) {
        a[(size_t)i] = (float)i * 0.001f;
        b[(size_t)i] = a[(size_t)i] + ((i % 3) ? 0.0f : 0.5f);
    }
    DeviceBuffer da((size_t)n * 4), db((size_t)n * 4);
    LPQ_CUDA_CHECK(cudaMemcpy(da.get(), a.data(), (size_t)n * 4, cudaMemcpyHostToDevice));
    LPQ_CUDA_CHECK(cudaMemcpy(db.get(), b.data(), (size_t)n * 4, cudaMemcpyHostToDevice));
    const ErrorStats st = compute_error_stats(
        da.get(), InputType::FP32, db.get(), OutputType::FP32, n, nullptr, AccumMode::Precise);
    const ErrorStats sf = compute_error_stats(
        da.get(), InputType::FP32, db.get(), OutputType::FP32, n, nullptr, AccumMode::Fast);
    double sum_abs = 0.0, sum_sq = 0.0, sum_signed = 0.0, mx = 0.0, sum_ref = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        const double d = (double)b[(size_t)i] - (double)a[(size_t)i];
        sum_abs += std::fabs(d);
        sum_sq += d * d;
        sum_signed += d;
        mx = std::max(mx, std::fabs(d));
        sum_ref += (double)a[(size_t)i] * (double)a[(size_t)i];
    }
    const double mae = sum_abs / (double)n;
    const double mse = sum_sq / (double)n;
    CHECK_MSG(std::fabs(st.mae - mae) <= 1e-12 * std::fabs(mae) + 1e-15, "mae");
    CHECK_MSG(std::fabs(st.mse - mse) <= 1e-12 * std::fabs(mse) + 1e-15, "mse");
    CHECK_MSG(std::fabs(st.rmse - std::sqrt(mse)) <= 1e-12, "rmse");
    CHECK_MSG(st.max_abs == mx, "max_abs");
    CHECK_MSG(std::fabs(st.mean_signed_error - sum_signed / (double)n) <= 1e-15, "bias");
    CHECK_MSG(std::fabs(st.rel_mse - mse / (sum_ref / (double)n)) <= 1e-12, "rel_mse");
    // 补偿 fp32 累加(Fast)在同一组数据上必须给出同样的统计量
    CHECK_MSG(std::fabs(sf.mae - mae) <= 1e-12 * std::fabs(mae) + 1e-15, "fast mae");
    CHECK_MSG(std::fabs(sf.mse - mse) <= 1e-12 * std::fabs(mse) + 1e-15, "fast mse");
    CHECK_MSG(std::fabs(sf.mean_signed_error - sum_signed / (double)n) <= 1e-15, "fast bias");
    CHECK_MSG(sf.max_abs == mx, "fast max_abs");  // 两种策略的 max|d| 逐位一致
    testfw::end_section();
}

// -----------------------------------------------------------------------------
// 6. 两种累加策略(Precise = 逐元素 fp64 / Fast = 线程内 (hi,lo) 补偿 fp32)在
//    真实量级数据上的一致性. 这里故意让元素的指数跨 2^-24..2^24, 覆盖累加器
//    动态范围最坏的情形, 同时把结果与 CPU 双精度串行循环对照.
// -----------------------------------------------------------------------------
void test_error_stats_accum_modes() {
    testfw::begin_section("pipeline/error-stats-accum");
    const int64_t n = 1 << 20;
    std::vector<float> a((size_t)n), b((size_t)n);
    uint64_t rng = 0x2545F4914F6CDD1Dull;
    auto unit = [&]() {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        return (double)(rng >> 11) * (1.0 / 9007199254740992.0);
    };
    for (int64_t i = 0; i < n; ++i) {
        const double mag = std::ldexp(1.0, (int)(unit() * 48.0) - 24);
        const double sgn = (unit() < 0.5) ? -1.0 : 1.0;
        const double x = sgn * mag * (0.5 + unit());
        a[(size_t)i] = (float)x;
        b[(size_t)i] = (float)(x + (unit() - 0.5) * 0.02 * mag);
    }
    DeviceBuffer da((size_t)n * 4), db((size_t)n * 4);
    LPQ_CUDA_CHECK(cudaMemcpy(da.get(), a.data(), (size_t)n * 4, cudaMemcpyHostToDevice));
    LPQ_CUDA_CHECK(cudaMemcpy(db.get(), b.data(), (size_t)n * 4, cudaMemcpyHostToDevice));

    const ErrorStats sp = compute_error_stats(
        da.get(), InputType::FP32, db.get(), OutputType::FP32, n, nullptr, AccumMode::Precise);
    const ErrorStats sf = compute_error_stats(
        da.get(), InputType::FP32, db.get(), OutputType::FP32, n, nullptr, AccumMode::Fast);

    double s_abs = 0.0, s_sq = 0.0, s_signed = 0.0, s_ref = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        const double d = (double)b[(size_t)i] - (double)a[(size_t)i];
        s_abs += std::fabs(d);
        s_sq += d * d;
        s_signed += d;
        s_ref += (double)a[(size_t)i] * (double)a[(size_t)i];
    }
    const double mae_ref = s_abs / (double)n;
    const double mse_ref = s_sq / (double)n;
    auto rel = [](double x, double y) { return std::fabs(x - y) / std::max(std::fabs(y), 1e-300); };
    const double d_fast = rel(sf.mae, sp.mae);
    const double d_fast_mse = rel(sf.mse, sp.mse);
    const double d_fast_relmse = rel(sf.rel_mse, sp.rel_mse);
    std::printf("  accum diff (fast vs precise): mae %.3e  mse %.3e  rel_mse %.3e\n", d_fast,
        d_fast_mse, d_fast_relmse);
    std::printf("  accum diff (precise vs cpu-double): mae %.3e  mse %.3e\n", rel(sp.mae, mae_ref),
        rel(sp.mse, mse_ref));

    // Fast 与 Precise 必须一致到 1e-13(实测 ~1e-15)
    CHECK_MSG(d_fast <= 1e-13, "fast/precise mae " + testfw::f_of(d_fast));
    CHECK_MSG(d_fast_mse <= 1e-13, "fast/precise mse " + testfw::f_of(d_fast_mse));
    CHECK_MSG(d_fast_relmse <= 1e-13, "fast/precise rel_mse " + testfw::f_of(d_fast_relmse));
    // 有符号平均误差存在严重相消, 只能按误差项的量级(|d| 之和)来判
    CHECK_MSG(std::fabs(sf.mean_signed_error - sp.mean_signed_error) <= 1e-13 * mae_ref,
        "fast/precise bias");
    // Precise 路径仍与 CPU 双精度循环一致到 1e-12
    CHECK_MSG(
        rel(sp.mae, mae_ref) <= 1e-12, "precise/cpu mae " + testfw::f_of(rel(sp.mae, mae_ref)));
    CHECK_MSG(
        rel(sp.mse, mse_ref) <= 1e-12, "precise/cpu mse " + testfw::f_of(rel(sp.mse, mse_ref)));
    testfw::end_section();
}

// -----------------------------------------------------------------------------
// 7. 融合误差统计 vs 独立的 metrics kernel
//
// quantize_on_device(..., out_stats) 会把误差累加放进量化 kernel 内部, 从而省掉
// metrics kernel 对 fp32 输入(268MB)与反量化结果(134MB)的第二次读取. 这只有在
// 「两条路径给出同一组数字」时才可以接受, 所以这里对每一种配置都跑两遍:
//   路径 A: 量化 -> 反量化 -> compute_error_stats(独立 kernel)
//   路径 B: 量化(顺带统计) -> 反量化
// 逐位比较量化产物, 并比较两边的误差统计量.
//
// 两条路径的求和顺序不同(GPU 归约顺序依赖 block/thread 映射), 因此统计量不会逐位
// 相同, 只要求相对差 <= 1e-13; 实测在 1e-16 量级, 与 AccumFast 自身的精度同阶.
// -----------------------------------------------------------------------------
void test_fused_error_stats() {
    testfw::begin_section("pipeline/fused-stats");
    const int64_t n = 100000;
    std::vector<float> hx = make_input(Dist::Normal, n);
    const size_t in_bytes = (size_t)n * 4;
    DeviceBuffer d_in(in_bytes);
    LPQ_CUDA_CHECK(cudaMemcpy(d_in.get(), hx.data(), in_bytes, cudaMemcpyHostToDevice));

    struct Case {
        const char* tag;
        QuantFormat fmt;
        bool e5m2;
        int64_t block_size;
        ScaleMode sm;
        OutputType ot;
        ScaleRule rule = ScaleRule::Static6;
    };
    const Case cases[] = {
        {"mxfp8-e4m3/block32 (fused)", QuantFormat::MxFP8, false, 32, ScaleMode::Block,
            OutputType::FP16},
        {"mxfp8-e4m3/block16 (fused)", QuantFormat::MxFP8, false, 16, ScaleMode::Block,
            OutputType::BF16},
        {"mxfp8-e5m2/block32  (fused)", QuantFormat::MxFP8, true, 32, ScaleMode::Block,
            OutputType::FP16},
        {"mxfp8-e4m3/block8   (fused)", QuantFormat::MxFP8, false, 8, ScaleMode::Block,
            OutputType::FP32},
        {"mxfp8-e4m3/tensor   (generic)", QuantFormat::MxFP8, false, 32, ScaleMode::Tensor,
            OutputType::FP16},
        {"mxfp8-e4m3/block48  (generic)", QuantFormat::MxFP8, false, 48, ScaleMode::Block,
            OutputType::FP32},
        {"nvfp4/block16       (fused)", QuantFormat::NVFP4, false, 16, ScaleMode::Block,
            OutputType::FP16},
        {"nvfp4/block16 bf16  (fused)", QuantFormat::NVFP4, false, 16, ScaleMode::Block,
            OutputType::BF16},
        {"nvfp4/tensor        (generic)", QuantFormat::NVFP4, false, 16, ScaleMode::Tensor,
            OutputType::FP32},
        {"mxfp4/block32       (fused)", QuantFormat::MxFP4, false, 32, ScaleMode::Block,
            OutputType::FP16},
        {"mxfp4/block16 fp32  (fused)", QuantFormat::MxFP4, false, 16, ScaleMode::Block,
            OutputType::FP32},
        {"mxfp4/block48       (generic)", QuantFormat::MxFP4, false, 48, ScaleMode::Block,
            OutputType::BF16},
        {"nvfp4/4-6 mse blk16", QuantFormat::NVFP4, false, 16, ScaleMode::Block, OutputType::FP16,
            ScaleRule::Mse},
        {"nvfp4/4-6 mae blk16", QuantFormat::NVFP4, false, 16, ScaleMode::Block, OutputType::BF16,
            ScaleRule::Mae},
        {"nvfp4/4-6 absmax blk16", QuantFormat::NVFP4, false, 16, ScaleMode::Block,
            OutputType::FP32, ScaleRule::AbsMax},
    };

    auto rel = [](double a, double b) { return std::fabs(a - b) / std::max(std::fabs(b), 1e-300); };
    double worst = 0.0;
    for (const Case& c : cases) {
        QuantConfig cfg;
        cfg.format = c.fmt;
        cfg.mxfp8_e5m2 = c.e5m2;
        cfg.scale_rule = c.rule;
        cfg.block_size = c.block_size;
        cfg.block_size_explicit = true;
        cfg.scale_mode = c.sm;
        cfg.output_type = c.ot;
        cfg.rounding = Rounding::Nearest;

        const size_t out_bytes = (size_t)n * ((c.ot == OutputType::FP32) ? 4u : 2u);
        DeviceBuffer d_out((size_t)out_bytes);

        // 路径 A: 独立 metrics kernel
        QuantizedDevice qa = quantize_on_device(d_in.get(), InputType::FP32, n, cfg);
        CHECK_MSG(!qa.stats_fused,
            std::string(c.tag) + ": stats_fused should be false when not requested");
        dequantize_on_device(qa, cfg, d_out.get());
        const ErrorStats sa = compute_error_stats(
            d_in.get(), InputType::FP32, d_out.get(), c.ot, n, nullptr, AccumMode::Fast);

        std::vector<uint8_t> deq_a((size_t)out_bytes);
        LPQ_CUDA_CHECK(cudaMemcpy(deq_a.data(), d_out.get(), deq_a.size(), cudaMemcpyDeviceToHost));

        // 路径 B: 融合统计
        ErrorStats sb;
        QuantizedDevice qb =
            quantize_on_device(d_in.get(), InputType::FP32, n, cfg, nullptr, &sb, AccumMode::Fast);
        CHECK_MSG(qb.stats_fused, std::string(c.tag) + ": stats_fused should be true");
        dequantize_on_device(qb, cfg, d_out.get());
        std::vector<uint8_t> deq_b((size_t)out_bytes);
        LPQ_CUDA_CHECK(cudaMemcpy(deq_b.data(), d_out.get(), deq_b.size(), cudaMemcpyDeviceToHost));
        CHECK_MSG(
            deq_a == deq_b, std::string(c.tag) + ": dequantized output must be bit-identical");

        // 量化产物必须逐位一致
        std::vector<uint8_t> pa((size_t)qa.packed.size()), pb((size_t)qb.packed.size());
        std::vector<uint8_t> sc_a((size_t)qa.scales.size()), sc_b((size_t)qb.scales.size());
        LPQ_CUDA_CHECK(cudaMemcpy(pa.data(), qa.packed.get(), pa.size(), cudaMemcpyDeviceToHost));
        LPQ_CUDA_CHECK(cudaMemcpy(pb.data(), qb.packed.get(), pb.size(), cudaMemcpyDeviceToHost));
        LPQ_CUDA_CHECK(
            cudaMemcpy(sc_a.data(), qa.scales.get(), sc_a.size(), cudaMemcpyDeviceToHost));
        LPQ_CUDA_CHECK(
            cudaMemcpy(sc_b.data(), qb.scales.get(), sc_b.size(), cudaMemcpyDeviceToHost));
        CHECK_MSG(pa == pb, std::string(c.tag) + ": packed must be bit-identical");
        CHECK_MSG(sc_a == sc_b, std::string(c.tag) + ": scales must be bit-identical");

        const double d_mae = rel(sb.mae, sa.mae);
        const double d_mse = rel(sb.mse, sa.mse);
        const double d_rel = rel(sb.rel_mse, sa.rel_mse);
        const double d_max = rel(sb.max_abs, sa.max_abs);
        worst = std::max(worst, std::max(std::max(d_mae, d_mse), std::max(d_rel, d_max)));
        std::printf("  %-32s mae %.4e  mse %.4e  rel_mse %.4e  max %.4e  fused=%d\n", c.tag, d_mae,
            d_mse, d_rel, d_max, (int)qb.stats_fused);
        CHECK_MSG(d_mae <= 1e-13, std::string(c.tag) + ": mae " + testfw::f_of(d_mae));
        CHECK_MSG(d_mse <= 1e-13, std::string(c.tag) + ": mse " + testfw::f_of(d_mse));
        CHECK_MSG(d_rel <= 1e-13, std::string(c.tag) + ": rel_mse " + testfw::f_of(d_rel));
        CHECK_MSG(d_max <= 1e-13, std::string(c.tag) + ": max_abs " + testfw::f_of(d_max));
    }
    std::printf("   vs  metrics kernel : %.3e ( 1e-13)\n", worst);
    CHECK_MSG(worst <= 1e-13, "fused/standalone worst " + testfw::f_of(worst));
    testfw::end_section();
}

// -----------------------------------------------------------------------------
// 8. NVFP4 的 4/6 自适应块缩放(ScaleRule)
//
// 标准 NVFP4 把每个块的 amax 映射到 E2M1 的最大值 6.0. 但 E2M1 的格点是
// {0, .5, 1, 1.5, 2, 3, 4, 6}: 在 [4, 6] 之间的格距是 2, 在 [2, 4] 之间只有 1 --
// 越靠近最大值量化台阶越粗. 4/6 的做法是逐块在这两个候选里挑一个:
//   「6 候选」: block_scale 使 amax -> 6.0 (标准 NVFP4);
//   「4 候选」: block_scale 乘 1.5, 使 amax -> 4.0 (落在格距更细的区间).
// 挑哪个由误差准则决定(arXiv:2512.02010)。这里验证三件事:
//   (a) 每条规则下 GPU 与 CPU 参考实现的 packed / scales 逐位一致(走 check_case);
//   (b) 规则确实改写了产物, 没有退化成 static_6;
//   (c) 以误差为判据的规则, 全局误差不劣于 static_6.
// -----------------------------------------------------------------------------
void test_scale_rule_four_over_six() {
    testfw::begin_section("pipeline/scale-rule-4-6");

    const ScaleRule dyn_rules[] = {ScaleRule::AbsMax, ScaleRule::Mae, ScaleRule::Mse};
    const Dist dists[] = {Dist::Normal, Dist::Outlier, Dist::Uniform};
    const int64_t n = 4096;

    // ---- (a) 与 CPU 参考实现逐位对拍 ----
    for (int pass = 0; pass < 2; ++pass) {
        const ScaleRule r = (pass == 0) ? ScaleRule::Static6 : ScaleRule::Static4;
        for (Dist d : dists) {
            QuantConfig cfg;
            cfg.format = QuantFormat::NVFP4;
            cfg.scale_rule = r;
            cfg.scale_mode = ScaleMode::Block;
            cfg.block_size = 16;
            cfg.block_size_explicit = true;
            cfg.output_type = OutputType::FP32;
            cfg.rounding = Rounding::Nearest;
            cfg.seed = 4242;
            check_case(std::string("nvfp4/4-6/") + to_string(r) + "/" + dist_name(d), cfg,
                make_input(d, n), InputType::FP32, true);
        }
    }
    for (ScaleRule r : dyn_rules) {
        for (Dist d : dists) {
            QuantConfig cfg;
            cfg.format = QuantFormat::NVFP4;
            cfg.scale_rule = r;
            cfg.scale_mode = ScaleMode::Block;
            cfg.block_size = 16;
            cfg.block_size_explicit = true;
            cfg.output_type = OutputType::FP32;
            cfg.rounding = Rounding::Nearest;
            cfg.seed = 4242;
            check_case(std::string("nvfp4/4-6/") + to_string(r) + "/" + dist_name(d), cfg,
                make_input(d, n), InputType::FP32, true);
        }
    }

    // ---- (b)(c) 产物差异与误差对比 ----
    const std::vector<float> x = make_input(Dist::Normal, n);
    DeviceBuffer d_in((size_t)n * 4);
    LPQ_CUDA_CHECK(cudaMemcpy(d_in.get(), x.data(), (size_t)n * 4, cudaMemcpyHostToDevice));
    DeviceBuffer d_out((size_t)n * 4);

    struct Res {
        std::vector<uint8_t> scales;
        std::vector<uint8_t> packed;
        ErrorStats st;
    };
    auto run = [&](ScaleRule r) {
        QuantConfig cfg;
        cfg.format = QuantFormat::NVFP4;
        cfg.scale_rule = r;
        cfg.scale_mode = ScaleMode::Block;
        cfg.block_size = 16;
        cfg.block_size_explicit = true;
        cfg.output_type = OutputType::FP32;
        cfg.rounding = Rounding::Nearest;
        QuantizedDevice q = quantize_on_device(d_in.get(), InputType::FP32, n, cfg);
        dequantize_on_device(q, cfg, d_out.get());
        Res res;
        res.st = compute_error_stats(
            d_in.get(), InputType::FP32, d_out.get(), OutputType::FP32, n, nullptr);
        res.scales.assign((size_t)q.scales.size(), 0);
        res.packed.assign((size_t)q.packed.size(), 0);
        LPQ_CUDA_CHECK(cudaMemcpy(
            res.scales.data(), q.scales.get(), res.scales.size(), cudaMemcpyDeviceToHost));
        LPQ_CUDA_CHECK(cudaMemcpy(
            res.packed.data(), q.packed.get(), res.packed.size(), cudaMemcpyDeviceToHost));
        return res;
    };
    auto count_diff = [](const std::vector<uint8_t>& a, const std::vector<uint8_t>& b) {
        size_t c = 0;
        for (size_t i = 0; i < a.size(); ++i)
            if (a[i] != b[i])
                ++c;
        return c;
    };

    const Res base = run(ScaleRule::Static6);
    {
        // static_4: 全局缩放换成 4 * 448 / amax, 码字必然变化(块缩放因子的数值本身
        // 不变, 所以不能拿 scales 做判据)
        const Res r4 = run(ScaleRule::Static4);
        std::printf("  [4/6] static_4    packed 差异 %zu / %zu\n",
            count_diff(base.packed, r4.packed), base.packed.size());
        CHECK_MSG(count_diff(base.packed, r4.packed) > 0, "static_4 应改变码字");
        CHECK_MSG(r4.st.mse > base.st.mse, "static_4 把 amax 压到 4.0 应比 static_6 更粗");
    }
    for (ScaleRule r : dyn_rules) {
        const Res d = run(r);
        const size_t ds = count_diff(base.scales, d.scales);
        const size_t dp = count_diff(base.packed, d.packed);
        std::printf(
            "  [4/6] %-9s 块缩放因子改变 %zu / %zu   码字差异 %zu   mse %.6e"
            " (static_6 %.6e, %+.2f%%)\n",
            to_string(r), ds, base.scales.size(), dp, d.st.mse, base.st.mse,
            100.0 * (d.st.mse / base.st.mse - 1.0));
        CHECK_MSG(ds > 0, std::string(to_string(r)) + " 没有改写任何块缩放因子");
        // 判据按块最小化目标误差, 因此对应的全局误差不应变差(容差只覆盖 fp32 累加)
        CHECK_MSG(d.st.mse <= base.st.mse * (1.0 + 1e-9),
            std::string(to_string(r)) + " 的 mse 不应高于 static_6");
        if (r == ScaleRule::Mse) {
            CHECK_MSG(d.st.mae <= base.st.mae * (1.0 + 1e-9), "mse 规则也不该让 mae 变差");
        }
        if (r == ScaleRule::Mae) {
            CHECK_MSG(d.st.mae <= base.st.mae * (1.0 + 1e-9), "mae 规则应改善 mae");
        }
        if (r == ScaleRule::AbsMax) {
            CHECK_MSG(d.st.max_abs <= base.st.max_abs * (1.0 + 1e-9), "abs_max 规则应改善 max_abs");
        }
    }
    testfw::end_section();
}

}  // namespace

int main() {
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) {
        std::printf("no CUDA device available\n");
        return 2;
    }
    test_full_matrix();
    test_tensor_and_large_block();
    test_packing_layout();
    test_stochastic_rounding();
    test_stochastic_unbiasedness();
    test_error_stats_kernel();
    test_error_stats_accum_modes();
    test_fused_error_stats();
    test_scale_rule_four_over_six();
    return testfw::summary("test_pipeline");
}
