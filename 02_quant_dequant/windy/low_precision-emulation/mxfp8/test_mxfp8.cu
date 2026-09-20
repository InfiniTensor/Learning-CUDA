#include "mxfp8.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = (call);                                            \
        if (err != cudaSuccess) {                                            \
            std::fprintf(stderr, "%s:%d CUDA error: %s\n",                   \
                         __FILE__, __LINE__, cudaGetErrorString(err));        \
            std::exit(EXIT_FAILURE);                                         \
        }                                                                    \
    } while (0)

namespace {

    float bf16_to_float(mxfp8::BF16 x) {
        const uint32_t bits = static_cast<uint32_t>(x.bits) << 16;
        float out;
        std::memcpy(&out, &bits, sizeof(out));
        return out;
    }

    template <typename T>
    T from_float(float x);

    template <>
    float from_float<float>(float x) {
        return x;
    }

    template <>
    __half from_float<__half>(float x) {
        return __float2half_rn(x);
    }

    template <typename T>
    float to_float(T x);

    template <>
    float to_float<float>(float x) {
        return x;
    }

    template <>
    float to_float<__half>(__half x) {
        return __half2float(x);
    }

    template <>
    float to_float<mxfp8::BF16>(mxfp8::BF16 x) {
        return bf16_to_float(x);
    }

    struct ErrorStats {
        double max_abs = 0.0;
        double mae = 0.0;
        double mse = 0.0;
    };

    ErrorStats compute_error(const std::vector<float>& reference,
        const std::vector<float>& actual) {
        ErrorStats stats;
        double abs_sum = 0.0;
        double sq_sum = 0.0;
        int count = 0;

        for (size_t i = 0; i < reference.size(); ++i) {
            if (!std::isfinite(reference[i]) || !std::isfinite(actual[i])) {
                continue;
            }

            const double diff =
                std::fabs(static_cast<double>(reference[i]) -
                    static_cast<double>(actual[i]));

            stats.max_abs = std::fmax(stats.max_abs, diff);
            abs_sum += diff;
            sq_sum += diff * diff;
            ++count;
        }

        if (count > 0) {
            stats.mae = abs_sum / count;
            stats.mse = sq_sum / count;
        }

        return stats;
    }

    std::vector<float> make_input(int n) {
        std::vector<float> x(n, 0.0f);

        // 首个 block 用固定数据检查 scale 和编码。
        x[0] = 50.0f;
        x[1] = 100.0f;
        x[2] = 150.0f;
        x[3] = 200.0f;

        std::mt19937 rng(1234);
        std::normal_distribution<float> normal(0.0f, 3.0f);

        for (int i = 32; i < 64 && i < n; ++i) {
            x[i] = normal(rng);
        }

        for (int i = 64; i < 96 && i < n; ++i) {
            x[i] = 0.01f * static_cast<float>(i - 63);
        }
        if (n > 95) {
            x[95] = 10000.0f;
        }

        // 全零 block。

        // 尾部使用极小值测试下溢路径。
        for (int i = 128; i < n; ++i) {
            x[i] = std::numeric_limits<float>::denorm_min() *
                static_cast<float>(i - 127);
        }

        return x;
    }

    template <typename InputT, typename OutputT>
    void run_case(const char* name, const std::vector<float>& source) {
        const int n = static_cast<int>(source.size());
        const int scale_count = mxfp8::num_scales(n);

        std::vector<InputT> h_input(n);
        std::vector<float> reference(n);

        for (int i = 0; i < n; ++i) {
            h_input[i] = from_float<InputT>(source[i]);

            // 使用实际进入 kernel 的输入值作为误差基准。
            reference[i] = to_float<InputT>(h_input[i]);
        }

        InputT* d_input = nullptr;
        uint8_t* d_quantized = nullptr;
        uint8_t* d_scales = nullptr;
        OutputT* d_output = nullptr;

        CUDA_CHECK(cudaMalloc(&d_input, n * sizeof(InputT)));
        CUDA_CHECK(cudaMalloc(&d_quantized, n * sizeof(uint8_t)));
        CUDA_CHECK(cudaMalloc(&d_scales, scale_count * sizeof(uint8_t)));
        CUDA_CHECK(cudaMalloc(&d_output, n * sizeof(OutputT)));

        CUDA_CHECK(cudaMemcpy(d_input,
            h_input.data(),
            n * sizeof(InputT),
            cudaMemcpyHostToDevice));

        mxfp8::quantize(d_input, d_quantized, d_scales, n);
        CUDA_CHECK(cudaGetLastError());

        mxfp8::dequantize(d_quantized, d_scales, d_output, n);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<uint8_t> h_quantized(n);
        std::vector<uint8_t> h_scales(scale_count);
        std::vector<OutputT> h_output(n);

        CUDA_CHECK(cudaMemcpy(h_quantized.data(),
            d_quantized,
            n * sizeof(uint8_t),
            cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_scales.data(),
            d_scales,
            scale_count * sizeof(uint8_t),
            cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_output.data(),
            d_output,
            n * sizeof(OutputT),
            cudaMemcpyDeviceToHost));

        std::vector<float> actual(n);
        for (int i = 0; i < n; ++i) {
            actual[i] = to_float<OutputT>(h_output[i]);
        }

        const ErrorStats stats = compute_error(reference, actual);

        // 检查首个 block 的 scale 和 E4M3 编码。
        const uint8_t expected_q[4] = {0x6c, 0x74, 0x79, 0x7c};

        bool pass = (h_scales[0] == 0x7e);
        for (int i = 0; i < 4; ++i) {
            pass = pass && (h_quantized[i] == expected_q[i]);
        }

        std::printf("\n[%s]\n", name);
        std::printf("golden     : %s\n", pass ? "PASS" : "FAIL");
        std::printf("scale[0]   : 0x%02X\n", static_cast<unsigned>(h_scales[0]));
        std::printf("max_abs    : %.9e\n", stats.max_abs);
        std::printf("MAE        : %.9e\n", stats.mae);
        std::printf("MSE        : %.9e\n", stats.mse);

        for (int i = 0; i < 4; ++i) {
            std::printf("i=%d  in=%8.3f  q=0x%02X  out=%8.3f\n",
                i,
                reference[i],
                static_cast<unsigned>(h_quantized[i]),
                actual[i]);
        }

        CUDA_CHECK(cudaFree(d_input));
        CUDA_CHECK(cudaFree(d_quantized));
        CUDA_CHECK(cudaFree(d_scales));
        CUDA_CHECK(cudaFree(d_output));

        if (!pass) {
            std::fprintf(stderr, "%s failed golden validation.\n", name);
            std::exit(EXIT_FAILURE);
        }
    }

    void test_e4m3_encode() {
        struct Case {
            const char* name;
            float value;
            uint8_t expected;
        };

        // 用 448 固定 scale exponent 为 0，单独测试 E4M3 编码。
        const Case cases[] = {
            {"+zero",              0.0f,         0x00},
            {"-zero",             -0.0f,         0x80},
            {"min subnormal",      0x1p-9f,      0x01},
            {"max subnormal",      7.0f * 0x1p-9f, 0x07},
            {"min normal",         0x1p-6f,      0x08},
            {"one",                1.0f,         0x38},
            {"one point five",     1.5f,         0x3C},

            // RNE 中点，舍入到偶数低位。
            {"RNE tie -> even low", 1.0625f,      0x38},

            // RNE 中点，舍入到偶数高位。
            {"RNE tie -> even high", 1.1875f,     0x3A},

            // RNE 舍入进位。
            {"RNE carry",          1.9375f,      0x40},

            // 测试次正规数中点舍入。
            {"subnormal tie zero", 0x1p-10f,     0x00},

            // 测试次正规数的偶数舍入。
            {"subnormal tie even", 3.0f * 0x1p-10f, 0x02},

            // 测试次正规数到最小正规数的舍入。
            {"subnormal -> normal", 15.0f * 0x1p-10f, 0x08},

            {"negative",           -1.5f,        0xBC},
            {"max finite",         448.0f,       0x7E},
            {"NaN",                std::numeric_limits<float>::quiet_NaN(), 0x7F},
        };

        constexpr int n = 32;
        std::vector<float> h_input(n, 0.0f);

        const int case_count = static_cast<int>(sizeof(cases) / sizeof(cases[0]));
        for (int i = 0; i < case_count; ++i) {
            h_input[i] = cases[i].value;
        }

        // 用 448 固定 block scale。
        h_input[n - 1] = 448.0f;

        float* d_input = nullptr;
        uint8_t* d_quantized = nullptr;
        uint8_t* d_scale = nullptr;

        CUDA_CHECK(cudaMalloc(&d_input, n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_quantized, n * sizeof(uint8_t)));
        CUDA_CHECK(cudaMalloc(&d_scale, sizeof(uint8_t)));

        CUDA_CHECK(cudaMemcpy(d_input,
            h_input.data(),
            n * sizeof(float),
            cudaMemcpyHostToDevice));

        mxfp8::quantize(d_input, d_quantized, d_scale, n);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<uint8_t> h_quantized(n);
        uint8_t h_scale = 0;

        CUDA_CHECK(cudaMemcpy(h_quantized.data(),
            d_quantized,
            n * sizeof(uint8_t),
            cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&h_scale,
            d_scale,
            sizeof(uint8_t),
            cudaMemcpyDeviceToHost));

        bool pass = (h_scale == 0x7F);

        std::printf("\n[E4M3 编码单元测试]\n");
        std::printf("forced scale: 0x%02X (%s)\n",
            static_cast<unsigned>(h_scale),
            h_scale == 0x7F ? "PASS" : "FAIL");

        for (int i = 0; i < case_count; ++i) {
            const bool ok = h_quantized[i] == cases[i].expected;
            pass = pass && ok;

            std::printf("%-24s q=0x%02X expected=0x%02X  %s\n",
                cases[i].name,
                static_cast<unsigned>(h_quantized[i]),
                static_cast<unsigned>(cases[i].expected),
                ok ? "PASS" : "FAIL");
        }

        CUDA_CHECK(cudaFree(d_input));
        CUDA_CHECK(cudaFree(d_quantized));
        CUDA_CHECK(cudaFree(d_scale));

        if (!pass) {
            std::fprintf(stderr, "E4M3 编码单元测试失败。\n");
            std::exit(EXIT_FAILURE);
        }
    }

    void test_e4m3_decode() {
        struct Case {
            const char* name;
            uint8_t code;
            float expected;
            bool expect_nan;
            bool expect_negative_zero;
        };

        const Case cases[] = {
            {"+zero",          0x00, 0.0f,              false, false},
            {"-zero",          0x80, -0.0f,             false, true},
            {"min subnormal",  0x01, 0x1p-9f,          false, false},
            {"max subnormal",  0x07, 7.0f * 0x1p-9f,   false, false},
            {"min normal",     0x08, 0x1p-6f,          false, false},
            {"one",            0x38, 1.0f,              false, false},
            {"one point five", 0x3C, 1.5f,              false, false},
            {"two",            0x40, 2.0f,              false, false},
            {"288",            0x79, 288.0f,            false, false},
            {"max finite",     0x7E, 448.0f,            false, false},
            {"negative max",   0xFE, -448.0f,           false, false},
            {"NaN",            0x7F, 0.0f,              true,  false},
            {"negative NaN",   0xFF, 0.0f,              true,  false},
        };

        constexpr int n = 32;
        std::vector<uint8_t> h_quantized(n, 0x00);
        const int case_count = static_cast<int>(sizeof(cases) / sizeof(cases[0]));

        for (int i = 0; i < case_count; ++i) {
            h_quantized[i] = cases[i].code;
        }

        // k = 0，对应 E8M0 code = 0x7F。
        const uint8_t h_scale = 0x7F;

        uint8_t* d_quantized = nullptr;
        uint8_t* d_scale = nullptr;
        float* d_output = nullptr;

        CUDA_CHECK(cudaMalloc(&d_quantized, n * sizeof(uint8_t)));
        CUDA_CHECK(cudaMalloc(&d_scale, sizeof(uint8_t)));
        CUDA_CHECK(cudaMalloc(&d_output, n * sizeof(float)));

        CUDA_CHECK(cudaMemcpy(d_quantized,
            h_quantized.data(),
            n * sizeof(uint8_t),
            cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_scale,
            &h_scale,
            sizeof(uint8_t),
            cudaMemcpyHostToDevice));

        mxfp8::dequantize(d_quantized, d_scale, d_output, n);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<float> h_output(n);
        CUDA_CHECK(cudaMemcpy(h_output.data(),
            d_output,
            n * sizeof(float),
            cudaMemcpyDeviceToHost));

        bool pass = true;

        std::printf("\n[E4M3 解码单元测试]\n");

        for (int i = 0; i < case_count; ++i) {
            bool ok;

            if (cases[i].expect_nan) {
                ok = std::isnan(h_output[i]);
            } else if (cases[i].expect_negative_zero) {
                ok = h_output[i] == 0.0f && std::signbit(h_output[i]);
            } else {
                ok = h_output[i] == cases[i].expected;
            }

            pass = pass && ok;

            if (cases[i].expect_nan) {
                std::printf("%-24s code=0x%02X out=NaN  %s\n",
                    cases[i].name,
                    static_cast<unsigned>(cases[i].code),
                    ok ? "PASS" : "FAIL");
            } else {
                std::printf("%-24s code=0x%02X out=% .9e expected=% .9e  %s\n",
                    cases[i].name,
                    static_cast<unsigned>(cases[i].code),
                    h_output[i],
                    cases[i].expected,
                    ok ? "PASS" : "FAIL");
            }
        }

        CUDA_CHECK(cudaFree(d_quantized));
        CUDA_CHECK(cudaFree(d_scale));
        CUDA_CHECK(cudaFree(d_output));

        if (!pass) {
            std::fprintf(stderr, "E4M3 解码单元测试失败。\n");
            std::exit(EXIT_FAILURE);
        }
    }

    void test_scale_boundaries() {
        struct Case {
            float amax;
            uint8_t expected_scale;
        };

        const Case cases[] = {
            {0.0f,   0x7F},  // 全零 block 的约定：k = 0
            {200.0f, 0x7E},  // k = -1
            {224.0f, 0x7E},  // 224 / 2^-1 = 448，刚好达到 E4M3 最大有限值
            {225.0f, 0x7F},  // k = 0
            {448.0f, 0x7F},  // k = 0
            {449.0f, 0x80},  // k = 1
            {896.0f, 0x80},  // 896 / 2 = 448，刚好达到 E4M3 最大有限值
            {897.0f, 0x81},  // k = 2
        };

        constexpr int case_count = static_cast<int>(sizeof(cases) / sizeof(cases[0]));
        constexpr int n = case_count * mxfp8::kBlockSize;

        std::vector<float> h_input(n, 0.0f);

        for (int b = 0; b < case_count; ++b) {
            h_input[b * mxfp8::kBlockSize] = cases[b].amax;
        }

        float* d_input = nullptr;
        uint8_t* d_quantized = nullptr;
        uint8_t* d_scales = nullptr;

        CUDA_CHECK(cudaMalloc(&d_input, n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_quantized, n * sizeof(uint8_t)));
        CUDA_CHECK(cudaMalloc(&d_scales, case_count * sizeof(uint8_t)));

        CUDA_CHECK(cudaMemcpy(d_input,
            h_input.data(),
            n * sizeof(float),
            cudaMemcpyHostToDevice));

        mxfp8::quantize(d_input, d_quantized, d_scales, n);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<uint8_t> h_scales(case_count);
        CUDA_CHECK(cudaMemcpy(h_scales.data(),
            d_scales,
            case_count * sizeof(uint8_t),
            cudaMemcpyDeviceToHost));

        bool pass = true;

        std::printf("\n[E8M0 scale 边界测试]\n");

        for (int i = 0; i < case_count; ++i) {
            const bool ok = h_scales[i] == cases[i].expected_scale;
            pass = pass && ok;

            std::printf("amax=%7.1f scale=0x%02X expected=0x%02X  %s\n",
                cases[i].amax,
                static_cast<unsigned>(h_scales[i]),
                static_cast<unsigned>(cases[i].expected_scale),
                ok ? "PASS" : "FAIL");
        }

        CUDA_CHECK(cudaFree(d_input));
        CUDA_CHECK(cudaFree(d_quantized));
        CUDA_CHECK(cudaFree(d_scales));

        if (!pass) {
            std::fprintf(stderr, "E8M0 scale 边界测试失败。\n");
            std::exit(EXIT_FAILURE);
        }
    }

    void test_inf_encoding() {
        constexpr int n = 32;
        std::vector<float> h_input(n, 0.0f);
        h_input[0] = std::numeric_limits<float>::infinity();
        h_input[1] = -std::numeric_limits<float>::infinity();

        float* d_input = nullptr;
        uint8_t* d_quantized = nullptr;
        uint8_t* d_scale = nullptr;

        CUDA_CHECK(cudaMalloc(&d_input, n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_quantized, n * sizeof(uint8_t)));
        CUDA_CHECK(cudaMalloc(&d_scale, sizeof(uint8_t)));

        CUDA_CHECK(cudaMemcpy(d_input,
            h_input.data(),
            n * sizeof(float),
            cudaMemcpyHostToDevice));

        mxfp8::quantize(d_input, d_quantized, d_scale, n);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<uint8_t> h_quantized(n);
        uint8_t h_scale = 0;

        CUDA_CHECK(cudaMemcpy(h_quantized.data(),
            d_quantized,
            n * sizeof(uint8_t),
            cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&h_scale,
            d_scale,
            sizeof(uint8_t),
            cudaMemcpyDeviceToHost));

        const bool pass =
            h_scale == 0xFE &&
            h_quantized[0] == 0x7E &&
            h_quantized[1] == 0xFE;

        std::printf("\n[Inf 编码测试]\n");
        std::printf("scale=0x%02X, +Inf q=0x%02X, -Inf q=0x%02X  %s\n",
            static_cast<unsigned>(h_scale),
            static_cast<unsigned>(h_quantized[0]),
            static_cast<unsigned>(h_quantized[1]),
            pass ? "PASS" : "FAIL");

        CUDA_CHECK(cudaFree(d_input));
        CUDA_CHECK(cudaFree(d_quantized));
        CUDA_CHECK(cudaFree(d_scale));

        if (!pass) {
            std::fprintf(stderr, "Inf 编码测试失败。\n");
            std::exit(EXIT_FAILURE);
        }
    }

    enum class DataPattern {
        Uniform,
        Normal,
        Outlier
    };

    const char* pattern_name(DataPattern pattern) {
        switch (pattern) {
            case DataPattern::Uniform: return "均匀随机矩阵";
            case DataPattern::Normal:  return "正态分布矩阵";
            case DataPattern::Outlier: return "含异常值矩阵";
        }
        return "未知";
    }

    std::vector<float> make_matrix_input(int n, DataPattern pattern) {
        std::vector<float> x(n);
        std::mt19937 rng(2026);

        if (pattern == DataPattern::Uniform) {
            // 均匀分布。
            std::uniform_real_distribution<float> dist(-8.0f, 8.0f);
            for (int i = 0; i < n; ++i) {
                x[i] = dist(rng);
            }
            return x;
        }

        if (pattern == DataPattern::Normal) {
            // 正态分布。
            std::normal_distribution<float> dist(0.0f, 3.0f);
            for (int i = 0; i < n; ++i) {
                x[i] = dist(rng);
            }
            return x;
        }

        // 正态分布中周期性插入异常值。
        std::normal_distribution<float> dist(0.0f, 1.0f);
        for (int i = 0; i < n; ++i) {
            x[i] = dist(rng);
        }

        for (int i = 0; i < n; i += 4096) {
            x[i] = ((i / 4096) & 1) ? -1000.0f : 1000.0f;
        }

        return x;
    }

    template <typename InputT, typename OutputT>
    void run_project_case(const char* case_name,
        const std::vector<float>& source,
        int warmup_iters,
        int bench_iters) {
        const int n = static_cast<int>(source.size());
        const int scale_count = mxfp8::num_scales(n);

        std::vector<InputT> h_input(n);
        std::vector<float> reference(n);

        for (int i = 0; i < n; ++i) {
            h_input[i] = from_float<InputT>(source[i]);

            // 使用实际进入 kernel 的输入值作为误差基准。
            reference[i] = to_float<InputT>(h_input[i]);
        }

        InputT* d_input = nullptr;
        uint8_t* d_quantized = nullptr;
        uint8_t* d_scales = nullptr;
        OutputT* d_output = nullptr;

        CUDA_CHECK(cudaMalloc(&d_input, n * sizeof(InputT)));
        CUDA_CHECK(cudaMalloc(&d_quantized, n * sizeof(uint8_t)));
        CUDA_CHECK(cudaMalloc(&d_scales, scale_count * sizeof(uint8_t)));
        CUDA_CHECK(cudaMalloc(&d_output, n * sizeof(OutputT)));

        CUDA_CHECK(cudaMemcpy(d_input,
            h_input.data(),
            n * sizeof(InputT),
            cudaMemcpyHostToDevice));

        // 量化计时。

        for (int i = 0; i < warmup_iters; ++i) {
            mxfp8::quantize(d_input, d_quantized, d_scales, n);
        }
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        cudaEvent_t quant_start;
        cudaEvent_t quant_stop;
        CUDA_CHECK(cudaEventCreate(&quant_start));
        CUDA_CHECK(cudaEventCreate(&quant_stop));

        CUDA_CHECK(cudaEventRecord(quant_start));
        for (int i = 0; i < bench_iters; ++i) {
            mxfp8::quantize(d_input, d_quantized, d_scales, n);
        }
        CUDA_CHECK(cudaEventRecord(quant_stop));
        CUDA_CHECK(cudaEventSynchronize(quant_stop));

        float quant_total_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&quant_total_ms, quant_start, quant_stop));
        const float quant_ms = quant_total_ms / bench_iters;

        // 反量化计时。

        for (int i = 0; i < warmup_iters; ++i) {
            mxfp8::dequantize(d_quantized, d_scales, d_output, n);
        }
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        cudaEvent_t dequant_start;
        cudaEvent_t dequant_stop;
        CUDA_CHECK(cudaEventCreate(&dequant_start));
        CUDA_CHECK(cudaEventCreate(&dequant_stop));

        CUDA_CHECK(cudaEventRecord(dequant_start));
        for (int i = 0; i < bench_iters; ++i) {
            mxfp8::dequantize(d_quantized, d_scales, d_output, n);
        }
        CUDA_CHECK(cudaEventRecord(dequant_stop));
        CUDA_CHECK(cudaEventSynchronize(dequant_stop));

        float dequant_total_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&dequant_total_ms,
            dequant_start,
            dequant_stop));
        const float dequant_ms = dequant_total_ms / bench_iters;

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // 拷回输出并统计误差。

        std::vector<OutputT> h_output(n);
        CUDA_CHECK(cudaMemcpy(h_output.data(),
            d_output,
            n * sizeof(OutputT),
            cudaMemcpyDeviceToHost));

        std::vector<float> actual(n);
        bool finite_ok = true;

        for (int i = 0; i < n; ++i) {
            actual[i] = to_float<OutputT>(h_output[i]);

            // 有限输入应得到有限输出。
            if (!std::isfinite(actual[i])) {
                finite_ok = false;
            }
        }

        const ErrorStats stats = compute_error(reference, actual);

        // 压缩率只统计 E4M3 数据和 E8M0 scale。

        const double input_bytes =
            static_cast<double>(n) * sizeof(InputT);

        const double compressed_bytes =
            static_cast<double>(n) * sizeof(uint8_t) +
            static_cast<double>(scale_count) * sizeof(uint8_t);

        const double compression_ratio =
            input_bytes / compressed_bytes;

        // 按逻辑读写字节数计算有效带宽。

        const double quant_bytes =
            static_cast<double>(n) * sizeof(InputT) +
            static_cast<double>(n) * sizeof(uint8_t) +
            static_cast<double>(scale_count) * sizeof(uint8_t);

        const double dequant_bytes =
            static_cast<double>(n) * sizeof(uint8_t) +
            static_cast<double>(scale_count) * sizeof(uint8_t) +
            static_cast<double>(n) * sizeof(OutputT);

        // bytes / (ms * 1e6) = GB/s。
        const double quant_gbps =
            quant_bytes / (static_cast<double>(quant_ms) * 1.0e6);

        const double dequant_gbps =
            dequant_bytes / (static_cast<double>(dequant_ms) * 1.0e6);

        std::printf("\n[%s]\n", case_name);
        std::printf("元素数量            : %d\n", n);
        std::printf("scale 数量          : %d\n", scale_count);
        std::printf("有限值检查          : %s\n", finite_ok ? "PASS" : "FAIL");
        std::printf("最大绝对误差        : %.9e\n", stats.max_abs);
        std::printf("MAE                 : %.9e\n", stats.mae);
        std::printf("MSE                 : %.9e\n", stats.mse);
        std::printf("压缩率              : %.4fx\n", compression_ratio);
        std::printf("量化 kernel 时间    : %.6f ms\n", quant_ms);
        std::printf("反量化 kernel 时间  : %.6f ms\n", dequant_ms);
        std::printf("量化有效带宽        : %.3f GB/s\n", quant_gbps);
        std::printf("反量化有效带宽      : %.3f GB/s\n", dequant_gbps);

        CUDA_CHECK(cudaEventDestroy(quant_start));
        CUDA_CHECK(cudaEventDestroy(quant_stop));
        CUDA_CHECK(cudaEventDestroy(dequant_start));
        CUDA_CHECK(cudaEventDestroy(dequant_stop));

        CUDA_CHECK(cudaFree(d_input));
        CUDA_CHECK(cudaFree(d_quantized));
        CUDA_CHECK(cudaFree(d_scales));
        CUDA_CHECK(cudaFree(d_output));

        if (!finite_ok) {
            std::fprintf(stderr, "%s 出现非有限反量化结果。\n", case_name);
            std::exit(EXIT_FAILURE);
        }
    }

    void run_project_matrix_tests() {
        // 使用 4096 x 4096 矩阵进行稳定测试。
        constexpr int rows = 4096;
        constexpr int cols = 4096;
        constexpr int n = rows * cols;

        constexpr int warmup_iters = 10;
        constexpr int bench_iters = 100;

        const DataPattern patterns[] = {
            DataPattern::Uniform,
            DataPattern::Normal,
            DataPattern::Outlier,
        };

        std::printf("\n============================================================\n");
        std::printf("项目要求：三类矩阵端到端误差与性能测试\n");
        std::printf("矩阵形状：%d x %d\n", rows, cols);
        std::printf("warmup=%d, benchmark iterations=%d\n",
            warmup_iters,
            bench_iters);
        std::printf("============================================================\n");

        for (DataPattern pattern : patterns) {
            const std::vector<float> source = make_matrix_input(n, pattern);

            std::printf("\n------------------------------------------------------------\n");
            std::printf("数据分布：%s\n", pattern_name(pattern));
            std::printf("------------------------------------------------------------\n");

            // FP32 输入。
            run_project_case<float, float>(
                "FP32 -> MXFP8 -> FP32",
                source,
                warmup_iters,
                bench_iters);

            run_project_case<float, __half>(
                "FP32 -> MXFP8 -> FP16",
                source,
                warmup_iters,
                bench_iters);

            run_project_case<float, mxfp8::BF16>(
                "FP32 -> MXFP8 -> BF16",
                source,
                warmup_iters,
                bench_iters);

            // FP16 输入。
            run_project_case<__half, float>(
                "FP16 -> MXFP8 -> FP32",
                source,
                warmup_iters,
                bench_iters);

            run_project_case<__half, __half>(
                "FP16 -> MXFP8 -> FP16",
                source,
                warmup_iters,
                bench_iters);

            run_project_case<__half, mxfp8::BF16>(
                "FP16 -> MXFP8 -> BF16",
                source,
                warmup_iters,
                bench_iters);
        }
    }

}  // namespace

int main() {
    test_e4m3_encode();
    test_e4m3_decode();
    test_scale_boundaries();
    test_inf_encoding();

    constexpr int n = 150;  // 非 32 的整数倍，用于测试尾部 block。
    const std::vector<float> source = make_input(n);

    // 覆盖 FP32 / FP16 输入与三种输出类型。
    run_case<float, float>("FP32 -> MXFP8 -> FP32", source);
    run_case<float, __half>("FP32 -> MXFP8 -> FP16", source);
    run_case<float, mxfp8::BF16>("FP32 -> MXFP8 -> BF16", source);

    run_case<__half, float>("FP16 -> MXFP8 -> FP32", source);
    run_case<__half, __half>("FP16 -> MXFP8 -> FP16", source);
    run_case<__half, mxfp8::BF16>("FP16 -> MXFP8 -> BF16", source);

    run_project_matrix_tests();

    std::printf("\n所有 MXFP8 测试均通过。\n");
    return 0;
}
