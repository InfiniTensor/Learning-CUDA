#include "nvfp4.cuh"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <type_traits>
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

// 测试侧 16 位存储类型。
struct FP16 {
    uint16_t bits;
};

struct BF16 {
    uint16_t bits;
};

static_assert(sizeof(FP16) == sizeof(uint16_t));
static_assert(sizeof(BF16) == sizeof(uint16_t));

// Host 侧 FP16 / BF16 转换。
uint32_t round_shift_rne_host(uint32_t value, int shift) {
    if (shift <= 0) return value;
    if (shift >= 32) return 0;

    uint32_t kept = value >> shift;
    const uint32_t remainder = value & ((1u << shift) - 1u);
    const uint32_t halfway = 1u << (shift - 1);

    if (remainder > halfway ||
        (remainder == halfway && (kept & 1u))) {
        ++kept;
    }

    return kept;
}

uint16_t float_to_fp16_bits(float x) {
    uint32_t bits;
    std::memcpy(&bits, &x, sizeof(bits));

    const uint32_t sign = (bits >> 16) & 0x8000u;
    const uint32_t e32 = (bits >> 23) & 0xffu;
    const uint32_t frac32 = bits & 0x7fffffu;

    if (e32 == 0xffu) {
        return static_cast<uint16_t>(
            sign | (frac32 ? 0x7e00u : 0x7c00u));
    }

    if (e32 == 0) {
        return static_cast<uint16_t>(sign);
    }

    int E = static_cast<int>(e32) - 127;
    const uint32_t sig = (1u << 23) | frac32;

    if (E > 15) {
        return static_cast<uint16_t>(sign | 0x7c00u);
    }

    if (E >= -14) {
        uint32_t rounded_sig = round_shift_rne_host(sig, 13);

        if (rounded_sig == 2048u) {
            rounded_sig = 1024u;
            ++E;

            if (E > 15) {
                return static_cast<uint16_t>(sign | 0x7c00u);
            }
        }

        const uint32_t exp16 =
            static_cast<uint32_t>(E + 15);

        return static_cast<uint16_t>(
            sign |
            (exp16 << 10) |
            (rounded_sig & 0x03ffu));
    }

    const uint32_t q =
        round_shift_rne_host(sig, -E - 1);

    if (q >= 1024u) {
        return static_cast<uint16_t>(sign | 0x0400u);
    }

    return static_cast<uint16_t>(sign | q);
}

float fp16_bits_to_float(uint16_t h) {
    const uint32_t sign = h >> 15;
    const uint32_t exp = (h >> 10) & 0x1fu;
    const uint32_t frac = h & 0x03ffu;

    if (exp == 0) {
        if (frac == 0) {
            return sign ? -0.0f : 0.0f;
        }

        const float value =
            std::ldexp(static_cast<float>(frac), -24);

        return sign ? -value : value;
    }

    if (exp == 0x1fu) {
        if (frac != 0) {
            return std::numeric_limits<float>::quiet_NaN();
        }

        return sign
            ? -std::numeric_limits<float>::infinity()
            :  std::numeric_limits<float>::infinity();
    }

    const float value =
        std::ldexp(
            1.0f + static_cast<float>(frac) / 1024.0f,
            static_cast<int>(exp) - 15);

    return sign ? -value : value;
}

uint16_t float_to_bf16_bits(float x) {
    uint32_t bits;
    std::memcpy(&bits, &x, sizeof(bits));

    const uint32_t exp = bits & 0x7f800000u;
    const uint32_t frac = bits & 0x007fffffu;

    if (exp == 0x7f800000u && frac != 0) {
        return static_cast<uint16_t>(
            (bits >> 16) | 0x0040u);
    }

    const uint32_t lsb =
        (bits >> 16) & 1u;

    bits += 0x7fffu + lsb;

    return static_cast<uint16_t>(bits >> 16);
}

float bf16_bits_to_float(uint16_t b) {
    const uint32_t bits =
        static_cast<uint32_t>(b) << 16;

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
FP16 from_float<FP16>(float x) {
    return FP16{float_to_fp16_bits(x)};
}

template <>
BF16 from_float<BF16>(float x) {
    return BF16{float_to_bf16_bits(x)};
}

template <typename T>
float to_float(T x);

template <>
float to_float<float>(float x) {
    return x;
}

template <>
float to_float<FP16>(FP16 x) {
    return fp16_bits_to_float(x.bits);
}

template <>
float to_float<BF16>(BF16 x) {
    return bf16_bits_to_float(x.bits);
}

// 根据输入类型分发量化 API。
template <typename InputT>
cudaError_t quantize_dispatch(
    const InputT* d_input,
    int n_padded,
    uint8_t* d_packed,
    uint8_t* d_scales,
    float* d_global_scale,
    uint32_t* d_global_amax_bits_workspace = nullptr) {

    if constexpr (std::is_same_v<InputT, float>) {
        return nvfp4::quantize(
            d_input,
            n_padded,
            d_packed,
            d_scales,
            d_global_scale,
            0,
            d_global_amax_bits_workspace);
    } else {
        static_assert(std::is_same_v<InputT, FP16>);

        return nvfp4::quantize_fp16(
            reinterpret_cast<const uint16_t*>(d_input),
            n_padded,
            d_packed,
            d_scales,
            d_global_scale,
            0,
            d_global_amax_bits_workspace);
    }
}

// 根据输出类型分发反量化 API。
template <typename OutputT>
cudaError_t dequantize_dispatch(
    const uint8_t* d_packed,
    const uint8_t* d_scales,
    const float* d_global_scale,
    int n_padded,
    OutputT* d_output) {

    if constexpr (std::is_same_v<OutputT, float>) {
        return nvfp4::dequantize(
            d_packed,
            d_scales,
            d_global_scale,
            n_padded,
            d_output);
    } else if constexpr (std::is_same_v<OutputT, FP16>) {
        return nvfp4::dequantize_fp16(
            d_packed,
            d_scales,
            d_global_scale,
            n_padded,
            reinterpret_cast<uint16_t*>(d_output));
    } else {
        static_assert(std::is_same_v<OutputT, BF16>);

        return nvfp4::dequantize_bf16(
            d_packed,
            d_scales,
            d_global_scale,
            n_padded,
            reinterpret_cast<uint16_t*>(d_output));
    }
}

// 通用误差统计。
struct ErrorStats {
    double max_abs = 0.0;
    double mae = 0.0;
    double mse = 0.0;
};

ErrorStats compute_error(
    const std::vector<float>& reference,
    const std::vector<float>& actual) {

    ErrorStats stats;
    double abs_sum = 0.0;
    double sq_sum = 0.0;
    int count = 0;

    for (size_t i = 0; i < reference.size(); ++i) {
        if (!std::isfinite(reference[i]) ||
            !std::isfinite(actual[i])) {
            continue;
        }

        const double diff =
            std::fabs(
                static_cast<double>(reference[i]) -
                static_cast<double>(actual[i]));

        stats.max_abs =
            std::fmax(stats.max_abs, diff);

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

uint8_t unpack_e2m1(
    const std::vector<uint8_t>& packed,
    int index) {

    const uint8_t byte =
        packed[index >> 1];

    return (index & 1)
        ? static_cast<uint8_t>((byte >> 4) & 0x0fu)
        : static_cast<uint8_t>(byte & 0x0fu);
}

// E2M1 编码测试。
void test_e2m1_encode() {
    struct Case {
        const char* name;
        float value;
        uint8_t expected;
    };

    const Case cases[] = {
        {"+zero",                  0.0f,  0x0},
        {"-zero",                 -0.0f,  0x8},

        {"subnormal 0.5",          0.5f,  0x1},
        {"one",                    1.0f,  0x2},
        {"one point five",         1.5f,  0x3},
        {"two",                    2.0f,  0x4},
        {"three",                  3.0f,  0x5},
        {"four",                   4.0f,  0x6},
        {"max finite 6",           6.0f,  0x7},

        {"RNE tie 0 / 0.5",        0.25f, 0x0},
        {"RNE tie 0.5 / 1",        0.75f, 0x2},
        {"RNE tie 1 / 1.5",        1.25f, 0x2},
        {"RNE tie 1.5 / 2",        1.75f, 0x4},
        {"RNE tie 2 / 3",          2.5f,  0x4},

        {"RNE rounding carry",     3.5f,  0x6},

        {"negative",              -1.5f,  0xB},
    };

    constexpr int n = 64;
    constexpr int case_count =
        static_cast<int>(sizeof(cases) / sizeof(cases[0]));

    static_assert(
        case_count == nvfp4::kBlockSize,
        "E2M1 cases should occupy exactly one NVFP4 block.");

    std::vector<float> h_input(n, 0.0f);

    for (int i = 0; i < case_count; ++i) {
        h_input[i] = cases[i].value;
    }

    // 固定 global_scale = 1，便于验证 E2M1。
    h_input[16] = 2688.0f;

    float* d_input = nullptr;
    uint8_t* d_packed = nullptr;
    uint8_t* d_scales = nullptr;
    float* d_global_scale = nullptr;

    CUDA_CHECK(cudaMalloc(&d_input, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(
        &d_packed,
        nvfp4::packed_bytes(n) * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(
        &d_scales,
        nvfp4::block_scale_count(n) * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(
        &d_global_scale,
        sizeof(float)));

    CUDA_CHECK(cudaMemcpy(
        d_input,
        h_input.data(),
        n * sizeof(float),
        cudaMemcpyHostToDevice));

    CUDA_CHECK(nvfp4::quantize(
        d_input,
        n,
        d_packed,
        d_scales,
        d_global_scale));

    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<uint8_t> h_packed(
        nvfp4::packed_bytes(n));

    std::vector<uint8_t> h_scales(
        nvfp4::block_scale_count(n));

    float h_global_scale = 0.0f;

    CUDA_CHECK(cudaMemcpy(
        h_packed.data(),
        d_packed,
        h_packed.size() * sizeof(uint8_t),
        cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaMemcpy(
        h_scales.data(),
        d_scales,
        h_scales.size() * sizeof(uint8_t),
        cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaMemcpy(
        &h_global_scale,
        d_global_scale,
        sizeof(float),
        cudaMemcpyDeviceToHost));

    bool pass =
        h_global_scale == 1.0f &&
        h_scales[0] == 0x38;

    std::printf("\n[E2M1 编码单元测试]\n");
    std::printf(
        "forced global scale: %.1f (%s)\n",
        h_global_scale,
        h_global_scale == 1.0f ? "PASS" : "FAIL");

    std::printf(
        "forced block scale : 0x%02X (%s)\n",
        static_cast<unsigned>(h_scales[0]),
        h_scales[0] == 0x38 ? "PASS" : "FAIL");

    for (int i = 0; i < case_count; ++i) {
        const uint8_t q =
            unpack_e2m1(h_packed, i);

        const bool ok =
            q == cases[i].expected;

        pass = pass && ok;

        std::printf(
            "%-24s q=0x%X expected=0x%X  %s\n",
            cases[i].name,
            static_cast<unsigned>(q),
            static_cast<unsigned>(cases[i].expected),
            ok ? "PASS" : "FAIL");
    }

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_packed));
    CUDA_CHECK(cudaFree(d_scales));
    CUDA_CHECK(cudaFree(d_global_scale));

    if (!pass) {
        std::fprintf(
            stderr,
            "E2M1 编码单元测试失败。\n");

        std::exit(EXIT_FAILURE);
    }
}

// E2M1 解码测试。
void test_e2m1_decode() {
    constexpr int n = 64;

    std::vector<uint8_t> h_packed(
        nvfp4::packed_bytes(n),
        0x00);

    for (int i = 0; i < 8; ++i) {
        const uint8_t q0 =
            static_cast<uint8_t>(2 * i);

        const uint8_t q1 =
            static_cast<uint8_t>(2 * i + 1);

        h_packed[i] =
            static_cast<uint8_t>(
                (q1 << 4) | q0);
    }

    std::vector<uint8_t> h_scales(
        nvfp4::block_scale_count(n),
        0x38);

    const float h_global_scale = 1.0f;

    const float expected[16] = {
         0.0f,  0.5f,  1.0f,  1.5f,
         2.0f,  3.0f,  4.0f,  6.0f,
        -0.0f, -0.5f, -1.0f, -1.5f,
        -2.0f, -3.0f, -4.0f, -6.0f,
    };

    uint8_t* d_packed = nullptr;
    uint8_t* d_scales = nullptr;
    float* d_global_scale = nullptr;
    float* d_output = nullptr;

    CUDA_CHECK(cudaMalloc(
        &d_packed,
        h_packed.size() * sizeof(uint8_t)));

    CUDA_CHECK(cudaMalloc(
        &d_scales,
        h_scales.size() * sizeof(uint8_t)));

    CUDA_CHECK(cudaMalloc(
        &d_global_scale,
        sizeof(float)));

    CUDA_CHECK(cudaMalloc(
        &d_output,
        n * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(
        d_packed,
        h_packed.data(),
        h_packed.size() * sizeof(uint8_t),
        cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(
        d_scales,
        h_scales.data(),
        h_scales.size() * sizeof(uint8_t),
        cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(
        d_global_scale,
        &h_global_scale,
        sizeof(float),
        cudaMemcpyHostToDevice));

    CUDA_CHECK(nvfp4::dequantize(
        d_packed,
        d_scales,
        d_global_scale,
        n,
        d_output));

    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> h_output(n);

    CUDA_CHECK(cudaMemcpy(
        h_output.data(),
        d_output,
        n * sizeof(float),
        cudaMemcpyDeviceToHost));

    bool pass = true;

    std::printf("\n[E2M1 解码单元测试]\n");

    for (int i = 0; i < 16; ++i) {
        bool ok;

        if (i == 8) {
            ok =
                h_output[i] == 0.0f &&
                std::signbit(h_output[i]);
        } else {
            ok =
                h_output[i] == expected[i];
        }

        pass = pass && ok;

        std::printf(
            "code=0x%X out=% .9e expected=% .9e  %s\n",
            i,
            h_output[i],
            expected[i],
            ok ? "PASS" : "FAIL");
    }

    CUDA_CHECK(cudaFree(d_packed));
    CUDA_CHECK(cudaFree(d_scales));
    CUDA_CHECK(cudaFree(d_global_scale));
    CUDA_CHECK(cudaFree(d_output));

    if (!pass) {
        std::fprintf(
            stderr,
            "E2M1 解码单元测试失败。\n");

        std::exit(EXIT_FAILURE);
    }
}

// E2M1 特殊值测试。
void test_e2m1_special_values() {

    {
        constexpr int n = 64;
        std::vector<float> h_input(n, 0.0f);

        h_input[0] =
            std::numeric_limits<float>::quiet_NaN();

        h_input[1] = 6.0f;
        h_input[16] = 2688.0f;

        float* d_input = nullptr;
        uint8_t* d_packed = nullptr;
        uint8_t* d_scales = nullptr;
        float* d_global_scale = nullptr;

        CUDA_CHECK(cudaMalloc(&d_input, n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(
            &d_packed,
            nvfp4::packed_bytes(n) * sizeof(uint8_t)));
        CUDA_CHECK(cudaMalloc(
            &d_scales,
            nvfp4::block_scale_count(n) * sizeof(uint8_t)));
        CUDA_CHECK(cudaMalloc(
            &d_global_scale,
            sizeof(float)));

        CUDA_CHECK(cudaMemcpy(
            d_input,
            h_input.data(),
            n * sizeof(float),
            cudaMemcpyHostToDevice));

        CUDA_CHECK(nvfp4::quantize(
            d_input,
            n,
            d_packed,
            d_scales,
            d_global_scale));

        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<uint8_t> h_packed(
            nvfp4::packed_bytes(n));

        CUDA_CHECK(cudaMemcpy(
            h_packed.data(),
            d_packed,
            h_packed.size() * sizeof(uint8_t),
            cudaMemcpyDeviceToHost));

        const uint8_t q =
            unpack_e2m1(h_packed, 0);

        const bool pass =
            q == 0x7;

        std::printf("\n[NaN 编码测试]\n");
        std::printf(
            "NaN q=0x%X expected=0x7  %s\n",
            static_cast<unsigned>(q),
            pass ? "PASS" : "FAIL");

        CUDA_CHECK(cudaFree(d_input));
        CUDA_CHECK(cudaFree(d_packed));
        CUDA_CHECK(cudaFree(d_scales));
        CUDA_CHECK(cudaFree(d_global_scale));

        if (!pass) {
            std::fprintf(
                stderr,
                "NaN 编码测试失败。\n");

            std::exit(EXIT_FAILURE);
        }
    }

    {
        constexpr int n = 64;
        std::vector<float> h_input(n, 0.0f);

        h_input[0] =
            std::numeric_limits<float>::infinity();

        h_input[1] =
            -std::numeric_limits<float>::infinity();

        float* d_input = nullptr;
        uint8_t* d_packed = nullptr;
        uint8_t* d_scales = nullptr;
        float* d_global_scale = nullptr;

        CUDA_CHECK(cudaMalloc(&d_input, n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(
            &d_packed,
            nvfp4::packed_bytes(n) * sizeof(uint8_t)));
        CUDA_CHECK(cudaMalloc(
            &d_scales,
            nvfp4::block_scale_count(n) * sizeof(uint8_t)));
        CUDA_CHECK(cudaMalloc(
            &d_global_scale,
            sizeof(float)));

        CUDA_CHECK(cudaMemcpy(
            d_input,
            h_input.data(),
            n * sizeof(float),
            cudaMemcpyHostToDevice));

        CUDA_CHECK(nvfp4::quantize(
            d_input,
            n,
            d_packed,
            d_scales,
            d_global_scale));

        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<uint8_t> h_packed(
            nvfp4::packed_bytes(n));

        CUDA_CHECK(cudaMemcpy(
            h_packed.data(),
            d_packed,
            h_packed.size() * sizeof(uint8_t),
            cudaMemcpyDeviceToHost));

        const uint8_t q_pos =
            unpack_e2m1(h_packed, 0);

        const uint8_t q_neg =
            unpack_e2m1(h_packed, 1);

        const bool pass =
            q_pos == 0x7 &&
            q_neg == 0xF;

        std::printf("\n[Inf 编码测试]\n");
        std::printf(
            "+Inf q=0x%X, -Inf q=0x%X, expected=(0x7, 0xF)  %s\n",
            static_cast<unsigned>(q_pos),
            static_cast<unsigned>(q_neg),
            pass ? "PASS" : "FAIL");

        CUDA_CHECK(cudaFree(d_input));
        CUDA_CHECK(cudaFree(d_packed));
        CUDA_CHECK(cudaFree(d_scales));
        CUDA_CHECK(cudaFree(d_global_scale));

        if (!pass) {
            std::fprintf(
                stderr,
                "Inf 编码测试失败。\n");

            std::exit(EXIT_FAILURE);
        }
    }
}

// 4-bit 打包测试。
void test_packing() {
    constexpr int n = 64;
    std::vector<float> h_input(n, 0.0f);

    h_input[0] = 6.0f;
    h_input[1] = 3.0f;

    h_input[16] = 2688.0f;

    float* d_input = nullptr;
    uint8_t* d_packed = nullptr;
    uint8_t* d_scales = nullptr;
    float* d_global_scale = nullptr;

    CUDA_CHECK(cudaMalloc(&d_input, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(
        &d_packed,
        nvfp4::packed_bytes(n) * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(
        &d_scales,
        nvfp4::block_scale_count(n) * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(
        &d_global_scale,
        sizeof(float)));

    CUDA_CHECK(cudaMemcpy(
        d_input,
        h_input.data(),
        n * sizeof(float),
        cudaMemcpyHostToDevice));

    CUDA_CHECK(nvfp4::quantize(
        d_input,
        n,
        d_packed,
        d_scales,
        d_global_scale));

    CUDA_CHECK(cudaDeviceSynchronize());

    uint8_t packed0 = 0;

    CUDA_CHECK(cudaMemcpy(
        &packed0,
        d_packed,
        sizeof(uint8_t),
        cudaMemcpyDeviceToHost));

    const bool pass =
        packed0 == 0x57;

    std::printf("\n[4-bit packing 测试]\n");
    std::printf(
        "packed[0]=0x%02X expected=0x57  %s\n",
        static_cast<unsigned>(packed0),
        pass ? "PASS" : "FAIL");

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_packed));
    CUDA_CHECK(cudaFree(d_scales));
    CUDA_CHECK(cudaFree(d_global_scale));

    if (!pass) {
        std::fprintf(
            stderr,
            "4-bit packing 测试失败。\n");

        std::exit(EXIT_FAILURE);
    }
}

// E4M3 block scale 边界测试。
void test_block_scale_boundaries() {
    struct Case {
        float ideal_scale;
        uint8_t expected_scale;
    };

    const Case cases[] = {
        {0.0f,                 0x00},

        {0x1p-9f,              0x01},
        {7.0f * 0x1p-9f,       0x07},
        {0x1p-6f,              0x08},

        {1.0f,                 0x38},
        {1.01f,                0x39},
        {1.125f,               0x39},
        {1.126f,               0x3A},

        {200.0f,               0x75},
        {208.0f,               0x75},
        {209.0f,               0x76},

        {448.0f,               0x7E},
    };

    constexpr int case_count =
        static_cast<int>(sizeof(cases) / sizeof(cases[0]));

    constexpr int n =
        case_count * nvfp4::kBlockSize;

    static_assert(
        n % nvfp4::kWarpValues == 0,
        "scale boundary test size must be a multiple of 64.");

    std::vector<float> h_input(n, 0.0f);

    for (int b = 0; b < case_count; ++b) {
        h_input[b * nvfp4::kBlockSize] =
            cases[b].ideal_scale * 6.0f;
    }

    float* d_input = nullptr;
    uint8_t* d_packed = nullptr;
    uint8_t* d_scales = nullptr;
    float* d_global_scale = nullptr;

    CUDA_CHECK(cudaMalloc(&d_input, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(
        &d_packed,
        nvfp4::packed_bytes(n) * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(
        &d_scales,
        case_count * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(
        &d_global_scale,
        sizeof(float)));

    CUDA_CHECK(cudaMemcpy(
        d_input,
        h_input.data(),
        n * sizeof(float),
        cudaMemcpyHostToDevice));

    CUDA_CHECK(nvfp4::quantize(
        d_input,
        n,
        d_packed,
        d_scales,
        d_global_scale));

    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<uint8_t> h_scales(case_count);
    float h_global_scale = 0.0f;

    CUDA_CHECK(cudaMemcpy(
        h_scales.data(),
        d_scales,
        case_count * sizeof(uint8_t),
        cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaMemcpy(
        &h_global_scale,
        d_global_scale,
        sizeof(float),
        cudaMemcpyDeviceToHost));

    bool pass =
        h_global_scale == 1.0f;

    std::printf("\n[E4M3 block scale 边界测试]\n");
    std::printf(
        "forced global scale: %.1f (%s)\n",
        h_global_scale,
        h_global_scale == 1.0f ? "PASS" : "FAIL");

    for (int i = 0; i < case_count; ++i) {
        const bool ok =
            h_scales[i] == cases[i].expected_scale;

        pass = pass && ok;

        std::printf(
            "ideal=%9.4f scale=0x%02X expected=0x%02X  %s\n",
            cases[i].ideal_scale,
            static_cast<unsigned>(h_scales[i]),
            static_cast<unsigned>(cases[i].expected_scale),
            ok ? "PASS" : "FAIL");
    }

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_packed));
    CUDA_CHECK(cudaFree(d_scales));
    CUDA_CHECK(cudaFree(d_global_scale));

    if (!pass) {
        std::fprintf(
            stderr,
            "E4M3 block scale 边界测试失败。\n");

        std::exit(EXIT_FAILURE);
    }
}

// FP32 global scale 边界测试。
void test_global_scale_boundaries() {
    struct Case {
        float amax;
        float expected_global_scale;
    };

    const Case cases[] = {
        {0.0f,    1.0f},
        {1344.0f, 0.5f},
        {2688.0f, 1.0f},
        {5376.0f, 2.0f},
    };

    constexpr int case_count =
        static_cast<int>(sizeof(cases) / sizeof(cases[0]));

    bool pass = true;

    std::printf("\n[FP32 global scale 边界测试]\n");

    for (int c = 0; c < case_count; ++c) {
        constexpr int n = 64;

        std::vector<float> h_input(n, 0.0f);
        h_input[0] = cases[c].amax;

        float* d_input = nullptr;
        uint8_t* d_packed = nullptr;
        uint8_t* d_scales = nullptr;
        float* d_global_scale = nullptr;

        CUDA_CHECK(cudaMalloc(&d_input, n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(
            &d_packed,
            nvfp4::packed_bytes(n) * sizeof(uint8_t)));
        CUDA_CHECK(cudaMalloc(
            &d_scales,
            nvfp4::block_scale_count(n) * sizeof(uint8_t)));
        CUDA_CHECK(cudaMalloc(
            &d_global_scale,
            sizeof(float)));

        CUDA_CHECK(cudaMemcpy(
            d_input,
            h_input.data(),
            n * sizeof(float),
            cudaMemcpyHostToDevice));

        CUDA_CHECK(nvfp4::quantize(
            d_input,
            n,
            d_packed,
            d_scales,
            d_global_scale));

        CUDA_CHECK(cudaDeviceSynchronize());

        float h_global_scale = 0.0f;

        CUDA_CHECK(cudaMemcpy(
            &h_global_scale,
            d_global_scale,
            sizeof(float),
            cudaMemcpyDeviceToHost));

        const bool ok =
            h_global_scale ==
            cases[c].expected_global_scale;

        pass = pass && ok;

        std::printf(
            "amax=%8.1f global_scale=%6.3f expected=%6.3f  %s\n",
            cases[c].amax,
            h_global_scale,
            cases[c].expected_global_scale,
            ok ? "PASS" : "FAIL");

        CUDA_CHECK(cudaFree(d_input));
        CUDA_CHECK(cudaFree(d_packed));
        CUDA_CHECK(cudaFree(d_scales));
        CUDA_CHECK(cudaFree(d_global_scale));
    }

    if (!pass) {
        std::fprintf(
            stderr,
            "FP32 global scale 边界测试失败。\n");

        std::exit(EXIT_FAILURE);
    }
}

// 三类项目级测试矩阵。
enum class DataPattern {
    Uniform,
    Normal,
    Outlier
};

const char* pattern_name(DataPattern pattern) {
    switch (pattern) {
        case DataPattern::Uniform:
            return "均匀随机矩阵";

        case DataPattern::Normal:
            return "正态分布矩阵";

        case DataPattern::Outlier:
            return "含异常值矩阵";
    }

    return "未知";
}

std::vector<float> make_matrix_input(
    int n,
    DataPattern pattern) {

    std::vector<float> x(n);
    std::mt19937 rng(2026);

    if (pattern == DataPattern::Uniform) {
        std::uniform_real_distribution<float>
            dist(-8.0f, 8.0f);

        for (int i = 0; i < n; ++i) {
            x[i] = dist(rng);
        }

        return x;
    }

    if (pattern == DataPattern::Normal) {
        std::normal_distribution<float>
            dist(0.0f, 3.0f);

        for (int i = 0; i < n; ++i) {
            x[i] = dist(rng);
        }

        return x;
    }

    std::normal_distribution<float>
        dist(0.0f, 1.0f);

    for (int i = 0; i < n; ++i) {
        x[i] = dist(rng);
    }

    for (int i = 0; i < n; i += 4096) {
        x[i] =
            ((i / 4096) & 1)
                ? -1000.0f
                :  1000.0f;
    }

    return x;
}

// 端到端误差与性能测试。
template <typename InputT, typename OutputT>
void run_project_case(
    const char* case_name,
    const std::vector<float>& source,
    int warmup_iters,
    int bench_iters) {

    const int n =
        static_cast<int>(source.size());

    const int n_padded =
        nvfp4::padded_elements(n);

    const int block_scale_count =
        nvfp4::block_scale_count(n_padded);

    const int packed_count =
        nvfp4::packed_bytes(n_padded);

    const int scale_count =
        block_scale_count + 1;

    std::vector<InputT> h_input(n_padded);
    std::vector<float> reference(n);

    for (int i = 0; i < n_padded; ++i) {
        const float x =
            i < n ? source[i] : 0.0f;

        h_input[i] =
            from_float<InputT>(x);

        if (i < n) {

            reference[i] =
                to_float<InputT>(h_input[i]);
        }
    }

    InputT* d_input = nullptr;
    uint8_t* d_packed = nullptr;
    uint8_t* d_scales = nullptr;
    float* d_global_scale = nullptr;
    uint32_t* d_global_amax_bits_workspace = nullptr;
    OutputT* d_output = nullptr;

    CUDA_CHECK(cudaMalloc(
        &d_input,
        n_padded * sizeof(InputT)));

    CUDA_CHECK(cudaMalloc(
        &d_packed,
        packed_count * sizeof(uint8_t)));

    CUDA_CHECK(cudaMalloc(
        &d_scales,
        block_scale_count * sizeof(uint8_t)));

    CUDA_CHECK(cudaMalloc(
        &d_global_scale,
        sizeof(float)));

    // 复用量化 workspace，避免热路径反复申请临时显存。
    CUDA_CHECK(cudaMalloc(
        &d_global_amax_bits_workspace,
        sizeof(uint32_t)));

    CUDA_CHECK(cudaMalloc(
        &d_output,
        n_padded * sizeof(OutputT)));

    CUDA_CHECK(cudaMemcpy(
        d_input,
        h_input.data(),
        n_padded * sizeof(InputT),
        cudaMemcpyHostToDevice));

    // 量化：global amax -> global scale -> block scale + E2M1。
    for (int i = 0; i < warmup_iters; ++i) {
        CUDA_CHECK(quantize_dispatch(
            d_input,
            n_padded,
            d_packed,
            d_scales,
            d_global_scale,
            d_global_amax_bits_workspace));
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t quant_start;
    cudaEvent_t quant_stop;

    CUDA_CHECK(cudaEventCreate(&quant_start));
    CUDA_CHECK(cudaEventCreate(&quant_stop));

    CUDA_CHECK(cudaEventRecord(quant_start));

    for (int i = 0; i < bench_iters; ++i) {
        CUDA_CHECK(quantize_dispatch(
            d_input,
            n_padded,
            d_packed,
            d_scales,
            d_global_scale,
            d_global_amax_bits_workspace));
    }

    CUDA_CHECK(cudaEventRecord(quant_stop));
    CUDA_CHECK(cudaEventSynchronize(quant_stop));

    float quant_total_ms = 0.0f;

    CUDA_CHECK(cudaEventElapsedTime(
        &quant_total_ms,
        quant_start,
        quant_stop));

    const float quant_ms =
        quant_total_ms / bench_iters;

    for (int i = 0; i < warmup_iters; ++i) {
        CUDA_CHECK(dequantize_dispatch(
            d_packed,
            d_scales,
            d_global_scale,
            n_padded,
            d_output));
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t dequant_start;
    cudaEvent_t dequant_stop;

    CUDA_CHECK(cudaEventCreate(&dequant_start));
    CUDA_CHECK(cudaEventCreate(&dequant_stop));

    CUDA_CHECK(cudaEventRecord(dequant_start));

    for (int i = 0; i < bench_iters; ++i) {
        CUDA_CHECK(dequantize_dispatch(
            d_packed,
            d_scales,
            d_global_scale,
            n_padded,
            d_output));
    }

    CUDA_CHECK(cudaEventRecord(dequant_stop));
    CUDA_CHECK(cudaEventSynchronize(dequant_stop));

    float dequant_total_ms = 0.0f;

    CUDA_CHECK(cudaEventElapsedTime(
        &dequant_total_ms,
        dequant_start,
        dequant_stop));

    const float dequant_ms =
        dequant_total_ms / bench_iters;

    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<OutputT> h_output(n_padded);

    CUDA_CHECK(cudaMemcpy(
        h_output.data(),
        d_output,
        n_padded * sizeof(OutputT),
        cudaMemcpyDeviceToHost));

    std::vector<float> actual(n);
    bool finite_ok = true;

    for (int i = 0; i < n; ++i) {
        actual[i] =
            to_float<OutputT>(h_output[i]);

        if (!std::isfinite(actual[i])) {
            finite_ok = false;
        }
    }

    const ErrorStats stats =
        compute_error(reference, actual);

    const double input_bytes =
        static_cast<double>(n) *
        sizeof(InputT);

    // payload：E2M1 数据 + E4M3 block scale + 1 个 FP32 global scale。
    const double compressed_bytes =
        static_cast<double>(packed_count) *
            sizeof(uint8_t) +
        static_cast<double>(block_scale_count) *
            sizeof(uint8_t) +
        sizeof(float);

    const double compression_ratio =
        input_bytes /
        compressed_bytes;

    // 量化包含两次输入遍历，因此按两次 input read 统计。
    const double quant_bytes =
        2.0 *
            static_cast<double>(n_padded) *
            sizeof(InputT) +
        static_cast<double>(packed_count) *
            sizeof(uint8_t) +
        static_cast<double>(block_scale_count) *
            sizeof(uint8_t) +
        sizeof(float);

    const double dequant_bytes =
        static_cast<double>(packed_count) *
            sizeof(uint8_t) +
        static_cast<double>(block_scale_count) *
            sizeof(uint8_t) +
        sizeof(float) +
        static_cast<double>(n_padded) *
            sizeof(OutputT);

    const double quant_gbps =
        quant_bytes /
        (static_cast<double>(quant_ms) * 1.0e6);

    const double dequant_gbps =
        dequant_bytes /
        (static_cast<double>(dequant_ms) * 1.0e6);

    std::printf("\n[%s]\n", case_name);
    std::printf("元素数量            : %d\n", n);
    std::printf("scale 数量          : %d\n", scale_count);
    std::printf("有限值检查          : %s\n",
                finite_ok ? "PASS" : "FAIL");
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
    CUDA_CHECK(cudaFree(d_packed));
    CUDA_CHECK(cudaFree(d_scales));
    CUDA_CHECK(cudaFree(d_global_scale));
    CUDA_CHECK(cudaFree(d_global_amax_bits_workspace));
    CUDA_CHECK(cudaFree(d_output));

    if (!finite_ok) {
        std::fprintf(
            stderr,
            "%s 出现非有限反量化结果。\n",
            case_name);

        std::exit(EXIT_FAILURE);
    }
}

// 运行三类矩阵测试。
void run_project_matrix_tests() {

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

    std::printf(
        "\n============================================================\n");

    std::printf(
        "项目要求：三类矩阵端到端误差与性能测试\n");

    std::printf(
        "矩阵形状：%d x %d\n",
        rows,
        cols);

    std::printf(
        "warmup=%d, benchmark iterations=%d\n",
        warmup_iters,
        bench_iters);

    std::printf(
        "============================================================\n");

    for (DataPattern pattern : patterns) {
        const std::vector<float> source =
            make_matrix_input(n, pattern);

        std::printf(
            "\n------------------------------------------------------------\n");

        std::printf(
            "数据分布：%s\n",
            pattern_name(pattern));

        std::printf(
            "------------------------------------------------------------\n");

        run_project_case<float, float>(
            "FP32 -> NVFP4 -> FP32",
            source,
            warmup_iters,
            bench_iters);

        run_project_case<float, FP16>(
            "FP32 -> NVFP4 -> FP16",
            source,
            warmup_iters,
            bench_iters);

        run_project_case<float, BF16>(
            "FP32 -> NVFP4 -> BF16",
            source,
            warmup_iters,
            bench_iters);

        run_project_case<FP16, float>(
            "FP16 -> NVFP4 -> FP32",
            source,
            warmup_iters,
            bench_iters);

        run_project_case<FP16, FP16>(
            "FP16 -> NVFP4 -> FP16",
            source,
            warmup_iters,
            bench_iters);

        run_project_case<FP16, BF16>(
            "FP16 -> NVFP4 -> BF16",
            source,
            warmup_iters,
            bench_iters);
    }
}

}  // 匿名命名空间

int main() {

    // E2M1 数据格式。
    test_e2m1_encode();
    test_e2m1_decode();
    test_e2m1_special_values();
    test_packing();

    // NVFP4 两级缩放。
    test_block_scale_boundaries();
    test_global_scale_boundaries();

    // 项目级矩阵测试。
    run_project_matrix_tests();

    std::printf("\n所有 NVFP4 测试均通过。\n");

    return 0;
}
