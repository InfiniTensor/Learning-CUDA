// Correctness harness.
//
// M1 covers the CPU references; the GPU comparisons are already wired so that M2
// and M3 only have to fill in the launchers. Comparison methodology: the input is
// rounded to the target precision first, and the reference is computed on that
// rounded input, so what the tolerance measures is kernel error, not input
// quantization error.
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "hadamard/common.h"
#include "hadamard/cuda_utils.h"
#include "hadamard/hadamard.h"

// The helpers below are declared at file scope, so pull the project symbols in.
using namespace hadamard;

namespace {

int g_failed = 0;
int g_skipped = 0;

struct Case {
    int batch;
    int seq_len;
    int num_heads;
    int head_dim;
};

const Case kCases[] = {
    {1, 1, 1, 2}, {1, 1, 1, 64}, {2, 3, 4, 128}, {1, 7, 8, 256}, {3, 5, 6, 512},
};

void report(bool ok, const std::string& name, const std::string& detail) {
    std::printf("[%s] %-46s %s\n", ok ? "PASS" : "FAIL", name.c_str(), detail.c_str());
    if (!ok) {
        ++g_failed;
    }
}

void skip(const std::string& name, const std::string& detail) {
    std::printf("[SKIP] %-46s %s\n", name.c_str(), detail.c_str());
    ++g_skipped;
}

std::string fmt(const Case& c) {
    return "d=" + std::to_string(c.head_dim) +
           " rows=" + std::to_string(static_cast<long>(c.batch) * c.seq_len * c.num_heads);
}

float max_abs_diff(const float* a, const float* b, long n) {
    float worst = 0.0f;
    for (long i = 0; i < n; ++i) {
        worst = std::max(worst, std::fabs(a[i] - b[i]));
    }
    return worst;
}

// The O(d^2) matmul reference and the butterfly must agree: they are two
// independent constructions of the same matrix.
void test_reference_agreement() {
    const float kTol = 1e-4f;
    for (const Case& c : kCases) {
        const Shape shape{c.batch, c.seq_len, c.num_heads, c.head_dim};
        const long n = shape.elems();
        const std::vector<float> x = random_host_data(n, 0xABCDu + c.head_dim);
        std::vector<float> y_matmul(n);
        std::vector<float> y_fwht(n);
        ref_matmul_fp32(x.data(), y_matmul.data(), shape.rows(), shape.head_dim);
        ref_fwht_fp32(x.data(), y_fwht.data(), shape.rows(), shape.head_dim);
        const float diff = max_abs_diff(y_matmul.data(), y_fwht.data(), n);
        report(diff <= kTol, "reference agreement (matmul vs butterfly) " + fmt(c),
               "max|diff|=" + std::to_string(diff));
    }
}

// H is orthogonal up to the scale factor, so applying the scaled transform twice
// must return the input. This catches sign, ordering, and normalization mistakes.
void test_orthogonality() {
    const float kTol = 1e-4f;
    for (const Case& c : kCases) {
        const Shape shape{c.batch, c.seq_len, c.num_heads, c.head_dim};
        const long n = shape.elems();
        const std::vector<float> x = random_host_data(n, 0x1234u + c.head_dim);
        std::vector<float> tmp(n);
        std::vector<float> back(n);
        ref_fwht_fp32(x.data(), tmp.data(), shape.rows(), shape.head_dim);
        ref_fwht_fp32(tmp.data(), back.data(), shape.rows(), shape.head_dim);
        const float diff = max_abs_diff(x.data(), back.data(), n);
        report(diff <= kTol, "orthogonality (transform applied twice) " + fmt(c),
               "max|diff|=" + std::to_string(diff));
    }
}

// Sanity check on the tolerances: rounding an input in [-1, 1) to the target
// precision must cost far less than the allowed absolute error.
void test_precision_roundtrip() {
    for (DType dtype : {DType::kFp16, DType::kBFloat16}) {
        const long n = 4096;
        const std::vector<float> x = random_host_data(n, 0x5EEDu);
        std::vector<uint16_t> packed(n);
        std::vector<float> back(n);
        pack_fp32(x.data(), n, packed.data(), dtype);
        unpack_fp32(packed.data(), n, back.data(), dtype);
        const float diff = max_abs_diff(x.data(), back.data(), n);
        const float limit = 0.1f * tolerance_for(dtype);
        report(diff <= limit, std::string("precision round-trip ") + dtype_name(dtype),
               "max|diff|=" + std::to_string(diff) + " limit=" + std::to_string(limit));
    }
}

// Full GPU comparison against the CPU reference. Reports SKIP when the launcher rejects the
// shape or the element type.
void test_gpu_implementation(const char* impl_name, bool (*launch)(const void*, void*, const Shape&,
                                                                   DType, cudaStream_t)) {
    for (const Case& c : kCases) {
        const Shape shape{c.batch, c.seq_len, c.num_heads, c.head_dim};
        const long rows = shape.rows();
        const long n = shape.elems();
        for (DType dtype : {DType::kFp16, DType::kBFloat16}) {
            const std::string name =
                std::string(impl_name) + " " + dtype_name(dtype) + " " + fmt(c);

            const std::vector<float> host_x = random_host_data(n, 0xBEEFu + c.head_dim);
            std::vector<uint16_t> packed_x(n);
            pack_fp32(host_x.data(), n, packed_x.data(), dtype);

            // Reference computed on the rounded input, i.e. the exact input the
            // kernel sees.
            std::vector<float> rounded_x(n);
            unpack_fp32(packed_x.data(), n, rounded_x.data(), dtype);
            std::vector<float> expected(n);
            ref_matmul_fp32(rounded_x.data(), expected.data(), rows, shape.head_dim);

            void* d_x = nullptr;
            void* d_y = nullptr;
            HW_CUDA_CHECK(cudaMalloc(&d_x, n * sizeof(uint16_t)));
            HW_CUDA_CHECK(cudaMalloc(&d_y, n * sizeof(uint16_t)));
            HW_CUDA_CHECK(
                cudaMemcpy(d_x, packed_x.data(), n * sizeof(uint16_t), cudaMemcpyHostToDevice));

            const bool available = launch(d_x, d_y, shape, dtype, nullptr);
            if (!available) {
                skip(name, "launcher unsupported for this shape");
                HW_CUDA_CHECK(cudaFree(d_x));
                HW_CUDA_CHECK(cudaFree(d_y));
                continue;
            }

            std::vector<uint16_t> packed_y(n);
            HW_CUDA_CHECK(
                cudaMemcpy(packed_y.data(), d_y, n * sizeof(uint16_t), cudaMemcpyDeviceToHost));
            HW_CUDA_CHECK(cudaFree(d_x));
            HW_CUDA_CHECK(cudaFree(d_y));

            std::vector<float> got(n);
            unpack_fp32(packed_y.data(), n, got.data(), dtype);
            const float diff = max_abs_diff(expected.data(), got.data(), n);
            const float limit = tolerance_for(dtype);
            report(diff <= limit, name,
                   "max|diff|=" + std::to_string(diff) + " limit=" + std::to_string(limit));
        }
    }
}

// ---------------------------------------------------------------------------
// M4: FP8 E4M3 quantization
// ---------------------------------------------------------------------------

using TransformLaunch = bool (*)(const void*, void*, const Shape&, DType, cudaStream_t);
using FusedLaunch = bool (*)(const void*, uint8_t*, float*, const Shape&, DType, cudaStream_t);

// Every finite E4M3 code, its two float neighbours, and the midpoint between adjacent
// codes (the rounding boundary itself). Everything here comes from decoding codes, so
// none of it depends on the conversion the kernels use.
std::vector<float> fp8_grid_probes() {
    std::vector<float> grid;
    for (int code = 0x00; code <= 0x7E; ++code) {
        grid.push_back(fp8_e4m3_dequant(static_cast<uint8_t>(code)));
    }
    std::vector<float> probes = grid;
    for (size_t i = 0; i < grid.size(); ++i) {
        probes.push_back(std::nextafterf(grid[i], 0.0f));
        probes.push_back(std::nextafterf(grid[i], 1000.0f));
        if (i + 1 < grid.size()) {
            const float mid = 0.5f * (grid[i] + grid[i + 1]);
            probes.push_back(mid);
            probes.push_back(std::nextafterf(mid, 0.0f));
            probes.push_back(std::nextafterf(mid, 1000.0f));
        }
    }
    return probes;  // magnitudes; the caller adds the signs
}

// Round-to-nearest-even on the E4M3 grid, derived from the decoded codes alone. This is an
// independent statement of the rounding rule: the code conversion in the kernels is what
// it is being checked against, so one code of disagreement has to be a failure even
// though it is far inside any error tolerance. The sign bit is copied, which is what the
// conversion does with -0 as well.
uint8_t fp8_expected_code(float v) {
    const float a = std::fabs(v);
    int best = 0;
    float best_err = a;  // code 0 decodes to exactly 0
    for (int code = 1; code <= 0x7E; ++code) {
        const float err = std::fabs(a - fp8_e4m3_dequant(static_cast<uint8_t>(code)));
        // A tie keeps the code with an even mantissa, i.e. bit 0 clear. Scanning upwards,
        // the code already held is the lower one, so it wins ties unless it is the odd one
        // and the candidate is even.
        if (err < best_err || (err == best_err && (code & 1) == 0)) {
            best = code;
            best_err = err;
        }
    }
    return static_cast<uint8_t>(best | (std::signbit(v) ? 0x80 : 0x00));
}

// Half an E4M3 ulp is at most |v| / 16 <= amax / 16 for any element of a row, so that is
// the bound a correct round-to-nearest-even quantizer has to stay inside.
float fp8_quant_error_limit(float amax) {
    return amax * (1.0f / 16.0f) + 1e-6f;
}

void test_fp8_reference_roundtrip() {
    // 1) The rounding rule itself. A unit scale leaves the probes untouched, so the code
    //    that comes out is a direct answer to "which grid point is nearest?" -- including
    //    at the boundaries, where the answer is one code away from being wrong.
    const std::vector<float> magnitudes = fp8_grid_probes();
    std::vector<float> probes = magnitudes;
    for (float m : magnitudes) {
        probes.push_back(-m);
    }
    for (float edge : {448.0f, -448.0f, 449.0f, -449.0f, 1e10f, -1e10f, 1e-45f}) {
        probes.push_back(edge);
    }
    probes.push_back(0.0f);
    probes.push_back(-0.0f);

    const int n = static_cast<int>(probes.size());
    const float unit_scale = 1.0f;  // rscale = 1 / 1 = 1
    std::vector<uint8_t> q(static_cast<size_t>(n));
    ref_quantize_fp8(probes.data(), &unit_scale, q.data(), 1, n);

    long mismatches = 0;
    long first_bad = -1;
    for (int i = 0; i < n; ++i) {
        const uint8_t expected = fp8_expected_code(probes[static_cast<size_t>(i)]);
        if (q[static_cast<size_t>(i)] != expected) {
            if (first_bad < 0) {
                first_bad = i;
            }
            ++mismatches;
        }
    }
    std::string detail = "mismatches=" + std::to_string(mismatches) + "/" + std::to_string(n);
    if (first_bad >= 0) {
        detail += " first " + std::to_string(probes[static_cast<size_t>(first_bad)]) + " -> " +
                  std::to_string(q[static_cast<size_t>(first_bad)]) + " want " +
                  std::to_string(fp8_expected_code(probes[static_cast<size_t>(first_bad)]));
    }
    report(mismatches == 0, "fp8 rounding rule vs independent RTNE decode", detail);

    // 2) The per-token pipeline as the kernels see it: derive the scale from amax, quantize,
    //    dequantize. The row holds only grid probes, so amax is the top of the grid and the
    //    half-ulp bound below is tight rather than several orders of magnitude loose.
    std::vector<float> row = magnitudes;
    for (float m : magnitudes) {
        row.push_back(-m);
    }
    const int row_n = static_cast<int>(row.size());
    float amax = 0.0f;
    for (float v : row) {
        amax = std::max(amax, std::fabs(v));
    }
    std::vector<float> scales(1, 0.0f);
    std::vector<uint8_t> row_q(static_cast<size_t>(row_n));
    std::vector<float> back(static_cast<size_t>(row_n));
    ref_quant_scales_fp8(row.data(), scales.data(), 1, row_n);
    ref_quantize_fp8(row.data(), scales.data(), row_q.data(), 1, row_n);
    ref_dequantize_fp8(row_q.data(), scales.data(), 1, row_n, back.data());

    const float limit = fp8_quant_error_limit(amax);
    float worst = 0.0f;
    for (int i = 0; i < row_n; ++i) {
        worst =
            std::max(worst, std::fabs(back[static_cast<size_t>(i)] - row[static_cast<size_t>(i)]));
    }
    report(worst <= limit, "fp8 round-trip within half an E4M3 ulp",
           "err/(amax/16)=" + std::to_string(worst / limit) + " (amax=" + std::to_string(amax) +
               ", scale=" + std::to_string(scales[0]) + ")");

    // An all-zero row would divide by a zero scale when quantizing, and 0 * inf is a NaN:
    // the convention is scale = 1 and all codes zero.
    std::vector<float> zeros(64, 0.0f);
    std::vector<uint8_t> zero_q(zeros.size(), 0xABu);
    float zero_scale = -1.0f;
    ref_quant_scales_fp8(zeros.data(), &zero_scale, 1, 64);
    ref_quantize_fp8(zeros.data(), &zero_scale, zero_q.data(), 1, 64);
    bool zero_ok = zero_scale == 1.0f;
    for (uint8_t code : zero_q) {
        zero_ok = zero_ok && code == 0x00u;
    }
    report(zero_ok, "fp8 all-zero row", "scale=" + std::to_string(zero_scale) + " codes=0x00");
}

// The acceptance criterion for the fusion: the fused kernel has to produce exactly the
// codes and scales the two-stage pipeline produces, and both have to be within half an
// E4M3 ulp of the activation the non-fused transform wrote out.
void test_fused_fp8_consistency(const char* name, TransformLaunch transform, FusedLaunch fused) {
    for (const Case& c : kCases) {
        const Shape shape{c.batch, c.seq_len, c.num_heads, c.head_dim};
        const long rows = shape.rows();
        const long n = shape.elems();
        const int d = shape.head_dim;
        for (DType dtype : {DType::kFp16, DType::kBFloat16}) {
            const std::string label =
                std::string("fused fp8 ") + name + " " + dtype_name(dtype) + " " + fmt(c);

            const std::vector<float> host_x = random_host_data(n, 0xC0DEu + c.head_dim);
            std::vector<uint16_t> packed_x(static_cast<size_t>(n));
            pack_fp32(host_x.data(), n, packed_x.data(), dtype);

            void* d_x = nullptr;
            void* d_y = nullptr;
            uint8_t* d_q_ref = nullptr;
            uint8_t* d_q_fused = nullptr;
            float* d_s_ref = nullptr;
            float* d_s_fused = nullptr;
            HW_CUDA_CHECK(cudaMalloc(&d_x, static_cast<size_t>(n) * sizeof(uint16_t)));
            HW_CUDA_CHECK(cudaMalloc(&d_y, static_cast<size_t>(n) * sizeof(uint16_t)));
            HW_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_q_ref), static_cast<size_t>(n)));
            HW_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_q_fused), static_cast<size_t>(n)));
            HW_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_s_ref), rows * sizeof(float)));
            HW_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_s_fused), rows * sizeof(float)));
            HW_CUDA_CHECK(cudaMemcpy(d_x, packed_x.data(),
                                     static_cast<size_t>(n) * sizeof(uint16_t),
                                     cudaMemcpyHostToDevice));

            const bool two_stage_ok =
                transform(d_x, d_y, shape, dtype, nullptr) &&
                launch_quantize_fp8(d_y, d_q_ref, d_s_ref, shape, dtype, nullptr);
            const bool fused_ok = fused(d_x, d_q_fused, d_s_fused, shape, dtype, nullptr);
            if (!two_stage_ok || !fused_ok) {
                skip(label, "launcher unsupported for this shape");
                HW_CUDA_CHECK(cudaFree(d_x));
                HW_CUDA_CHECK(cudaFree(d_y));
                HW_CUDA_CHECK(cudaFree(d_q_ref));
                HW_CUDA_CHECK(cudaFree(d_q_fused));
                HW_CUDA_CHECK(cudaFree(d_s_ref));
                HW_CUDA_CHECK(cudaFree(d_s_fused));
                continue;
            }

            std::vector<uint8_t> q_ref(static_cast<size_t>(n));
            std::vector<uint8_t> q_fused(static_cast<size_t>(n));
            std::vector<float> s_ref(static_cast<size_t>(rows));
            std::vector<float> s_fused(static_cast<size_t>(rows));
            std::vector<uint16_t> packed_y(static_cast<size_t>(n));
            HW_CUDA_CHECK(
                cudaMemcpy(q_ref.data(), d_q_ref, static_cast<size_t>(n), cudaMemcpyDeviceToHost));
            HW_CUDA_CHECK(cudaMemcpy(q_fused.data(), d_q_fused, static_cast<size_t>(n),
                                     cudaMemcpyDeviceToHost));
            HW_CUDA_CHECK(
                cudaMemcpy(s_ref.data(), d_s_ref, rows * sizeof(float), cudaMemcpyDeviceToHost));
            HW_CUDA_CHECK(cudaMemcpy(s_fused.data(), d_s_fused, rows * sizeof(float),
                                     cudaMemcpyDeviceToHost));
            HW_CUDA_CHECK(cudaMemcpy(packed_y.data(), d_y,
                                     static_cast<size_t>(n) * sizeof(uint16_t),
                                     cudaMemcpyDeviceToHost));
            HW_CUDA_CHECK(cudaFree(d_x));
            HW_CUDA_CHECK(cudaFree(d_y));
            HW_CUDA_CHECK(cudaFree(d_q_ref));
            HW_CUDA_CHECK(cudaFree(d_q_fused));
            HW_CUDA_CHECK(cudaFree(d_s_ref));
            HW_CUDA_CHECK(cudaFree(d_s_fused));

            long codes_differing = 0;
            for (long i = 0; i < n; ++i) {
                if (q_ref[static_cast<size_t>(i)] != q_fused[static_cast<size_t>(i)]) {
                    ++codes_differing;
                }
            }
            float scale_diff = 0.0f;
            for (long r = 0; r < rows; ++r) {
                scale_diff = std::max(scale_diff, std::fabs(s_ref[static_cast<size_t>(r)] -
                                                            s_fused[static_cast<size_t>(r)]));
            }

            // Dequantize the fused output and compare it against the activation the
            // two-stage path actually wrote, row by row: the error is bounded by half an
            // E4M3 ulp, which is what the ratio below normalises by.
            std::vector<float> activation(static_cast<size_t>(n));
            std::vector<float> dequant(static_cast<size_t>(n));
            unpack_fp32(packed_y.data(), n, activation.data(), dtype);
            ref_dequantize_fp8(q_fused.data(), s_fused.data(), rows, d, dequant.data());
            float worst_ratio = 0.0f;
            for (long r = 0; r < rows; ++r) {
                float amax = 0.0f;
                for (int i = 0; i < d; ++i) {
                    amax = std::max(amax, std::fabs(activation[static_cast<size_t>(r * d + i)]));
                }
                const float limit = fp8_quant_error_limit(amax);
                for (int i = 0; i < d; ++i) {
                    const float err = std::fabs(dequant[static_cast<size_t>(r * d + i)] -
                                                activation[static_cast<size_t>(r * d + i)]);
                    worst_ratio = std::max(worst_ratio, err / limit);
                }
            }

            const bool ok = codes_differing == 0 && scale_diff == 0.0f && worst_ratio <= 1.0f;
            report(ok, label,
                   "codes differing=" + std::to_string(codes_differing) +
                       " |scale diff|=" + std::to_string(scale_diff) +
                       " err/(amax/16)=" + std::to_string(worst_ratio));
        }
    }
}

// The fused kernels have to follow the same all-zero convention as the reference.
void test_fp8_zero_input(const char* name, FusedLaunch fused) {
    const Shape shape{1, 1, 1, 128};
    const long n = shape.elems();
    const long rows = shape.rows();
    const std::vector<float> host_x(static_cast<size_t>(n), 0.0f);
    std::vector<uint16_t> packed_x(static_cast<size_t>(n));
    pack_fp32(host_x.data(), n, packed_x.data(), DType::kFp16);

    void* d_x = nullptr;
    uint8_t* d_q = nullptr;
    float* d_s = nullptr;
    HW_CUDA_CHECK(cudaMalloc(&d_x, static_cast<size_t>(n) * sizeof(uint16_t)));
    HW_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_q), static_cast<size_t>(n)));
    HW_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_s), rows * sizeof(float)));
    HW_CUDA_CHECK(cudaMemcpy(d_x, packed_x.data(), static_cast<size_t>(n) * sizeof(uint16_t),
                             cudaMemcpyHostToDevice));

    const bool launched = fused(d_x, d_q, d_s, shape, DType::kFp16, nullptr);
    if (!launched) {
        skip(std::string("fused fp8 ") + name + " all-zero input", "launcher unsupported");
    } else {
        std::vector<uint8_t> q(static_cast<size_t>(n), 0xABu);
        std::vector<float> s(static_cast<size_t>(rows), -1.0f);
        HW_CUDA_CHECK(cudaMemcpy(q.data(), d_q, static_cast<size_t>(n), cudaMemcpyDeviceToHost));
        HW_CUDA_CHECK(cudaMemcpy(s.data(), d_s, rows * sizeof(float), cudaMemcpyDeviceToHost));
        bool ok = true;
        for (uint8_t code : q) {
            ok = ok && code == 0x00u;
        }
        for (float scale : s) {
            ok = ok && scale == 1.0f;
        }
        report(ok, std::string("fused fp8 ") + name + " all-zero input",
               "scale=" + std::to_string(s[0]) + " codes=0x00");
    }
    HW_CUDA_CHECK(cudaFree(d_x));
    HW_CUDA_CHECK(cudaFree(d_q));
    HW_CUDA_CHECK(cudaFree(d_s));
}

}  // namespace

int main() {
    std::printf("== Hadamard correctness harness ==\n\n");
    test_reference_agreement();
    test_orthogonality();
    test_precision_roundtrip();
    std::printf("\n-- GPU implementations (M2 butterfly, M3 Tensor Core) --\n");
    test_gpu_implementation("fwht baseline", &hadamard::launch_fwht_baseline);
    test_gpu_implementation("tensor core", &hadamard::launch_hadamard_tc);

    std::printf("\n-- FP8 E4M3 quantization (M4) --\n");
    test_fp8_reference_roundtrip();
    test_fused_fp8_consistency("tensor core", &hadamard::launch_hadamard_tc,
                               &hadamard::launch_hadamard_tc_fp8);
    test_fused_fp8_consistency("butterfly", &hadamard::launch_fwht_baseline,
                               &hadamard::launch_fwht_baseline_fp8);
    test_fp8_zero_input("tensor core", &hadamard::launch_hadamard_tc_fp8);
    test_fp8_zero_input("butterfly", &hadamard::launch_fwht_baseline_fp8);

    std::printf("\n%d failed, %d skipped\n", g_failed, g_skipped);
    return g_failed == 0 ? 0 : 1;
}
