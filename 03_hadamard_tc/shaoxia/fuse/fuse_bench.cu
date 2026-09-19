#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include "cuda_check.h"
#include "fuse/fuse_per_row_quant.cuh"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

template <typename T>
void hadamard_v4(const T*, T*, int, int, cudaStream_t);

template <typename T>
T from_float(float x);

template <>
__half from_float(float x) { return __float2half_rn(x); }

template <>
__nv_bfloat16 from_float(float x) { return __float2bfloat16_rn(x); }

template <typename T>
__global__ void per_row_quant_kernel(
    const T* input, uint8_t* output, float* scales, int rows, int cols,
    uint8_t* zero_points, int quant_type, bool asymmetric)
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    if (row >= rows) return;

    float amax = 0.0f;
    float row_min = 0.0f;
    float row_max = 0.0f;
    if (asymmetric) {
        row_min = __int_as_float(0x7f800000);
        row_max = -__int_as_float(0x7f800000);
    }
    for (int col = tid; col < cols; col += blockDim.x) {
        const float v = float(input[(size_t)row * cols + col]);
        amax = fmaxf(amax, fabsf(v));
        if (asymmetric) {
            row_min = fminf(row_min, v);
            row_max = fmaxf(row_max, v);
        }
    }

    __shared__ float reduction_max[256];
    __shared__ float reduction_min[256];
    reduction_max[tid] = asymmetric ? row_max : amax;
    if (asymmetric) reduction_min[tid] = row_min;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            reduction_max[tid] = fmaxf(
                reduction_max[tid], reduction_max[tid + offset]);
            if (asymmetric)
                reduction_min[tid] = fminf(
                    reduction_min[tid], reduction_min[tid + offset]);
        }
        __syncthreads();
    }

    const float symmetric_qmax = quant_type == int(FuseQuantType::Fp8E4M3)
                                     ? 448.0f
                                     : (quant_type == int(FuseQuantType::Int4)
                                            ? 7.0f : 127.0f);
    const float asymmetric_qmax =
        quant_type == int(FuseQuantType::Int4) ? 15.0f : 255.0f;
    const float range = reduction_max[0] - reduction_min[0];
    const float scale = asymmetric
        ? (range > 0.0f ? range / asymmetric_qmax : 1.0f)
        : (reduction_max[0] > 0.0f
               ? reduction_max[0] / symmetric_qmax : 1.0f);
    const float inv_scale = asymmetric
        ? (range > 0.0f ? asymmetric_qmax / range : 1.0f)
        : (reduction_max[0] > 0.0f
               ? symmetric_qmax / reduction_max[0] : 1.0f);
    const uint8_t zero_point = asymmetric
        ? uint8_t(fminf(asymmetric_qmax,
                        fmaxf(0.0f, rintf(-reduction_min[0] * inv_scale))))
        : 0;
    if (tid == 0) {
        scales[row] = scale;
        if (zero_points) zero_points[row] = zero_point;
    }

    auto quantize = [&](float v) {
        if (quant_type == int(FuseQuantType::Fp8E4M3))
            return uint8_t(__nv_cvt_float_to_fp8(
                v * inv_scale, __NV_SATFINITE, __NV_E4M3));
        float q = rintf(v * inv_scale) +
                  (asymmetric ? float(zero_point) : 0.0f);
        if (quant_type == int(FuseQuantType::Int4)) {
            q = asymmetric ? fminf(15.0f, fmaxf(0.0f, q))
                           : fminf(7.0f, fmaxf(-7.0f, q));
            return uint8_t(uint8_t(int(q)) & 0x0f);
        }
        q = asymmetric ? fminf(255.0f, fmaxf(0.0f, q))
                       : fminf(127.0f, fmaxf(-127.0f, q));
        return asymmetric ? uint8_t(q) : uint8_t(int8_t(q));
    };

    if (quant_type == int(FuseQuantType::Int4)) {
        for (int pair = tid; pair < cols / 2; pair += blockDim.x) {
            const size_t col = size_t(pair) * 2;
            const uint8_t lo = quantize(float(input[(size_t)row * cols + col]));
            const uint8_t hi = quantize(float(input[(size_t)row * cols + col + 1]));
            output[(size_t)row * (cols / 2) + pair] = lo | (hi << 4);
        }
    } else {
        for (int col = tid; col < cols; col += blockDim.x)
            output[(size_t)row * cols + col] =
                quantize(float(input[(size_t)row * cols + col]));
    }
}

template <typename F>
float time_ms(F launch, int iters, float warmup_ms)
{
    cudaEvent_t begin, end;
    CUDA_CHECK(cudaEventCreate(&begin));
    CUDA_CHECK(cudaEventCreate(&end));

    CUDA_CHECK(cudaEventRecord(begin));
    float elapsed = 0.0f;
    do {
        launch();
        CUDA_CHECK(cudaEventRecord(end));
        CUDA_CHECK(cudaEventSynchronize(end));
        CUDA_CHECK(cudaEventElapsedTime(&elapsed, begin, end));
    } while (elapsed < warmup_ms);

    CUDA_CHECK(cudaEventRecord(begin));
    for (int i = 0; i < iters; ++i) launch();
    CUDA_CHECK(cudaEventRecord(end));
    CUDA_CHECK(cudaEventSynchronize(end));
    CUDA_CHECK(cudaEventElapsedTime(&elapsed, begin, end));

    CUDA_CHECK(cudaEventDestroy(begin));
    CUDA_CHECK(cudaEventDestroy(end));
    return elapsed / iters;
}

std::vector<int> parse_list(const char* text)
{
    std::vector<int> values;
    std::string s(text);
    size_t begin = 0;
    while (begin < s.size()) {
        size_t end = s.find(',', begin);
        std::string token = s.substr(begin, end - begin);
        int value = std::atoi(token.c_str());
        if (value <= 0) {
            std::fprintf(stderr, "invalid list: %s\n", text);
            std::exit(1);
        }
        values.push_back(value);
        if (end == std::string::npos) break;
        begin = end + 1;
    }
    return values;
}

template <typename T>
bool run_one(int rows, int cols, int iters, float warmup_ms,
             const char* dtype, FuseQuantType quant, FuseQuantScheme scheme,
             FuseKernelPolicy policy)
{
    const size_t count = (size_t)rows * cols;
    const size_t output_count = quant == FuseQuantType::Int4 ? count / 2 : count;
    const bool asymmetric = scheme == FuseQuantScheme::Asymmetric;
    std::vector<T> input(count);
    for (size_t i = 0; i < count; ++i)
        input[i] = from_float<T>(3.0f * sinf(0.017f * float(i)));

    T *d_input = nullptr, *d_hadamard = nullptr;
    uint8_t *d_fused = nullptr, *d_reference = nullptr;
    float *d_fused_scale = nullptr, *d_reference_scale = nullptr;
    uint8_t *d_fused_zero = nullptr, *d_reference_zero = nullptr;
    CUDA_CHECK(cudaMalloc(&d_input, count * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_hadamard, count * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_fused, output_count));
    CUDA_CHECK(cudaMalloc(&d_reference, output_count));
    CUDA_CHECK(cudaMalloc(&d_fused_scale, rows * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_reference_scale, rows * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_fused_zero, rows));
    CUDA_CHECK(cudaMalloc(&d_reference_zero, rows));
    CUDA_CHECK(cudaMemcpy(d_input, input.data(), count * sizeof(T),
                          cudaMemcpyHostToDevice));

    auto fused = [&] {
        fused_hadamard_per_row_quantize_with_policy<T>(
            d_input, d_fused, d_fused_scale,
            asymmetric ? d_fused_zero : nullptr,
            rows, cols, quant, scheme, policy, 0);
    };
    auto separate = [&] {
        hadamard_v4<T>(d_input, d_hadamard, rows, cols, 0);
        per_row_quant_kernel<T><<<rows, 256>>>(
            d_hadamard, d_reference, d_reference_scale, rows, cols,
            asymmetric ? d_reference_zero : nullptr,
            int(quant), asymmetric);
    };

    fused();
    separate();
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_LAUNCH_CHECK();

    std::vector<uint8_t> fused_out(output_count), reference_out(output_count);
    std::vector<float> fused_scale(rows), reference_scale(rows);
    CUDA_CHECK(cudaMemcpy(fused_out.data(), d_fused, output_count,
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(reference_out.data(), d_reference, output_count,
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(fused_scale.data(), d_fused_scale,
                          rows * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(reference_scale.data(), d_reference_scale,
                          rows * sizeof(float), cudaMemcpyDeviceToHost));

    size_t mismatches = 0;
    for (size_t i = 0; i < output_count; ++i)
        mismatches += fused_out[i] != reference_out[i];
    size_t zero_mismatches = 0;
    if (asymmetric) {
        std::vector<uint8_t> fused_zero(rows), reference_zero(rows);
        CUDA_CHECK(cudaMemcpy(fused_zero.data(), d_fused_zero, rows,
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(reference_zero.data(), d_reference_zero, rows,
                              cudaMemcpyDeviceToHost));
        for (int i = 0; i < rows; ++i)
            zero_mismatches += fused_zero[i] != reference_zero[i];
    }
    float max_scale_error = 0.0f;
    for (int i = 0; i < rows; ++i)
        max_scale_error = std::max(
            max_scale_error, fabsf(fused_scale[i] - reference_scale[i]));

    const float fused_ms = time_ms(fused, iters, warmup_ms);
    const float separate_ms = time_ms(separate, iters, warmup_ms);
    std::printf(
        "quant=%-4s scheme=%-10s policy=%-5s dtype=%-4s rows=%6d cols=%5d "
        "| mismatch=%zu zp_mismatch=%zu scale_err=%.4e [%s] "
        "| fused=%8.3f ms | separate=%8.3f ms | speedup=%5.2fx\n",
        quant == FuseQuantType::Fp8E4M3 ? "fp8" :
            (quant == FuseQuantType::Int4 ? "int4" : "int8"),
        scheme == FuseQuantScheme::Asymmetric ? "asymmetric" :
            (scheme == FuseQuantScheme::ScaleOnly ? "scale_only" : "symmetric"),
        policy == FuseKernelPolicy::MultiWarp ? "multi" :
            (policy == FuseKernelPolicy::Vec ? "vec" : "auto"),
        dtype, rows, cols, mismatches, zero_mismatches, max_scale_error,
        mismatches == 0 && zero_mismatches == 0 && max_scale_error == 0.0f
            ? "PASS" : "FAIL",
        fused_ms, separate_ms, separate_ms / fused_ms);

    CUDA_CHECK(cudaFree(d_reference_zero));
    CUDA_CHECK(cudaFree(d_fused_zero));
    CUDA_CHECK(cudaFree(d_reference_scale));
    CUDA_CHECK(cudaFree(d_fused_scale));
    CUDA_CHECK(cudaFree(d_reference));
    CUDA_CHECK(cudaFree(d_fused));
    CUDA_CHECK(cudaFree(d_hadamard));
    CUDA_CHECK(cudaFree(d_input));
    return mismatches == 0 && zero_mismatches == 0 && max_scale_error == 0.0f;
}

int main(int argc, char** argv)
{
    const char* rows_text = "2048";
    const char* dims_text = "256,512,1024,2048,4096,8192";
    std::string dtype = "all";
    std::string quant = "all";
    std::string scheme = "all";
    std::string policy = "auto";
    int iters = 100;
    float warmup_ms = 20.0f;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto next = [&]() {
            if (++i >= argc) {
                std::fprintf(stderr, "missing value for %s\n", arg.c_str());
                std::exit(1);
            }
            return argv[i];
        };
        if (arg == "--rows") rows_text = next();
        else if (arg == "--dims") dims_text = next();
        else if (arg == "--dtype") dtype = next();
        else if (arg == "--quant") quant = next();
        else if (arg == "--scheme") scheme = next();
        else if (arg == "--policy") policy = next();
        else if (arg == "--iters") iters = std::atoi(next());
        else if (arg == "--warmup-ms") warmup_ms = std::atof(next());
        else {
            std::fprintf(stderr, "unknown argument: %s\n", arg.c_str());
            return 1;
        }
    }
    if (dtype != "fp16" && dtype != "bf16" && dtype != "all") {
        std::fprintf(stderr, "--dtype must be fp16, bf16, or all\n");
        return 1;
    }
    if (quant != "int8" && quant != "int4" && quant != "fp8" && quant != "all") {
        std::fprintf(stderr, "--quant must be int8, int4, fp8, or all\n");
        return 1;
    }
    if (scheme != "symmetric" && scheme != "asymmetric" &&
        scheme != "scale_only" && scheme != "all") {
        std::fprintf(stderr,
                     "--scheme must be symmetric, asymmetric, scale_only, or all\n");
        return 1;
    }
    if (quant == "fp8" && scheme != "all" && scheme != "scale_only") {
        std::fprintf(stderr, "fp8 only supports --scheme scale_only\n");
        return 1;
    }
    if ((quant == "int8" || quant == "int4") && scheme == "scale_only") {
        std::fprintf(stderr, "integer quantization requires symmetric or asymmetric\n");
        return 1;
    }
    if (policy != "auto" && policy != "vec" && policy != "multi" &&
        policy != "all") {
        std::fprintf(stderr, "--policy must be auto, vec, multi, or all\n");
        return 1;
    }

    bool passed = true;
    for (int rows : parse_list(rows_text)) {
        for (int cols : parse_list(dims_text)) {
            if ((cols & (cols - 1)) != 0 || cols > 32768) {
                std::fprintf(stderr, "cols must be a power of two in [2, 32768]\n");
                return 1;
            }
            for (FuseQuantType q : {FuseQuantType::Int8,
                                    FuseQuantType::Int4,
                                    FuseQuantType::Fp8E4M3}) {
                const char* qname = q == FuseQuantType::Int8 ? "int8" :
                    (q == FuseQuantType::Int4 ? "int4" : "fp8");
                if (quant != "all" && quant != qname) continue;
                for (FuseQuantScheme s : {FuseQuantScheme::Symmetric,
                                          FuseQuantScheme::Asymmetric,
                                          FuseQuantScheme::ScaleOnly}) {
                    const char* sname = s == FuseQuantScheme::Symmetric
                        ? "symmetric" : (s == FuseQuantScheme::Asymmetric
                            ? "asymmetric" : "scale_only");
                    if (scheme != "all" && scheme != sname) continue;
                    if (q == FuseQuantType::Fp8E4M3) {
                        if (s != FuseQuantScheme::ScaleOnly) continue;
                    } else if (s == FuseQuantScheme::ScaleOnly) {
                        continue;
                    }
                    std::vector<FuseKernelPolicy> policies;
                    if (policy == "all" && cols >= 256 && cols <= 8192) {
                        policies = {FuseKernelPolicy::Vec,
                                    FuseKernelPolicy::MultiWarp};
                    } else if (policy == "vec") {
                        policies = {FuseKernelPolicy::Vec};
                    } else if (policy == "multi") {
                        policies = {FuseKernelPolicy::MultiWarp};
                    } else {
                        policies = {FuseKernelPolicy::Auto};
                    }
                    for (FuseKernelPolicy p : policies) {
                        if (dtype == "fp16" || dtype == "all")
                            passed &= run_one<__half>(
                                rows, cols, iters, warmup_ms, "fp16", q, s, p);
                        if (dtype == "bf16" || dtype == "all")
                            passed &= run_one<__nv_bfloat16>(
                                rows, cols, iters, warmup_ms, "bf16", q, s, p);
                    }
                }
            }
        }
    }
    return passed ? 0 : 1;
}
