#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include "cuda_check.h"
#include "fuse/fuse_per_group_quant.cuh"

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
T from_float(float value);

template <>
__half from_float(float value) { return __float2half_rn(value); }

template <>
__nv_bfloat16 from_float(float value)
{
    return __float2bfloat16_rn(value);
}

template <typename T>
__global__ void per_group_reference(
    const T* input, uint8_t* output, float* scales, uint8_t* zero_points,
    int rows, int cols, int group_size, int quant_type, bool asymmetric)
{
    const int groups_per_row = cols / group_size;
    const int flat_group = blockIdx.x;
    const int row = flat_group / groups_per_row;
    const int group = flat_group % groups_per_row;
    const int begin = row * cols + group * group_size;
    const int tid = threadIdx.x;
    __shared__ float scale;
    __shared__ float inv_scale;
    __shared__ uint8_t zero_point;

    if (tid == 0) {
        float amax = 0.0f;
        float group_min = float(input[begin]);
        float group_max = group_min;
        for (int i = 0; i < group_size; ++i) {
            const float value = float(input[begin + i]);
            amax = fmaxf(amax, fabsf(value));
            if (asymmetric) {
                group_min = fminf(group_min, value);
                group_max = fmaxf(group_max, value);
            }
        }
        const float qmax = quant_type == int(FuseQuantType::Fp8E4M3)
            ? 448.0f
            : (quant_type == int(FuseQuantType::Int4) ? 7.0f : 127.0f);
        const float asymmetric_qmax =
            quant_type == int(FuseQuantType::Int4) ? 15.0f : 255.0f;
        if (asymmetric) {
            const float range = group_max - group_min;
            scale = range > 0.0f ? range / asymmetric_qmax : 1.0f;
            inv_scale = range > 0.0f ? asymmetric_qmax / range : 1.0f;
            zero_point = uint8_t(fminf(
                asymmetric_qmax,
                fmaxf(0.0f, rintf(-group_min * inv_scale))));
        } else {
            scale = amax > 0.0f ? amax / qmax : 1.0f;
            inv_scale = amax > 0.0f ? qmax / amax : 1.0f;
            zero_point = 0;
        }
        scales[flat_group] = scale;
        if (zero_points) zero_points[flat_group] = zero_point;
    }
    __syncthreads();

    auto quantize = [&](float value) {
        if (quant_type == int(FuseQuantType::Fp8E4M3))
            return uint8_t(__nv_cvt_float_to_fp8(
                value * inv_scale, __NV_SATFINITE, __NV_E4M3));
        float q = rintf(value * inv_scale) +
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
        for (int pair = tid; pair < group_size / 2; pair += blockDim.x) {
            const int col = group * group_size + 2 * pair;
            const uint8_t low = quantize(float(input[row * cols + col]));
            const uint8_t high = quantize(float(input[row * cols + col + 1]));
            output[row * (cols / 2) + col / 2] = low | (high << 4);
        }
    } else {
        for (int i = tid; i < group_size; i += blockDim.x) {
            const int col = group * group_size + i;
            output[row * cols + col] =
                quantize(float(input[row * cols + col]));
        }
    }
}

// Optimized standalone baseline for the common 8..256 group sizes. One warp
// reads a contiguous 256-element segment and partitions itself into sub-warps,
// matching the fused kernel's group reduction without recomputing Hadamard.
template <typename T>
__global__ void per_group_reference_vec(
    const T* input, uint8_t* output, float* scales, uint8_t* zero_points,
    int rows, int cols, int group_size, int quant_type, bool asymmetric)
{
    constexpr int kWarpSize = 32;
    constexpr int kVecSize = 8;
    constexpr int kSegmentSize = kWarpSize * kVecSize;
    const int lane = threadIdx.x & (kWarpSize - 1);
    const int warp_in_block = threadIdx.x / kWarpSize;
    const int warps_per_block = blockDim.x / kWarpSize;
    const int segments_per_row = cols / kSegmentSize;
    const int global_warp = blockIdx.x * warps_per_block + warp_in_block;
    if (global_warp >= rows * segments_per_row) return;
    const int row = global_warp / segments_per_row;
    const int segment = global_warp % segments_per_row;
    const int base_col = segment * kSegmentSize + lane * kVecSize;
    float value[kVecSize];
#pragma unroll
    for (int i = 0; i < kVecSize; ++i)
        value[i] = float(input[row * cols + base_col + i]);

    float amax = 0.0f;
    float group_min = value[0];
    float group_max = value[0];
#pragma unroll
    for (int i = 0; i < kVecSize; ++i) {
        amax = fmaxf(amax, fabsf(value[i]));
        if (asymmetric) {
            group_min = fminf(group_min, value[i]);
            group_max = fmaxf(group_max, value[i]);
        }
    }
    const int lanes_per_group = group_size / kVecSize;
    for (int offset = lanes_per_group / 2; offset > 0; offset >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(
            0xffffffffu, amax, offset, lanes_per_group));
        if (asymmetric) {
            group_min = fminf(group_min, __shfl_xor_sync(
                0xffffffffu, group_min, offset, lanes_per_group));
            group_max = fmaxf(group_max, __shfl_xor_sync(
                0xffffffffu, group_max, offset, lanes_per_group));
        }
    }

    const float qmax = quant_type == int(FuseQuantType::Fp8E4M3)
        ? 448.0f : (quant_type == int(FuseQuantType::Int4) ? 7.0f : 127.0f);
    const float asymmetric_qmax =
        quant_type == int(FuseQuantType::Int4) ? 15.0f : 255.0f;
    const float range = group_max - group_min;
    const float scale = asymmetric
        ? (range > 0.0f ? range / asymmetric_qmax : 1.0f)
        : (amax > 0.0f ? amax / qmax : 1.0f);
    const float inv_scale = asymmetric
        ? (range > 0.0f ? asymmetric_qmax / range : 1.0f)
        : (amax > 0.0f ? qmax / amax : 1.0f);
    const uint8_t zero_point = asymmetric
        ? uint8_t(fminf(asymmetric_qmax,
                        fmaxf(0.0f, rintf(-group_min * inv_scale))))
        : 0;
    const int groups_per_row = cols / group_size;
    const int group = segment * (kSegmentSize / group_size) +
                      lane / lanes_per_group;
    if ((lane % lanes_per_group) == 0) {
        scales[row * groups_per_row + group] = scale;
        if (zero_points)
            zero_points[row * groups_per_row + group] = zero_point;
    }

    auto quantize = [&](float x) {
        float q = rintf(x * inv_scale) +
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

    if (quant_type == int(FuseQuantType::Fp8E4M3)) {
        __nv_fp8x2_storage_t packed[kVecSize / 2];
#pragma unroll
        for (int i = 0; i < kVecSize / 2; ++i) {
            const float2 pair = make_float2(
                value[2 * i] * inv_scale, value[2 * i + 1] * inv_scale);
            packed[i] = __nv_cvt_float2_to_fp8x2(
                pair, __NV_SATFINITE, __NV_E4M3);
        }
        *reinterpret_cast<unsigned long long*>(
            output + row * cols + base_col) =
            *reinterpret_cast<const unsigned long long*>(packed);
    } else if (quant_type == int(FuseQuantType::Int4)) {
        uint32_t packed = 0;
#pragma unroll
        for (int i = 0; i < kVecSize / 2; ++i)
            packed |= uint32_t(quantize(value[2 * i]) |
                               (quantize(value[2 * i + 1]) << 4)) << (8 * i);
        *reinterpret_cast<uint32_t*>(
            output + row * (cols / 2) + base_col / 2) = packed;
    } else {
        uint8_t packed[kVecSize];
#pragma unroll
        for (int i = 0; i < kVecSize; ++i)
            packed[i] = quantize(value[i]);
        *reinterpret_cast<unsigned long long*>(
            output + row * cols + base_col) =
            *reinterpret_cast<const unsigned long long*>(packed);
    }
}

template <typename T>
bool check_one(const char* dtype, int rows, int cols, int group_size,
               FuseQuantType quant, FuseQuantScheme scheme,
               int iters, float warmup_ms)
{
    const size_t count = size_t(rows) * cols;
    const size_t output_bytes =
        quant == FuseQuantType::Int4 ? count / 2 : count;
    const int scale_count = rows * cols / group_size;
    const bool asymmetric = scheme == FuseQuantScheme::Asymmetric;
    std::vector<T> input(count);
    for (size_t i = 0; i < count; ++i)
        input[i] = from_float<T>(3.0f * sinf(0.017f * float(i)));

    T* d_input = nullptr;
    T* d_hadamard = nullptr;
    uint8_t* d_output = nullptr;
    uint8_t* d_reference = nullptr;
    float* d_scales = nullptr;
    float* d_reference_scales = nullptr;
    uint8_t* d_zero_points = nullptr;
    uint8_t* d_reference_zero_points = nullptr;
    CUDA_CHECK(cudaMalloc(&d_input, count * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_hadamard, count * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_output, output_bytes));
    CUDA_CHECK(cudaMalloc(&d_reference, output_bytes));
    CUDA_CHECK(cudaMalloc(&d_scales, scale_count * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_reference_scales, scale_count * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_zero_points, scale_count));
    CUDA_CHECK(cudaMalloc(&d_reference_zero_points, scale_count));
    CUDA_CHECK(cudaMemcpy(
        d_input, input.data(), count * sizeof(T), cudaMemcpyHostToDevice));

    auto fused = [&] {
        fused_hadamard_per_group_quantize<T>(
            d_input, d_output, d_scales,
            asymmetric ? d_zero_points : nullptr,
            rows, cols, group_size, quant, scheme);
    };
    auto separate = [&] {
        hadamard_v4<T>(d_input, d_hadamard, rows, cols, 0);
        if (cols >= 256 && group_size >= 8 && group_size <= 256) {
            constexpr int threads = 256;
            constexpr int warps_per_block = threads / 32;
            const int total_warps = rows * cols / 256;
            const int blocks =
                (total_warps + warps_per_block - 1) / warps_per_block;
            per_group_reference_vec<<<blocks, threads>>>(
                d_hadamard, d_reference, d_reference_scales,
                asymmetric ? d_reference_zero_points : nullptr,
                rows, cols, group_size, int(quant), asymmetric);
        } else {
            per_group_reference<<<scale_count, 256>>>(
                d_hadamard, d_reference, d_reference_scales,
                asymmetric ? d_reference_zero_points : nullptr,
                rows, cols, group_size, int(quant), asymmetric);
        }
    };
    fused();
    separate();
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_LAUNCH_CHECK();

    std::vector<uint8_t> output(output_bytes), reference(output_bytes);
    std::vector<float> scales(scale_count), reference_scales(scale_count);
    CUDA_CHECK(cudaMemcpy(
        output.data(), d_output, output_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(
        reference.data(), d_reference, output_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(scales.data(), d_scales,
                          scale_count * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(reference_scales.data(), d_reference_scales,
                          scale_count * sizeof(float), cudaMemcpyDeviceToHost));

    size_t mismatch = 0;
    for (size_t i = 0; i < output_bytes; ++i)
        mismatch += output[i] != reference[i];
    float scale_error = 0.0f;
    for (int i = 0; i < scale_count; ++i)
        scale_error = fmaxf(
            scale_error, fabsf(scales[i] - reference_scales[i]));
    size_t zero_mismatch = 0;
    if (asymmetric) {
        std::vector<uint8_t> zero(scale_count), reference_zero(scale_count);
        CUDA_CHECK(cudaMemcpy(zero.data(), d_zero_points,
                              scale_count, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(reference_zero.data(), d_reference_zero_points,
                              scale_count, cudaMemcpyDeviceToHost));
        for (int i = 0; i < scale_count; ++i)
            zero_mismatch += zero[i] != reference_zero[i];
    }

    const char* qname = quant == FuseQuantType::Int8 ? "int8" :
        (quant == FuseQuantType::Int4 ? "int4" : "fp8");
    const char* sname = asymmetric ? "asym" :
        (scheme == FuseQuantScheme::ScaleOnly ? "scale" : "sym");
    const bool passed =
        mismatch == 0 && zero_mismatch == 0 && scale_error == 0.0f;
    auto time_ms = [&](auto launch) {
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
    };
    const float fused_ms = time_ms(fused);
    const float separate_ms = time_ms(separate);
    std::printf(
        "dtype=%-4s quant=%-4s scheme=%-5s rows=%d cols=%5d group=%5d "
        "| mismatch=%zu zp=%zu scale=%.1e [%s] "
        "| fused=%8.3f ms separate=%8.3f ms speedup=%5.2fx\n",
        dtype, qname, sname, rows, cols, group_size,
        mismatch, zero_mismatch, scale_error, passed ? "PASS" : "FAIL",
        fused_ms, separate_ms, separate_ms / fused_ms);

    CUDA_CHECK(cudaFree(d_reference_zero_points));
    CUDA_CHECK(cudaFree(d_zero_points));
    CUDA_CHECK(cudaFree(d_reference_scales));
    CUDA_CHECK(cudaFree(d_scales));
    CUDA_CHECK(cudaFree(d_reference));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_hadamard));
    CUDA_CHECK(cudaFree(d_input));
    return passed;
}

int main(int argc, char** argv)
{
    int rows = 2048;
    int cols = 8192;
    int group_size = 128;
    int iters = 100;
    float warmup_ms = 20.0f;
    std::string dtype = "all";
    std::string quant = "all";
    std::string scheme = "all";
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto next = [&]() {
            if (++i >= argc) {
                std::fprintf(stderr, "missing value for %s\n", arg.c_str());
                std::exit(1);
            }
            return argv[i];
        };
        if (arg == "--rows") rows = std::atoi(next());
        else if (arg == "--cols") cols = std::atoi(next());
        else if (arg == "--group-size") group_size = std::atoi(next());
        else if (arg == "--dtype") dtype = next();
        else if (arg == "--quant") quant = next();
        else if (arg == "--scheme") scheme = next();
        else if (arg == "--iters") iters = std::atoi(next());
        else if (arg == "--warmup-ms") warmup_ms = std::atof(next());
        else {
            std::fprintf(stderr, "unknown argument: %s\n", arg.c_str());
            return 1;
        }
    }
    if (rows <= 0 || cols < 2 || cols > 32768 ||
        (cols & (cols - 1)) != 0 || group_size <= 0 ||
        (group_size & (group_size - 1)) != 0 || cols % group_size != 0 ||
        iters <= 0 || warmup_ms < 0.0f) {
        std::fprintf(stderr, "invalid rows/cols/group-size/iters/warmup-ms\n");
        return 1;
    }
    if (dtype != "fp16" && dtype != "bf16" && dtype != "all") {
        std::fprintf(stderr, "--dtype must be fp16, bf16, or all\n");
        return 1;
    }
    if (quant != "int8" && quant != "int4" &&
        quant != "fp8" && quant != "all") {
        std::fprintf(stderr, "--quant must be int8, int4, fp8, or all\n");
        return 1;
    }
    if (scheme != "symmetric" && scheme != "asymmetric" &&
        scheme != "scale_only" && scheme != "all") {
        std::fprintf(stderr,
                     "--scheme must be symmetric, asymmetric, scale_only, or all\n");
        return 1;
    }
    if ((quant == "int8" || quant == "int4") && scheme == "scale_only") {
        std::fprintf(stderr, "integer quantization has no scale_only mode\n");
        return 1;
    }
    if (quant == "fp8" && scheme != "scale_only" && scheme != "all") {
        std::fprintf(stderr, "FP8 only supports scale_only\n");
        return 1;
    }
    if ((quant == "int4" || quant == "all") && group_size < 2) {
        std::fprintf(stderr, "INT4 requires group-size >= 2\n");
        return 1;
    }

    bool passed = true;
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
            if (dtype == "fp16" || dtype == "all")
                passed &= check_one<__half>(
                    "fp16", rows, cols, group_size, q, s,
                    iters, warmup_ms);
            if (dtype == "bf16" || dtype == "all")
                passed &= check_one<__nv_bfloat16>(
                    "bf16", rows, cols, group_size, q, s,
                    iters, warmup_ms);
        }
    }
    return passed ? 0 : 1;
}
