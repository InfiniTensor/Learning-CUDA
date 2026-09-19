#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "reference.hpp"

#include <cstdio>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

#define CUDA_CHECK(call) do { const cudaError_t e = (call); if (e != cudaSuccess) \
    throw std::runtime_error(std::string(#call) + ": " + cudaGetErrorString(e)); } while (0)

template<class T> __host__ __device__ float read_value(T x);
template<> __host__ __device__ float read_value(__half x) { return __half2float(x); }
template<> __host__ __device__ float read_value(__nv_bfloat16 x) { return __bfloat162float(x); }
template<class T> __host__ __device__ T store_value(float x);
template<> __host__ __device__ __half store_value(float x) { return __float2half_rn(x); }
template<> __host__ __device__ __nv_bfloat16 store_value(float x) { return __float2bfloat16_rn(x); }

// 不依赖 warp 宽度；每层先读到寄存器，再同步写回共享内存。
template<class T, bool Transform, bool Quantize>
__global__ void shared_fwht(const T* input, T* output, unsigned char* packed,
                            float* scales, int n, float scale) {
    __shared__ float values[256];
    __shared__ float row_scale;
    const int i = threadIdx.x;
    const std::size_t offset = static_cast<std::size_t>(blockIdx.x) * n;
    if (i < n) values[i] = read_value(input[offset + i]);
    __syncthreads();
    if constexpr (Transform) {
        for (int stride = 1; stride < n; stride *= 2) {
            float next = 0;
            if (i < n) {
                const float a = values[i], b = values[i ^ stride];
                next = (i & stride) ? b - a : a + b;
            }
            __syncthreads();
            if (i < n) values[i] = next;
            __syncthreads();
        }
        // 先舍入到公开输出精度，融合量化必须与分步路径保持相同语义。
        if (i < n) values[i] = read_value(store_value<T>(values[i] * scale));
        __syncthreads();
    }
    if constexpr (!Quantize) {
        if (i < n) output[offset + i] = store_value<T>(values[i]);
    } else {
        if (i == 0) {
            float magnitude = 0;
            for (int j = 0; j < n; ++j) magnitude = fmaxf(magnitude, fabsf(values[j]));
            row_scale = magnitude == 0 ? 1.0f : magnitude / 7.0f;
            scales[blockIdx.x] = row_scale;
        }
        __syncthreads();
        const int bytes = (n + 1) / 2;
        if (i < bytes) {
            unsigned char byte = 0;
            for (int k = 0; k < 2 && 2 * i + k < n; ++k) {
                const float x = values[2 * i + k] / row_scale;
                const float lower = floorf(x), fraction = x - lower;
                int q = static_cast<int>(lower);
                if (fraction > 0.5f || (fraction == 0.5f && q % 2 != 0)) ++q;
                q = q < -7 ? -7 : (q > 7 ? 7 : q);
                byte |= static_cast<unsigned char>((q & 15) << (4 * k));
            }
            packed[static_cast<std::size_t>(blockIdx.x) * bytes + i] = byte;
        }
    }
}

template<class T> T* allocate(std::size_t count) {
    T* ptr = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ptr), count * sizeof(T)));
    return ptr;
}

struct Summary { int cases = 0; std::size_t elements = 0; double max_error = 0; };

template<class T>
void validate_case(cudaStream_t stream, int rows, int n, float scale, int pattern,
                   const char* dtype, Summary& summary) {
    const std::size_t size = static_cast<std::size_t>(rows) * n;
    const std::size_t bytes = static_cast<std::size_t>(rows) * ((n + 1) / 2);
    std::vector<T> input(size), output(size);
    std::vector<float> rounded(size);
    std::uint32_t state = 0x735123abu;
    for (std::size_t i = 0; i < size; ++i) {
        state = state * 1664525u + 1013904223u;
        float value = (static_cast<float>(state >> 8) / 16777216.0f - 0.5f) * 0.5f;
        if (pattern == 0) value = 0;
        if (pattern == 1) value = i % n == (i / n) % n ? 0.625f : 0;
        input[i] = store_value<T>(value);
        rounded[i] = read_value(input[i]);
    }
    std::vector<unsigned char> split(bytes), fused(bytes);
    std::vector<float> split_scales(rows), fused_scales(rows);
    T* dx = allocate<T>(size);
    T* dy = allocate<T>(size);
    auto* dq = allocate<unsigned char>(bytes);
    auto* df = allocate<unsigned char>(bytes);
    float* ds = allocate<float>(rows);
    float* dfs = allocate<float>(rows);
    CUDA_CHECK(cudaMemcpyAsync(dx, input.data(), size * sizeof(T), cudaMemcpyHostToDevice, stream));
    const int threads = std::max(64, n);
    shared_fwht<T, true, false><<<rows, threads, 0, stream>>>(dx, dy, nullptr, nullptr, n, scale);
    CUDA_CHECK(cudaGetLastError());
    shared_fwht<T, false, true><<<rows, threads, 0, stream>>>(dy, nullptr, dq, ds, n, 1.0f);
    CUDA_CHECK(cudaGetLastError());
    shared_fwht<T, true, true><<<rows, threads, 0, stream>>>(dx, nullptr, df, dfs, n, scale);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpyAsync(output.data(), dy, size * sizeof(T), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(split.data(), dq, bytes, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(fused.data(), df, bytes, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(split_scales.data(), ds, rows * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(fused_scales.data(), dfs, rows * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaFree(dx)); CUDA_CHECK(cudaFree(dy)); CUDA_CHECK(cudaFree(dq));
    CUDA_CHECK(cudaFree(df)); CUDA_CHECK(cudaFree(ds)); CUDA_CHECK(cudaFree(dfs));

    const double tolerance = std::string(dtype) == "fp16" ? 1e-2 : 5e-2;
    const std::string context = std::string(dtype) + " rows=" + std::to_string(rows)
        + " n=" + std::to_string(n) + " pattern=" + std::to_string(pattern)
        + " scale=" + std::to_string(scale);
    std::vector<float> actual(size), expected = rounded;
    hadamard::fwht(expected.data(), rows, n, scale);
    for (std::size_t i = 0; i < size; ++i) {
        actual[i] = read_value(output[i]);
        const double error = std::abs(static_cast<double>(actual[i]) - expected[i]);
        if (!std::isfinite(actual[i]) || !(error < tolerance))
            throw std::runtime_error("CPU FWHT mismatch: " + context + " index=" + std::to_string(i));
        summary.max_error = std::max(summary.max_error, error);
    }
    for (int row : {0, rows / 2, rows - 1}) {
        const std::vector<float> sample(rounded.begin() + row * n, rounded.begin() + (row + 1) * n);
        const auto dense = hadamard::dense_reference(sample, n, scale);
        for (int i = 0; i < n; ++i) {
            const double error = std::abs(actual[row * n + i] - dense[i]);
            if (!(error < tolerance)) throw std::runtime_error("Dense oracle mismatch: " + context);
            summary.max_error = std::max(summary.max_error, error);
        }
    }
    const auto cpu_quant = hadamard::quantize_int4(actual, n);
    if (split != cpu_quant.packed || fused != split || split_scales != cpu_quant.scales
        || fused_scales != split_scales)
        throw std::runtime_error("INT4 bytes/scales mismatch: " + context);
    ++summary.cases;
    summary.elements += size;
}

// 独立验证正负半整数：scale=1，低位/高位和 N=1 的空高位另由全套用例覆盖。
template<class T> void validate_ties(cudaStream_t stream) {
    const std::vector<float> values{7, -7, 0.5f, 1.5f, 2.5f, -0.5f, -1.5f, -2.5f};
    std::vector<T> input;
    for (float value : values) input.push_back(store_value<T>(value));
    T* dx = allocate<T>(8);
    auto* dq = allocate<unsigned char>(4);
    float* ds = allocate<float>(1);
    std::vector<unsigned char> packed(4);
    float scale = 0;
    CUDA_CHECK(cudaMemcpyAsync(dx, input.data(), 8 * sizeof(T), cudaMemcpyHostToDevice, stream));
    shared_fwht<T, false, true><<<1, 64, 0, stream>>>(dx, nullptr, dq, ds, 8, 1.0f);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpyAsync(packed.data(), dq, 4, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(&scale, ds, sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaFree(dx)); CUDA_CHECK(cudaFree(dq)); CUDA_CHECK(cudaFree(ds));
    if (scale != 1.0f || packed != hadamard::quantize_int4(values, 8).packed)
        throw std::runtime_error("Positive/negative ties-to-even quantization failed");
}

template<class T> void validate_dtype(cudaStream_t stream, const char* dtype) {
    Summary summary;
    for (int n = 1; n <= 256; n *= 2)
        for (int rows : {1, 17, 257})
            for (float scale : {1.0f, 1.0f / std::sqrt(static_cast<float>(n))})
                for (int pattern = 0; pattern < 3; ++pattern)
                    validate_case<T>(stream, rows, n, scale, pattern, dtype, summary);
    validate_ties<T>(stream);
    std::cout << "PASS dtype=" << dtype << " cases=" << summary.cases
              << " elements=" << summary.elements << " max_abs_error="
              << std::setprecision(12) << summary.max_error << " ties_even=PASS\n";
}

int main() {
    try {
        cudaDeviceProp prop{};
        CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
        CUDA_CHECK(cudaSetDevice(0));
        int runtime = 0, driver = 0;
        CUDA_CHECK(cudaRuntimeGetVersion(&runtime));
        CUDA_CHECK(cudaDriverGetVersion(&driver));
        std::printf("VALIDATION_ONLY device=%s warp=%d runtime=%d driver=%d\n",
                    prop.name, prop.warpSize, runtime, driver);
        cudaStream_t stream;
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        validate_dtype<__half>(stream, "fp16");
        validate_dtype<__nv_bfloat16>(stream, "bf16");
        CUDA_CHECK(cudaStreamDestroy(stream));
        std::puts("PASS shared-memory FWHT, CPU oracle, split/fused INT4; no performance claim");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL %s\n", error.what());
        return 1;
    }
}
