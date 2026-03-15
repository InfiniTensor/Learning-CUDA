#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <cstdlib>
#include <cuda_runtime.h>
#include "../src/bilateral_filter.cuh"
#include "../src/reference.h"
#include "../src/metrics.h"

struct BenchResult {
    std::string name;
    float gpu_ms;
    float cpu_ms;
    float mae;
    float psnr;
};

BenchResult benchmark(
    const char* name, int W, int H, int C,
    const BilateralParams& params,
    void (*launcher)(const float*, float*, int, int, int, const BilateralParams&),
    int warmup = 3, int iters = 10)
{
    int N = W * H * C;
    std::vector<float> input(N);
    std::srand(123);
    for (auto& v : input) v = static_cast<float>(std::rand() % 256);

    float *d_in, *d_out;
    cudaMalloc(&d_in,  N * sizeof(float));
    cudaMalloc(&d_out, N * sizeof(float));
    cudaMemcpy(d_in, input.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    // Warmup
    for (int i = 0; i < warmup; i++) launcher(d_in, d_out, W, H, C, params);

    // Timed iters
    cudaEvent_t s, e;
    cudaEventCreate(&s); cudaEventCreate(&e);
    cudaEventRecord(s);
    for (int i = 0; i < iters; i++) launcher(d_in, d_out, W, H, C, params);
    cudaEventRecord(e);
    cudaEventSynchronize(e);
    float total_ms; cudaEventElapsedTime(&total_ms, s, e);
    float gpu_ms = total_ms / iters;

    std::vector<float> gpu_out(N);
    cudaMemcpy(gpu_out.data(), d_out, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_in); cudaFree(d_out);

    // CPU 参考（只跑一次）
    std::vector<float> ref(N);
    auto t0 = std::chrono::high_resolution_clock::now();
    bilateral_filter_cpu(input.data(), ref.data(), W, H, C, params);
    auto t1 = std::chrono::high_resolution_clock::now();
    float cpu_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();

    float mae  = compute_mae (gpu_out.data(), ref.data(), N);
    float psnr = compute_psnr(gpu_out.data(), ref.data(), N);

    float mpix = (float)(W * H) / 1e6f;
    printf("[%-20s] %dx%d C=%d  GPU=%.2fms  Throughput=%.1fMPix/s  CPU=%.1fms  Speedup=%.1fx  MAE=%.4f  PSNR=%.2fdB\n",
           name, W, H, C, gpu_ms, mpix/(gpu_ms/1000.f), cpu_ms, cpu_ms/gpu_ms, mae, psnr);

    return {name, gpu_ms, cpu_ms, mae, psnr};
}

int main() {
    BilateralParams p{5, 3.0f, 30.0f};

    printf("=== Bilateral Filter Benchmark ===\n\n");

    // 512x512 对比三个 kernel
    benchmark("naive_512_rgb",    512,  512,  3, p, launch_bilateral_naive);
    benchmark("shared_512_rgb",   512,  512,  3, p, launch_bilateral_shared);
    benchmark("adaptive_512_rgb", 512,  512,  3, p, launch_bilateral_adaptive);

    // 4K 性能测试（主要目标）
    benchmark("naive_4K_rgb",     3840, 2160, 3, p, launch_bilateral_naive);
    benchmark("shared_4K_rgb",    3840, 2160, 3, p, launch_bilateral_shared);
    benchmark("adaptive_4K_rgb",  3840, 2160, 3, p, launch_bilateral_adaptive);

    return 0;
}
