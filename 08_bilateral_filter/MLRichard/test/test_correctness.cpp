#include <iostream>
#include <vector>
#include <cstdlib>
#include <cuda_runtime.h>
#include "../src/bilateral_filter.cuh"
#include "../src/reference.h"
#include "../src/metrics.h"

namespace {

bool check_cuda(cudaError_t err, const char* step) {
    if (err == cudaSuccess) {
        return true;
    }

    std::cerr << "CUDA error at " << step << ": "
              << cudaGetErrorString(err) << std::endl;
    return false;
}

}

// 生成随机噪声图
std::vector<float> make_random_image(int w, int h, int c, unsigned seed = 42) {
    std::srand(seed);
    std::vector<float> img(w * h * c);
    for (auto& v : img) v = static_cast<float>(std::rand() % 256);
    return img;
}

bool run_test(const char* name, int W, int H, int C,
              const BilateralParams& params,
              void (*launcher)(const float*, float*, int, int, int, const BilateralParams&))
{
    auto input = make_random_image(W, H, C);
    int N = W * H * C;

    // CPU 参考
    std::vector<float> ref(N);
    bilateral_filter_cpu(input.data(), ref.data(), W, H, C, params);

    // GPU
    float *d_in = nullptr, *d_out = nullptr;
    if (!check_cuda(cudaMalloc(&d_in,  N * sizeof(float)), "cudaMalloc d_in") ||
        !check_cuda(cudaMalloc(&d_out, N * sizeof(float)), "cudaMalloc d_out") ||
        !check_cuda(cudaMemcpy(d_in, input.data(), N * sizeof(float), cudaMemcpyHostToDevice),
                    "cudaMemcpy H2D")) {
        if (d_in) cudaFree(d_in);
        if (d_out) cudaFree(d_out);
        return false;
    }

    std::vector<float> gpu_out(N);

    launcher(d_in, d_out, W, H, C, params);

    if (!check_cuda(cudaGetLastError(), "kernel launch/sync") ||
        !check_cuda(cudaMemcpy(gpu_out.data(), d_out, N * sizeof(float), cudaMemcpyDeviceToHost),
                    "cudaMemcpy D2H")) {
        cudaFree(d_in);
        cudaFree(d_out);
        return false;
    }
    cudaFree(d_in); cudaFree(d_out);

    float mae  = compute_mae(gpu_out.data(), ref.data(), N);
    float psnr = compute_psnr(gpu_out.data(), ref.data(), N);
    bool pass = mae < 1.0f;
    printf("[%s] W=%d H=%d C=%d  MAE=%.4f  PSNR=%.2f dB  %s\n",
           name, W, H, C, mae, psnr, pass ? "PASS" : "FAIL");
    return pass;
}

int main() {
    BilateralParams p{5, 3.0f, 30.0f};
    bool ok = true;

    ok &= run_test("naive_gray",   512, 512, 1, p, launch_bilateral_naive);
    ok &= run_test("naive_rgb",    512, 512, 3, p, launch_bilateral_naive);
    ok &= run_test("shared_gray",  512, 512, 1, p, launch_bilateral_shared);
    ok &= run_test("shared_rgb",   512, 512, 3, p, launch_bilateral_shared);
    ok &= run_test("adaptive_rgb", 512, 512, 3, p, launch_bilateral_adaptive);

    if (!ok) {
        std::cerr << "Correctness tests failed!" << std::endl;
        return 1;
    }

    std::cout << "All tests passed!" << std::endl;
    return 0;
}
