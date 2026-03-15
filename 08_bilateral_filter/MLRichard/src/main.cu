#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>
#include "io.h"
#include "bilateral_filter.cuh"
#include "reference.h"
#include "metrics.h"

BilateralParams load_params(const std::string& path) {
    BilateralParams p{5, 3.0f, 30.0f};
    std::ifstream f(path);
    if (!f) { std::cerr << "params file not found, using defaults\n"; return p; }
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        auto eq = line.find('=');
        if (eq == std::string::npos) continue;
        std::string key = line.substr(0, eq);
        std::string val = line.substr(eq + 1);
        key.erase(0, key.find_first_not_of(" \t"));
        key.erase(key.find_last_not_of(" \t") + 1);
        val.erase(0, val.find_first_not_of(" \t"));
        auto hash = val.find('#');
        if (hash != std::string::npos) val = val.substr(0, hash);
        if (key == "radius")             p.radius        = std::stoi(val);
        else if (key == "sigma_spatial") p.sigma_spatial = std::stof(val);
        else if (key == "sigma_color")   p.sigma_color   = std::stof(val);
    }
    return p;
}

int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0]
                  << " <input.raw> <output.raw> <params.txt> [kernel: naive|shared|adaptive]\n";
        return 1;
    }
    std::string input_path  = argv[1];
    std::string output_path = argv[2];
    std::string params_path = argv[3];
    std::string kernel_type = argc > 4 ? argv[4] : "shared";

    BilateralParams params = load_params(params_path);
    Image img = read_raw(input_path);

    int W = img.width, H = img.height, C = img.channels;
    int N = W * H * C;

    auto input_f = image_to_float(img);

    float *d_in, *d_out;
    cudaMalloc(&d_in,  N * sizeof(float));
    cudaMalloc(&d_out, N * sizeof(float));
    cudaMemcpy(d_in, input_f.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);

    if      (kernel_type == "naive")    launch_bilateral_naive   (d_in, d_out, W, H, C, params);
    else if (kernel_type == "shared")   launch_bilateral_shared  (d_in, d_out, W, H, C, params);
    else if (kernel_type == "adaptive") launch_bilateral_adaptive(d_in, d_out, W, H, C, params);
    else { std::cerr << "Unknown kernel: " << kernel_type << "\n"; return 1; }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float gpu_ms = 0.f;
    cudaEventElapsedTime(&gpu_ms, start, stop);

    std::vector<float> output_f(N);
    cudaMemcpy(output_f.data(), d_out, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_in); cudaFree(d_out);

    Image out_img = float_to_image(output_f.data(), W, H, C);
    write_raw(output_path, out_img);

    std::vector<float> ref_f(N);
    auto t0 = std::chrono::high_resolution_clock::now();
    bilateral_filter_cpu(input_f.data(), ref_f.data(), W, H, C, params);
    auto t1 = std::chrono::high_resolution_clock::now();
    float cpu_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();

    float mae  = compute_mae (output_f.data(), ref_f.data(), N);
    float psnr = compute_psnr(output_f.data(), ref_f.data(), N);
    float mpix = (float)(W * H) / 1e6f;

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    printf("\n=== Performance Log ===\n");
    printf("Platform     : %s\n", prop.name);
    printf("Image        : %dx%d %s\n", W, H, C == 3 ? "RGB" : "Gray");
    printf("Kernel       : %s\n", kernel_type.c_str());
    printf("Radius       : %d, sigma_s=%.1f, sigma_c=%.1f\n",
           params.radius, params.sigma_spatial, params.sigma_color);
    printf("GPU Time     : %.2f ms\n", gpu_ms);
    printf("Throughput   : %.1f MPix/s\n", mpix / (gpu_ms / 1000.f));
    printf("CPU Time     : %.2f ms (OpenCV)\n", cpu_ms);
    printf("Speedup      : %.1fx\n", cpu_ms / gpu_ms);
    printf("MAE          : %.4f\n", mae);
    printf("PSNR         : %.2f dB\n", psnr);

    return 0;
}
