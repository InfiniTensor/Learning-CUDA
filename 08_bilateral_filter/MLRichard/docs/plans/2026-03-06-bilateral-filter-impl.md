# 实时图像双边滤波（CUDA）实现计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 实现一个基于 CUDA 的实时双边滤波器，分三阶段优化，支持灰度和 RGB 图像，MAE < 1.0，并在 A100 上达到 4K 60fps。

**Architecture:** 模块化 C++/CUDA 项目，分离 IO、参考实现、CUDA kernel 和测试。三阶段 kernel：朴素版（基线）→ Shared Memory 优化版 → 自适应半径 + 查找表版。CPU 端调用 OpenCV bilateralFilter 作为正确性参考。

**Tech Stack:** CUDA 12+, CMake 3.20+, OpenCV 4.x, C++17

---

## Task 1: 项目脚手架与构建系统

**Files:**
- Create: `CMakeLists.txt`
- Create: `params.txt`
- Create: `src/main.cu`
- Create: `src/io.h`
- Create: `src/io.cpp`
- Create: `src/metrics.h`
- Create: `src/metrics.cpp`
- Create: `src/reference.h`
- Create: `src/reference.cpp`
- Create: `src/bilateral_filter.cuh`
- Create: `src/bilateral_filter.cu`
- Create: `test/test_correctness.cpp`
- Create: `test/test_performance.cpp`

**Step 1: 创建 CMakeLists.txt**

```cmake
cmake_minimum_required(VERSION 3.20)
project(bilateral_filter CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 75;80;86;89)

find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS} src)

# 主程序
add_executable(bilateral_filter
    src/main.cu
    src/bilateral_filter.cu
    src/io.cpp
    src/reference.cpp
    src/metrics.cpp
)
target_link_libraries(bilateral_filter ${OpenCV_LIBS})
set_target_properties(bilateral_filter PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# 正确性测试
add_executable(test_correctness
    test/test_correctness.cpp
    src/bilateral_filter.cu
    src/io.cpp
    src/reference.cpp
    src/metrics.cpp
)
target_link_libraries(test_correctness ${OpenCV_LIBS})
set_target_properties(test_correctness PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# 性能测试
add_executable(test_performance
    test/test_performance.cpp
    src/bilateral_filter.cu
    src/io.cpp
    src/reference.cpp
    src/metrics.cpp
)
target_link_libraries(test_performance ${OpenCV_LIBS})
set_target_properties(test_performance PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
```

**Step 2: 创建 params.txt**

```text
radius = 5
sigma_spatial = 3.0
sigma_color = 30.0
```

**Step 3: 创建目录结构并确认编译**

```bash
mkdir -p build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j$(nproc)
```

Expected: 编译成功，生成 `bilateral_filter`、`test_correctness`、`test_performance`

**Step 4: Commit**

```bash
git add CMakeLists.txt params.txt
git commit -m "feat: add CMake build system"
```

---

## Task 2: IO 模块（raw 文件读写）

**Files:**
- Modify: `src/io.h`
- Modify: `src/io.cpp`

**Step 1: 编写 io.h**

```cpp
#pragma once
#include <string>
#include <vector>
#include <cstdint>

struct Image {
    uint32_t width;
    uint32_t height;
    uint32_t channels;  // 1 (灰度) 或 3 (RGB)
    std::vector<uint8_t> data;  // 行主序，交错存储
};

// 读取 raw 格式文件
Image read_raw(const std::string& path);

// 写入 raw 格式文件
void write_raw(const std::string& path, const Image& img);

// 从 PNG/JPG 转换为 raw（借助 OpenCV）
Image load_image(const std::string& path);

// 将 float 数组（0-255）转换回 uint8 Image
Image float_to_image(const float* data, uint32_t w, uint32_t h, uint32_t c);

// 将 Image 转换为 float 数组（0-255）
std::vector<float> image_to_float(const Image& img);
```

**Step 2: 编写 io.cpp**

```cpp
#include "io.h"
#include <fstream>
#include <stdexcept>
#include <opencv2/opencv.hpp>

Image read_raw(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open: " + path);
    Image img;
    f.read(reinterpret_cast<char*>(&img.width),    4);
    f.read(reinterpret_cast<char*>(&img.height),   4);
    f.read(reinterpret_cast<char*>(&img.channels), 4);
    size_t n = (size_t)img.width * img.height * img.channels;
    img.data.resize(n);
    f.read(reinterpret_cast<char*>(img.data.data()), n);
    return img;
}

void write_raw(const std::string& path, const Image& img) {
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot write: " + path);
    f.write(reinterpret_cast<const char*>(&img.width),    4);
    f.write(reinterpret_cast<const char*>(&img.height),   4);
    f.write(reinterpret_cast<const char*>(&img.channels), 4);
    f.write(reinterpret_cast<const char*>(img.data.data()), img.data.size());
}

Image load_image(const std::string& path) {
    cv::Mat mat = cv::imread(path, cv::IMREAD_UNCHANGED);
    if (mat.empty()) throw std::runtime_error("Cannot load image: " + path);
    Image img;
    img.width    = mat.cols;
    img.height   = mat.rows;
    img.channels = mat.channels();
    if (img.channels == 3) cv::cvtColor(mat, mat, cv::COLOR_BGR2RGB);
    img.data.assign(mat.data, mat.data + mat.total() * mat.elemSize());
    return img;
}

Image float_to_image(const float* data, uint32_t w, uint32_t h, uint32_t c) {
    Image img;
    img.width = w; img.height = h; img.channels = c;
    size_t n = (size_t)w * h * c;
    img.data.resize(n);
    for (size_t i = 0; i < n; i++) {
        float v = data[i];
        if (v < 0.f) v = 0.f;
        if (v > 255.f) v = 255.f;
        img.data[i] = static_cast<uint8_t>(v + 0.5f);
    }
    return img;
}

std::vector<float> image_to_float(const Image& img) {
    size_t n = (size_t)img.width * img.height * img.channels;
    std::vector<float> f(n);
    for (size_t i = 0; i < n; i++) f[i] = static_cast<float>(img.data[i]);
    return f;
}
```

**Step 3: 确认编译通过**

```bash
cd build && make -j$(nproc) 2>&1 | grep -E "error|warning|Built"
```

**Step 4: Commit**

```bash
git add src/io.h src/io.cpp
git commit -m "feat: add raw image IO module"
```

---

## Task 3: Metrics 模块（MAE / PSNR）

**Files:**
- Modify: `src/metrics.h`
- Modify: `src/metrics.cpp`

**Step 1: 编写 metrics.h**

```cpp
#pragma once
#include <vector>

// 计算平均绝对误差 MAE（输入为 float 数组，值域 0-255）
float compute_mae(const float* a, const float* b, int n);

// 计算峰值信噪比 PSNR（最大值 255）
float compute_psnr(const float* a, const float* b, int n);
```

**Step 2: 编写 metrics.cpp**

```cpp
#include "metrics.h"
#include <cmath>
#include <numeric>

float compute_mae(const float* a, const float* b, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) sum += std::fabs(a[i] - b[i]);
    return static_cast<float>(sum / n);
}

float compute_psnr(const float* a, const float* b, int n) {
    double mse = 0.0;
    for (int i = 0; i < n; i++) {
        double d = a[i] - b[i];
        mse += d * d;
    }
    mse /= n;
    if (mse < 1e-10) return 100.0f;
    return static_cast<float>(10.0 * std::log10(255.0 * 255.0 / mse));
}
```

**Step 3: Commit**

```bash
git add src/metrics.h src/metrics.cpp
git commit -m "feat: add MAE/PSNR metrics module"
```

---

## Task 4: CPU 参考实现（OpenCV bilateralFilter）

**Files:**
- Modify: `src/reference.h`
- Modify: `src/reference.cpp`

**Step 1: 编写 reference.h**

```cpp
#pragma once
#include <vector>
#include <cstdint>

struct BilateralParams {
    int   radius;
    float sigma_spatial;
    float sigma_color;
};

// CPU 参考实现，调用 OpenCV cv::bilateralFilter
// 输入/输出均为 float 数组（0-255），行主序，交错 RGB 或灰度
void bilateral_filter_cpu(
    const float* input,
    float*       output,
    int width, int height, int channels,
    const BilateralParams& params
);
```

**Step 2: 编写 reference.cpp**

```cpp
#include "reference.h"
#include <opencv2/opencv.hpp>
#include <stdexcept>

void bilateral_filter_cpu(
    const float* input, float* output,
    int w, int h, int c,
    const BilateralParams& p)
{
    // 转换为 OpenCV Mat（uint8）
    cv::Mat src(h, w, c == 3 ? CV_8UC3 : CV_8UC1);
    for (int i = 0; i < h * w * c; i++)
        src.data[i] = static_cast<uint8_t>(std::min(std::max(input[i], 0.f), 255.f));

    // OpenCV bilateralFilter 要求 BGR，我们存的是 RGB，需要转换
    if (c == 3) cv::cvtColor(src, src, cv::COLOR_RGB2BGR);

    cv::Mat dst;
    // d = 2*radius+1，sigma_color 和 sigma_space 与我们的参数对应
    cv::bilateralFilter(src, dst, 2 * p.radius + 1, p.sigma_color, p.sigma_spatial);

    if (c == 3) cv::cvtColor(dst, dst, cv::COLOR_BGR2RGB);

    for (int i = 0; i < h * w * c; i++)
        output[i] = static_cast<float>(dst.data[i]);
}
```

**Step 3: Commit**

```bash
git add src/reference.h src/reference.cpp
git commit -m "feat: add OpenCV CPU reference implementation"
```

---

## Task 5: Phase 1 — 朴素 CUDA Kernel

**Files:**
- Modify: `src/bilateral_filter.cuh`
- Modify: `src/bilateral_filter.cu`

**Step 1: 编写 bilateral_filter.cuh**

```cpp
#pragma once
#include "reference.h"  // BilateralParams

// Phase 1：朴素 kernel，每线程处理一个像素，直接读取 Global Memory
void launch_bilateral_naive(
    const float* d_input,
    float*       d_output,
    int width, int height, int channels,
    const BilateralParams& params
);

// Phase 2：Shared Memory 优化 kernel
void launch_bilateral_shared(
    const float* d_input,
    float*       d_output,
    int width, int height, int channels,
    const BilateralParams& params
);

// Phase 3：自适应半径 kernel
void launch_bilateral_adaptive(
    const float* d_input,
    float*       d_output,
    int width, int height, int channels,
    const BilateralParams& params
);
```

**Step 2: 编写 Phase 1 kernel（bilateral_filter.cu 初始版本）**

```cuda
#include "bilateral_filter.cuh"
#include <cuda_runtime.h>
#include <cmath>

#define BLOCK_SIZE 16

__device__ inline float color_dist_sq(
    const float* img, int idx_a, int idx_b, int channels)
{
    float sum = 0.f;
    for (int c = 0; c < channels; c++) {
        float d = img[idx_a * channels + c] - img[idx_b * channels + c];
        sum += d * d;
    }
    return sum;
}

__global__ void kernel_naive(
    const float* input, float* output,
    int W, int H, int C,
    int radius, float inv2_sigma_s2, float inv2_sigma_c2)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;

    int center = y * W + x;
    float weight_sum = 0.f;
    float val[3] = {0.f, 0.f, 0.f};

    for (int dy = -radius; dy <= radius; dy++) {
        int ny = y + dy;
        if (ny < 0 || ny >= H) continue;
        for (int dx = -radius; dx <= radius; dx++) {
            int nx = x + dx;
            if (nx < 0 || nx >= W) continue;
            int nbr = ny * W + nx;

            float spatial = (dx*dx + dy*dy) * inv2_sigma_s2;
            float color   = color_dist_sq(input, center, nbr, C) * inv2_sigma_c2;
            float w       = expf(-(spatial + color));

            weight_sum += w;
            for (int c = 0; c < C; c++)
                val[c] += w * input[nbr * C + c];
        }
    }

    for (int c = 0; c < C; c++)
        output[center * C + c] = val[c] / weight_sum;
}

void launch_bilateral_naive(
    const float* d_in, float* d_out,
    int W, int H, int C,
    const BilateralParams& p)
{
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((W + BLOCK_SIZE - 1) / BLOCK_SIZE,
              (H + BLOCK_SIZE - 1) / BLOCK_SIZE);
    float inv2s2 = -1.f / (2.f * p.sigma_spatial * p.sigma_spatial);
    float inv2c2 = -1.f / (2.f * p.sigma_color   * p.sigma_color);
    kernel_naive<<<grid, block>>>(d_in, d_out, W, H, C,
                                   p.radius, inv2s2, inv2c2);
    cudaDeviceSynchronize();
}

// Phase 2 和 Phase 3 占位（后续 Task 实现）
void launch_bilateral_shared(
    const float* d_in, float* d_out,
    int W, int H, int C, const BilateralParams& p)
{
    launch_bilateral_naive(d_in, d_out, W, H, C, p);  // 临时占位
}

void launch_bilateral_adaptive(
    const float* d_in, float* d_out,
    int W, int H, int C, const BilateralParams& p)
{
    launch_bilateral_naive(d_in, d_out, W, H, C, p);  // 临时占位
}
```

**Step 3: Commit**

```bash
git add src/bilateral_filter.cuh src/bilateral_filter.cu
git commit -m "feat: add Phase 1 naive CUDA bilateral filter kernel"
```

---

## Task 6: 正确性测试（test_correctness）

**Files:**
- Modify: `test/test_correctness.cpp`

**Step 1: 编写 test_correctness.cpp**

```cpp
#include <iostream>
#include <vector>
#include <cassert>
#include <cstdlib>
#include <cuda_runtime.h>
#include "../src/bilateral_filter.cuh"
#include "../src/reference.h"
#include "../src/metrics.h"

// 生成随机噪声图
std::vector<float> make_random_image(int w, int h, int c, unsigned seed = 42) {
    std::srand(seed);
    std::vector<float> img(w * h * c);
    for (auto& v : img) v = static_cast<float>(std::rand() % 256);
    return img;
}

void run_test(const char* name, int W, int H, int C,
              const BilateralParams& params,
              void (*launcher)(const float*, float*, int, int, int, const BilateralParams&))
{
    auto input = make_random_image(W, H, C);
    int N = W * H * C;

    // CPU 参考
    std::vector<float> ref(N);
    bilateral_filter_cpu(input.data(), ref.data(), W, H, C, params);

    // GPU
    float *d_in, *d_out;
    cudaMalloc(&d_in,  N * sizeof(float));
    cudaMalloc(&d_out, N * sizeof(float));
    cudaMemcpy(d_in, input.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    launcher(d_in, d_out, W, H, C, params);

    std::vector<float> gpu_out(N);
    cudaMemcpy(gpu_out.data(), d_out, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_in); cudaFree(d_out);

    float mae  = compute_mae(gpu_out.data(), ref.data(), N);
    float psnr = compute_psnr(gpu_out.data(), ref.data(), N);
    printf("[%s] W=%d H=%d C=%d  MAE=%.4f  PSNR=%.2f dB  %s\n",
           name, W, H, C, mae, psnr, mae < 1.0f ? "PASS" : "FAIL");
    assert(mae < 1.0f && "MAE >= 1.0, correctness check failed!");
}

int main() {
    BilateralParams p{5, 3.0f, 30.0f};

    run_test("naive_gray",  512, 512, 1, p, launch_bilateral_naive);
    run_test("naive_rgb",   512, 512, 3, p, launch_bilateral_naive);
    run_test("shared_gray", 512, 512, 1, p, launch_bilateral_shared);
    run_test("shared_rgb",  512, 512, 3, p, launch_bilateral_shared);

    std::cout << "All tests passed!" << std::endl;
    return 0;
}
```

**Step 2: 编译并运行测试**

```bash
cd build && make test_correctness -j$(nproc) && ./test_correctness
```

Expected output:
```
[naive_gray]  W=512 H=512 C=1  MAE=0.xxxx  PSNR=xx.xx dB  PASS
[naive_rgb]   W=512 H=512 C=3  MAE=0.xxxx  PSNR=xx.xx dB  PASS
All tests passed!
```

**Step 3: Commit**

```bash
git add test/test_correctness.cpp
git commit -m "test: add correctness validation with MAE/PSNR checks"
```

---

## Task 7: main.cu — 程序入口和参数解析

**Files:**
- Modify: `src/main.cu`

**Step 1: 编写 main.cu**

```cpp
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
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
        // 去除空格
        key.erase(0, key.find_first_not_of(" \t"));
        key.erase(key.find_last_not_of(" \t") + 1);
        val.erase(0, val.find_first_not_of(" \t"));
        // 去除注释
        auto hash = val.find('#');
        if (hash != std::string::npos) val = val.substr(0, hash);
        if (key == "radius")        p.radius        = std::stoi(val);
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

    // GPU 计时
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

    // 写输出
    Image out_img = float_to_image(output_f.data(), W, H, C);
    write_raw(output_path, out_img);

    // CPU 参考和指标
    std::vector<float> ref_f(N);
    auto t0 = std::chrono::high_resolution_clock::now();
    bilateral_filter_cpu(input_f.data(), ref_f.data(), W, H, C, params);
    auto t1 = std::chrono::high_resolution_clock::now();
    float cpu_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();

    float mae  = compute_mae (output_f.data(), ref_f.data(), N);
    float psnr = compute_psnr(output_f.data(), ref_f.data(), N);
    float mpix = (float)(W * H) / 1e6f;

    // 获取 GPU 名称
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
```

注意：需要在 main.cu 顶部添加 `#include <chrono>`。

**Step 2: 编译并用自动生成的测试图验证**

```bash
cd build && make bilateral_filter -j$(nproc)
# 生成测试 raw（可用 test_correctness 内部逻辑或 Python 脚本生成）
./bilateral_filter test.raw out.raw ../params.txt naive
```

**Step 3: Commit**

```bash
git add src/main.cu
git commit -m "feat: add main entry point with param parsing and perf logging"
```

---

## Task 8: Phase 2 — Shared Memory 优化 Kernel

**Files:**
- Modify: `src/bilateral_filter.cu`（替换 launch_bilateral_shared 占位实现）

**Step 1: 在 bilateral_filter.cu 中添加常量内存和 Shared Memory kernel**

在文件顶部添加常量内存声明：
```cuda
#define MAX_RADIUS 15
#define MAX_WEIGHT_SIZE ((2*MAX_RADIUS+1)*(2*MAX_RADIUS+1))
__constant__ float c_spatial_weight[MAX_WEIGHT_SIZE];
```

添加 Shared Memory kernel：
```cuda
#define TILE 16

__global__ void kernel_shared(
    const float* __restrict__ input,
    float* output,
    int W, int H, int C,
    int radius, float inv2_sigma_c2)
{
    // shared memory 布局：(TILE + 2*radius)^2 * C 个 float
    extern __shared__ float smem[];

    int tx = threadIdx.x, ty = threadIdx.y;
    int x  = blockIdx.x * TILE + tx;
    int y  = blockIdx.y * TILE + ty;

    int tile_w = TILE + 2 * radius;
    // 将 tile + halo 加载到 shared memory（多次加载覆盖 halo）
    for (int j = ty; j < tile_w; j += TILE) {
        for (int i = tx; i < tile_w; i += TILE) {
            int gx = blockIdx.x * TILE - radius + i;
            int gy = blockIdx.y * TILE - radius + j;
            // 边界钳制
            gx = max(0, min(W - 1, gx));
            gy = max(0, min(H - 1, gy));
            int src = (gy * W + gx) * C;
            int dst = (j * tile_w + i) * C;
            for (int c = 0; c < C; c++) smem[dst + c] = input[src + c];
        }
    }
    __syncthreads();

    if (x >= W || y >= H) return;

    int lx = tx + radius, ly = ty + radius;  // 当前像素在 smem 中的坐标
    int ci = (ly * tile_w + lx) * C;         // 中心像素在 smem 的偏移

    float weight_sum = 0.f;
    float val[3] = {0.f, 0.f, 0.f};
    int wi = 0;  // 空间权重索引

    for (int dy = -radius; dy <= radius; dy++) {
        for (int dx = -radius; dx <= radius; dx++, wi++) {
            int ni = ((ly + dy) * tile_w + (lx + dx)) * C;
            float color_sq = 0.f;
            for (int c = 0; c < C; c++) {
                float d = smem[ci + c] - smem[ni + c];
                color_sq += d * d;
            }
            float w = c_spatial_weight[wi] * expf(color_sq * inv2_sigma_c2);
            weight_sum += w;
            for (int c = 0; c < C; c++) val[c] += w * smem[ni + c];
        }
    }

    int out_idx = (y * W + x) * C;
    for (int c = 0; c < C; c++)
        output[out_idx + c] = val[c] / weight_sum;
}
```

更新 `launch_bilateral_shared`：
```cuda
void launch_bilateral_shared(
    const float* d_in, float* d_out,
    int W, int H, int C, const BilateralParams& p)
{
    int r = p.radius;
    // 预计算空间权重并上传到 Constant Memory
    int diam = 2 * r + 1;
    std::vector<float> sw(diam * diam);
    float inv2s2 = 1.f / (2.f * p.sigma_spatial * p.sigma_spatial);
    for (int dy = -r; dy <= r; dy++)
        for (int dx = -r; dx <= r; dx++)
            sw[(dy+r)*diam + (dx+r)] = expf(-(dx*dx + dy*dy) * inv2s2);
    cudaMemcpyToSymbol(c_spatial_weight, sw.data(), diam*diam*sizeof(float));

    float inv2c2 = -1.f / (2.f * p.sigma_color * p.sigma_color);
    int tile_w = TILE + 2 * r;
    size_t smem = (size_t)tile_w * tile_w * C * sizeof(float);

    dim3 block(TILE, TILE);
    dim3 grid((W + TILE - 1) / TILE, (H + TILE - 1) / TILE);
    kernel_shared<<<grid, block, smem>>>(d_in, d_out, W, H, C, r, inv2c2);
    cudaDeviceSynchronize();
}
```

**Step 2: 运行正确性测试确认 Phase 2 通过**

```bash
cd build && make test_correctness -j$(nproc) && ./test_correctness
```

Expected: 所有 PASS，包括 `shared_gray` 和 `shared_rgb`

**Step 3: Commit**

```bash
git add src/bilateral_filter.cu src/bilateral_filter.cuh
git commit -m "feat: add Phase 2 shared memory bilateral filter kernel"
```

---

## Task 9: Phase 3 — 自适应半径 + 查找表 Kernel

**Files:**
- Modify: `src/bilateral_filter.cu`（替换 launch_bilateral_adaptive 占位）

**Step 1: 添加颜色权重查找表常量内存**

```cuda
__constant__ float c_color_lut[256];  // lut[i] = exp(-i^2 / (2*sigma_c^2))
```

**Step 2: 添加 Sobel 梯度 kernel**

```cuda
__global__ void kernel_sobel(
    const float* __restrict__ input,
    float* gradient, int W, int H, int C)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < 1 || x >= W-1 || y < 1 || y >= H-1) {
        if (x < W && y < H) gradient[y*W+x] = 0.f;
        return;
    }
    // 用灰度值（取第一通道或均值）计算梯度
    auto px = [&](int xi, int yi) {
        float v = 0.f;
        for (int c = 0; c < C; c++) v += input[(yi*W+xi)*C+c];
        return v / C;
    };
    float gx = -px(x-1,y-1) + px(x+1,y-1)
               -2*px(x-1,y) + 2*px(x+1,y)
               -px(x-1,y+1) + px(x+1,y+1);
    float gy = -px(x-1,y-1) - 2*px(x,y-1) - px(x+1,y-1)
               +px(x-1,y+1) + 2*px(x,y+1) + px(x+1,y+1);
    gradient[y*W+x] = sqrtf(gx*gx + gy*gy);
}
```

**Step 3: 添加自适应 kernel**

```cuda
__global__ void kernel_adaptive(
    const float* __restrict__ input,
    const float* __restrict__ gradient,
    float* output,
    int W, int H, int C,
    int r_small, int r_large,
    float grad_thresh,
    float inv2_sigma_s2)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;

    int radius = gradient[y*W+x] > grad_thresh ? r_small : r_large;
    int center = y * W + x;
    float weight_sum = 0.f;
    float val[3] = {0.f, 0.f, 0.f};

    for (int dy = -radius; dy <= radius; dy++) {
        int ny = max(0, min(H-1, y+dy));
        for (int dx = -radius; dx <= radius; dx++) {
            int nx = max(0, min(W-1, x+dx));
            int nbr = ny * W + nx;
            float spatial_w = expf(-(dx*dx + dy*dy) * inv2_sigma_s2);
            // 颜色距离：各通道差绝对值的均值（近似）
            float cdist = 0.f;
            for (int c = 0; c < C; c++) {
                float d = fabsf(input[center*C+c] - input[nbr*C+c]);
                cdist += d;
            }
            cdist /= C;
            int lut_idx = min((int)cdist, 255);
            float color_w = c_color_lut[lut_idx];
            float w = spatial_w * color_w;
            weight_sum += w;
            for (int c = 0; c < C; c++) val[c] += w * input[nbr*C+c];
        }
    }
    for (int c = 0; c < C; c++)
        output[center*C+c] = val[c] / weight_sum;
}
```

**Step 4: 更新 launch_bilateral_adaptive**

```cuda
void launch_bilateral_adaptive(
    const float* d_in, float* d_out,
    int W, int H, int C, const BilateralParams& p)
{
    // 上传颜色查找表
    std::vector<float> lut(256);
    float inv2c2 = 1.f / (2.f * p.sigma_color * p.sigma_color);
    for (int i = 0; i < 256; i++) lut[i] = expf(-(float)(i*i) * inv2c2);
    cudaMemcpyToSymbol(c_color_lut, lut.data(), 256*sizeof(float));

    // 计算梯度
    float* d_grad;
    cudaMalloc(&d_grad, W * H * sizeof(float));
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((W+BLOCK_SIZE-1)/BLOCK_SIZE, (H+BLOCK_SIZE-1)/BLOCK_SIZE);
    kernel_sobel<<<grid, block>>>(d_in, d_grad, W, H, C);

    float inv2s2 = -1.f / (2.f * p.sigma_spatial * p.sigma_spatial);
    int r_small = max(1, p.radius / 2);
    int r_large = p.radius;
    float grad_thresh = 20.f;  // 可配置

    kernel_adaptive<<<grid, block>>>(d_in, d_grad, d_out, W, H, C,
                                      r_small, r_large, grad_thresh, inv2s2);
    cudaDeviceSynchronize();
    cudaFree(d_grad);
}
```

**Step 5: 运行正确性测试（Phase 3 允许 MAE 略高，但仍需 < 1.0）**

在 test_correctness.cpp 中添加：
```cpp
run_test("adaptive_rgb", 512, 512, 3, p, launch_bilateral_adaptive);
```

```bash
cd build && make -j$(nproc) && ./test_correctness
```

**Step 6: Commit**

```bash
git add src/bilateral_filter.cu
git commit -m "feat: add Phase 3 adaptive radius bilateral filter with LUT"
```

---

## Task 10: 性能 Benchmark（test_performance）

**Files:**
- Modify: `test/test_performance.cpp`

**Step 1: 编写 test_performance.cpp**

```cpp
#include <iostream>
#include <vector>
#include <string>
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

    cudaDeviceProp prop; cudaGetDeviceProperties(&prop, 0);
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
```

注意：需要在文件顶部添加 `#include <chrono>`。

**Step 2: 编译并运行**

```bash
cd build && make test_performance -j$(nproc) && ./test_performance
```

**Step 3: Commit**

```bash
git add test/test_performance.cpp
git commit -m "test: add performance benchmark covering 512x512 and 4K"
```

---

## Task 11: 工具脚本——PNG/JPG 转 raw

**Files:**
- Create: `tools/img2raw.py`

**Step 1: 编写 img2raw.py**

```python
#!/usr/bin/env python3
"""将 PNG/JPG 图像转换为项目使用的 raw 格式"""
import sys
import struct
import numpy as np
from PIL import Image

def img2raw(src, dst, grayscale=False):
    img = Image.open(src)
    if grayscale:
        img = img.convert('L')
        arr = np.array(img)
        c = 1
    else:
        img = img.convert('RGB')
        arr = np.array(img)
        c = 3
    h, w = arr.shape[:2]
    with open(dst, 'wb') as f:
        f.write(struct.pack('<III', w, h, c))
        f.write(arr.tobytes())
    print(f"Saved: {dst}  ({w}x{h} C={c})")

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: img2raw.py <input.png> <output.raw> [--gray]")
        sys.exit(1)
    img2raw(sys.argv[1], sys.argv[2], '--gray' in sys.argv)
```

**Step 2: Commit**

```bash
git add tools/img2raw.py
git commit -m "tools: add img2raw.py for converting PNG/JPG to raw format"
```

---

## Task 12: ncu/nsys Profiling

**Step 1: 用 ncu 分析 Phase 1 和 Phase 2 kernel**

```bash
# 在远程 A100 上执行
ncu --set full -o profile_naive    ./bilateral_filter test_4k.raw /dev/null ../params.txt naive
ncu --set full -o profile_shared   ./bilateral_filter test_4k.raw /dev/null ../params.txt shared
ncu --set full -o profile_adaptive ./bilateral_filter test_4k.raw /dev/null ../params.txt adaptive
```

**Step 2: 用 nsys 分析整体流程**

```bash
nsys profile -o timeline ./bilateral_filter test_4k.raw out.raw ../params.txt shared
```

**Step 3: 在本地用 Nsight Compute UI 打开 .ncu-rep 文件查看分析**

重点关注指标：
- `l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum.per_second`（Global Memory 带宽）
- `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum`（Shared Memory bank conflict）
- `sm__warps_active.avg.pct_of_peak_sustained_active`（Occupancy）

**Step 4: 将 profiling 截图和分析写入报告**

```bash
git add docs/report.md
git commit -m "docs: add profiling analysis and performance report"
```

---

## 验收标准

| 检查项 | 标准 |
|--------|------|
| MAE（Phase 1/2） | < 1.0 |
| MAE（Phase 3 自适应） | < 1.0 |
| 4K RGB Phase 2 throughput（A100） | > 1000 MPix/s（目标 60fps = 497 MPix/s） |
| 编译 | 无 error，无关键 warning |
| 测试 | test_correctness 全部 PASS |
