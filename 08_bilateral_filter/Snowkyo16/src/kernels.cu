#include "bilateral_filter.cuh"
#include "utils.cuh"

#include <cmath>
#include <iostream>

using namespace std;

// V1: Naive CUDA 双边滤波 Kernel
__global__ void bilateral_filter_kernel_v1(
    const uint8_t* input,   // 输入图像（device 全局内存）
    uint8_t* output,        // 输出图像（device 全局内存）
    int width,
    int height,
    int channels,
    int radius,
    float spatial_denom,    // 预计算: 2 * sigma_spatial^2
    float color_denom       // 预计算: 2 * sigma_color^2
) {
    // 计算当前线程负责的像素坐标
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // 边界保护：当图像尺寸不是 blockDim 的整数倍时，
    // 最后一排线程块会有多余线程，直接返回
    if (x >= width || y >= height) {
        return;
    }

    // ---- 以下逻辑与 CPU 版完全一致 ----

    // 中心像素在一维数组中的偏移量
    int p_idx = (y * width + x) * channels;

    // 读取中心像素的颜色值
    float p_color[3];
    for (int ch = 0; ch < channels; ch++) {
        p_color[ch] = (float)input[p_idx + ch];
    }

    // 累加器
    float sum_value[3] = {0.0f, 0.0f, 0.0f};
    float sum_weight = 0.0f;

    // 遍历邻域窗口
    for (int dy = -radius; dy <= radius; dy++) {
        for (int dx = -radius; dx <= radius; dx++) {
            int qx = x + dx;
            int qy = y + dy;

            // 边界处理：跳过越界像素
            if (qx < 0 || qx >= width || qy < 0 || qy >= height) {
                continue;
            }

            // 邻域像素的颜色值
            int q_idx = (qy * width + qx) * channels;
            float q_color[3];
            for (int ch = 0; ch < channels; ch++) {
                q_color[ch] = (float)input[q_idx + ch];
            }

            // 空间距离平方
            float spatial_dist_sq = (float)(dx * dx + dy * dy);

            // 颜色差异平方（各通道差的平方和）
            float color_diff_sq = 0.0f;
            for (int ch = 0; ch < channels; ch++) {
                float diff = p_color[ch] - q_color[ch];
                color_diff_sq += diff * diff;
            }

            // 高斯权重
            float w_spatial = expf(-spatial_dist_sq / spatial_denom);
            float w_color = expf(-color_diff_sq / color_denom);
            float weight = w_spatial * w_color;

            // 累加
            sum_weight += weight;
            for (int ch = 0; ch < channels; ch++) {
                sum_value[ch] += weight * q_color[ch];
            }
        }
    }

    // 归一化并写回输出
    int out_idx = (y * width + x) * channels;
    for (int ch = 0; ch < channels; ch++) {
        float val = sum_value[ch] / sum_weight;
        // roundf 四舍五入，clamp 到 [0, 255]
        val = fminf(255.0f, fmaxf(0.0f, roundf(val)));
        output[out_idx + ch] = (uint8_t)val;
    }
}

// V1 包装函数：CPU 端调用，负责内存管理和 kernel 启动
Image bilateral_filter_gpu_v1(const Image& input, const FilterParams& params) {
    int w = input.width;
    int h = input.height;
    int c = input.channels;
    size_t img_size = (size_t)w * h * c * sizeof(uint8_t);

    // 预计算高斯分母
    float spatial_denom = 2.0f * params.sigma_spatial * params.sigma_spatial;
    float color_denom = 2.0f * params.sigma_color * params.sigma_color;

    // step 1: 在 GPU 上分配输入/输出内存 ----
    uint8_t* d_input = nullptr;
    uint8_t* d_output = nullptr;
    RUNTIME_CHECK(cudaMalloc(&d_input, img_size));
    RUNTIME_CHECK(cudaMalloc(&d_output, img_size));

    // step 2: 把输入图像从 CPU 内存拷贝到 GPU 内存 ----
    RUNTIME_CHECK(cudaMemcpy(d_input, input.data.data(), img_size,
                             cudaMemcpyHostToDevice));

    // step 3: 配置 kernel 启动参数 ----
    // block 大小: 16×16 = 256 个线程（A100 每个 SM 最多 2048 线程）
    // grid 大小: 向上取整，确保覆盖所有像素
    dim3 block(16, 16);
    dim3 grid((w + block.x - 1) / block.x,
              (h + block.y - 1) / block.y);

    // 用CUDA Event精确测量kernel执行时间（不含内存传输）
    cudaEvent_t start, stop;
    RUNTIME_CHECK(cudaEventCreate(&start));
    RUNTIME_CHECK(cudaEventCreate(&stop));
    RUNTIME_CHECK(cudaEventRecord(start));
    
    // 启动 kernel
    bilateral_filter_kernel_v1<<<grid, block>>>(
        d_input, d_output,
        w, h, c,
        params.radius,
        spatial_denom, color_denom
    );

    RUNTIME_CHECK(cudaEventRecord(stop));
    RUNTIME_CHECK(cudaEventSynchronize(stop));

    // 打印 kernel纯计算时间
    float kernel_ms = 0.0f;
    RUNTIME_CHECK(cudaEventElapsedTime(&kernel_ms, start, stop));
    cout << "  Kernel时间:    " << kernel_ms << " ms" << endl;

    RUNTIME_CHECK(cudaEventDestroy(start));
    RUNTIME_CHECK(cudaEventDestroy(stop));

    // 检查 kernel 是否启动成功
    RUNTIME_CHECK(cudaGetLastError());

    // step 4: 把结果从 GPU 拷回 CPU ----
    Image output;
    output.width = w;
    output.height = h;
    output.channels = c;
    output.data.resize(w * h * c);
    RUNTIME_CHECK(cudaMemcpy(output.data.data(), d_output, img_size,
                             cudaMemcpyDeviceToHost));

    // step 5: 释放 GPU 内存 ----
    RUNTIME_CHECK(cudaFree(d_input));
    RUNTIME_CHECK(cudaFree(d_output));

    return output;
}
