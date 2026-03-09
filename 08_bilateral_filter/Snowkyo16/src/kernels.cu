#include "bilateral_filter.cuh"
#include "utils.cuh"

#include <cmath>
#include <iostream>

using namespace std;

// V1: Naive CUDA 双边滤波 Kernel
__global__ void bilateral_filter_kernel_v1(
    const uint8_t* input,   
    uint8_t* output,        
    int width,
    int height,
    int channels,
    int radius,
    float spatial_denom,    
    float color_denom      
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
            // 圆形窗口：跳过角落像素，与OpenCV一致
            if (dx * dx + dy * dy > radius * radius) {
                continue;
            }
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

            // 颜色差异: L1范数，与OpenCV一致
            float color_diff_l1 = 0.0f;
            for (int ch = 0; ch < channels; ch++) {
                float diff = p_color[ch] - q_color[ch];
                color_diff_l1 += fabsf(diff);
            }

            // 高斯权重
            float w_spatial = expf(-spatial_dist_sq / spatial_denom);
            float w_color = expf(-color_diff_l1 * color_diff_l1 / color_denom);
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

// V2: Shared Memory 双边滤波 Kernel

// block 大小常量（与V1一致）
#define BLOCK_SIZE 16

__global__ void bilateral_filter_kernel_v2(
    const uint8_t* input,
    uint8_t* output,
    int width,
    int height,
    int channels,
    int radius,
    float spatial_denom,
    float color_denom
) {
    // step 1：计算当前线程负责的输出像素坐标
    int x = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int y = blockIdx.y * BLOCK_SIZE + threadIdx.y;

    // tile尺寸 = block尺寸 + 两侧halo
    int tile_w = BLOCK_SIZE + 2 * radius;
    int tile_h = BLOCK_SIZE + 2 * radius;

    // step 2：声明 shared memory 
    extern __shared__ uint8_t smem[];

    // step 3：协作加载tile ——> shared memory
    int tid = threadIdx.y * BLOCK_SIZE + threadIdx.x;
    int num_threads = BLOCK_SIZE * BLOCK_SIZE;
    int tile_size = tile_w * tile_h * channels;

    // tile左上角在全局图像中的坐标（可能为负，表示超出边界）
    int tile_origin_x = blockIdx.x * BLOCK_SIZE - radius;
    int tile_origin_y = blockIdx.y * BLOCK_SIZE - radius;

    for (int i = tid; i < tile_size; i += num_threads) {
        // 从一维索引 i 反算出 tile 内的 (ty, tx, ch)
        int ch = i % channels;
        int tx = (i / channels) % tile_w;
        int ty = (i / channels) / tile_w;

        // 对应的全局坐标
        int gx = tile_origin_x + tx;
        int gy = tile_origin_y + ty;

        // clamp 边界处理：超出图像范围时钳制到边界像素
        gx = min(max(gx, 0), width - 1);
        gy = min(max(gy, 0), height - 1);

        // 从全局内存读取并写入 shared memory
        smem[i] = input[(gy * width + gx) * channels + ch];
    }

    // 同步: 确保 tile 全部加载完毕
    __syncthreads();

    // step 4：边界保护, 多余线程不参与计算）
    if (x >= width || y >= height) {
        return;
    }

    // step 5：从 shared memory 读取并执行双边滤波
    int sx = threadIdx.x + radius;  
    int sy = threadIdx.y + radius;  

    // 读取中心像素颜色
    float p_color[3];
    for (int ch = 0; ch < channels; ch++) {
        p_color[ch] = (float)smem[(sy * tile_w + sx) * channels + ch];
    }

    // 累加器
    float sum_value[3] = {0.0f, 0.0f, 0.0f};
    float sum_weight = 0.0f;

    // 遍历邻域窗口
    for (int dy = -radius; dy <= radius; dy++) {
        for (int dx = -radius; dx <= radius; dx++) {
            // 圆形窗口：跳过角落像素，与OpenCV一致
            if (dx * dx + dy * dy > radius *  radius) {
                continue;
            }
            
            // 边界检查：跳过越界像素
            int gqx = x + dx;
            int gqy = y + dy;
            if (gqx < 0 || gqx >= width || gqy < 0 || gqy > height) {
                continue;
            }
            
            // 在 shared memory 中的坐标
            int qsx = sx + dx;
            int qsy = sy + dy;

            // 从 shared memory 读取邻域像素颜色
            float q_color[3];
            for (int ch = 0; ch < channels; ch++) {
                q_color[ch] = (float)smem[(qsy * tile_w + qsx) * channels + ch];
            }

            // 以下计算逻辑与 V1 完全一致
            float spatial_dist_sq = (float)(dx * dx + dy * dy);

            // 颜色差异：L1范数，与OpenCV一致
            float color_diff_l1 = 0.0f;
            for (int ch = 0; ch < channels; ch++) {
                float diff = p_color[ch] - q_color[ch];
                color_diff_l1 += fabsf(diff);
            }

            float w_spatial = expf(-spatial_dist_sq / spatial_denom);
            float w_color = expf(-color_diff_l1 * color_diff_l1 / color_denom);
            float weight = w_spatial * w_color;

            sum_weight += weight;
            for (int ch = 0; ch < channels; ch++) {
                sum_value[ch] += weight * q_color[ch];
            }
        }
    }

    // 归一化并写回全局内存
    int out_idx = (y * width + x) * channels;
    for (int ch = 0; ch < channels; ch++) {
        float val = sum_value[ch] / sum_weight;
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

    // step 1: 在GPU上分配内存
    uint8_t* d_input = nullptr;
    uint8_t* d_output = nullptr;
    RUNTIME_CHECK(cudaMalloc(&d_input, img_size));
    RUNTIME_CHECK(cudaMalloc(&d_output, img_size));

    // step 2: 从CPU ——> GPU
    RUNTIME_CHECK(cudaMemcpy(d_input, input.data.data(), img_size,
                             cudaMemcpyHostToDevice));

    // step 3: 配置kernel启动参数
    dim3 block(16, 16);
    dim3 grid((w + block.x - 1) / block.x,
              (h + block.y - 1) / block.y);

    // 精确测量kernel执行时间
    cudaEvent_t start, stop;
    RUNTIME_CHECK(cudaEventCreate(&start));
    RUNTIME_CHECK(cudaEventCreate(&stop));
    RUNTIME_CHECK(cudaEventRecord(start));
    
    // 启动kernel
    bilateral_filter_kernel_v1<<<grid, block>>>(
        d_input, d_output,
        w, h, c,
        params.radius,
        spatial_denom, color_denom
    );

    RUNTIME_CHECK(cudaEventRecord(stop));
    RUNTIME_CHECK(cudaEventSynchronize(stop));

    // 打印kernel纯计算时间
    float kernel_ms = 0.0f;
    RUNTIME_CHECK(cudaEventElapsedTime(&kernel_ms, start, stop));
    cout << "  Kernel时间:    " << kernel_ms << " ms" << endl;

    RUNTIME_CHECK(cudaEventDestroy(start));
    RUNTIME_CHECK(cudaEventDestroy(stop));

    // 检查kernel是否启动成功
    RUNTIME_CHECK(cudaGetLastError());

    // step 4: 把结果从GPU ——> CPU 
    Image output;
    output.width = w;
    output.height = h;
    output.channels = c;
    output.data.resize(w * h * c);
    RUNTIME_CHECK(cudaMemcpy(output.data.data(), d_output, img_size,
                             cudaMemcpyDeviceToHost));

    // step 5: 释放GPU内存 
    RUNTIME_CHECK(cudaFree(d_input));
    RUNTIME_CHECK(cudaFree(d_output));

    return output;
}

// V2 包装函数：与V1结构相同，增加shared memory大小计算
Image bilateral_filter_gpu_v2(const Image& input, const FilterParams& params) {
    int w = input.width;
    int h = input.height;
    int c = input.channels;
    size_t img_size = (size_t)w * h * c * sizeof(uint8_t);

    float spatial_denom = 2.0f * params.sigma_spatial * params.sigma_spatial;
    float color_denom = 2.0f * params.sigma_color * params.sigma_color;

    // 分配device内存
    uint8_t* d_input = nullptr;
    uint8_t* d_output = nullptr;
    RUNTIME_CHECK(cudaMalloc(&d_input, img_size));
    RUNTIME_CHECK(cudaMalloc(&d_output, img_size));

    RUNTIME_CHECK(cudaMemcpy(d_input, input.data.data(), img_size,
                             cudaMemcpyHostToDevice));

    // 配置 kernel 启动参数
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((w + BLOCK_SIZE - 1) / BLOCK_SIZE,
              (h + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // 计算 shared memory 大小
    // tile大小 = (BLOCK_SIZE + 2*radius)^2，每个元素 channels 字节
    int tile_w = BLOCK_SIZE + 2 * params.radius;
    int tile_h = BLOCK_SIZE + 2 * params.radius;
    size_t smem_size = (size_t)tile_w * tile_h * c * sizeof(uint8_t);
    cout << "  Shared Memory: " << smem_size << " bytes/block ("
         << tile_w << "x" << tile_h << "x" << c << ")" << endl;

    // CUDA Event 计时
    cudaEvent_t start, stop;
    RUNTIME_CHECK(cudaEventCreate(&start));
    RUNTIME_CHECK(cudaEventCreate(&stop));

    RUNTIME_CHECK(cudaEventRecord(start));

    // 启动 V2 kernel
    bilateral_filter_kernel_v2<<<grid, block, smem_size>>>(
        d_input, d_output,
        w, h, c,
        params.radius,
        spatial_denom, color_denom
    );

    RUNTIME_CHECK(cudaEventRecord(stop));
    RUNTIME_CHECK(cudaEventSynchronize(stop));

    float kernel_ms = 0.0f;
    RUNTIME_CHECK(cudaEventElapsedTime(&kernel_ms, start, stop));
    cout << "  Kernel 时间:   " << kernel_ms << " ms" << endl;

    RUNTIME_CHECK(cudaEventDestroy(start));
    RUNTIME_CHECK(cudaEventDestroy(stop));

    RUNTIME_CHECK(cudaGetLastError());

    // 拷回结果
    Image output;
    output.width = w;
    output.height = h;
    output.channels = c;
    output.data.resize(w * h * c);
    RUNTIME_CHECK(cudaMemcpy(output.data.data(), d_output, img_size,
                             cudaMemcpyDeviceToHost));

    RUNTIME_CHECK(cudaFree(d_input));
    RUNTIME_CHECK(cudaFree(d_output));

    return output;
}
