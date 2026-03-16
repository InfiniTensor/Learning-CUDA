#include "bilateral_filter.cuh"
#include "utils.cuh"

#include <cmath>
#include <cstring>
#include <iostream>

using namespace std;

// block 大小常量
#define BLOCK_SIZE 16

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


// V2: Shared Memory 双边滤波 Kernel
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
            if (gqx < 0 || gqx >= width || gqy < 0 || gqy >= height) {
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


// V3: 常量内存LUT + __expf + #pragma unroll
#define MAX_RADIUS 16
#define MAX_DIAMETER (2 * MAX_RADIUS + 1)
__constant__ float d_spatial_LUT[MAX_DIAMETER * MAX_DIAMETER];  // 只读常量内存

__global__ void bilateral_filter_kernel_v3(
    const uint8_t* input,
    uint8_t* output,
    int width,
    int height,
    int channels,
    int radius,
    float color_denom , // 预计算颜色分母，空间分母已在LUT中
    int y_offset  // strip 起始行偏移(V4 strea用， V3传0)
) {
    // step 1: 计算输出像素坐标
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y + y_offset;

    // tile尺寸 = block尺寸 + 两侧halo
    int tile_w = blockDim.x + 2 * radius;
    int tile_h = blockDim.y + 2 * radius;

    // step 2: shared memory协作加载
    extern __shared__ uint8_t smem[];

    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int num_threads = blockDim.x * blockDim.y;
    int tile_size = tile_w * tile_h * channels;

    int tile_origin_x = blockIdx.x * blockDim.x - radius;
    int tile_origin_y = blockIdx.y * blockDim.y + y_offset - radius;

    for (int i = tid; i < tile_size; i += num_threads) {
        int ch = i % channels;
        int tx = (i / channels) % tile_w;
        int ty = (i / channels) / tile_w;

        int gx = tile_origin_x + tx;
        int gy = tile_origin_y + ty;

        gx = min(max(gx, 0), width - 1);
        gy = min(max(gy, 0), height - 1);

        smem[i] = input[(gy * width + gx) * channels + ch];
    }

    __syncthreads();

    // step 3: 边界保护
    if (x >= width || y >= height) {
        return;
    }

    // step 4:  从shared memory读中心像素
    int sx = threadIdx.x + radius;
    int sy = threadIdx.y + radius;

    float p_color[3];
    for (int ch = 0; ch < channels; ch++) {
        p_color[ch] = (float)smem[(sy * tile_w + sx) * channels + ch];
    }

    // step 5: 邻域遍历（V3 优化版）----
    float sum_value[3] = {0.0f, 0.0f, 0.0f};
    float sum_weight = 0.0f;

    int diameter = 2 * radius + 1;

    for (int dy = -radius; dy <= radius; dy++) {
        for (int dx = -radius; dx <= radius; dx++) {
            // 【优化1】从常量内存查表获取空间权重
            // LUT 中圆形窗口外的值为 -1，直接跳过
            float w_spatial = d_spatial_LUT[(dy + radius) * diameter + (dx + radius)];
            if (w_spatial <= 0.0f) {
                continue;  // 圆形窗口外，跳过
            }

            // 边界检查
            int gqx = x + dx;
            int gqy = y + dy;
            if (gqx < 0 || gqx >= width || gqy < 0 || gqy >= height) {
                continue;
            }

            // 从 shared memory 读邻域像素
            int qsx = sx + dx;
            int qsy = sy + dy;

            float q_color[3];
            #pragma unroll 3
            for (int ch = 0; ch < 3; ch++) {
                q_color[ch] = (float)smem[(qsy * tile_w + qsx) * channels + ch];
            }

            // 【优化3】通道循环展开计算 L1 颜色差异
            float color_diff_l1 = 0.0f;
            #pragma unroll 3
            for (int ch = 0; ch < 3; ch++) {
                color_diff_l1 += fabsf(p_color[ch] - q_color[ch]);
            }

            // 【优化2】__expf 快速数学计算颜色权重
            float w_color = __expf(-color_diff_l1 * color_diff_l1 / color_denom);
            float weight = w_spatial * w_color;

            // 累加
            sum_weight += weight;
            #pragma unroll 3
            for (int ch = 0; ch < 3; ch++) {
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

// V3 初始化函数：预计算空间权重LUT并上传到常量内存
// 只需在参数变化时调用一次，不计入每帧耗时
void bilateral_filter_gpu_v3_init(const FilterParams& params) {
    int r = params.radius;
    int diameter = 2 * r + 1;

    // CPU 端预计算空间权重 LUT
    float spatial_denom = 2.0f * params.sigma_spatial * params.sigma_spatial;
    float h_spatial_LUT[MAX_DIAMETER * MAX_DIAMETER];

    for (int dy = -r; dy <= r; dy++) {
        for (int dx = -r; dx <= r; dx++) {
            int idx = (dy + r) * diameter + (dx + r);
            int dist_sq = dx * dx + dy * dy;

            if (dist_sq > r * r) {
                h_spatial_LUT[idx] = -1.0f;
            } else {
                h_spatial_LUT[idx] = expf(-(float)dist_sq / spatial_denom);
            }
        }
    }

    // 写入 GPU 常量内存
    size_t lut_bytes = (size_t)diameter * diameter * sizeof(float);
    RUNTIME_CHECK(cudaMemcpyToSymbol(d_spatial_LUT, h_spatial_LUT, lut_bytes));
}

// V3 包装函数，纯滤波，LUT已由init函数上传
Image bilateral_filter_gpu_v3(const Image& input, const FilterParams& params) {
    int w = input.width;
    int h = input.height;
    int c = input.channels;
    size_t img_size = (size_t)w * h * c * sizeof(uint8_t);
    int r = params.radius;

    float color_denom = 2.0f * params.sigma_color * params.sigma_color;

    //分配device内存
    uint8_t* d_input = nullptr;
    uint8_t* d_output = nullptr;
    RUNTIME_CHECK(cudaMalloc(&d_input, img_size));
    RUNTIME_CHECK(cudaMalloc(&d_output, img_size));

    RUNTIME_CHECK(cudaMemcpy(d_input, input.data.data(), img_size,
                             cudaMemcpyHostToDevice));

    // 配置kernel启动参数
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((w + BLOCK_SIZE - 1) / BLOCK_SIZE,
              (h + BLOCK_SIZE - 1) / BLOCK_SIZE);

    int tile_w = BLOCK_SIZE + 2 * r;
    int tile_h = BLOCK_SIZE + 2 * r;
    size_t smem_size = (size_t)tile_w * tile_h * c * sizeof(uint8_t);
    int diameter = 2 * r + 1;
    size_t lut_bytes = (size_t)diameter * diameter * sizeof(float);
    cout << "  LUT 大小:      " << diameter << "x" << diameter
         << " = " << lut_bytes << " bytes" << endl;
    cout << "  Shared Memory: " << smem_size << " bytes/block ("
         << tile_w << "x" << tile_h << "x" << c << ")" << endl;

    // CUDA Event 计时
    cudaEvent_t start, stop;
    RUNTIME_CHECK(cudaEventCreate(&start));
    RUNTIME_CHECK(cudaEventCreate(&stop));

    RUNTIME_CHECK(cudaEventRecord(start));

    // 启动 V3 kernel,不再传 spatial_denom
    bilateral_filter_kernel_v3<<<grid, block, smem_size>>>(
        d_input, d_output,
        w, h, c,
        r,
        color_denom,
        0  // y_offset = 0 (V3处理全图)
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


// V4: Pinned Memory + CUDA Streams流水线
static uint8_t* d_v4_input  = nullptr;  // 预分配的输入 buffer(device)
static uint8_t* d_v4_output = nullptr;  // 预分配的输出 buffer(device)
static uint8_t* h_v4_pinned_in = nullptr;  // pinned host 输入缓冲
static uint8_t* h_v4_pinned_out =  nullptr;  // pinned host 输出缓冲
static size_t   d_v4_buf_size = 0;      // 当前 buffer 大小（字节）

#define V4_BLOCK_SIZE 32  // 可独立调优
#define V4_NUM_STREAMS 4  // CUDA Streams：多路流水线
static cudaStream_t v4_streams[V4_NUM_STREAMS];


// V4初始化：上传LUT + 预分配 device/pinned buffer + 创建streams
void bilateral_filter_gpu_v4_init(const Image& input, const FilterParams& params) {
    // 1. 上传空间权重LUT复用V3
    bilateral_filter_gpu_v3_init(params);

    // 2. 预分配 device buffer
    size_t img_size = (size_t)input.width * input.height * input.channels * sizeof(uint8_t);

    // 如果已经分配过且大小足够，跳过
    if (d_v4_input != nullptr && d_v4_buf_size >= img_size) {
        cout << "  Buffer 复用:   " << d_v4_buf_size << " bytes (已分配)" << endl;
        return;
    }

    // 释放旧buffer
    if (d_v4_input != nullptr) {
        RUNTIME_CHECK(cudaFree(d_v4_input));
        RUNTIME_CHECK(cudaFree(d_v4_output));
    }

    // 分配新buffer(device + pinned host)
    RUNTIME_CHECK(cudaMalloc(&d_v4_input, img_size));
    RUNTIME_CHECK(cudaMalloc(&d_v4_output, img_size));
    RUNTIME_CHECK(cudaHostAlloc(&h_v4_pinned_in, img_size, cudaHostAllocDefault));
    RUNTIME_CHECK(cudaHostAlloc(&h_v4_pinned_out, img_size, cudaHostAllocDefault));
    d_v4_buf_size = img_size;

    cout << "  Buffer 预分配: " << img_size << " bytes x 2 (device)" << endl;
    cout << "  Pinned Memory: " << img_size << " bytes x 2 (host)" << endl;

    // 3. 创建CUDA Streams
    for (int i = 0; i < V4_NUM_STREAMS; i++) {
        RUNTIME_CHECK(cudaStreamCreate(&v4_streams[i]));
    }
    cout << "  CUDA Streams: " << V4_NUM_STREAMS << " 路流水线"  << endl;
}


// V4清理：释放device/pinned buffer + 销毁streams

void bilateral_filter_gpu_v4_cleanup() {
    if (d_v4_input != nullptr) {
        // 销毁 streams
        for (int i =0; i < V4_NUM_STREAMS; i++) {
            RUNTIME_CHECK(cudaStreamDestroy(v4_streams[i]));
        }
        RUNTIME_CHECK(cudaFree(d_v4_input));
        RUNTIME_CHECK(cudaFree(d_v4_output));
        RUNTIME_CHECK(cudaFreeHost(h_v4_pinned_in));
        RUNTIME_CHECK(cudaFreeHost(h_v4_pinned_out));
        d_v4_input = nullptr;
        d_v4_output = nullptr;
        h_v4_pinned_in = nullptr;
        h_v4_pinned_out = nullptr;
        d_v4_buf_size = 0;
    }
}

// V4 每帧滤波：stream流水线
Image bilateral_filter_gpu_v4(const Image& input, const FilterParams& params) {
    int w = input.width;
    int h = input.height;
    int c = input.channels;
    size_t img_size = (size_t)w * h * c * sizeof(uint8_t);
    int r = params.radius;

    float color_denom = 2.0f * params.sigma_color * params.sigma_color;

    // 1. CPU -> pinned buffer
    memcpy(h_v4_pinned_in, input.data.data(), img_size);
    
    // 2. H2D: pinned -> devive 异步
    RUNTIME_CHECK(cudaMemcpyAsync(d_v4_input, h_v4_pinned_in, img_size,
                             cudaMemcpyHostToDevice, v4_streams[0]));

    // H2D 完成事件：其他stream需要等待H2D完成才能启动kernel
    cudaEvent_t h2d_done;
    RUNTIME_CHECK(cudaEventCreate(&h2d_done));
    RUNTIME_CHECK(cudaEventRecord(h2d_done, v4_streams[0]));

    // 3. kernel配置
    dim3 block(V4_BLOCK_SIZE, V4_BLOCK_SIZE);
    int tile_w = V4_BLOCK_SIZE + 2 * r;
    int tile_h = V4_BLOCK_SIZE + 2 * r;
    size_t smem_size = (size_t)tile_w * tile_h * c * sizeof(uint8_t);
    int diameter = 2 * r + 1;
    size_t lut_bytes = (size_t)diameter * diameter * sizeof(float);
    cout << "  LUT 大小:      " << diameter << "x" << diameter
         << " = " << lut_bytes << " bytes" << endl;
    cout << "  Shared Memory: " << smem_size << " bytes/block ("
         << tile_w << "x" << tile_h << "x" << c << ")" << endl;

    // 4. 分strip启动kernel + D2H
    // 计算每个strip的高度，对齐到V4_BLOCK_SIZE
    int strip_h = (h + V4_NUM_STREAMS - 1) / V4_NUM_STREAMS;
    strip_h = ((strip_h + V4_BLOCK_SIZE -  1) / V4_BLOCK_SIZE) * V4_BLOCK_SIZE;

    cout << "  Strip 高度:     " << strip_h << " row x " << V4_NUM_STREAMS
         << " strips" << endl;

    // Kernel计时：从第一个kernel开始到最后一个D2H完成
    cudaEvent_t start, stop;
    RUNTIME_CHECK(cudaEventCreate(&start));
    RUNTIME_CHECK(cudaEventCreate(&stop));

    // start记录在stream[0] (H2D完成后)
    RUNTIME_CHECK(cudaStreamWaitEvent(v4_streams[0], h2d_done, 0));
    RUNTIME_CHECK(cudaEventRecord(start, v4_streams[0]));

    int last_stream = 0;
    for (int s = 0; s < V4_NUM_STREAMS; s++) {
        int y_start = s * strip_h;
        if (y_start >= h) break;
        int y_end = min(y_start + strip_h, h);
        int actual_h = y_end - y_start;

        cudaStream_t stream =v4_streams[s];
        // 确保 H2D 完成后再启动 kernel
        RUNTIME_CHECK(cudaStreamWaitEvent(stream, h2d_done, 0));

        // 启动 kernel：只处理本 strip 的行
        dim3 grid_s((w + V4_BLOCK_SIZE - 1) / V4_BLOCK_SIZE,
                    (actual_h + V4_BLOCK_SIZE - 1) / V4_BLOCK_SIZE);

        bilateral_filter_kernel_v3<<<grid_s, block, smem_size, stream>>>(
            d_v4_input, d_v4_output,
            w, h, c, r, color_denom,
            y_start     // y_offset: 本 strip 的起始行
        );

        // D2H：只拷回本 strip 的输出行（同一 stream，自动排在 kernel 之后）
        size_t strip_offset = (size_t)y_start * w * c;
        size_t strip_bytes = (size_t)actual_h * w * c;
        RUNTIME_CHECK(cudaMemcpyAsync(
            h_v4_pinned_out + strip_offset,
            d_v4_output + strip_offset,
            strip_bytes,
            cudaMemcpyDeviceToHost,
            stream
        ));

        last_stream = s;
    }

    // stop 记录在最后一个活跃的 stream
    RUNTIME_CHECK(cudaEventRecord(stop, v4_streams[last_stream]));

    // 5. 等待所有 stream 完成
    RUNTIME_CHECK(cudaDeviceSynchronize());

    float pipeline_ms = 0.0f;
    RUNTIME_CHECK(cudaEventElapsedTime(&pipeline_ms, start, stop));
    cout << "  Pipeline 时间: " << pipeline_ms << " ms (kernel + D2H 重叠)" << endl;

    RUNTIME_CHECK(cudaEventDestroy(start));
    RUNTIME_CHECK(cudaEventDestroy(stop));
    RUNTIME_CHECK(cudaEventDestroy(h2d_done));
    RUNTIME_CHECK(cudaGetLastError());

    // 6. pinned -> output
    Image output;
    output.width = w;
    output.height = h;
    output.channels = c;
    output.data.resize(w * h * c);
    memcpy(output.data.data(), h_v4_pinned_out, img_size);

    return output;
}


