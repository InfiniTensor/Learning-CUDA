// V6: Moore Threads S5000 (MUSA) 平台适配

#include "bilateral_filter.cuh"

#include <musa_runtime.h>
#include <cmath>
#include <cstring>
#include <iostream>

using namespace std;

// MUSA 运行时错误检查宏

#define RUNTIME_CHECK(call)                                                    \
  do {                                                                         \
    musaError_t err = call;                                                    \
    if (err != musaSuccess) {                                                  \
      std::cerr << "Runtime error at " << __FILE__ << ":" << __LINE__ << " - " \
                << musaGetErrorString(err) << "\n";                            \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

// 常量定义
#define V6_BLOCK_SIZE 32
#define V6_NUM_STREAMS 4

#define MAX_RADIUS 16
#define MAX_DIAMETER (2 * MAX_RADIUS + 1)

// GPU 常量内存：存放预计算的空间高斯权重
__constant__ float d_spatial_LUT[MAX_DIAMETER * MAX_DIAMETER];

// V6 Kernel：常量内存 LUT + Shared Memory + 循环展开
__global__ void bilateral_filter_kernel_v6(
    const uint8_t* input,
    uint8_t* output,
    int width,
    int height,
    int channels,
    int radius,
    float color_denom,      // 预计算: 2 * sigma_color²（空间分母已在 LUT 中）
    int y_offset            // strip 起始行偏移
) {
    // step 1：计算输出像素坐标 
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y + y_offset;

    // tile 尺寸 = block 尺寸 + 两侧 halo
    int tile_w = blockDim.x + 2 * radius;
    int tile_h = blockDim.y + 2 * radius;

    // step 2：shared memory 协作加载 
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

    // step 3：边界保护
    if (x >= width || y >= height) {
        return;
    }

    // step 4：从 shared memory 读中心像素
    int sx = threadIdx.x + radius;
    int sy = threadIdx.y + radius;

    float p_color[3];
    for (int ch = 0; ch < channels; ch++) {
        p_color[ch] = (float)smem[(sy * tile_w + sx) * channels + ch];
    }

    // step 5：邻域遍历 
    float sum_value[3] = {0.0f, 0.0f, 0.0f};
    float sum_weight = 0.0f;

    int diameter = 2 * radius + 1;

    for (int dy = -radius; dy <= radius; dy++) {
        for (int dx = -radius; dx <= radius; dx++) {
            // 从常量内存查表获取空间权重
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

            // 通道循环展开计算 L1 颜色差异
            float color_diff_l1 = 0.0f;
            #pragma unroll 3
            for (int ch = 0; ch < 3; ch++) {
                color_diff_l1 += fabsf(p_color[ch] - q_color[ch]);
            }

            // 使用标准 expf（MUSA 可能不支持 __expf device intrinsic）
            float w_color = expf(-color_diff_l1 * color_diff_l1 / color_denom);
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

// 文件作用域静态变量：预分配的 device buffer + streams
static uint8_t* d_v6_input  = nullptr;
static uint8_t* d_v6_output = nullptr;
static uint8_t* h_v6_pinned_in  = nullptr;
static uint8_t* h_v6_pinned_out = nullptr;
static size_t   d_v6_buf_size = 0;

static musaStream_t v6_streams[V6_NUM_STREAMS];

// V6 初始化：上传 LUT + 预分配 device/pinned buffer + 创建 streams
void bilateral_filter_gpu_v6_init(const Image& input, const FilterParams& params) {
    // ---- 1. 预计算空间权重 LUT 并上传到常量内存 ----
    int r = params.radius;
    int diameter = 2 * r + 1;
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

    size_t lut_bytes = (size_t)diameter * diameter * sizeof(float);
    RUNTIME_CHECK(musaMemcpyToSymbol(d_spatial_LUT, h_spatial_LUT, lut_bytes));

    // ---- 2. 预分配 device buffer ----
    size_t img_size = (size_t)input.width * input.height * input.channels * sizeof(uint8_t);

    if (d_v6_input != nullptr && d_v6_buf_size >= img_size) {
        cout << "  Buffer 复用:   " << d_v6_buf_size << " bytes (已分配)" << endl;
        return;
    }

    // 释放旧 buffer
    if (d_v6_input != nullptr) {
        RUNTIME_CHECK(musaFree(d_v6_input));
        RUNTIME_CHECK(musaFree(d_v6_output));
        RUNTIME_CHECK(musaFreeHost(h_v6_pinned_in));
        RUNTIME_CHECK(musaFreeHost(h_v6_pinned_out));
    }

    // 分配新 buffer（device + pinned host）
    RUNTIME_CHECK(musaMalloc(&d_v6_input, img_size));
    RUNTIME_CHECK(musaMalloc(&d_v6_output, img_size));
    RUNTIME_CHECK(musaMallocHost((void**)&h_v6_pinned_in, img_size));
    RUNTIME_CHECK(musaMallocHost((void**)&h_v6_pinned_out, img_size));
    d_v6_buf_size = img_size;

    cout << "  Buffer 预分配: " << img_size << " bytes x 2 (device)"  << endl;
    cout << "  Pinned Memory: " << img_size << " bytes x 2 (host)" << endl;

    // 3. 创建 Streams 
    for (int i = 0; i < V6_NUM_STREAMS; i++) {
        RUNTIME_CHECK(musaStreamCreate(&v6_streams[i]));
    }
    cout << "  Streams:       " << V6_NUM_STREAMS << " 路流水线" << endl;
}

// V6 清理：释放 device/pinned buffer + 销毁 streams
void bilateral_filter_gpu_v6_cleanup() {
    if (d_v6_input != nullptr) {
        for (int i = 0; i < V6_NUM_STREAMS; i++) {
            RUNTIME_CHECK(musaStreamDestroy(v6_streams[i]));
        }
        RUNTIME_CHECK(musaFree(d_v6_input));
        RUNTIME_CHECK(musaFree(d_v6_output));
        RUNTIME_CHECK(musaFreeHost(h_v6_pinned_in));
        RUNTIME_CHECK(musaFreeHost(h_v6_pinned_out));
        d_v6_input = nullptr;
        d_v6_output = nullptr;
        h_v6_pinned_in = nullptr;
        h_v6_pinned_out = nullptr;
        d_v6_buf_size = 0;
    }
}


// V6 每帧滤波：stream 流水线
Image bilateral_filter_gpu_v6(const Image& input, const FilterParams& params) {
    int w = input.width;
    int h = input.height;
    int c = input.channels;
    size_t img_size = (size_t)w * h * c * sizeof(uint8_t);
    int r = params.radius;

    float color_denom = 2.0f * params.sigma_color * params.sigma_color;

    // 1. CPU → pinned buffer
    memcpy(h_v6_pinned_in, input.data.data(), img_size);

    // 2. H2D: pinned → device（异步，stream[0]）
    RUNTIME_CHECK(musaMemcpyAsync(d_v6_input, h_v6_pinned_in, img_size,
                                   musaMemcpyHostToDevice, v6_streams[0]));

    // H2D 完成事件：其他 stream 需要等待 H2D 完成才能启动 kernel
    musaEvent_t h2d_done;
    RUNTIME_CHECK(musaEventCreate(&h2d_done));
    RUNTIME_CHECK(musaEventRecord(h2d_done, v6_streams[0]));

    // 3. kernel 配置 
    dim3 block(V6_BLOCK_SIZE, V6_BLOCK_SIZE);
    int tile_w = V6_BLOCK_SIZE + 2 * r;
    int tile_h = V6_BLOCK_SIZE + 2 * r;
    size_t smem_size = (size_t)tile_w * tile_h * c * sizeof(uint8_t);
    int diameter = 2 * r + 1;
    size_t lut_bytes = (size_t)diameter * diameter * sizeof(float);
    cout << "  LUT 大小:      " << diameter << "x" << diameter
         << " = " << lut_bytes << " bytes" << endl;
    cout << "  Shared Memory: " << smem_size << " bytes/block ("
         << tile_w << "x" << tile_h << "x" << c << ")" << endl;

    // 4. 分 strip 启动 kernel + D2H 
    int strip_h = (h + V6_NUM_STREAMS - 1) / V6_NUM_STREAMS;
    strip_h = ((strip_h + V6_BLOCK_SIZE - 1) / V6_BLOCK_SIZE) * V6_BLOCK_SIZE;

    cout << "  Strip 高度:    " << strip_h << " rows x " << V6_NUM_STREAMS
         << " strips" << endl;

    // Kernel 计时：从第一个 kernel 开始到最后一个 D2H 完成
    musaEvent_t start, stop;
    RUNTIME_CHECK(musaEventCreate(&start));
    RUNTIME_CHECK(musaEventCreate(&stop));

    RUNTIME_CHECK(musaStreamWaitEvent(v6_streams[0], h2d_done));
    RUNTIME_CHECK(musaEventRecord(start, v6_streams[0]));

    int last_stream = 0;
    for (int s = 0; s < V6_NUM_STREAMS; s++) {
        int y_start = s * strip_h;
        if (y_start >= h) break;
        int y_end = min(y_start + strip_h, h);
        int actual_h = y_end - y_start;

        musaStream_t stream = v6_streams[s];

        // 确保 H2D 完成后再启动 kernel
        RUNTIME_CHECK(musaStreamWaitEvent(stream, h2d_done));

        // 启动 kernel：只处理本 strip 的行
        dim3 grid_s((w + V6_BLOCK_SIZE - 1) / V6_BLOCK_SIZE,
                    (actual_h + V6_BLOCK_SIZE - 1) / V6_BLOCK_SIZE);

        bilateral_filter_kernel_v6<<<grid_s, block, smem_size, stream>>>(
            d_v6_input, d_v6_output,
            w, h, c, r, color_denom,
            y_start     // y_offset: 本 strip 的起始行
        );

        // D2H：只拷回本 strip 的输出行
        size_t strip_offset = (size_t)y_start * w * c;
        size_t strip_bytes = (size_t)actual_h * w * c;
        RUNTIME_CHECK(musaMemcpyAsync(
            h_v6_pinned_out + strip_offset,
            d_v6_output + strip_offset,
            strip_bytes,
            musaMemcpyDeviceToHost,
            stream
        ));

        last_stream = s;
    }

    // stop 记录在最后一个活跃的 stream
    RUNTIME_CHECK(musaEventRecord(stop, v6_streams[last_stream]));

    // 5. 等待所有 stream 完成 
    RUNTIME_CHECK(musaDeviceSynchronize());

    float pipeline_ms = 0.0f;
    RUNTIME_CHECK(musaEventElapsedTime(&pipeline_ms, start, stop));
    cout << "  Pipeline 时间: " << pipeline_ms << " ms (kernel + D2H 重叠)" << endl;

    RUNTIME_CHECK(musaEventDestroy(start));
    RUNTIME_CHECK(musaEventDestroy(stop));
    RUNTIME_CHECK(musaEventDestroy(h2d_done));
    RUNTIME_CHECK(musaGetLastError());

    // 6. pinned → output 
    Image output;
    output.width = w;
    output.height = h;
    output.channels = c;
    output.data.resize(w * h * c);
    memcpy(output.data.data(), h_v6_pinned_out, img_size);

    return output;
}
