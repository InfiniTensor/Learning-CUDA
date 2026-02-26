// Standard library
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

// CUDA FP16 (half-precision) support
#include <cuda_fp16.h>

// Project headers (includes <cuda_runtime.h> via bilateral_filter_cuda.cuh)
#include "bilateral_filter_cuda.cuh"

// clang-format off
#define CUDA_CHECK(call)                                                \
    do {                                                                \
        cudaError_t err = call;                                         \
        if (err != cudaSuccess) {                                       \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                \
                    __FILE__, __LINE__, cudaGetErrorString(err));       \
            exit(EXIT_FAILURE);                                         \
        }                                                               \
    } while (0)
// clang-format on

#ifndef BLOCK_X
#define BLOCK_X 16 // Opt4: 16x16 = 256 threads; better smem cache vs 32x8
#endif
#ifndef BLOCK_Y
#define BLOCK_Y 16
#endif
// Opt A: launch bounds hints for register allocation
#define THREADS_PER_BLOCK (BLOCK_X * BLOCK_Y)
#define MIN_BLOCKS_PER_SM 4
#define MAX_RADIUS     16
#define LUT_SIZE       ((2 * MAX_RADIUS + 1) * (2 * MAX_RADIUS + 1))
#define COLOR_LUT_SIZE 256

__constant__ float d_spatial_lut[LUT_SIZE];
__constant__ float d_color_lut[COLOR_LUT_SIZE];

// ============================================================================
// Opt2: type-safe output helper for uint8/float kernel output
// ============================================================================

template <typename T>
__device__ inline T to_output(float v);
template <>
__device__ inline float to_output<float>(float v) {
    return v;
}
template <>
__device__ inline uint8_t to_output<uint8_t>(float v) {
    return static_cast<uint8_t>(fminf(255.0f, fmaxf(0.0f, v)));
}

// ============================================================================
// Template-based grayscale bilateral filter with compile-time radius
// ============================================================================

template <int RADIUS, typename InT = float, typename OutT = float>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_BLOCKS_PER_SM)
k_bilateral_filter_gray_template(const InT* __restrict__ input,
                                                 OutT* __restrict__ output, int width, int height) {

    constexpr int TILE_W = BLOCK_X + 2 * RADIUS;
    constexpr int TILE_H = BLOCK_Y + 2 * RADIUS;
    constexpr int TILE_SIZE = TILE_W * TILE_H;
    constexpr int LUT_WIDTH = 2 * RADIUS + 1;

    __shared__ float smem[TILE_SIZE];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int x = blockIdx.x * BLOCK_X + tx;
    const int y = blockIdx.y * BLOCK_Y + ty;

    const int thread_id = ty * BLOCK_X + tx;
    const int threads_per_block = BLOCK_X * BLOCK_Y;

// Collaborative loading
#pragma unroll
    for (int i = thread_id; i < TILE_SIZE; i += threads_per_block) {
        int sy = i / TILE_W;
        int sx = i % TILE_W;
        int gx = blockIdx.x * BLOCK_X + sx - RADIUS;
        int gy = blockIdx.y * BLOCK_Y + sy - RADIUS;
        gx = max(0, min(width - 1, gx));
        gy = max(0, min(height - 1, gy));
        smem[i] = static_cast<float>(input[gy * width + gx]);
    }

    __syncthreads();

    if (x >= width || y >= height)
        return;

    const int lx = tx + RADIUS;
    const int ly = ty + RADIUS;
    const float center = smem[ly * TILE_W + lx];

    float sum = 0.0f;
    float weight_sum = 0.0f;

#pragma unroll
    for (int dy = -RADIUS; dy <= RADIUS; ++dy) {
#pragma unroll
        for (int dx = -RADIUS; dx <= RADIUS; ++dx) {
            float spatial_weight = d_spatial_lut[(dy + RADIUS) * LUT_WIDTH + (dx + RADIUS)];
            if (spatial_weight == 0.0f) continue;  // Opt C/F: skip circular-window corners

            float neighbor = smem[(ly + dy) * TILE_W + (lx + dx)];

            int diff = static_cast<int>(fabsf(neighbor - center) + 0.5f);
            diff = min(diff, COLOR_LUT_SIZE - 1);
            float color_weight = d_color_lut[diff];

            float w = spatial_weight * color_weight;
            sum += neighbor * w;
            weight_sum += w;
        }
    }

    output[y * width + x] = to_output<OutT>(sum / weight_sum);
}

// ============================================================================
// Template-based RGB bilateral filter with compile-time radius
// Full loop unrolling for known radius values
// ============================================================================

template <int RADIUS, typename InT = float, typename OutT = float>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_BLOCKS_PER_SM)
k_bilateral_filter_rgb_template(const InT* __restrict__ input,
                                                OutT* __restrict__ output, int width, int height) {

    constexpr int TILE_W = BLOCK_X + 2 * RADIUS;
    constexpr int TILE_H = BLOCK_Y + 2 * RADIUS;
    constexpr int TILE_SIZE = TILE_W * TILE_H;
    constexpr int LUT_WIDTH = 2 * RADIUS + 1;

    __shared__ float smem_r[TILE_SIZE];
    __shared__ float smem_g[TILE_SIZE];
    __shared__ float smem_b[TILE_SIZE];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int x = blockIdx.x * BLOCK_X + tx;
    const int y = blockIdx.y * BLOCK_Y + ty;

    const int thread_id = ty * BLOCK_X + tx;
    const int threads_per_block = BLOCK_X * BLOCK_Y;

// Collaborative loading
#pragma unroll
    for (int i = thread_id; i < TILE_SIZE; i += threads_per_block) {
        int sy = i / TILE_W;
        int sx = i % TILE_W;
        int gx = blockIdx.x * BLOCK_X + sx - RADIUS;
        int gy = blockIdx.y * BLOCK_Y + sy - RADIUS;
        gx = max(0, min(width - 1, gx));
        gy = max(0, min(height - 1, gy));
        int gidx = (gy * width + gx) * 3;
        smem_r[i] = static_cast<float>(input[gidx]);
        smem_g[i] = static_cast<float>(input[gidx + 1]);
        smem_b[i] = static_cast<float>(input[gidx + 2]);
    }

    __syncthreads();

    if (x >= width || y >= height)
        return;

    const int lx = tx + RADIUS;
    const int ly = ty + RADIUS;
    const int lidx = ly * TILE_W + lx;

    const float center_r = smem_r[lidx];
    const float center_g = smem_g[lidx];
    const float center_b = smem_b[lidx];

    float sum_r = 0.0f, sum_g = 0.0f, sum_b = 0.0f;
    float wsum = 0.0f;

// Opt5: single shared color weight per neighbor, computed from mean absolute channel diff.
// Reduces 3 LUT lookups + 3 wsum accumulators to 1 each. Tradeoff: MAE rises ~0.65→0.80,
// which is acceptable (< 1.0). OpenCV actually uses Euclidean distance across channels,
// so this is a different (simpler) approximation.
#pragma unroll
    for (int dy = -RADIUS; dy <= RADIUS; ++dy) {
#pragma unroll
        for (int dx = -RADIUS; dx <= RADIUS; ++dx) {
            float spatial_weight = d_spatial_lut[(dy + RADIUS) * LUT_WIDTH + (dx + RADIUS)];
            if (spatial_weight == 0.0f) continue;  // Opt C/F: skip circular-window corners

            int nidx = (ly + dy) * TILE_W + (lx + dx);
            float nr = smem_r[nidx];
            float ng = smem_g[nidx];
            float nb = smem_b[nidx];

            // Single color distance: mean absolute channel difference
            int diff = static_cast<int>(
                (fabsf(nr - center_r) + fabsf(ng - center_g) + fabsf(nb - center_b)) *
                    (1.0f / 3.0f) +
                0.5f);
            diff = min(diff, COLOR_LUT_SIZE - 1);

            float w = spatial_weight * d_color_lut[diff];

            sum_r += nr * w;
            sum_g += ng * w;
            sum_b += nb * w;
            wsum += w;
        }
    }

    // Opt6b: replace 3 divisions with 1 reciprocal + 3 multiplications
    float rcp_wsum = __frcp_rn(wsum);
    int out_idx = (y * width + x) * 3;
    output[out_idx] = to_output<OutT>(sum_r * rcp_wsum);
    output[out_idx + 1] = to_output<OutT>(sum_g * rcp_wsum);
    output[out_idx + 2] = to_output<OutT>(sum_b * rcp_wsum);
}

// ============================================================================
// Separable approximation for grayscale: horizontal + vertical passes
// O(2r) complexity instead of O(r^2)
// ============================================================================

template <int RADIUS, typename InT = float, typename TmpT = float>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_BLOCKS_PER_SM)
k_bilateral_horizontal_gray(const InT* __restrict__ input,
                                            TmpT* __restrict__ output, int width, int height) {

    constexpr int TILE_W = BLOCK_X + 2 * RADIUS;
    constexpr int LUT_WIDTH = 2 * RADIUS + 1;

    __shared__ float smem[BLOCK_Y][TILE_W];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int x = blockIdx.x * BLOCK_X + tx;
    const int y = blockIdx.y * BLOCK_Y + ty;

    // Load row with halo
    for (int i = tx; i < TILE_W; i += BLOCK_X) {
        int gx = blockIdx.x * BLOCK_X + i - RADIUS;
        gx = max(0, min(width - 1, gx));
        int gy = min(y, height - 1);
        smem[ty][i] = static_cast<float>(input[gy * width + gx]);
    }

    __syncthreads();

    if (x >= width || y >= height)
        return;

    const int lx = tx + RADIUS;
    const float center = smem[ty][lx];

    float sum = 0.0f;
    float weight_sum = 0.0f;

#pragma unroll
    for (int dx = -RADIUS; dx <= RADIUS; ++dx) {
        float neighbor = smem[ty][lx + dx];
        float spatial_weight = d_spatial_lut[RADIUS * LUT_WIDTH + (dx + RADIUS)];

        int diff = static_cast<int>(fabsf(neighbor - center) + 0.5f);
        diff = min(diff, COLOR_LUT_SIZE - 1);
        float color_weight = d_color_lut[diff];

        float w = spatial_weight * color_weight;
        sum += neighbor * w;
        weight_sum += w;
    }

    output[y * width + x] = static_cast<TmpT>(sum / weight_sum);
}

template <int RADIUS, typename TmpT = float, typename OutT = float>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_BLOCKS_PER_SM)
k_bilateral_vertical_gray(const TmpT* __restrict__ input,
                                          OutT* __restrict__ output, int width, int height) {

    constexpr int TILE_H = BLOCK_Y + 2 * RADIUS;
    constexpr int LUT_WIDTH = 2 * RADIUS + 1;

    __shared__ float smem[TILE_H][BLOCK_X];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int x = blockIdx.x * BLOCK_X + tx;
    const int y = blockIdx.y * BLOCK_Y + ty;

    // Load column with halo (cast TmpT -> float for FP16 intermediate support)
    for (int i = ty; i < TILE_H; i += BLOCK_Y) {
        int gy = blockIdx.y * BLOCK_Y + i - RADIUS;
        gy = max(0, min(height - 1, gy));
        int gx = min(x, width - 1);
        smem[i][tx] = static_cast<float>(input[gy * width + gx]);
    }

    __syncthreads();

    if (x >= width || y >= height)
        return;

    const int ly = ty + RADIUS;
    const float center = smem[ly][tx];

    float sum = 0.0f;
    float weight_sum = 0.0f;

#pragma unroll
    for (int dy = -RADIUS; dy <= RADIUS; ++dy) {
        float neighbor = smem[ly + dy][tx];
        float spatial_weight = d_spatial_lut[(dy + RADIUS) * LUT_WIDTH + RADIUS];

        int diff = static_cast<int>(fabsf(neighbor - center) + 0.5f);
        diff = min(diff, COLOR_LUT_SIZE - 1);
        float color_weight = d_color_lut[diff];

        float w = spatial_weight * color_weight;
        sum += neighbor * w;
        weight_sum += w;
    }

    output[y * width + x] = to_output<OutT>(sum / weight_sum);
}

// ============================================================================
// Separable approximation for RGB: horizontal + vertical passes
// O(2r) complexity instead of O(r^2)
// ============================================================================

template <int RADIUS, typename InT = float, typename TmpT = float>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_BLOCKS_PER_SM)
k_bilateral_horizontal_rgb(const InT* __restrict__ input,
                                           TmpT* __restrict__ output, // intermediate (float or __half)
                                           int width, int height) {

    constexpr int TILE_W = BLOCK_X + 2 * RADIUS;
    constexpr int LUT_WIDTH = 2 * RADIUS + 1;

    __shared__ float smem_r[BLOCK_Y][TILE_W];
    __shared__ float smem_g[BLOCK_Y][TILE_W];
    __shared__ float smem_b[BLOCK_Y][TILE_W];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int x = blockIdx.x * BLOCK_X + tx;
    const int y = blockIdx.y * BLOCK_Y + ty;

    // Load row with halo
    for (int i = tx; i < TILE_W; i += BLOCK_X) {
        int gx = blockIdx.x * BLOCK_X + i - RADIUS;
        gx = max(0, min(width - 1, gx));
        int gy = min(y, height - 1);
        int gidx = (gy * width + gx) * 3;
        smem_r[ty][i] = static_cast<float>(input[gidx]);
        smem_g[ty][i] = static_cast<float>(input[gidx + 1]);
        smem_b[ty][i] = static_cast<float>(input[gidx + 2]);
    }

    __syncthreads();

    if (x >= width || y >= height)
        return;

    const int lx = tx + RADIUS;
    const float center_r = smem_r[ty][lx];
    const float center_g = smem_g[ty][lx];
    const float center_b = smem_b[ty][lx];

    float sum_r = 0.0f, sum_g = 0.0f, sum_b = 0.0f;
    float wsum_r = 0.0f, wsum_g = 0.0f, wsum_b = 0.0f;

#pragma unroll
    for (int dx = -RADIUS; dx <= RADIUS; ++dx) {
        float nr = smem_r[ty][lx + dx];
        float ng = smem_g[ty][lx + dx];
        float nb = smem_b[ty][lx + dx];

        // Use 1D spatial weight (center row of 2D LUT)
        float spatial_weight = d_spatial_lut[RADIUS * LUT_WIDTH + (dx + RADIUS)];

        int diff_r = static_cast<int>(fabsf(nr - center_r) + 0.5f);
        int diff_g = static_cast<int>(fabsf(ng - center_g) + 0.5f);
        int diff_b = static_cast<int>(fabsf(nb - center_b) + 0.5f);
        diff_r = min(diff_r, COLOR_LUT_SIZE - 1);
        diff_g = min(diff_g, COLOR_LUT_SIZE - 1);
        diff_b = min(diff_b, COLOR_LUT_SIZE - 1);

        float cw_r = d_color_lut[diff_r];
        float cw_g = d_color_lut[diff_g];
        float cw_b = d_color_lut[diff_b];

        float w_r = spatial_weight * cw_r;
        float w_g = spatial_weight * cw_g;
        float w_b = spatial_weight * cw_b;

        sum_r += nr * w_r;
        sum_g += ng * w_g;
        sum_b += nb * w_b;
        wsum_r += w_r;
        wsum_g += w_g;
        wsum_b += w_b;
    }

    int out_idx = (y * width + x) * 3;
    output[out_idx]     = static_cast<TmpT>(sum_r / wsum_r);
    output[out_idx + 1] = static_cast<TmpT>(sum_g / wsum_g);
    output[out_idx + 2] = static_cast<TmpT>(sum_b / wsum_b);
}

template <int RADIUS, typename TmpT = float, typename OutT = float>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_BLOCKS_PER_SM)
k_bilateral_vertical_rgb(const TmpT* __restrict__ input, // intermediate (float or __half)
                         OutT* __restrict__ output, int width, int height) {

    constexpr int TILE_H = BLOCK_Y + 2 * RADIUS;
    constexpr int LUT_WIDTH = 2 * RADIUS + 1;

    __shared__ float smem_r[TILE_H][BLOCK_X];
    __shared__ float smem_g[TILE_H][BLOCK_X];
    __shared__ float smem_b[TILE_H][BLOCK_X];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int x = blockIdx.x * BLOCK_X + tx;
    const int y = blockIdx.y * BLOCK_Y + ty;

    // Load column with halo (cast TmpT -> float for FP16 intermediate support)
    for (int i = ty; i < TILE_H; i += BLOCK_Y) {
        int gy = blockIdx.y * BLOCK_Y + i - RADIUS;
        gy = max(0, min(height - 1, gy));
        int gx = min(x, width - 1);
        int gidx = (gy * width + gx) * 3;
        smem_r[i][tx] = static_cast<float>(input[gidx]);
        smem_g[i][tx] = static_cast<float>(input[gidx + 1]);
        smem_b[i][tx] = static_cast<float>(input[gidx + 2]);
    }

    __syncthreads();

    if (x >= width || y >= height)
        return;

    const int ly = ty + RADIUS;
    const float center_r = smem_r[ly][tx];
    const float center_g = smem_g[ly][tx];
    const float center_b = smem_b[ly][tx];

    float sum_r = 0.0f, sum_g = 0.0f, sum_b = 0.0f;
    float wsum_r = 0.0f, wsum_g = 0.0f, wsum_b = 0.0f;

#pragma unroll
    for (int dy = -RADIUS; dy <= RADIUS; ++dy) {
        float nr = smem_r[ly + dy][tx];
        float ng = smem_g[ly + dy][tx];
        float nb = smem_b[ly + dy][tx];

        // Use 1D spatial weight (center column of 2D LUT)
        float spatial_weight = d_spatial_lut[(dy + RADIUS) * LUT_WIDTH + RADIUS];

        int diff_r = static_cast<int>(fabsf(nr - center_r) + 0.5f);
        int diff_g = static_cast<int>(fabsf(ng - center_g) + 0.5f);
        int diff_b = static_cast<int>(fabsf(nb - center_b) + 0.5f);
        diff_r = min(diff_r, COLOR_LUT_SIZE - 1);
        diff_g = min(diff_g, COLOR_LUT_SIZE - 1);
        diff_b = min(diff_b, COLOR_LUT_SIZE - 1);

        float cw_r = d_color_lut[diff_r];
        float cw_g = d_color_lut[diff_g];
        float cw_b = d_color_lut[diff_b];

        float w_r = spatial_weight * cw_r;
        float w_g = spatial_weight * cw_g;
        float w_b = spatial_weight * cw_b;

        sum_r += nr * w_r;
        sum_g += ng * w_g;
        sum_b += nb * w_b;
        wsum_r += w_r;
        wsum_g += w_g;
        wsum_b += w_b;
    }

    int out_idx = (y * width + x) * 3;
    output[out_idx] = to_output<OutT>(sum_r / wsum_r);
    output[out_idx + 1] = to_output<OutT>(sum_g / wsum_g);
    output[out_idx + 2] = to_output<OutT>(sum_b / wsum_b);
}

// ============================================================================
// Runtime-radius version (fallback)
// ============================================================================

// Grayscale runtime version
template <typename InT = float, typename OutT = float>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_BLOCKS_PER_SM)
k_bilateral_filter_shared(const InT* __restrict__ input, OutT* __restrict__ output,
                                          int width, int height, int radius) {

    extern __shared__ float smem[];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int x = blockIdx.x * BLOCK_X + tx;
    const int y = blockIdx.y * BLOCK_Y + ty;

    const int tile_w = BLOCK_X + 2 * radius;
    const int tile_h = BLOCK_Y + 2 * radius;
    const int smem_size = tile_w * tile_h;
    const int thread_id = ty * BLOCK_X + tx;
    const int threads_per_block = BLOCK_X * BLOCK_Y;

    // Collaborative loading
    for (int i = thread_id; i < smem_size; i += threads_per_block) {
        int sy = i / tile_w;
        int sx = i % tile_w;
        int gx = blockIdx.x * BLOCK_X + sx - radius;
        int gy = blockIdx.y * BLOCK_Y + sy - radius;
        gx = max(0, min(width - 1, gx));
        gy = max(0, min(height - 1, gy));
        smem[i] = static_cast<float>(input[gy * width + gx]);
    }

    __syncthreads();

    if (x >= width || y >= height)
        return;

    const int lx = tx + radius;
    const int ly = ty + radius;
    const float center = smem[ly * tile_w + lx];

    float sum = 0.0f;
    float weight_sum = 0.0f;
    const int lut_width = 2 * radius + 1;

#pragma unroll 4
    for (int dy = -radius; dy <= radius; ++dy) {
#pragma unroll 4
        for (int dx = -radius; dx <= radius; ++dx) {
            float spatial_weight = d_spatial_lut[(dy + radius) * lut_width + (dx + radius)];
            if (spatial_weight == 0.0f) continue;

            float neighbor = smem[(ly + dy) * tile_w + (lx + dx)];

            int diff = static_cast<int>(fabsf(neighbor - center) + 0.5f);
            diff = min(diff, COLOR_LUT_SIZE - 1);
            float color_weight = d_color_lut[diff];

            float w = spatial_weight * color_weight;
            sum += neighbor * w;
            weight_sum += w;
        }
    }

    output[y * width + x] = to_output<OutT>(sum / weight_sum);
}

// RGB runtime version
template <typename InT = float, typename OutT = float>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_BLOCKS_PER_SM)
k_bilateral_filter_rgb_shared(const InT* __restrict__ input,
                                              OutT* __restrict__ output, int width, int height,
                                              int radius) {

    extern __shared__ float smem[];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int x = blockIdx.x * BLOCK_X + tx;
    const int y = blockIdx.y * BLOCK_Y + ty;

    const int tile_w = BLOCK_X + 2 * radius;
    const int tile_h = BLOCK_Y + 2 * radius;
    const int tile_size = tile_w * tile_h;

    float* smem_r = smem;
    float* smem_g = smem + tile_size;
    float* smem_b = smem + 2 * tile_size;

    const int thread_id = ty * BLOCK_X + tx;
    const int threads_per_block = BLOCK_X * BLOCK_Y;

    for (int i = thread_id; i < tile_size; i += threads_per_block) {
        int sy = i / tile_w;
        int sx = i % tile_w;
        int gx = blockIdx.x * BLOCK_X + sx - radius;
        int gy = blockIdx.y * BLOCK_Y + sy - radius;
        gx = max(0, min(width - 1, gx));
        gy = max(0, min(height - 1, gy));
        int gidx = (gy * width + gx) * 3;
        smem_r[i] = static_cast<float>(input[gidx]);
        smem_g[i] = static_cast<float>(input[gidx + 1]);
        smem_b[i] = static_cast<float>(input[gidx + 2]);
    }

    __syncthreads();

    if (x >= width || y >= height)
        return;

    const int lx = tx + radius;
    const int ly = ty + radius;
    const int lidx = ly * tile_w + lx;

    const float center_r = smem_r[lidx];
    const float center_g = smem_g[lidx];
    const float center_b = smem_b[lidx];

    float sum_r = 0.0f, sum_g = 0.0f, sum_b = 0.0f;
    float wsum_r = 0.0f, wsum_g = 0.0f, wsum_b = 0.0f;
    const int lut_width = 2 * radius + 1;

#pragma unroll 4
    for (int dy = -radius; dy <= radius; ++dy) {
#pragma unroll 4
        for (int dx = -radius; dx <= radius; ++dx) {
            float spatial_weight = d_spatial_lut[(dy + radius) * lut_width + (dx + radius)];
            if (spatial_weight == 0.0f) continue;

            int nidx = (ly + dy) * tile_w + (lx + dx);
            float nr = smem_r[nidx];
            float ng = smem_g[nidx];
            float nb = smem_b[nidx];

            int diff_r = static_cast<int>(fabsf(nr - center_r) + 0.5f);
            int diff_g = static_cast<int>(fabsf(ng - center_g) + 0.5f);
            int diff_b = static_cast<int>(fabsf(nb - center_b) + 0.5f);
            diff_r = min(diff_r, COLOR_LUT_SIZE - 1);
            diff_g = min(diff_g, COLOR_LUT_SIZE - 1);
            diff_b = min(diff_b, COLOR_LUT_SIZE - 1);

            float cw_r = d_color_lut[diff_r];
            float cw_g = d_color_lut[diff_g];
            float cw_b = d_color_lut[diff_b];

            float w_r = spatial_weight * cw_r;
            float w_g = spatial_weight * cw_g;
            float w_b = spatial_weight * cw_b;

            sum_r += nr * w_r;
            sum_g += ng * w_g;
            sum_b += nb * w_b;
            wsum_r += w_r;
            wsum_g += w_g;
            wsum_b += w_b;
        }
    }

    int out_idx = (y * width + x) * 3;
    output[out_idx] = to_output<OutT>(sum_r / wsum_r);
    output[out_idx + 1] = to_output<OutT>(sum_g / wsum_g);
    output[out_idx + 2] = to_output<OutT>(sum_b / wsum_b);
}

// ============================================================================
// LUT initialization
// ============================================================================

static void init_spatial_lut(int radius, float sigma_spatial) {
    float coeff = -0.5f / (sigma_spatial * sigma_spatial);
    int w = 2 * radius + 1;
    std::vector<float> lut(LUT_SIZE, 0.0f);

    const int r2 = radius * radius;
    for (int dy = -radius; dy <= radius; ++dy) {
        for (int dx = -radius; dx <= radius; ++dx) {
            // Opt C/F: circular window - zero out corners outside inscribed circle.
            // For r=5: 121 positions → 81 inside, 40 outside (33%).
            // TEMPLATE kernels with #pragma unroll: compiler eliminates dead iterations
            // at compile time (dead code elimination), achieving +13%~+65% speedup.
            if (dx * dx + dy * dy > r2) {
                lut[(dy + radius) * w + (dx + radius)] = 0.0f;
            } else {
                lut[(dy + radius) * w + (dx + radius)] = expf((dx * dx + dy * dy) * coeff);
            }
        }
    }

    CUDA_CHECK(cudaMemcpyToSymbol(d_spatial_lut, lut.data(), w * w * sizeof(float)));
}

static void init_color_lut(float sigma_color) {
    float coeff = -0.5f / (sigma_color * sigma_color);
    std::vector<float> lut(COLOR_LUT_SIZE);

    for (int i = 0; i < COLOR_LUT_SIZE; ++i) {
        // i ∈ [0,255], i*i ∈ [0,65025]: no overflow for int
        lut[i] = expf(static_cast<float>(i * i) * coeff);
    }

    CUDA_CHECK(cudaMemcpyToSymbol(d_color_lut, lut.data(), COLOR_LUT_SIZE * sizeof(float)));
}

// Opt B (disabled): cudaFuncSetCacheConfig was tested and showed no benefit on sm_89.
// Moreover, calling it on many template instantiations triggers a JIT crash in the
// CUDA 13.1 PTX compiler under WSL2. The function has been removed entirely.

// Opt1: LUT cache - only re-upload when params change
static void ensure_luts(int radius, float sigma_spatial, float sigma_color) {
    static int cached_radius = -1;
    static float cached_sigma_s = -1.f;
    static float cached_sigma_c = -1.f;

    if (radius == cached_radius && sigma_spatial == cached_sigma_s &&
        sigma_color == cached_sigma_c) {
        return;
    }

    init_spatial_lut(radius, sigma_spatial);
    init_color_lut(sigma_color);

    cached_radius = radius;
    cached_sigma_s = sigma_spatial;
    cached_sigma_c = sigma_color;
}

// ============================================================================
// Kernel dispatch with template specialization
// ============================================================================

enum class FilterMode {
    STANDARD,       // Shared memory + LUT (runtime radius)
    TEMPLATE,       // Template-based (compile-time radius)
    SEPARABLE,      // Separable approximation (float intermediate)
    SEPARABLE_FP16, // Separable approximation (__half intermediate, half bandwidth)
    ADAPTIVE        // Adaptive radius: per-pixel radius from local gradient
};

static FilterMode g_filter_mode = FilterMode::TEMPLATE;

void set_bilateral_filter_mode(int mode) {
    g_filter_mode = static_cast<FilterMode>(mode);
}

static FilterMode get_filter_mode() {
    static bool initialized = false;
    if (!initialized) {
        const char* env = getenv("BILATERAL_MODE");
        if (env) {
            int mode = atoi(env);
            if (mode >= 0 && mode <= 4) {
                g_filter_mode = static_cast<FilterMode>(mode);
            }
        }
        initialized = true;
    }
    return g_filter_mode;
}

// Float-path template launchers removed: the float bilateral_filter_cuda() now uses
// only runtime-radius shared-memory kernels. All template specializations are
// reserved for the uint8 direct path (dispatch_u8_kernel).

// ============================================================================
// Opt2: uint8 I/O launchers - kernel reads/writes uint8 directly from GPU
// ============================================================================

// Grayscale template uint8 launcher
template <int RADIUS>
static void launch_u8_gray(const uint8_t* d_in, uint8_t* d_out, int w, int h, cudaStream_t s) {
    dim3 block(BLOCK_X, BLOCK_Y);
    dim3 grid((w + BLOCK_X - 1) / BLOCK_X, (h + BLOCK_Y - 1) / BLOCK_Y);
    k_bilateral_filter_gray_template<RADIUS, uint8_t, uint8_t>
        <<<grid, block, 0, s>>>(d_in, d_out, w, h);
}

// RGB template uint8 launcher
template <int RADIUS>
static void launch_u8_rgb(const uint8_t* d_in, uint8_t* d_out, int w, int h, cudaStream_t s) {
    dim3 block(BLOCK_X, BLOCK_Y);
    dim3 grid((w + BLOCK_X - 1) / BLOCK_X, (h + BLOCK_Y - 1) / BLOCK_Y);
    k_bilateral_filter_rgb_template<RADIUS, uint8_t, uint8_t>
        <<<grid, block, 0, s>>>(d_in, d_out, w, h);
}

// Grayscale separable uint8 launcher (uint8→float→uint8 via d_temp)
template <int RADIUS>
static void launch_u8_sep_gray(const uint8_t* d_in, uint8_t* d_out, float* d_temp, int w, int h,
                               cudaStream_t s) {
    dim3 block(BLOCK_X, BLOCK_Y);
    dim3 grid((w + BLOCK_X - 1) / BLOCK_X, (h + BLOCK_Y - 1) / BLOCK_Y);
    k_bilateral_horizontal_gray<RADIUS, uint8_t, float><<<grid, block, 0, s>>>(d_in, d_temp, w, h);
    k_bilateral_vertical_gray<RADIUS, float, uint8_t><<<grid, block, 0, s>>>(d_temp, d_out, w, h);
}

// RGB separable uint8 launcher (uint8→float→uint8 via d_temp)
template <int RADIUS>
static void launch_u8_sep_rgb(const uint8_t* d_in, uint8_t* d_out, float* d_temp, int w, int h,
                               cudaStream_t s) {
    dim3 block(BLOCK_X, BLOCK_Y);
    dim3 grid((w + BLOCK_X - 1) / BLOCK_X, (h + BLOCK_Y - 1) / BLOCK_Y);
    k_bilateral_horizontal_rgb<RADIUS, uint8_t, float><<<grid, block, 0, s>>>(d_in, d_temp, w, h);
    k_bilateral_vertical_rgb<RADIUS, float, uint8_t><<<grid, block, 0, s>>>(d_temp, d_out, w, h);
}

// FP16 separable launchers removed to reduce fatbin size (20 kernel instantiations).
// SEPARABLE_FP16 mode now falls through to float SEPARABLE at runtime.

// ============================================================================
// Opt1: persistent GPU buffers - allocated once, reused across calls
// Opt3: cudaHostRegister cache - page-lock caller's memory for DMA transfers
// ============================================================================

static struct {
    uint8_t* d_in_u8 = nullptr;
    uint8_t* d_out_u8 = nullptr;
    float*   d_temp = nullptr;      // separable float intermediate
    __half*  d_temp_h16 = nullptr;  // separable FP16 intermediate
    uint8_t* d_radius_map = nullptr; // adaptive mode: per-pixel radius
    size_t n_u8 = 0;
    size_t n_temp = 0;
    size_t n_temp_h16 = 0;
    size_t n_radius_map = 0;
    // Opt3: cached registered host pointers
    const uint8_t* h_in_reg = nullptr;
    uint8_t* h_out_reg = nullptr;
    size_t n_reg = 0;
} g_bufs;

// Opt3: register caller's heap memory as page-locked so cudaMemcpy uses DMA.
// Called once per unique (h_in, h_out, n) triple; re-registers when they change.
static void ensure_registered(const uint8_t* h_in, uint8_t* h_out, size_t n) {
    if (h_in == g_bufs.h_in_reg && h_out == g_bufs.h_out_reg && n == g_bufs.n_reg)
        return;
    // Unregister previous pointers (ignore errors on first call / already-unregistered)
    if (g_bufs.h_in_reg)
        cudaHostUnregister(const_cast<uint8_t*>(g_bufs.h_in_reg));
    if (g_bufs.h_out_reg)
        cudaHostUnregister(g_bufs.h_out_reg);
    g_bufs.h_in_reg = nullptr;
    g_bufs.h_out_reg = nullptr;
    g_bufs.n_reg = 0;
    // Register new pointers; if either fails, CUDA_CHECK will abort
    CUDA_CHECK(cudaHostRegister(const_cast<uint8_t*>(h_in), n, cudaHostRegisterDefault));
    CUDA_CHECK(cudaHostRegister(h_out, n, cudaHostRegisterDefault));
    g_bufs.h_in_reg = h_in;
    g_bufs.h_out_reg = h_out;
    g_bufs.n_reg = n;
}

static void ensure_io_buffers(size_t n_u8) {
    if (n_u8 == g_bufs.n_u8)
        return;
    cudaFree(g_bufs.d_in_u8);
    cudaFree(g_bufs.d_out_u8);
    CUDA_CHECK(cudaMalloc(&g_bufs.d_in_u8, n_u8));
    CUDA_CHECK(cudaMalloc(&g_bufs.d_out_u8, n_u8));
    g_bufs.n_u8 = n_u8;
}

static void ensure_temp_buffer(size_t n_bytes) {
    if (n_bytes == g_bufs.n_temp)
        return;
    cudaFree(g_bufs.d_temp);
    CUDA_CHECK(cudaMalloc(&g_bufs.d_temp, n_bytes));
    g_bufs.n_temp = n_bytes;
}

// Adaptive mode: per-pixel radius map buffer
static void ensure_radius_map_buffer(size_t n_pixels) {
    if (n_pixels == g_bufs.n_radius_map)
        return;
    cudaFree(g_bufs.d_radius_map);
    CUDA_CHECK(cudaMalloc(&g_bufs.d_radius_map, n_pixels));
    g_bufs.n_radius_map = n_pixels;
}

// ============================================================================
// Adaptive radius: compute per-pixel radius from local gradient magnitude
// ============================================================================

// Sobel-based gradient magnitude → radius map.
// High gradient (edge) → small radius; low gradient (flat) → large radius.
// r_min, r_max are the allowed radius bounds. grad_threshold controls the
// gradient value at which radius reaches r_min (fully edge).
__global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_BLOCKS_PER_SM)
k_compute_radius_map_gray(const uint8_t* __restrict__ input,
                          uint8_t* __restrict__ radius_map,
                          int width, int height,
                          int r_min, int r_max, float inv_grad_threshold) {
    const int x = blockIdx.x * BLOCK_X + threadIdx.x;
    const int y = blockIdx.y * BLOCK_Y + threadIdx.y;
    if (x >= width || y >= height)
        return;

    // Clamp coords for border pixels
    int x0 = max(x - 1, 0), x2 = min(x + 1, width - 1);
    int y0 = max(y - 1, 0), y2 = min(y + 1, height - 1);

    // Sobel gradient: Gx and Gy using 3x3 Sobel operator
    float p00 = input[y0 * width + x0], p01 = input[y0 * width + x], p02 = input[y0 * width + x2];
    float p10 = input[y  * width + x0],                               p12 = input[y  * width + x2];
    float p20 = input[y2 * width + x0], p21 = input[y2 * width + x], p22 = input[y2 * width + x2];

    float gx = -p00 + p02 - 2.0f * p10 + 2.0f * p12 - p20 + p22;
    float gy = -p00 - 2.0f * p01 - p02 + p20 + 2.0f * p21 + p22;
    float grad = sqrtf(gx * gx + gy * gy);

    // Map gradient to radius: linear interpolation
    // grad=0 → r_max (flat area, smooth more), grad>=threshold → r_min (edge, smooth less)
    float t = fminf(grad * inv_grad_threshold, 1.0f);
    int r = __float2int_rn(static_cast<float>(r_max) - t * static_cast<float>(r_max - r_min));
    r = max(r_min, min(r_max, r));

    radius_map[y * width + x] = static_cast<uint8_t>(r);
}

__global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_BLOCKS_PER_SM)
k_compute_radius_map_rgb(const uint8_t* __restrict__ input,
                         uint8_t* __restrict__ radius_map,
                         int width, int height,
                         int r_min, int r_max, float inv_grad_threshold) {
    const int x = blockIdx.x * BLOCK_X + threadIdx.x;
    const int y = blockIdx.y * BLOCK_Y + threadIdx.y;
    if (x >= width || y >= height)
        return;

    int x0 = max(x - 1, 0), x2 = min(x + 1, width - 1);
    int y0 = max(y - 1, 0), y2 = min(y + 1, height - 1);

    // Compute gradient on luminance approximation: (R + G + B) / 3
    auto lum = [&](int py, int px) -> float {
        int idx = (py * width + px) * 3;
        return (static_cast<float>(input[idx]) +
                static_cast<float>(input[idx + 1]) +
                static_cast<float>(input[idx + 2])) * (1.0f / 3.0f);
    };

    float p00 = lum(y0, x0), p01 = lum(y0, x), p02 = lum(y0, x2);
    float p10 = lum(y,  x0),                    p12 = lum(y,  x2);
    float p20 = lum(y2, x0), p21 = lum(y2, x), p22 = lum(y2, x2);

    float gx = -p00 + p02 - 2.0f * p10 + 2.0f * p12 - p20 + p22;
    float gy = -p00 - 2.0f * p01 - p02 + p20 + 2.0f * p21 + p22;
    float grad = sqrtf(gx * gx + gy * gy);

    float t = fminf(grad * inv_grad_threshold, 1.0f);
    int r = __float2int_rn(static_cast<float>(r_max) - t * static_cast<float>(r_max - r_min));
    r = max(r_min, min(r_max, r));

    radius_map[y * width + x] = static_cast<uint8_t>(r);
}

// ============================================================================
// Adaptive bilateral filter kernels: read per-pixel radius from radius_map
// Shared memory is allocated for r_max halo (worst case) so all threads
// in a block can access the full neighborhood regardless of their radius.
// ============================================================================

template <typename InT = uint8_t, typename OutT = uint8_t>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_BLOCKS_PER_SM)
k_bilateral_adaptive_gray(const InT* __restrict__ input,
                          OutT* __restrict__ output,
                          const uint8_t* __restrict__ radius_map,
                          int width, int height, int r_max) {

    extern __shared__ float smem[];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int x = blockIdx.x * BLOCK_X + tx;
    const int y = blockIdx.y * BLOCK_Y + ty;

    const int tile_w = BLOCK_X + 2 * r_max;
    const int tile_h = BLOCK_Y + 2 * r_max;
    const int tile_size = tile_w * tile_h;
    const int thread_id = ty * BLOCK_X + tx;
    const int threads_per_block = BLOCK_X * BLOCK_Y;

    // Collaborative loading with r_max halo
    for (int i = thread_id; i < tile_size; i += threads_per_block) {
        int sy = i / tile_w;
        int sx = i % tile_w;
        int gx = blockIdx.x * BLOCK_X + sx - r_max;
        int gy = blockIdx.y * BLOCK_Y + sy - r_max;
        gx = max(0, min(width - 1, gx));
        gy = max(0, min(height - 1, gy));
        smem[i] = static_cast<float>(input[gy * width + gx]);
    }

    __syncthreads();

    if (x >= width || y >= height)
        return;

    // Read per-pixel radius
    const int my_radius = static_cast<int>(radius_map[y * width + x]);
    const int lx = tx + r_max;
    const int ly = ty + r_max;
    const float center = smem[ly * tile_w + lx];
    // Spatial LUT was built with r_max, so use r_max-based width and offset
    const int lut_width = 2 * r_max + 1;
    const int lut_center = r_max;  // center offset in the LUT

    float sum = 0.0f;
    float weight_sum = 0.0f;

    for (int dy = -my_radius; dy <= my_radius; ++dy) {
        for (int dx = -my_radius; dx <= my_radius; ++dx) {
            float neighbor = smem[(ly + dy) * tile_w + (lx + dx)];

            float spatial_weight = d_spatial_lut[(dy + lut_center) * lut_width + (dx + lut_center)];

            int diff = static_cast<int>(fabsf(neighbor - center) + 0.5f);
            diff = min(diff, COLOR_LUT_SIZE - 1);
            float color_weight = d_color_lut[diff];

            float w = spatial_weight * color_weight;
            sum += neighbor * w;
            weight_sum += w;
        }
    }

    output[y * width + x] = to_output<OutT>(sum / weight_sum);
}

template <typename InT = uint8_t, typename OutT = uint8_t>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_BLOCKS_PER_SM)
k_bilateral_adaptive_rgb(const InT* __restrict__ input,
                         OutT* __restrict__ output,
                         const uint8_t* __restrict__ radius_map,
                         int width, int height, int r_max) {

    extern __shared__ float smem[];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int x = blockIdx.x * BLOCK_X + tx;
    const int y = blockIdx.y * BLOCK_Y + ty;

    const int tile_w = BLOCK_X + 2 * r_max;
    const int tile_h = BLOCK_Y + 2 * r_max;
    const int tile_size = tile_w * tile_h;
    const int thread_id = ty * BLOCK_X + tx;
    const int threads_per_block = BLOCK_X * BLOCK_Y;

    float* smem_r = smem;
    float* smem_g = smem + tile_size;
    float* smem_b = smem + 2 * tile_size;

    // Collaborative loading with r_max halo
    for (int i = thread_id; i < tile_size; i += threads_per_block) {
        int sy = i / tile_w;
        int sx = i % tile_w;
        int gx = blockIdx.x * BLOCK_X + sx - r_max;
        int gy = blockIdx.y * BLOCK_Y + sy - r_max;
        gx = max(0, min(width - 1, gx));
        gy = max(0, min(height - 1, gy));
        int gidx = (gy * width + gx) * 3;
        smem_r[i] = static_cast<float>(input[gidx]);
        smem_g[i] = static_cast<float>(input[gidx + 1]);
        smem_b[i] = static_cast<float>(input[gidx + 2]);
    }

    __syncthreads();

    if (x >= width || y >= height)
        return;

    const int my_radius = static_cast<int>(radius_map[y * width + x]);
    const int lx = tx + r_max;
    const int ly = ty + r_max;
    const int lidx = ly * tile_w + lx;
    // Spatial LUT was built with r_max, so use r_max-based width and offset
    const int lut_width = 2 * r_max + 1;
    const int lut_center = r_max;

    const float center_r = smem_r[lidx];
    const float center_g = smem_g[lidx];
    const float center_b = smem_b[lidx];

    float sum_r = 0.0f, sum_g = 0.0f, sum_b = 0.0f;
    float wsum = 0.0f;

    for (int dy = -my_radius; dy <= my_radius; ++dy) {
        for (int dx = -my_radius; dx <= my_radius; ++dx) {
            int nidx = (ly + dy) * tile_w + (lx + dx);
            float nr = smem_r[nidx];
            float ng = smem_g[nidx];
            float nb = smem_b[nidx];

            float spatial_weight = d_spatial_lut[(dy + lut_center) * lut_width + (dx + lut_center)];

            // Mean absolute channel difference (approximation of OpenCV's L2 distance)
            int diff = static_cast<int>(
                (fabsf(nr - center_r) + fabsf(ng - center_g) + fabsf(nb - center_b)) *
                    (1.0f / 3.0f) + 0.5f);
            diff = min(diff, COLOR_LUT_SIZE - 1);

            float w = spatial_weight * d_color_lut[diff];
            sum_r += nr * w;
            sum_g += ng * w;
            sum_b += nb * w;
            wsum += w;
        }
    }

    float rcp_wsum = __frcp_rn(wsum);
    int out_idx = (y * width + x) * 3;
    output[out_idx]     = to_output<OutT>(sum_r * rcp_wsum);
    output[out_idx + 1] = to_output<OutT>(sum_g * rcp_wsum);
    output[out_idx + 2] = to_output<OutT>(sum_b * rcp_wsum);
}

// Float-path bilateral_filter_cuda: simplified to use only runtime-radius shared-memory
// kernels (no template specializations). Template kernels are only used in the uint8
// direct path (dispatch_u8_kernel) which is the primary performance path.
void bilateral_filter_cuda(const float* d_input, float* d_output, int width, int height,
                           int channels, int radius, float sigma_spatial, float sigma_color,
                           cudaStream_t stream) {

    radius = min(radius, MAX_RADIUS);
    ensure_luts(radius, sigma_spatial, sigma_color);

    dim3 block(BLOCK_X, BLOCK_Y);
    dim3 grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y);

    if (channels == 1) {
        int tile_w = BLOCK_X + 2 * radius;
        int tile_h = BLOCK_Y + 2 * radius;
        size_t smem = tile_w * tile_h * sizeof(float);
        k_bilateral_filter_shared<float, float><<<grid, block, smem, stream>>>(
            d_input, d_output, width, height, radius);
    } else {
        int tile_w = BLOCK_X + 2 * radius;
        int tile_h = BLOCK_Y + 2 * radius;
        size_t smem = 3 * tile_w * tile_h * sizeof(float);
        k_bilateral_filter_rgb_shared<float, float><<<grid, block, smem, stream>>>(
            d_input, d_output, width, height, radius);
    }

    CUDA_CHECK(cudaGetLastError());
    if (stream == 0) {
        CUDA_CHECK(cudaDeviceSynchronize());
    }
}

// ============================================================================
// Opt E: Strip pipelining - internal kernel-only dispatcher (no H2D/D2H)
// Called per-strip or once for the full image.
// All GPU buffers (d_in, d_out, d_temp, d_temp_h16, d_rmap) are pre-allocated
// by the caller. ensure_luts() must also be called before this function.
// ============================================================================
static void dispatch_u8_kernel(
        const uint8_t* d_in, uint8_t* d_out,
        float* d_temp, __half* d_temp_h16, uint8_t* d_rmap,
        int width, int height, int channels,
        int radius, float sigma_spatial, float sigma_color,
        FilterMode mode, cudaStream_t stream) {

    const dim3 block(BLOCK_X, BLOCK_Y);
    const dim3 grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y);

    if (channels == 1) {
        if (mode == FilterMode::ADAPTIVE) {
            int r_max = radius;
            int r_min = max(2, radius - 1);
            float grad_threshold = fmaxf(sigma_color * 1.5f, 20.0f);
            float inv_grad_threshold = 1.0f / grad_threshold;
            k_compute_radius_map_gray<<<grid, block, 0, stream>>>(
                d_in, d_rmap, width, height, r_min, r_max, inv_grad_threshold);
            ensure_luts(r_max, sigma_spatial, sigma_color);
            int tile_w = BLOCK_X + 2 * r_max;
            int tile_h = BLOCK_Y + 2 * r_max;
            size_t smem_bytes = static_cast<size_t>(tile_w) * tile_h * sizeof(float);
            k_bilateral_adaptive_gray<uint8_t, uint8_t>
                <<<grid, block, smem_bytes, stream>>>(d_in, d_out, d_rmap, width, height, r_max);
        } else if (mode == FilterMode::SEPARABLE_FP16 || mode == FilterMode::SEPARABLE) {
            // Only radius=5 specialization; others fallback to runtime shared-memory
            if (radius == 5) {
                launch_u8_sep_gray<5>(d_in, d_out, d_temp, width, height, stream);
            } else {
                size_t smem = (BLOCK_X + 2*radius) * (BLOCK_Y + 2*radius) * sizeof(float);
                k_bilateral_filter_shared<uint8_t, uint8_t>
                    <<<grid, block, smem, stream>>>(d_in, d_out, width, height, radius);
            }
        } else if (mode == FilterMode::TEMPLATE) {
            if (radius == 5) {
                launch_u8_gray<5>(d_in, d_out, width, height, stream);
            } else {
                size_t smem = (BLOCK_X + 2*radius) * (BLOCK_Y + 2*radius) * sizeof(float);
                k_bilateral_filter_shared<uint8_t, uint8_t>
                    <<<grid, block, smem, stream>>>(d_in, d_out, width, height, radius);
            }
        } else {
            // STANDARD
            size_t smem = (BLOCK_X + 2*radius) * (BLOCK_Y + 2*radius) * sizeof(float);
            k_bilateral_filter_shared<uint8_t, uint8_t>
                <<<grid, block, smem, stream>>>(d_in, d_out, width, height, radius);
        }
    } else {
        // RGB
        if (mode == FilterMode::ADAPTIVE) {
            int r_max = radius;
            int r_min = max(2, radius - 1);
            float grad_threshold = fmaxf(sigma_color * 1.5f, 20.0f);
            float inv_grad_threshold = 1.0f / grad_threshold;
            k_compute_radius_map_rgb<<<grid, block, 0, stream>>>(
                d_in, d_rmap, width, height, r_min, r_max, inv_grad_threshold);
            ensure_luts(r_max, sigma_spatial, sigma_color);
            int tile_w = BLOCK_X + 2 * r_max;
            int tile_h = BLOCK_Y + 2 * r_max;
            size_t smem_bytes = 3 * static_cast<size_t>(tile_w) * tile_h * sizeof(float);
            k_bilateral_adaptive_rgb<uint8_t, uint8_t>
                <<<grid, block, smem_bytes, stream>>>(d_in, d_out, d_rmap, width, height, r_max);
        } else if (mode == FilterMode::SEPARABLE_FP16 || mode == FilterMode::SEPARABLE) {
            if (radius == 5) {
                launch_u8_sep_rgb<5>(d_in, d_out, d_temp, width, height, stream);
            } else {
                size_t smem = 3 * (BLOCK_X + 2*radius) * (BLOCK_Y + 2*radius) * sizeof(float);
                k_bilateral_filter_rgb_shared<uint8_t, uint8_t>
                    <<<grid, block, smem, stream>>>(d_in, d_out, width, height, radius);
            }
        } else if (mode == FilterMode::TEMPLATE) {
            if (radius == 5) {
                launch_u8_rgb<5>(d_in, d_out, width, height, stream);
            } else {
                size_t smem = 3 * (BLOCK_X + 2*radius) * (BLOCK_Y + 2*radius) * sizeof(float);
                k_bilateral_filter_rgb_shared<uint8_t, uint8_t>
                    <<<grid, block, smem, stream>>>(d_in, d_out, width, height, radius);
            }
        } else {
            // STANDARD
            size_t smem = 3 * (BLOCK_X + 2*radius) * (BLOCK_Y + 2*radius) * sizeof(float);
            k_bilateral_filter_rgb_shared<uint8_t, uint8_t>
                <<<grid, block, smem, stream>>>(d_in, d_out, width, height, radius);
        }
    }
}

// ============================================================================
// Opt E: Strip pipeline state - per-strip GPU buffers + CUDA streams.
// Enables H2D / kernel / D2H overlap across strips, hiding PCIe latency.
// ============================================================================

static constexpr int MAX_STRIPS = 8;

static struct {
    int          count          = 0;
    cudaStream_t streams[MAX_STRIPS];
    uint8_t*     d_in  [MAX_STRIPS];
    uint8_t*     d_out [MAX_STRIPS];
    float*       d_temp[MAX_STRIPS];   // separable float intermediate
    __half*      d_temph[MAX_STRIPS];  // separable FP16 intermediate
    uint8_t*     d_rmap[MAX_STRIPS];   // adaptive radius map
    size_t       cap_in  [MAX_STRIPS]; // current allocated bytes
    size_t       cap_temp[MAX_STRIPS];
    size_t       cap_rmap[MAX_STRIPS];
} g_strip;

// Lazily allocate / resize per-strip buffers.
static void ensure_strip_buffers(int s, size_t in_bytes, size_t temp_bytes,
                                  size_t temph_bytes, size_t rmap_bytes) {
    if (in_bytes > g_strip.cap_in[s]) {
        cudaFree(g_strip.d_in[s]);
        cudaFree(g_strip.d_out[s]);
        CUDA_CHECK(cudaMalloc(&g_strip.d_in[s],  in_bytes));
        CUDA_CHECK(cudaMalloc(&g_strip.d_out[s], in_bytes));
        g_strip.cap_in[s] = in_bytes;
    }
    if (temp_bytes > g_strip.cap_temp[s]) {
        cudaFree(g_strip.d_temp[s]);
        cudaFree(g_strip.d_temph[s]);
        CUDA_CHECK(cudaMalloc(&g_strip.d_temp[s],  temp_bytes));
        CUDA_CHECK(cudaMalloc(&g_strip.d_temph[s], temph_bytes));
        g_strip.cap_temp[s] = temp_bytes;
    }
    if (rmap_bytes > g_strip.cap_rmap[s]) {
        cudaFree(g_strip.d_rmap[s]);
        CUDA_CHECK(cudaMalloc(&g_strip.d_rmap[s], rmap_bytes));
        g_strip.cap_rmap[s] = rmap_bytes;
    }
}

// Create streams on first use.
static void ensure_strip_streams(int num_strips) {
    if (g_strip.count >= num_strips) return;
    for (int s = g_strip.count; s < num_strips; ++s) {
        CUDA_CHECK(cudaStreamCreateWithFlags(&g_strip.streams[s], cudaStreamNonBlocking));
        g_strip.d_in[s] = g_strip.d_out[s] = nullptr;
        g_strip.d_temp[s] = nullptr;
        g_strip.d_temph[s] = nullptr;
        g_strip.d_rmap[s] = nullptr;
        g_strip.cap_in[s] = g_strip.cap_temp[s] = g_strip.cap_rmap[s] = 0;
    }
    g_strip.count = num_strips;
}

// Strip pipeline entry point.
// Splits the image into num_strips horizontal strips and pipelines:
//   H2D(i) → kernel(i) → D2H(i)  on stream[i]
// Streams run concurrently; GPU copy engine overlaps with compute engine.
//
// Each strip input includes halo rows (±radius) so border pixels are correct.
// Only the non-halo rows are copied back to the host output.
static void apply_strip_pipeline(
        const uint8_t* h_input, uint8_t* h_output,
        int width, int height, int channels,
        int radius, float sigma_spatial, float sigma_color,
        FilterMode mode, int num_strips) {

    ensure_strip_streams(num_strips);

    const int rows_per_strip = (height + num_strips - 1) / num_strips;
    const size_t row_bytes   = static_cast<size_t>(width) * channels;

    for (int s = 0; s < num_strips; ++s) {
        const int row_start = s * rows_per_strip;
        const int row_end   = std::min(row_start + rows_per_strip, height);
        if (row_start >= height) break;

        const int halo_top    = std::min(radius, row_start);
        const int halo_bot    = std::min(radius, height - row_end);
        const int strip_h     = halo_top + (row_end - row_start) + halo_bot;
        const size_t in_bytes = static_cast<size_t>(strip_h) * row_bytes;

        // Temp buffer for separable float intermediate
        const size_t temp_bytes  = static_cast<size_t>(strip_h) * width * channels * sizeof(float);
        const size_t temph_bytes = static_cast<size_t>(strip_h) * width * channels * sizeof(__half);
        const size_t rmap_bytes  = static_cast<size_t>(strip_h) * width; // 1 byte/pixel

        ensure_strip_buffers(s, in_bytes, temp_bytes, temph_bytes, rmap_bytes);

        // Async H2D: copy strip input with halo
        const uint8_t* src = h_input + (row_start - halo_top) * row_bytes;
        CUDA_CHECK(cudaMemcpyAsync(g_strip.d_in[s], src, in_bytes,
                                   cudaMemcpyHostToDevice, g_strip.streams[s]));

        // Kernel on this strip (local dimensions: width × strip_h)
        dispatch_u8_kernel(
            g_strip.d_in[s], g_strip.d_out[s],
            g_strip.d_temp[s], g_strip.d_temph[s], g_strip.d_rmap[s],
            width, strip_h, channels,
            radius, sigma_spatial, sigma_color, mode, g_strip.streams[s]);

        // Async D2H: copy back only the non-halo rows
        const size_t out_offset  = static_cast<size_t>(halo_top) * row_bytes;
        const size_t out_bytes   = static_cast<size_t>(row_end - row_start) * row_bytes;
        uint8_t* dst = h_output + static_cast<size_t>(row_start) * row_bytes;
        CUDA_CHECK(cudaMemcpyAsync(dst, g_strip.d_out[s] + out_offset, out_bytes,
                                   cudaMemcpyDeviceToHost, g_strip.streams[s]));
    }

    // Wait for all strips to complete
    CUDA_CHECK(cudaDeviceSynchronize());
}

void apply_bilateral_filter_cuda(const uint8_t* h_input, uint8_t* h_output, int width, int height,
                                 int channels, int radius, float sigma_spatial, float sigma_color) {
    // Opt B: ensure_l1_cache_prefer() was disabled - triggers JIT crash on WSL2
    // (cudaFuncSetCacheConfig on all template instantiations forces simultaneous JIT,
    //  which crashes libnvidia-ptxjitcompiler.so in WSL2 environment.
    //  The optimization itself also showed no benefit on sm_89 Ada architecture.)

    const size_t n_u8 = static_cast<size_t>(width) * height * channels;

    // (Re)allocate only when image size changes
    ensure_io_buffers(n_u8);

    // Opt3: page-lock caller's buffers for faster H2D/D2H transfers
    ensure_registered(h_input, h_output, n_u8);

    radius = min(radius, MAX_RADIUS);
    ensure_luts(radius, sigma_spatial, sigma_color);

    // Opt E: check BILATERAL_STRIP env var for strip pipeline
    static int num_strips = -1;
    if (num_strips < 0) {
        const char* env = getenv("BILATERAL_STRIP");
        num_strips = (env && atoi(env) > 1) ? std::min(atoi(env), MAX_STRIPS) : 1;
    }

    // Opt2: dispatch uint8 I/O kernels directly - no float conversion pipeline
    const FilterMode mode = get_filter_mode();

    // Opt E: strip pipeline - H2D/kernel/D2H overlap across num_strips streams
    if (num_strips > 1) {
        apply_strip_pipeline(h_input, h_output, width, height, channels,
                             radius, sigma_spatial, sigma_color, mode, num_strips);
        return;
    }

    // Single-shot path: one synchronous H2D → kernel → D2H
    CUDA_CHECK(cudaMemcpy(g_bufs.d_in_u8, h_input, n_u8, cudaMemcpyHostToDevice));

    // Pre-allocate intermediate buffers required by the selected mode
    if (mode == FilterMode::SEPARABLE || mode == FilterMode::SEPARABLE_FP16) {
        // FP16 intermediate was removed; both modes use float temp buffer
        ensure_temp_buffer(static_cast<size_t>(width) * height * channels * sizeof(float));
    } else if (mode == FilterMode::ADAPTIVE) {
        ensure_radius_map_buffer(static_cast<size_t>(width) * height);
    }

    dispatch_u8_kernel(g_bufs.d_in_u8, g_bufs.d_out_u8,
                       g_bufs.d_temp, g_bufs.d_temp_h16, g_bufs.d_radius_map,
                       width, height, channels,
                       radius, sigma_spatial, sigma_color, mode, 0);

    // Check kernel launch errors, then sync
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_output, g_bufs.d_out_u8, n_u8, cudaMemcpyDeviceToHost));
}
