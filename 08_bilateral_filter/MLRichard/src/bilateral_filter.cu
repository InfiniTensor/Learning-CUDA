#include "bilateral_filter.cuh"
#include <cuda_runtime.h>
#include <cmath>
#include <vector>

#define BLOCK_SIZE 16
#define TILE 16
#define MAX_RADIUS 15
#define MAX_WEIGHT_SIZE ((2 * MAX_RADIUS + 1) * (2 * MAX_RADIUS + 1))

__constant__ float c_spatial_weight[MAX_WEIGHT_SIZE];
__constant__ float c_color_lut[256];

__device__ inline int clamp_int(int value, int low, int high) {
    return value < low ? low : (value > high ? high : value);
}

__device__ inline float color_distance_l1_sq(
    const float* img, int idx_a, int idx_b, int channels)
{
    float sum_abs = 0.f;
    for (int channel = 0; channel < channels; ++channel) {
        float diff = img[idx_a * channels + channel] - img[idx_b * channels + channel];
        sum_abs += fabsf(diff);
    }
    return sum_abs * sum_abs;
}

__device__ inline float pixel_gray(const float* input, int pixel_idx, int channels) {
    float value = 0.f;
    int base = pixel_idx * channels;
    for (int channel = 0; channel < channels; ++channel) {
        value += input[base + channel];
    }
    return value / channels;
}

__global__ void kernel_naive(
    const float* input, float* output,
    int width, int height, int channels,
    int radius, float inv2_sigma_s2, float inv2_sigma_c2)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) {
        return;
    }

    int center = y * width + x;
    float weight_sum = 0.f;
    float value[3] = {0.f, 0.f, 0.f};

    for (int dy = -radius; dy <= radius; ++dy) {
        int ny = y + dy;
        if (ny < 0 || ny >= height) {
            continue;
        }
        for (int dx = -radius; dx <= radius; ++dx) {
            int nx = x + dx;
            if (nx < 0 || nx >= width) {
                continue;
            }

            int neighbor = ny * width + nx;
            float spatial = (dx * dx + dy * dy) * inv2_sigma_s2;
            float color = color_distance_l1_sq(input, center, neighbor, channels) * inv2_sigma_c2;
            float weight = expf(-(spatial + color));

            weight_sum += weight;
            for (int channel = 0; channel < channels; ++channel) {
                value[channel] += weight * input[neighbor * channels + channel];
            }
        }
    }

    for (int channel = 0; channel < channels; ++channel) {
        output[center * channels + channel] = value[channel] / weight_sum;
    }
}

__global__ void kernel_shared(
    const float* __restrict__ input,
    float* output,
    int width, int height, int channels,
    int radius, float inv2_sigma_c2)
{
    extern __shared__ float smem[];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int x = blockIdx.x * TILE + tx;
    int y = blockIdx.y * TILE + ty;
    int tile_w = TILE + 2 * radius;

    for (int j = ty; j < tile_w; j += blockDim.y) {
        for (int i = tx; i < tile_w; i += blockDim.x) {
            int gx = clamp_int(blockIdx.x * TILE - radius + i, 0, width - 1);
            int gy = clamp_int(blockIdx.y * TILE - radius + j, 0, height - 1);
            int src = (gy * width + gx) * channels;
            int dst = (j * tile_w + i) * channels;
            for (int channel = 0; channel < channels; ++channel) {
                smem[dst + channel] = input[src + channel];
            }
        }
    }
    __syncthreads();

    if (x >= width || y >= height) {
        return;
    }

    int lx = tx + radius;
    int ly = ty + radius;
    int center = (ly * tile_w + lx) * channels;
    float weight_sum = 0.f;
    float value[3] = {0.f, 0.f, 0.f};
    int wi = 0;

    for (int dy = -radius; dy <= radius; ++dy) {
        for (int dx = -radius; dx <= radius; ++dx, ++wi) {
            int neighbor = ((ly + dy) * tile_w + (lx + dx)) * channels;
            float sum_abs = 0.f;
            for (int channel = 0; channel < channels; ++channel) {
                sum_abs += fabsf(smem[center + channel] - smem[neighbor + channel]);
            }
            float color = (sum_abs * sum_abs) * inv2_sigma_c2;
            float weight = c_spatial_weight[wi] * expf(-color);
            weight_sum += weight;
            for (int channel = 0; channel < channels; ++channel) {
                value[channel] += weight * smem[neighbor + channel];
            }
        }
    }

    int out = (y * width + x) * channels;
    for (int channel = 0; channel < channels; ++channel) {
        output[out + channel] = value[channel] / weight_sum;
    }
}

__global__ void kernel_sobel(
    const float* __restrict__ input,
    float* gradient,
    int width, int height, int channels)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    if (x < 1 || x >= width - 1 || y < 1 || y >= height - 1) {
        gradient[y * width + x] = 0.f;
        return;
    }

    float g00 = pixel_gray(input, (y - 1) * width + (x - 1), channels);
    float g01 = pixel_gray(input, (y - 1) * width + x, channels);
    float g02 = pixel_gray(input, (y - 1) * width + (x + 1), channels);
    float g10 = pixel_gray(input, y * width + (x - 1), channels);
    float g12 = pixel_gray(input, y * width + (x + 1), channels);
    float g20 = pixel_gray(input, (y + 1) * width + (x - 1), channels);
    float g21 = pixel_gray(input, (y + 1) * width + x, channels);
    float g22 = pixel_gray(input, (y + 1) * width + (x + 1), channels);

    float gx = -g00 + g02 - 2.f * g10 + 2.f * g12 - g20 + g22;
    float gy = -g00 - 2.f * g01 - g02 + g20 + 2.f * g21 + g22;
    gradient[y * width + x] = sqrtf(gx * gx + gy * gy);
}

__global__ void kernel_adaptive(
    const float* __restrict__ input,
    const float* __restrict__ gradient,
    float* output,
    int width, int height, int channels,
    int radius_small, int radius_large,
    float gradient_threshold,
    float inv2_sigma_s2)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) {
        return;
    }

    int radius = gradient[y * width + x] > gradient_threshold ? radius_small : radius_large;
    int center = y * width + x;
    float weight_sum = 0.f;
    float value[3] = {0.f, 0.f, 0.f};

    for (int dy = -radius; dy <= radius; ++dy) {
        int ny = clamp_int(y + dy, 0, height - 1);
        for (int dx = -radius; dx <= radius; ++dx) {
            int nx = clamp_int(x + dx, 0, width - 1);
            int neighbor = ny * width + nx;
            float spatial = expf(-((dx * dx + dy * dy) * inv2_sigma_s2));
            float color_sum_abs = 0.f;
            for (int channel = 0; channel < channels; ++channel) {
                color_sum_abs += fabsf(input[center * channels + channel] -
                                       input[neighbor * channels + channel]);
            }
            int lut_idx = min(static_cast<int>(color_sum_abs), 255);
            float weight = spatial * c_color_lut[lut_idx];
            weight_sum += weight;
            for (int channel = 0; channel < channels; ++channel) {
                value[channel] += weight * input[neighbor * channels + channel];
            }
        }
    }

    for (int channel = 0; channel < channels; ++channel) {
        output[center * channels + channel] = value[channel] / weight_sum;
    }
}

void launch_bilateral_naive(
    const float* d_in, float* d_out,
    int width, int height, int channels,
    const BilateralParams& params)
{
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((width + BLOCK_SIZE - 1) / BLOCK_SIZE,
              (height + BLOCK_SIZE - 1) / BLOCK_SIZE);
    float inv2s2 = 1.f / (2.f * params.sigma_spatial * params.sigma_spatial);
    float inv2c2 = 1.f / (2.f * params.sigma_color * params.sigma_color);
    kernel_naive<<<grid, block>>>(d_in, d_out, width, height, channels,
                                  params.radius, inv2s2, inv2c2);
    cudaDeviceSynchronize();
}

void launch_bilateral_shared(
    const float* d_in, float* d_out,
    int width, int height, int channels,
    const BilateralParams& params)
{
    int radius = params.radius;
    int diameter = 2 * radius + 1;
    std::vector<float> spatial_weights(diameter * diameter);
    float inv2s2 = 1.f / (2.f * params.sigma_spatial * params.sigma_spatial);
    for (int dy = -radius; dy <= radius; ++dy) {
        for (int dx = -radius; dx <= radius; ++dx) {
            spatial_weights[(dy + radius) * diameter + (dx + radius)] =
                expf(-((dx * dx + dy * dy) * inv2s2));
        }
    }
    cudaMemcpyToSymbol(c_spatial_weight, spatial_weights.data(),
                       spatial_weights.size() * sizeof(float));

    float inv2c2 = 1.f / (2.f * params.sigma_color * params.sigma_color);
    int tile_w = TILE + 2 * radius;
    size_t shared_mem = static_cast<size_t>(tile_w) * tile_w * channels * sizeof(float);

    dim3 block(TILE, TILE);
    dim3 grid((width + TILE - 1) / TILE, (height + TILE - 1) / TILE);
    kernel_shared<<<grid, block, shared_mem>>>(d_in, d_out, width, height, channels,
                                               radius, inv2c2);
    cudaDeviceSynchronize();
}

void launch_bilateral_adaptive(
    const float* d_in, float* d_out,
    int width, int height, int channels,
    const BilateralParams& params)
{
    std::vector<float> lut(256);
    float inv2c2 = 1.f / (2.f * params.sigma_color * params.sigma_color);
    for (int value = 0; value < 256; ++value) {
        lut[value] = expf(-((value * value) * inv2c2));
    }
    cudaMemcpyToSymbol(c_color_lut, lut.data(), lut.size() * sizeof(float));

    float* d_gradient = nullptr;
    cudaMalloc(&d_gradient, width * height * sizeof(float));

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((width + BLOCK_SIZE - 1) / BLOCK_SIZE,
              (height + BLOCK_SIZE - 1) / BLOCK_SIZE);
    kernel_sobel<<<grid, block>>>(d_in, d_gradient, width, height, channels);

    float inv2s2 = 1.f / (2.f * params.sigma_spatial * params.sigma_spatial);
    int radius_small = max(1, params.radius - 2);
    int radius_large = params.radius;
    float gradient_threshold = 20.f;

    kernel_adaptive<<<grid, block>>>(d_in, d_gradient, d_out, width, height, channels,
                                     radius_small, radius_large, gradient_threshold, inv2s2);
    cudaDeviceSynchronize();
    cudaFree(d_gradient);
}
