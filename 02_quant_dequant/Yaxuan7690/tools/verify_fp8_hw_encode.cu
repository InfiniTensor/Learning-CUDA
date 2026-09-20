
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include "lpq/formats.hpp"

using namespace lpq;

static float h_bits_to_f32(unsigned int b) {
    union {
        unsigned int u;
        float f;
    } v;
    v.u = b;
    return v.f;
}

__device__ __forceinline__ unsigned int hw_e4m3x2(float a, float b) {
    unsigned short r;
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;" : "=h"(r) : "f"(a), "f"(b));
    return (unsigned int)r;
}
__device__ __forceinline__ unsigned int hw_e5m2x2(float a, float b) {
    unsigned short r;
    asm("cvt.rn.satfinite.e5m2x2.f32 %0, %2, %1;" : "=h"(r) : "f"(a), "f"(b));
    return (unsigned int)r;
}

// rec[3*i+0]=bits, +1=sw, +2=hw
__global__ void k_cmp8(
    unsigned long long n, int which, unsigned long long* bad, unsigned int* rec) {
    unsigned long long stride = (unsigned long long)gridDim.x * blockDim.x;
    for (unsigned long long idx = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
        idx < n; idx += stride) {
        unsigned int bits = (unsigned int)idx;
        float f = bits_to_f32(bits);
        unsigned int sw, hw;
        if (which == 0) {
            sw = f32_to_fmt<FmtE4M3>(f) & 0xFFu;
            hw = hw_e4m3x2(f, f) & 0xFFu;
        } else {
            sw = f32_to_fmt<FmtE5M2>(f) & 0xFFu;
            hw = hw_e5m2x2(f, f) & 0xFFu;
        }
        if (sw != hw) {
            unsigned long long k = atomicAdd(bad, 1ull);
            if (k < 24) {
                rec[3 * k] = bits;
                rec[3 * k + 1] = sw;
                rec[3 * k + 2] = hw;
            }
        }
    }
}

int main() {
    unsigned long long* bad;
    unsigned int* rec;
    cudaMalloc(&bad, 8);
    cudaMalloc(&rec, 72 * 4);
    unsigned long long n = 1ull << 32;
    cudaEvent_t a, b;
    cudaEventCreate(&a);
    cudaEventCreate(&b);
    for (int which = 0; which < 2; ++which) {
        cudaMemset(bad, 0, 8);
        cudaMemset(rec, 0, 72 * 4);
        cudaEventRecord(a);
        k_cmp8<<<4096, 256>>>(n, which, bad, rec);
        cudaEventRecord(b);
        cudaDeviceSynchronize();
        unsigned long long h;
        cudaMemcpy(&h, bad, 8, cudaMemcpyDeviceToHost);
        float ms;
        cudaEventElapsedTime(&ms, a, b);
        printf("%s : mismatches = %llu   (%.0f ms)\n", which == 0 ? "e4m3" : "e5m2", h, ms);
        if (h) {
            unsigned int r[72];
            cudaMemcpy(r, rec, 72 * 4, cudaMemcpyDeviceToHost);
            for (int i = 0; i < 24; ++i)
                printf("   bits=0x%08X sw=0x%02X hw=0x%02X  float=%.9g\n", r[3 * i], r[3 * i + 1],
                    r[3 * i + 2], (double)h_bits_to_f32(r[3 * i]));
        }
    }
    return 0;
}
