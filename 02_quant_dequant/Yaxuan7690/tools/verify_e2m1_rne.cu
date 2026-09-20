#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include "lpq/formats.hpp"

using namespace lpq;

// exhaustive over all 2^32 fp32 patterns: public encode entry (fast path) vs the
// generic magnitude encoder (already verified against the CPU reference).
__global__ void k_cmp(unsigned long long n, unsigned long long* bad, unsigned int* rec) {
    unsigned long long stride = (unsigned long long)gridDim.x * blockDim.x;
    for (unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x; i < n;
        i += stride) {
        unsigned int a = (unsigned int)i;
        unsigned int slow =
            ((a >> 31) << 3) | f32_bits_to_fmt_mag<FmtE2M1>(a & 0x7FFFFFFFu, 0ull, false);
        unsigned int fast = f32_to_fmt<FmtE2M1>(bits_to_f32(a));
        if (fast != slow) {
            unsigned long long k = atomicAdd(bad, 1ull);
            if (k < 24) {
                rec[3 * k] = a;
                rec[3 * k + 1] = slow;
                rec[3 * k + 2] = fast;
            }
        }
    }
}

int main() {
    unsigned long long* bad;
    unsigned int* rec;
    cudaMalloc(&bad, 8);
    cudaMalloc(&rec, 72 * 4);
    cudaMemset(bad, 0, 8);
    cudaMemset(rec, 0, 72 * 4);
    cudaEvent_t a, b;
    cudaEventCreate(&a);
    cudaEventCreate(&b);
    cudaEventRecord(a);
    k_cmp<<<8192, 256>>>(1ull << 32, bad, rec);
    cudaEventRecord(b);
    cudaDeviceSynchronize();
    unsigned long long h;
    cudaMemcpy(&h, bad, 8, cudaMemcpyDeviceToHost);
    float ms;
    cudaEventElapsedTime(&ms, a, b);
    printf("e2m1 encode: mismatches = %llu over 2^32 patterns (%.0f ms)  err=%s\n", h, ms,
        cudaGetErrorString(cudaGetLastError()));
    if (h) {
        unsigned int r[72];
        cudaMemcpy(r, rec, 72 * 4, cudaMemcpyDeviceToHost);
        for (int i = 0; i < 24; ++i)
            printf(
                "   bits=0x%08X slow=0x%02X fast=0x%02X\n", r[3 * i], r[3 * i + 1], r[3 * i + 2]);
    }
    return 0;
}
