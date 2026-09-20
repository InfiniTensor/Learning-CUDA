
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include "lpq/formats.hpp"
using namespace lpq;

__device__ __forceinline__ unsigned int hw_e4m3_to_h2(unsigned int pair) {
    unsigned int r;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(r) : "h"((unsigned short)pair));
    return r;
}
__device__ __forceinline__ unsigned int bits(float f) {
    return *(unsigned int*)&f;
}
__device__ __forceinline__ unsigned int hw_e5m2_to_h2(unsigned int pair) {
    unsigned int r;
    asm("cvt.rn.f16x2.e5m2x2 %0, %1;" : "=r"(r) : "h"((unsigned short)pair));
    return r;
}

__global__ void k_test(unsigned int* bad, unsigned int* rec, int which) {
    unsigned int p = threadIdx.x;
    if (p >= 65536u)
        return;
    unsigned int h2 = which ? hw_e5m2_to_h2(p) : hw_e4m3_to_h2(p);
    unsigned int lo = h2 & 0xFFFFu, hi = h2 >> 16;
    float lo_f = __half2float(*(__half*)&lo);
    float hi_f = __half2float(*(__half*)&hi);
    unsigned int sw0 =
        which ? bits(fmt_to_f32<FmtE5M2>(p & 0xFFu)) : bits(fmt_to_f32<FmtE4M3>(p & 0xFFu));
    unsigned int sw1 =
        which ? bits(fmt_to_f32<FmtE5M2>(p >> 8)) : bits(fmt_to_f32<FmtE4M3>(p >> 8));
    if (bits(lo_f) != sw0 || bits(hi_f) != sw1) {
        unsigned int k = atomicAdd(bad, 1u);
        if (k < 8)
            rec[k] = p;
    }
}

int main() {
    unsigned int *bad, *rec;
    cudaMalloc(&bad, 4);
    cudaMalloc(&rec, 32);
    cudaMemset(bad, 0, 4);
    cudaMemset(rec, 0, 32);
    for (int which = 0; which < 2; ++which) {
        cudaMemset(bad, 0, 4);
        cudaMemset(rec, 0, 32);
        k_test<<<1, 65536>>>(bad, rec, which);
        cudaDeviceSynchronize();
        unsigned int a;
        cudaMemcpy(&a, bad, 4, cudaMemcpyDeviceToHost);
        printf("%s -> f16x2 (bitwise) mismatches = %u / 65536\n", which ? "e5m2x2" : "e4m3x2", a);
        if (a) {
            unsigned int r[8];
            cudaMemcpy(r, rec, 32, cudaMemcpyDeviceToHost);
            for (int i = 0; i < 8; ++i)
                printf("   pair=0x%04X\n", r[i]);
        }
    }
    return 0;
}
