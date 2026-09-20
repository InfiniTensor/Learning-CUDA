// 用墙上时钟复核 musaEventElapsedTime: 同一段纯读 kernel 各测一遍
#include <musa_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <chrono>

__global__ static void k_read(const char* __restrict__ s, char* __restrict__ d, long long n) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    unsigned int a;
    __builtin_memcpy(&a, s + i * 4, 4);
    unsigned int b;
    __builtin_memcpy(&b, s + i * 4 + 4, 4);
    unsigned int c;
    __builtin_memcpy(&c, s + i * 4 + 8, 4);
    unsigned int e;
    __builtin_memcpy(&e, s + i * 4 + 12, 4);
    unsigned long long acc = ((unsigned long long)a << 32) ^ b ^ ((unsigned long long)c << 16) ^ e;
    acc = (acc >> 32) ^ (acc & 0xFFFFFFFFull);
    if (acc == 0x0F0F0F0Full) *reinterpret_cast<volatile unsigned char*>(d) = 1;
}

int main() {
    const long long numel = 67108864LL;
    const long long n = numel / 4;
    char *ds = 0, *dd = 0;
    if (musaMalloc((void**)&ds, (size_t)numel * 4) != musaSuccess) { printf("alloc fail\n"); return 1; }
    if (musaMalloc((void**)&dd, (size_t)numel * 2) != musaSuccess) { printf("alloc fail2\n"); return 1; }
    musaMemset(ds, 0x5A, (size_t)numel * 4);
    musaMemset(dd, 0x00, (size_t)numel * 2);
    const int threads = 256;
    const unsigned blocks = (unsigned)((n + threads - 1) / threads);
    for (int i = 0; i < 5; ++i) k_read<<<blocks, threads>>>(ds, dd, n);
    musaDeviceSynchronize();

    // 事件计时 (30 次取最小)
    musaEvent_t e0, e1;
    musaEventCreate(&e0); musaEventCreate(&e1);
    float best = 1e30f;
    for (int i = 0; i < 30; ++i) {
        musaEventRecord(e0);
        k_read<<<blocks, threads>>>(ds, dd, n);
        musaEventRecord(e1);
        musaEventSynchronize(e1);
        float ms = 0.f;
        musaEventElapsedTime(&ms, e0, e1);
        if (ms < best) best = ms;
    }
    // 墙上时钟 (整批计时, 含 200 次 kernel)
    const int iters = 200;
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) k_read<<<blocks, threads>>>(ds, dd, n);
    musaDeviceSynchronize();
    auto t1 = std::chrono::steady_clock::now();
    const double wall_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;

    const double mb = (double)numel * 4.0 / 1e6;
    printf("事件: %.3f ms/kernel -> %.1f GB/s\n", best, mb / best);
    printf("墙上: %.3f ms/kernel -> %.1f GB/s   (event/wall = %.3f)\n", wall_ms, mb / wall_ms,
           best / wall_ms);
    return 0;
}
