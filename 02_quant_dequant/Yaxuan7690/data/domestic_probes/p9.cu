// 探针 9: 验证 hi + dh*dh 是否被收缩成 fma(dh, dh, hi)
#include <musa_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <vector>

__global__ static void k(const float* __restrict__ h, const float* __restrict__ d,
                         int64_t n, int* cnt, float* ex) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float hi = h[i], dh = d[i];
    const float p = dh * dh;
    const float a = hi + p;                // 普通写法
    const float b = fmaf(dh, dh, hi);      // 显式融合
    const float p2 = __fmul_rn(dh, dh);
    const float c = hi + p2;               // 显式 round-to-nearest 乘法
    if (a != b) atomicAdd(&cnt[0], 1);
    if (c != b) atomicAdd(&cnt[1], 1);
    if (a != c) atomicAdd(&cnt[2], 1);
    if (i < 8) { ex[i * 4 + 0] = a; ex[i * 4 + 1] = b; ex[i * 4 + 2] = c; ex[i * 4 + 3] = p; }
}

int main() {
    const int64_t n = 1 << 20;
    std::vector<float> hh(n), dd(n);
    srand(4242);
    for (int64_t i = 0; i < n; ++i) {
        const double mag = ldexp(1.0, (rand() % 40) - 20);
        hh[i] = (float)(mag * (0.5 + (rand() % 1000) / 1000.0));
        dd[i] = (float)(mag * 1e-3 * ((rand() % 2000) / 1000.0 - 1.0));
    }
    float *dh, *dx, *ex; int* cnt;
    musaMalloc(&dh, n * 4); musaMalloc(&dx, n * 4); musaMalloc(&ex, 32 * 4); musaMalloc(&cnt, 3 * 4);
    musaMemcpy(dh, hh.data(), n * 4, musaMemcpyHostToDevice);
    musaMemcpy(dx, dd.data(), n * 4, musaMemcpyHostToDevice);
    musaMemset(cnt, 0, 12); musaMemset(ex, 0, 128);
    k<<<(int)((n + 255) / 256), 256>>>(dh, dx, n, cnt, ex);
    musaError_t e = musaDeviceSynchronize();
    if (e != musaSuccess) { printf("err %s\n", musaGetErrorString(e)); return 1; }
    int c[3]; musaMemcpy(c, cnt, 12, musaMemcpyDeviceToHost);
    printf("(hi + dh*dh)  != fma(dh,dh,hi)          : %d / %lld\n", c[0], (long long)n);
    printf("(hi + __fmul_rn(dh,dh)) != fma(dh,dh,hi): %d / %lld\n", c[1], (long long)n);
    printf("(hi + dh*dh)  != (hi + __fmul_rn(dh,dh)): %d / %lld\n", c[2], (long long)n);
    printf("=> 若第 1 行≈0 而第 2 行很多, 说明普通乘法被收缩进加法了\n");
    return 0;
}
