// 探测 MUSA (MTT S4000 / mp_22) 上 fp32 除法/倒数的舍入行为
// 判据: 与 x86 host 的 IEEE 正确舍入结果逐位比较
#include <musa_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <vector>

__device__ static float k1(float x) { return 1.0f / x; }
__device__ static float k2(float x) { return __frcp_rn(x); }
__device__ static float k3(float x) { return __fdiv_rn(1.0f, x); }
__device__ static float k4(float x) { return (float)(1.0 / (double)x); }
__device__ static float k5(float a, float b) { return a / b; }
__device__ static float k6(float a, float b) { return __fdiv_rn(a, b); }

__global__ static void kern(const float* x, const float* y, float* o1, float* o2,
                            float* o3, float* o4, float* o5, float* o6, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    o1[i] = k1(x[i]);
    o2[i] = k2(x[i]);
    o3[i] = k3(x[i]);
    o4[i] = k4(x[i]);
    o5[i] = k5(y[i], x[i]);
    o6[i] = k6(y[i], x[i]);
}

static int diff_ulp(float a, float b) {
    uint32_t ua, ub;
    memcpy(&ua, &a, 4); memcpy(&ub, &b, 4);
    if (ua == ub) return 0;
    if ((ua >> 31) != (ub >> 31)) return 2000000000;
    return (int)((ua > ub) ? (ua - ub) : (ub - ua));
}

static void check(const char* name, const std::vector<float>& host, const float* dev,
                  int n) {
    std::vector<float> h(n);
    musaMemcpy(h.data(), dev, sizeof(float) * n, musaMemcpyDeviceToHost);
    long bad = 0; int worst = 0;
    for (int i = 0; i < n; ++i) {
        int d = diff_ulp(h[i], host[i]);
        if (d) { ++bad; if (worst < d) worst = d; }
    }
    printf("%-9s mismatches=%8ld / %8d (%7.3f%%)  worst_diff=%d\n", name, bad, n,
           100.0 * (double)bad / (double)n, worst);
}

int main() {
    musaDeviceProp prop;
    musaGetDeviceProperties(&prop, 0);
    printf("device=%s cc=%d.%d sm=%d  MUSA_ARCH=%d\n", prop.name, prop.major, prop.minor,
           prop.multiProcessorCount,
#ifdef __MUSA_ARCH__
           __MUSA_ARCH__
#else
           -1
#endif
    );

    const int n = 1 << 22;
    std::vector<float> hx(n), hy(n);
    srand(12345);
    for (int i = 0; i < n; ++i) {
        uint32_t u;
        switch (i % 4) {
            case 0: u = ((uint32_t)rand() << 16) ^ (uint32_t)rand(); break;
            case 1: u = 0x3F000000u + (uint32_t)(rand() & 0xFFFFFF); break;
            case 2: u = 0x40000000u + (uint32_t)(rand() & 0xFFFFFF); break;
            default: u = ((uint32_t)rand() << 16) ^ (uint32_t)rand();
        }
        memcpy(&hx[i], &u, 4);
        uint32_t v = ((uint32_t)rand() << 16) ^ (uint32_t)rand();
        memcpy(&hy[i], &v, 4);
    }

    std::vector<float> hr(n), hd(n);
    for (int i = 0; i < n; ++i) {
        hr[i] = 1.0f / hx[i];   // host: x86 SSE, IEEE 正确舍入
        hd[i] = hy[i] / hx[i];
    }

    float *dx, *dy, *d1, *d2, *d3, *d4, *d5, *d6;
    musaMalloc(&dx, sizeof(float) * n); musaMalloc(&dy, sizeof(float) * n);
    musaMalloc(&d1, sizeof(float) * n); musaMalloc(&d2, sizeof(float) * n);
    musaMalloc(&d3, sizeof(float) * n); musaMalloc(&d4, sizeof(float) * n);
    musaMalloc(&d5, sizeof(float) * n); musaMalloc(&d6, sizeof(float) * n);
    musaMemcpy(dx, hx.data(), sizeof(float) * n, musaMemcpyHostToDevice);
    musaMemcpy(dy, hy.data(), sizeof(float) * n, musaMemcpyHostToDevice);
    kern<<<(n + 255) / 256, 256>>>(dx, dy, d1, d2, d3, d4, d5, d6, n);
    musaError_t e = musaDeviceSynchronize();
    if (e != musaSuccess) { printf("kernel err: %s\n", musaGetErrorString(e)); return 1; }

    printf("--- vs host 1.0f/x ---\n");
    check("1/x", hr, d1, n);
    check("rcp_rn", hr, d2, n);
    check("fdiv_rn", hr, d3, n);
    check("dbl", hr, d4, n);
    printf("--- vs host y/x ---\n");
    check("a/b", hd, d5, n);
    check("fdiv_rn2", hd, d6, n);
    return 0;
}
