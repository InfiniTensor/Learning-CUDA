// 探测 v2: 过滤 NaN/Inf/0/次正规输入, 给出逐位 ULP 直方图
#include <musa_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <vector>

__device__ static float k1(float x) { return 1.0f / x; }
__device__ static float k2(float x) { return __frcp_rn(x); }
__device__ static float k3(float x) { return __fdiv_rn(1.0f, x); }
__device__ static float k4(float x) { return (float)(1.0 / (double)x); }
__device__ static float k5(float a, float b) { return a / b; }
__device__ static float k6(float a, float b) { return __fdiv_rn(a, b); }
__device__ static float k7(float a, float b) { return a * b; }
__device__ static float k8(float a, float b) { return __fmul_rn(a, b); }
__device__ static float k9(float a, float b, float c) { return a * b + c; }
__device__ static float k10(float a, float b, float c) { return fmaf(a, b, c); }

__global__ static void kern(const float* x, const float* y, float* o, int n, int which) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    switch (which) {
        case 1: o[i] = k1(x[i]); break;
        case 2: o[i] = k2(x[i]); break;
        case 3: o[i] = k3(x[i]); break;
        case 4: o[i] = k4(x[i]); break;
        case 5: o[i] = k5(y[i], x[i]); break;
        case 6: o[i] = k6(y[i], x[i]); break;
        case 7: o[i] = k7(y[i], x[i]); break;
        case 8: o[i] = k8(y[i], x[i]); break;
        case 9: o[i] = k9(y[i], x[i], 1.0f); break;
        case 10: o[i] = k10(y[i], x[i], 1.0f); break;
    }
}

static int diff_ulp(float a, float b) {
    uint32_t ua, ub;
    memcpy(&ua, &a, 4); memcpy(&ub, &b, 4);
    if (ua == ub) return 0;
    if ((ua >> 31) != (ub >> 31)) return 1 << 20;
    return (int)((ua > ub) ? (ua - ub) : (ub - ua));
}

static void run(const char* name, int which, const float* dx, const float* dy, int n,
                const std::vector<float>& gold) {
    float* d; musaMalloc(&d, sizeof(float) * n);
    kern<<<(n + 255) / 256, 256>>>(dx, dy, d, n, which);
    musaDeviceSynchronize();
    std::vector<float> h(n);
    musaMemcpy(h.data(), d, sizeof(float) * n, musaMemcpyDeviceToHost);
    musaFree(d);
    long exact = 0, u1 = 0, u2 = 0, ubad = 0, spec = 0;
    for (int i = 0; i < n; ++i) {
        int dd = diff_ulp(h[i], gold[i]);
        if (dd == 0) ++exact;
        else if (dd == 1) ++u1;
        else if (dd == 2) ++u2;
        else if (dd >= (1 << 20)) ++spec;
        else ++ubad;
    }
    printf("%-9s exact=%8ld 1ulp=%7ld 2ulp=%6ld >2ulp=%5ld special=%5ld\n",
           name, exact, u1, u2, ubad, spec);
}

int main() {
    const int n = 1 << 22;
    std::vector<float> hx(n), hy(n);
    srand(12345);
    int k = 0;
    while (k < n) {
        uint32_t u = ((uint32_t)rand() << 16) ^ (uint32_t)rand();
        uint32_t v = ((uint32_t)rand() << 16) ^ (uint32_t)rand();
        float x, y; memcpy(&x, &u, 4); memcpy(&y, &v, 4);
        if (!std::isfinite(x) || x == 0.0f || std::fabs(x) < 1e-30f) continue;
        if (!std::isfinite(y)) continue;
        hx[k] = x; hy[k] = y; ++k;
    }
    std::vector<float> hr(n), hd(n), hm(n), hf(n);
    for (int i = 0; i < n; ++i) {
        hr[i] = 1.0f / hx[i];
        hd[i] = hy[i] / hx[i];
        hm[i] = hy[i] * hx[i];
        hf[i] = fmaf(hy[i], hx[i], 1.0f);
    }
    float *dx, *dy;
    musaMalloc(&dx, sizeof(float) * n); musaMalloc(&dy, sizeof(float) * n);
    musaMemcpy(dx, hx.data(), sizeof(float) * n, musaMemcpyHostToDevice);
    musaMemcpy(dy, hy.data(), sizeof(float) * n, musaMemcpyHostToDevice);
    printf("valid samples = %d (finite, |x|>=1e-30, y finite)\n", n);
    printf("--- 1/x ---\n");
    run("1/x", 1, dx, dy, n, hr);
    run("rcp_rn", 2, dx, dy, n, hr);
    run("fdiv_rn1", 3, dx, dy, n, hr);
    run("dbl", 4, dx, dy, n, hr);
    printf("--- y/x ---\n");
    run("a/b", 5, dx, dy, n, hd);
    run("fdiv_rn2", 6, dx, dy, n, hd);
    printf("--- y*x ---\n");
    run("a*b", 7, dx, dy, n, hm);
    run("mul_rn", 8, dx, dy, n, hm);
    printf("--- y*x+1 ---\n");
    run("a*b+1", 9, dx, dy, n, hf);
    run("fma", 10, dx, dy, n, hf);
    return 0;
}
