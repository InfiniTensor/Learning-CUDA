// 探针 4: 把 stats.cuh 的 AccumFast 原样搬到设备上, 与逐元素 fp64 累加对拍
#include <musa_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>

__device__ __forceinline__ void two_sum_accum(float& hi, float& lo, float x) {
    const float t = hi + x;
    const float z = t - hi;
    lo += (hi - (t - z)) + (x - z);
    hi = t;
}

struct AccFast {
    float hi[4] = {0.f, 0.f, 0.f, 0.f};
    float lo[4] = {0.f, 0.f, 0.f, 0.f};
    float m = 0.0f;
    __device__ __forceinline__ void add(float test, float ref) {
        const float dh = test - ref;
        const float zz = dh - test;
        const float dl = (test - (dh - zz)) + ((-ref) - zz);
        const float adh = fabsf(dh);
        const float dsum = dh + dl;
        const float p1 = dh * dh;
        const float p1e = fmaf(dh, dh, -p1);
        const float p2 = ref * ref;
        const float p2e = fmaf(ref, ref, -p2);
        two_sum_accum(hi[0], lo[0], adh);
        two_sum_accum(hi[1], lo[1], p1);
        two_sum_accum(hi[2], lo[2], p2);
        two_sum_accum(hi[3], lo[3], dh);
        lo[0] += (dsum < 0.0f) ? -dl : dl;
        lo[1] += p1e + (2.0f * (dh * dl) + dl * dl);
        lo[2] += p2e;
        lo[3] += dl;
        m = fmaxf(m, fabsf(dsum));
    }
};

struct Acc64 {
    double s[4] = {0, 0, 0, 0};
    __device__ __forceinline__ void add(float test, float ref) {
        const double r = (double)ref;
        const double d = (double)test - r;
        s[0] += fabs(d); s[1] += d * d; s[2] += r * r; s[3] += d;
    }
};

__global__ static void k_acc(const float* __restrict__ a, const float* __restrict__ b,
                             int64_t n, int64_t chunk, double* out) {
    const int64_t t = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t lo = t * chunk, hi = min(lo + chunk, n);
    AccFast f; Acc64 p;
    for (int64_t i = lo; i < hi; ++i) { f.add(b[i], a[i]); p.add(b[i], a[i]); }
    // out 布局: 每线程 8 个 double [fast0..3, prec0..3]
    for (int j = 0; j < 4; ++j) {
        out[t * 8 + j] = (double)f.hi[j] + (double)f.lo[j];
        out[t * 8 + 4 + j] = p.s[j];
    }
    if (t == (int64_t)gridDim.x * blockDim.x - 1) {
        out[16] = (double)f.hi[2];
        out[17] = (double)f.lo[2];
    }
    if (t == 1) {
        out[18] = (double)f.hi[2];
        out[19] = (double)f.lo[2];
    }
}

int main() {
    const int64_t n = 1 << 20;
    std::vector<float> a(n), b(n);
    uint64_t rng = 0x2545F4914F6CDD1Dull;
    auto unit = [&]() {
        rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
        return (double)(rng >> 11) * (1.0 / 9007199254740992.0);
    };
    for (int64_t i = 0; i < n; ++i) {
        const double mag = ldexp(1.0, (int)(unit() * 48.0) - 24);
        const double sgn = (unit() < 0.5) ? -1.0 : 1.0;
        const double x = sgn * mag * (0.5 + unit());
        a[i] = (float)x;
        b[i] = (float)(x + (unit() - 0.5) * 0.02 * mag);
    }
    const int64_t threads = 4096;
    const int64_t chunk = (n + threads - 1) / threads;
    std::vector<double> out((size_t)threads * 8, 0.0);
    float *da, *db; double* dout;
    musaMalloc(&da, n * 4); musaMalloc(&db, n * 4); musaMalloc(&dout, threads * 64);
    musaMemcpy(da, a.data(), n * 4, musaMemcpyHostToDevice);
    musaMemcpy(db, b.data(), n * 4, musaMemcpyHostToDevice);
    k_acc<<<(int)((threads + 255) / 256), 256>>>(da, db, n, chunk, dout);
    musaError_t e = musaDeviceSynchronize();
    if (e != musaSuccess) { printf("err %s\n", musaGetErrorString(e)); return 1; }
    musaMemcpy(out.data(), dout, threads * 64, musaMemcpyDeviceToHost);

    const char* nm[4] = {"sum|d|", "sum d^2", "sum ref^2", "sum d"};
    for (int j = 0; j < 4; ++j) {
        double tf = 0, tp = 0;
        for (int64_t t = 0; t < threads; ++t) { tf += out[t * 8 + j]; tp += out[t * 8 + 4 + j]; }
        printf("%-10s fast=%.10e precise=%.10e  rel=%.3e\n", nm[j], tf, tp,
               fabs(tf - tp) / std::max(fabs(tp), 1e-300));
    }
    // CPU 双精度真值
    double s_abs = 0, s_sq = 0, s_ref = 0, s_sgn = 0;
    for (int64_t i = 0; i < n; ++i) {
        const double d = (double)b[i] - (double)a[i];
        s_abs += fabs(d); s_sq += d * d; s_ref += (double)a[i] * (double)a[i]; s_sgn += d;
    }
    printf("\nCPU double: |d|=%.10e d2=%.10e ref2=%.10e d=%.10e\n", s_abs, s_sq, s_ref, s_sgn);
    printf("mse fast  rel vs cpu = %.3e\n", fabs(out[1] - s_sq / n) / (s_sq / n));
    return 0;
}
