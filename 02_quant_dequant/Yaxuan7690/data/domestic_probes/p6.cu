// 探针 6: AccumFast 逐线程 vs 逐元素 fp64 (输出缓冲完全分离, 不再互相覆盖)
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

__global__ static void k_acc(const float* __restrict__ a, const float* __restrict__ b,
                             int64_t n, int64_t chunk, double* fast, double* prec,
                             double* dump) {
    const int64_t t = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t lo = t * chunk, hi = min(lo + chunk, n);
    AccFast f;
    double p[4] = {0, 0, 0, 0};
    for (int64_t i = lo; i < hi; ++i) {
        f.add(b[i], a[i]);
        const double d = (double)b[i] - (double)a[i];
        p[0] += fabs(d); p[1] += d * d; p[2] += (double)a[i] * (double)a[i]; p[3] += d;
    }
    for (int j = 0; j < 4; ++j) { fast[t * 4 + j] = (double)f.hi[j] + (double)f.lo[j]; prec[t * 4 + j] = p[j]; }
    for (int j = 0; j < 4; ++j) { dump[t * 8 + j] = (double)f.hi[j]; dump[t * 8 + 4 + j] = (double)f.lo[j]; }
}

int main(int argc, char** argv) {
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
    const int64_t threads = (argc > 1) ? atoi(argv[1]) : 4096;
    const int64_t chunk = (n + threads - 1) / threads;
    std::vector<double> fast((size_t)threads * 4), prec((size_t)threads * 4), dump((size_t)threads * 8);
    float *da, *db; double *df, *dp, *dd;
    musaMalloc(&da, n * 4); musaMalloc(&db, n * 4);
    musaMalloc(&df, threads * 32); musaMalloc(&dp, threads * 32); musaMalloc(&dd, threads * 64);
    musaMemcpy(da, a.data(), n * 4, musaMemcpyHostToDevice);
    musaMemcpy(db, b.data(), n * 4, musaMemcpyHostToDevice);
    k_acc<<<(int)((threads + 255) / 256), 256>>>(da, db, n, chunk, df, dp, dd);
    musaError_t e = musaDeviceSynchronize();
    if (e != musaSuccess) { printf("err %s\n", musaGetErrorString(e)); return 1; }
    musaMemcpy(fast.data(), df, threads * 32, musaMemcpyDeviceToHost);
    musaMemcpy(prec.data(), dp, threads * 32, musaMemcpyDeviceToHost);
    musaMemcpy(dump.data(), dd, threads * 64, musaMemcpyDeviceToHost);

    const char* nm[4] = {"sum|d|", "sum d^2", "sum ref^2", "sum d"};
    printf("threads=%lld chunk=%lld\n", (long long)threads, (long long)chunk);
    for (int j = 0; j < 4; ++j) {
        double tf = 0, tp = 0;
        for (int64_t t = 0; t < threads; ++t) { tf += fast[t * 4 + j]; tp += prec[t * 4 + j]; }
        double wr = 0; int64_t wt = -1;
        for (int64_t t = 0; t < threads; ++t) {
            const double r = fabs(fast[t*4+j] - prec[t*4+j]) / std::max(fabs(prec[t*4+j]), 1e-300);
            if (r > wr) { wr = r; wt = t; }
        }
        double mxlo = 0;
        for (int64_t t = 0; t < threads; ++t) {
            const double h = fabs(dump[t*8+j]);
            if (h > 0) mxlo = std::max(mxlo, fabs(dump[t*8+4+j]) / h);
        }
        printf("%-10s fast=%.10e precise=%.10e  rel_total=%.3e | 最差线程 rel=%.3e (t=%lld) | max|lo|/|hi|=%.3e\n",
               nm[j], tf, tp, fabs(tf-tp)/std::max(fabs(tp),1e-300), wr, (long long)wt, mxlo);
    }
    double s_sq = 0;
    for (int64_t i = 0; i < n; ++i) { const double d = (double)b[i] - (double)a[i]; s_sq += d*d; }
    printf("CPU fp64 sum d^2 = %.10e\n", s_sq);
    return 0;
}
