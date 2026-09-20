// 探针 5: 逐元素拆开 AccumFast 的 d^2 补偿项
#include <musa_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <vector>

__device__ __forceinline__ void two_sum_accum(float& hi, float& lo, float x) {
    const float t = hi + x;
    const float z = t - hi;
    lo += (hi - (t - z)) + (x - z);
    hi = t;
}

__global__ static void k_dump(const float* __restrict__ a, const float* __restrict__ b,
                              int64_t n, double* out, float* fout) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    float hi1 = 0.f, lo1 = 0.f, hi0 = 0.f, lo0 = 0.f, hi2 = 0.f, lo2 = 0.f;
    for (int64_t i = 0; i < n; ++i) {
        const float test = b[i], ref = a[i];
        const float dh = test - ref;
        const float zz = dh - test;
        const float dl = (test - (dh - zz)) + ((-ref) - zz);
        const float p1 = dh * dh;
        const float p1e = fmaf(dh, dh, -p1);
        const float cross = 2.0f * (dh * dl);
        const float dlsq = dl * dl;
        const float corr = p1e + (cross + dlsq);
        two_sum_accum(hi0, lo0, fabsf(dh));
        two_sum_accum(hi1, lo1, p1);
        two_sum_accum(hi2, lo2, ref * ref);
        lo0 += (dh < 0.0f) ? -dl : dl;
        lo1 += corr;
        if (i < 12) {
            const double d64 = (double)test - (double)ref;
            out[i * 8 + 0] = (double)dh;
            out[i * 8 + 1] = (double)dl;
            out[i * 8 + 2] = (double)p1;
            out[i * 8 + 3] = (double)p1e;
            out[i * 8 + 4] = (double)cross;
            out[i * 8 + 5] = (double)dlsq;
            out[i * 8 + 6] = d64 * d64;
            out[i * 8 + 7] = (double)hi1 + (double)lo1;
            fout[i * 4 + 0] = p1;
            fout[i * 4 + 1] = corr;
            fout[i * 4 + 2] = p1e;
            fout[i * 4 + 3] = (float)d64;
        }
    }
    fout[48] = hi1; fout[49] = lo1;
    fout[50] = hi2; fout[51] = lo2;
    fout[52] = hi0; fout[53] = lo0;
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
    float *da, *db, *fo; double* dout;
    musaMalloc(&da, n * 4); musaMalloc(&db, n * 4);
    musaMalloc(&dout, 12 * 8 * sizeof(double)); musaMalloc(&fo, 64 * 4);
    musaMemcpy(da, a.data(), n * 4, musaMemcpyHostToDevice);
    musaMemcpy(db, b.data(), n * 4, musaMemcpyHostToDevice);
    k_dump<<<1, 1>>>(da, db, n, dout, fo);
    musaDeviceSynchronize();
    std::vector<double> out(12 * 8); std::vector<float> f(64);
    musaMemcpy(out.data(), dout, 12 * 8 * sizeof(double), musaMemcpyDeviceToHost);
    musaMemcpy(f.data(), fo, 64 * 4, musaMemcpyDeviceToHost);
    printf("%3s %14s %14s %14s %14s %14s %14s %14s\n", "i", "dh", "dl", "p1", "p1e", "2*dh*dl",
           "dl*dl", "d^2(64bit)");
    for (int i = 0; i < 12; ++i)
        printf("%3d %14.6e %14.6e %14.6e %14.6e %14.6e %14.6e %14.6e\n", i, out[i*8], out[i*8+1],
               out[i*8+2], out[i*8+3], out[i*8+4], out[i*8+5], out[i*8+6]);
    printf("\n单线程整体: hi1=%.6e lo1=%.6e  (hi0=%.6e lo0=%.6e, hi2=%.6e lo2=%.6e)\n",
           (double)f[48], (double)f[49], (double)f[52], (double)f[53], (double)f[50], (double)f[51]);
    double s_sq = 0, s_ref = 0, s_abs = 0;
    for (int64_t i = 0; i < n; ++i) {
        const double d = (double)b[i] - (double)a[i];
        s_sq += d * d; s_ref += (double)a[i] * (double)a[i]; s_abs += fabs(d);
    }
    printf("fp64 真值:  d2=%.6e  ref2=%.6e  |d|=%.6e\n", s_sq, s_ref, s_abs);
    printf("fast 结果:  d2=%.6e  ref2=%.6e  |d|=%.6e\n", (double)f[48] + (double)f[49],
           (double)f[50] + (double)f[51], (double)f[52] + (double)f[53]);
    return 0;
}
