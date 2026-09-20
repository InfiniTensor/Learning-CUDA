// 探针 8: 修正项分别累加到 普通 fp32 / lo 分量 两条路, 看谁把修正项丢了
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

__global__ static void k(const float* __restrict__ a, const float* __restrict__ b, int64_t n,
                         int64_t chunk, int64_t variant, double* out) {
    const int64_t t = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t i0 = t * chunk, i1 = min(i0 + chunk, n);
    float hi1 = 0.f, lo1 = 0.f;
    float plain = 0.f;          // 修正项单独放一个普通 fp32
    float chi = 0.f, clo = 0.f; // 修正项走补偿累加
    double P1 = 0, P1E = 0;
    for (int64_t i = i0; i < i1; ++i) {
        const float test = b[i], ref = a[i];
        const float dh = test - ref;
        const float zz = dh - test;
        const float dl = (test - (dh - zz)) + ((-ref) - zz);
        const float p1 = dh * dh;
        const float p1e = fmaf(dh, dh, -p1);
        const float corr = p1e + (2.0f * (dh * dl) + dl * dl);
        two_sum_accum(hi1, lo1, p1);
        if (variant == 0) lo1 += corr;             // 原样
        if (variant == 1) plain += corr;           // 单独普通累加
        if (variant == 2) two_sum_accum(chi, clo, corr);  // 修正项也补偿累加
        P1 += (double)p1; P1E += (double)corr;
    }
    double* o = out + t * 8;
    o[0] = (double)hi1; o[1] = (double)lo1; o[2] = (double)plain;
    o[3] = (double)chi; o[4] = (double)clo; o[5] = P1; o[6] = P1E; o[7] = 0;
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
    std::vector<double> out((size_t)threads * 8);
    float *da, *db; double* dout;
    musaMalloc(&da, n * 4); musaMalloc(&db, n * 4); musaMalloc(&dout, threads * 64);
    musaMemcpy(da, a.data(), n * 4, musaMemcpyHostToDevice);
    musaMemcpy(db, b.data(), n * 4, musaMemcpyHostToDevice);
    for (int64_t variant = 0; variant < 3; ++variant) {
        musaMemset(dout, 0, threads * 64);
        k<<<(int)((threads + 255) / 256), 256>>>(da, db, n, chunk, variant, dout);
        musaDeviceSynchronize();
        musaMemcpy(out.data(), dout, threads * 64, musaMemcpyDeviceToHost);
        double H = 0, L = 0, P = 0, CH = 0, CL = 0, S1 = 0, S1E = 0;
        for (int64_t t = 0; t < threads; ++t) {
            H += out[t*8+0]; L += out[t*8+1]; P += out[t*8+2];
            CH += out[t*8+3]; CL += out[t*8+4]; S1 += out[t*8+5]; S1E += out[t*8+6];
        }
        printf("variant=%lld\n", (long long)variant);
        printf("  hi1=%.8e lo1=%.8e plain=%.8e chi=%.8e clo=%.8e\n", H, L, P, CH, CL);
        printf("  fp64 Σp1=%.8e  fp64 Σ修正=%.8e\n", S1, S1E);
        printf("  hi+lo           = %.10e  (rel vs Σp1+Σ修正 = %.3e)\n", H + L,
               fabs((H + L) - (S1 + S1E)) / fabs(S1 + S1E));
        if (variant == 1)
            printf("  plain 单独累加修正 = %.8e  (fp64 应为 %.8e)\n", P, S1E);
        if (variant == 2)
            printf("  chi+clo 补偿累加修正 = %.8e  (fp64 应为 %.8e)\n", CH + CL, S1E);
    }
    return 0;
}
