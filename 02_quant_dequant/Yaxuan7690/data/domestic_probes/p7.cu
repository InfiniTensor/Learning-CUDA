// 探针 7: 把 AccumFast 的 hi / lo 两条链分开, 各自与 fp64 对照
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
                         int64_t chunk, double* out) {
    const int64_t t = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t lo_i = t * chunk, hi_i = min(lo_i + chunk, n);
    float hi1 = 0.f, lo1 = 0.f, hi2 = 0.f, lo2 = 0.f;
    double P1 = 0, P1E = 0, CROSS = 0, P2 = 0, P2E = 0;
    for (int64_t i = lo_i; i < hi_i; ++i) {
        const float test = b[i], ref = a[i];
        const float dh = test - ref;
        const float zz = dh - test;
        const float dl = (test - (dh - zz)) + ((-ref) - zz);
        const float p1 = dh * dh;
        const float p1e = fmaf(dh, dh, -p1);
        const float p2 = ref * ref;
        const float p2e = fmaf(ref, ref, -p2);
        two_sum_accum(hi1, lo1, p1);
        two_sum_accum(hi2, lo2, p2);
        lo1 += p1e + (2.0f * (dh * dl) + dl * dl);
        lo2 += p2e;
        P1 += (double)p1; P1E += (double)p1e; CROSS += (double)(2.0f * (dh * dl) + dl * dl);
        P2 += (double)p2; P2E += (double)p2e;
    }
    double* o = out + t * 10;
    o[0] = (double)hi1; o[1] = (double)lo1; o[2] = P1; o[3] = P1E + CROSS;
    o[4] = (double)hi2; o[5] = (double)lo2; o[6] = P2; o[7] = P2E;
    o[8] = fabsf(hi1); o[9] = 0.0;
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
    std::vector<double> out((size_t)threads * 10);
    float *da, *db; double* dout;
    musaMalloc(&da, n * 4); musaMalloc(&db, n * 4); musaMalloc(&dout, threads * 80);
    musaMemcpy(da, a.data(), n * 4, musaMemcpyHostToDevice);
    musaMemcpy(db, b.data(), n * 4, musaMemcpyHostToDevice);
    k<<<(int)((threads + 255) / 256), 256>>>(da, db, n, chunk, dout);
    musaError_t e = musaDeviceSynchronize();
    if (e != musaSuccess) { printf("err %s\n", musaGetErrorString(e)); return 1; }
    musaMemcpy(out.data(), dout, threads * 80, musaMemcpyDeviceToHost);

    double H1=0,L1=0,S1=0,S1E=0,H2=0,L2=0,S2=0,S2E=0;
    for (int64_t t = 0; t < threads; ++t) {
        H1 += out[t*10+0]; L1 += out[t*10+1]; S1 += out[t*10+2]; S1E += out[t*10+3];
        H2 += out[t*10+4]; L2 += out[t*10+5]; S2 += out[t*10+6]; S2E += out[t*10+7];
    }
    auto r = [](double x, double y){ return fabs(x-y)/std::max(fabs(y),1e-300); };
    printf("=== d^2 链 ===\n");
    printf("  hi[1] 累加器 = %.10e   fp64 Σp1      = %.10e   rel=%.3e\n", H1, S1, r(H1,S1));
    printf("  lo[1] 累加器 = %.10e   fp64 Σ修正项  = %.10e   rel=%.3e\n", L1, S1E, r(L1,S1E));
    printf("  hi+lo        = %.10e   fp64 总       = %.10e   rel=%.3e\n", H1+L1, S1+S1E, r(H1+L1,S1+S1E));
    printf("  lo/hi = %.3e\n", L1/H1);
    printf("=== ref^2 链 ===\n");
    printf("  hi[2] 累加器 = %.10e   fp64 Σp2      = %.10e   rel=%.3e\n", H2, S2, r(H2,S2));
    printf("  lo[2] 累加器 = %.10e   fp64 Σp2e     = %.10e   rel=%.3e\n", L2, S2E, r(L2,S2E));
    printf("  hi+lo        = %.10e   fp64 总       = %.10e   rel=%.3e\n", H2+L2, S2+S2E, r(H2+L2,S2+S2E));
    printf("  lo/hi = %.3e\n", L2/H2);
    return 0;
}
