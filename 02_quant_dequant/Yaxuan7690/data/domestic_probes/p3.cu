// 探针 3: 设备端 fmaf / TwoSum / 精确拆平方 是否与 IEEE 语义一致
#include <musa_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <vector>

__global__ static void k_fma(const float* a, const float* b, float* r, float* c1, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float p = a[i] * b[i];
    r[i] = fmaf(a[i], b[i], -p);   // 真 FMA 时应给出精确残差
    c1[i] = a[i] * b[i] + 1.0f;    // 若被收缩成 FMA, 应与下面 fmaf 逐位相同
}
__global__ static void k_fma2(const float* a, const float* b, float* c2, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    c2[i] = fmaf(a[i], b[i], 1.0f);
}
__global__ static void k_twosum(const float* h, const float* x, float* t_o, float* lo_o, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float hi = h[i], xx = x[i];
    const float t = hi + xx;
    const float z = t - hi;
    t_o[i] = t;
    lo_o[i] = (hi - (t - z)) + (xx - z);
}
__global__ static void k_sq(const float* d, float* p_o, float* pe_o, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float v = d[i];
    const float p = v * v;
    p_o[i] = p;
    pe_o[i] = fmaf(v, v, -p);
}

int main() {
    const int n = 1 << 20;
    std::vector<float> a(n), b(n), h(n), x(n), d(n);
    srand(999);
    for (int i = 0; i < n; ++i) {
        const double mag = ldexp(1.0, (rand() % 48) - 24);
        const double sgn = (rand() & 1) ? -1.0 : 1.0;
        a[i] = (float)(sgn * mag * (0.5 + (rand() % 1000) / 1000.0));
        const double mag2 = ldexp(1.0, (rand() % 48) - 24);
        b[i] = (float)(((rand() & 1) ? -1.0 : 1.0) * mag2 * (0.5 + (rand() % 1000) / 1000.0));
        h[i] = 1e6f * (float)(rand() % 1000) / 1000.0f;
        x[i] = 1e-3f * (float)(rand() % 1000) / 1000.0f;
        d[i] = a[i] * 1e-5f;
    }
    float *da, *db, *dh, *dx, *dd, *r, *c1, *c2, *t_o, *lo_o, *p_o, *pe_o;
    musaMalloc(&da, n * 4); musaMalloc(&db, n * 4); musaMalloc(&dh, n * 4);
    musaMalloc(&dx, n * 4); musaMalloc(&dd, n * 4);
    musaMalloc(&r, n * 4); musaMalloc(&c1, n * 4); musaMalloc(&c2, n * 4);
    musaMalloc(&t_o, n * 4); musaMalloc(&lo_o, n * 4); musaMalloc(&p_o, n * 4); musaMalloc(&pe_o, n * 4);
    musaMemcpy(da, a.data(), n * 4, musaMemcpyHostToDevice);
    musaMemcpy(db, b.data(), n * 4, musaMemcpyHostToDevice);
    musaMemcpy(dh, h.data(), n * 4, musaMemcpyHostToDevice);
    musaMemcpy(dx, x.data(), n * 4, musaMemcpyHostToDevice);
    musaMemcpy(dd, d.data(), n * 4, musaMemcpyHostToDevice);
    k_fma<<<(n + 255) / 256, 256>>>(da, db, r, c1, n);
    k_fma2<<<(n + 255) / 256, 256>>>(da, db, c2, n);
    k_twosum<<<(n + 255) / 256, 256>>>(dh, dx, t_o, lo_o, n);
    k_sq<<<(n + 255) / 256, 256>>>(dd, p_o, pe_o, n);
    musaError_t e = musaDeviceSynchronize();
    if (e != musaSuccess) { printf("kernel err %s\n", musaGetErrorString(e)); return 1; }

    std::vector<float> hr(n), hc1(n), hc2(n), ht(n), hlo(n), hp(n), hpe(n);
    musaMemcpy(hr.data(), r, n * 4, musaMemcpyDeviceToHost);
    musaMemcpy(hc1.data(), c1, n * 4, musaMemcpyDeviceToHost);
    musaMemcpy(hc2.data(), c2, n * 4, musaMemcpyDeviceToHost);
    musaMemcpy(ht.data(), t_o, n * 4, musaMemcpyDeviceToHost);
    musaMemcpy(hlo.data(), lo_o, n * 4, musaMemcpyDeviceToHost);
    musaMemcpy(hp.data(), p_o, n * 4, musaMemcpyDeviceToHost);
    musaMemcpy(hpe.data(), pe_o, n * 4, musaMemcpyDeviceToHost);

    long resid_nonzero = 0;
    for (int i = 0; i < n; ++i) if (hr[i] != 0.0f) ++resid_nonzero;
    printf("[fmaf 真融合?] fmaf(a,b,-fl(a*b)) != 0 的样本: %ld / %d (%.1f%%)\n",
           resid_nonzero, n, 100.0 * resid_nonzero / n);

    long contr = 0;
    for (int i = 0; i < n; ++i) if (hc1[i] != hc2[i]) ++contr;
    printf("[a*b+1 是否收缩成 fma] 与 fmaf(a,b,1) 不同的样本: %ld / %d\n", contr, n);

    long ts_bad = 0, ts_bad_lo = 0;
    for (int i = 0; i < n; ++i) {
        const double exact = (double)h[i] + (double)x[i];
        if (ht[i] != (float)exact) ++ts_bad;
        if ((double)ht[i] + (double)hlo[i] != exact) ++ts_bad_lo;
    }
    printf("[TwoSum] t != fl(h+x): %ld / %d ;  (t+lo) != 精确和: %ld / %d\n", ts_bad, n,
           ts_bad_lo, n);

    long sq_bad = 0, sq_bad_pe0 = 0;
    for (int i = 0; i < n; ++i) {
        const double exact = (double)d[i] * (double)d[i];
        if ((double)hp[i] + (double)hpe[i] != exact) ++sq_bad;
        if (hpe[i] == 0.0f && (double)hp[i] != exact) ++sq_bad_pe0;
    }
    printf("[精确拆平方] (p+pe) != 精确平方: %ld / %d ;  其中 pe==0 且 p 不精确: %ld / %d\n",
           sq_bad, n, sq_bad_pe0, n);
    return 0;
}
