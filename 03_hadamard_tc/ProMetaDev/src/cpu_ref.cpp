// cpu_ref.cpp - CPU 参考实现
//
// 提供两种互相独立的参考：
//   1) cpu_matmul_ref：直接按定义 y = scale * H_n @ x（O(n^2)），
//      其中 H[i][j] = (-1)^popcount(i & j)（Sylvester 构造），
//      不依赖任何快速算法，用于验证 FWHT 本身的正确性。
//   2) cpu_fht：标准蝶形快速 Walsh-Hadamard 变换（O(n log n)）。
// 两者应给出相同结果，再与 GPU kernel 对照。
#include "hadamard.h"
#include "fp8.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace hd {

namespace {
inline int log2_pow2(int n) {
    int s = 0;
    while ((1 << s) < n)
        s++;
    return s;
}
} // namespace

void cpu_fht(const Tensor& in, Tensor& out, float scale) {
    out.rows = in.rows;
    out.cols = in.cols;
    out.dtype = in.dtype;
    out.data = in.data;
    const int n = in.cols;
    if (n <= 0)
        return;
    const int stages = log2_pow2(n);
    if ((1 << stages) != n) {
        std::fprintf(stderr, "[cpu_fht] cols=%d 不是 2 的幂\n", n);
        return;
    }
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int32_t r = 0; r < in.rows; r++) {
        float* row = out.data.data() + (int64_t)r * n;
        // 标准蝶形 FWHT，产生自然（Sylvester）序的 H_n @ x
        for (int s = 1; s < n; s <<= 1) {
            for (int i = 0; i < n; i += (s << 1)) {
                for (int j = 0; j < s; j++) {
                    float a = row[i + j];
                    float b = row[i + j + s];
                    row[i + j] = a + b;
                    row[i + j + s] = a - b;
                }
            }
        }
        if (scale != 1.0f)
            for (int i = 0; i < n; i++)
                row[i] *= scale;
    }
}

void cpu_matmul_ref(const Tensor& in, Tensor& out, float scale) {
    out.rows = in.rows;
    out.cols = in.cols;
    out.dtype = in.dtype;
    out.data.assign(in.data.size(), 0.0f);
    const int n = in.cols;
    const int stages = log2_pow2(n);
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 8)
#endif
    for (int32_t r = 0; r < in.rows; r++) {
        const float* x = in.data.data() + (int64_t)r * n;
        float* y = out.data.data() + (int64_t)r * n;
        for (int i = 0; i < n; i++) {
            double acc = 0.0;
            for (int j = 0; j < n; j++) {
                // H[i][j] = (-1)^popcount(i & j)
                int sign = (__builtin_popcount((unsigned)(i & j)) & 1) ? -1 : 1;
                acc += sign * (double)x[j];
            }
            y[i] = (float)(acc * (double)scale);
        }
    }
    (void)stages;
}

void cpu_dequant(const QuantResult& qr, int32_t rows, int32_t cols, int32_t block_size,
                 QuantFormat fmt, Tensor& out) {
    out.rows = rows;
    out.cols = cols;
    out.dtype = Dtype::FP32;
    out.data.assign((size_t)((int64_t)rows * cols), 0.0f);
    if (block_size <= 0)
        block_size = 32;
    const bool e5m2 = (fmt == QuantFormat::FP8_E5M2);
    for (int64_t i = 0; i < (int64_t)rows * cols; i++) {
        int64_t blk = i / block_size;
        float sc = e8m0_to_f32(qr.scales[(size_t)blk]);
        uint8_t c = qr.qdata[(size_t)i];
        out.data[(size_t)i] = (e5m2 ? e5m2_to_f32(c) : e4m3_to_f32(c)) * sc;
    }
}

double bandwidth_gbps(double bytes, double ms) {
    if (ms <= 0.0)
        return 0.0;
    return bytes / (ms * 1e-3) / 1e9;
}

} // namespace hd
