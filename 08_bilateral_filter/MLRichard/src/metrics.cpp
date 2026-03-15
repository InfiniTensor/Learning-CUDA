#include "metrics.h"
#include <cmath>
#include <numeric>

float compute_mae(const float* a, const float* b, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) sum += std::fabs(a[i] - b[i]);
    return static_cast<float>(sum / n);
}

float compute_psnr(const float* a, const float* b, int n) {
    double mse = 0.0;
    for (int i = 0; i < n; i++) {
        double d = a[i] - b[i];
        mse += d * d;
    }
    mse /= n;
    if (mse < 1e-10) return 100.0f;
    return static_cast<float>(10.0 * std::log10(255.0 * 255.0 / mse));
}
