// metrics.cpp - 误差指标与性能统计实现
#include "metrics.h"

#include <algorithm>
#include <cmath>

namespace lowp {

ErrMetrics compute_error(const std::vector<float>& ref, const std::vector<float>& got) {
    ErrMetrics m;
    int64_t n = (int64_t)std::min(ref.size(), got.size());
    if (n == 0)
        return m;
    double sum_abs = 0.0, sum_sq = 0.0, mx = 0.0;
    double sum_ref_abs = 0.0, sum_ref_sq = 0.0;
    int64_t valid = 0;
    for (int64_t i = 0; i < n; i++) {
        double r = (double)ref[(size_t)i];
        double g = (double)got[(size_t)i];
        double d = std::fabs(r - g);
        if (std::isnan(r) || std::isnan(g))
            continue;
        valid++;
        sum_abs += d;
        sum_sq += d * d;
        sum_ref_abs += std::fabs(r);
        sum_ref_sq += r * r;
        mx = std::max(mx, d);
    }
    if (valid == 0)
        return m;
    m.max_abs = mx;
    m.mae = sum_abs / (double)valid;
    m.mse = sum_sq / (double)valid;
    m.rmse = std::sqrt(m.mse);
    // 归一化 L1 误差：与数值尺度无关，便于跨张量/跨格式比较
    m.nmae = (sum_ref_abs > 0.0) ? (sum_abs / sum_ref_abs) : 0.0;
    // 信噪比；含 inf 等病态数据时保持 0 以免产生 nan
    if (sum_sq > 0.0 && std::isfinite(sum_sq) && std::isfinite(sum_ref_sq) && sum_ref_sq > 0.0)
        m.sqnr_db = 10.0 * std::log10(sum_ref_sq / sum_sq);
    return m;
}

MatchResult compare_exact(const std::vector<float>& a, const std::vector<float>& b) {
    MatchResult r;
    int64_t n = (int64_t)std::min(a.size(), b.size());
    for (int64_t i = 0; i < n; i++) {
        double d = std::fabs((double)a[(size_t)i] - (double)b[(size_t)i]);
        bool same = (a[(size_t)i] == b[(size_t)i]) ||
                    (std::isnan(a[(size_t)i]) && std::isnan(b[(size_t)i]));
        if (!same) {
            r.mismatch++;
            if (r.first_bad_index < 0)
                r.first_bad_index = i;
        }
        r.max_diff = std::max(r.max_diff, d);
    }
    return r;
}

double effective_bw_gbps(double bytes_read, double bytes_write, double ms) {
    if (ms <= 0.0)
        return 0.0;
    return (bytes_read + bytes_write) / (ms * 1e-3) / 1e9;
}

} // namespace lowp
