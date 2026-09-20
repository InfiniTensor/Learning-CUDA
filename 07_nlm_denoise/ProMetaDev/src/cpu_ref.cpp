// cpu_ref.cpp - CPU 精确 NLM 参考实现（OpenMP） + OpenCV 参考封装
#include "nlm.h"
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace nlm {

namespace {
double now_ms() {
    using clk = std::chrono::high_resolution_clock;
    return std::chrono::duration<double, std::milli>(clk::now().time_since_epoch()).count();
}
inline int clamp_i(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}
} // namespace

// ============================================================
// CPU 精确 NLM
//   对每个像素 p，遍历搜索窗口内（裁剪到图像内）的所有候选 q，
//   计算 patch 平均平方差 D/|P|，按 w = exp(-max(D/|P| - 2σ²,0)/h²) 加权平均。
//   为了保证 patch 始终完整，patch 读取在边界处按"复制边缘"处理。
//   用 double 累加以提高参考精度。
// ============================================================
Image cpu_nlm(const Image& in, const NLMParams& p, double* out_ms) {
    const int W = in.width, H = in.height, C = in.channels;
    if (W <= 0 || H <= 0 || (C != 1 && C != 3))
        throw std::runtime_error("cpu_nlm: 图像尺寸/通道非法");

    const int rp = p.patch_radius, rs = p.search_radius;
    const int pstep = p.patch_step, sstep = p.search_step;
    // 采样点偏移：0, pstep, 2*pstep, ...（以 patch 中心对齐）
    const int nk = (2 * rp) / pstep + 1;
    // 归一化必须除以"patch 采样点数 × 通道数"：
    // D 累加了 nk²·C 个平方差，若只除以 nk²，RGB 下会把距离放大 C 倍，
    // 使 dd ≈ (C-1)·2σ² 远大于 0 → 除中心外所有候选权重趋近 0 → 输出≈输入。
    const double inv_patch = 1.0 / (double)(nk * nk * C);
    const double h2 = p.h * p.h;
    const double two_sigma2 = 2.0 * p.sigma * p.sigma;

    std::vector<float> f(in.data.begin(), in.data.end());

    Image out;
    out.width = W;
    out.height = H;
    out.channels = C;
    out.data.resize(in.data.size());

    const double t0 = now_ms();

#pragma omp parallel for schedule(dynamic, 1)
    for (int y = 0; y < H; ++y) {
        std::vector<double> num((size_t)C, 0.0);
        std::vector<int> koff((size_t)nk);
        for (int i = 0; i < nk; ++i)
            koff[(size_t)i] = i * pstep; // 0..2rp

        for (int x = 0; x < W; ++x) {
            std::fill(num.begin(), num.end(), 0.0);
            double den = 0.0;

            const int dy_lo = std::max(-rs, -y), dy_hi = std::min(rs, H - 1 - y);
            const int dx_lo = std::max(-rs, -x), dx_hi = std::min(rs, W - 1 - x);

            for (int dy = dy_lo; dy <= dy_hi; dy += sstep) {
                for (int dx = dx_lo; dx <= dx_hi; dx += sstep) {
                    // ---- patch 平均平方差（边界复制） ----
                    double D = 0.0;
                    for (int ky = 0; ky < nk; ++ky) {
                        const int py = clamp_i(y + koff[ky] - rp, 0, H - 1);
                        const int qy = clamp_i(y + dy + koff[ky] - rp, 0, H - 1);
                        const float* rp_row = f.data() + ((size_t)py * W) * C;
                        const float* rq_row = f.data() + ((size_t)qy * W) * C;
                        for (int kx = 0; kx < nk; ++kx) {
                            const int px = clamp_i(x + koff[kx] - rp, 0, W - 1);
                            const int qx = clamp_i(x + dx + koff[kx] - rp, 0, W - 1);
                            for (int c = 0; c < C; ++c) {
                                const double d = (double)rp_row[(size_t)px * C + c] -
                                                 (double)rq_row[(size_t)qx * C + c];
                                D += d * d;
                            }
                        }
                    }
                    // ---- 权重 ----
                    const double dd = std::max(D * inv_patch - two_sigma2, 0.0);
                    const double w = std::exp(-dd / h2);

                    const size_t iq = ((size_t)(y + dy) * W + (x + dx)) * C;
                    for (int c = 0; c < C; ++c)
                        num[(size_t)c] += w * (double)f[iq + c];
                    den += w;
                }
            }

            const size_t io = ((size_t)y * W + x) * C;
            for (int c = 0; c < C; ++c) {
                const double v = (den > 0.0) ? num[(size_t)c] / den : (double)f[io + c];
                out.data[io + c] = (uint8_t)clamp_i((int)std::lround(v), 0, 255);
            }
        }
    }
    if (out_ms)
        *out_ms = now_ms() - t0;
    return out;
}

// ============================================================
// OpenCV 参考（外部对照）
//   注意：OpenCV 的 fastNlMeansDenoising* 内部对 patch 权重做了"快速"近似
//   （按阈值二值化 + 不同的边界策略），因此与上面的精确语义不会逐像素相同。
// ============================================================
Image opencv_nlm(const Image& in, const NLMParams& p, double* out_ms) {
    if (!p.is_exact())
        throw std::runtime_error(
            "opencv_nlm: OpenCV 参考仅用于精确参数（不支持 search_step/patch_step/LUT）");
    cv::Mat m(in.height, in.width, in.channels == 1 ? CV_8UC1 : CV_8UC3,
              const_cast<uint8_t*>(in.data.data()));
    cv::Mat o;
    const int tw = 2 * p.patch_radius + 1;
    const int sw = 2 * p.search_radius + 1;

    const double t0 = now_ms();
    if (in.channels == 1)
        cv::fastNlMeansDenoising(m, o, (float)p.h, tw, sw);
    else
        cv::fastNlMeansDenoisingColored(m, o, (float)p.h, (float)p.h, tw, sw);
    if (out_ms)
        *out_ms = now_ms() - t0;

    Image out;
    out.width = in.width;
    out.height = in.height;
    out.channels = in.channels;
    cv::Mat cont = o.isContinuous() ? o : o.clone();
    out.data.assign(cont.data, cont.data + (size_t)cont.total() * in.channels);
    return out;
}

} // namespace nlm
