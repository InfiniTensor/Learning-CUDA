#include "nlm_cpu_ref.h"

#include <cmath>
#include <vector>

namespace {

inline int ClampI(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

} // namespace

void NlmDenoiseCpuRef(const ImageU8& src, ImageU8* dst, const NlmParams& params) {
    const int w = src.width, h = src.height, c = src.channels;
    const size_t plane = (size_t)w * h;
    const int pr = params.patch_radius, sr = params.search_radius;

    // interleaved u8 -> planar f32
    std::vector<float> in(plane * c);
    for (size_t i = 0; i < plane; ++i)
        for (int ch = 0; ch < c; ++ch)
            in[(size_t)ch * plane + i] = (float)src.data[i * c + ch];

    std::vector<float> out(plane * c);

    const float inv_h2 = 1.0f / (params.h * params.h);
    const float n_patch = (float)((2 * pr + 1) * (2 * pr + 1) * c);
    const float two_sigma2_n = 2.0f * params.sigma * params.sigma * n_patch;

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            float sum_w = 0.0f;
            float sum[3] = {0.0f, 0.0f, 0.0f};

            for (int dy = -sr; dy <= sr; ++dy) {
                for (int dx = -sr; dx <= sr; ++dx) {
                    const int qx = x + dx, qy = y + dy;
                    // patch L2 距离（跨通道求和）
                    float dist = 0.0f;
                    for (int py = -pr; py <= pr; ++py) {
                        const int ay = ClampI(y + py, 0, h - 1);
                        const int by = ClampI(qy + py, 0, h - 1);
                        for (int px = -pr; px <= pr; ++px) {
                            const int ax = ClampI(x + px, 0, w - 1);
                            const int bx = ClampI(qx + px, 0, w - 1);
                            for (int ch = 0; ch < c; ++ch) {
                                const float d = in[(size_t)ch * plane + (size_t)ay * w + ax] -
                                                in[(size_t)ch * plane + (size_t)by * w + bx];
                                dist += d * d;
                            }
                        }
                    }
                    const float dsub = dist - two_sigma2_n;
                    const float wgt = std::exp(-(dsub > 0.0f ? dsub : 0.0f) * inv_h2);
                    sum_w += wgt;
                    const int cqx = ClampI(qx, 0, w - 1);
                    const int cqy = ClampI(qy, 0, h - 1);
                    for (int ch = 0; ch < c; ++ch)
                        sum[ch] += wgt * in[(size_t)ch * plane + (size_t)cqy * w + cqx];
                }
            }

            const float inv = 1.0f / sum_w;  // sum_w >= 1，无除零
            for (int ch = 0; ch < c; ++ch)
                out[(size_t)ch * plane + (size_t)y * w + x] = sum[ch] * inv;
        }
    }

    // planar f32 -> interleaved u8（round + clamp）
    dst->width = w;
    dst->height = h;
    dst->channels = c;
    dst->data.resize(plane * c);
    for (size_t i = 0; i < plane; ++i) {
        for (int ch = 0; ch < c; ++ch) {
            const float v = out[(size_t)ch * plane + i];
            int q = (int)(v + 0.5f);
            q = q < 0 ? 0 : (q > 255 ? 255 : q);
            dst->data[i * c + ch] = (uint8_t)q;
        }
    }
}
