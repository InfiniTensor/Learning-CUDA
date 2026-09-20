// reference.cpp - CPU 标量参考实现
//
// 该实现定义量化的“标准语义”，GPU kernel 必须与其逐位一致。
// 采用与 kernel 完全相同的缩放因子推导、舍入规则与打包布局。
#include "quant.h"

#include <algorithm>
#include <chrono>
#include <cmath>

namespace lowp {

static inline int64_t ceil_div_i64(int64_t a, int64_t b) {
    return (a + b - 1) / b;
}

QuantResult run_cpu(const std::vector<float>& in, int64_t n, const Config& cfg) {
    QuantResult qr;
    const bool is_nvfp4 = (cfg.format == LowFormat::NVFP4);

    // tensor 模式下整张量视为一个 block
    int bs = (cfg.scale_mode == ScaleMode::TENSOR) ? (int)n : cfg.block_size;
    if (bs <= 0)
        bs = 1;
    int64_t nb = ceil_div_i64(n, bs);
    qr.block_size = bs;
    qr.num_blocks = nb;

    int64_t packed_bytes = is_nvfp4 ? ceil_div_i64(n, 2) : n;
    qr.packed.assign((size_t)packed_bytes, 0);
    qr.scales.assign((size_t)nb, 0);
    qr.dequant.assign((size_t)n, 0.0f);

    // 全张量 amax（NVFP4 全局缩放 / tensor 模式共用）
    float gmax = 0.0f;
    for (int64_t i = 0; i < n; i++)
        gmax = std::max(gmax, std::fabs(in[(size_t)i]));

    float gscale = 1.0f;
    if (is_nvfp4)
        gscale = nvfp4_global_scale(gmax);
    qr.global_scale = gscale;

    const int rmode = (cfg.round == RoundMode::STOCHASTIC) ? 1 : 0;

    const auto t_begin = std::chrono::high_resolution_clock::now();

    for (int64_t b = 0; b < nb; b++) {
        int64_t lo = b * bs;
        int64_t hi = std::min(n, lo + bs);

        float amax = 0.0f;
        for (int64_t i = lo; i < hi; i++)
            amax = std::max(amax, std::fabs(in[(size_t)i]));

        float scale_val;
        if (!is_nvfp4) {
            // MXFP8：E8M0 共享指数缩放
            int e = mxfp8_scale_exp(amax, elem_max_val(cfg.elem));
            qr.scales[(size_t)b] = e8m0_from_exp(e);
            scale_val = ldexpf(1.0f, e);
        } else {
            // NVFP4：block 局部 E4M3 缩放 + 全局 FP32 缩放
            float target = amax / NVFP4_ELEM_MAX;
            uint8_t sc = f32_to_e4m3(target / gscale, 0, 0.0f); // 缩放因子采用最近舍入
            qr.scales[(size_t)b] = sc;
            scale_val = e4m3_to_f32(sc) * gscale;
        }
        float inv = (scale_val > 0.0f) ? (1.0f / scale_val) : 0.0f;

        for (int64_t i = lo; i < hi; i++) {
            float x = in[(size_t)i];
            float rnd = (rmode == 1) ? rng_uniform(cfg.seed, (uint64_t)i) : 0.0f;
            uint8_t code;
            float dq;

            if (!is_nvfp4) {
                if (cfg.elem == ElemFmt::E4M3) {
                    code = f32_to_e4m3(x * inv, rmode, rnd);
                    dq = e4m3_to_f32(code) * scale_val;
                } else {
                    code = f32_to_e5m2(x * inv, rmode, rnd);
                    dq = e5m2_to_f32(code) * scale_val;
                }
                qr.packed[(size_t)i] = code;
            } else {
                code = f32_to_e2m1(x * inv, rmode, rnd);
                dq = e2m1_to_f32(code) * scale_val;
                int64_t byte = i >> 1;
                if ((i & 1) == 0)
                    qr.packed[(size_t)byte] |= (uint8_t)(code & 0x0Fu);
                else
                    qr.packed[(size_t)byte] |= (uint8_t)((code & 0x0Fu) << 4);
            }
            // 与 GPU 输出保持一致：按 out_dtype 做一次舍入
            qr.dequant[(size_t)i] = round_to_out_dtype(dq, cfg.out_dtype);
        }
    }
    const auto t_end = std::chrono::high_resolution_clock::now();
    qr.cpu_ms = std::chrono::duration<double, std::milli>(t_end - t_begin).count();
    return qr;
}

std::vector<float> host_dequant_fp32(const std::vector<uint8_t>& packed,
                                     const std::vector<uint8_t>& scales, float gscale, int64_t n,
                                     int64_t bs, bool is_nvfp4, ElemFmt elem) {
    std::vector<float> out((size_t)std::max<int64_t>(n, 0), 0.0f);
    if (bs <= 0)
        bs = 1;
    for (int64_t i = 0; i < n; i++) {
        const int64_t blk = i / bs;
        float sc;
        uint8_t code;
        if (!is_nvfp4) {
            sc = e8m0_to_f32(scales[(size_t)blk]);
            code = packed[(size_t)i];
            out[(size_t)i] = ((elem == ElemFmt::E4M3) ? e4m3_to_f32(code) : e5m2_to_f32(code)) * sc;
        } else {
            sc = e4m3_to_f32(scales[(size_t)blk]) * gscale;
            uint8_t b = packed[(size_t)(i >> 1)];
            code = (i & 1) ? (uint8_t)(b >> 4) : (uint8_t)(b & 0x0Fu);
            out[(size_t)i] = e2m1_to_f32(code) * sc;
        }
    }
    return out;
}

} // namespace lowp
