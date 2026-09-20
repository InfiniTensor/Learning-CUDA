// =============================================================================
//  reference.cu -- 独立的 CPU 参考实现(查表转换 + 串行流水线)
//
//  该路径刻意避开 formats.hpp 中所有的位运算技巧:
//    * 转换   :枚举格式的全部可表示幅值 -> 二分查找 -> 比较/随机选择
//    * log2   :反复折半/加倍计数(精确,无 log2f 舍入风险)
//    * 流水线 :纯标量串行
//  缩放因子的生成沿用 formats.hpp 中定义的公式常量(公式属于规范,不是实现技巧),
//  以便两端送入转换器的数值逐位相同,使比较结果有意义.
// =============================================================================
#include "lpq/reference.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

#include "lpq/common.hpp"

namespace lpq {
namespace ref {

RefFmt ref_fmt_of(int kind) {
    RefFmt f;
    switch (kind) {
        case 1:  // e5m2
            f.ebits = 5;
            f.mbits = 2;
            f.bias = 15;
            f.has_inf = true;
            f.has_nan = true;
            break;
        case 2:  // e2m1
            f.ebits = 2;
            f.mbits = 1;
            f.bias = 1;
            f.has_inf = false;
            f.has_nan = false;
            break;
        default:  // e4m3
            f.ebits = 4;
            f.mbits = 3;
            f.bias = 7;
            f.has_inf = false;
            f.has_nan = true;
            break;
    }
    return f;
}

int exact_log2_floor(double v) {
    if (!(v > 0.0))
        return INT32_MIN;
    int e = 0;
    while (v >= 2.0) {
        v *= 0.5;
        ++e;
    }
    while (v < 1.0) {
        v *= 2.0;
        --e;
    }
    return e;
}

RefTable build_table(const RefFmt& f) {
    RefTable t;
    t.f = f;
    const uint32_t max_ef = (1u << f.ebits) - 1u;
    const uint32_t mag_mask = (1u << (f.ebits + f.mbits)) - 1u;
    const uint32_t total = (uint32_t)(f.ebits + f.mbits);
    const int emin = 1 - f.bias;

    t.decode.assign((size_t)mag_mask + 1, 0.0);
    t.decode[0] = 0.0;
    // 0 也作为一个候选幅值参与二分,否则「小于最小可表示幅值」的输入会被错误地
    // 舍入到最小值而不是 0 (例如 e2m1 的 0.0076 应量化为 0,而不是 0.5).
    t.vals.push_back(0.0);
    t.codes.push_back(0u);
    for (uint32_t code = 1; code <= mag_mask; ++code) {
        const uint32_t ef = code >> f.mbits;
        const uint32_t mf = code & ((1u << f.mbits) - 1u);
        if (f.has_inf && ef == max_ef) {
            t.decode[code] = std::numeric_limits<double>::quiet_NaN();
            continue;
        }
        if (f.has_nan && !f.has_inf && ef == max_ef && mf == ((1u << f.mbits) - 1u)) {
            t.decode[code] = std::numeric_limits<double>::quiet_NaN();
            continue;
        }
        double v;
        if (ef == 0) {
            v = (double)mf * std::pow(2.0, (double)(emin - f.mbits));
        } else {
            v = (1.0 + (double)mf * std::pow(2.0, -(double)f.mbits)) *
                std::pow(2.0, (double)ef - f.bias);
        }
        t.decode[code] = v;
        t.vals.push_back(v);
        t.codes.push_back(code);
    }
    t.max_code = t.codes.empty() ? 0u : t.codes.back();
    if (f.has_inf) {
        t.inf_code = (max_ef << f.mbits) & mag_mask;
        t.qnan_code = ((max_ef << f.mbits) | (1u << (f.mbits - 1))) & mag_mask;
    } else if (f.has_nan) {
        t.qnan_code = mag_mask;
    } else {
        t.qnan_code = t.max_code;
    }
    (void)total;
    return t;
}

double table_value(const RefTable& t, uint32_t code) {
    const uint32_t mag_mask = (1u << (t.f.ebits + t.f.mbits)) - 1u;
    const uint32_t sign = (code >> (t.f.ebits + t.f.mbits)) & 1u;
    const uint32_t mag = code & mag_mask;
    const double v = t.decode[mag];
    return sign ? -v : v;
}

uint32_t table_quantize(const RefTable& t, float x, bool stochastic, uint64_t rnd) {
    const int total = t.f.ebits + t.f.mbits;
    const uint32_t sign = (std::signbit(x) ? 1u : 0u) << total;
    const double ax = std::fabs((double)x);
    const size_t n = t.vals.size();

    if (std::isnan(x))
        return sign | t.qnan_code;
    if (std::isinf(x))
        return sign | (t.f.has_inf ? t.inf_code : t.max_code);
    if (ax == 0.0)
        return sign;  // 保留 ±0

    if (ax >= t.vals[n - 1])
        return sign | t.max_code;  // 上溢饱和

    // 二分:找到最大的 lo 使 vals[lo] <= ax
    size_t lo = 0, hi = n - 1;
    while (hi - lo > 1) {
        const size_t mid = lo + (hi - lo) / 2;
        if (t.vals[mid] <= ax) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    if (t.vals[lo] == ax)
        return sign | t.codes[lo];

    size_t sel;
    if (!stochastic) {
        const double sum = t.vals[lo] + t.vals[hi];
        const double twice = 2.0 * ax;
        if (twice < sum) {
            sel = lo;
        } else if (twice > sum) {
            sel = hi;
        } else {
            sel = ((t.codes[lo] & 1u) == 0u) ? lo : hi;  // ties-to-even(按编码奇偶)
        }
    } else {
        const double p = (ax - t.vals[lo]) / (t.vals[hi] - t.vals[lo]);
        // 低 32 位即 random_at 的 32bit 哈希(见 include/lpq/common.hpp); 直接由低
        // 32 位构造 [0,1) 均匀分布, 不再依赖 splitmix64 的高 53 位.
        const double u = (double)(uint32_t)rnd * (1.0 / 4294967296.0);  // [0,1)
        sel = (u < p) ? hi : lo;
    }
    return sign | t.codes[sel];
}

namespace {

inline uint32_t mag_mask_of(const RefFmt& f) {
    return (1u << (f.ebits + f.mbits)) - 1u;
}

// 完整编码掩码(含符号位)
inline uint32_t full_mask_of(const RefFmt& f) {
    return (1u << (f.ebits + f.mbits + 1)) - 1u;
}

}  // namespace

CpuQuantized cpu_quantize(const float* x, int64_t numel, const QuantConfig& cfg) {
    CpuQuantized q;
    q.block_size = (cfg.scale_mode == ScaleMode::Tensor) ? numel : (int64_t)cfg.block_size;
    if (q.block_size <= 0)
        q.block_size = numel;
    q.num_blocks = (numel + q.block_size - 1) / q.block_size;
    if (q.num_blocks < 1)
        q.num_blocks = 1;

    const bool stoch = (cfg.rounding == Rounding::Stochastic);

    // ---- 阶段 1:每块 amax ----
    std::vector<float> amax((size_t)q.num_blocks, 0.0f);
    for (int64_t i = 0; i < numel; ++i) {
        const float a = std::fabs(x[i]);
        const size_t b = (size_t)(i / q.block_size);
        if (a > amax[b])
            amax[b] = a;
    }

    q.scales.assign((size_t)q.num_blocks, 0);

    if (cfg.is_mx()) {
        // ---- 阶段 2:共享指数(E8M0) ----
        // 元素格式:e2m1(MXFP4) / e4m3 或 e5m2(MXFP8)
        const int elem_kind = (cfg.format == QuantFormat::MxFP4) ? 2 : (cfg.mxfp8_e5m2 ? 1 : 0);
        const RefFmt ef = ref_fmt_of(elem_kind);
        const RefTable et = build_table(ef);
        const uint32_t max_ef = (1u << ef.ebits) - 1u;
        // e4m3: 8, e5m2: 15, e2m1: 2
        const int emax_elem = (int)(ef.has_inf ? (max_ef - 1u) : max_ef) - ef.bias;

        for (int64_t b = 0; b < q.num_blocks; ++b) {
            int e = -127;
            if (amax[(size_t)b] > 0.0f) {
                e = exact_log2_floor((double)amax[(size_t)b]) - emax_elem;
                e = std::max(-127, std::min(127, e));
            }
            q.scales[(size_t)b] = (uint8_t)(e + 127);
        }

        // ---- 阶段 3:量化 + 打包 ----
        // e2m1(4bit)每字节 2 个元素、偶下标在低半字节; e4m3 / e5m2 每字节 1 个元素
        const bool pack4 = (ef.ebits + ef.mbits) <= 4;
        q.packed.assign((size_t)(pack4 ? ((numel + 1) / 2) : numel), 0);
        for (int64_t i = 0; i < numel; ++i) {
            const uint8_t sc = q.scales[(size_t)(i / q.block_size)];
            const float inv = e8m0_rcp(sc);
            const uint64_t r = stoch ? random_at(cfg.seed, (uint64_t)i) : 0ull;
            const uint32_t code = table_quantize(et, x[i] * inv, stoch, r) & full_mask_of(ef);
            if (!pack4) {
                q.packed[(size_t)i] = (uint8_t)code;
                continue;
            }
            const size_t byte = (size_t)(i >> 1);
            if ((i & 1) == 0)
                q.packed[byte] = (uint8_t)code;
            else
                q.packed[byte] = (uint8_t)((q.packed[byte] & 0x0Fu) | (code << 4));
        }
    } else {
        const RefTable e2m1 = build_table(ref_fmt_of(2));
        const RefTable e4m3 = build_table(ref_fmt_of(0));

        // ---- 阶段 2:FP32 全局缩放 + E4M3 局部缩放 ----
        // max_q 是块内 amax 要映射到的值(6 或 4), max_s 是全局缩放里的块缩放上限
        // (static 规则 448, 动态规则降到 256 给 1.5 倍扩张留余量), 见 formats.hpp.
        const float max_q = (cfg.scale_rule == ScaleRule::Static4) ? 4.0f : kMaxE2M1;
        const float max_s = cfg.scale_rule_adaptive() ? kFourOverSixScaleMax : kMaxE4M3;

        float tensor_amax = 0.0f;
        for (int64_t b = 0; b < q.num_blocks; ++b) {
            tensor_amax = std::max(tensor_amax, amax[(size_t)b]);
        }
        q.global_scale = nvfp4_global_scale(tensor_amax, max_q, max_s);
        q.global_scale_inv = f32_div_rn(1.0f, q.global_scale);

        // 误差准则:0 = abs_max, 1 = mae, 2 = mse;-1 表示静态规则(不做 4/6 选择)
        const int rule =
            cfg.scale_rule_adaptive()
                ? (cfg.scale_rule == ScaleRule::AbsMax ? 0
                                                       : (cfg.scale_rule == ScaleRule::Mae ? 1 : 2))
                : -1;
        const float gf = q.global_scale_inv;
        for (int64_t b = 0; b < q.num_blocks; ++b) {
            const float raw = nvfp4_block_scale_f32(amax[(size_t)b], q.global_scale, max_q);
            const uint32_t c6 = table_quantize(e4m3, raw, false, 0);
            if (rule < 0) {
                q.scales[(size_t)b] = (uint8_t)c6;
                continue;
            }
            // 4/6:块内按下标顺序逐个量化-反量化, 累加次序与设备 kernel 完全一致
            const uint32_t c4 = table_quantize(e4m3, mul_rn(raw, kFourOverSixExpansion), false, 0);
            const float bs6 = (float)table_value(e4m3, c6);
            const float bs4 = (float)table_value(e4m3, c4);
            const int64_t i0 = b * q.block_size;
            const int64_t n = std::min<int64_t>(q.block_size, numel - i0);
            float err6 = 0.0f, err4 = 0.0f;
            for (int64_t k = 0; k < n; ++k) {
                const float xv = x[i0 + k];
                const uint64_t r = stoch ? random_at(cfg.seed, (uint64_t)(i0 + k)) : 0ull;
                const float q6 = (bs6 > 0.0f) ? f32_div_rn(xv * q.global_scale, bs6) : 0.0f;
                const float q4 = (bs4 > 0.0f) ? f32_div_rn(xv * q.global_scale, bs4) : 0.0f;
                const float d6 =
                    mul_rn((float)table_value(
                               e2m1, table_quantize(e2m1, q6, stoch, r) & full_mask_of(e2m1.f)),
                        mul_rn(bs6, gf));
                const float d4 =
                    mul_rn((float)table_value(
                               e2m1, table_quantize(e2m1, q4, stoch, r) & full_mask_of(e2m1.f)),
                        mul_rn(bs4, gf));
                if (rule == 0) {
                    err6 = std::max(err6, std::fabs(d6 - xv));
                    err4 = std::max(err4, std::fabs(d4 - xv));
                } else if (rule == 1) {
                    err6 = err6 + std::fabs(d6 - xv);
                    err4 = err4 + std::fabs(d4 - xv);
                } else {
                    err6 = fma_rn(d6 - xv, d6 - xv, err6);
                    err4 = fma_rn(d4 - xv, d4 - xv, err4);
                }
            }
            q.scales[(size_t)b] = (uint8_t)((err4 < err6) ? c4 : c6);  // 平手保留 6 候选
        }

        // ---- 阶段 3:量化 + 打包(每字节 2 个元素,偶下标在低半字节) ----
        q.packed.assign((size_t)((numel + 1) / 2), 0);
        for (int64_t i = 0; i < numel; ++i) {
            const uint8_t sc = q.scales[(size_t)(i / q.block_size)];
            const float bs = (float)table_value(e4m3, sc);
            const float qv = (bs > 0.0f) ? f32_div_rn(x[i] * q.global_scale, bs) : 0.0f;
            const uint64_t r = stoch ? random_at(cfg.seed, (uint64_t)i) : 0ull;
            const uint32_t code = table_quantize(e2m1, qv, stoch, r) & full_mask_of(e2m1.f);
            const size_t byte = (size_t)(i >> 1);
            if ((i & 1) == 0) {
                q.packed[byte] = (uint8_t)code;
            } else {
                q.packed[byte] = (uint8_t)((q.packed[byte] & 0x0Fu) | (code << 4));
            }
        }
    }
    return q;
}

std::vector<float> cpu_dequantize(const CpuQuantized& q, int64_t numel, const QuantConfig& cfg) {
    std::vector<float> out((size_t)numel, 0.0f);
    if (cfg.is_mx()) {
        const int elem_kind = (cfg.format == QuantFormat::MxFP4) ? 2 : (cfg.mxfp8_e5m2 ? 1 : 0);
        const RefFmt ef = ref_fmt_of(elem_kind);
        const RefTable et = build_table(ef);
        const bool pack4 = (ef.ebits + ef.mbits) <= 4;
        for (int64_t i = 0; i < numel; ++i) {
            const uint8_t sc = q.scales[(size_t)(i / q.block_size)];
            const float s = e8m0_to_f32(sc);
            // 4bit 元素: 偶下标取低半字节, 奇下标取高半字节
            const uint32_t code =
                pack4 ? (uint32_t)((q.packed[(size_t)(i >> 1)] >> (4 * (uint32_t)(i & 1))) & 0xFu)
                      : (uint32_t)q.packed[(size_t)i];
            out[(size_t)i] = (float)table_value(et, code) * s;
        }
    } else {
        const RefTable e2m1 = build_table(ref_fmt_of(2));
        const RefTable e4m3 = build_table(ref_fmt_of(0));
        const float gf = f32_div_rn(1.0f, q.global_scale);
        for (int64_t i = 0; i < numel; ++i) {
            const uint8_t byte = q.packed[(size_t)(i >> 1)];
            const uint32_t code = (i & 1) ? ((byte >> 4) & 0xFu) : (byte & 0xFu);
            const uint8_t sc = q.scales[(size_t)(i / q.block_size)];
            // 注意乘法结合顺序必须与 GPU kernel 一致: q * (bs * gf).
            // 若写成 (q * bs) * gf, 在 fp32 输出下会有 ULP 级差异.
            const float bs_gf = (float)table_value(e4m3, sc) * gf;
            out[(size_t)i] = (float)table_value(e2m1, code) * bs_gf;
        }
    }
    return out;
}

}  // namespace ref
}  // namespace lpq