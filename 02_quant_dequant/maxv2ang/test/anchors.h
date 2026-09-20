// ============================================================================
// 锚点自测（格式层）
// ============================================================================
// 把关键格式事实钉死成断言：次正规数系数、特殊值码点、舍入平局、打包字节序等
// 一旦有人改坏，这里立刻失败。由 test/test_all.cpp 调用。
//
// 注意：这是**测试代码**，不属于库。库（include/ + src/）里没有测试。
// ============================================================================

#pragma once

#include "quant_format.h"
#include "quant_io.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>


inline int g_qc_checks = 0;
inline void qc_check(const std::string &name, bool cond) {
    g_qc_checks++;
    std::cout << "  [" << (cond ? "PASS" : "FAIL") << "] " << name << "\n";
    if (!cond) {
        std::cerr << "测试失败: " << name << "\n";
        std::exit(1);
    }
}

inline int run_anchor_tests() {
    std::cout << "== 1. 解码锚点（与核实手册一致）==\n";
    qc_check("E4M3 0x00=0", fmt_decode(FMT_E4M3, 0x00) == 0.0);
    qc_check("E4M3 0x38=1.0", fmt_decode(FMT_E4M3, 0x38) == 1.0);
    qc_check("E4M3 0x01=2^-9 (subnormal!)", fmt_decode(FMT_E4M3, 0x01) == 0x1p-9);
    qc_check("E4M3 0x07=7×2^-9", fmt_decode(FMT_E4M3, 0x07) == 7 * 0x1p-9);
    qc_check("E4M3 0x7E=448 (exp=1111 不是特殊值)", fmt_decode(FMT_E4M3, 0x7E) == 448.0);
    qc_check("E4M3 0x7F=NaN (OCP 无 Inf)", std::isnan(fmt_decode(FMT_E4M3, 0x7F)));
    qc_check("E4M3 0xFF=NaN", std::isnan(fmt_decode(FMT_E4M3, 0xFF)));
    qc_check("E5M2 0x7B=57344", fmt_decode(FMT_E5M2, 0x7B) == 57344.0);
    qc_check("E5M2 0x7C=+Inf", std::isinf(fmt_decode(FMT_E5M2, 0x7C)));
    {
        std::vector<double> s;
        for (int c = 0; c < 16; c++)
            s.push_back(fmt_decode(FMT_E2M1, (uint8_t)c));
        std::sort(s.begin(), s.end());
        s.erase(std::unique(s.begin(), s.end(), [](double a, double b) { return a == b; }),
                s.end());
        qc_check("E2M1 值集 = {0,±0.5,±1,±1.5,±2,±3,±4,±6}",
                 s.size() == 15 && s[0] == -6 && s[7] == 0 && s[14] == 6);
    }

    std::cout << "== 2. 编码（RNE 平局取偶 / 饱和）==\n";
    qc_check("E4M3 编码 1.0 → 0x38", fmt_encode(FMT_E4M3, 1.0, MODE_NEAREST, 0) == 0x38);
    qc_check("E4M3 编码 -2.0 → 0xC0", fmt_encode(FMT_E4M3, -2.0, MODE_NEAREST, 0) == 0xC0);
    qc_check("E4M3 平局 1.0625 → 1.0 (0x38 偶)",
             fmt_encode(FMT_E4M3, 1.0625, MODE_NEAREST, 0) == 0x38);
    qc_check("E4M3 饱和 449 → 448",
             fmt_decode(FMT_E4M3, fmt_encode(FMT_E4M3, 449.0, MODE_NEAREST, 0)) == 448.0);
    qc_check("E2M1 平局 2.5 → 2 (偶码点)",
             fmt_decode(FMT_E2M1, fmt_encode(FMT_E2M1, 2.5, MODE_NEAREST, 0)) == 2.0);
    qc_check("E2M1 平局 5.0 → 4 (偶码点)",
             fmt_decode(FMT_E2M1, fmt_encode(FMT_E2M1, 5.0, MODE_NEAREST, 0)) == 4.0);
    qc_check("E2M1 饱和 100 → 6",
             fmt_decode(FMT_E2M1, fmt_encode(FMT_E2M1, 100.0, MODE_NEAREST, 0)) == 6.0);

    std::cout << "== 3. 4bit 打包（与手算一致）==\n";
    {
        std::vector<uint8_t> b = pack_nibbles({0x2, 0xA});
        qc_check("codes(2,A) → 字节 0xA2 (低4bit=偶数下标)", b[0] == 0xA2);
        std::vector<uint8_t> u = unpack_nibbles(b, 2);
        qc_check("解包 0xA2 → (2, A)", u[0] == 0x2 && u[1] == 0xA);
        qc_check("奇数长度补零后截断", unpack_nibbles(pack_nibbles({7}), 1)[0] == 7);
    }

    std::cout << "== 4. MXFP8 手算对照 ==\n";
    {
        std::vector<float> w;
        for (int r = 0; r < 4; r++)
            for (float v : {8.0f, 4.0f, 2.0f, 1.0f, 0.5f, 0.25f, 0.125f, 0.0625f})
                w.push_back(v);
        Quantized q = quantize_mxfp8_host(w, 32, FMT_E4M3, "block", MODE_NEAREST, 0);
        qc_check("E8M0 字节 = 127 (scale=1.0)", q.scale_bytes[0] == 127);
        std::vector<double> dq = dequantize_mxfp8_host(q);
        bool exact = dq.size() == w.size() && std::equal(w.begin(), w.end(), dq.begin(),
                                                         [](double a, double b) { return a == b; });
        qc_check("全 2 的幂 → round-trip 精确还原", exact);
        std::vector<float> w2(32, 15.0f);
        std::vector<double> dq2 =
            dequantize_mxfp8_host(quantize_mxfp8_host(w2, 32, FMT_E4M3, "block", MODE_NEAREST, 0));
        qc_check("amax=15 → scale=1, 精确还原",
                 std::equal(w2.begin(), w2.end(), dq2.begin(),
                            [](double a, double b) { return a == b; }));
    }

    std::cout << "== 5. NVFP4 手算对照 ==\n";
    {
        // amax=448×6=2688 → gs=1.0；块 amax=2688 → s_block=448(精确 E4M3)
        std::vector<float> w3;
        for (int r = 0; r < 4; r++)
            for (double g :
                 {0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0})
                w3.push_back((float)(448.0 * g));
        Quantized q = quantize_nvfp4_host(w3, 16, "block", MODE_NEAREST, 0);
        qc_check("global_scale = 1.0", q.global_scale == 1.0f);
        bool all7e = true;
        for (uint8_t c : q.scale_bytes)
            all7e = all7e && c == 0x7E;
        qc_check("块缩放码点 = 0x7E (448)", all7e);
        std::vector<double> dq = dequantize_nvfp4_host(q);
        qc_check("round-trip 精确还原", std::equal(w3.begin(), w3.end(), dq.begin(),
                                                   [](double a, double b) { return a == b; }));
        qc_check("首打包字节 = 0x10 (lo=code0, hi=code1)", q.packed[0] == 0x10);
    }

    std::cout << "== 6. 随机舍入无偏性（E2M1，10^6 样本）==\n";
    {
        double sr_err = 0, rne_err = 0;
        const int N = 1000000;
        for (int i = 0; i < N; i++) {
            uint64_t st = elem_rng(7, i);
            double x = rand01(&st) * 6.0;
            uint64_t st2 = elem_rng(7, i + 1);
            sr_err += fmt_decode(FMT_E2M1, fmt_encode(FMT_E2M1, x, MODE_STOCHASTIC, st2)) - x;
            rne_err += fmt_decode(FMT_E2M1, fmt_encode(FMT_E2M1, x, MODE_NEAREST, 0)) - x;
        }
        qc_check("SR 平均误差 ≈ 0", fabs(sr_err / N) < 0.005);
        qc_check("RNE 平均误差 ≈ 0", fabs(rne_err / N) < 0.005);
    }

    std::cout << "\n格式锚点 " << g_qc_checks << " 项 PASS ✔\n";
    return 0;
}

