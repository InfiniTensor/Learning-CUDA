// ============================================================================
// 测试输入生成
// ============================================================================
// 按分布生成测试矩阵并落盘，供 test/test_all.cpp 使用。
// 这是**测试工具**，不属于库。
// ============================================================================

#pragma once

#include "quant_io.h"

#include <cstdint>
#include <random>
#include <string>
#include <vector>

// ---- 三种分布的数值生成 ----

inline std::vector<float> gen_dist(const std::string &dist, size_t n, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::vector<float> v(n);
    if (dist == "random") {
        std::uniform_real_distribution<double> ud(0.0, 1.0);
        for (auto &x : v)
            x = (float)ud(rng);
    } else if (dist == "normal") {
        std::normal_distribution<double> nd(0.0, 1.0);
        for (auto &x : v)
            x = (float)nd(rng);
    } else { // outlier: 正态 + 0.5% 元素 ×100
        std::normal_distribution<double> nd(0.0, 1.0);
        std::uniform_real_distribution<double> ud(0.0, 1.0);
        for (auto &x : v) {
            double d = nd(rng);
            if (ud(rng) < 0.005)
                d *= 100.0;
            x = (float)d;
        }
    }
    return v;
}

// 生成 rows×cols 的矩阵并写盘（dtype: fp32 / fp16 / bf16）
inline void gen_input_file(const std::string &path, const std::string &dist, int rows, int cols,
                           const std::string &dtype = "fp32", uint64_t seed = 42) {
    std::vector<float> m = gen_dist(dist, (size_t)rows * cols, seed);
    write_tensor(path, rows, cols, dtype, m);
}
