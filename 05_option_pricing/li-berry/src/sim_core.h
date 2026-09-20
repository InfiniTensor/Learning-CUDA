// sim_core.h - host/device 共享的单条路径模拟核心
// CPU 引擎与 GPU 核共用本文件的同一份数学逻辑，保证"相同/可比采样条件下结果一致"
#pragma once

#include "payoff.h"
#include "rng.h"

// 4 步展开主循环的公共体。gen 提供 next4(float z[4])（4 个独立标准正态）
template <OptionType OPT, typename Gen>
OP_HD void sim_steps(const OptionParams& p, Gen& gen, float& S, float& acc, float& maxS,
                     float& logsum) {
    float z[4];
    int k = 0;
    // 主循环：每轮 4 步（256 步 = 64 次 next4）
    for (; k + 4 <= p.num_steps; k += 4) {
        gen.next4(z);
        advance<OPT>(p.drift_step, p.vol_step, z[0], S, acc, maxS, logsum);
        advance<OPT>(p.drift_step, p.vol_step, z[1], S, acc, maxS, logsum);
        advance<OPT>(p.drift_step, p.vol_step, z[2], S, acc, maxS, logsum);
        advance<OPT>(p.drift_step, p.vol_step, z[3], S, acc, maxS, logsum);
    }
    // 余数循环（num_steps 非 4 的倍数时 < 4 步）
    for (; k < p.num_steps; ++k) {
        gen.next4(z);
        advance<OPT>(p.drift_step, p.vol_step, z[0], S, acc, maxS, logsum);
    }
}

// 单条路径：生成路径并输出（未缩减的）折现收益 y 与折现控制变量 c
// Gen 需提供 init(seed, path) 与 next4(float z[4])
template <OptionType OPT, typename Gen>
OP_HD void sim_one_path(const OptionParams& p, unsigned long long seed,
                        unsigned long long path, float& y, float& c) {
    Gen gen;
    gen.init(seed, path);
    float S = p.s0;
    float acc = 0.0f;     // 亚式：算术累计和（t_1..t_n，不含 t_0）
    float maxS = p.s0;    // 障碍：运行最大值
    float logsum = 0.0f;  // 亚式：log S 累计和（几何平均用）
    sim_steps<OPT>(p, gen, S, acc, maxS, logsum);
    float avg = acc * p.inv_steps;
    float geo_mean = expf(logsum * p.inv_steps);
    y = p.disc * payoff<OPT>(S, avg, maxS, p);
    c = control_variate<OPT>(S, geo_mean, p);
}

// 对偶变量对：同一正态流上锁步模拟 (Z, -Z) 两条路径，样本取两者收益的均值
// y 为对均值（所用估计量样本），c 为对均控制变量，y_raw_leg 为第一条腿的收益
// （用于估计朴素估计量方差 → 方差缩减因子）
template <OptionType OPT, typename Gen>
OP_HD void sim_antithetic_pair(const OptionParams& p, unsigned long long seed,
                               unsigned long long path, float& y, float& c,
                               float& y_raw_leg) {
    Gen gen;
    gen.init(seed, path);
    float Sa = p.s0, Sb = p.s0;
    float accA = 0.0f, accB = 0.0f;
    float maxA = p.s0, maxB = p.s0;
    float logA = 0.0f, logB = 0.0f;
    float z[4];
    int k = 0;
    for (; k + 4 <= p.num_steps; k += 4) {
        gen.next4(z);
        for (int i = 0; i < 4; ++i) {
            advance<OPT>(p.drift_step, p.vol_step, z[i], Sa, accA, maxA, logA);
            advance<OPT>(p.drift_step, p.vol_step, -z[i], Sb, accB, maxB, logB);
        }
    }
    for (; k < p.num_steps; ++k) {
        gen.next4(z);
        advance<OPT>(p.drift_step, p.vol_step, z[0], Sa, accA, maxA, logA);
        advance<OPT>(p.drift_step, p.vol_step, -z[0], Sb, accB, maxB, logB);
    }
    float avgA = accA * p.inv_steps, avgB = accB * p.inv_steps;
    float geoA = expf(logA * p.inv_steps), geoB = expf(logB * p.inv_steps);
    float yA = p.disc * payoff<OPT>(Sa, avgA, maxA, p);
    float yB = p.disc * payoff<OPT>(Sb, avgB, maxB, p);
    float cA = control_variate<OPT>(Sa, geoA, p);
    float cB = control_variate<OPT>(Sb, geoB, p);
    y = 0.5f * (yA + yB);
    c = 0.5f * (cA + cB);
    y_raw_leg = yA;
}
