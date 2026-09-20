// maxflow.h - 最大流算法接口（CPU 参考 + GPU）
#pragma once
#include "types.h"

namespace mf {

// ======================= CPU 参考：Edmonds-Karp 算法 =======================
// 在残量图上从 s 到 t 求最大流（int64 返回）
int64_t cpu_edmonds_karp_maxflow(const ResidualGraph& g, int32_t s, int32_t t);

// ======================= GPU Push-Relabel =======================
class GpuMaxFlow {
public:
    GpuMaxFlow();
    ~GpuMaxFlow();

    // 上传残量图到 GPU（图拓扑 + 原始容量常驻显存）
    void upload(const ResidualGraph& g);

    // 对单个 (s, t) 查询求最大流（在 GPU 内重置残量网络）
    // num_phases 输出本次查询经历的阶段数
    int64_t solve(int32_t s, int32_t t, int* num_phases = nullptr);

    // 释放 GPU 资源
    void release();

private:
    struct Impl;
    Impl* impl_;
};

} // namespace mf
