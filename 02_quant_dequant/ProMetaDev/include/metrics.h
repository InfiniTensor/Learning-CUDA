// metrics.h - 误差指标与性能统计
#pragma once
#include "lowp.h"
#include <cstdint>
#include <string>
#include <vector>

namespace lowp {

// ======================= 误差指标 =======================
struct ErrMetrics {
    double max_abs = 0.0; // 最大绝对误差
    double mae = 0.0;     // 平均绝对误差
    double mse = 0.0;     // 均方误差
    double rmse = 0.0;    // 均方根误差
    double nmae = 0.0;    // 归一化 MAE = Σ|err| / Σ|ref|，无量纲、跨张量可比
    double sqnr_db = 0.0; // 信噪比 10·log10(Σref² / Σerr²)
};

// 在 fp32 域计算 ref 与 got 的误差（长度需一致）
ErrMetrics compute_error(const std::vector<float>& ref, const std::vector<float>& got);

// 逐元素一致性检查（CPU 参考与 GPU 结果应完全逐位一致）
struct MatchResult {
    int64_t mismatch = 0;
    double max_diff = 0.0;
    int64_t first_bad_index = -1;
};
MatchResult compare_exact(const std::vector<float>& a, const std::vector<float>& b);

// ======================= 性能统计 =======================
struct PerfStats {
    double quant_ms = 0.0;          // 量化 kernel 总耗时
    double dequant_ms = 0.0;        // 反量化 kernel 总耗时
    double bytes_in = 0.0;          // 输入张量字节数
    double bytes_quant_out = 0.0;   // 压缩权重字节数(packed + scales + global)
    double bytes_dequant_out = 0.0; // 反量化输出字节数
    double compression_ratio = 0.0; // bytes_in / bytes_quant_out
    double quant_bw_gbps = 0.0;     // 量化有效带宽
    double dequant_bw_gbps = 0.0;   // 反量化有效带宽
};

// 量化有效带宽 = (读入字节 + 写出字节) / 时间
// 读入 = 元素数 * 4 (fp32)
// 写出 = packed + scales
double effective_bw_gbps(double bytes_read, double bytes_write, double ms);

} // namespace lowp
