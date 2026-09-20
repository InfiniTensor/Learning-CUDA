// output.h - 定价结果文件与性能日志输出
#pragma once

#include <string>
#include <vector>

#include "types.h"

// 一次引擎运行（GPU 或 CPU）的完整记录
struct RunReport {
    Engine engine = Engine::Gpu;
    PriceResult result;
    Timing timing;
    long long paths_run = 0;        // 实际模拟的路径数
    double paths_per_sec = 0.0;
    bool cpu_rng_fallback = false;  // CPU 引擎在 rng=curand 时降级为 custom
};

// 写定价结果文件（engine=both 时包含两段）
void write_pricing_result(const std::string& path, const OptionSpec& os, const SimSpec& ss,
                          const EngineOptions& opts, const std::vector<RunReport>& runs);

// 写性能日志（含 CPU 基线对比与加速比）。cpu_run 为 CPU 引擎运行记录（可为空）
void write_perf_log(const std::string& path, const OptionSpec& os, const SimSpec& ss,
                    const EngineOptions& opts, const std::vector<RunReport>& runs,
                    const RunReport* cpu_run);
