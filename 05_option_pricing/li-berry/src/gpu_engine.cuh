// gpu_engine.cuh - CUDA 蒙特卡洛引擎
#pragma once

#include "stats.h"
#include "types.h"

// GPU 蒙特卡洛定价（一次运行）。params 需已由 make_params 构造（含 BGK/有效步数）
RawRun run_gpu(const OptionSpec& os, const SimSpec& ss, const EngineOptions& opts,
               const OptionParams& params);

// 转储 RNG 流（--dump-rng）：每条路径输出前 16 个 32 位 uniform（十六进制）
// 用于 CPU/GPU 一致性验证；count 为总 u32 数量（须为 16 的倍数）
void dump_gpu_rng_stream(const SimSpec& ss, int count);
