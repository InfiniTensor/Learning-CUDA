// cpu_engine.h - CPU 单线程蒙特卡洛基线引擎
#pragma once

#include "stats.h"
#include "types.h"

// CPU 蒙特卡洛引擎。与 GPU 核共用 sim_core.h 的同一份数学逻辑；
// 始终使用 custom RNG（rng=curand 时降级并给出警告，报告中说明）
RawRun run_cpu(const OptionSpec& os, const SimSpec& ss, const EngineOptions& opts,
               const OptionParams& params);
