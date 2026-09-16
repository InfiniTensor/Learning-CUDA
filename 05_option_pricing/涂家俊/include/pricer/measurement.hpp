#pragma once

// 统一重复运行、预热和中位数计时的测量接口。

#include <cstdint>
#include <functional>
#include <optional>

#include "pricer/cpu_pricer.hpp"
#include "pricer/cuda_pricer.hpp"

namespace pricer {

// 允许测量逻辑复用真实定价器或测试替身。
using CpuPriceFunction = std::function<CpuPricingRun(const OptionParams &,
                                                     const SimulationParams &)>;
using GpuPriceFunction = std::function<CudaPricingRun(
    const OptionParams &, const SimulationParams &, int)>;
using AnalyzeFunction =
    std::function<PricingResult(const RawMoments &, std::optional<double>)>;

// 对多次 CPU 运行取中位数后的报告数据。
struct CpuMeasurement {
    PricingResult result;
    double total_runtime_ms;
    double compute_runtime_ms;
};

// 对多次 GPU 运行取中位数后的报告数据与设备信息。
struct GpuMeasurement {
    PricingResult result;
    double total_runtime_ms;
    double compute_runtime_ms;
    std::optional<double> rng_ms;
    double reduction_ms;
    std::uint64_t actual_batch_size;
    std::uint64_t grid_size;
    CudaDeviceInfo device;
};

// 运行 CPU 基线；每次重复都包含统计分析，避免计时口径不一致。
CpuMeasurement
measure_cpu(const OptionParams &option, const SimulationParams &simulation,
            std::uint32_t repetitions, std::optional<double> reference_price,
            const CpuPriceFunction &price, const AnalyzeFunction &analyze);

// 先执行 GPU 预热，再测量正式重复运行。
GpuMeasurement measure_gpu(const OptionParams &option,
                           const SimulationParams &simulation, int device,
                           std::uint32_t warmup, std::uint32_t repetitions,
                           std::optional<double> reference_price,
                           const GpuPriceFunction &price,
                           const AnalyzeFunction &analyze);

} // namespace pricer
