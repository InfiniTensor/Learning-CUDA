// 执行预热和重复测量，并以中位数降低偶发波动的影响。
#include "pricer/measurement.hpp"

#include <chrono>
#include <stdexcept>
#include <utility>
#include <vector>

#include "pricer/output.hpp"

namespace pricer {
namespace {

// 同一固定 seed 的重复正式运行必须产生相同统计量。
void require_same_moments(const RawMoments &left, const RawMoments &right) {
    if (left.count != right.count || left.sum != right.sum ||
        left.sum_squares != right.sum_squares) {
        throw std::runtime_error(
            "fixed config and seed did not reproduce identical moments");
    }
}

} // namespace

CpuMeasurement
// CPU 没有预热阶段；每次测量同时产出可比较的统计结果。
measure_cpu(const OptionParams &option, const SimulationParams &simulation,
            std::uint32_t repetitions, std::optional<double> reference_price,
            const CpuPriceFunction &price, const AnalyzeFunction &analyze) {
    if (repetitions == 0U) {
        throw std::invalid_argument("CPU measurement requires repetitions");
    }
    std::vector<double> totals;
    std::vector<double> computes;
    std::optional<RawMoments> moments;
    std::optional<PricingResult> result;
    for (std::uint32_t repetition = 0U; repetition < repetitions;
         ++repetition) {
        const auto start = std::chrono::steady_clock::now();
        const auto run = price(option, simulation);
        if (moments) {
            require_same_moments(*moments, run.moments);
        } else {
            moments = run.moments;
        }
        auto analyzed = analyze(run.moments, reference_price);
        const auto stop = std::chrono::steady_clock::now();
        if (!result) {
            result = std::move(analyzed);
        }
        totals.push_back(
            std::chrono::duration<double, std::milli>(stop - start).count());
        computes.push_back(run.compute_runtime_ms);
    }
    return {std::move(*result), median(std::move(totals)),
            median(std::move(computes))};
}

// GPU 预热不计入正式中位数，以隔离上下文初始化的偶发成本。
GpuMeasurement measure_gpu(const OptionParams &option,
                           const SimulationParams &simulation, int device,
                           std::uint32_t warmup, std::uint32_t repetitions,
                           std::optional<double> reference_price,
                           const GpuPriceFunction &price,
                           const AnalyzeFunction &analyze) {
    if (repetitions == 0U) {
        throw std::invalid_argument("GPU measurement requires repetitions");
    }
    for (std::uint32_t index = 0U; index < warmup; ++index) {
        static_cast<void>(price(option, simulation, device));
    }

    SimulationParams formal_simulation = simulation;
    std::vector<double> totals;
    std::vector<double> computes;
    std::vector<double> reductions;
    std::vector<double> rngs;
    std::optional<RawMoments> moments;
    std::optional<PricingResult> result;
    std::optional<CudaPricingRun> first;
    for (std::uint32_t repetition = 0U; repetition < repetitions;
         ++repetition) {
        const auto start = std::chrono::steady_clock::now();
        auto run = price(option, formal_simulation, device);
        if (!first) {
            first = run;
            if (formal_simulation.batch_size == 0U) {
                formal_simulation.batch_size = run.batch_size;
            }
        }
        if (moments) {
            require_same_moments(*moments, run.moments);
        } else {
            moments = run.moments;
        }
        auto analyzed = analyze(run.moments, reference_price);
        const auto stop = std::chrono::steady_clock::now();
        if (!result) {
            result = std::move(analyzed);
        }
        totals.push_back(
            std::chrono::duration<double, std::milli>(stop - start).count());
        computes.push_back(run.gpu_compute_ms);
        reductions.push_back(run.reduction_ms);
        if (run.rng_ms) {
            rngs.push_back(*run.rng_ms);
        }
    }
    return {std::move(*result),
            median(std::move(totals)),
            median(std::move(computes)),
            rngs.empty() ? std::nullopt
                         : std::optional<double>(median(std::move(rngs))),
            median(std::move(reductions)),
            first->batch_size,
            first->grid_size,
            std::move(first->device)};
}

} // namespace pricer
