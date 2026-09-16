// 使用 CPU 单线程生成路径，作为 GPU 结果的统计基线。
#include "pricer/cpu_pricer.hpp"

#include <chrono>
#include <cmath>
#include <random>
#include <stdexcept>

#include "pricer/payoff.hpp"

namespace pricer {
namespace {

// CPU 与 GPU 在启动前执行同一类基础请求校验。
void validate_request(const OptionParams &option,
                      const SimulationParams &simulation) {
    if (simulation.num_paths == 0U) {
        throw std::invalid_argument("CPU pricing requires at least one path");
    }
    if (option.type != OptionType::EuropeanCall &&
        option.type != OptionType::AsianArithmeticCall) {
        throw std::domain_error("CPU pricing supports only P0 call options");
    }
    if (option.type == OptionType::AsianArithmeticCall &&
        simulation.num_steps == 0U) {
        throw std::domain_error("Asian CPU pricing requires at least one step");
    }
}

// 将单条路径收益折叠进原始矩，避免保存整条收益数组。
void accumulate(RawMoments &moments, double payoff) {
    // 不保存全部路径收益：只保留 sum 和 sum_squares 已足够在最后算均值、方差和
    // SE。
    ++moments.count;
    moments.sum += payoff;
    moments.sum_squares += payoff * payoff;
}

} // namespace

// 按期权类型选择一次终值采样或多步路径采样。
CpuPricingRun CpuMonteCarloPricer::price(const OptionParams &option,
                                         const SimulationParams &simulation) {
    validate_request(option, simulation);

    const auto start = std::chrono::steady_clock::now();
    // 每次 price 都从同一个 seed 重新开始，保证同一 CPU
    // 二进制上的重复运行可复现。
    std::mt19937_64 engine(simulation.seed);
    std::normal_distribution<double> normal(0.0, 1.0);
    RawMoments moments{0U, 0.0, 0.0};

    if (option.type == OptionType::EuropeanCall) {
        // 欧式期权只关心到期价，所以一条路径只需要一个标准正态随机数和一个 GBM
        // 步。
        const auto step = make_gbm_step_constants(
            option.maturity, option.risk_free_rate, option.volatility, 1U);
        for (std::uint64_t path = 0; path < simulation.num_paths; ++path) {
            const double terminal_spot =
                option.spot *
                std::exp(step.drift + step.diffusion * normal(engine));
            accumulate(moments,
                       european_call_payoff(terminal_spot, option.strike,
                                            step.discount));
        }
    } else {
        // 亚式期权关心路径上的平均价，因此一条路径必须逐步演化并累计每个监控价。
        const auto step =
            make_gbm_step_constants(option.maturity, option.risk_free_rate,
                                    option.volatility, simulation.num_steps);
        for (std::uint64_t path = 0; path < simulation.num_paths; ++path) {
            double spot = option.spot;
            double running_sum = 0.0; // 不包含初始 S0，见 Asian payoff 的定义。
            for (std::uint32_t index = 0; index < simulation.num_steps;
                 ++index) {
                spot *= std::exp(step.drift + step.diffusion * normal(engine));
                running_sum += spot;
            }
            accumulate(moments, asian_arithmetic_call_payoff(
                                    running_sum, simulation.num_steps,
                                    option.strike, step.discount));
        }
    }

    const auto stop = std::chrono::steady_clock::now();
    const double compute_runtime_ms =
        std::chrono::duration<double, std::milli>(stop - start).count();
    return {moments, compute_runtime_ms};
}

} // namespace pricer
