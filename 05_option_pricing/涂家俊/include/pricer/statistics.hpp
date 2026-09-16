#pragma once

// 定义原始矩和价格统计量之间的转换接口。

#include <cstdint>
#include <optional>

namespace pricer {

// 不保存所有样本时仍可计算统计量的一阶、二阶原始矩。
struct RawMoments {
    std::uint64_t count;
    double sum;
    double sum_squares;
};

// 面向用户输出的价格、误差和置信区间。
struct PricingResult {
    double price;
    std::optional<double> sample_stddev;
    std::optional<double> standard_error;
    std::optional<double> ci_lower;
    std::optional<double> ci_upper;
    std::optional<double> reference_price;
    std::optional<double> absolute_error;
    std::optional<double> relative_error;
    RawMoments moments;
};

// 合并批次或分段运行产生的原始矩。
RawMoments merge_raw_moments(const RawMoments &left, const RawMoments &right);

class ResultAnalyzer {
  public:
    // 从原始矩推导样本标准差、标准误和 95% 置信区间。
    static PricingResult
    analyze(const RawMoments &moments,
            std::optional<double> reference_price = std::nullopt);
};

} // namespace pricer
