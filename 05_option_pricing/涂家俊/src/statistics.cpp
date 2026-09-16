// 从路径收益的原始矩计算均值、标准误和置信区间。
#include "pricer/statistics.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace pricer {
namespace {

// 统计公式假定收益和平方收益均有效且非负。
void validate_moments(const RawMoments &moments) {
    if (!std::isfinite(moments.sum) || !std::isfinite(moments.sum_squares)) {
        throw std::domain_error("moments must be finite");
    }
    if (moments.sum < 0.0 || moments.sum_squares < 0.0) {
        throw std::domain_error("moments must be non-negative");
    }
}

double require_finite(double value, const char *name) {
    if (!std::isfinite(value)) {
        throw std::overflow_error(std::string(name) + " is not finite");
    }
    return value;
}

} // namespace

// 分批计算时，原始矩可以直接相加而无需回放每条路径。
RawMoments merge_raw_moments(const RawMoments &left, const RawMoments &right) {
    // GPU 分批后，每一批只传回两个标量；此函数把它们合成一次完整实验的矩。
    validate_moments(left);
    validate_moments(right);
    if (right.count > std::numeric_limits<std::uint64_t>::max() - left.count) {
        throw std::overflow_error("moment count overflow");
    }

    const double sum = require_finite(left.sum + right.sum, "merged sum");
    const double sum_squares = require_finite(
        left.sum_squares + right.sum_squares, "merged sum of squares");
    return {left.count + right.count, sum, sum_squares};
}

// 使用样本方差而非总体方差，并为单样本保留不可用统计量。
PricingResult ResultAnalyzer::analyze(const RawMoments &moments,
                                      std::optional<double> reference_price) {
    validate_moments(moments);
    if (moments.count == 0U) {
        throw std::domain_error("at least one sample is required");
    }
    if (reference_price.has_value() && !std::isfinite(*reference_price)) {
        throw std::domain_error("reference price must be finite");
    }

    const double count = static_cast<double>(moments.count);
    // 折现收益的样本均值就是蒙特卡洛价格估计。
    const double price = require_finite(moments.sum / count, "price");
    PricingResult result{price,        std::nullopt, std::nullopt,
                         std::nullopt, std::nullopt, reference_price,
                         std::nullopt, std::nullopt, moments};

    if (moments.count > 1U) {
        // E[X^2] - E[X]^2 给出样本方差所需的分子；再由 stddev/sqrt(N) 得到 SE。
        const double expected_square =
            require_finite(count * price * price, "variance term");
        double variance_numerator = moments.sum_squares - expected_square;
        const double variance_scale = std::max(
            {1.0, std::abs(moments.sum_squares), std::abs(expected_square)});
        const double roundoff_tolerance =
            64.0 * std::numeric_limits<double>::epsilon() * variance_scale;
        if (variance_numerator < 0.0) {
            if (variance_numerator < -roundoff_tolerance) {
                throw std::domain_error(
                    "sample variance is materially negative");
            }
            variance_numerator = 0.0;
        }

        const double variance = require_finite(
            variance_numerator / static_cast<double>(moments.count - 1U),
            "sample variance");
        const double sample_stddev =
            require_finite(std::sqrt(variance), "sample standard deviation");
        const double standard_error =
            require_finite(sample_stddev / std::sqrt(count), "standard error");
        // 1.96 对应正态近似下的双侧 95% 置信区间。
        const double ci_lower = require_finite(price - 1.96 * standard_error,
                                               "lower confidence interval");
        const double ci_upper = require_finite(price + 1.96 * standard_error,
                                               "upper confidence interval");
        if (standard_error < 0.0 || ci_lower > ci_upper) {
            throw std::domain_error("invalid confidence interval");
        }
        result.sample_stddev = sample_stddev;
        result.standard_error = standard_error;
        result.ci_lower = ci_lower;
        result.ci_upper = ci_upper;
    }

    if (reference_price.has_value()) {
        const double absolute_error = require_finite(
            std::abs(price - *reference_price), "absolute error");
        result.absolute_error = absolute_error;
        if (std::abs(*reference_price) > 1e-15) {
            result.relative_error = require_finite(
                absolute_error / std::abs(*reference_price), "relative error");
        }
    }
    return result;
}

} // namespace pricer
