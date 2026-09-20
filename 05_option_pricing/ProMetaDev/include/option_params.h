// ============================================================
// option_params.h
// 期权/市场参数结构体 & 模拟参数结构体 & 枚举
// ============================================================
#ifndef CUDA_PRICING_OPTION_PARAMS_H
#define CUDA_PRICING_OPTION_PARAMS_H

#include <string>
#include <cstdint>

// -------- 期权类型枚举 --------
enum class OptionType {
    UNKNOWN = 0,
    EUROPEAN_CALL, // 欧式看涨
    EUROPEAN_PUT,  // 欧式看跌
    ASIAN_CALL,    // 亚式算术平均看涨
    ASIAN_PUT,     // 亚式算术平均看跌
    BARRIER_CALL,  // 障碍看涨
    BARRIER_PUT    // 障碍看跌
};

// -------- 障碍期权细分方向 --------
enum class BarrierDir { UP = 1, DOWN = -1 };
enum class BarrierType { KNOCK_IN = 0, KNOCK_OUT = 1 };

// -------- 方差缩减方式 --------
enum class VarianceReduction {
    NONE = 0,
    ANTITHETIC,     // 对偶变量法
    CONTROL_VARIATE // 控制变量法
};

// -------- RNG 类型 --------
// 注: 项目始终使用 curand Philox4_32_10 (可复现 RNG), 早期声明的
//      CUSTOM_XORWOW 从未实现, 已清理移除。
enum class RNGType { CURAND_DEFAULT = 0 };

// -------- 期权 & 市场参数 --------
struct OptionParams {
    OptionType type = OptionType::EUROPEAN_CALL;
    double spot = 100.0;
    double strike = 100.0;
    double risk_free = 0.03;
    double volatility = 0.2;
    double maturity = 1.0;
    double barrier = 130.0;
    BarrierDir barrier_dir = BarrierDir::UP;
    BarrierType barrier_t = BarrierType::KNOCK_IN;

    static OptionType parse_option_type(const std::string& s);
    static const char* to_string(OptionType t);
};

// -------- 模拟参数 --------
struct SimParams {
    int64_t num_paths = 10000000;
    int num_steps = 256;
    unsigned int seed = 1234;
    RNGType rng = RNGType::CURAND_DEFAULT;
    VarianceReduction variance_reduction = VarianceReduction::NONE;
    int block_size = 256;
};

// -------- 定价结果 --------
struct PricingResult {
    std::string option_type_str;
    std::string method_str;
    double price = 0.0;
    double ref_price = 0.0;
    double abs_error = 0.0;
    double std_error = 0.0;
    double ci_half = 0.0;
    double cpu_time_ms = 0.0;
    double gpu_time_ms = 0.0;
    double paths_per_sec = 0.0;
    double speedup = 0.0;
};

#endif // CUDA_PRICING_OPTION_PARAMS_H
