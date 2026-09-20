// config.h - 期权/模拟参数文件解析
#pragma once

#include <stdexcept>
#include <string>

#include "types.h"

// 配置解析错误（携带行号，便于定位）
struct ConfigError : std::runtime_error {
    int line;
    ConfigError(int line_, const std::string& msg)
        : std::runtime_error("line " + std::to_string(line_) + ": " + msg), line(line_) {}
};

// 解析期权参数文件（option_type/spot/strike/risk_free_rate/volatility/maturity/barrier）
OptionSpec parse_option_file(const std::string& path);

// 解析模拟参数文件（num_paths/num_steps/seed/rng/variance_reduction，缺省用默认值）
SimSpec parse_sim_file(const std::string& path);
