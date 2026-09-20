// ============================================================
// file_io.h - 参数文件解析 / 结果输出 (CSV + JSON)
// ============================================================
#ifndef CUDA_PRICING_FILE_IO_H
#define CUDA_PRICING_FILE_IO_H

#include "option_params.h"
#include "bs_formula.h"
#include <string>

// -------- 参数文件解析 --------
OptionParams parse_option_params(const std::string& path);
SimParams parse_sim_params(const std::string& path);

// -------- 结果输出 (greeks 可为 nullptr, 表示不输出) --------
void write_result_csv(const std::string& path, const OptionParams& o, const SimParams& s,
                      const PricingResult& r, double occupancy_pct, const GreeksBS* g = nullptr);
void write_result_json(const std::string& path, const OptionParams& o, const SimParams& s,
                       const PricingResult& r, double occupancy_pct, const GreeksBS* g = nullptr);

// -------- 性能日志 --------
void write_perf_log(const std::string& path, const OptionParams& o, const SimParams& s,
                    const PricingResult& r, double occupancy_pct, const GreeksBS* g = nullptr);

#endif // CUDA_PRICING_FILE_IO_H
