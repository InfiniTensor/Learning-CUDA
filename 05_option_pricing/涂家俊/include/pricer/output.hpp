#pragma once

// 定义定价结果、性能记录及 JSON/CSV 输出接口。

#include <cstdint>
#include <filesystem>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "pricer/config.hpp"
#include "pricer/statistics.hpp"

namespace pricer {

// 一次后端运行的时间、吞吐和加速比数据。
struct PerformanceData {
    double total_runtime_ms;
    double compute_runtime_ms;
    std::optional<double> rng_ms;
    std::optional<double> reduction_ms;
    double paths_per_second;
    std::uint32_t block_size;
    std::uint64_t grid_size;
    std::uint32_t repetitions;
    std::string aggregation;
    std::optional<double> cpu_runtime_ms;
    std::optional<double> speedup_vs_cpu;
};

// 记录复现实验需要的硬件和构建环境。
struct EnvironmentData {
    std::optional<std::string> gpu_name;
    std::optional<std::string> compute_capability;
    std::optional<std::string> cuda_runtime_version;
    std::optional<std::string> build_type;
    std::optional<std::string> git_commit;
};

// 序列化为 JSON 和 CSV 的完整一次运行记录。
struct OutputRecord {
    std::string run_id;
    std::string timestamp_utc;
    std::string backend;
    OptionParams option;
    SimulationParams simulation;
    PricingResult result;
    PerformanceData performance;
    EnvironmentData environment;
};

// 成功发布后返回两个输出文件的位置。
struct WrittenOutput {
    std::filesystem::path json_path;
    std::filesystem::path csv_path;
    std::string run_id;
};

class OutputError : public std::runtime_error {
  public:
    using std::runtime_error::runtime_error;
};

// 只构建 JSON 内容，不访问文件系统，便于单元测试。
nlohmann::json make_result_json(const OutputRecord &record);

std::filesystem::path
// 临时写入 JSON 后再 rename，避免中途失败留下半个结果文件。
write_result_json_atomic(const std::filesystem::path &output_dir,
                         OutputRecord record);

// 在进程锁保护下创建或追加性能 CSV。
void append_performance_csv(const std::filesystem::path &path,
                            const OutputRecord &record);

// 将 JSON 和 CSV 作为一个输出事务发布。
WrittenOutput write_output_transaction(const std::filesystem::path &output_dir,
                                       OutputRecord record);

// 返回计时样本中位数，降低单次抖动的影响。
double median(std::vector<double> measurements);
std::string utc_timestamp();

} // namespace pricer
