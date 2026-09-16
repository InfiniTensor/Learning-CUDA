#pragma once

// 定义期权、模拟和命令行配置的数据结构与读取接口。

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <stdexcept>
#include <string>

namespace pricer {

// P0 支持的期权类型；亚式期权需要完整路径上的多个监控点。
enum class OptionType { EuropeanCall, AsianArithmeticCall };
// GPU 路径演化可选精度；统计累加始终使用双精度。
enum class Precision { Fp32, Fp64 };
enum class VarianceReduction { None };
enum class RngType { CurandPhilox, Mt19937_64 };

// 一份期权合约的金融参数。
struct OptionParams {
    OptionType type;
    double spot;
    double strike;
    double risk_free_rate;
    double volatility;
    double maturity;
    std::optional<double> barrier;
};

// 一次蒙特卡洛运行的随机数、精度和 GPU 执行参数。
struct SimulationParams {
    std::uint64_t num_paths;
    std::uint32_t num_steps;
    std::uint64_t seed;
    RngType rng;
    Precision precision;
    VarianceReduction variance_reduction;
    std::uint32_t block_size;
    std::uint64_t batch_size;
};

// 将合约、模拟策略和 CLI 后端选择组合为完整运行配置。
struct RunConfig {
    OptionParams option;
    SimulationParams simulation;
    std::filesystem::path output_dir;
    bool run_cpu;
    bool run_gpu;
};

enum class ErrorCode { ConfigInvalid, FileRead, UnsupportedFeature };

// 保留文件、行号和字段，方便用户定位配置错误。
class ConfigError : public std::runtime_error {
  public:
    ConfigError(ErrorCode code, std::filesystem::path source_file,
                std::optional<std::size_t> line,
                std::optional<std::string> field, std::string reason);

    ErrorCode code() const noexcept;
    const std::filesystem::path &source_file() const noexcept;
    const std::optional<std::size_t> &line() const noexcept;
    const std::optional<std::string> &field() const noexcept;
    const std::string &reason() const noexcept;
    std::string render() const;

  private:
    ErrorCode code_;
    std::filesystem::path source_file_;
    std::optional<std::size_t> line_;
    std::optional<std::string> field_;
    std::string reason_;
};

// 将配置错误稳定映射为 CLI 退出码。
int exit_code_for(ErrorCode code) noexcept;

// 读取两份 INI 文件，并在返回前完成跨字段校验。
RunConfig load_run_config(const std::filesystem::path &option_file,
                          const std::filesystem::path &simulation_file,
                          std::filesystem::path output_dir = "results",
                          bool run_cpu = true, bool run_gpu = true);

} // namespace pricer
