#pragma once

// 声明 CUDA 蒙特卡洛后端及其设备错误转换接口。

#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>

#include "pricer/config.hpp"
#include "pricer/statistics.hpp"

namespace pricer {

// 将 CUDA API 调用失败转换为带调用位置的 C++ 异常。
class CudaError : public std::runtime_error {
  public:
    CudaError(std::string api, int code, std::string cuda_text,
              std::string source_file, int source_line);

    const std::string &api() const noexcept;
    int code() const noexcept;
    const std::string &cuda_text() const noexcept;
    const std::string &source_file() const noexcept;
    int source_line() const noexcept;

  private:
    std::string api_;
    int code_;
    std::string cuda_text_;
    std::string source_file_;
    int source_line_;
};

// 输出报告所需的目标 GPU 基本信息。
struct CudaDeviceInfo {
    int id;
    std::string name;
    int compute_capability_major;
    int compute_capability_minor;
    std::size_t total_memory_bytes;
    std::size_t free_memory_bytes;
    int max_threads_per_block;
};

// GPU 一次定价的累计矩、分批策略和分段计时。
struct CudaPricingRun {
    RawMoments moments;
    double gpu_compute_ms;
    double reduction_ms;
    std::optional<double> rng_ms;
    std::uint64_t batch_size;
    std::uint64_t grid_size;
    CudaDeviceInfo device;
};

class CudaMonteCarloPricer {
  public:
    // 在指定设备上模拟路径并完成收益归约。
    static CudaPricingRun price(const OptionParams &option,
                                const SimulationParams &simulation,
                                int device_id = 0);
};

} // namespace pricer
