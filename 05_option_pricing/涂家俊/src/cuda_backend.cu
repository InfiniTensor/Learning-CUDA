// 在 GPU 上生成蒙特卡洛路径，并归约收益的一阶、二阶矩。
#include "pricer/cuda_pricer.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

#include <cub/device/device_reduce.cuh>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <thrust/iterator/transform_iterator.h>

#ifdef PRICER_ENABLE_NVTX
#include <nvtx3/nvToolsExt.h>
#endif

namespace pricer {
namespace {

[[noreturn]] void throw_cuda_error(const char *api, cudaError_t status,
                                   const char *source_file, int source_line) {
    throw CudaError(api, static_cast<int>(status), cudaGetErrorString(status),
                    source_file, source_line);
}

void check_cuda(cudaError_t status, const char *api, const char *source_file,
                int source_line) {
    if (status != cudaSuccess) {
        throw_cuda_error(api, status, source_file, source_line);
    }
}

#define PRICER_CUDA_CHECK(api_name, expression)                                \
    ::pricer::check_cuda((expression), (api_name), __FILE__, __LINE__)

class DeviceBuffer {
  public:
    DeviceBuffer() = default;

    explicit DeviceBuffer(std::size_t bytes) { allocate(bytes); }

    DeviceBuffer(const DeviceBuffer &) = delete;
    DeviceBuffer &operator=(const DeviceBuffer &) = delete;

    ~DeviceBuffer() noexcept {
        if (pointer_ != nullptr) {
            const cudaError_t status = cudaFree(pointer_);
            if (status != cudaSuccess) {
                // Destructors cannot replace an active exception. Normal-path
                // cleanup uses release(), which reports a structured error.
            }
        }
    }

    void allocate(std::size_t bytes) {
        if (bytes == 0U) {
            throw std::invalid_argument(
                "CUDA allocation size must be positive");
        }
        PRICER_CUDA_CHECK("cudaMalloc", cudaMalloc(&pointer_, bytes));
    }

    void *get() const noexcept { return pointer_; }

    void release() {
        if (pointer_ != nullptr) {
            void *released = pointer_;
            pointer_ = nullptr;
            PRICER_CUDA_CHECK("cudaFree", cudaFree(released));
        }
    }

  private:
    void *pointer_ = nullptr;
};

class CudaEvent {
  public:
    CudaEvent() {
        PRICER_CUDA_CHECK("cudaEventCreate", cudaEventCreate(&event_));
    }

    CudaEvent(const CudaEvent &) = delete;
    CudaEvent &operator=(const CudaEvent &) = delete;

    ~CudaEvent() noexcept {
        if (event_ != nullptr) {
            const cudaError_t status = cudaEventDestroy(event_);
            if (status != cudaSuccess) {
                // See DeviceBuffer::~DeviceBuffer().
            }
        }
    }

    void record() {
        PRICER_CUDA_CHECK("cudaEventRecord", cudaEventRecord(event_));
    }

    void synchronize() {
        PRICER_CUDA_CHECK("cudaEventSynchronize", cudaEventSynchronize(event_));
    }

    float elapsed_since(const CudaEvent &start) const {
        float milliseconds = 0.0F;
        PRICER_CUDA_CHECK(
            "cudaEventElapsedTime",
            cudaEventElapsedTime(&milliseconds, start.event_, event_));
        return milliseconds;
    }

    void release() {
        if (event_ != nullptr) {
            cudaEvent_t released = event_;
            event_ = nullptr;
            PRICER_CUDA_CHECK("cudaEventDestroy", cudaEventDestroy(released));
        }
    }

  private:
    cudaEvent_t event_ = nullptr;
};

class NvtxRange {
  public:
    explicit NvtxRange(const char *label) {
#ifdef PRICER_ENABLE_NVTX
        nvtxRangePushA(label);
#else
        static_cast<void>(label);
#endif
    }

    NvtxRange(const NvtxRange &) = delete;
    NvtxRange &operator=(const NvtxRange &) = delete;

    ~NvtxRange() noexcept {
#ifdef PRICER_ENABLE_NVTX
        nvtxRangePop();
#endif
    }
};

struct Square {
    __host__ __device__ double operator()(const double &value) const {
        return value * value;
    }
};

template <typename Real> struct PathMath;

template <> struct PathMath<float> {
    __device__ static float normal(curandStatePhilox4_32_10_t *state) {
        return curand_normal(state);
    }

    __device__ static float exponential(float value) { return expf(value); }

    __device__ static float maximum(float left, float right) {
        return fmaxf(left, right);
    }
};

template <> struct PathMath<double> {
    __device__ static double normal(curandStatePhilox4_32_10_t *state) {
        return curand_normal_double(state);
    }

    __device__ static double exponential(double value) { return exp(value); }

    __device__ static double maximum(double left, double right) {
        return fmax(left, right);
    }
};

template <typename Real>
__global__ void
european_payoff_kernel(double *payoffs, std::uint64_t batch_paths,
                       std::uint64_t batch_offset, std::uint64_t seed,
                       Real spot, Real strike, Real drift, Real diffusion,
                       Real discount) {
    const std::uint64_t thread =
        static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::uint64_t stride =
        static_cast<std::uint64_t>(gridDim.x) * blockDim.x;
    for (std::uint64_t local_path = thread; local_path < batch_paths;
         local_path += stride) {
        // 一个 CUDA 线程负责一条路径。batch_offset + local_path
        // 是全局路径编号，让分批与不分批时也取到同一条 Philox
        // 随机子序列，而不会重复抽样。
        curandStatePhilox4_32_10_t state;
        curand_init(static_cast<unsigned long long>(seed),
                    static_cast<unsigned long long>(batch_offset + local_path),
                    0ULL, &state);
        const Real normal = PathMath<Real>::normal(&state);
        const Real terminal_spot =
            spot * PathMath<Real>::exponential(drift + diffusion * normal);
        const Real payoff =
            discount * PathMath<Real>::maximum(terminal_spot - strike, Real{0});
        payoffs[local_path] = static_cast<double>(payoff);
    }
}

template <typename Real>
__global__ void asian_payoff_kernel(double *payoffs, std::uint64_t batch_paths,
                                    std::uint64_t batch_offset,
                                    std::uint64_t seed, Real initial_spot,
                                    Real strike, Real drift, Real diffusion,
                                    Real discount, std::uint32_t num_steps) {
    const std::uint64_t thread =
        static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::uint64_t stride =
        static_cast<std::uint64_t>(gridDim.x) * blockDim.x;
    for (std::uint64_t local_path = thread; local_path < batch_paths;
         local_path += stride) {
        // 亚式版本与 CPU 的双层循环一一对应：外层是路径，内层是时间监控点。
        curandStatePhilox4_32_10_t state;
        curand_init(static_cast<unsigned long long>(seed),
                    static_cast<unsigned long long>(batch_offset + local_path),
                    0ULL, &state);
        Real spot = initial_spot;
        Real running_sum = Real{0};
        for (std::uint32_t step = 0; step < num_steps; ++step) {
            spot *= PathMath<Real>::exponential(
                drift + diffusion * PathMath<Real>::normal(&state));
            running_sum += spot;
        }
        const Real average = running_sum / static_cast<Real>(num_steps);
        const Real payoff =
            discount * PathMath<Real>::maximum(average - strike, Real{0});
        payoffs[local_path] = static_cast<double>(payoff);
    }
}

template <typename Real>
void launch_payoff_kernel(const OptionParams &option,
                          const SimulationParams &simulation,
                          double *payoff_pointer, std::uint64_t batch_paths,
                          std::uint64_t batch_offset, std::uint64_t grid,
                          double drift, double diffusion, double discount) {
    if (option.type == OptionType::EuropeanCall) {
        european_payoff_kernel<Real>
            <<<static_cast<unsigned int>(grid), simulation.block_size>>>(
                payoff_pointer, batch_paths, batch_offset, simulation.seed,
                static_cast<Real>(option.spot),
                static_cast<Real>(option.strike), static_cast<Real>(drift),
                static_cast<Real>(diffusion), static_cast<Real>(discount));
    } else {
        asian_payoff_kernel<Real>
            <<<static_cast<unsigned int>(grid), simulation.block_size>>>(
                payoff_pointer, batch_paths, batch_offset, simulation.seed,
                static_cast<Real>(option.spot),
                static_cast<Real>(option.strike), static_cast<Real>(drift),
                static_cast<Real>(diffusion), static_cast<Real>(discount),
                simulation.num_steps);
    }
}

std::size_t checked_add(std::size_t left, std::size_t right) {
    if (right > std::numeric_limits<std::size_t>::max() - left) {
        throw std::overflow_error("CUDA batch byte count overflow");
    }
    return left + right;
}

std::size_t payoff_bytes(std::uint64_t batch_paths) {
    if (batch_paths >
        std::numeric_limits<std::size_t>::max() / sizeof(double)) {
        throw std::overflow_error("CUDA payoff byte count overflow");
    }
    return static_cast<std::size_t>(batch_paths) * sizeof(double);
}

std::size_t reduction_temp_bytes(std::uint64_t batch_paths) {
    std::size_t sum_bytes = 0U;
    PRICER_CUDA_CHECK(
        "cub::DeviceReduce::Sum(query payoff)",
        cub::DeviceReduce::Sum(nullptr, sum_bytes,
                               static_cast<const double *>(nullptr),
                               static_cast<double *>(nullptr), batch_paths));

    std::size_t square_bytes = 0U;
    const auto squares = thrust::make_transform_iterator(
        static_cast<const double *>(nullptr), Square{});
    PRICER_CUDA_CHECK("cub::DeviceReduce::Sum(query square)",
                      cub::DeviceReduce::Sum(nullptr, square_bytes, squares,
                                             static_cast<double *>(nullptr),
                                             batch_paths));
    return std::max(sum_bytes, square_bytes);
}

struct BatchStorage {
    std::uint64_t paths;
    std::size_t payoff_bytes;
    std::size_t temporary_bytes;
    std::size_t total_bytes;
};

BatchStorage describe_storage(std::uint64_t batch_paths) {
    const std::size_t payoffs = payoff_bytes(batch_paths);
    const std::size_t temporary = reduction_temp_bytes(batch_paths);
    std::size_t total = checked_add(payoffs, temporary);
    total = checked_add(total, 2U * sizeof(double));
    return {batch_paths, payoffs, temporary, total};
}

BatchStorage choose_batch(const SimulationParams &simulation,
                          std::size_t free_memory_bytes) {
    if (simulation.batch_size > 0U) {
        const auto storage = describe_storage(
            std::min(simulation.batch_size, simulation.num_paths));
        if (storage.total_bytes > free_memory_bytes) {
            throw_cuda_error("batch_memory", cudaErrorMemoryAllocation,
                             __FILE__, __LINE__);
        }
        return storage;
    }

    // 自动模式只使用约 80% 的空闲显存，给 CUDA 运行时和显示任务保留余量。
    const std::size_t budget = (free_memory_bytes / 10U) * 8U;
    const std::uint64_t maximum_by_size = static_cast<std::uint64_t>(
        std::numeric_limits<std::size_t>::max() / sizeof(double));
    std::uint64_t low = 0U;
    std::uint64_t high = std::min(simulation.num_paths, maximum_by_size);
    while (low < high) {
        const std::uint64_t middle = low + (high - low + 1U) / 2U;
        if (describe_storage(middle).total_bytes <= budget) {
            low = middle;
        } else {
            high = middle - 1U;
        }
    }
    if (low == 0U) {
        throw_cuda_error("automatic_batch_memory", cudaErrorMemoryAllocation,
                         __FILE__, __LINE__);
    }
    if (low > simulation.block_size) {
        low = (low / simulation.block_size) * simulation.block_size;
    }
    low = std::max<std::uint64_t>(low, 1U);
    return describe_storage(low);
}

void validate_request(const OptionParams &option,
                      const SimulationParams &simulation) {
    if (simulation.num_paths == 0U) {
        throw std::invalid_argument("CUDA pricing requires at least one path");
    }
    if (option.type != OptionType::EuropeanCall &&
        option.type != OptionType::AsianArithmeticCall) {
        throw std::domain_error("CUDA pricing supports only P0 call options");
    }
    if (option.type == OptionType::AsianArithmeticCall &&
        simulation.num_steps == 0U) {
        throw std::domain_error(
            "Asian CUDA pricing requires at least one step");
    }
}

std::uint64_t grid_size_for(std::uint64_t paths, std::uint32_t block_size,
                            int maximum_grid_size) {
    const std::uint64_t blocks =
        paths / block_size + (paths % block_size == 0U ? 0U : 1U);
    return std::min(blocks, static_cast<std::uint64_t>(maximum_grid_size));
}

} // namespace

CudaError::CudaError(std::string api, int code, std::string cuda_text,
                     std::string source_file, int source_line)
    : std::runtime_error(api + " failed with CUDA error " +
                         std::to_string(code) + ": " + cuda_text),
      api_(std::move(api)), code_(code), cuda_text_(std::move(cuda_text)),
      source_file_(std::move(source_file)), source_line_(source_line) {}

const std::string &CudaError::api() const noexcept { return api_; }
int CudaError::code() const noexcept { return code_; }
const std::string &CudaError::cuda_text() const noexcept { return cuda_text_; }
const std::string &CudaError::source_file() const noexcept {
    return source_file_;
}
int CudaError::source_line() const noexcept { return source_line_; }

CudaPricingRun CudaMonteCarloPricer::price(const OptionParams &option,
                                           const SimulationParams &simulation,
                                           int device_id) {
    validate_request(option, simulation);

    int device_count = 0;
    PRICER_CUDA_CHECK("cudaGetDeviceCount", cudaGetDeviceCount(&device_count));
    if (device_id < 0 || device_id >= device_count) {
        throw_cuda_error("device_id", cudaErrorInvalidDevice, __FILE__,
                         __LINE__);
    }
    PRICER_CUDA_CHECK("cudaSetDevice", cudaSetDevice(device_id));

    cudaDeviceProp properties{};
    PRICER_CUDA_CHECK("cudaGetDeviceProperties",
                      cudaGetDeviceProperties(&properties, device_id));
    if (simulation.block_size == 0U ||
        simulation.block_size >
            static_cast<std::uint32_t>(properties.maxThreadsPerBlock)) {
        throw std::invalid_argument(
            "CUDA block size must be within the selected device limit");
    }
    if (properties.maxGridSize[0] <= 0) {
        throw std::domain_error("CUDA device reports no usable x-grid size");
    }

    std::size_t free_memory_bytes = 0U;
    std::size_t total_memory_bytes = 0U;
    PRICER_CUDA_CHECK("cudaMemGetInfo",
                      cudaMemGetInfo(&free_memory_bytes, &total_memory_bytes));
    // 批处理只限制“同时放在显存里的路径数”；所有批次的统计量随后在主机合并。
    const BatchStorage storage = choose_batch(simulation, free_memory_bytes);
    const std::uint64_t representative_grid = grid_size_for(
        storage.paths, simulation.block_size, properties.maxGridSize[0]);
    const double discount = std::exp(-option.risk_free_rate * option.maturity);
    const double step_maturity =
        option.type == OptionType::EuropeanCall
            ? option.maturity
            : option.maturity / static_cast<double>(simulation.num_steps);
    const double drift =
        (option.risk_free_rate - 0.5 * option.volatility * option.volatility) *
        step_maturity;
    const double diffusion = option.volatility * std::sqrt(step_maturity);

    DeviceBuffer payoffs(storage.payoff_bytes);
    DeviceBuffer reduced_sum(sizeof(double));
    DeviceBuffer reduced_sum_squares(sizeof(double));
    DeviceBuffer temporary(storage.temporary_bytes);
    CudaEvent compute_start;
    CudaEvent compute_stop;
    CudaEvent reduction_start;
    CudaEvent reduction_stop;

    auto *payoff_pointer = static_cast<double *>(payoffs.get());
    auto *sum_pointer = static_cast<double *>(reduced_sum.get());
    auto *sum_squares_pointer =
        static_cast<double *>(reduced_sum_squares.get());

    RawMoments moments{0U, 0.0, 0.0};
    double gpu_compute_ms = 0.0;
    double reduction_ms = 0.0;
    for (std::uint64_t batch_offset = 0U;
         batch_offset < simulation.num_paths;) {
        const std::uint64_t batch_paths =
            std::min(storage.paths, simulation.num_paths - batch_offset);
        const std::uint64_t grid = grid_size_for(
            batch_paths, simulation.block_size, properties.maxGridSize[0]);

        // 这个循环是一批完整 GPU 工作：生成并计算每条路径收益，再归约为两个数。
        compute_start.record();
        {
            NvtxRange range("simulate_paths");
            if (simulation.precision == Precision::Fp32) {
                launch_payoff_kernel<float>(option, simulation, payoff_pointer,
                                            batch_paths, batch_offset, grid,
                                            drift, diffusion, discount);
            } else {
                launch_payoff_kernel<double>(option, simulation, payoff_pointer,
                                             batch_paths, batch_offset, grid,
                                             drift, diffusion, discount);
            }
        }
        PRICER_CUDA_CHECK("cudaGetLastError", cudaGetLastError());

        reduction_start.record();
        {
            NvtxRange range("reduce_moments");
            std::size_t temporary_bytes = storage.temporary_bytes;
            PRICER_CUDA_CHECK("cub::DeviceReduce::Sum(payoff)",
                              cub::DeviceReduce::Sum(
                                  temporary.get(), temporary_bytes,
                                  payoff_pointer, sum_pointer, batch_paths));
            const auto squares =
                thrust::make_transform_iterator(payoff_pointer, Square{});
            temporary_bytes = storage.temporary_bytes;
            PRICER_CUDA_CHECK("cub::DeviceReduce::Sum(square)",
                              cub::DeviceReduce::Sum(
                                  temporary.get(), temporary_bytes, squares,
                                  sum_squares_pointer, batch_paths));
        }
        reduction_stop.record();
        compute_stop.record();
        compute_stop.synchronize();
        gpu_compute_ms += compute_stop.elapsed_since(compute_start);
        reduction_ms += reduction_stop.elapsed_since(reduction_start);

        {
            NvtxRange range("copy_batch_moments");
            double host_sum = 0.0;
            double host_sum_squares = 0.0;
            PRICER_CUDA_CHECK("cudaMemcpy(sum)",
                              cudaMemcpy(&host_sum, sum_pointer,
                                         sizeof(host_sum),
                                         cudaMemcpyDeviceToHost));
            PRICER_CUDA_CHECK("cudaMemcpy(sum_squares)",
                              cudaMemcpy(&host_sum_squares, sum_squares_pointer,
                                         sizeof(host_sum_squares),
                                         cudaMemcpyDeviceToHost));
            moments = merge_raw_moments(
                moments, {batch_paths, host_sum, host_sum_squares});
        }
        // 每批只回传两个标量，随后与先前批次在主机合并。
        batch_offset += batch_paths;
    }

    CudaDeviceInfo device{device_id,
                          properties.name,
                          properties.major,
                          properties.minor,
                          total_memory_bytes,
                          free_memory_bytes,
                          properties.maxThreadsPerBlock};
    reduction_stop.release();
    reduction_start.release();
    compute_stop.release();
    compute_start.release();
    temporary.release();
    reduced_sum_squares.release();
    reduced_sum.release();
    payoffs.release();

    return {moments,       gpu_compute_ms,      reduction_ms,     std::nullopt,
            storage.paths, representative_grid, std::move(device)};
}

} // namespace pricer
