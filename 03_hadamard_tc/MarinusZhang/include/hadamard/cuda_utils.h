// Small CUDA helpers: error checking, event timing, device query.
#ifndef HADAMARD_CUDA_UTILS_H_
#define HADAMARD_CUDA_UTILS_H_

#include <cuda_runtime.h>

#include <cstdio>
#include <stdexcept>
#include <string>

namespace hadamard {

inline void check_cuda(cudaError_t status, const char* expr, const char* file, int line) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "CUDA error at %s:%d (%s): %s\n", file, line, expr,
                     cudaGetErrorString(status));
        throw std::runtime_error("CUDA call failed");
    }
}

#define HW_CUDA_CHECK(expr) ::hadamard::check_cuda((expr), #expr, __FILE__, __LINE__)

// Wall-clock timing of GPU work through CUDA events, reported in milliseconds,
// which is the metric the project asks for.
class CudaEventTimer {
   public:
    CudaEventTimer() {
        HW_CUDA_CHECK(cudaEventCreate(&start_));
        HW_CUDA_CHECK(cudaEventCreate(&stop_));
    }
    ~CudaEventTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }
    CudaEventTimer(const CudaEventTimer&) = delete;
    CudaEventTimer& operator=(const CudaEventTimer&) = delete;

    void start(cudaStream_t stream = nullptr) { HW_CUDA_CHECK(cudaEventRecord(start_, stream)); }

    void stop(cudaStream_t stream = nullptr) {
        HW_CUDA_CHECK(cudaEventRecord(stop_, stream));
        HW_CUDA_CHECK(cudaEventSynchronize(stop_));
    }

    float elapsed_ms() const {
        float ms = 0.0f;
        HW_CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
        return ms;
    }

   private:
    cudaEvent_t start_{};
    cudaEvent_t stop_{};
};

struct DeviceInfo {
    std::string name;
    int major = 0;
    int minor = 0;
    int sm_count = 0;
    int clock_khz = 0;
    size_t global_mem_bytes = 0;
    int driver_version = 0;
    int runtime_version = 0;
};

inline DeviceInfo query_device(int device_id = 0) {
    DeviceInfo info;
    cudaDeviceProp prop{};
    HW_CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));
    info.name = prop.name;
    info.major = prop.major;
    info.minor = prop.minor;
    info.sm_count = prop.multiProcessorCount;
    info.clock_khz = prop.clockRate;
    info.global_mem_bytes = prop.totalGlobalMem;
    HW_CUDA_CHECK(cudaDriverGetVersion(&info.driver_version));
    HW_CUDA_CHECK(cudaRuntimeGetVersion(&info.runtime_version));
    return info;
}

// Effective bandwidth of a memory-bound kernel, assuming `bytes` moved in `ms`.
inline double effective_bandwidth_gbps(size_t bytes, float ms) {
    if (ms <= 0.0f) {
        return 0.0;
    }
    return static_cast<double>(bytes) / (static_cast<double>(ms) * 1e-3) / 1e9;
}

}  // namespace hadamard

#endif  // HADAMARD_CUDA_UTILS_H_
