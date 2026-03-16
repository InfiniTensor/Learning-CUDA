//
// Created by core_dump on 2026/2/25.
//

#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <chrono>
#include <iostream>

__host__ __device__ __forceinline__
float mix_mul(float fp, __half h) {
    return fp * __half2float(h);
}

__host__ __device__ __forceinline__
float mix_mul(float fp, __nv_bfloat16 h) {
    return fp * __bfloat162float(h);
}

__host__ __device__ __forceinline__
float f162float(__half h) {
    return __half2float(h);
}

__host__ __device__ __forceinline__
float f162float(__nv_bfloat16 h) {
    return __bfloat162float(h);
}


#define CUDA_CHECK(call)                                                                    \
{                                                                                           \
    cudaError_t err = call;                                                                 \
    if (err != cudaSuccess) {                                                               \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__                        \
                  << " - " << cudaGetErrorString(err) << "\n";                              \
        std::exit(-1);                                                                      \
    }                                                                                       \
}

class Timer {
public:
    using clock = std::chrono::high_resolution_clock;

    Timer() : running_(false), elapsed_ms_(0.0) {}

    void tic() {
        start_ = clock::now();
        running_ = true;
    }

    double toc() {
        if (!running_) {
            return elapsed_ms_;
        }
        auto end = clock::now();
        elapsed_ms_ = std::chrono::duration<double, std::milli>(end - start_).count();
        running_ = false;
        return elapsed_ms_;
    }

    double elapsed() const {
        if (!running_) {
            return elapsed_ms_;
        }
        auto now = clock::now();
        return std::chrono::duration<double, std::milli>(now - start_).count();
    }

    void reset() {
        running_ = false;
        elapsed_ms_ = 0.0;
    }

private:
    clock::time_point start_;
    bool running_;
    double elapsed_ms_;
};

class Tracer {
public:
    Tracer() {}

    void start() {
        timer_.reset();
        timer_.tic();
    }

    void stop() {
        total_elapsed_ms_ += timer_.toc();
    }

    Tracer& memcpy_accumulate(uint64_t cpy_size_in_byte) {
        total_data_cpy_in_bytes_ += cpy_size_in_byte;
        return *this;
    }

    double bandwidth_bytes_per_s() const {
        if (total_elapsed_ms_ <= 0.0) {
            return 0.0;
        }
        return static_cast<double>(total_data_cpy_in_bytes_) * 1000.0 / total_elapsed_ms_;
    }

    double bandwidth_gib_per_s() const {
        if (total_elapsed_ms_ <= 0.0) {
            return 0.0;
        }
        constexpr double kBytesPerGiB = 1024.0 * 1024.0 * 1024.0;
        return static_cast<double>(total_data_cpy_in_bytes_) * 1000.0 / total_elapsed_ms_ / kBytesPerGiB;
    }

    void print(std::ostream& os = std::cout) const {
        os << "elapsed: " << total_elapsed_ms_ << " ms, "
           << "effective bandwidth: " << bandwidth_gib_per_s() << " GiB/s\n";
    }

private:
    Timer timer_;

    uint64_t total_data_cpy_in_bytes_ = 0;
    double total_elapsed_ms_;
};
