#pragma once

#include "benchmark.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace luxorion::benchmark_detail {

inline void check(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": "
                             + cudaGetErrorString(status));
  }
}

class Event {
 public:
  Event() { check(cudaEventCreate(&value_), "cudaEventCreate failed"); }
  ~Event() { cudaEventDestroy(value_); }
  Event(const Event&) = delete;
  Event& operator=(const Event&) = delete;
  cudaEvent_t get() const { return value_; }

 private:
  cudaEvent_t value_{};
};

inline TimingStats summarize(std::vector<double> samples) {
  if (samples.empty()) throw std::invalid_argument("benchmark samples are empty");
  std::sort(samples.begin(), samples.end());
  const std::size_t n = samples.size();
  const double median = n % 2 != 0
      ? samples[n / 2]
      : (samples[n / 2 - 1] + samples[n / 2]) / 2.0;
  const std::size_t p95_index = static_cast<std::size_t>(
      std::ceil(0.95 * static_cast<double>(n))) - 1;
  return {median, samples[p95_index]};
}

template <typename Launch>
TimingStats measure_cuda_events(unsigned warmup, unsigned repeats,
                                Launch launch) {
  if (repeats == 0) throw std::invalid_argument("benchmark repeats must be positive");
  for (unsigned i = 0; i < warmup; ++i) launch();
  check(cudaDeviceSynchronize(), "CUDA benchmark warmup failed");

  Event begin;
  Event end;
  std::vector<double> samples;
  samples.reserve(repeats);
  for (unsigned i = 0; i < repeats; ++i) {
    check(cudaEventRecord(begin.get()), "recording CUDA start event failed");
    launch();
    check(cudaEventRecord(end.get()), "recording CUDA stop event failed");
    check(cudaEventSynchronize(end.get()), "waiting for CUDA stop event failed");
    float milliseconds = 0.0F;
    check(cudaEventElapsedTime(&milliseconds, begin.get(), end.get()),
          "measuring CUDA event time failed");
    samples.push_back(milliseconds);
  }
  return summarize(std::move(samples));
}

template <typename Run>
TimingStats measure_host(unsigned warmup, unsigned repeats, Run run) {
  if (repeats == 0) throw std::invalid_argument("benchmark repeats must be positive");
  for (unsigned i = 0; i < warmup; ++i) run();
  std::vector<double> samples;
  samples.reserve(repeats);
  for (unsigned i = 0; i < repeats; ++i) {
    const auto begin = std::chrono::steady_clock::now();
    run();
    const auto end = std::chrono::steady_clock::now();
    samples.push_back(std::chrono::duration<double, std::milli>(end - begin).count());
  }
  return summarize(std::move(samples));
}

inline double effective_gbps(std::size_t logical_bytes, double median_ms) {
  return median_ms == 0.0 ? 0.0
      : static_cast<double>(logical_bytes) / median_ms / 1.0e6;
}

}  // namespace luxorion::benchmark_detail
