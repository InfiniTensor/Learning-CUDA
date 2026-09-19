#include "low_precision/cuda_quant.hpp"
#include "low_precision/quant.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

#ifndef LP_GPU_BACKEND_NAME
#define LP_GPU_BACKEND_NAME "cuda"
#endif

enum class Distribution { Random, Normal, Outlier };

struct Metrics {
    double max_abs = 0.0;
    double mae = 0.0;
    double mse = 0.0;
};

struct TimingStats {
    double p50_ms = 0.0;
    double min_ms = 0.0;
    double stddev_ms = 0.0;
};

struct Timings {
    TimingStats api_quant;
    TimingStats device_quant;
    TimingStats api_dequant;
    TimingStats device_dequant;
};

struct TimedSample {
    double api_ms = 0.0;
    double device_ms = 0.0;
};

struct GpuBuffers {
    float* input = nullptr;
    std::uint8_t* values = nullptr;
    std::uint8_t* scales = nullptr;
    float* output = nullptr;
};

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

template <typename T>
T* device_alloc(std::size_t count) {
    T* pointer = nullptr;
    check_cuda(cudaMalloc(&pointer, count * sizeof(T)), "cudaMalloc");
    return pointer;
}

void free_buffers(GpuBuffers& buffers) {
    cudaFree(buffers.output);
    cudaFree(buffers.scales);
    cudaFree(buffers.values);
    cudaFree(buffers.input);
    buffers = {};
}

std::vector<float> make_input(std::size_t rows,
                              std::size_t cols,
                              Distribution distribution) {
    std::mt19937 generator(20260824u);
    std::vector<float> input(rows * cols);
    if (distribution == Distribution::Random) {
        std::uniform_real_distribution<float> random(-1.0f, 1.0f);
        for (float& value : input) {
            value = random(generator);
        }
    } else {
        std::normal_distribution<float> normal(0.0f, 1.0f);
        for (float& value : input) {
            value = normal(generator);
        }
        if (distribution == Distribution::Outlier) {
            for (std::size_t index = 0; index < input.size(); index += 97) {
                input[index] *= 32.0f;
            }
        }
    }
    return input;
}

Metrics compare(const std::vector<float>& expected,
                const std::vector<float>& actual) {
    assert(expected.size() == actual.size());
    Metrics metrics;
    double squared_error = 0.0;
    double absolute_error = 0.0;
    for (std::size_t index = 0; index < expected.size(); ++index) {
        const double error = static_cast<double>(actual[index]) - expected[index];
        metrics.max_abs = std::max(metrics.max_abs, std::fabs(error));
        absolute_error += std::fabs(error);
        squared_error += error * error;
    }
    const double count = static_cast<double>(expected.size());
    metrics.mae = absolute_error / count;
    metrics.mse = squared_error / count;
    return metrics;
}

template <typename T, typename Decode>
Metrics compare_typed(const std::vector<float>& expected,
                      const std::vector<T>& actual,
                      Decode decode) {
    assert(expected.size() == actual.size());
    Metrics metrics;
    double squared_error = 0.0;
    double absolute_error = 0.0;
    for (std::size_t index = 0; index < expected.size(); ++index) {
        const double error = static_cast<double>(decode(actual[index])) - expected[index];
        metrics.max_abs = std::max(metrics.max_abs, std::fabs(error));
        absolute_error += std::fabs(error);
        squared_error += error * error;
    }
    const double count = static_cast<double>(expected.size());
    metrics.mae = absolute_error / count;
    metrics.mse = squared_error / count;
    return metrics;
}

std::vector<low_precision::Fp16> to_fp16(const std::vector<float>& input) {
    std::vector<low_precision::Fp16> output(input.size());
    for (std::size_t index = 0; index < input.size(); ++index) {
        output[index] = low_precision::fp16_from_float(input[index]);
    }
    return output;
}

template <typename T>
double milliseconds(T start, T end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

TimingStats summarize(std::vector<double> samples) {
    if (samples.empty()) {
        return {};
    }
    std::sort(samples.begin(), samples.end());
    const std::size_t middle = samples.size() / 2;
    const double p50 = samples.size() % 2 == 0
                           ? (samples[middle - 1] + samples[middle]) / 2.0
                           : samples[middle];
    const double mean = std::accumulate(samples.begin(), samples.end(), 0.0) /
                        static_cast<double>(samples.size());
    double squared_delta = 0.0;
    for (const double sample : samples) {
        const double delta = sample - mean;
        squared_delta += delta * delta;
    }
    return {p50, samples.front(),
            std::sqrt(squared_delta / static_cast<double>(samples.size()))};
}

TimingStats single_sample(double milliseconds_value) {
    return {milliseconds_value, milliseconds_value, 0.0};
}

struct CudaEvents {
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;

    CudaEvents() {
        check_cuda(cudaEventCreate(&start), "cudaEventCreate(start)");
        check_cuda(cudaEventCreate(&stop), "cudaEventCreate(stop)");
    }

    ~CudaEvents() {
        cudaEventDestroy(stop);
        cudaEventDestroy(start);
    }
};

template <typename Function>
TimedSample measure_cuda(Function&& function, CudaEvents& events) {
    check_cuda(cudaEventRecord(events.start), "cudaEventRecord(start)");
    const auto api_start = Clock::now();
    function();
    check_cuda(cudaEventRecord(events.stop), "cudaEventRecord(stop)");
    check_cuda(cudaEventSynchronize(events.stop), "cudaEventSynchronize(stop)");
    const auto api_end = Clock::now();
    float device_ms = 0.0f;
    check_cuda(cudaEventElapsedTime(&device_ms, events.start, events.stop),
               "cudaEventElapsedTime");
    return {milliseconds(api_start, api_end), static_cast<double>(device_ms)};
}

double effective_bandwidth(std::size_t bytes, double milliseconds_value) {
    if (milliseconds_value <= 0.0) {
        return 0.0;
    }
    return static_cast<double>(bytes) / (milliseconds_value * 1.0e6);
}

const char* distribution_name(Distribution distribution) {
    switch (distribution) {
        case Distribution::Random:
            return "random";
        case Distribution::Normal:
            return "normal";
        case Distribution::Outlier:
            return "outlier";
    }
    return "unknown";
}

std::size_t count_mismatches(const std::vector<std::uint8_t>& expected,
                             const std::vector<std::uint8_t>& actual) {
    assert(expected.size() == actual.size());
    std::size_t mismatches = 0;
    for (std::size_t index = 0; index < expected.size(); ++index) {
        mismatches += expected[index] != actual[index] ? 1 : 0;
    }
    return mismatches;
}

void print_header() {
    std::cout << "distribution,format,backend,input_type,output_type,rows,cols,elements,"
                 "api_quant_p50_ms,api_quant_min_ms,api_quant_std_ms,"
                 "device_quant_p50_ms,device_quant_min_ms,device_quant_std_ms,"
                 "api_dequant_p50_ms,api_dequant_min_ms,api_dequant_std_ms,"
                 "device_dequant_p50_ms,device_dequant_min_ms,device_dequant_std_ms,"
                 "max_abs,mae,mse,compression_ratio,"
                 "quant_read_bytes,quant_write_bytes,dequant_read_bytes,dequant_write_bytes,"
                 "api_quant_gb_s,device_quant_gb_s,api_dequant_gb_s,device_dequant_gb_s,"
                 "payload_mismatches\n";
}

void print_timing(const TimingStats& stats) {
    std::cout << std::fixed << std::setprecision(4) << stats.p50_ms << ','
              << stats.min_ms << ',' << stats.stddev_ms << ',';
}

void print_row(Distribution distribution,
               const char* format,
               const char* backend,
               const char* input_type,
               const char* output_type,
               std::size_t rows,
               std::size_t cols,
               const Timings& timings,
               const Metrics& metrics,
               double compression_ratio,
               std::size_t input_bytes,
               std::size_t output_bytes,
               std::size_t payload_bytes,
               std::size_t payload_mismatches) {
    std::cout << distribution_name(distribution) << ',' << format << ',' << backend << ','
              << input_type << ',' << output_type << ','
              << rows << ',' << cols << ',' << rows * cols << ','
              ;
    print_timing(timings.api_quant);
    print_timing(timings.device_quant);
    print_timing(timings.api_dequant);
    print_timing(timings.device_dequant);
    std::cout
              << std::scientific << std::setprecision(6) << metrics.max_abs << ','
              << metrics.mae << ',' << metrics.mse << ','
              << std::fixed << std::setprecision(4) << compression_ratio << ','
              << input_bytes << ',' << payload_bytes << ','
              << payload_bytes << ',' << output_bytes << ','
              << effective_bandwidth(input_bytes + payload_bytes,
                                     timings.api_quant.p50_ms) << ','
              << effective_bandwidth(input_bytes + payload_bytes,
                                     timings.device_quant.p50_ms) << ','
              << effective_bandwidth(input_bytes + payload_bytes,
                                     timings.api_dequant.p50_ms) << ','
              << effective_bandwidth(input_bytes + payload_bytes,
                                     timings.device_dequant.p50_ms) << ','
              << payload_mismatches << '\n';
}

void benchmark_mxfp8(const std::vector<float>& input,
                     std::size_t rows,
                     std::size_t cols,
                     Distribution distribution,
                     int repeats) {
    const auto cpu_quant_start = Clock::now();
    low_precision::MxFp8Tensor cpu_quantized =
        low_precision::quantize_mxfp8(input, rows, cols);
    const auto cpu_quant_end = Clock::now();
    const auto cpu_dequant_start = Clock::now();
    const std::vector<float> cpu_decoded = low_precision::dequantize_mxfp8(cpu_quantized);
    const auto cpu_dequant_end = Clock::now();
    const Metrics cpu_metrics = compare(input, cpu_decoded);
    const std::size_t payload_bytes = cpu_quantized.values.size() + cpu_quantized.scales.size();
    const std::size_t input_bytes = input.size() * sizeof(float);
    Timings cpu_timings;
    cpu_timings.api_quant = single_sample(milliseconds(cpu_quant_start, cpu_quant_end));
    cpu_timings.api_dequant = single_sample(milliseconds(cpu_dequant_start, cpu_dequant_end));
    print_row(distribution, "mxfp8", "cpu", "fp32", "fp32", rows, cols,
              cpu_timings, cpu_metrics,
              static_cast<double>(input_bytes) / payload_bytes,
              input_bytes, input_bytes, payload_bytes, 0);

    GpuBuffers buffers;
    buffers.input = device_alloc<float>(input.size());
    buffers.values = device_alloc<std::uint8_t>(cpu_quantized.values.size());
    buffers.scales = device_alloc<std::uint8_t>(cpu_quantized.scales.size());
    buffers.output = device_alloc<float>(input.size());
    check_cuda(cudaMemcpy(buffers.input, input.data(), input_bytes, cudaMemcpyHostToDevice),
               "copy input");

    low_precision::cuda_quantize_mxfp8(buffers.input, buffers.values, buffers.scales,
                                        rows, cols);
    low_precision::cuda_dequantize_mxfp8(buffers.values, buffers.scales, buffers.output,
                                          rows, cols);

    std::vector<std::uint8_t> gpu_values(cpu_quantized.values.size());
    std::vector<std::uint8_t> gpu_scales(cpu_quantized.scales.size());
    std::vector<float> gpu_decoded(input.size());
    std::vector<double> api_quant_samples;
    std::vector<double> device_quant_samples;
    std::vector<double> api_dequant_samples;
    std::vector<double> device_dequant_samples;
    api_quant_samples.reserve(static_cast<std::size_t>(repeats));
    device_quant_samples.reserve(static_cast<std::size_t>(repeats));
    api_dequant_samples.reserve(static_cast<std::size_t>(repeats));
    device_dequant_samples.reserve(static_cast<std::size_t>(repeats));
    CudaEvents events;
    for (int iteration = 0; iteration < repeats; ++iteration) {
        const TimedSample quant = measure_cuda(
            [&] {
                low_precision::cuda_quantize_mxfp8(buffers.input, buffers.values,
                                                   buffers.scales, rows, cols);
            },
            events);
        api_quant_samples.push_back(quant.api_ms);
        device_quant_samples.push_back(quant.device_ms);

        const TimedSample dequant = measure_cuda(
            [&] {
                low_precision::cuda_dequantize_mxfp8(buffers.values, buffers.scales,
                                                     buffers.output, rows, cols);
            },
            events);
        api_dequant_samples.push_back(dequant.api_ms);
        device_dequant_samples.push_back(dequant.device_ms);
    }
    check_cuda(cudaMemcpy(gpu_values.data(), buffers.values, gpu_values.size(),
                          cudaMemcpyDeviceToHost),
               "copy MXFP8 values");
    check_cuda(cudaMemcpy(gpu_scales.data(), buffers.scales, gpu_scales.size(),
                          cudaMemcpyDeviceToHost),
               "copy MXFP8 scales");
    check_cuda(cudaMemcpy(gpu_decoded.data(), buffers.output, input_bytes,
                          cudaMemcpyDeviceToHost),
               "copy MXFP8 output");
    const Metrics gpu_metrics = compare(input, gpu_decoded);
    Timings gpu_timings;
    gpu_timings.api_quant = summarize(std::move(api_quant_samples));
    gpu_timings.device_quant = summarize(std::move(device_quant_samples));
    gpu_timings.api_dequant = summarize(std::move(api_dequant_samples));
    gpu_timings.device_dequant = summarize(std::move(device_dequant_samples));
    print_row(distribution, "mxfp8", LP_GPU_BACKEND_NAME, "fp32", "fp32", rows, cols,
              gpu_timings, gpu_metrics,
              static_cast<double>(input_bytes) / payload_bytes,
              input_bytes, input_bytes, payload_bytes,
              count_mismatches(cpu_quantized.values, gpu_values) +
                  count_mismatches(cpu_quantized.scales, gpu_scales));
    free_buffers(buffers);
}

void benchmark_nvfp4(const std::vector<float>& input,
                     std::size_t rows,
                     std::size_t cols,
                     Distribution distribution,
                     int repeats) {
    const auto cpu_quant_start = Clock::now();
    low_precision::NvFp4Tensor cpu_quantized =
        low_precision::quantize_nvfp4(input, rows, cols);
    const auto cpu_quant_end = Clock::now();
    const auto cpu_dequant_start = Clock::now();
    const std::vector<float> cpu_decoded = low_precision::dequantize_nvfp4(cpu_quantized);
    const auto cpu_dequant_end = Clock::now();
    const Metrics cpu_metrics = compare(input, cpu_decoded);
    const std::size_t payload_bytes = cpu_quantized.values.size() +
                                      cpu_quantized.block_scales.size() + sizeof(float);
    const std::size_t input_bytes = input.size() * sizeof(float);
    Timings cpu_timings;
    cpu_timings.api_quant = single_sample(milliseconds(cpu_quant_start, cpu_quant_end));
    cpu_timings.api_dequant = single_sample(milliseconds(cpu_dequant_start, cpu_dequant_end));
    print_row(distribution, "nvfp4", "cpu", "fp32", "fp32", rows, cols,
              cpu_timings, cpu_metrics,
              static_cast<double>(input_bytes) / payload_bytes,
              input_bytes, input_bytes, payload_bytes, 0);

    GpuBuffers buffers;
    buffers.input = device_alloc<float>(input.size());
    buffers.values = device_alloc<std::uint8_t>(cpu_quantized.values.size());
    buffers.scales = device_alloc<std::uint8_t>(cpu_quantized.block_scales.size());
    buffers.output = device_alloc<float>(input.size());
    float gpu_global_scale = 0.0f;
    check_cuda(cudaMemcpy(buffers.input, input.data(), input_bytes, cudaMemcpyHostToDevice),
               "copy input");

    low_precision::cuda_quantize_nvfp4(buffers.input, buffers.values, buffers.scales,
                                        rows, cols, &gpu_global_scale);
    low_precision::cuda_dequantize_nvfp4(buffers.values, buffers.scales,
                                          gpu_global_scale, buffers.output, rows, cols);

    std::vector<std::uint8_t> gpu_values(cpu_quantized.values.size());
    std::vector<std::uint8_t> gpu_scales(cpu_quantized.block_scales.size());
    std::vector<float> gpu_decoded(input.size());
    std::vector<double> api_quant_samples;
    std::vector<double> device_quant_samples;
    std::vector<double> api_dequant_samples;
    std::vector<double> device_dequant_samples;
    api_quant_samples.reserve(static_cast<std::size_t>(repeats));
    device_quant_samples.reserve(static_cast<std::size_t>(repeats));
    api_dequant_samples.reserve(static_cast<std::size_t>(repeats));
    device_dequant_samples.reserve(static_cast<std::size_t>(repeats));
    CudaEvents events;
    for (int iteration = 0; iteration < repeats; ++iteration) {
        const TimedSample quant = measure_cuda(
            [&] {
                low_precision::cuda_quantize_nvfp4(buffers.input, buffers.values,
                                                   buffers.scales, rows, cols,
                                                   &gpu_global_scale);
            },
            events);
        api_quant_samples.push_back(quant.api_ms);
        device_quant_samples.push_back(quant.device_ms);

        const TimedSample dequant = measure_cuda(
            [&] {
                low_precision::cuda_dequantize_nvfp4(buffers.values, buffers.scales,
                                                     gpu_global_scale, buffers.output,
                                                     rows, cols);
            },
            events);
        api_dequant_samples.push_back(dequant.api_ms);
        device_dequant_samples.push_back(dequant.device_ms);
    }
    check_cuda(cudaMemcpy(gpu_values.data(), buffers.values, gpu_values.size(),
                          cudaMemcpyDeviceToHost),
               "copy NVFP4 values");
    check_cuda(cudaMemcpy(gpu_scales.data(), buffers.scales, gpu_scales.size(),
                          cudaMemcpyDeviceToHost),
               "copy NVFP4 scales");
    check_cuda(cudaMemcpy(gpu_decoded.data(), buffers.output, input_bytes,
                          cudaMemcpyDeviceToHost),
               "copy NVFP4 output");
    const Metrics gpu_metrics = compare(input, gpu_decoded);
    Timings gpu_timings;
    gpu_timings.api_quant = summarize(std::move(api_quant_samples));
    gpu_timings.device_quant = summarize(std::move(device_quant_samples));
    gpu_timings.api_dequant = summarize(std::move(api_dequant_samples));
    gpu_timings.device_dequant = summarize(std::move(device_dequant_samples));
    print_row(distribution, "nvfp4", LP_GPU_BACKEND_NAME, "fp32", "fp32", rows, cols,
              gpu_timings, gpu_metrics,
              static_cast<double>(input_bytes) / payload_bytes,
              input_bytes, input_bytes, payload_bytes,
              count_mismatches(cpu_quantized.values, gpu_values) +
                  count_mismatches(cpu_quantized.block_scales, gpu_scales) +
                  (gpu_global_scale != cpu_quantized.global_scale ? 1 : 0));
    free_buffers(buffers);
}

template <typename Output, typename Decode>
void benchmark_mxfp8_typed(const std::vector<float>& input,
                            std::size_t rows,
                            std::size_t cols,
                            Distribution distribution,
                            int repeats,
                            const char* output_type,
                            Decode decode) {
    const std::vector<low_precision::Fp16> half_input = to_fp16(input);
    const auto cpu_quantized = low_precision::quantize_mxfp8(half_input, rows, cols);
    const std::size_t payload_bytes = cpu_quantized.values.size() + cpu_quantized.scales.size();
    const std::size_t input_bytes = half_input.size() * sizeof(low_precision::Fp16);
    const std::size_t output_bytes = input.size() * sizeof(Output);

    auto* d_input = device_alloc<low_precision::Fp16>(half_input.size());
    auto* d_values = device_alloc<std::uint8_t>(cpu_quantized.values.size());
    auto* d_scales = device_alloc<std::uint8_t>(cpu_quantized.scales.size());
    auto* d_output = device_alloc<Output>(input.size());
    check_cuda(cudaMemcpy(d_input, half_input.data(), input_bytes, cudaMemcpyHostToDevice),
               "copy typed input");
    low_precision::cuda_quantize_mxfp8(d_input, d_values, d_scales, rows, cols);
    low_precision::cuda_dequantize_mxfp8(d_values, d_scales, d_output, rows, cols);

    std::vector<std::uint8_t> gpu_values(cpu_quantized.values.size());
    std::vector<std::uint8_t> gpu_scales(cpu_quantized.scales.size());
    std::vector<Output> gpu_output(input.size());
    std::vector<double> api_quant_samples;
    std::vector<double> device_quant_samples;
    std::vector<double> api_dequant_samples;
    std::vector<double> device_dequant_samples;
    api_quant_samples.reserve(static_cast<std::size_t>(repeats));
    device_quant_samples.reserve(static_cast<std::size_t>(repeats));
    api_dequant_samples.reserve(static_cast<std::size_t>(repeats));
    device_dequant_samples.reserve(static_cast<std::size_t>(repeats));
    CudaEvents events;
    for (int iteration = 0; iteration < repeats; ++iteration) {
        const TimedSample quant = measure_cuda(
            [&] {
                low_precision::cuda_quantize_mxfp8(d_input, d_values, d_scales,
                                                   rows, cols);
            },
            events);
        api_quant_samples.push_back(quant.api_ms);
        device_quant_samples.push_back(quant.device_ms);
        const TimedSample dequant = measure_cuda(
            [&] {
                low_precision::cuda_dequantize_mxfp8(d_values, d_scales, d_output,
                                                     rows, cols);
            },
            events);
        api_dequant_samples.push_back(dequant.api_ms);
        device_dequant_samples.push_back(dequant.device_ms);
    }
    check_cuda(cudaMemcpy(gpu_values.data(), d_values, gpu_values.size(), cudaMemcpyDeviceToHost),
               "copy typed MXFP8 values");
    check_cuda(cudaMemcpy(gpu_scales.data(), d_scales, gpu_scales.size(), cudaMemcpyDeviceToHost),
               "copy typed MXFP8 scales");
    check_cuda(cudaMemcpy(gpu_output.data(), d_output, output_bytes, cudaMemcpyDeviceToHost),
               "copy typed MXFP8 output");
    Timings timings;
    timings.api_quant = summarize(std::move(api_quant_samples));
    timings.device_quant = summarize(std::move(device_quant_samples));
    timings.api_dequant = summarize(std::move(api_dequant_samples));
    timings.device_dequant = summarize(std::move(device_dequant_samples));
    const Metrics metrics = compare_typed(input, gpu_output, decode);
    print_row(distribution, "mxfp8", LP_GPU_BACKEND_NAME, "fp16", output_type, rows, cols,
              timings, metrics,
              static_cast<double>(input_bytes) / payload_bytes,
              input_bytes, output_bytes, payload_bytes,
              count_mismatches(cpu_quantized.values, gpu_values) +
                  count_mismatches(cpu_quantized.scales, gpu_scales));
    cudaFree(d_output);
    cudaFree(d_scales);
    cudaFree(d_values);
    cudaFree(d_input);
}

template <typename Output, typename Decode>
void benchmark_nvfp4_typed(const std::vector<float>& input,
                            std::size_t rows,
                            std::size_t cols,
                            Distribution distribution,
                            int repeats,
                            const char* output_type,
                            Decode decode) {
    const std::vector<low_precision::Fp16> half_input = to_fp16(input);
    const auto cpu_quantized = low_precision::quantize_nvfp4(half_input, rows, cols);
    const std::size_t payload_bytes = cpu_quantized.values.size() +
                                      cpu_quantized.block_scales.size() + sizeof(float);
    const std::size_t input_bytes = half_input.size() * sizeof(low_precision::Fp16);
    const std::size_t output_bytes = input.size() * sizeof(Output);

    auto* d_input = device_alloc<low_precision::Fp16>(half_input.size());
    auto* d_values = device_alloc<std::uint8_t>(cpu_quantized.values.size());
    auto* d_scales = device_alloc<std::uint8_t>(cpu_quantized.block_scales.size());
    auto* d_output = device_alloc<Output>(input.size());
    float global_scale = 0.0f;
    check_cuda(cudaMemcpy(d_input, half_input.data(), input_bytes, cudaMemcpyHostToDevice),
               "copy typed input");
    low_precision::cuda_quantize_nvfp4(d_input, d_values, d_scales, rows, cols,
                                       &global_scale);
    low_precision::cuda_dequantize_nvfp4(d_values, d_scales, global_scale,
                                         d_output, rows, cols);

    std::vector<std::uint8_t> gpu_values(cpu_quantized.values.size());
    std::vector<std::uint8_t> gpu_scales(cpu_quantized.block_scales.size());
    std::vector<Output> gpu_output(input.size());
    std::vector<double> api_quant_samples;
    std::vector<double> device_quant_samples;
    std::vector<double> api_dequant_samples;
    std::vector<double> device_dequant_samples;
    api_quant_samples.reserve(static_cast<std::size_t>(repeats));
    device_quant_samples.reserve(static_cast<std::size_t>(repeats));
    api_dequant_samples.reserve(static_cast<std::size_t>(repeats));
    device_dequant_samples.reserve(static_cast<std::size_t>(repeats));
    CudaEvents events;
    for (int iteration = 0; iteration < repeats; ++iteration) {
        const TimedSample quant = measure_cuda(
            [&] {
                low_precision::cuda_quantize_nvfp4(d_input, d_values, d_scales,
                                                   rows, cols, &global_scale);
            },
            events);
        api_quant_samples.push_back(quant.api_ms);
        device_quant_samples.push_back(quant.device_ms);
        const TimedSample dequant = measure_cuda(
            [&] {
                low_precision::cuda_dequantize_nvfp4(d_values, d_scales, global_scale,
                                                     d_output, rows, cols);
            },
            events);
        api_dequant_samples.push_back(dequant.api_ms);
        device_dequant_samples.push_back(dequant.device_ms);
    }
    check_cuda(cudaMemcpy(gpu_values.data(), d_values, gpu_values.size(), cudaMemcpyDeviceToHost),
               "copy typed NVFP4 values");
    check_cuda(cudaMemcpy(gpu_scales.data(), d_scales, gpu_scales.size(), cudaMemcpyDeviceToHost),
               "copy typed NVFP4 scales");
    check_cuda(cudaMemcpy(gpu_output.data(), d_output, output_bytes, cudaMemcpyDeviceToHost),
               "copy typed NVFP4 output");
    Timings timings;
    timings.api_quant = summarize(std::move(api_quant_samples));
    timings.device_quant = summarize(std::move(device_quant_samples));
    timings.api_dequant = summarize(std::move(api_dequant_samples));
    timings.device_dequant = summarize(std::move(device_dequant_samples));
    const Metrics metrics = compare_typed(input, gpu_output, decode);
    print_row(distribution, "nvfp4", LP_GPU_BACKEND_NAME, "fp16", output_type, rows, cols,
              timings, metrics,
              static_cast<double>(input_bytes) / payload_bytes,
              input_bytes, output_bytes, payload_bytes,
              count_mismatches(cpu_quantized.values, gpu_values) +
                  count_mismatches(cpu_quantized.block_scales, gpu_scales) +
                  (global_scale != cpu_quantized.global_scale ? 1 : 0));
    cudaFree(d_output);
    cudaFree(d_scales);
    cudaFree(d_values);
    cudaFree(d_input);
}

}  // namespace

int main(int argc, char** argv) {
    try {
        std::vector<std::pair<std::size_t, std::size_t>> sizes;
        int repeats = 5;
        if (argc > 1 && std::string(argv[1]) == "--sweep") {
            repeats = argc > 2 ? std::stoi(argv[2]) : 5;
            sizes = {{256, 256}, {1024, 2048}, {4096, 4096}};
        } else {
            const std::size_t rows = argc > 1 ? std::stoull(argv[1]) : 1024;
            const std::size_t cols = argc > 2 ? std::stoull(argv[2]) : 2048;
            repeats = argc > 3 ? std::stoi(argv[3]) : 5;
            sizes.emplace_back(rows, cols);
        }
        if (repeats <= 0) {
            throw std::invalid_argument("repeats must be positive");
        }
        for (const auto [rows, cols] : sizes) {
            if (rows == 0 || cols == 0) {
                throw std::invalid_argument("rows and cols must be positive");
            }
        }

        int device_count = 0;
        check_cuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
        if (device_count == 0) {
            std::cerr << "No CUDA device found\n";
            return 1;
        }
        check_cuda(cudaSetDevice(0), "cudaSetDevice");
        std::cout << "# device=";
        cudaDeviceProp properties{};
        check_cuda(cudaGetDeviceProperties(&properties, 0), "cudaGetDeviceProperties");
        std::cout << properties.name << ",repeats=" << repeats
                  << ",sizes=" << sizes.size() << '\n';
        print_header();

        for (const auto [rows, cols] : sizes) {
            for (const Distribution distribution : {Distribution::Random,
                                                     Distribution::Normal,
                                                     Distribution::Outlier}) {
                const std::vector<float> input = make_input(rows, cols, distribution);
                benchmark_mxfp8(input, rows, cols, distribution, repeats);
                benchmark_nvfp4(input, rows, cols, distribution, repeats);
                benchmark_mxfp8_typed<low_precision::Fp16>(
                    input, rows, cols, distribution, repeats,
                    "fp16", low_precision::fp16_to_float);
                benchmark_mxfp8_typed<low_precision::Bf16>(
                    input, rows, cols, distribution, repeats,
                    "bf16", low_precision::bf16_to_float);
                benchmark_nvfp4_typed<low_precision::Fp16>(
                    input, rows, cols, distribution, repeats,
                    "fp16", low_precision::fp16_to_float);
                benchmark_nvfp4_typed<low_precision::Bf16>(
                    input, rows, cols, distribution, repeats,
                    "bf16", low_precision::bf16_to_float);
            }
        }
    } catch (const std::exception& error) {
        std::cerr << "benchmark failed: " << error.what() << '\n';
        return 1;
    }
}
