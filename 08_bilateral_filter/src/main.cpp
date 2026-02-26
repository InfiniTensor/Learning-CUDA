#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <vector>

#include "bilateral_filter.h"
#include "bilateral_filter_cuda.cuh"
#include "bilateral_filter_opencv.h"
#include "image_io.h"

constexpr int WARMUP_RUNS = 5;
constexpr int BENCHMARK_RUNS = 50;

void print_usage(const char* prog) {
    fprintf(stderr, "Usage:\n");
    fprintf(stderr, "  %s input.raw params.txt output.raw              # CPU only\n", prog);
    fprintf(stderr, "  %s --cuda input.raw params.txt output.raw       # CUDA only\n", prog);
    fprintf(stderr, "  %s --opencv input.raw params.txt output.raw     # OpenCV only\n", prog);
    fprintf(stderr,
            "  %s --bench input.raw params.txt                 # Benchmark CUDA vs OpenCV\n", prog);
    fprintf(stderr, "  %s --compare input.raw params.txt               # Compare CPU vs OpenCV\n",
            prog);
    fprintf(stderr,
            "  %s --compare-all input.raw params.txt           # Compare CPU vs CUDA vs OpenCV\n",
            prog);
}

struct BenchmarkResult {
    double mean_ms;
    double min_ms;
    double max_ms;
    double stddev_ms;
};

BenchmarkResult compute_stats(const std::vector<double>& times) {
    BenchmarkResult result = {0.0, 0.0, 0.0, 0.0};
    if (times.empty()) {
        return result;
    }

    double sum = std::accumulate(times.begin(), times.end(), 0.0);
    result.mean_ms = sum / times.size();

    result.min_ms = *std::min_element(times.begin(), times.end());
    result.max_ms = *std::max_element(times.begin(), times.end());

    double sq_sum = 0.0;
    for (double t : times) {
        sq_sum += (t - result.mean_ms) * (t - result.mean_ms);
    }
    result.stddev_ms = sqrt(sq_sum / times.size());

    return result;
}

int main(int argc, char* argv[]) {
    if (argc < 4) {
        print_usage(argv[0]);
        return 1;
    }

    bool compare_mode = (strcmp(argv[1], "--compare") == 0);
    bool compare_all_mode = (strcmp(argv[1], "--compare-all") == 0);
    bool bench_mode = (strcmp(argv[1], "--bench") == 0);
    bool cuda_only = (strcmp(argv[1], "--cuda") == 0);
    bool opencv_only = (strcmp(argv[1], "--opencv") == 0);

    const char* input_path;
    const char* params_path;
    const char* output_path = nullptr;

    if (compare_mode || compare_all_mode || bench_mode) {
        if (argc != 4) {
            print_usage(argv[0]);
            return 1;
        }
        input_path = argv[2];
        params_path = argv[3];
    } else if (cuda_only || opencv_only) {
        if (argc != 5) {
            print_usage(argv[0]);
            return 1;
        }
        input_path = argv[2];
        params_path = argv[3];
        output_path = argv[4];
    } else {
        if (argc != 4) {
            print_usage(argv[0]);
            return 1;
        }
        input_path = argv[1];
        params_path = argv[2];
        output_path = argv[3];
    }

    ImageData input;
    if (!load_raw_image(input_path, &input)) {
        return 1;
    }
    printf("Loaded image: %dx%d, channels=%d\n", input.width, input.height, input.channels);

    FilterParams params;
    if (!load_params(params_path, &params)) {
        return 1;
    }
    printf("Parameters: radius=%d, sigma_spatial=%.2f, sigma_color=%.2f\n", params.radius,
           params.sigma_spatial, params.sigma_color);

    double megapixels = static_cast<double>(input.width) * input.height / 1e6;

    if (compare_all_mode) {
        ImageData output_cpu, output_cuda, output_opencv;

        output_cuda.width = input.width;
        output_cuda.height = input.height;
        output_cuda.channels = input.channels;
        output_cuda.data.resize(input.size());

        printf("\nBenchmarking (warmup=%d, runs=%d)...\n", WARMUP_RUNS, BENCHMARK_RUNS);

        // Warmup runs
        for (int i = 0; i < WARMUP_RUNS; ++i) {
            apply_bilateral_filter_cpu(input, output_cpu, params);
            apply_bilateral_filter_cuda(input.data.data(), output_cuda.data.data(), input.width,
                                        input.height, input.channels, params.radius,
                                        params.sigma_spatial, params.sigma_color);
            apply_bilateral_filter_opencv(input, output_opencv, params);
        }

        // Benchmark CPU
        std::vector<double> cpu_times;
        for (int i = 0; i < BENCHMARK_RUNS; ++i) {
            auto start = std::chrono::high_resolution_clock::now();
            apply_bilateral_filter_cpu(input, output_cpu, params);
            auto end = std::chrono::high_resolution_clock::now();
            cpu_times.push_back(std::chrono::duration<double, std::milli>(end - start).count());
        }

        // Benchmark CUDA
        std::vector<double> cuda_times;
        for (int i = 0; i < BENCHMARK_RUNS; ++i) {
            auto start = std::chrono::high_resolution_clock::now();
            apply_bilateral_filter_cuda(input.data.data(), output_cuda.data.data(), input.width,
                                        input.height, input.channels, params.radius,
                                        params.sigma_spatial, params.sigma_color);
            auto end = std::chrono::high_resolution_clock::now();
            cuda_times.push_back(std::chrono::duration<double, std::milli>(end - start).count());
        }

        // Benchmark OpenCV
        std::vector<double> cv_times;
        for (int i = 0; i < BENCHMARK_RUNS; ++i) {
            auto start = std::chrono::high_resolution_clock::now();
            apply_bilateral_filter_opencv(input, output_opencv, params);
            auto end = std::chrono::high_resolution_clock::now();
            cv_times.push_back(std::chrono::duration<double, std::milli>(end - start).count());
        }

        auto cpu_stats = compute_stats(cpu_times);
        auto cuda_stats = compute_stats(cuda_times);
        auto cv_stats = compute_stats(cv_times);

        double mae_cpu_cv = compute_mae(output_cpu, output_opencv);
        double mae_cuda_cv = compute_mae(output_cuda, output_opencv);

        printf("\n=== Benchmark Results (mean ± stddev) ===\n");
        printf("CPU    : %.2f ± %.2f ms [min=%.2f, max=%.2f] (%.2f MP/s)\n", cpu_stats.mean_ms,
               cpu_stats.stddev_ms, cpu_stats.min_ms, cpu_stats.max_ms,
               megapixels / (cpu_stats.mean_ms / 1000.0));
        printf("CUDA   : %.2f ± %.2f ms [min=%.2f, max=%.2f] (%.2f MP/s)\n", cuda_stats.mean_ms,
               cuda_stats.stddev_ms, cuda_stats.min_ms, cuda_stats.max_ms,
               megapixels / (cuda_stats.mean_ms / 1000.0));
        printf("OpenCV : %.2f ± %.2f ms [min=%.2f, max=%.2f] (%.2f MP/s)\n", cv_stats.mean_ms,
               cv_stats.stddev_ms, cv_stats.min_ms, cv_stats.max_ms,
               megapixels / (cv_stats.mean_ms / 1000.0));

        printf("\nSpeedup (based on mean):\n");
        printf("  CUDA vs CPU:    %.2fx\n", cpu_stats.mean_ms / cuda_stats.mean_ms);
        printf("  CUDA vs OpenCV: %.2fx\n", cv_stats.mean_ms / cuda_stats.mean_ms);
        printf("  OpenCV vs CPU:  %.2fx\n", cpu_stats.mean_ms / cv_stats.mean_ms);

        printf("\nMAE (vs OpenCV):\n");
        printf("  CPU:  %.4f %s\n", mae_cpu_cv, mae_cpu_cv < 1.0 ? "✓" : "✗");
        printf("  CUDA: %.4f %s\n", mae_cuda_cv, mae_cuda_cv < 1.0 ? "✓" : "✗");

    } else if (bench_mode) {
        // Benchmark CUDA vs OpenCV only (skip slow CPU)
        ImageData output_cuda, output_opencv;

        output_cuda.width = input.width;
        output_cuda.height = input.height;
        output_cuda.channels = input.channels;
        output_cuda.data.resize(input.size());

        printf("\nBenchmarking CUDA vs OpenCV (warmup=%d, runs=%d)...\n", WARMUP_RUNS,
               BENCHMARK_RUNS);

        // Warmup
        for (int i = 0; i < WARMUP_RUNS; ++i) {
            apply_bilateral_filter_cuda(input.data.data(), output_cuda.data.data(), input.width,
                                        input.height, input.channels, params.radius,
                                        params.sigma_spatial, params.sigma_color);
            apply_bilateral_filter_opencv(input, output_opencv, params);
        }

        // Benchmark CUDA
        std::vector<double> cuda_times;
        for (int i = 0; i < BENCHMARK_RUNS; ++i) {
            auto start = std::chrono::high_resolution_clock::now();
            apply_bilateral_filter_cuda(input.data.data(), output_cuda.data.data(), input.width,
                                        input.height, input.channels, params.radius,
                                        params.sigma_spatial, params.sigma_color);
            auto end = std::chrono::high_resolution_clock::now();
            cuda_times.push_back(std::chrono::duration<double, std::milli>(end - start).count());
        }

        // Benchmark OpenCV
        std::vector<double> cv_times;
        for (int i = 0; i < BENCHMARK_RUNS; ++i) {
            auto start = std::chrono::high_resolution_clock::now();
            apply_bilateral_filter_opencv(input, output_opencv, params);
            auto end = std::chrono::high_resolution_clock::now();
            cv_times.push_back(std::chrono::duration<double, std::milli>(end - start).count());
        }

        auto cuda_stats = compute_stats(cuda_times);
        auto cv_stats = compute_stats(cv_times);
        double mae = compute_mae(output_cuda, output_opencv);

        printf("\n=== Benchmark Results ===\n");
        printf("CUDA   : %.2f ± %.2f ms [min=%.2f, max=%.2f] (%.2f MP/s)\n", cuda_stats.mean_ms,
               cuda_stats.stddev_ms, cuda_stats.min_ms, cuda_stats.max_ms,
               megapixels / (cuda_stats.mean_ms / 1000.0));
        printf("OpenCV : %.2f ± %.2f ms [min=%.2f, max=%.2f] (%.2f MP/s)\n", cv_stats.mean_ms,
               cv_stats.stddev_ms, cv_stats.min_ms, cv_stats.max_ms,
               megapixels / (cv_stats.mean_ms / 1000.0));
        printf("\nSpeedup: CUDA is %.2fx %s than OpenCV\n",
               cv_stats.mean_ms > cuda_stats.mean_ms ? cv_stats.mean_ms / cuda_stats.mean_ms
                                                     : cuda_stats.mean_ms / cv_stats.mean_ms,
               cv_stats.mean_ms > cuda_stats.mean_ms ? "faster" : "slower");
        printf("MAE: %.4f %s\n", mae, mae < 1.0 ? "✓" : "✗");

    } else if (compare_mode) {
        ImageData output_cpu, output_opencv;

        printf("\nBenchmarking (warmup=%d, runs=%d)...\n", WARMUP_RUNS, BENCHMARK_RUNS);

        // Warmup
        for (int i = 0; i < WARMUP_RUNS; ++i) {
            apply_bilateral_filter_cpu(input, output_cpu, params);
            apply_bilateral_filter_opencv(input, output_opencv, params);
        }

        // Benchmark
        std::vector<double> cpu_times, cv_times;
        for (int i = 0; i < BENCHMARK_RUNS; ++i) {
            auto start = std::chrono::high_resolution_clock::now();
            apply_bilateral_filter_cpu(input, output_cpu, params);
            auto end = std::chrono::high_resolution_clock::now();
            cpu_times.push_back(std::chrono::duration<double, std::milli>(end - start).count());
        }
        for (int i = 0; i < BENCHMARK_RUNS; ++i) {
            auto start = std::chrono::high_resolution_clock::now();
            apply_bilateral_filter_opencv(input, output_opencv, params);
            auto end = std::chrono::high_resolution_clock::now();
            cv_times.push_back(std::chrono::duration<double, std::milli>(end - start).count());
        }

        auto cpu_stats = compute_stats(cpu_times);
        auto cv_stats = compute_stats(cv_times);
        double mae = compute_mae(output_cpu, output_opencv);

        printf("\n=== Benchmark Results ===\n");
        printf("CPU    : %.2f ± %.2f ms (%.2f MP/s)\n", cpu_stats.mean_ms, cpu_stats.stddev_ms,
               megapixels / (cpu_stats.mean_ms / 1000.0));
        printf("OpenCV : %.2f ± %.2f ms (%.2f MP/s)\n", cv_stats.mean_ms, cv_stats.stddev_ms,
               megapixels / (cv_stats.mean_ms / 1000.0));
        printf("Speedup: %.2fx\n", cpu_stats.mean_ms / cv_stats.mean_ms);
        printf("MAE: %.4f %s\n", mae, mae < 1.0 ? "✓" : "✗");

    } else if (cuda_only) {
        ImageData output;
        output.width = input.width;
        output.height = input.height;
        output.channels = input.channels;
        output.data.resize(input.size());

        // Warmup
        for (int i = 0; i < WARMUP_RUNS; ++i) {
            apply_bilateral_filter_cuda(input.data.data(), output.data.data(), input.width,
                                        input.height, input.channels, params.radius,
                                        params.sigma_spatial, params.sigma_color);
        }

        // Benchmark
        std::vector<double> times;
        for (int i = 0; i < BENCHMARK_RUNS; ++i) {
            auto start = std::chrono::high_resolution_clock::now();
            apply_bilateral_filter_cuda(input.data.data(), output.data.data(), input.width,
                                        input.height, input.channels, params.radius,
                                        params.sigma_spatial, params.sigma_color);
            auto end = std::chrono::high_resolution_clock::now();
            times.push_back(std::chrono::duration<double, std::milli>(end - start).count());
        }

        auto stats = compute_stats(times);
        printf("CUDA: %.2f ± %.2f ms [min=%.2f, max=%.2f] (%.2f MP/s)\n", stats.mean_ms,
               stats.stddev_ms, stats.min_ms, stats.max_ms, megapixels / (stats.mean_ms / 1000.0));

        if (!save_raw_image(output_path, output)) {
            return 1;
        }
        printf("Output saved to: %s\n", output_path);

    } else if (opencv_only) {
        ImageData output;

        // Warmup
        for (int i = 0; i < WARMUP_RUNS; ++i) {
            apply_bilateral_filter_opencv(input, output, params);
        }

        // Benchmark
        std::vector<double> times;
        for (int i = 0; i < BENCHMARK_RUNS; ++i) {
            auto start = std::chrono::high_resolution_clock::now();
            apply_bilateral_filter_opencv(input, output, params);
            auto end = std::chrono::high_resolution_clock::now();
            times.push_back(std::chrono::duration<double, std::milli>(end - start).count());
        }

        auto stats = compute_stats(times);
        printf("OpenCV: %.2f ± %.2f ms (%.2f MP/s)\n", stats.mean_ms, stats.stddev_ms,
               megapixels / (stats.mean_ms / 1000.0));

        if (!save_raw_image(output_path, output)) {
            return 1;
        }
        printf("Output saved to: %s\n", output_path);

    } else {
        ImageData output;

        auto start = std::chrono::high_resolution_clock::now();
        apply_bilateral_filter_cpu(input, output, params);
        auto end = std::chrono::high_resolution_clock::now();

        double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
        printf("CPU time: %.2f ms (%.2f MP/s)\n", elapsed_ms, megapixels / (elapsed_ms / 1000.0));

        if (!save_raw_image(output_path, output)) {
            return 1;
        }
        printf("Output saved to: %s\n", output_path);
    }

    return 0;
}
