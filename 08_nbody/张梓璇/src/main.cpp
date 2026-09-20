#include "cpu_reference.h"
#include "io.h"
#include "nbody.cuh"

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct CommandLine {
    std::string particle_path;
    std::string config_path;
    std::string trajectory_path = "trajectory.bin";
    std::string performance_path = "performance.log";
    std::string kernel_override;
    int block_size_override = 0;
    int source_chunks_override = 0;
    int profile_steps = 0;
    std::string cpu_validation = "auto";
    bool skip_conservation = false;
    bool cuda_graph = false;
};

void print_usage(const char* program) {
    std::cout
        << "用法: " << program << " <particles.txt> <config.txt> [选项]\n\n"
        << "选项:\n"
        << "  --kernel MODE             baseline、tiled、tiled-float4 或 split-source\n"
        << "  --block-size N            覆盖配置文件中的 block_size\n"
        << "  --source-chunks N         split-source 的 source 分块数，默认 2\n"
        << "  --profile-steps N         只运行前 N 步，方便 ncu 少量 replay\n"
        << "  --output PATH             轨迹文件路径，默认 trajectory.bin\n"
        << "  --performance-log PATH    性能日志路径，默认 performance.log\n"
        << "  --cpu-validation MODE     auto、on 或 off；auto 只验证 N <= 1024\n"
        << "  --skip-conservation       跳过 CPU O(N^2) 守恒量计算，大规模 profiler 更快\n"
        << "  --cuda-graph              实验性地用 CUDA Graph 提交每一步，默认关闭\n"
        << "  --help                    显示帮助\n";
}

CommandLine parse_arguments(int argc, char** argv) {
    if (argc == 2 && std::string(argv[1]) == "--help") {
        print_usage(argv[0]);
        std::exit(0);
    }
    if (argc < 3) {
        print_usage(argv[0]);
        throw std::runtime_error("缺少粒子文件或配置文件");
    }

    CommandLine args;
    args.particle_path = argv[1];
    args.config_path = argv[2];
    for (int i = 3; i < argc; ++i) {
        const std::string option = argv[i];
        if (option == "--skip-conservation") {
            args.skip_conservation = true;
            continue;
        }
        if (option == "--cuda-graph") {
            args.cuda_graph = true;
            continue;
        }
        if (option == "--kernel" || option == "--block-size" ||
            option == "--profile-steps" ||
            option == "--source-chunks" ||
            option == "--output" || option == "--performance-log" ||
            option == "--cpu-validation") {
            if (i + 1 >= argc) {
                throw std::runtime_error("选项缺少参数: " + option);
            }
            const std::string value = argv[++i];
            if (option == "--kernel") args.kernel_override = value;
            else if (option == "--block-size") {
                try {
                    args.block_size_override = std::stoi(value);
                } catch (const std::exception&) {
                    throw std::runtime_error("--block-size 必须是整数");
                }
                if (args.block_size_override <= 0) {
                    throw std::runtime_error("--block-size 必须在 1 到 1024 之间");
                }
            }
            else if (option == "--profile-steps") {
                try {
                    args.profile_steps = std::stoi(value);
                } catch (const std::exception&) {
                    throw std::runtime_error("--profile-steps 必须是正整数");
                }
                if (args.profile_steps <= 0) {
                    throw std::runtime_error("--profile-steps 必须是正整数");
                }
            }
            else if (option == "--source-chunks") {
                try {
                    args.source_chunks_override = std::stoi(value);
                } catch (const std::exception&) {
                    throw std::runtime_error("--source-chunks 必须是整数");
                }
            }
            else if (option == "--output") args.trajectory_path = value;
            else if (option == "--performance-log") args.performance_path = value;
            else args.cpu_validation = value;
        } else {
            throw std::runtime_error("未知命令行选项: " + option);
        }
    }
    if (!args.kernel_override.empty() && args.kernel_override != "baseline" &&
        args.kernel_override != "tiled" && args.kernel_override != "tiled-float4" &&
        args.kernel_override != "split-source") {
        throw std::runtime_error(
            "--kernel 只支持 baseline、tiled、tiled-float4 或 split-source");
    }
    if (args.block_size_override < 0 || args.block_size_override > 1024) {
        throw std::runtime_error("--block-size 必须在 1 到 1024 之间");
    }
    if (args.source_chunks_override < 0 || args.source_chunks_override > 64) {
        throw std::runtime_error("--source-chunks 必须在 1 到 64 之间");
    }
    if (args.cpu_validation != "auto" && args.cpu_validation != "on" &&
        args.cpu_validation != "off") {
        throw std::runtime_error("--cpu-validation 只支持 auto、on 或 off");
    }
    return args;
}

void print_conservation(const ConservationValues& initial,
                        const ConservationValues& final) {
    const double momentum_error = std::sqrt(
        (final.px - initial.px) * (final.px - initial.px) +
        (final.py - initial.py) * (final.py - initial.py) +
        (final.pz - initial.pz) * (final.pz - initial.pz));
    const double relative_energy_error = std::abs(initial.total_energy) > 0.0
        ? std::abs((final.total_energy - initial.total_energy) / initial.total_energy)
        : std::abs(final.total_energy - initial.total_energy);

    std::cout << std::setprecision(10)
              << "initial momentum: " << initial.px << ' ' << initial.py << ' ' << initial.pz << '\n'
              << "final momentum:   " << final.px << ' ' << final.py << ' ' << final.pz << '\n'
              << "momentum error:   " << momentum_error << '\n'
              << "initial energy:   " << initial.total_energy << '\n'
              << "final energy:     " << final.total_energy << '\n'
              << "relative energy error: " << relative_energy_error << '\n';
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const CommandLine args = parse_arguments(argc, argv);
        const auto input_start = std::chrono::steady_clock::now();
        const std::vector<Particle> initial = read_particles(args.particle_path);
        SimConfig config = read_config(args.config_path);
        if (!args.kernel_override.empty()) config.kernel = args.kernel_override;
        if (args.block_size_override > 0) config.block_size = args.block_size_override;
        if (args.source_chunks_override > 0) config.source_chunks = args.source_chunks_override;
        if (args.cuda_graph) config.cuda_graph = true;
        if (args.profile_steps > 0 && args.profile_steps < config.num_steps) {
            config.num_steps = args.profile_steps;
        }
        const double input_read_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - input_start).count();

        ConservationValues initial_values;
        if (!args.skip_conservation) {
            initial_values = compute_conservation(initial, config.G, config.softening);
        }

        std::cout << "particles: " << initial.size() << '\n'
                  << "steps: " << config.num_steps << '\n'
                  << "integrator: " << config.integrator << '\n'
                  << "kernel: " << config.kernel << '\n'
                  << "block size: " << config.block_size << '\n';
        if (config.kernel == "split-source") {
            std::cout << "source chunks: " << config.source_chunks << '\n';
        }
        std::cout << "CUDA Graph: " << (config.cuda_graph ? "on" : "off") << '\n';

        GpuSimulationResult gpu = simulate_gpu(initial, config, args.trajectory_path);
        gpu.performance.input_read_ms = input_read_ms;
        ConservationValues final_values;
        if (!args.skip_conservation) {
            final_values = compute_conservation(gpu.particles, config.G, config.softening);
        } else {
            gpu.performance.conservation_available = false;
        }

        const bool run_cpu = args.cpu_validation == "on" ||
            (args.cpu_validation == "auto" && initial.size() <= 1024);
        if (run_cpu) {
            std::vector<Particle> cpu_particles = initial;
            const auto cpu_start = std::chrono::steady_clock::now();
            simulate_cpu(cpu_particles, config);
            gpu.performance.cpu_simulation_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - cpu_start).count();
            state_error(cpu_particles, gpu.particles,
                        gpu.performance.max_position_error,
                        gpu.performance.rms_position_error,
                        gpu.performance.max_velocity_error,
                        gpu.performance.rms_velocity_error);
            std::cout << "CPU validation max position error: "
                      << gpu.performance.max_position_error << '\n'
                      << "CPU validation RMS position error: "
                      << gpu.performance.rms_position_error << '\n'
                      << "CPU validation max velocity error: "
                      << gpu.performance.max_velocity_error << '\n'
                      << "CPU validation RMS velocity error: "
                      << gpu.performance.rms_velocity_error << '\n';
        } else {
            std::cout << "CPU validation: skipped (use --cpu-validation on to force it)\n";
        }

        if (gpu.performance.conservation_available) {
            print_conservation(initial_values, final_values);
        } else {
            std::cout << "conservation: skipped\n";
        }
        write_performance_log(args.performance_path,
                              static_cast<int>(initial.size()),
                              config,
                              config.kernel,
                              gpu.performance,
                              initial_values,
                              final_values);

        std::cout << "GPU simulation time: " << gpu.performance.gpu_kernel_ms << " ms\n"
                  << "simulation wall time: " << gpu.performance.simulation_wall_ms << " ms\n"
                  << "trajectory: " << std::filesystem::absolute(args.trajectory_path).string() << '\n'
                  << "performance log: " << std::filesystem::absolute(args.performance_path).string() << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "错误: " << error.what() << '\n';
        return 1;
    }
}
