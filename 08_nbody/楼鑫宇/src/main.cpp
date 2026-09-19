#include <chrono>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>

#include "nbody/config.hpp"
#include "nbody/cpu_solver.hpp"
#ifdef NBODY_WITH_CUDA
#include "nbody/cuda_solver.hpp"
#endif
#ifdef NBODY_WITH_MACA
#include "nbody/maca_solver.hpp"
#endif
#include "nbody/io.hpp"

namespace {

struct Arguments {
    std::filesystem::path input;
    std::filesystem::path config;
    std::filesystem::path output;
    std::string backend = "cpu";
    int block_size = 128;
    double theta = 0.5;
    bool theta_supplied = false;
    nbody::RunOptions run_options;
};

void print_usage(const char* program) {
    std::cerr << "用法: " << program
              << " --input particles.txt --config simulation.cfg --output results/run"
              << " [--backend cpu|cuda-naive|cuda-tiled|cuda-mixed|cuda-bh|maca-naive|maca-tiled]"
              << " [--block-size 64|128|256|512]"
              << " [--theta 0.5 (仅 cuda-bh)]"
              << " [--record on|off] [--diagnostics final|off]\n";
}

Arguments parse_arguments(int argc, char** argv) {
    Arguments arguments;
    for (int i = 1; i < argc; ++i) {
        const std::string option = argv[i];
        if (option == "--backend") {
            if (++i >= argc) throw std::runtime_error("--backend 缺少值");
            arguments.backend = argv[i];
            if (arguments.backend != "cpu" &&
                arguments.backend != "cuda-naive" &&
                arguments.backend != "cuda-tiled" &&
                arguments.backend != "cuda-mixed" &&
                arguments.backend != "cuda-bh" &&
                arguments.backend != "maca-naive" &&
                arguments.backend != "maca-tiled") {
                throw std::runtime_error(
                    "未知的 --backend；支持 cpu、cuda-* 和 maca-*");
            }
            continue;
        }
        if (option == "--block-size") {
            if (++i >= argc) throw std::runtime_error("--block-size 缺少值");
            std::size_t used = 0;
            try {
                arguments.block_size = std::stoi(argv[i], &used);
            } catch (const std::exception&) {
                throw std::runtime_error("--block-size 必须是整数");
            }
            if (used != std::string(argv[i]).size() ||
                (arguments.block_size != 64 && arguments.block_size != 128 &&
                 arguments.block_size != 256 && arguments.block_size != 512)) {
                throw std::runtime_error("--block-size 只支持 64、128、256 或 512");
            }
            continue;
        }
        if (option == "--theta") {
            if (++i >= argc) throw std::runtime_error("--theta 缺少值");
            std::size_t used = 0;
            try {
                arguments.theta = std::stod(argv[i], &used);
            } catch (const std::exception&) {
                throw std::runtime_error("--theta 必须是正数");
            }
            if (used != std::string(argv[i]).size() ||
                !std::isfinite(arguments.theta) ||
                !(arguments.theta > 0.0)) {
                throw std::runtime_error("--theta 必须是有限正数");
            }
            arguments.theta_supplied = true;
            continue;
        }
        if (option == "--record") {
            if (++i >= argc) throw std::runtime_error("--record 缺少值");
            const std::string value = argv[i];
            if (value != "on" && value != "off") {
                throw std::runtime_error("--record 只支持 on 或 off");
            }
            arguments.run_options.record_enabled = value == "on";
            continue;
        }
        if (option == "--diagnostics") {
            if (++i >= argc) throw std::runtime_error("--diagnostics 缺少值");
            const std::string value = argv[i];
            if (value != "final" && value != "off") {
                throw std::runtime_error("--diagnostics 只支持 final 或 off");
            }
            arguments.run_options.diagnostics_enabled = value == "final";
            continue;
        }
        if (option != "--input" && option != "--config" && option != "--output") {
            throw std::runtime_error("未知参数: " + option);
        }
        if (++i >= argc) throw std::runtime_error(option + " 缺少值");
        if (option == "--input") arguments.input = argv[i];
        if (option == "--config") arguments.config = argv[i];
        if (option == "--output") arguments.output = argv[i];
    }
    if (arguments.input.empty() || arguments.config.empty() || arguments.output.empty()) {
        throw std::runtime_error("必须提供 --input、--config 和 --output");
    }
    if (arguments.theta_supplied && arguments.backend != "cuda-bh") {
        throw std::runtime_error("--theta 只适用于 cuda-bh");
    }
    return arguments;
}

}  // namespace

int main(int argc, char** argv) {
    using Clock = std::chrono::steady_clock;
    try {
        const auto wall_begin = Clock::now();
        const Arguments arguments = parse_arguments(argc, argv);
#ifndef NBODY_WITH_CUDA
        if (arguments.backend.rfind("cuda-", 0) == 0) {
            throw std::runtime_error("当前构建未启用 CUDA 后端");
        }
#endif
#ifndef NBODY_WITH_MACA
        if (arguments.backend.rfind("maca-", 0) == 0) {
            throw std::runtime_error("当前构建未启用 MACA 后端");
        }
#endif
        if (std::filesystem::exists(arguments.output)) {
            throw std::runtime_error("输出目录已存在，请使用新目录: " +
                                     arguments.output.string());
        }
        const nbody::Config config = nbody::read_config(arguments.config);
        const auto particles = nbody::read_particles(arguments.input);
        std::filesystem::create_directories(arguments.output);
        nbody::CpuRunResult result;
        std::string precision;
        if (arguments.backend == "cpu") {
            result = nbody::simulate_cpu(particles, config,
                                         arguments.run_options);
            precision = "fp64";
#ifdef NBODY_WITH_CUDA
        } else if (arguments.backend == "cuda-naive") {
            result = nbody::simulate_cuda_naive(particles, config,
                                                arguments.block_size,
                                                arguments.run_options);
            precision = "fp32";
        } else if (arguments.backend == "cuda-tiled") {
            result = nbody::simulate_cuda_tiled(particles, config,
                                                arguments.block_size,
                                                arguments.run_options);
            precision = "fp32";
        } else if (arguments.backend == "cuda-mixed") {
            result = nbody::simulate_cuda_mixed(particles, config,
                                                arguments.block_size,
                                                arguments.run_options);
            precision = "fp16-pair-fp32-accumulation";
        } else if (arguments.backend == "cuda-bh") {
            result = nbody::simulate_cuda_bh(particles, config,
                                             arguments.block_size,
                                             arguments.theta,
                                             arguments.run_options);
            precision = "fp32";
#endif
#ifdef NBODY_WITH_MACA
        } else if (arguments.backend == "maca-naive") {
            result = nbody::simulate_maca_naive(particles, config,
                                                arguments.block_size,
                                                arguments.run_options);
            precision = "fp32";
        } else if (arguments.backend == "maca-tiled") {
            result = nbody::simulate_maca_tiled(particles, config,
                                                arguments.block_size,
                                                arguments.run_options);
            precision = "fp32";
#endif
        } else {
            throw std::runtime_error("当前构建未启用后端: " + arguments.backend);
        }
        const double before_write_ms = std::chrono::duration<double, std::milli>(
            Clock::now() - wall_begin).count();
        nbody::write_run_outputs(arguments.output, arguments.input, config, result,
                                 before_write_ms, arguments.backend, precision,
                                 arguments.backend == "cuda-bh"
                                     ? std::optional<double>(arguments.theta)
                                     : std::nullopt);
        std::cout << arguments.backend << ' ' << precision << " 模拟完成\n"
                  << "粒子: " << particles.size() << ", 步数: " << config.num_steps
                  << ", 积分器: " << nbody::to_string(config.integrator) << '\n'
                  << "力计算: " << result.force_ms << " ms, 模拟: "
                  << result.simulation_ms << " ms\n"
                  << "输出目录: " << arguments.output << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "错误: " << error.what() << '\n';
        print_usage(argv[0]);
        return 1;
    }
}
