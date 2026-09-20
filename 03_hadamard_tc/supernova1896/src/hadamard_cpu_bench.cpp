#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Args {
    int rows = 2 * 512 * 16;
    int dim = 128;
    int repeats = 10;
};

Args parse_args(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        auto need_value = [&](const char* name) -> int {
            if (i + 1 >= argc) {
                throw std::invalid_argument(std::string("缺少参数：") + name);
            }
            return std::stoi(argv[++i]);
        };
        if (key == "--rows") {
            args.rows = need_value("--rows");
        } else if (key == "--dim") {
            args.dim = need_value("--dim");
        } else if (key == "--repeats") {
            args.repeats = need_value("--repeats");
        } else if (key == "--help") {
            std::cout << "用法：hadamard_cpu_bench [--rows N] [--dim 64|128|256] [--repeats N]\n";
            std::exit(0);
        } else {
            throw std::invalid_argument("未知参数：" + key);
        }
    }
    if (args.rows <= 0 || args.repeats <= 0 || args.dim <= 0 || (args.dim & (args.dim - 1)) != 0) {
        throw std::invalid_argument("rows/repeats 必须为正数，dim 必须是 2 的幂");
    }
    return args;
}

void cpu_fht(float* values, int dim) {
    for (int stride = 1; stride < dim; stride <<= 1) {
        for (int base = 0; base < dim; base += 2 * stride) {
            for (int i = 0; i < stride; ++i) {
                const float a = values[base + i];
                const float b = values[base + i + stride];
                values[base + i] = a + b;
                values[base + i + stride] = a - b;
            }
        }
    }
    const float scale = 1.0f / std::sqrt(static_cast<float>(dim));
    for (int i = 0; i < dim; ++i) {
        values[i] *= scale;
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Args args = parse_args(argc, argv);
        std::mt19937 rng(42);
        std::normal_distribution<float> normal(0.0f, 1.0f);
        std::vector<float> input(static_cast<std::size_t>(args.rows) * args.dim);
        for (std::size_t i = 0; i < input.size(); ++i) {
            input[i] = normal(rng);
        }

        std::vector<float> output = input;
        // 预热一次，避免首次运行时的缓存和频率变化影响结果。
        for (int row = 0; row < args.rows; ++row) {
            cpu_fht(output.data() + static_cast<std::size_t>(row) * args.dim, args.dim);
        }

        const auto start = std::chrono::steady_clock::now();
        for (int repeat = 0; repeat < args.repeats; ++repeat) {
            output = input;
            for (int row = 0; row < args.rows; ++row) {
                cpu_fht(output.data() + static_cast<std::size_t>(row) * args.dim, args.dim);
            }
        }
        const auto stop = std::chrono::steady_clock::now();
        const double total_ms = std::chrono::duration<double, std::milli>(stop - start).count();
        const double avg_ms = total_ms / args.repeats;
        const double bytes = static_cast<double>(args.rows) * args.dim * sizeof(float) * 2;
        const double gb_s = bytes / (avg_ms * 1.0e6);

        std::cout << std::fixed << std::setprecision(6)
                  << "cpu_fht_ms=" << avg_ms
                  << " effective_gb_s=" << gb_s
                  << " rows=" << args.rows
                  << " dim=" << args.dim
                  << " repeats=" << args.repeats << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "错误：" << error.what() << '\n';
        return 1;
    }
}
