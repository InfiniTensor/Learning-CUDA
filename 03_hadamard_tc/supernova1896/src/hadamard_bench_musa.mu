#include "hadamard_musa.muh"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Args {
    int batch = 8;
    int seq = 1024;
    int heads = 16;
    int dim = 128;
    int repeats = 100;
    hadamard_musa::DType dtype = hadamard_musa::DType::FP16;
    std::string log_path = "perf_musa.csv";
};

Args parse_args(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        auto need_value = [&](const char* name) -> const char* {
            if (i + 1 >= argc) {
                throw std::invalid_argument(std::string("缺少参数: ") + name);
            }
            return argv[++i];
        };
        if (key == "--batch") {
            args.batch = std::stoi(need_value("--batch"));
        } else if (key == "--seq") {
            args.seq = std::stoi(need_value("--seq"));
        } else if (key == "--heads") {
            args.heads = std::stoi(need_value("--heads"));
        } else if (key == "--dim") {
            args.dim = std::stoi(need_value("--dim"));
        } else if (key == "--repeats") {
            args.repeats = std::stoi(need_value("--repeats"));
        } else if (key == "--dtype") {
            const std::string value = need_value("--dtype");
            if (value == "fp16") {
                args.dtype = hadamard_musa::DType::FP16;
            } else if (value == "bf16") {
                args.dtype = hadamard_musa::DType::BF16;
            } else {
                throw std::invalid_argument("dtype 必须是 fp16 或 bf16");
            }
        } else if (key == "--log") {
            args.log_path = need_value("--log");
        } else if (key == "--help") {
            std::cout
                << "用法: hadamard_bench_musa [--batch N] [--seq N] [--heads N]\n"
                << "       [--dim 64|128|256] [--dtype fp16|bf16]\n"
                << "       [--repeats N] [--log perf_musa.csv]\n";
            std::exit(0);
        } else {
            throw std::invalid_argument("未知参数: " + key);
        }
    }
    return args;
}

std::vector<std::uint16_t> make_input(
    int rows,
    int dim,
    hadamard_musa::DType dtype) {
    std::mt19937 rng(42);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::vector<std::uint16_t> data(static_cast<std::size_t>(rows) * dim);
    for (std::size_t i = 0; i < data.size(); ++i) {
        float value = normal(rng);
        if (i % 251 == 0) {
            value *= 12.0f;
        }
        data[i] = dtype == hadamard_musa::DType::FP16
            ? hadamard_musa::float_to_fp16_bits(value)
            : hadamard_musa::float_to_bf16_bits(value);
    }
    return data;
}

double bytes_per_second(std::size_t bytes, float milliseconds) {
    return static_cast<double>(bytes) /
        (milliseconds * 1.0e-3) / 1.0e9;
}

void append_log(
    const std::string& path,
    const Args& args,
    const std::string& kernel,
    float milliseconds,
    std::size_t bytes) {
    const bool exists = static_cast<bool>(std::ifstream(path));
    std::ofstream output(path, std::ios::app);
    if (!exists) {
        output << "kernel,dtype,batch,seq,heads,head_dim,rows,time_ms,effective_gb_s\n";
    }
    const int rows = args.batch * args.seq * args.heads;
    output << kernel << ','
           << hadamard_musa::dtype_name(args.dtype) << ','
           << args.batch << ',' << args.seq << ',' << args.heads << ','
           << args.dim << ',' << rows << ',' << milliseconds << ','
           << bytes_per_second(bytes, milliseconds) << '\n';
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Args args = parse_args(argc, argv);
        const int rows = args.batch * args.seq * args.heads;
        const std::size_t tensor_bytes =
            hadamard_musa::tensor_bytes(rows, args.dim);
        const std::vector<std::uint16_t> input =
            make_input(rows, args.dim, args.dtype);

        const int device = 0;
        HADAMARD_MUSA_CHECK(musaSetDevice(device));
        musaDeviceProp prop{};
        HADAMARD_MUSA_CHECK(musaGetDeviceProperties(&prop, device));
        std::cout << "GPU: " << prop.name
                  << " compute_capability=" << prop.major << '.' << prop.minor
                  << "\n";
        std::cout << "shape=[" << args.batch << ',' << args.seq << ','
                  << args.heads << ',' << args.dim << "] dtype="
                  << hadamard_musa::dtype_name(args.dtype)
                  << " rows=" << rows << " repeats=" << args.repeats << "\n";

        void* d_input = nullptr;
        void* d_output = nullptr;
        HADAMARD_MUSA_CHECK(musaMalloc(&d_input, tensor_bytes));
        HADAMARD_MUSA_CHECK(musaMalloc(&d_output, tensor_bytes));
        HADAMARD_MUSA_CHECK(musaMemcpy(
            d_input,
            input.data(),
            tensor_bytes,
            musaMemcpyHostToDevice));

        const auto shared = hadamard_musa::launch_hadamard_shared(
            d_input, d_output, rows, args.dim, args.dtype, args.repeats);
        std::cout << "shared_fht_ms=" << shared.milliseconds
                  << " effective_gb_s="
                  << bytes_per_second(2 * tensor_bytes, shared.milliseconds)
                  << "\n";
        append_log(
            args.log_path,
            args,
            "shared_fht",
            shared.milliseconds,
            2 * tensor_bytes);

        std::uint8_t* d_int4 = nullptr;
        float* d_scales = nullptr;
        const std::size_t packed_bytes =
            hadamard_musa::int4_packed_bytes(rows, args.dim);
        HADAMARD_MUSA_CHECK(musaMalloc(&d_int4, packed_bytes));
        HADAMARD_MUSA_CHECK(musaMalloc(&d_scales, rows * sizeof(float)));
        const auto fused_int4 = hadamard_musa::launch_fused_hadamard_int4(
            d_input,
            d_int4,
            d_scales,
            rows,
            args.dim,
            args.dtype,
            args.repeats);
        std::cout << "fused_int4_ms=" << fused_int4.milliseconds
                  << " effective_gb_s="
                  << bytes_per_second(
                      tensor_bytes + packed_bytes + rows * sizeof(float),
                      fused_int4.milliseconds)
                  << "\n";
        append_log(
            args.log_path,
            args,
            "fused_int4",
            fused_int4.milliseconds,
            tensor_bytes + packed_bytes + rows * sizeof(float));

        std::uint8_t* d_e4m3 = nullptr;
        HADAMARD_MUSA_CHECK(musaMalloc(
            &d_e4m3,
            static_cast<std::size_t>(rows) * args.dim));
        const auto fused_e4m3 = hadamard_musa::launch_fused_hadamard_e4m3(
            d_input,
            d_e4m3,
            rows,
            args.dim,
            args.dtype,
            args.repeats);
        std::cout << "fused_e4m3_ms=" << fused_e4m3.milliseconds
                  << " effective_gb_s="
                  << bytes_per_second(
                      tensor_bytes + static_cast<std::size_t>(rows) * args.dim,
                      fused_e4m3.milliseconds)
                  << "\n";
        append_log(
            args.log_path,
            args,
            "fused_e4m3",
            fused_e4m3.milliseconds,
            tensor_bytes + static_cast<std::size_t>(rows) * args.dim);

        HADAMARD_MUSA_CHECK(musaFree(d_e4m3));
        HADAMARD_MUSA_CHECK(musaFree(d_int4));
        HADAMARD_MUSA_CHECK(musaFree(d_scales));
        HADAMARD_MUSA_CHECK(musaFree(d_input));
        HADAMARD_MUSA_CHECK(musaFree(d_output));
        std::cout << "日志已追加到 " << args.log_path << "\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "错误: " << error.what() << "\n";
        return 1;
    }
}
