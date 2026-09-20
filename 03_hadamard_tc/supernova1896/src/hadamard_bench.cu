#include "hadamard_cuda.cuh"

#include <cuda_fp16.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
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
    hadamard::DType dtype = hadamard::DType::FP16;
    std::string log_path = "perf.csv";
};

std::uint16_t float_to_fp16_bits(float value) {
    const half h = __float2half_rn(value);
    std::uint16_t bits;
    std::memcpy(&bits, &h, sizeof(bits));
    return bits;
}

Args parse_args(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        auto need_value = [&](const char* name) -> const char* {
            if (i + 1 >= argc) {
                throw std::invalid_argument(std::string("missing value for ") + name);
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
                args.dtype = hadamard::DType::FP16;
            } else if (value == "bf16") {
                args.dtype = hadamard::DType::BF16;
            } else {
                throw std::invalid_argument("dtype must be fp16 or bf16");
            }
        } else if (key == "--log") {
            args.log_path = need_value("--log");
        } else if (key == "--help") {
            std::cout
                << "Usage: hadamard_bench [--batch N] [--seq N] [--heads N]\n"
                << "                      [--dim 64|128|256] [--dtype fp16|bf16]\n"
                << "                      [--repeats N] [--log perf.csv]\n";
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown argument: " + key);
        }
    }
    return args;
}

std::vector<std::uint16_t> make_input(int rows, int dim, hadamard::DType dtype) {
    std::mt19937 rng(42);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::vector<std::uint16_t> data(static_cast<std::size_t>(rows) * dim);
    for (std::size_t i = 0; i < data.size(); ++i) {
        float v = normal(rng);
        if (i % 251 == 0) {
            // 注入少量异常值，让 benchmark 输入更接近量化前激活的长尾分布。
            v *= 12.0f;
        }
        data[i] = dtype == hadamard::DType::FP16 ? float_to_fp16_bits(v) : hadamard::float_to_bf16_bits(v);
    }
    return data;
}

double bytes_per_second(std::size_t bytes, float ms) {
    return (static_cast<double>(bytes) / (ms * 1.0e-3)) / 1.0e9;
}

void append_log(
    const std::string& path,
    const Args& args,
    const std::string& kernel,
    float ms,
    std::size_t bytes) {
    const bool exists = static_cast<bool>(std::ifstream(path));
    std::ofstream out(path, std::ios::app);
    if (!exists) {
        // CSV 字段保持英文，便于后续用脚本或表格软件处理。
        out << "kernel,dtype,batch,seq,heads,head_dim,rows,time_ms,effective_gb_s\n";
    }
    const int rows = args.batch * args.seq * args.heads;
    out << kernel << ','
        << hadamard::dtype_name(args.dtype) << ','
        << args.batch << ','
        << args.seq << ','
        << args.heads << ','
        << args.dim << ','
        << rows << ','
        << ms << ','
        << bytes_per_second(bytes, ms) << '\n';
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Args args = parse_args(argc, argv);
        const int rows = args.batch * args.seq * args.heads;
        const std::size_t tensor_bytes = hadamard::tensor_bytes(rows, args.dim, args.dtype);
        const std::vector<std::uint16_t> input = make_input(rows, args.dim, args.dtype);

        int device = 0;
        HADAMARD_CUDA_CHECK(cudaSetDevice(device));
        cudaDeviceProp prop{};
        HADAMARD_CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
        std::cout << "GPU: " << prop.name << " sm_" << prop.major << prop.minor << "\n";
        std::cout << "shape=[" << args.batch << ',' << args.seq << ',' << args.heads << ',' << args.dim
                  << "] dtype=" << hadamard::dtype_name(args.dtype)
                  << " rows=" << rows << " repeats=" << args.repeats << "\n";

        void* d_input = nullptr;
        void* d_output = nullptr;
        HADAMARD_CUDA_CHECK(cudaMalloc(&d_input, tensor_bytes));
        HADAMARD_CUDA_CHECK(cudaMalloc(&d_output, tensor_bytes));
        HADAMARD_CUDA_CHECK(cudaMemcpy(d_input, input.data(), tensor_bytes, cudaMemcpyHostToDevice));

        // 主路径：shared-memory FHT，统计读输入和写输出的近似有效带宽。
        const auto shared = hadamard::launch_hadamard_shared(d_input, d_output, rows, args.dim, args.dtype, args.repeats);
        std::cout << "shared_fht_ms=" << shared.milliseconds
                  << " effective_gb_s=" << bytes_per_second(2 * tensor_bytes, shared.milliseconds) << "\n";
        append_log(args.log_path, args, "shared_fht", shared.milliseconds, 2 * tensor_bytes);

        std::uint8_t* d_int4 = nullptr;
        float* d_scales = nullptr;
        const std::size_t packed_bytes = hadamard::int4_packed_bytes(rows, args.dim);
        HADAMARD_CUDA_CHECK(cudaMalloc(&d_int4, packed_bytes));
        HADAMARD_CUDA_CHECK(cudaMalloc(&d_scales, rows * sizeof(float)));
        // 融合 INT4 路径：一次 kernel 完成旋转、scale 统计和 packed store。
        const auto fused_int4 = hadamard::launch_fused_hadamard_int4(
            d_input,
            d_int4,
            d_scales,
            rows,
            args.dim,
            args.dtype,
            args.repeats);
        std::cout << "fused_int4_ms=" << fused_int4.milliseconds
                  << " effective_gb_s=" << bytes_per_second(tensor_bytes + packed_bytes + rows * sizeof(float), fused_int4.milliseconds) << "\n";
        append_log(args.log_path, args, "fused_int4", fused_int4.milliseconds, tensor_bytes + packed_bytes + rows * sizeof(float));

        std::uint8_t* d_e4m3 = nullptr;
        HADAMARD_CUDA_CHECK(cudaMalloc(&d_e4m3, static_cast<std::size_t>(rows) * args.dim));
        // 融合 E4M3 路径：Hadamard 输出不落地，直接写入 FP8 编码字节。
        const auto fused_e4m3 = hadamard::launch_fused_hadamard_e4m3(
            d_input,
            d_e4m3,
            rows,
            args.dim,
            args.dtype,
            args.repeats);
        std::cout << "fused_e4m3_ms=" << fused_e4m3.milliseconds
                  << " effective_gb_s=" << bytes_per_second(tensor_bytes + static_cast<std::size_t>(rows) * args.dim, fused_e4m3.milliseconds) << "\n";
        append_log(args.log_path, args, "fused_e4m3", fused_e4m3.milliseconds, tensor_bytes + static_cast<std::size_t>(rows) * args.dim);

#if !defined(HADAMARD_MACA)
        if (args.dtype == hadamard::DType::FP16) {
            // FP16 额外跑 WMMA/Tensor Core 对照路径，用于和 FHT 主实现比较。
            const auto hmat = hadamard::make_hadamard_matrix_fp16(args.dim);
            void* d_hmat = nullptr;
            void* d_wmma = nullptr;
            HADAMARD_CUDA_CHECK(cudaMalloc(&d_hmat, hmat.size() * sizeof(std::uint16_t)));
            HADAMARD_CUDA_CHECK(cudaMalloc(&d_wmma, tensor_bytes));
            HADAMARD_CUDA_CHECK(cudaMemcpy(d_hmat, hmat.data(), hmat.size() * sizeof(std::uint16_t), cudaMemcpyHostToDevice));
            const auto wmma = hadamard::launch_hadamard_wmma_fp16(
                d_input,
                d_wmma,
                d_hmat,
                rows,
                args.dim,
                args.repeats);
            std::cout << "wmma_fp16_ms=" << wmma.milliseconds
                      << " effective_gb_s=" << bytes_per_second(2 * tensor_bytes + hmat.size() * sizeof(std::uint16_t), wmma.milliseconds) << "\n";
            append_log(args.log_path, args, "wmma_fp16", wmma.milliseconds, 2 * tensor_bytes + hmat.size() * sizeof(std::uint16_t));
            HADAMARD_CUDA_CHECK(cudaFree(d_hmat));
            HADAMARD_CUDA_CHECK(cudaFree(d_wmma));
        }
#else
        // C500 版本不调用 NVIDIA 专用 WMMA 对照路径。
        (void)args;
#endif

        HADAMARD_CUDA_CHECK(cudaFree(d_e4m3));
        HADAMARD_CUDA_CHECK(cudaFree(d_int4));
        HADAMARD_CUDA_CHECK(cudaFree(d_scales));
        HADAMARD_CUDA_CHECK(cudaFree(d_input));
        HADAMARD_CUDA_CHECK(cudaFree(d_output));
        std::cout << "log appended to " << args.log_path << "\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
