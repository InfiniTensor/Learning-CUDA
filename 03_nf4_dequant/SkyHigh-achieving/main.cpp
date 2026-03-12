#include "dequant_kernel.h"
#include <cuda_runtime.h>
#include <iostream>
#include <string>
#include <vector>

namespace {

ComputeType parse_compute_type(const std::string& s) {
    if (s == "bf16") {
        return ComputeType::BF16;
    }
    return ComputeType::FP16;
}

}  

int main(int argc, char** argv) {
    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    if (error != cudaSuccess) {
        std::cerr << "cudaGetDeviceCount failed: " << cudaGetErrorString(error) << std::endl;
        std::cerr << "Ensure this binary is executed inside a GPU allocation (srun/sbatch)." << std::endl;
        return -1;
    }
    if (deviceCount == 0) {
        std::cerr << "No CUDA-capable devices found in current context." << std::endl;
        std::cerr << "Use: srun --partition=nvidia --gres=gpu:nvidia:1 ... ./nf4_dequant" << std::endl;
        return -1;
    }

    cudaDeviceProp prop;
    cudaError_t propErr = cudaGetDeviceProperties(&prop, 0);
    if (propErr == cudaSuccess) {
        std::cout << "Using device 0: " << prop.name << " (Compute Capability " << prop.major << "." << prop.minor << ")" << std::endl;
    } else {
        std::cerr << "cudaGetDeviceProperties failed: " << cudaGetErrorString(propErr) << std::endl;
        return -1;
    }

    if (argc < 4) {
        std::cerr << "Usage: nf4_dequant <input.bin> <fp16|bf16> <output.bin>" << std::endl;
        return 1;
    }

    NF4Binary input;
    if (!load_nf4_binary(argv[1], input)) {
        std::cerr << "Failed to load input binary: " << argv[1] << std::endl;
        return 2;
    }
    input.config.compute_type = parse_compute_type(argv[2]);

    std::vector<float> output;
    float mae = 0.0f;
    if (!run_dequant_cuda(input, output, mae)) {
        std::cerr << "CUDA run failed" << std::endl;
        return 3;
    }

    if (!save_float_output(argv[3], output)) {
        std::cerr << "Failed to save output: " << argv[3] << std::endl;
        return 4;
    }

    std::cout << "rows=" << input.config.rows
              << " cols=" << input.config.cols
              << " blocksize=" << input.config.blocksize
              << " mae=" << mae << std::endl;

    if (mae >= 1e-2f) {
        std::cerr << "MAE threshold failed" << std::endl;
        return 5;
    }

    return 0;
}
