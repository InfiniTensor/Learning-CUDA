#include "hadamard_cuda.cuh"

#include <cuda_fp16.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::uint16_t float_to_fp16_bits(float value) {
    const half h = __float2half_rn(value);
    std::uint16_t bits;
    std::memcpy(&bits, &h, sizeof(bits));
    return bits;
}

float fp16_bits_to_float(std::uint16_t bits) {
    half h;
    std::memcpy(&h, &bits, sizeof(bits));
    return __half2float(h);
}

std::vector<float> generate_input(int rows, int dim) {
    std::mt19937 rng(20260918);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::vector<float> data(static_cast<std::size_t>(rows) * dim);
    for (std::size_t i = 0; i < data.size(); ++i) {
        float v = normal(rng);
        if (i % 97 == 0) {
            // 人为加入少量异常值，用来覆盖 Hadamard 旋转常见的量化前场景。
            v *= 16.0f;
        }
        data[i] = v;
    }
    return data;
}

std::vector<std::uint16_t> encode_input(const std::vector<float>& input, hadamard::DType dtype) {
    std::vector<std::uint16_t> encoded(input.size());
    for (std::size_t i = 0; i < input.size(); ++i) {
        encoded[i] = dtype == hadamard::DType::FP16
            ? float_to_fp16_bits(input[i])
            : hadamard::float_to_bf16_bits(input[i]);
    }
    return encoded;
}

float decode_value(std::uint16_t bits, hadamard::DType dtype) {
    return dtype == hadamard::DType::FP16
        ? fp16_bits_to_float(bits)
        : hadamard::bf16_bits_to_float(bits);
}

std::vector<float> cpu_reference_from_encoded(
    const std::vector<std::uint16_t>& encoded,
    int rows,
    int dim,
    hadamard::DType dtype) {
    std::vector<float> ref(encoded.size());
    for (int row = 0; row < rows; ++row) {
        std::vector<float> tmp(dim);
        for (int i = 0; i < dim; ++i) {
            tmp[i] = decode_value(encoded[row * dim + i], dtype);
        }
        // CPU reference 使用同一套蝶形 FHT 公式，作为 GPU shared-memory
        // kernel 的数值正确性基准。
        for (int stride = 1; stride < dim; stride <<= 1) {
            for (int base = 0; base < dim; base += 2 * stride) {
                for (int i = 0; i < stride; ++i) {
                    const float a = tmp[base + i];
                    const float b = tmp[base + i + stride];
                    tmp[base + i] = a + b;
                    tmp[base + i + stride] = a - b;
                }
            }
        }
        const float norm = 1.0f / std::sqrt(static_cast<float>(dim));
        for (int i = 0; i < dim; ++i) {
            ref[row * dim + i] = tmp[i] * norm;
        }
    }
    return ref;
}

void assert_close(
    const std::vector<float>& ref,
    const std::vector<std::uint16_t>& got,
    int rows,
    int dim,
    hadamard::DType dtype,
    float tolerance,
    const std::string& label) {
    float max_abs = 0.0f;
    for (std::size_t i = 0; i < got.size(); ++i) {
        max_abs = std::max(max_abs, std::fabs(ref[i] - decode_value(got[i], dtype)));
    }
    std::cout << label << " max_abs=" << max_abs << " tolerance=" << tolerance << "\n";
    if (max_abs > tolerance) {
        throw std::runtime_error(label + " exceeded tolerance");
    }
}

void run_case(int rows, int dim, hadamard::DType dtype) {
    const float tolerance = dtype == hadamard::DType::FP16 ? 1e-2f : 5e-2f;
    const std::vector<float> input = generate_input(rows, dim);
    const std::vector<std::uint16_t> encoded = encode_input(input, dtype);
    const std::vector<float> ref = cpu_reference_from_encoded(encoded, rows, dim, dtype);

    void* d_input = nullptr;
    void* d_output = nullptr;
    HADAMARD_CUDA_CHECK(cudaMalloc(&d_input, hadamard::tensor_bytes(rows, dim, dtype)));
    HADAMARD_CUDA_CHECK(cudaMalloc(&d_output, hadamard::tensor_bytes(rows, dim, dtype)));
    HADAMARD_CUDA_CHECK(cudaMemcpy(
        d_input,
        encoded.data(),
        hadamard::tensor_bytes(rows, dim, dtype),
        cudaMemcpyHostToDevice));

    hadamard::launch_hadamard_shared(d_input, d_output, rows, dim, dtype, 3);
    std::vector<std::uint16_t> output(encoded.size());
    HADAMARD_CUDA_CHECK(cudaMemcpy(
        output.data(),
        d_output,
        hadamard::tensor_bytes(rows, dim, dtype),
        cudaMemcpyDeviceToHost));
    assert_close(ref, output, rows, dim, dtype, tolerance, std::string("shared ") + hadamard::dtype_name(dtype) + " dim=" + std::to_string(dim));

    // 题目要求 fused 量化需与“先变换后量化”一致。这里逐字节比较
    // INT4 packed 输出，并逐元素比较 per-row scale。
    std::uint8_t* d_sep_int4 = nullptr;
    std::uint8_t* d_fused_int4 = nullptr;
    float* d_sep_scales = nullptr;
    float* d_fused_scales = nullptr;
    const std::size_t packed_bytes = hadamard::int4_packed_bytes(rows, dim);
    HADAMARD_CUDA_CHECK(cudaMalloc(&d_sep_int4, packed_bytes));
    HADAMARD_CUDA_CHECK(cudaMalloc(&d_fused_int4, packed_bytes));
    HADAMARD_CUDA_CHECK(cudaMalloc(&d_sep_scales, rows * sizeof(float)));
    HADAMARD_CUDA_CHECK(cudaMalloc(&d_fused_scales, rows * sizeof(float)));
    hadamard::launch_quantize_int4(d_output, d_sep_int4, d_sep_scales, rows, dim, dtype, 1);
    hadamard::launch_fused_hadamard_int4(d_input, d_fused_int4, d_fused_scales, rows, dim, dtype, 1);
    std::vector<std::uint8_t> sep_int4(packed_bytes);
    std::vector<std::uint8_t> fused_int4(packed_bytes);
    std::vector<float> sep_scales(rows);
    std::vector<float> fused_scales(rows);
    HADAMARD_CUDA_CHECK(cudaMemcpy(sep_int4.data(), d_sep_int4, packed_bytes, cudaMemcpyDeviceToHost));
    HADAMARD_CUDA_CHECK(cudaMemcpy(fused_int4.data(), d_fused_int4, packed_bytes, cudaMemcpyDeviceToHost));
    HADAMARD_CUDA_CHECK(cudaMemcpy(sep_scales.data(), d_sep_scales, rows * sizeof(float), cudaMemcpyDeviceToHost));
    HADAMARD_CUDA_CHECK(cudaMemcpy(fused_scales.data(), d_fused_scales, rows * sizeof(float), cudaMemcpyDeviceToHost));
    if (sep_int4 != fused_int4) {
        throw std::runtime_error("fused INT4 bytes differ from transform-then-quantize");
    }
    for (int i = 0; i < rows; ++i) {
        if (std::fabs(sep_scales[i] - fused_scales[i]) > 1e-6f) {
            throw std::runtime_error("fused INT4 scales differ from transform-then-quantize");
        }
    }

    std::uint8_t* d_sep_e4m3 = nullptr;
    std::uint8_t* d_fused_e4m3 = nullptr;
    HADAMARD_CUDA_CHECK(cudaMalloc(&d_sep_e4m3, encoded.size()));
    HADAMARD_CUDA_CHECK(cudaMalloc(&d_fused_e4m3, encoded.size()));
    hadamard::launch_quantize_e4m3(d_output, d_sep_e4m3, rows, dim, dtype, 1);
    hadamard::launch_fused_hadamard_e4m3(d_input, d_fused_e4m3, rows, dim, dtype, 1);
    std::vector<std::uint8_t> sep_e4m3(encoded.size());
    std::vector<std::uint8_t> fused_e4m3(encoded.size());
    HADAMARD_CUDA_CHECK(cudaMemcpy(sep_e4m3.data(), d_sep_e4m3, encoded.size(), cudaMemcpyDeviceToHost));
    HADAMARD_CUDA_CHECK(cudaMemcpy(fused_e4m3.data(), d_fused_e4m3, encoded.size(), cudaMemcpyDeviceToHost));
    if (sep_e4m3 != fused_e4m3) {
        throw std::runtime_error("fused E4M3 bytes differ from transform-then-quantize");
    }

#if !defined(HADAMARD_MACA)
    // NVIDIA 路径额外验证 WMMA/Tensor Core 对照实现，误差阈值放宽到 2e-2。
    if (dtype == hadamard::DType::FP16) {
        void* d_hmat = nullptr;
        void* d_wmma = nullptr;
        const auto hmat = hadamard::make_hadamard_matrix_fp16(dim);
        HADAMARD_CUDA_CHECK(cudaMalloc(&d_hmat, hmat.size() * sizeof(std::uint16_t)));
        HADAMARD_CUDA_CHECK(cudaMalloc(&d_wmma, hadamard::tensor_bytes(rows, dim, dtype)));
        HADAMARD_CUDA_CHECK(cudaMemcpy(d_hmat, hmat.data(), hmat.size() * sizeof(std::uint16_t), cudaMemcpyHostToDevice));
        hadamard::launch_hadamard_wmma_fp16(d_input, d_wmma, d_hmat, rows, dim, 1);
        std::vector<std::uint16_t> wmma_output(encoded.size());
        HADAMARD_CUDA_CHECK(cudaMemcpy(
            wmma_output.data(),
            d_wmma,
            hadamard::tensor_bytes(rows, dim, dtype),
            cudaMemcpyDeviceToHost));
        assert_close(ref, wmma_output, rows, dim, dtype, 2e-2f, "wmma fp16 dim=" + std::to_string(dim));
        HADAMARD_CUDA_CHECK(cudaFree(d_hmat));
        HADAMARD_CUDA_CHECK(cudaFree(d_wmma));
    }
#endif

    HADAMARD_CUDA_CHECK(cudaFree(d_sep_e4m3));
    HADAMARD_CUDA_CHECK(cudaFree(d_fused_e4m3));
    HADAMARD_CUDA_CHECK(cudaFree(d_sep_int4));
    HADAMARD_CUDA_CHECK(cudaFree(d_fused_int4));
    HADAMARD_CUDA_CHECK(cudaFree(d_sep_scales));
    HADAMARD_CUDA_CHECK(cudaFree(d_fused_scales));
    HADAMARD_CUDA_CHECK(cudaFree(d_input));
    HADAMARD_CUDA_CHECK(cudaFree(d_output));
}

}  // namespace

int main() {
    try {
        int device = 0;
        HADAMARD_CUDA_CHECK(cudaSetDevice(device));
        cudaDeviceProp prop{};
        HADAMARD_CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
        std::cout << "GPU: " << prop.name << " sm_" << prop.major << prop.minor << "\n";

        for (int dim : {64, 128, 256}) {
            run_case(37, dim, hadamard::DType::FP16);
            run_case(37, dim, hadamard::DType::BF16);
        }
        std::cout << "All Hadamard correctness and fused-quantization tests passed.\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Test failed: " << e.what() << "\n";
        return 1;
    }
}
