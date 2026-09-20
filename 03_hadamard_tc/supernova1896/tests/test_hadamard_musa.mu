#include "hadamard_musa.muh"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::vector<float> make_input(int rows, int dim) {
    std::mt19937 rng(20260920);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::vector<float> data(static_cast<std::size_t>(rows) * dim);
    for (std::size_t i = 0; i < data.size(); ++i) {
        data[i] = normal(rng);
        if (i % 97 == 0) {
            data[i] *= 16.0f;
        }
    }
    return data;
}

std::vector<std::uint16_t> encode_input(
    const std::vector<float>& input,
    hadamard_musa::DType dtype) {
    std::vector<std::uint16_t> output(input.size());
    for (std::size_t i = 0; i < input.size(); ++i) {
        output[i] = dtype == hadamard_musa::DType::FP16
            ? hadamard_musa::float_to_fp16_bits(input[i])
            : hadamard_musa::float_to_bf16_bits(input[i]);
    }
    return output;
}

float decode_value(
    std::uint16_t bits,
    hadamard_musa::DType dtype) {
    return dtype == hadamard_musa::DType::FP16
        ? hadamard_musa::fp16_bits_to_float(bits)
        : hadamard_musa::bf16_bits_to_float(bits);
}

std::vector<float> reference(
    const std::vector<std::uint16_t>& encoded,
    int rows,
    int dim,
    hadamard_musa::DType dtype) {
    std::vector<float> result(encoded.size());
    for (int row = 0; row < rows; ++row) {
        std::vector<float> values(dim);
        for (int i = 0; i < dim; ++i) {
            values[i] = decode_value(encoded[row * dim + i], dtype);
        }
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
            result[row * dim + i] = values[i] * scale;
        }
    }
    return result;
}

void assert_close(
    const std::vector<float>& expected,
    const std::vector<std::uint16_t>& actual,
    int dim,
    hadamard_musa::DType dtype,
    float tolerance,
    const std::string& label) {
    float max_abs = 0.0f;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        max_abs = std::max(
            max_abs,
            std::fabs(expected[i] - decode_value(actual[i], dtype)));
    }
    std::cout << label << " max_abs=" << max_abs
              << " tolerance=" << tolerance << "\n";
    if (max_abs > tolerance) {
        throw std::runtime_error(label + " 超过误差阈值");
    }
    (void)dim;
}

void run_case(int rows, int dim, hadamard_musa::DType dtype) {
    const float tolerance = dtype == hadamard_musa::DType::FP16
        ? 1e-2f
        : 5e-2f;
    const auto floating_input = make_input(rows, dim);
    const auto encoded_input = encode_input(floating_input, dtype);
    const auto expected = reference(encoded_input, rows, dim, dtype);
    const std::size_t tensor_bytes = hadamard_musa::tensor_bytes(rows, dim);

    void* d_input = nullptr;
    void* d_output = nullptr;
    HADAMARD_MUSA_CHECK(musaMalloc(&d_input, tensor_bytes));
    HADAMARD_MUSA_CHECK(musaMalloc(&d_output, tensor_bytes));
    HADAMARD_MUSA_CHECK(musaMemcpy(
        d_input,
        encoded_input.data(),
        tensor_bytes,
        musaMemcpyHostToDevice));

    hadamard_musa::launch_hadamard_shared(
        d_input, d_output, rows, dim, dtype, 2);
    std::vector<std::uint16_t> output(encoded_input.size());
    HADAMARD_MUSA_CHECK(musaMemcpy(
        output.data(),
        d_output,
        tensor_bytes,
        musaMemcpyDeviceToHost));
    assert_close(
        expected,
        output,
        dim,
        dtype,
        tolerance,
        std::string("shared ") + hadamard_musa::dtype_name(dtype) +
            " dim=" + std::to_string(dim));

    const std::size_t packed_bytes =
        hadamard_musa::int4_packed_bytes(rows, dim);
    std::uint8_t* d_sep_int4 = nullptr;
    std::uint8_t* d_fused_int4 = nullptr;
    float* d_sep_scales = nullptr;
    float* d_fused_scales = nullptr;
    HADAMARD_MUSA_CHECK(musaMalloc(&d_sep_int4, packed_bytes));
    HADAMARD_MUSA_CHECK(musaMalloc(&d_fused_int4, packed_bytes));
    HADAMARD_MUSA_CHECK(musaMalloc(
        &d_sep_scales, rows * sizeof(float)));
    HADAMARD_MUSA_CHECK(musaMalloc(
        &d_fused_scales, rows * sizeof(float)));
    hadamard_musa::launch_quantize_int4(
        d_output,
        d_sep_int4,
        d_sep_scales,
        rows,
        dim,
        dtype);
    hadamard_musa::launch_fused_hadamard_int4(
        d_input,
        d_fused_int4,
        d_fused_scales,
        rows,
        dim,
        dtype);

    std::vector<std::uint8_t> sep_int4(packed_bytes);
    std::vector<std::uint8_t> fused_int4(packed_bytes);
    std::vector<float> sep_scales(rows);
    std::vector<float> fused_scales(rows);
    HADAMARD_MUSA_CHECK(musaMemcpy(
        sep_int4.data(), d_sep_int4, packed_bytes, musaMemcpyDeviceToHost));
    HADAMARD_MUSA_CHECK(musaMemcpy(
        fused_int4.data(), d_fused_int4, packed_bytes, musaMemcpyDeviceToHost));
    HADAMARD_MUSA_CHECK(musaMemcpy(
        sep_scales.data(),
        d_sep_scales,
        rows * sizeof(float),
        musaMemcpyDeviceToHost));
    HADAMARD_MUSA_CHECK(musaMemcpy(
        fused_scales.data(),
        d_fused_scales,
        rows * sizeof(float),
        musaMemcpyDeviceToHost));
    if (sep_int4 != fused_int4) {
        throw std::runtime_error("MUSA fused INT4 字节结果不一致");
    }
    for (int i = 0; i < rows; ++i) {
        if (std::fabs(sep_scales[i] - fused_scales[i]) > 1e-6f) {
            throw std::runtime_error("MUSA fused INT4 scale 不一致");
        }
    }

    std::uint8_t* d_sep_e4m3 = nullptr;
    std::uint8_t* d_fused_e4m3 = nullptr;
    HADAMARD_MUSA_CHECK(musaMalloc(
        &d_sep_e4m3, encoded_input.size()));
    HADAMARD_MUSA_CHECK(musaMalloc(
        &d_fused_e4m3, encoded_input.size()));
    hadamard_musa::launch_quantize_e4m3(
        d_output, d_sep_e4m3, rows, dim, dtype);
    hadamard_musa::launch_fused_hadamard_e4m3(
        d_input, d_fused_e4m3, rows, dim, dtype);
    std::vector<std::uint8_t> sep_e4m3(encoded_input.size());
    std::vector<std::uint8_t> fused_e4m3(encoded_input.size());
    HADAMARD_MUSA_CHECK(musaMemcpy(
        sep_e4m3.data(),
        d_sep_e4m3,
        encoded_input.size(),
        musaMemcpyDeviceToHost));
    HADAMARD_MUSA_CHECK(musaMemcpy(
        fused_e4m3.data(),
        d_fused_e4m3,
        encoded_input.size(),
        musaMemcpyDeviceToHost));
    if (sep_e4m3 != fused_e4m3) {
        throw std::runtime_error("MUSA fused E4M3 字节结果不一致");
    }

    HADAMARD_MUSA_CHECK(musaFree(d_sep_e4m3));
    HADAMARD_MUSA_CHECK(musaFree(d_fused_e4m3));
    HADAMARD_MUSA_CHECK(musaFree(d_sep_int4));
    HADAMARD_MUSA_CHECK(musaFree(d_fused_int4));
    HADAMARD_MUSA_CHECK(musaFree(d_sep_scales));
    HADAMARD_MUSA_CHECK(musaFree(d_fused_scales));
    HADAMARD_MUSA_CHECK(musaFree(d_input));
    HADAMARD_MUSA_CHECK(musaFree(d_output));
}

}  // namespace

int main() {
    try {
        HADAMARD_MUSA_CHECK(musaSetDevice(0));
        musaDeviceProp prop{};
        HADAMARD_MUSA_CHECK(musaGetDeviceProperties(&prop, 0));
        std::cout << "GPU: " << prop.name
                  << " compute_capability=" << prop.major << '.'
                  << prop.minor << "\n";
        for (int dim : {64, 128, 256}) {
            run_case(37, dim, hadamard_musa::DType::FP16);
            run_case(37, dim, hadamard_musa::DType::BF16);
        }
        std::cout << "所有 MUSA Hadamard 正确性和融合量化测试通过。\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "测试失败: " << error.what() << "\n";
        return 1;
    }
}
