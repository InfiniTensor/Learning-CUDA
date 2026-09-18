#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>

#include <cuda_runtime.h>

#include "formats/nvfp4_codec.cuh"

namespace {

/**
 * @brief 检查 CUDA runtime 调用，并在失败时打印诊断。
 *
 * @param status 待检查的 CUDA runtime 返回值。
 * @param operation 正在执行的 CUDA 操作说明。
 * @return `status == cudaSuccess` 时返回 true。
 */
[[nodiscard]] bool check_cuda(const cudaError_t status, const char* operation) {
    if (status == cudaSuccess) {
        return true;
    }

    std::cerr << operation << " 失败：" << cudaGetErrorString(status) << '\n';
    return false;
}

/**
 * @brief 比较两个有限 FP32 值是否在小误差范围内相等。
 *
 * @param left 左操作数。
 * @param right 右操作数。
 * @return 绝对误差不超过 `1e-6` 时返回 true。
 */
[[nodiscard]] bool nearly_equal(const float left, const float right) {
    return std::fabs(left - right) <= 1.0e-6F;
}

/**
 * @brief 在 device 路径验证 E2M1 编码、NVFP4 scale 和 nibble 打包。
 *
 * 单线程 kernel 只验证 codec 的 `__device__` 版本是否可调用且与 host 结果
 * 一致，并不涉及真实量化 kernel 的并行映射。
 *
 * @param output device 指针，指向至少 5 字节可写空间。
 */
__global__ void encodeNvfp4CodecKernel(std::uint8_t* __restrict__ output) {
    using quant_dequant::RoundingMode;
    using namespace quant_dequant::formats;

    const float global_scale = compute_nvfp4_global_scale(6.0F);
    const std::uint8_t local_scale =
        compute_nvfp4_local_scale_code(6.0F, global_scale);
    const std::uint8_t positive = encode_nvfp4_element(
        6.0F, local_scale, global_scale, RoundingMode::kNearest);
    const std::uint8_t negative = encode_nvfp4_element(
        -6.0F, local_scale, global_scale, RoundingMode::kNearest);

    output[0] = encode_e2m1_rne_sat(5.0F);
    output[1] = local_scale;
    output[2] = positive;
    output[3] = pack_e2m1_nibbles(positive, negative);
    output[4] = unpack_e2m1_high_nibble(output[3]);
}

/**
 * @brief 验证 E2M1 的完整正 codebook、符号位和正零规范化。
 *
 * @return 全部 eight magnitude code 的解码及零规则正确时返回 true。
 */
[[nodiscard]] bool test_e2m1_codebook_and_zero() {
    using namespace quant_dequant::formats;

    constexpr std::array<float, 8U> kExpectedMagnitude{
        0.0F, 0.5F, 1.0F, 1.5F, 2.0F, 3.0F, 4.0F, 6.0F,
    };

    for (std::uint8_t code = 0U; code <= kE2M1PositiveMaxCode; ++code) {
        if (decode_e2m1(code) != kExpectedMagnitude[code] ||
            decode_e2m1(static_cast<std::uint8_t>(code | 0x08U)) !=
                -kExpectedMagnitude[code]) {
            return false;
        }
    }

    const float negative_zero = decode_e2m1(0x08U);
    return negative_zero == 0.0F && fp32::has_negative_sign(negative_zero) &&
           encode_e2m1_rne_sat(-0.0F) == 0x00U;
}

/**
 * @brief 验证 E2M1 RNE 的不等间距中点、stochastic 端点和非有限回退。
 *
 * @return 所有编码边界符合项目规则时返回 true。
 */
[[nodiscard]] bool test_e2m1_rounding_and_saturation() {
    using namespace quant_dequant::formats;

    const bool rne_ties_to_even =
        encode_e2m1_rne_sat(0.25F) == 0x00U &&
        encode_e2m1_rne_sat(0.75F) == 0x02U &&
        encode_e2m1_rne_sat(1.25F) == 0x02U &&
        encode_e2m1_rne_sat(1.75F) == 0x04U &&
        encode_e2m1_rne_sat(2.5F) == 0x04U &&
        encode_e2m1_rne_sat(3.5F) == 0x06U &&
        encode_e2m1_rne_sat(5.0F) == 0x06U;

    const bool stochastic_endpoints =
        // codec 与 MXFP8 的约定相同：uniform_random < p_upper 时选上界。
        encode_e2m1_stochastic_sat(1.25F, 0.0F) == 0x03U &&
        encode_e2m1_stochastic_sat(1.25F, 0.999F) == 0x02U &&
        encode_e2m1_stochastic_sat(-1.25F, 0.999F) == 0x0aU;

    const bool nonfinite_policy =
        encode_e2m1_rne_sat(fp32::canonical_quiet_nan()) == 0x00U &&
        encode_e2m1_rne_sat(fp32::kPositiveInfinity) == 0x07U &&
        encode_e2m1_rne_sat(-fp32::kPositiveInfinity) == 0x0fU;

    return rne_ties_to_even && stochastic_endpoints && nonfinite_policy;
}

/**
 * @brief 验证标准 NVFP4 双层 scale、单元素变换和真实 nibble 打包。
 *
 * 使用格式规范中的 `amax = 6` 向量：global scale 应为 `1/448`，local
 * E4M3 scale 应为 `0x7e`（448），`{+6, -6}` 应编码并打包为 `0xf7`。
 *
 * @return scale、payload 与反量化值均正确时返回 true。
 */
[[nodiscard]] bool test_nvfp4_scaling_and_packing() {
    using quant_dequant::RoundingMode;
    using namespace quant_dequant::formats;

    const float global_scale = compute_nvfp4_global_scale(6.0F);
    const std::uint8_t local_scale =
        compute_nvfp4_local_scale_code(6.0F, global_scale);
    const std::uint8_t positive = encode_nvfp4_element(
        6.0F, local_scale, global_scale, RoundingMode::kNearest);
    const std::uint8_t negative = encode_nvfp4_element(
        -6.0F, local_scale, global_scale, RoundingMode::kNearest);
    const std::uint8_t packed = pack_e2m1_nibbles(positive, negative);

    const bool standard_vector = nearly_equal(global_scale, 1.0F / 448.0F) &&
        local_scale == 0x7eU && positive == 0x07U && negative == 0x0fU &&
        packed == 0xf7U && unpack_e2m1_low_nibble(packed) == positive &&
        unpack_e2m1_high_nibble(packed) == negative &&
        nearly_equal(decode_nvfp4_element(positive, local_scale, global_scale), 6.0F) &&
        nearly_equal(decode_nvfp4_element(negative, local_scale, global_scale), -6.0F);

    const bool zero_and_invalid_scales =
        compute_nvfp4_global_scale(0.0F) == 1.0F &&
        compute_nvfp4_local_scale_code(0.0F, 1.0F) == 0x00U &&
        compute_nvfp4_local_scale_code(1.0F, 0.0F) == kE4M3CanonicalNaNCode &&
        encode_nvfp4_element(1.0F, 0x00U, 1.0F, RoundingMode::kNearest) == 0x00U;

    return standard_vector && zero_and_invalid_scales;
}

/**
 * @brief 验证 device 版本的 NVFP4 codec 与已知 host bit pattern 一致。
 *
 * @return 无 CUDA device 时跳过并返回 true；否则全部结果相符时返回 true。
 */
[[nodiscard]] bool test_device_nvfp4_codec() {
    int device_count = 0;
    const cudaError_t device_query_status = cudaGetDeviceCount(&device_count);
    if (device_query_status == cudaErrorNoDevice ||
        device_query_status == cudaErrorInsufficientDriver) {
        std::cout << "CUDA runtime 不可用，跳过 NVFP4 device codec 测试。\n";
        return true;
    }

    if (!check_cuda(device_query_status, "cudaGetDeviceCount")) {
        return false;
    }

    if (device_count == 0) {
        std::cout << "未发现 CUDA 设备，跳过 NVFP4 device codec 测试。\n";
        return true;
    }

    constexpr std::size_t kOutputBytes = 5U;
    std::uint8_t* device_output = nullptr;
    if (!check_cuda(cudaMalloc(&device_output, kOutputBytes), "cudaMalloc")) {
        return false;
    }

    encodeNvfp4CodecKernel<<<1, 1>>>(device_output);
    if (!check_cuda(cudaGetLastError(), "encodeNvfp4CodecKernel 发射")) {
        static_cast<void>(cudaFree(device_output));
        return false;
    }

    std::array<std::uint8_t, kOutputBytes> host_output{};
    if (!check_cuda(cudaMemcpy(host_output.data(), device_output, kOutputBytes,
                               cudaMemcpyDeviceToHost),
                    "cudaMemcpy(DeviceToHost)")) {
        static_cast<void>(cudaFree(device_output));
        return false;
    }

    if (!check_cuda(cudaFree(device_output), "cudaFree")) {
        return false;
    }

    return host_output == std::array<std::uint8_t, kOutputBytes>{
                              0x06U, 0x7eU, 0x07U, 0xf7U, 0x0fU,
                          };
}

}  // namespace

/**
 * @brief 运行 NVFP4 codec 的 host/device 数值语义测试。
 *
 * @return 所有测试通过时返回 0；否则打印失败原因并返回 1。
 */
int run_nvfp4_codec_tests() {
    if (!test_e2m1_codebook_and_zero()) {
        std::cerr << "E2M1 codebook 或 signed-zero 规则测试失败。\n";
        return 1;
    }

    if (!test_e2m1_rounding_and_saturation()) {
        std::cerr << "E2M1 舍入、饱和或非有限回退测试失败。\n";
        return 1;
    }

    if (!test_nvfp4_scaling_and_packing()) {
        std::cerr << "NVFP4 双层 scale、编码或 nibble 打包测试失败。\n";
        return 1;
    }

    if (!test_device_nvfp4_codec()) {
        std::cerr << "NVFP4 device codec 测试失败。\n";
        return 1;
    }

    return 0;
}
