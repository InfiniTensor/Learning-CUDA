#include <array>
#include <cstdint>
#include <iostream>

#include <cuda_runtime.h>

#include "formats/mxfp8_codec.cuh"

namespace {

/**
 * @brief 检查 CUDA runtime 调用状态，并在失败时打印诊断信息。
 *
 * @param status 待检查的 CUDA runtime 返回值。
 * @param operation 正在执行的 CUDA 操作说明。
 * @return status 为 cudaSuccess 时返回 true。
 */
[[nodiscard]] bool check_cuda(const cudaError_t status, const char* operation) {
    if (status == cudaSuccess) {
        return true;
    }

    std::cerr << operation << " 失败：" << cudaGetErrorString(status) << '\n';
    return false;
}

/**
 * @brief 在 device 路径验证 -0 的 scale 与 payload 规范化规则。
 *
 * grid 只包含一个 CTA 和一个线程：本测试不测并行性能，只确认
 * `__device__` 版本的 codec 与 host 版本具有相同的零语义。
 *
 * @param output device 指针，指向至少 4 字节的输出数组。
 */
__global__ void encodeCanonicalZeroKernel(std::uint8_t* __restrict__ output) {
    output[0] = quant_dequant::formats::compute_mxfp8_scale_code(-0.0F);
    output[1] = quant_dequant::formats::encode_e4m3_rne_sat(-0.0F);
    output[2] = quant_dequant::formats::encode_e4m3_stochastic_sat(-0.0F, 0.5F);
    output[3] = quant_dequant::formats::encode_mxfp8_element(
        -0.0F, 0x00U, quant_dequant::RoundingMode::kNearest);
}

/**
 * @brief 验证 host codec 对 -0 的项目规范化规则。
 *
 * @return 三个编码接口都产生正零 code 时返回 true。
 */
[[nodiscard]] bool test_host_canonical_zero() {
    using quant_dequant::RoundingMode;
    using namespace quant_dequant::formats;

    const bool scale_is_min_finite =
        compute_mxfp8_scale_code(-0.0F) == 0x00U;
    const bool rne_payload_is_positive_zero =
        encode_e4m3_rne_sat(-0.0F) == 0x00U;
    const bool stochastic_payload_is_positive_zero =
        encode_e4m3_stochastic_sat(-0.0F, 0.5F) == 0x00U;
    const bool mxfp8_payload_is_positive_zero =
        encode_mxfp8_element(-0.0F, 0x00U, RoundingMode::kNearest) == 0x00U;

    return scale_is_min_finite && rne_payload_is_positive_zero &&
           stochastic_payload_is_positive_zero && mxfp8_payload_is_positive_zero;
}

/**
 * @brief 验证解码器仍能保留外部文件中的合法 E4M3 -0。
 *
 * 编码器选择 canonical +0 是项目输出约定；读取已有文件时，`0x80` 仍应
 * 解码为带负号的 FP32 零，不能把两种编码混为一谈。
 *
 * @return `0x80` 解码为 -0 时返回 true。
 */
[[nodiscard]] bool test_signed_zero_decode() {
    const float decoded = quant_dequant::formats::decode_e4m3(0x80U);
    return decoded == 0.0F &&
           quant_dequant::formats::fp32::has_negative_sign(decoded);
}

/**
 * @brief 验证 device codec 对 -0 的项目规范化规则。
 *
 * @return device 计算出的 scale 与三个 payload 均为 `0x00` 时返回 true。
 */
[[nodiscard]] bool test_device_canonical_zero() {
    int device_count = 0;
    const cudaError_t device_query_status = cudaGetDeviceCount(&device_count);

    if (device_query_status == cudaErrorNoDevice ||
        device_query_status == cudaErrorInsufficientDriver) {
        std::cout << "CUDA runtime 不可用，跳过 MXFP8 device 零边界测试。\n";
        return true;
    }

    if (!check_cuda(device_query_status, "cudaGetDeviceCount")) {
        return false;
    }

    if (device_count == 0) {
        std::cout << "未发现 CUDA 设备，跳过 MXFP8 device 零边界测试。\n";
        return true;
    }

    std::uint8_t* device_output = nullptr;
    constexpr std::size_t kOutputBytes = 4U;

    if (!check_cuda(cudaMalloc(&device_output, kOutputBytes), "cudaMalloc")) {
        return false;
    }

    encodeCanonicalZeroKernel<<<1, 1>>>(device_output);
    if (!check_cuda(cudaGetLastError(), "encodeCanonicalZeroKernel 发射")) {
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
                              0x00U, 0x00U, 0x00U, 0x00U,
                          };
}

}  // namespace

/**
 * @brief 运行 MXFP8 codec 的零边界规则测试。
 *
 * @return 所有 host/device 零边界测试通过时返回 0；失败时返回 1。
 */
int run_mxfp8_codec_tests() {
    if (!test_host_canonical_zero()) {
        std::cerr << "MXFP8 host 的 -0 规范化测试失败。\n";
        return 1;
    }

    if (!test_signed_zero_decode()) {
        std::cerr << "MXFP8 E4M3 -0 解码测试失败。\n";
        return 1;
    }

    if (!test_device_canonical_zero()) {
        std::cerr << "MXFP8 device 的 -0 规范化测试失败。\n";
        return 1;
    }

    return 0;
}
