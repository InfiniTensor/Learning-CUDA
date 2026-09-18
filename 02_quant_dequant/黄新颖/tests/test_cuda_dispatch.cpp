#include <cuda_runtime_api.h>

#include <exception>
#include <iostream>
#include <optional>
#include <string_view>

#include "quant_dequant/quantize.hpp"
#include "pipeline/pipeline_detail.hpp"

namespace {

using quant_dequant::CudaPipelineError;
using quant_dequant::DequantizationConfig;
using quant_dequant::DType;
using quant_dequant::HostTensor;
using quant_dequant::QuantFormat;
using quant_dequant::QuantizationConfig;
using quant_dequant::QuantizedTensor;
using quant_dequant::QuantizedTensorDesc;
using quant_dequant::RoundingMode;
using quant_dequant::ScaleMode;
using quant_dequant::TensorDesc;
using quant_dequant::kMxfp8BlockSize;
using quant_dequant::kNvfp4BlockSize;

/**
 * @brief 验证调用以包含指定文本的 `CudaPipelineError` 失败。
 *
 * 本测试不检查 CUDA runtime，也不要求机器有可用 GPU。它验证公共入口会拒绝
 * 无效参数，并在无 device 时确认合法请求确实抵达格式专用路径。
 *
 * @tparam Callable 无参且会调用 CUDA pipeline 公共接口的可调用对象。
 * @param callable 待验证的调用。
 * @param expected_detail 错误诊断中应出现的阶段或格式名称。
 * @return 预期异常类型及诊断均匹配时为 `true`。
 */
template <typename Callable>
[[nodiscard]] bool throws_cuda_pipeline_error(
    Callable&& callable,
    std::string_view expected_detail) {
    try {
        callable();
    } catch (const CudaPipelineError& error) {
        return std::string_view{error.what()}.find(expected_detail) !=
               std::string_view::npos;
    } catch (const std::exception&) {
        return false;
    }

    return false;
}

/**
 * @brief 判断本机是否能执行会实际分配 device memory 的 MXFP8 路径。
 *
 * 无 CUDA device 时，公共 MXFP8 入口会在 H2D / allocation 阶段失败，无法到达
 * block/tensor launcher；这种环境仍可验证其公共错误边界，但不能断言 launcher
 * 的专用错误文本。
 *
 * @return CUDA runtime 成功发现至少一个 device 时返回 true。
 */
[[nodiscard]] bool has_cuda_device() {
    int device_count = 0;
    return cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0;
}

/**
 * @brief 构造一元素的合法 host FP32 输入。
 *
 * @return 可供 MXFP8 与 NVFP4 格式分派测试共享的输入张量。
 */
[[nodiscard]] HostTensor make_input() {
    return HostTensor{
        .desc = TensorDesc{
            .num_rows = 1,
            .num_cols = 1,
            .dtype = DType::kFloat32,
        },
        .values = {1.0F},
    };
}

/**
 * @brief 构造可通过一致性校验的最小 MXFP8 host 量化张量。
 *
 * @return 一元素、一个 E8M0 local scale 的 MXFP8 张量。
 */
[[nodiscard]] QuantizedTensor make_mxfp8_tensor() {
    return QuantizedTensor{
        .desc = QuantizedTensorDesc{
            .source_desc = TensorDesc{
                .num_rows = 1,
                .num_cols = 1,
                .dtype = DType::kFloat32,
            },
            .format = QuantFormat::kMxfp8,
            .scale_mode = ScaleMode::kBlock,
            .rounding = RoundingMode::kNearest,
            .stochastic_seed = 0,
            .block_size = kMxfp8BlockSize,
        },
        .payload = {0x00U},
        .local_scales = {0x00U},
        .global_scale = std::nullopt,
    };
}

/**
 * @brief 构造可通过一致性校验的最小 NVFP4 host 量化张量。
 *
 * 一字节保存两个 FP4 nibble；一元素张量中高四位是 tail padding。其数值对于
 * 本测试无关；有 CUDA device 时的数值正确性由专门对照测试覆盖。
 *
 * @return 一元素、一个 E4M3 local scale 与一个 FP32 global scale 的 NVFP4 张量。
 */
[[nodiscard]] QuantizedTensor make_nvfp4_tensor() {
    return QuantizedTensor{
        .desc = QuantizedTensorDesc{
            .source_desc = TensorDesc{
                .num_rows = 1,
                .num_cols = 1,
                .dtype = DType::kFloat16,
            },
            .format = QuantFormat::kNvfp4,
            .scale_mode = ScaleMode::kBlock,
            .rounding = RoundingMode::kNearest,
            .stochastic_seed = 0,
            .block_size = kNvfp4BlockSize,
        },
        .payload = {0x00U},
        .local_scales = {0x00U},
        .global_scale = 1.0F,
    };
}

}  // namespace

int run_cuda_dispatch_tests() {
    const HostTensor input = make_input();
    const bool cuda_device_available = has_cuda_device();

    QuantizationConfig mxfp8_config{};
    mxfp8_config.format = QuantFormat::kMxfp8;
    mxfp8_config.block_size = kMxfp8BlockSize;
    mxfp8_config.scale_mode = ScaleMode::kBlock;
    mxfp8_config.rounding = RoundingMode::kNearest;

    // MXFP8 block-scale 已是可执行 CUDA 路径。无 CUDA device 的环境只能在
    // H2D/分配阶段验证错误边界；有 device 时的实际数值对照由
    // test_mxfp8_cuda.cu 负责，避免此分派测试重复执行 kernel。
    if (!cuda_device_available && !throws_cuda_pipeline_error(
                                      [&] {
                                          static_cast<void>(
                                              quant_dequant::quantize_cuda(
                                                  input, mxfp8_config));
                                      },
                                      "mxfp8")) {
        std::cerr << "CUDA MXFP8 量化没有分派到预期的格式专用入口。\n";
        return 1;
    }

    QuantizationConfig mxfp8_tensor_config = mxfp8_config;
    mxfp8_tensor_config.scale_mode = ScaleMode::kTensor;
    // MXFP8 tensor-scale 现在也会实际完成两阶段 amax reduction 和编码；有
    // device 时由 test_mxfp8_cuda.cu 比较 CPU reference。这里仅在无 device
    // 环境确认公共入口到达 CUDA 格式专用路径后正确报告 runtime/H2D 边界错误。
    if (!cuda_device_available && !throws_cuda_pipeline_error(
                                      [&] {
                                          static_cast<void>(
                                              quant_dequant::quantize_cuda(
                                                  input, mxfp8_tensor_config));
                                      },
                                      "mxfp8")) {
        std::cerr << "CUDA MXFP8 tensor-scale 量化没有分派到预期的 launcher。\n";
        return 1;
    }

    QuantizationConfig nvfp4_config{};
    nvfp4_config.format = QuantFormat::kNvfp4;
    nvfp4_config.block_size = kNvfp4BlockSize;
    nvfp4_config.scale_mode = ScaleMode::kBlock;
    nvfp4_config.rounding = RoundingMode::kNearest;

    // NVFP4 量化现已可执行。无 device 时它应和 MXFP8 一样在 H2D、allocation 或
    // launcher 查询边界报告 nvfp4 CUDA 异常；有 device 时的逐字节 CPU 对照由
    // test_nvfp4_cuda.cu 负责，避免这个 dispatcher 测试重复发射 kernel。
    if (!cuda_device_available && !throws_cuda_pipeline_error(
                                      [&] {
                                          static_cast<void>(
                                              quant_dequant::quantize_cuda(
                                                  input, nvfp4_config));
                                      },
                                      "nvfp4")) {
        std::cerr << "CUDA NVFP4 量化没有分派到预期的格式专用入口。\n";
        return 1;
    }

    const DequantizationConfig dequantization_config{
        .output_type = DType::kFloat32,
    };
    const QuantizedTensor mxfp8 = make_mxfp8_tensor();
    // MXFP8 反量化已是可执行 CUDA 路径。无 CUDA device 时这里只验证 H2D/
    // 分配阶段的错误边界；有 device 时的 CPU reference 数值对照由
    // test_mxfp8_dequantize_cuda.cu 覆盖。
    if (!cuda_device_available && !throws_cuda_pipeline_error(
                                      [&] {
                                          static_cast<void>(
                                              quant_dequant::dequantize_cuda(
                                                  mxfp8, dequantization_config));
                                      },
                                      "mxfp8")) {
        std::cerr << "CUDA MXFP8 反量化没有分派到预期的格式专用入口。\n";
        return 1;
    }

    const QuantizedTensor nvfp4 = make_nvfp4_tensor();
    if (!cuda_device_available && !throws_cuda_pipeline_error(
                                      [&] {
                                          static_cast<void>(
                                              quant_dequant::dequantize_cuda(
                                                  nvfp4, dequantization_config));
                                      },
                                      "nvfp4")) {
        std::cerr << "CUDA NVFP4 反量化没有分派到预期的格式专用入口。\n";
        return 1;
    }

    const HostTensor invalid_input{
        .desc = TensorDesc{
            .num_rows = 1,
            .num_cols = 1,
            .dtype = DType::kBFloat16,
        },
        .values = {1.0F},
    };
    if (!throws_cuda_pipeline_error(
            [&] {
                static_cast<void>(
                    quant_dequant::quantize_cuda(invalid_input, mxfp8_config));
            },
            "FP16/FP32")) {
        std::cerr << "CUDA 量化入口没有拒绝不支持的 BF16 输入描述。\n";
        return 1;
    }

    // `pipeline_detail` 不是公共 API，但内部入口也必须自行维护 MXFP8 的固定
    // 格式前提，避免将来某个内部调用绕过 dispatcher 后进入错误的 kernel 映射。
    QuantizationConfig invalid_mxfp8_config = mxfp8_config;
    invalid_mxfp8_config.block_size = kNvfp4BlockSize;
    if (!throws_cuda_pipeline_error(
            [&] {
                static_cast<void>(quant_dequant::pipeline_detail::quantize_mxfp8_cuda(
                    input, invalid_mxfp8_config));
            },
            "block_size = 32")) {
        std::cerr << "MXFP8 CUDA 内部入口没有拒绝非 32 元素 block 配置。\n";
        return 1;
    }

    if (!throws_cuda_pipeline_error(
            [&] {
                static_cast<void>(quant_dequant::pipeline_detail::quantize_mxfp8_cuda(
                    invalid_input, mxfp8_config));
            },
            "FP16/FP32")) {
        std::cerr << "MXFP8 CUDA 内部入口没有拒绝非法输入描述。\n";
        return 1;
    }

    if (!throws_cuda_pipeline_error(
            [&] {
                static_cast<void>(quant_dequant::dequantize_cuda(
                    QuantizedTensor{}, dequantization_config));
            },
            "自洽")) {
        std::cerr << "CUDA 反量化入口没有拒绝不自洽的量化张量。\n";
        return 1;
    }

    return 0;
}
