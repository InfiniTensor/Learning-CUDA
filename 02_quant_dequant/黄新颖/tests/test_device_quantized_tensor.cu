#include <cuda_runtime_api.h>

#include <cstdint>
#include <exception>
#include <iostream>
#include <type_traits>

#include "pipeline/device_quantized_tensor.cuh"

namespace {

/**
 * @brief 构造一个包含尾 block 的合法 MXFP8 device 输出描述。
 *
 * @return 形状为 1 x 33、block 模式的 MXFP8 描述。
 */
[[nodiscard]] quant_dequant::QuantizedTensorDesc make_mxfp8_desc() {
    return {
        .source_desc = {
            .num_rows = 1U,
            .num_cols = 33U,
            .dtype = quant_dequant::DType::kFloat32,
        },
        .format = quant_dequant::QuantFormat::kMxfp8,
        .scale_mode = quant_dequant::ScaleMode::kBlock,
        .rounding = quant_dequant::RoundingMode::kNearest,
        .stochastic_seed = 0U,
        .block_size = quant_dequant::kMxfp8BlockSize,
        .scale_layout = quant_dequant::ScaleLayout::kRowwise,
    };
}

/**
 * @brief 构造一个包含奇数 payload 尾元素的合法 NVFP4 device 输出描述。
 *
 * @return 形状为 1 x 17、block 模式的 NVFP4 描述。
 */
[[nodiscard]] quant_dequant::QuantizedTensorDesc make_nvfp4_desc() {
    return {
        .source_desc = {
            .num_rows = 1U,
            .num_cols = 17U,
            .dtype = quant_dequant::DType::kFloat16,
        },
        .format = quant_dequant::QuantFormat::kNvfp4,
        .scale_mode = quant_dequant::ScaleMode::kBlock,
        .rounding = quant_dequant::RoundingMode::kNearest,
        .stochastic_seed = 0U,
        .block_size = quant_dequant::kNvfp4BlockSize,
        .scale_layout = quant_dequant::ScaleLayout::kRowwise,
    };
}

/**
 * @brief 构造一个合法的 BF16 反量化 device 输出描述。
 *
 * device 上仍将分配 FP32 数组；BF16 只记录最终 QDTENSOR 应写出的物理类型。
 *
 * @return 形状为 1 x 33、目标物理类型为 BF16 的输出描述。
 */
[[nodiscard]] quant_dequant::TensorDesc make_dequantized_output_desc() {
    return {
        .num_rows = 1U,
        .num_cols = 33U,
        .dtype = quant_dequant::DType::kBFloat16,
    };
}

/**
 * @brief 判断当前运行环境是否可以执行会分配 device memory 的测试。
 *
 * @return 已发现可用 CUDA device 时返回 true；无 driver 或无 device 时返回 false。
 */
[[nodiscard]] bool has_cuda_device() {
    int device_count = 0;
    const cudaError_t status = cudaGetDeviceCount(&device_count);

    if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver) {
        std::cout << "CUDA runtime 不可用，跳过 device buffer 分配测试。\n";
        return false;
    }

    if (status != cudaSuccess) {
        std::cerr << "cudaGetDeviceCount 失败：" << cudaGetErrorString(status) << '\n';
        return false;
    }

    if (device_count == 0) {
        std::cout << "未发现 CUDA device，跳过 device buffer 分配测试。\n";
        return false;
    }

    return true;
}

}  // namespace

/**
 * @brief 验证 pipeline 内部 Thrust device 所有权对象的结构不变量。
 *
 * 无 device 的环境仍会验证 move-only 语义、默认对象与非法描述拒绝；可用 GPU
 * 上会进一步验证 MXFP8/NVFP4 的精确 buffer 长度和 global scale 存在性。
 *
 * @return 所有检查通过时返回 0；否则打印诊断并返回 1。
 */
int run_device_quantized_tensor_tests() {
    using quant_dequant::pipeline::DeviceQuantizationInput;
    using quant_dequant::pipeline::DeviceDequantizationOutput;
    using quant_dequant::pipeline::DeviceQuantizedTensor;
    using quant_dequant::pipeline::DeviceTensorQuantizationWorkspace;

    static_assert(!std::is_copy_constructible_v<DeviceQuantizationInput>);
    static_assert(!std::is_copy_assignable_v<DeviceQuantizationInput>);
    static_assert(std::is_move_constructible_v<DeviceQuantizationInput>);
    static_assert(!std::is_copy_constructible_v<DeviceDequantizationOutput>);
    static_assert(!std::is_copy_assignable_v<DeviceDequantizationOutput>);
    static_assert(std::is_move_constructible_v<DeviceDequantizationOutput>);
    static_assert(!std::is_copy_constructible_v<DeviceTensorQuantizationWorkspace>);
    static_assert(!std::is_copy_assignable_v<DeviceTensorQuantizationWorkspace>);
    static_assert(std::is_move_constructible_v<DeviceTensorQuantizationWorkspace>);
    static_assert(!std::is_copy_constructible_v<DeviceQuantizedTensor>);
    static_assert(!std::is_copy_assignable_v<DeviceQuantizedTensor>);
    static_assert(std::is_move_constructible_v<DeviceQuantizedTensor>);

    const DeviceQuantizationInput empty_input{};
    const DeviceDequantizationOutput empty_dequantized_output{};
    const DeviceQuantizedTensor empty_output{};
    const DeviceTensorQuantizationWorkspace empty_tensor_workspace{};
    if (empty_input.isConsistent() || empty_dequantized_output.isConsistent() ||
        empty_output.isConsistent() || empty_tensor_workspace.isConsistent()) {
        std::cerr << "默认 device 数据对象不应被识别为有效张量。\n";
        return 1;
    }

    try {
        static_cast<void>(
            DeviceQuantizedTensor::allocate(quant_dequant::QuantizedTensorDesc{}));
        std::cerr << "非法量化描述未被 device buffer 分配器拒绝。\n";
        return 1;
    } catch (const std::invalid_argument&) {
    } catch (const std::exception& error) {
        std::cerr << "非法描述产生了错误的异常类型：" << error.what() << '\n';
        return 1;
    }

    try {
        static_cast<void>(DeviceTensorQuantizationWorkspace::allocate(0U));
        std::cerr << "零长度 tensor-scale partial 工作区未被拒绝。\n";
        return 1;
    } catch (const std::invalid_argument&) {
    } catch (const std::exception& error) {
        std::cerr << "非法 tensor-scale 工作区产生了错误异常类型："
                  << error.what() << '\n';
        return 1;
    }

    if (!has_cuda_device()) {
        return 0;
    }

    try {
        const DeviceQuantizedTensor mxfp8_output =
            DeviceQuantizedTensor::allocate(make_mxfp8_desc());
        if (!mxfp8_output.isConsistent() || mxfp8_output.payload.size() != 33U ||
            mxfp8_output.local_scales.size() != 2U ||
            mxfp8_output.global_scale.has_value()) {
            std::cerr << "MXFP8 device buffer 结构或长度不符合预期。\n";
            return 1;
        }

        const DeviceQuantizedTensor nvfp4_output =
            DeviceQuantizedTensor::allocate(make_nvfp4_desc());
        if (!nvfp4_output.isConsistent() || nvfp4_output.payload.size() != 9U ||
            nvfp4_output.local_scales.size() != 2U ||
            !nvfp4_output.global_scale.has_value() ||
            nvfp4_output.global_scale->size() != 1U) {
            std::cerr << "NVFP4 device buffer 结构或长度不符合预期。\n";
            return 1;
        }

        const DeviceDequantizationOutput dequantized_output =
            DeviceDequantizationOutput::allocate(make_dequantized_output_desc());
        if (!dequantized_output.isConsistent() ||
            dequantized_output.values.size() != 33U ||
            dequantized_output.desc.dtype != quant_dequant::DType::kBFloat16) {
            std::cerr << "反量化 device FP32 输出结构或长度不符合预期。\n";
            return 1;
        }

        const DeviceTensorQuantizationWorkspace tensor_workspace =
            DeviceTensorQuantizationWorkspace::allocate(3U);
        if (!tensor_workspace.isConsistent() ||
            tensor_workspace.partial_amax.size() != 3U ||
            tensor_workspace.partial_nonfinite.size() != 3U) {
            std::cerr << "tensor-scale partial 工作区结构或长度不符合预期。\n";
            return 1;
        }

        DeviceQuantizationInput input{};
        input.desc = make_mxfp8_desc().source_desc;
        input.values.resize(33U);
        if (!input.isConsistent()) {
            std::cerr << "合法的 device FP32 输入没有通过结构校验。\n";
            return 1;
        }
    } catch (const std::exception& error) {
        std::cerr << "device 数据结构测试意外失败：" << error.what() << '\n';
        return 1;
    }

    return 0;
}
