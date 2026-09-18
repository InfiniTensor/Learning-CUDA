#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>

#include <thrust/device_vector.h>

#include "quant_dequant/quantized_tensor.hpp"

namespace quant_dequant::pipeline {
namespace detail {

/**
 * @brief 将文件/描述中的无符号长度安全转换为 Thrust 容器使用的 size_t。
 *
 * 本函数只检查 host 侧容器长度是否可表示；实际 CUDA allocation 是否成功由
 * `thrust::device_vector` 自身报告。它被刻意放在 pipeline 内部，避免把 Thrust
 * 相关的长度约束泄漏到公共的数据模型中。
 *
 * @param count 要转换的元素或字节数量。
 * @param buffer_name 用于异常信息的 device buffer 名称。
 * @return 可安全传给 `thrust::device_vector` 构造函数的长度。
 * @throws std::overflow_error count 超出当前平台 size_t 范围时抛出。
 */
[[nodiscard]] inline std::size_t checked_device_buffer_size(
    const std::uint64_t count,
    const char* const buffer_name) {
    constexpr std::uint64_t kMaxSize =
        static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max());

    if (count > kMaxSize) {
        throw std::overflow_error{
            "device buffer \"" + std::string{buffer_name} +
            "\" 的长度超出当前平台 size_t 范围。"};
    }

    return static_cast<std::size_t>(count);
}

}  // namespace detail

/**
 * @brief 量化 pipeline 在 device 端持有的统一 FP32 输入张量。
 *
 * QDTENSOR 输入文件可以是 FP16 或 FP32，但在 host I/O 边界已经统一扩展为 FP32。
 * 因而 CUDA quantize kernel 始终读取 `values` 中的 FP32；`desc.dtype` 仍保留
 * 原始文件的物理 dtype，以便最终 QDWGT 正确记录 `source_dtype`。
 *
 * 这是量化流程专用对象，不是泛用 GPU Tensor。它只能在 `pipeline/` 层使用，
 * CUDA kernel 由 pipeline 通过 `thrust::raw_pointer_cast(values.data())` 取得
 * 非拥有 raw device pointer，不应直接接收本对象。
 */
struct DeviceQuantizationInput {
    /** 原始输入文件的形状和物理 dtype；只在 host 侧解释。 */
    TensorDesc desc{};

    /** 按 row-major 顺序保存的 device FP32 输入值。 */
    thrust::device_vector<float> values{};

    DeviceQuantizationInput() = default;
    DeviceQuantizationInput(const DeviceQuantizationInput&) = delete;
    DeviceQuantizationInput& operator=(const DeviceQuantizationInput&) = delete;
    DeviceQuantizationInput(DeviceQuantizationInput&&) = default;
    DeviceQuantizationInput& operator=(DeviceQuantizationInput&&) = default;
    ~DeviceQuantizationInput() = default;

    /**
     * @brief 检查 host 描述和 device FP32 输入数组的结构是否匹配。
     *
     * 本检查不会访问 device 数据，因此不会触发 H2D/D2H 拷贝或 CUDA 同步；它只
     * 验证 shape、第一版允许的物理输入 dtype 与 device vector 长度。
     *
     * @return 描述合法、输入 dtype 为 FP16/FP32 且 values 长度正确时返回 true。
     */
    [[nodiscard]] bool isConsistent() const noexcept {
        const auto element_count = desc.elementCount();
        if (!desc.isValid() || !is_supported_input_dtype(desc.dtype) ||
            !element_count.has_value() ||
            *element_count >
                static_cast<std::uint64_t>(
                    std::numeric_limits<std::size_t>::max())) {
            return false;
        }

        return values.size() == static_cast<std::size_t>(*element_count);
    }
};

/**
 * @brief MXFP8 tensor-scale 量化的临时 device 规约工作区。
 *
 * tensor mode 的唯一 E8M0 scale 依赖整张矩阵的 `amax`，不能像 block mode 那样
 * 在单个 warp 内完成。第一阶段 persistent reduction kernel 为每个 CTA 写一项
 * partial `amax` 与“是否发现 NaN/Inf”的标记；第二阶段单 CTA 再规约这两个数组，
 * 并直接写 `local_scales[0]`。
 *
 * 工作区只在一次 `quantize_mxfp8_cuda()` 调用内存活，既不属于最终 QDWGT，也
 * 不应被序列化。它用两个简单的 device vector 而非一个 struct vector，使两种
 * 标量的对齐、访问和最终 warp 规约都保持直接。
 */
struct DeviceTensorQuantizationWorkspace {
    /** 每个第一阶段 CTA 写出的有限输入最大绝对值。 */
    thrust::device_vector<float> partial_amax{};

    /** 每个第一阶段 CTA 写出的非有限输入标记；0 表示无，非 0 表示存在。 */
    thrust::device_vector<std::uint32_t> partial_nonfinite{};

    DeviceTensorQuantizationWorkspace() = default;
    DeviceTensorQuantizationWorkspace(
        const DeviceTensorQuantizationWorkspace&) = delete;
    DeviceTensorQuantizationWorkspace& operator=(
        const DeviceTensorQuantizationWorkspace&) = delete;
    DeviceTensorQuantizationWorkspace(DeviceTensorQuantizationWorkspace&&) = default;
    DeviceTensorQuantizationWorkspace& operator=(
        DeviceTensorQuantizationWorkspace&&) = default;
    ~DeviceTensorQuantizationWorkspace() = default;

    /**
     * @brief 分配具有指定 partial CTA 数量的临时规约数组。
     *
     * `partial_count` 必须与 tensor-scale launcher 将要发射的第一阶段 grid.x
     * 相同。pipeline 先向 launcher 查询该数量再分配，使工作区在最终编码和
     * D2H 完成前都持续拥有 kernel 所借用的 device 内存。
     *
     * @param partial_count 第一阶段 persistent reduction CTA 数量。
     * @return 持有同长度 amax 和非有限标记数组的 move-only 工作区。
     * @throws std::invalid_argument partial_count 为 0 时抛出。
     * @throws thrust::system_error CUDA runtime 或 device allocation 失败时抛出。
     */
    [[nodiscard]] static DeviceTensorQuantizationWorkspace allocate(
        const std::size_t partial_count) {
        if (partial_count == 0U) {
            throw std::invalid_argument{
                "DeviceTensorQuantizationWorkspace 需要至少一项 partial 结果。"};
        }

        DeviceTensorQuantizationWorkspace workspace{};
        workspace.partial_amax = thrust::device_vector<float>{partial_count};
        workspace.partial_nonfinite =
            thrust::device_vector<std::uint32_t>{partial_count};
        return workspace;
    }

    /**
     * @brief 检查两个临时规约数组的长度是否匹配且非空。
     *
     * 该检查只读取 Thrust 容器元数据，不访问 device 数值或同步 CUDA stream。
     *
     * @return 两个数组均非空且长度相同返回 true。
     */
    [[nodiscard]] bool isConsistent() const noexcept {
        return !partial_amax.empty() &&
               partial_amax.size() == partial_nonfinite.size();
    }
};

/**
 * @brief CUDA quantize pipeline 在 device 端持有的低精度量化结果。
 *
 * 该对象的 host 元数据复用 `QuantizedTensorDesc`，使 CPU reference、QDWGT I/O
 * 与 CUDA 路径对 payload 字节数、局部 scale 数量、block 布局和格式约束使用同一
 * 套规则。真正位于 device 的 payload、local_scales 与 NVFP4 global_scale 全部
 * 使用 `thrust::device_vector` 管理，遵循 RAII（资源获取即初始化）生命周期。
 *
 * MXFP8 中 `global_scale` 必须为空；NVFP4 中它必须存在且只含一个 FP32 元素。
 * 这个元素的数值由 future NVFP4 quantize kernel 写入。`isConsistent()` 只做
 * 不同步的结构检查，不能也不应读取该 device scalar 验证其数值。
 */
struct DeviceQuantizedTensor {
    /** 与三个 device buffer 共同解释的 host 侧量化元数据。 */
    QuantizedTensorDesc desc{};

    /** 真实位宽的 device payload：MXFP8 为 byte，NVFP4 为 packed nibble byte。 */
    thrust::device_vector<std::uint8_t> payload{};

    /** device 局部 scale code：MXFP8 为 E8M0，NVFP4 为 E4M3。 */
    thrust::device_vector<std::uint8_t> local_scales{};

    /** NVFP4 的单元素 device FP32 解码方向 global scale；MXFP8 不存在。 */
    std::optional<thrust::device_vector<float>> global_scale{};

    DeviceQuantizedTensor() = default;
    DeviceQuantizedTensor(const DeviceQuantizedTensor&) = delete;
    DeviceQuantizedTensor& operator=(const DeviceQuantizedTensor&) = delete;
    DeviceQuantizedTensor(DeviceQuantizedTensor&&) = default;
    DeviceQuantizedTensor& operator=(DeviceQuantizedTensor&&) = default;
    ~DeviceQuantizedTensor() = default;

    /**
     * @brief 按已验证的量化描述分配所有需要的 device buffer。
     *
     * MXFP8 只分配 payload 和 local_scales；NVFP4 额外分配长度为 1 的
     * `global_scale`。本函数不初始化量化 code，也不执行 H2D 拷贝；调用方应在
     * 后续 kernel 或显式拷贝中填充各 buffer。
     *
     * @param tensor_desc 已知格式、shape、block size 和 scale 布局的量化描述。
     * @return 持有精确长度 device buffer 的 move-only DeviceQuantizedTensor。
     * @throws std::invalid_argument 描述不是合法的量化结果元数据时抛出。
     * @throws std::overflow_error 推导长度超出当前平台 size_t 时抛出。
     * @throws thrust::system_error CUDA runtime 或 device allocation 失败时抛出。
     */
    [[nodiscard]] static DeviceQuantizedTensor allocate(
        const QuantizedTensorDesc& tensor_desc) {
        const auto expected_payload_bytes = tensor_desc.expectedPayloadBytes();
        const auto expected_local_scale_count =
            tensor_desc.expectedLocalScaleCount();

        if (!tensor_desc.isMetadataValid() || !expected_payload_bytes.has_value() ||
            !expected_local_scale_count.has_value()) {
            throw std::invalid_argument{
                "DeviceQuantizedTensor 需要合法的 QuantizedTensorDesc。"};
        }

        DeviceQuantizedTensor output{};
        output.desc = tensor_desc;
        output.payload = thrust::device_vector<std::uint8_t>{
            detail::checked_device_buffer_size(
                *expected_payload_bytes, "payload")};
        output.local_scales = thrust::device_vector<std::uint8_t>{
            detail::checked_device_buffer_size(
                *expected_local_scale_count, "local_scales")};

        if (tensor_desc.usesGlobalScale()) {
            output.global_scale.emplace(1U);
        }

        return output;
    }

    /**
     * @brief 检查 host 元数据和 device buffer 长度的结构是否匹配。
     *
     * 此函数绝不解引用 device pointer，因此不会同步 CUDA stream。对 NVFP4，
     * 它只要求 global_scale buffer 存在且长度为 1；其中是否已经写入有限正值
     * 由量化 kernel 完成后、D2H 构造 Host QuantizedTensor 时验证。
     *
     * @return 所有 buffer 长度及 MXFP8/NVFP4 的 global scale 存在性正确时返回 true。
     */
    [[nodiscard]] bool isConsistent() const noexcept {
        const auto expected_payload_bytes = desc.expectedPayloadBytes();
        const auto expected_local_scale_count =
            desc.expectedLocalScaleCount();
        constexpr std::uint64_t kMaxSize =
            static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max());

        if (!desc.isMetadataValid() || !expected_payload_bytes.has_value() ||
            !expected_local_scale_count.has_value() ||
            *expected_payload_bytes > kMaxSize ||
            *expected_local_scale_count > kMaxSize ||
            payload.size() != static_cast<std::size_t>(*expected_payload_bytes) ||
            local_scales.size() !=
                static_cast<std::size_t>(*expected_local_scale_count)) {
            return false;
        }

        if (desc.format == QuantFormat::kMxfp8) {
            return !global_scale.has_value();
        }

        return desc.format == QuantFormat::kNvfp4 && global_scale.has_value() &&
               global_scale->size() == 1U;
    }
};

/**
 * @brief CUDA dequantize pipeline 在 device 端持有的统一 FP32 输出张量。
 *
 * 反量化 kernel 的计算结果始终先写为 FP32：E4M3 payload 与 E8M0 scale 的
 * 解码、相乘都可以直接在 FP32 中完成，避免为 FP16、BF16、FP32 维护三套 kernel。
 * `desc.dtype` 仅记录随后 QDTENSOR I/O 应写出的物理类型；D2H 后的
 * `HostTensor::values` 仍保持 FP32，实际窄化由 `write_dequantized_tensor()`
 * 统一完成。
 *
 * 这是 dequantize 流程专用对象，而非泛用 GPU Tensor。CUDA kernel 通过
 * `thrust::raw_pointer_cast(values.data())` 取得不拥有所有权的 raw device
 * pointer，不应按值接收本对象。
 */
struct DeviceDequantizationOutput {
    /** 输出矩阵的 row-major 形状与目标 QDTENSOR 物理 dtype。 */
    TensorDesc desc{};

    /** 按 row-major 顺序保存的 device FP32 反量化数值。 */
    thrust::device_vector<float> values{};

    DeviceDequantizationOutput() = default;
    DeviceDequantizationOutput(const DeviceDequantizationOutput&) = delete;
    DeviceDequantizationOutput& operator=(const DeviceDequantizationOutput&) = delete;
    DeviceDequantizationOutput(DeviceDequantizationOutput&&) = default;
    DeviceDequantizationOutput& operator=(DeviceDequantizationOutput&&) = default;
    ~DeviceDequantizationOutput() = default;

    /**
     * @brief 按目标输出描述分配 device FP32 结果数组。
     *
     * 虽然 `desc.dtype` 可以是 FP16、BF16 或 FP32，device 端数组始终使用
     * `float`。这与 HostTensor 的统一 FP32 表示一致，也使格式 kernel 与最终
     * 文件的物理类型转换解耦。
     *
     * @param output_desc 已知形状和目标物理输出类型的张量描述。
     * @return 持有精确数量 FP32 device 元素的 move-only 输出对象。
     * @throws std::invalid_argument 描述、输出 dtype 或元素数量不合法时抛出。
     * @throws std::overflow_error 元素数量超出当前平台 size_t 时抛出。
     * @throws thrust::system_error CUDA runtime 或 device allocation 失败时抛出。
     */
    [[nodiscard]] static DeviceDequantizationOutput allocate(
        const TensorDesc& output_desc) {
        const auto element_count = output_desc.elementCount();
        if (!output_desc.isValid() ||
            !is_supported_output_dtype(output_desc.dtype) ||
            !element_count.has_value()) {
            throw std::invalid_argument{
                "DeviceDequantizationOutput 需要合法的 FP16/BF16/FP32 输出描述。"};
        }

        DeviceDequantizationOutput output{};
        output.desc = output_desc;
        output.values = thrust::device_vector<float>{
            detail::checked_device_buffer_size(*element_count, "dequantized values")};
        return output;
    }

    /**
     * @brief 检查输出描述和 device FP32 数组的结构是否匹配。
     *
     * 本函数不会读取 device 数值，也不会同步 CUDA stream。
     *
     * @return 形状、目标输出 dtype 和 device 数组长度均正确时返回 true。
     */
    [[nodiscard]] bool isConsistent() const noexcept {
        const auto element_count = desc.elementCount();
        if (!desc.isValid() || !is_supported_output_dtype(desc.dtype) ||
            !element_count.has_value() ||
            *element_count >
                static_cast<std::uint64_t>(
                    std::numeric_limits<std::size_t>::max())) {
            return false;
        }

        return values.size() == static_cast<std::size_t>(*element_count);
    }
};

}  // namespace quant_dequant::pipeline
