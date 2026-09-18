#include "quant_dequant/quantize.hpp"

#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <cuda_runtime_api.h>

#include "cuda/mxfp8_quantize.cuh"
#include "cuda/nvfp4_quantize.cuh"
#include "common/cuda_stream.cuh"
#include "common/cuda_timer.cuh"
#include "formats/mxfp8_codec.cuh"
#include "formats/nvfp4_codec.cuh"
#include "pipeline/device_quantized_tensor.cuh"
#include "pipeline/pipeline_detail.hpp"

#include <thrust/device_ptr.h>

namespace quant_dequant {
namespace {

/**
 * @brief 验证 CUDA 量化入口能够安全接收的统一 host 输入。
 *
 * `HostTensor` 在内存中始终使用 FP32 `values`，而 `desc.dtype` 仅保存输入文件
 * 的物理类型。因此即使原始输入是 FP16，也应当在进入 CUDA pipeline 前由
 * tensor I/O 层扩展为 FP32。
 *
 * @param input 调用方传入的 host 张量。
 * @throws CudaPipelineError 描述、元素数或输入物理类型不符合约定时抛出。
 */
void validate_cuda_quantization_input(const HostTensor& input) {
    if (!input.desc.isValid() ||
        (input.desc.dtype != DType::kFloat16 && input.desc.dtype != DType::kFloat32)) {
        throw CudaPipelineError{
            "CUDA pipeline 量化要求合法的 FP16/FP32 输入描述。"};
    }

    const auto expected_count = input.desc.elementCount();
    if (!expected_count.has_value() ||
        *expected_count > std::numeric_limits<std::size_t>::max() ||
        input.values.size() != static_cast<std::size_t>(*expected_count)) {
        throw CudaPipelineError{
            "CUDA pipeline 量化输入的 FP32 values 长度与张量描述不一致。"};
    }
}

/**
 * @brief 在指定 stream 上异步复制一段 host 数组到 device 数组。
 *
 * @tparam T 两端元素类型，必须具有相同的平凡二进制表示。
 * @param host_source 指向只读连续 host 数组的非拥有指针。
 * @param device_destination 指向已分配连续 device 数组的非拥有指针。
 * @param count 要复制的元素数量。
 * @param stream 承担本次 H2D 的 non-blocking CUDA stream。
 * @param field_name 用于诊断的数组名称。
 * @throws CudaPipelineError 数量、地址或 cudaMemcpyAsync 调用不合法时抛出。
 */
template <typename T>
void copy_host_to_device_async(const T* const host_source,
                               T* const device_destination,
                               const std::size_t count,
                               const cudaStream_t stream,
                               const char* const field_name) {
    if (host_source == nullptr || device_destination == nullptr || count == 0U ||
        count > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
        throw CudaPipelineError{
            "CUDA pipeline 无法构造 \"" + std::string{field_name} +
            "\" 的 H2D 复制范围。"};
    }

    const cudaError_t status = cudaMemcpyAsync(
        device_destination,
        host_source,
        count * sizeof(T),
        cudaMemcpyHostToDevice,
        stream);
    if (status != cudaSuccess) {
        throw CudaPipelineError{
            "CUDA pipeline H2D 复制 \"" + std::string{field_name} + "\" 失败：" +
            cudaGetErrorString(status)};
    }
}

/**
 * @brief 在指定 stream 上异步复制一段 device 数组到 host 数组。
 *
 * @tparam T 两端元素类型，必须具有相同的平凡二进制表示。
 * @param device_source 指向只读连续 device 数组的非拥有指针。
 * @param host_destination 指向已分配连续 host 数组的非拥有指针。
 * @param count 要复制的元素数量。
 * @param stream 承担本次 D2H 的 non-blocking CUDA stream。
 * @param field_name 用于诊断的数组名称。
 * @throws CudaPipelineError 数量、地址或 cudaMemcpyAsync 调用不合法时抛出。
 */
template <typename T>
void copy_device_to_host_async(const T* const device_source,
                               T* const host_destination,
                               const std::size_t count,
                               const cudaStream_t stream,
                               const char* const field_name) {
    if (device_source == nullptr || host_destination == nullptr || count == 0U ||
        count > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
        throw CudaPipelineError{
            "CUDA pipeline 无法构造 \"" + std::string{field_name} +
            "\" 的 D2H 复制范围。"};
    }

    const cudaError_t status = cudaMemcpyAsync(
        host_destination,
        device_source,
        count * sizeof(T),
        cudaMemcpyDeviceToHost,
        stream);
    if (status != cudaSuccess) {
        throw CudaPipelineError{
            "CUDA pipeline D2H 复制 \"" + std::string{field_name} + "\" 失败：" +
            cudaGetErrorString(status)};
    }
}

}  // namespace

CudaPipelineError::CudaPipelineError(std::string detail)
    : std::runtime_error(std::move(detail)) {}

QuantizedTensor quantize_cuda(
    const HostTensor& input,
    const QuantizationConfig& config) {
    return quantize_cuda_profiled(input, config).tensor;
}

ProfiledQuantizationResult quantize_cuda_profiled(
    const HostTensor& input,
    const QuantizationConfig& config) {
    validate_cuda_quantization_input(input);

    if (!config.isValid()) {
        throw CudaPipelineError{"CUDA pipeline 量化配置不合法。"};
    }

    switch (config.format) {
        case QuantFormat::kMxfp8:
            return pipeline_detail::quantize_mxfp8_cuda_profiled(input, config);
        case QuantFormat::kNvfp4:
            return pipeline_detail::quantize_nvfp4_cuda_profiled(input, config);

        case QuantFormat::kUnknown:
            break;
    }

    throw CudaPipelineError{"CUDA pipeline 量化格式不合法。"};
}

}  // namespace quant_dequant

namespace quant_dequant::pipeline_detail {
namespace {

/**
 * @brief 验证 MXFP8 CUDA 专用入口可安全使用的 host 输入。
 *
 * 公共 `quantize_cuda()` 已执行相同的检查。这里仍重复验证，使内部调用者无法
 * 绕过公共 dispatcher 后把不完整的 `HostTensor` 传给未来的 H2D 路径。
 *
 * @param input 待量化的统一 host FP32 张量。
 * @throws CudaPipelineError 描述、物理输入类型或 FP32 元素数量不符合约定时抛出。
 */
void validate_mxfp8_cuda_input(const HostTensor& input) {
    if (!input.desc.isValid() || !is_supported_input_dtype(input.desc.dtype)) {
        throw CudaPipelineError{
            "MXFP8 CUDA pipeline 收到了不合法的 FP16/FP32 输入描述。"};
    }

    const auto element_count = input.desc.elementCount();
    if (!element_count.has_value() ||
        *element_count > std::numeric_limits<std::size_t>::max() ||
        input.values.size() != static_cast<std::size_t>(*element_count)) {
        throw CudaPipelineError{
            "MXFP8 CUDA pipeline 输入的 FP32 values 长度与张量描述不一致。"};
    }
}

/**
 * @brief 验证 MXFP8 格式专用入口的量化配置。
 *
 * 虽然 `QuantizationConfig::isValid()` 已经通过格式映射间接检查 block size，
 * 这里显式比较 `kMxfp8BlockSize`，将未来 warp-per-quantization-block kernel
 * 的固定 32 元素前提固定在该格式入口，而非仅隐含在通用配置校验中。
 *
 * @param config 待验证的量化方向配置。
 * @throws CudaPipelineError 配置不是合法 MXFP8 配置，或 block size 不为 32 时抛出。
 */
void validate_mxfp8_cuda_config(const QuantizationConfig& config) {
    if (!config.isValid() || config.format != QuantFormat::kMxfp8 ||
        config.block_size != kMxfp8BlockSize) {
        throw CudaPipelineError{
            "MXFP8 CUDA pipeline 收到了非法量化配置；"
            "该格式要求 format = mxfp8 且 block_size = 32。"};
    }
}

/**
 * @brief 验证未来 MXFP8 CUDA 输出所需的元数据和 buffer 长度。
 *
 * 该函数只推导 metadata，不分配 `thrust::device_vector`。它提前确保后续
 * `DeviceQuantizedTensor::allocate()` 可以得到一个 E4M3 一字节一元素的 payload
 * 和按 tensor/block 模式组织的 E8M0 local scale 数组。
 *
 * @param input 已通过输入校验的 host 张量。
 * @param config 已通过 MXFP8 配置校验的量化配置。
 * @return 供 `DeviceQuantizedTensor::allocate()` 使用的 MXFP8 输出描述。
 * @throws CudaPipelineError 元数据不可推导，或数组长度超出 host/device 容器可表示
 * 范围时抛出。
 */
[[nodiscard]] QuantizedTensorDesc make_mxfp8_cuda_output_desc(
    const HostTensor& input,
    const QuantizationConfig& config) {
    QuantizedTensorDesc output_desc{};
    output_desc.source_desc = input.desc;
    output_desc.format = QuantFormat::kMxfp8;
    output_desc.scale_mode = config.scale_mode;
    output_desc.rounding = config.rounding;
    output_desc.stochastic_seed = config.stochastic_seed;
    output_desc.block_size = config.block_size;
    output_desc.scale_layout = ScaleLayout::kRowwise;

    const auto payload_bytes = output_desc.expectedPayloadBytes();
    const auto local_scale_count = output_desc.expectedLocalScaleCount();
    if (!output_desc.isMetadataValid() || !payload_bytes.has_value() ||
        !local_scale_count.has_value()) {
        throw CudaPipelineError{
            "MXFP8 CUDA pipeline 无法构造合法的量化输出元数据。"};
    }

    constexpr std::uint64_t kMaxContainerSize =
        static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max());
    if (*payload_bytes > kMaxContainerSize ||
        *local_scale_count > kMaxContainerSize) {
        throw CudaPipelineError{
            "MXFP8 CUDA pipeline 的 payload 或 local scale 数量超出"
            "当前 host/device 容器可表示范围。"};
    }

    return output_desc;
}

/**
 * @brief 验证 NVFP4 CUDA 专用入口的 host 输入。
 *
 * @param input 待量化的统一 host FP32 张量。
 * @throws CudaPipelineError 描述、输入 dtype 或 FP32 values 长度不合法时抛出。
 */
void validate_nvfp4_cuda_input(const HostTensor& input) {
    if (!input.desc.isValid() || !is_supported_input_dtype(input.desc.dtype)) {
        throw CudaPipelineError{
            "NVFP4 CUDA pipeline 收到了不合法的 FP16/FP32 输入描述。"};
    }

    const auto element_count = input.desc.elementCount();
    if (!element_count.has_value() ||
        *element_count > std::numeric_limits<std::size_t>::max() ||
        input.values.size() != static_cast<std::size_t>(*element_count)) {
        throw CudaPipelineError{
            "NVFP4 CUDA pipeline 输入的 FP32 values 长度与张量描述不一致。"};
    }
}

/**
 * @brief 验证严格 NVFP4 rowwise 16 元素 block 量化配置。
 *
 * @param config 待验证的量化方向配置。
 * @throws CudaPipelineError 格式、block size 或 scale_mode 不符合 NVFP4 规则时抛出。
 */
void validate_nvfp4_cuda_config(const QuantizationConfig& config) {
    if (!config.isValid() || config.format != QuantFormat::kNvfp4 ||
        config.block_size != kNvfp4BlockSize ||
        config.scale_mode != ScaleMode::kBlock) {
        throw CudaPipelineError{
            "NVFP4 CUDA pipeline 要求 format = nvfp4、block_size = 16 且 "
            "scale_mode = block。"};
    }
}

/**
 * @brief 构造 NVFP4 CUDA 输出的共享 QuantizedTensorDesc。
 *
 * @param input 已通过输入校验的 host 张量。
 * @param config 已通过 NVFP4 专用校验的量化配置。
 * @return 可用于分配 packed payload、E4M3 local scale 与 global scalar 的描述。
 * @throws CudaPipelineError 元数据或容器长度无法安全推导时抛出。
 */
[[nodiscard]] QuantizedTensorDesc make_nvfp4_cuda_output_desc(
    const HostTensor& input,
    const QuantizationConfig& config) {
    QuantizedTensorDesc output_desc{};
    output_desc.source_desc = input.desc;
    output_desc.format = QuantFormat::kNvfp4;
    output_desc.scale_mode = ScaleMode::kBlock;
    output_desc.rounding = config.rounding;
    output_desc.stochastic_seed = config.stochastic_seed;
    output_desc.block_size = kNvfp4BlockSize;
    output_desc.scale_layout = ScaleLayout::kRowwise;

    const auto payload_bytes = output_desc.expectedPayloadBytes();
    const auto local_scale_count = output_desc.expectedLocalScaleCount();
    if (!output_desc.isMetadataValid() || !payload_bytes.has_value() ||
        !local_scale_count.has_value()) {
        throw CudaPipelineError{
            "NVFP4 CUDA pipeline 无法构造合法的量化输出元数据。"};
    }

    constexpr std::uint64_t kMaxContainerSize =
        static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max());
    if (*payload_bytes > kMaxContainerSize ||
        *local_scale_count > kMaxContainerSize) {
        throw CudaPipelineError{
            "NVFP4 CUDA pipeline 的 packed payload 或 local scale 数量超出"
            "当前 host/device 容器可表示范围。"};
    }

    return output_desc;
}

}  // namespace

QuantizedTensor quantize_mxfp8_cuda(
    const HostTensor& input,
    const QuantizationConfig& config) {
    return quantize_mxfp8_cuda_profiled(input, config).tensor;
}

ProfiledQuantizationResult quantize_mxfp8_cuda_profiled(
    const HostTensor& input,
    const QuantizationConfig& config) {
    // 阶段 0：即使经由内部入口调用，也保持与 CPU reference 相同的“先验证、
    // 再计算”约束。这样未来接入 H2D 与 kernel 后不会因错误描述或错误配置
    // 留下半成品 device buffer。
    validate_mxfp8_cuda_input(input);
    validate_mxfp8_cuda_config(config);

    // 阶段 1：根据输入描述和方向配置验证未来输出的精确 layout。MXFP8 的
    // block_size 在上一步已固定为 32，因此 block mode 下每个逻辑量化 block
    // 都能稳定映射为一个 warp 的 scale 工作单元。
    const QuantizedTensorDesc output_desc =
        make_mxfp8_cuda_output_desc(input, config);

    try {
        // 每次 profile 调用独占一条 non-blocking stream。所有 H2D、event、kernel
        // 和 D2H 都显式提交到它，避免默认 stream 与 Thrust 隐式绑定导致计时区间
        // 覆盖其他任务，或发生不必要的跨 stream 同步。
        common::CudaStream stream{};
        const cudaStream_t cuda_stream = stream.get();

        // 阶段 2：QDTENSOR I/O 已把输入统一为 host FP32；device_vector 只负责
        // 取得 device 所有权，随后 cudaMemcpyAsync 显式把 H2D 提交到本调用的
        // stream。原始 FP16/FP32 物理 dtype 仍保留在 desc 中，以便最终 QDWGT
        // 正确记录 source_dtype。
        pipeline::DeviceQuantizationInput device_input{};
        device_input.desc = input.desc;
        device_input.values.resize(input.values.size());
        copy_host_to_device_async(
            input.values.data(),
            thrust::raw_pointer_cast(device_input.values.data()),
            input.values.size(), cuda_stream, "量化输入 values");
        if (!device_input.isConsistent()) {
            throw CudaPipelineError{
                "mxfp8 CUDA pipeline 构造出了不自洽的 device 输入。"};
        }

        // 阶段 3：按已验证的精确字节数分配 E4M3 payload 和 E8M0 local scale。
        // MXFP8 不含 NVFP4 的 global_scale；allocate() 会将其保持为空。
        pipeline::DeviceQuantizedTensor device_output =
            pipeline::DeviceQuantizedTensor::allocate(output_desc);
        if (!device_output.isConsistent()) {
            throw CudaPipelineError{
                "mxfp8 CUDA pipeline 构造出了不自洽的 device 输出。"};
        }

        // 阶段 4：block mode 能在一个 warp 内完成 amax、E8M0 scale 和 E4M3
        // 编码；tensor mode 则需要跨 CTA 的全局 amax reduction。二者共享同样的
        // 输入/输出 device 所有权对象，但 tensor mode 额外需要一次调用期间始终
        // 存活的 partial 规约工作区，因此必须进入不同的 CUDA launcher。
        std::optional<pipeline::DeviceTensorQuantizationWorkspace>
            tensor_workspace{};
        // start event 排在 H2D 后面，因此后续 elapsed time 不包含输入传输；同一
        // stream 的顺序语义保证 event 一定在 copy 完成后才被记录。
        common::CudaEventTimer timer{cuda_stream};
        timer.start();
        switch (config.scale_mode) {
            case ScaleMode::kBlock:
                cuda::launch_mxfp8_block_quantize(
                    device_input, config, device_output, cuda_stream);
                break;

            case ScaleMode::kTensor:
                tensor_workspace.emplace(
                    pipeline::DeviceTensorQuantizationWorkspace::allocate(
                        cuda::mxfp8_tensor_partial_count()));
                cuda::launch_mxfp8_tensor_quantize(
                    device_input, config, device_output, *tensor_workspace, cuda_stream);
                break;

            case ScaleMode::kUnknown:
                throw CudaPipelineError{
                    "mxfp8 CUDA pipeline 收到了未知的 scale_mode。"};
        }
        timer.stop();
        const double kernel_ms = static_cast<double>(timer.elapsedMilliseconds());

        // 阶段 5：把 D2H 显式提交给同一条 stream，并在读取 host 数据前同步该
        // stream。先回传 scale：kernel 用 E8M0 的 0xff 标记包含 NaN/Inf 的逻辑
        // block；一旦发现该哨兵，就与 CPU reference 一样拒绝整个量化请求，避免
        // 把无数值意义的 payload 暴露出去。
        std::vector<std::uint8_t> host_local_scales(
            device_output.local_scales.size());
        copy_device_to_host_async(
            thrust::raw_pointer_cast(device_output.local_scales.data()),
            host_local_scales.data(), host_local_scales.size(), cuda_stream,
            "量化 local_scales");
        stream.synchronize();
        for (std::size_t block_index = 0;
             block_index < host_local_scales.size();
             ++block_index) {
            if (formats::is_e8m0_nan(host_local_scales[block_index])) {
                throw CudaPipelineError{
                    "mxfp8 CUDA 量化不接受 NaN 或 Inf 输入；发现非法 "
                    "E8M0 scale 的量化 block 为 " +
                    std::to_string(block_index) + "。"};
            }
        }

        // 所有 local scale 合法后，再回传同样由 kernel 写入的 E4M3 payload。
        // QuantizedTensor 是 CPU/GPU 共用的持久化结果类型；此处不额外转换编码，
        // 因而 QDWGT writer 可直接写出这些真实字节。
        QuantizedTensor result{
            .desc = output_desc,
            .payload = std::vector<std::uint8_t>(device_output.payload.size()),
            .local_scales = std::move(host_local_scales),
            .global_scale = std::nullopt,
        };
        copy_device_to_host_async(
            thrust::raw_pointer_cast(device_output.payload.data()),
            result.payload.data(), result.payload.size(), cuda_stream,
            "量化 payload");
        stream.synchronize();
        if (!result.isConsistent()) {
            throw CudaPipelineError{
                "mxfp8 CUDA pipeline D2H 后得到了不自洽的量化结果。"};
        }

        return {
            .tensor = std::move(result),
            .kernel_ms = kernel_ms,
        };
    } catch (const CudaPipelineError&) {
        throw;
    } catch (const std::exception& error) {
        // thrust 的 H2D/分配失败以 system_error 等标准异常报告。统一转换为公共
        // CUDA pipeline 错误，避免调用方必须了解 Thrust 的异常层级。D2H 的
        // device iterator 回传也遵循同一异常转换边界。
        throw CudaPipelineError{
            "mxfp8 CUDA pipeline 构造 device buffer 或执行 H2D/D2H 时失败：" +
            std::string{error.what()}};
    }

}

QuantizedTensor quantize_nvfp4_cuda(
    const HostTensor& input,
    const QuantizationConfig& config) {
    return quantize_nvfp4_cuda_profiled(input, config).tensor;
}

ProfiledQuantizationResult quantize_nvfp4_cuda_profiled(
    const HostTensor& input,
    const QuantizationConfig& config) {
    // 阶段 0：NVFP4 只能是 block mode，但同时必有整张张量的 FP32 global
    // scale。先固定这两个看似相反、实际分属不同层次的约束。
    validate_nvfp4_cuda_input(input);
    validate_nvfp4_cuda_config(config);
    const QuantizedTensorDesc output_desc =
        make_nvfp4_cuda_output_desc(input, config);

    try {
        common::CudaStream stream{};
        const cudaStream_t cuda_stream = stream.get();

        // 阶段 1：原始 QDTENSOR 的 FP16/FP32 已在 I/O 边界统一扩展为 host
        // FP32；从迭代器构造 device_vector 即执行一次 H2D。
        pipeline::DeviceQuantizationInput device_input{};
        device_input.desc = input.desc;
        device_input.values.resize(input.values.size());
        copy_host_to_device_async(
            input.values.data(),
            thrust::raw_pointer_cast(device_input.values.data()),
            input.values.size(), cuda_stream, "量化输入 values");
        if (!device_input.isConsistent()) {
            throw CudaPipelineError{
                "nvfp4 CUDA pipeline 构造出了不自洽的 device 输入。"};
        }

        // 阶段 2：allocate() 按 QuantizedTensorDesc 精确分配 ceil(N/2) 个
        // payload byte、每个 16 元素 block 一个 E4M3 scale，以及一个 FP32
        // global_scale device scalar。
        pipeline::DeviceQuantizedTensor device_output =
            pipeline::DeviceQuantizedTensor::allocate(output_desc);
        if (!device_output.isConsistent() || !device_output.global_scale.has_value()) {
            throw CudaPipelineError{
                "nvfp4 CUDA pipeline 构造出了不自洽的 device 输出。"};
        }

        // 阶段 3：NVFP4 即使配置为 block，也不能像 MXFP8 block mode 那样在
        // 单 kernel 内完成全部量化，因为每个 local scale 还依赖 tensor-wide
        // global scale。工作区必须覆盖三阶段 kernel 及后续 D2H。
        pipeline::DeviceTensorQuantizationWorkspace workspace =
            pipeline::DeviceTensorQuantizationWorkspace::allocate(
                cuda::nvfp4_tensor_partial_count());
        common::CudaEventTimer timer{cuda_stream};
        timer.start();
        cuda::launch_nvfp4_block_quantize(
            device_input, config, device_output, workspace, cuda_stream);
        timer.stop();
        const double kernel_ms = static_cast<double>(timer.elapsedMilliseconds());

        // 阶段 4：首先 D2H 唯一的 global scalar。NaN/Inf 输入会在全局规约
        // finalize kernel 中写 canonical NaN；必须在复制 payload 前拒绝。
        std::vector<float> host_global_scale(device_output.global_scale->size());
        copy_device_to_host_async(
            thrust::raw_pointer_cast(device_output.global_scale->data()),
            host_global_scale.data(), host_global_scale.size(), cuda_stream,
            "量化 global_scale");
        stream.synchronize();
        if (host_global_scale.size() != 1U ||
            !formats::fp32::is_finite(host_global_scale.front()) ||
            host_global_scale.front() <= 0.0F) {
            throw CudaPipelineError{
                "nvfp4 CUDA 量化不接受 NaN 或 Inf 输入，或生成了非法的 "
                "FP32 global_scale。"};
        }

        std::vector<std::uint8_t> host_local_scales(
            device_output.local_scales.size());
        copy_device_to_host_async(
            thrust::raw_pointer_cast(device_output.local_scales.data()),
            host_local_scales.data(), host_local_scales.size(), cuda_stream,
            "量化 local_scales");
        stream.synchronize();
        for (std::size_t block_index = 0U;
             block_index < host_local_scales.size();
             ++block_index) {
            if (formats::is_e4m3_nan(host_local_scales[block_index])) {
                throw CudaPipelineError{
                    "nvfp4 CUDA 量化不接受 NaN 或 Inf 输入；发现非法 "
                    "E4M3 local scale 的量化 block 为 " +
                    std::to_string(block_index) + "。"};
            }
        }

        // 阶段 5：所有 scale 均已通过 D2H 校验后才回传 packed payload。每个
        // 字节已经是两个 E2M1 code 的最终磁盘布局，无须 CPU 再打包。
        QuantizedTensor result{
            .desc = output_desc,
            .payload = std::vector<std::uint8_t>(device_output.payload.size()),
            .local_scales = std::move(host_local_scales),
            .global_scale = host_global_scale.front(),
        };
        copy_device_to_host_async(
            thrust::raw_pointer_cast(device_output.payload.data()),
            result.payload.data(), result.payload.size(), cuda_stream,
            "量化 payload");
        stream.synchronize();
        if (!result.isConsistent()) {
            throw CudaPipelineError{
                "nvfp4 CUDA pipeline D2H 后得到了不自洽的量化结果。"};
        }

        return {
            .tensor = std::move(result),
            .kernel_ms = kernel_ms,
        };
    } catch (const CudaPipelineError&) {
        throw;
    } catch (const std::exception& error) {
        throw CudaPipelineError{
            "nvfp4 CUDA pipeline 构造 device buffer 或执行 H2D/D2H 时失败：" +
            std::string{error.what()}};
    }
}

}  // namespace quant_dequant::pipeline_detail
