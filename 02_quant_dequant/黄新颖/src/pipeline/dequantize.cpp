#include "quant_dequant/quantize.hpp"

#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include <cuda_runtime_api.h>

#include "cuda/mxfp8_dequantize.cuh"
#include "cuda/nvfp4_dequantize.cuh"
#include "common/cuda_stream.cuh"
#include "common/cuda_timer.cuh"
#include "pipeline/device_quantized_tensor.cuh"
#include "pipeline/pipeline_detail.hpp"

#include <thrust/device_ptr.h>

namespace quant_dequant {
namespace {

/**
 * @brief 验证 CUDA 反量化入口能够安全接收的 host 侧低精度张量。
 *
 * 量化文件读入后由 `QuantizedTensor` 保存其 payload、local scale 和可选 global
 * scale。实际实现会在通过该校验后将这些数组传输到 GPU；因此不能允许不自洽的
 * 描述或 buffer 进入格式专用路径。
 *
 * @param input 调用方传入的 host 侧量化张量。
 * @throws CudaPipelineError 张量不自洽时抛出。
 */
void validate_cuda_dequantization_input(const QuantizedTensor& input) {
    if (!input.isConsistent()) {
        throw CudaPipelineError{
            "CUDA pipeline 反量化要求自洽的 QuantizedTensor。"};
    }
}

/**
 * @brief 在指定 stream 上异步复制连续 host 数组到 device 数组。
 *
 * @tparam T 两端元素类型，必须具有相同的平凡二进制表示。
 * @param host_source 指向只读 host 数据的非拥有指针。
 * @param device_destination 指向已分配 device 数据的非拥有指针。
 * @param count 复制元素数。
 * @param stream 本次 H2D 所属 CUDA stream。
 * @param field_name 用于异常诊断的数组名称。
 * @throws CudaPipelineError CUDA runtime 复制失败或范围非法时抛出。
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
        device_destination, host_source, count * sizeof(T), cudaMemcpyHostToDevice, stream);
    if (status != cudaSuccess) {
        throw CudaPipelineError{
            "CUDA pipeline H2D 复制 \"" + std::string{field_name} + "\" 失败：" +
            cudaGetErrorString(status)};
    }
}

/**
 * @brief 在指定 stream 上异步复制连续 device 数组到 host 数组。
 *
 * @tparam T 两端元素类型，必须具有相同的平凡二进制表示。
 * @param device_source 指向只读 device 数据的非拥有指针。
 * @param host_destination 指向已分配 host 数据的非拥有指针。
 * @param count 复制元素数。
 * @param stream 本次 D2H 所属 CUDA stream。
 * @param field_name 用于异常诊断的数组名称。
 * @throws CudaPipelineError CUDA runtime 复制失败或范围非法时抛出。
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
        host_destination, device_source, count * sizeof(T), cudaMemcpyDeviceToHost, stream);
    if (status != cudaSuccess) {
        throw CudaPipelineError{
            "CUDA pipeline D2H 复制 \"" + std::string{field_name} + "\" 失败：" +
            cudaGetErrorString(status)};
    }
}

}  // namespace

HostTensor dequantize_cuda(
    const QuantizedTensor& input,
    const DequantizationConfig& config) {
    return dequantize_cuda_profiled(input, config).tensor;
}

ProfiledDequantizationResult dequantize_cuda_profiled(
    const QuantizedTensor& input,
    const DequantizationConfig& config) {
    validate_cuda_dequantization_input(input);

    if (!config.isValid()) {
        throw CudaPipelineError{"CUDA pipeline 反量化配置不合法。"};
    }

    switch (input.desc.format) {
        case QuantFormat::kMxfp8:
            return pipeline_detail::dequantize_mxfp8_cuda_profiled(input, config);
        case QuantFormat::kNvfp4:
            return pipeline_detail::dequantize_nvfp4_cuda_profiled(input, config);

        case QuantFormat::kUnknown:
            break;
    }

    throw CudaPipelineError{"CUDA pipeline 反量化格式不合法。"};
}

}  // namespace quant_dequant

namespace quant_dequant::pipeline_detail {
namespace {

/**
 * @brief 推导 CUDA 反量化后应返回的统一 host 输出描述。
 *
 * 低精度 QDWGT 记录的是量化前的 FP16/FP32 `source_desc`；反量化不改变矩阵
 * 形状，仅由方向配置指定最终 QDTENSOR 的物理输出类型。无论目标是 FP16、
 * BF16 还是 FP32，CUDA kernel 和 HostTensor 在内存中都先使用 FP32。
 *
 * @param input 已验证的 host 量化张量。
 * @param config 已验证的反量化方向配置。
 * @return 形状不变、dtype 为 config.output_type 的输出描述。
 * @throws CudaPipelineError 输出描述或元素数量不能安全表示时抛出。
 */
[[nodiscard]] TensorDesc make_cuda_dequantized_output_desc(
    const QuantizedTensor& input,
    const DequantizationConfig& config) {
    TensorDesc output_desc = input.desc.source_desc;
    output_desc.dtype = config.output_type;

    const auto element_count = output_desc.elementCount();
    if (!output_desc.isValid() ||
        !is_supported_output_dtype(output_desc.dtype) ||
        !element_count.has_value() ||
        *element_count >
            static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
        throw CudaPipelineError{
            "CUDA pipeline 无法构造合法的反量化输出描述。"};
    }

    return output_desc;
}

}  // namespace

HostTensor dequantize_mxfp8_cuda(
    const QuantizedTensor& input,
    const DequantizationConfig& config) {
    return dequantize_mxfp8_cuda_profiled(input, config).tensor;
}

ProfiledDequantizationResult dequantize_mxfp8_cuda_profiled(
    const QuantizedTensor& input,
    const DequantizationConfig& config) {
    // 阶段 0：public dispatcher 已完成同样的检查；格式专用入口仍保持自己的
    // 不变量，避免内部调用绕过 API 后把 NVFP4 或不合法的目标 dtype 传入。
    if (!input.isConsistent() || !config.isValid() ||
        input.desc.format != QuantFormat::kMxfp8) {
        throw CudaPipelineError{
            "mxfp8 CUDA pipeline 反量化收到了非法输入或输出配置。"};
    }

    const TensorDesc output_desc =
        make_cuda_dequantized_output_desc(input, config);

    try {
        common::CudaStream stream{};
        const cudaStream_t cuda_stream = stream.get();

        // 阶段 1：用与 quantize 输出相同的 device 所有权对象容纳 QDWGT 中的
        // E4M3 payload 与 E8M0 local scale。MXFP8 不使用 global_scale，
        // allocate() 会相应保持它为空；因此不能也无需从 host 拷贝该字段。
        pipeline::DeviceQuantizedTensor device_input =
            pipeline::DeviceQuantizedTensor::allocate(input.desc);
        copy_host_to_device_async(
            input.payload.data(), thrust::raw_pointer_cast(device_input.payload.data()),
            input.payload.size(), cuda_stream, "反量化 payload");
        copy_host_to_device_async(
            input.local_scales.data(),
            thrust::raw_pointer_cast(device_input.local_scales.data()),
            input.local_scales.size(), cuda_stream, "反量化 local_scales");
        if (!device_input.isConsistent()) {
            throw CudaPipelineError{
                "mxfp8 CUDA pipeline 构造出了不自洽的反量化 device 输入。"};
        }

        // 阶段 2：分配统一的 device FP32 输出。output_desc.dtype 仅标记将来
        // QDTENSOR 的物理 payload 类型；kernel 始终写 float，以使一个实现
        // 覆盖 FP16、BF16、FP32 三种输出请求。
        pipeline::DeviceDequantizationOutput device_output =
            pipeline::DeviceDequantizationOutput::allocate(output_desc);
        if (!device_output.isConsistent()) {
            throw CudaPipelineError{
                "mxfp8 CUDA pipeline 构造出了不自洽的反量化 device 输出。"};
        }

        // 阶段 3：一个线程解码一个 E4M3 payload；launcher 根据 input.desc 的
        // scale_mode 在 kernel 内选择唯一 tensor scale 或 rowwise block scale。
        // 因而 pipeline 不需要为了两种 scale 模式复制 H2D/D2H 流程。
        common::CudaEventTimer timer{cuda_stream};
        timer.start();
        cuda::launch_mxfp8_dequantize(device_input, device_output, cuda_stream);
        timer.stop();
        const double kernel_ms = static_cast<double>(timer.elapsedMilliseconds());

        // 阶段 4：把 D2H 提交给相同 stream，再等待此前 kernel 与复制都完成。
        // HostTensor 固定保存 FP32 values；真正写成配置请求的 FP16/BF16/FP32
        // 字节由 tensor_io 的 write_dequantized_tensor() 完成。
        HostTensor result{
            .desc = output_desc,
            .values = std::vector<float>(device_output.values.size()),
        };
        copy_device_to_host_async(
            thrust::raw_pointer_cast(device_output.values.data()), result.values.data(),
            result.values.size(), cuda_stream, "反量化 values");
        stream.synchronize();
        const auto expected_count = result.desc.elementCount();
        if (!expected_count.has_value() ||
            *expected_count >
                static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()) ||
            result.values.size() != static_cast<std::size_t>(*expected_count)) {
            throw CudaPipelineError{
                "mxfp8 CUDA pipeline D2H 后得到了长度不自洽的反量化结果。"};
        }

        return {
            .tensor = std::move(result),
            .kernel_ms = kernel_ms,
        };
    } catch (const CudaPipelineError&) {
        throw;
    } catch (const std::exception& error) {
        // Thrust 的 device allocation、H2D 或 D2H 通常以标准异常传播。统一转换
        // 为公共 pipeline 错误，调用方无需依赖 Thrust 的异常类型层次。
        throw CudaPipelineError{
            "mxfp8 CUDA pipeline 反量化执行 H2D、kernel 或 D2H 时失败：" +
            std::string{error.what()}};
    }
}

HostTensor dequantize_nvfp4_cuda(
    const QuantizedTensor& input,
    const DequantizationConfig& config) {
    return dequantize_nvfp4_cuda_profiled(input, config).tensor;
}

ProfiledDequantizationResult dequantize_nvfp4_cuda_profiled(
    const QuantizedTensor& input,
    const DequantizationConfig& config) {
    // 阶段 0：NVFP4 的 global_scale 是每次 decode 都必需的第三层 scale，不能
    // 像 MXFP8 一样只依赖 payload/local scale。QuantizedTensor::isConsistent()
    // 已保证它存在、有限且为正；这里再次固定格式与 block 规则，阻止内部调用
    // 绕过 public dispatcher 后落入错误的 packed-byte kernel。
    if (!input.isConsistent() || !config.isValid() ||
        input.desc.format != QuantFormat::kNvfp4 ||
        input.desc.scale_mode != ScaleMode::kBlock ||
        input.desc.block_size != kNvfp4BlockSize ||
        !input.global_scale.has_value()) {
        throw CudaPipelineError{
            "nvfp4 CUDA pipeline 反量化收到了非法输入或输出配置。"};
    }

    const TensorDesc output_desc =
        make_cuda_dequantized_output_desc(input, config);

    try {
        common::CudaStream stream{};
        const cudaStream_t cuda_stream = stream.get();

        // 阶段 1：把 QDWGT 的 packed E2M1 bytes、E4M3 local scale 和单元素
        // FP32 global scale 都交由同一个 DeviceQuantizedTensor 持有。`assign()`
        // 使用 host iterator 触发 H2D；对象将在 kernel 与 D2H 完成前持续存活。
        pipeline::DeviceQuantizedTensor device_input =
            pipeline::DeviceQuantizedTensor::allocate(input.desc);
        copy_host_to_device_async(
            input.payload.data(), thrust::raw_pointer_cast(device_input.payload.data()),
            input.payload.size(), cuda_stream, "反量化 payload");
        copy_host_to_device_async(
            input.local_scales.data(),
            thrust::raw_pointer_cast(device_input.local_scales.data()),
            input.local_scales.size(), cuda_stream, "反量化 local_scales");
        copy_host_to_device_async(
            &*input.global_scale,
            thrust::raw_pointer_cast(device_input.global_scale->data()),
            1U, cuda_stream, "反量化 global_scale");
        if (!device_input.isConsistent()) {
            throw CudaPipelineError{
                "nvfp4 CUDA pipeline 构造出了不自洽的反量化 device 输入。"};
        }

        // 阶段 2：与 MXFP8 统一，device kernel 仅写 FP32。output_desc.dtype
        // 只是后续 tensor I/O 写出 FP16/BF16/FP32 的物理格式请求。
        pipeline::DeviceDequantizationOutput device_output =
            pipeline::DeviceDequantizationOutput::allocate(output_desc);
        if (!device_output.isConsistent()) {
            throw CudaPipelineError{
                "nvfp4 CUDA pipeline 构造出了不自洽的反量化 device 输出。"};
        }

        // 阶段 3：一个线程解包一个物理 byte，并分别为 low/high nibble 推导
        // local scale；奇数列导致的跨行 byte 不会错误共用一个 scale 下标。
        common::CudaEventTimer timer{cuda_stream};
        timer.start();
        cuda::launch_nvfp4_dequantize(device_input, device_output, cuda_stream);
        timer.stop();
        const double kernel_ms = static_cast<double>(timer.elapsedMilliseconds());

        // 阶段 4：Thrust range 构造函数执行 D2H 并等待此前 kernel。统一的
        // HostTensor 继续保存 FP32 数值，物理窄化仍留在 QDTENSOR 写出边界。
        HostTensor result{
            .desc = output_desc,
            .values = std::vector<float>(device_output.values.size()),
        };
        copy_device_to_host_async(
            thrust::raw_pointer_cast(device_output.values.data()), result.values.data(),
            result.values.size(), cuda_stream, "反量化 values");
        stream.synchronize();
        const auto expected_count = result.desc.elementCount();
        if (!expected_count.has_value() ||
            *expected_count >
                static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()) ||
            result.values.size() != static_cast<std::size_t>(*expected_count)) {
            throw CudaPipelineError{
                "nvfp4 CUDA pipeline D2H 后得到了长度不自洽的反量化结果。"};
        }

        return {
            .tensor = std::move(result),
            .kernel_ms = kernel_ms,
        };
    } catch (const CudaPipelineError&) {
        throw;
    } catch (const std::exception& error) {
        throw CudaPipelineError{
            "nvfp4 CUDA pipeline 反量化执行 H2D、kernel 或 D2H 时失败：" +
            std::string{error.what()}};
    }
}

}  // namespace quant_dequant::pipeline_detail
