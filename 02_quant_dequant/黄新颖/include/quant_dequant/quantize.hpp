#pragma once

#include <stdexcept>
#include <string>

#include "quant_dequant/quantized_tensor.hpp"
#include "quant_dequant/tensor_io.hpp"
#include "quant_dequant/types.hpp"

namespace quant_dequant {

/**
 * @brief 表示 CPU reference 量化或反量化流程无法继续时的错误。
 *
 * 该异常用于参数不自洽、量化结果不自洽，以及某个格式的 reference 后端尚未
 * 实现等情况。它不表示 CUDA runtime 错误；CUDA pipeline 的接口使用独立的
 * `CudaPipelineError`。
 */
class ReferenceError final : public std::runtime_error {
public:
    /**
     * @brief 构造带有具体诊断信息的 CPU reference 错误。
     *
     * @param detail 面向调用方的错误说明。
     */
    explicit ReferenceError(std::string detail);
};

/**
 * @brief 使用 CPU reference 量化一个 host FP32 张量。
 *
 * 这是正确性基准接口，不发射 CUDA kernel，也不接触 Thrust device buffer。
 * 函数根据 `config.format` 转发到格式专用 reference 实现。MXFP8 支持
 * tensor/block scaling；NVFP4 支持严格的 block scaling、E4M3 local scale、
 * FP32 global scale 和真实 nibble 打包。
 *
 * @param input 输入 host 张量；`values` 必须是 row-major FP32 数据，`desc.dtype`
 * 必须记录原输入的 FP16 或 FP32 物理类型。
 * @param config 仅包含量化方向数值语义的配置。
 * @return 量化完成后的 host `QuantizedTensor`。
 * @throws ReferenceError 输入、配置非法或目标格式的 CPU reference 尚未实现时抛出。
 */
[[nodiscard]] QuantizedTensor quantize_reference(
    const HostTensor& input,
    const QuantizationConfig& config);

/**
 * @brief 使用 CPU reference 反量化一个 host 侧低精度张量。
 *
 * 函数根据 `input.desc.format` 转发到格式专用 reference 实现。返回的
 * `HostTensor::values` 始终是 FP32；`HostTensor::desc.dtype` 则设置为
 * `config.output_type`，供随后写 QDTENSOR 时选择 FP16、BF16 或 FP32 payload。
 *
 * @param input 已通过 `QuantizedTensor::isConsistent()` 校验的量化结果。
 * @param config 仅包含反量化输出类型的配置。
 * @return 反量化后的 host FP32 张量及目标物理输出类型。
 * @throws ReferenceError 量化结果、配置非法或目标格式的 CPU reference 尚未实现时抛出。
 */
[[nodiscard]] HostTensor dequantize_reference(
    const QuantizedTensor& input,
    const DequantizationConfig& config);

/**
 * @brief 表示 CUDA pipeline 在启动格式专用实现前无法继续时的错误。
 *
 * 该异常用于 host 输入、量化描述或配置不自洽，以及某个 CUDA 格式路径尚未
 * 实现的情况。它与 `ReferenceError` 分开，使调用方能够区分 CPU reference 和
 * CUDA pipeline 后端的失败。未来 CUDA runtime API 与 kernel 的实际错误也会
 * 归一到这一异常类型，并保留底层诊断信息。
 */
class CudaPipelineError final : public std::runtime_error {
public:
    /**
     * @brief 构造带有具体诊断信息的 CUDA pipeline 错误。
     *
     * @param detail 面向调用方的错误说明。
     */
    explicit CudaPipelineError(std::string detail);
};

/**
 * @brief CUDA 量化的持久化结果及仅 kernel 区间的 Event 计时。
 *
 * `kernel_ms` 从同一条非阻塞 stream 上的 start/stop event 得到，不包含 host-device
 * 拷贝、Thrust 分配、D2H、文件 I/O、CUDA context 初始化或 CPU 元数据校验。
 */
struct ProfiledQuantizationResult {
    /** 已完成 D2H、可直接传给 QDWGT 写入器的量化结果。 */
    QuantizedTensor tensor{};

    /** 当前调用中所有量化 kernel 的合计时间，单位毫秒。 */
    double kernel_ms{0.0};
};

/**
 * @brief CUDA 反量化的 host 结果及仅 kernel 区间的 Event 计时。
 */
struct ProfiledDequantizationResult {
    /** 已完成 D2H 的统一 FP32 HostTensor；desc 记录目标物理输出类型。 */
    HostTensor tensor{};

    /** 当前调用中所有反量化 kernel 的合计时间，单位毫秒。 */
    double kernel_ms{0.0};
};

/**
 * @brief 使用 CUDA pipeline 量化一个 host 侧输入张量。
 *
 * 接口与 `quantize_reference()` 一样接收 `HostTensor`，避免调用方在 CPU 与 GPU
 * 后端之间维护两套输入类型。MXFP8 block-scale 路径在一个 warp 内处理一个
 * 32 元素量化 block；tensor-scale 路径使用 V7 风格的固定 CTA `amax` partial
 * reduction、单 CTA final reduction/scale 写入和独立编码 kernel。两条路径均已
 * 完成 H2D、device 所有权对象建立、kernel 发射和 D2H，返回值可直接交给
 * `write_quantized_tensor()` 序列化。
 *
 * NVFP4 路径先做全局两阶段 amax reduction 写 FP32 global scale，再以 16-lane
 * tile 写 E4M3 local scale，最后以 32-lane warp 写 packed E2M1 payload。两种
 * 格式的输入若含 NaN/Inf，公共接口都会拒绝该请求。
 *
 * @param input 输入 host 张量；`values` 必须是 row-major FP32 数据，`desc.dtype`
 * 必须记录原输入的 FP16 或 FP32 物理类型。
 * @param config 仅包含量化方向数值语义的配置。
 * @return 已完成的 host `QuantizedTensor`；支持 MXFP8 tensor/block 与 NVFP4 block。
 * @throws CudaPipelineError 输入、配置非法、CUDA runtime/kernel/D2H 失败、输入含
 * NaN/Inf 时抛出。
 */
[[nodiscard]] QuantizedTensor quantize_cuda(
    const HostTensor& input,
    const QuantizationConfig& config);

/**
 * @brief 使用独占 CUDA stream 执行量化并返回纯 kernel Event 计时。
 *
 * H2D 在 start event 前提交；格式专用路径的全部 kernel 都提交到同一条 stream；
 * stop event 紧跟最后一个 kernel，随后才执行 D2H。因此本接口适合 app 报告和
 * benchmark，不会把内存传输、分配或文件操作误计为 kernel 时间。
 *
 * @param input 输入 host FP32 张量，描述 dtype 为原始 FP16/FP32 物理类型。
 * @param config 合法的 MXFP8 或 NVFP4 量化配置。
 * @return 量化结果和该格式路径所有 kernel 的合计毫秒数。
 * @throws CudaPipelineError 输入、配置、stream、CUDA runtime、kernel 或 D2H 失败时抛出。
 */
[[nodiscard]] ProfiledQuantizationResult quantize_cuda_profiled(
    const HostTensor& input,
    const QuantizationConfig& config);

/**
 * @brief 使用 CUDA pipeline 反量化一个 host 侧低精度张量。
 *
 * MXFP8 路径会把 `QuantizedTensor` 的 E4M3 payload 与 E8M0 local scale 传输到
 * device，按 `input.desc.scale_mode` 计算 scale 下标；NVFP4 则传输 packed E2M1
 * payload、E4M3 local scale 与 FP32 global scale，由一个线程解包一个物理 byte。
 * 两条路径均发射格式专用反量化 kernel，并将结果回传为 FP32 `HostTensor::values`。
 * 返回描述中的 `dtype` 设为 `config.output_type`，由张量 I/O 层决定写出 FP16、
 * BF16 或 FP32 的物理 payload。
 *
 * 支持 MXFP8 的 tensor/block-scale，以及 NVFP4 的严格 rowwise block-scale。
 *
 * @param input 已通过 `QuantizedTensor::isConsistent()` 校验的量化结果。
 * @param config 仅包含反量化输出类型的配置。
 * @return 已完成的 host FP32 张量及目标物理输出类型。
 * @throws CudaPipelineError 量化结果、配置非法、CUDA runtime/kernel/D2H 失败，
 * 时抛出。
 */
[[nodiscard]] HostTensor dequantize_cuda(
    const QuantizedTensor& input,
    const DequantizationConfig& config);

/**
 * @brief 使用独占 CUDA stream 执行反量化并返回纯 kernel Event 计时。
 *
 * QDWGT buffer 的 H2D 完成后才记录 start event；格式专用反量化 kernel 结束后立刻
 * 记录 stop event；FP32 输出的 D2H 位于计时区间之外。
 *
 * @param input 已验证的 MXFP8 或 NVFP4 host 量化张量。
 * @param config 合法的反量化输出类型配置。
 * @return 反量化 HostTensor 和 kernel 毫秒数。
 * @throws CudaPipelineError 输入、配置、stream、CUDA runtime、kernel 或 D2H 失败时抛出。
 */
[[nodiscard]] ProfiledDequantizationResult dequantize_cuda_profiled(
    const QuantizedTensor& input,
    const DequantizationConfig& config);

}  // namespace quant_dequant
