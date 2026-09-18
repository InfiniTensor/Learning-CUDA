#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>

#include "quant_dequant/quantized_tensor.hpp"
#include "quant_dequant/tensor_io.hpp"
#include "quant_dequant/types.hpp"

namespace quant_dequant {

/** @brief 表示指标计算或运行报告构造/写入无法继续时的错误。 */
class MetricsError final : public std::runtime_error {
public:
    /**
     * @brief 构造带有面向调用方诊断信息的指标错误。
     *
     * @param detail 错误的具体原因。
     */
    explicit MetricsError(std::string detail);
};

/** @brief 原始 FP32 参考值和最终反量化值之间的误差统计。 */
struct ErrorMetrics {
    /** 全部元素绝对误差中的最大值。 */
    double max_abs{0.0};

    /** 全部元素绝对误差的算术平均值。 */
    double mae{0.0};

    /** 全部元素平方误差的算术平均值。 */
    double mse{0.0};
};

/** @brief 写入产物和逻辑量化布局的字节数。 */
struct ArtifactMetrics {
    /** 原始 QDTENSOR 数值 payload 的字节数，不包含 64-byte header。 */
    std::uint64_t input_payload_bytes{0U};

    /** 实际写出的完整 QDWGT 文件字节数，包含 header 和 section 对齐 padding。 */
    std::uint64_t quantized_file_bytes{0U};

    /** 实际写出的完整反量化 QDTENSOR 文件字节数，包含 header。 */
    std::uint64_t dequantized_file_bytes{0U};

    /** 低精度 payload 的真实物理字节数；NVFP4 已按两个 nibble 每字节计算。 */
    std::uint64_t payload_bytes{0U};

    /** local-scale 数组的物理字节数。当前 MXFP8/NVFP4 都是一项一字节。 */
    std::uint64_t local_scale_bytes{0U};

    /** NVFP4 为 4，MXFP8 为 0 的 global-scale 逻辑字节数。 */
    std::uint64_t global_scale_bytes{0U};
};

/** @brief 低精度逻辑布局和实际 QDWGT 文件得到的压缩率。 */
struct CompressionMetrics {
    /** payload、local scale 与 NVFP4 global scale 的逻辑总字节数。 */
    std::uint64_t logical_quantized_bytes{0U};

    /** input payload / logical quantized bytes，不包含任何文件 header 或 padding。 */
    double logical_compression_ratio{0.0};

    /** input payload / complete QDWGT file bytes，反映实际落盘空间占用。 */
    double on_disk_compression_ratio{0.0};
};

/** @brief CUDA kernel 计时及按项目逻辑读写量定义的有效带宽。 */
struct KernelPerformance {
    /** 量化阶段所有 kernel 的合计时间；CPU/reference 路径为 null。 */
    std::optional<double> quant_kernel_ms{};

    /** 反量化阶段所有 kernel 的合计时间；CPU/reference 路径为 null。 */
    std::optional<double> dequant_kernel_ms{};

    /**
     * @brief 量化逻辑读写字节数除以 quant_kernel_ms，单位 GB/s。
     *
     * 无计时或 CUDA Event 分辨率将极短 kernel 量为 0 ms 时为 null；后者的带宽
     * 无法定义为有限数，但 `quant_kernel_ms = 0.0` 仍会如实保留。
     */
    std::optional<double> quant_effective_bandwidth_gbps{};

    /**
     * @brief 反量化逻辑读写字节数除以 dequant_kernel_ms，单位 GB/s。
     *
     * 无计时或 CUDA Event 分辨率将极短 kernel 量为 0 ms 时为 null。
     */
    std::optional<double> dequant_effective_bandwidth_gbps{};
};

/**
 * @brief 一次完整 app 运行需要写入 JSON 的所有稳定报告字段。
 *
 * 该对象只保存值和元数据，不保存输入/输出大数组，也不持有 CUDA event、stream 或
 * 文件句柄。app 在量化、反量化、写文件后构造它，再交给 `write_run_report_json()`。
 */
struct RunReport {
    /** 原始输入 QDTENSOR 的形状与物理输入 dtype。 */
    TensorDesc input_desc{};

    /** 本次量化使用的格式、block、scale 和舍入配置。 */
    QuantizationConfig quantization{};

    /** 本次反量化写出 QDTENSOR 的目标物理 dtype。 */
    DequantizationConfig dequantization{};

    /** 仅用于实验记录的 GPU 名称。 */
    std::string target_gpu{};

    /** 输入、输出文件和低精度布局的字节统计。 */
    ArtifactMetrics artifacts{};

    /** 原始输入和最终物理输出之间的误差。 */
    ErrorMetrics error{};

    /** 逻辑布局与真实文件大小得到的压缩率。 */
    CompressionMetrics compression{};

    /** CUDA kernel 计时和带宽；CPU/reference 路径允许所有字段为 null。 */
    KernelPerformance performance{};
};

/**
 * @brief 计算两个等长、有限 FP32 数组的最大绝对误差、MAE 与 MSE。
 *
 * 调用方应传入最终会写入 QDTENSOR 的数值：当 `output_type` 为 FP16/BF16 时，
 * 必须先完成与实际文件写入一致的窄化再扩展，避免漏掉物理输出类型带来的误差。
 *
 * @param reference 原始输入扩展得到的 row-major FP32 参考值。
 * @param actual 最终反量化输出对应的 row-major FP32 值。
 * @return 使用 double 累积得到的三项误差统计。
 * @throws MetricsError 数组为空、长度不同或含 NaN/Inf 时抛出。
 */
[[nodiscard]] ErrorMetrics compute_error_metrics(
    std::span<const float> reference,
    std::span<const float> actual);

/**
 * @brief 推导给定输入描述的 QDTENSOR 数值 payload 字节数。
 *
 * @param input_desc 合法的 FP16 或 FP32 输入描述。
 * @return 不含 QDTENSOR header 的原始输入 payload 字节数。
 * @throws MetricsError 描述、dtype 或字节数溢出不合法时抛出。
 */
[[nodiscard]] std::uint64_t compute_input_payload_bytes(
    const TensorDesc& input_desc);

/**
 * @brief 推导目标反量化 QDTENSOR 数值 payload 的字节数。
 *
 * @param output_desc 合法的 FP16、BF16 或 FP32 输出描述。
 * @return 不含 QDTENSOR header 的输出 payload 字节数。
 * @throws MetricsError 描述、dtype 或字节数溢出不合法时抛出。
 */
[[nodiscard]] std::uint64_t compute_dequantized_payload_bytes(
    const TensorDesc& output_desc);

/**
 * @brief 从自洽 QuantizedTensor 推导量化逻辑存储字节数。
 *
 * 结果固定为 `payload + local_scales + (NVFP4 ? 4 : 0)`，不包含 QDWGT 128-byte
 * header 或 section padding，适用于逻辑压缩率和 kernel 逻辑读写量。
 *
 * @param quantized 已完成量化的 host 低精度张量。
 * @return 低精度逻辑存储的总字节数。
 * @throws MetricsError quantized 不自洽或加法溢出时抛出。
 */
[[nodiscard]] std::uint64_t compute_logical_quantized_bytes(
    const QuantizedTensor& quantized);

/**
 * @brief 构造量化文件布局及实际输出文件大小的字节统计。
 *
 * @param input_desc 原始 QDTENSOR 输入描述。
 * @param quantized 已完成量化的低精度结果。
 * @param dequantized_output_desc 最终写出 QDTENSOR 的输出描述。
 * @param quantized_file_bytes 实际 QDWGT 文件总字节数。
 * @param dequantized_file_bytes 实际输出 QDTENSOR 文件总字节数。
 * @return 可直接写入 `RunReport::artifacts` 的统计结构。
 * @throws MetricsError 描述、量化结果或计算出的字节数不合法时抛出。
 */
[[nodiscard]] ArtifactMetrics make_artifact_metrics(
    const TensorDesc& input_desc,
    const QuantizedTensor& quantized,
    const TensorDesc& dequantized_output_desc,
    std::uint64_t quantized_file_bytes,
    std::uint64_t dequantized_file_bytes);

/**
 * @brief 从产物字节统计计算逻辑压缩率和落盘压缩率。
 *
 * @param artifacts 由 `make_artifact_metrics()` 产生的字节统计。
 * @return 与 QDWGT/QDTENSOR 文件布局一致的压缩率结构。
 * @throws MetricsError 输入、逻辑量化或 QDWGT 文件字节数为零时抛出。
 */
[[nodiscard]] CompressionMetrics compute_compression_metrics(
    const ArtifactMetrics& artifacts);

/**
 * @brief 用逻辑读写字节数和 kernel 时间计算十进制 GB/s。
 *
 * @param logical_bytes 该阶段的逻辑读写字节总数。
 * @param kernel_ms CUDA Event 测得的 kernel 合计时间，单位毫秒。
 * @return `logical_bytes / (kernel_ms * 1e6)`，单位 GB/s。
 * @throws MetricsError logical_bytes 为零，或 kernel_ms 非有限/非正时抛出。
 */
[[nodiscard]] double compute_effective_bandwidth_gbps(
    std::uint64_t logical_bytes,
    double kernel_ms);

/**
 * @brief 从可选 CUDA kernel 时间构造完整性能指标。
 *
 * quantize 的逻辑读写量为输入 payload 加量化逻辑字节数；dequantize 的逻辑读写量
 * 为量化逻辑字节数加最终输出 payload。无 CUDA 计时时保持 null，不伪造 0.0。
 *
 * @param artifacts 当前运行的产物字节统计。
 * @param quantized 已完成量化的低精度张量。
 * @param dequantized_output_desc 最终写出 QDTENSOR 的输出描述，用于精确推导
 * 反量化 kernel 的逻辑写入 payload 字节数。
 * @param quant_kernel_ms 量化 kernel 合计时间；CPU/reference 路径传 null。
 * @param dequant_kernel_ms 反量化 kernel 合计时间；CPU/reference 路径传 null。
 * 已提供的合法 0 ms CUDA Event 时间会保留为 `0.0`，但对应带宽为 null，避免
 * 除以零；正常的正时间会同时产生带宽。
 *
 * @return 包含可选时间和可选带宽的性能结构。
 * @throws MetricsError 量化结果、字节数或已提供的时间不合法时抛出。
 */
[[nodiscard]] KernelPerformance make_kernel_performance(
    const ArtifactMetrics& artifacts,
    const QuantizedTensor& quantized,
    const TensorDesc& dequantized_output_desc,
    std::optional<double> quant_kernel_ms,
    std::optional<double> dequant_kernel_ms);

/**
 * @brief 将自洽 RunReport 序列化为稳定、带两个空格缩进的 JSON 文本。
 *
 * 可选 kernel 时间和带宽会写为 JSON `null`，而不是数值 0。该函数不访问文件，
 * 方便测试精确检查字段和让 app 自行决定写入位置。
 *
 * @param report 完整 app 运行的报告对象。
 * @return 以换行结尾的 JSON 文本。
 * @throws MetricsError 报告字段违反本项目约定时抛出。
 */
[[nodiscard]] std::string serialize_run_report_json(const RunReport& report);

/**
 * @brief 将 RunReport 以 JSON 写入指定路径。
 *
 * @param output_path 要创建或覆盖的报告 JSON 文件。
 * @param report 待序列化的完整报告。
 * @throws MetricsError 路径无法写入或报告不合法时抛出。
 */
void write_run_report_json(const std::filesystem::path& output_path,
                           const RunReport& report);

}  // namespace quant_dequant
