#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <limits>
#include <string_view>
#include <utility>
#include <vector>

#include "formats/fp32_utils.cuh"
#include "quant_dequant/quantize.hpp"

namespace {

using quant_dequant::CudaPipelineError;
using quant_dequant::DType;
using quant_dequant::HostTensor;
using quant_dequant::QuantFormat;
using quant_dequant::QuantizationConfig;
using quant_dequant::QuantizedTensor;
using quant_dequant::RoundingMode;
using quant_dequant::ScaleMode;
using quant_dequant::TensorDesc;
using quant_dequant::kNvfp4BlockSize;

/** NVFP4 local-scale persistent grid 的目标 CTA/SM 数量。 */
constexpr std::uint64_t kTargetCtasPerSm = 4U;

/** 每个 256-thread CTA 包含的 16 元素 NVFP4 logical block 数量。 */
constexpr std::uint64_t kNvfp4TilesPerCta = 16U;

/**
 * @brief 生成足以触发 local-scale 与 packed-encode grid-stride 循环的行数。
 *
 * 35 列矩阵每行有 3 个 NVFP4 local block，且为奇数列，因而同时覆盖尾 block
 * 和跨行 packed byte。此行数至少超过 local-scale persistent 首轮容量。
 *
 * @param properties 当前 CUDA device 属性。
 * @return 合法且非零的测试矩阵行数。
 */
[[nodiscard]] std::uint64_t persistent_test_row_count(
    const cudaDeviceProp& properties) {
    return static_cast<std::uint64_t>(properties.multiProcessorCount) *
               kTargetCtasPerSm * kNvfp4TilesPerCta +
           3U;
}

/**
 * @brief 构造确定性有限 FP32 输入，覆盖多种 E2M1 和 scale 边界。
 *
 * @param element_count 所需 row-major 元素数量。
 * @return 可供 CPU reference 与 CUDA pipeline 重复使用的输入数值。
 */
[[nodiscard]] std::vector<float> make_input_values(
    const std::size_t element_count) {
    std::vector<float> values(element_count, 0.0F);
    for (std::size_t index = 0U; index < values.size(); ++index) {
        const int bucket =
            static_cast<int>((index * 29U + 7U) % 113U) - 56;
        values[index] = static_cast<float>(bucket) * 0.125F;

        if (index % 131U == 0U) {
            values[index] = 0.0F;
        } else if (index % 251U == 0U) {
            values[index] = 6.0F;
        } else if (index % 509U == 0U) {
            values[index] = -18.0F;
        } else if (index % 347U == 0U) {
            values[index] = 0.5F;
        }
    }

    return values;
}

/**
 * @brief 比较 CPU reference 与 CUDA 量化结果的所有持久化字段。
 *
 * @param cpu CPU reference 量化结果。
 * @param cuda CUDA pipeline 经 D2H 得到的量化结果。
 * @param case_name 用于失败诊断的舍入模式名称。
 * @return 描述、packed payload、E4M3 local scale 和 FP32 global scale 位级一致时返回 true。
 */
[[nodiscard]] bool equal_quantized_results(
    const QuantizedTensor& cpu,
    const QuantizedTensor& cuda,
    const std::string_view case_name) {
    const bool same_desc =
        cpu.desc.source_desc.num_rows == cuda.desc.source_desc.num_rows &&
        cpu.desc.source_desc.num_cols == cuda.desc.source_desc.num_cols &&
        cpu.desc.source_desc.dtype == cuda.desc.source_desc.dtype &&
        cpu.desc.format == cuda.desc.format &&
        cpu.desc.scale_mode == cuda.desc.scale_mode &&
        cpu.desc.rounding == cuda.desc.rounding &&
        cpu.desc.stochastic_seed == cuda.desc.stochastic_seed &&
        cpu.desc.block_size == cuda.desc.block_size &&
        cpu.desc.scale_layout == cuda.desc.scale_layout;

    const bool same_global_scale = cpu.global_scale.has_value() &&
        cuda.global_scale.has_value() &&
        quant_dequant::formats::fp32::float_to_bits(*cpu.global_scale) ==
            quant_dequant::formats::fp32::float_to_bits(*cuda.global_scale);
    if (!same_desc || cpu.payload != cuda.payload ||
        cpu.local_scales != cuda.local_scales || !same_global_scale) {
        std::cerr << "NVFP4 CUDA " << case_name
                  << " 的量化结果与 CPU reference 不一致。\n";
        return false;
    }

    return true;
}

/**
 * @brief 执行一次 NVFP4 CPU/CUDA 逐字节量化对照。
 *
 * @param input 有限 FP32 host 输入。
 * @param config 严格 NVFP4 block-scale 配置。
 * @param case_name 用于失败诊断的测试名称。
 * @return CPU 与 CUDA 量化结果逐字段一致时返回 true。
 */
[[nodiscard]] bool compare_with_cpu_reference(
    const HostTensor& input,
    const QuantizationConfig& config,
    const std::string_view case_name) {
    try {
        const QuantizedTensor cpu_result =
            quant_dequant::quantize_reference(input, config);
        const QuantizedTensor cuda_result =
            quant_dequant::quantize_cuda(input, config);
        return equal_quantized_results(cpu_result, cuda_result, case_name);
    } catch (const std::exception& error) {
        std::cerr << "NVFP4 CUDA " << case_name
                  << " 路径意外失败：" << error.what() << '\n';
        return false;
    }
}

/**
 * @brief 验证 CUDA 全局 amax 规约会拒绝 NaN/Inf，而不会交付半成品 payload。
 *
 * @param input 正常有限输入的副本，将在中间位置注入 NaN。
 * @param config 严格 NVFP4 block-scale 配置。
 * @return 抛出包含 NaN/Inf 诊断的 CudaPipelineError 时返回 true。
 */
[[nodiscard]] bool rejects_nonfinite_input(
    HostTensor input,
    const QuantizationConfig& config) {
    input.values[input.values.size() / 2U] =
        std::numeric_limits<float>::quiet_NaN();
    try {
        static_cast<void>(quant_dequant::quantize_cuda(input, config));
    } catch (const CudaPipelineError& error) {
        return std::string_view{error.what()}.find("NaN 或 Inf") !=
               std::string_view::npos;
    } catch (const std::exception& error) {
        std::cerr << "NVFP4 CUDA 非有限输入产生了错误异常类型："
                  << error.what() << '\n';
        return false;
    }

    return false;
}

}  // namespace

/**
 * @brief 运行 NVFP4 CUDA block-scale 与 CPU reference 的逐字节对照测试。
 *
 * 无 CUDA device 或 driver 不可用时明确跳过。可用 GPU 上覆盖 nearest、
 * stochastic、全局两阶段 amax reduction、16-lane local-scale tile、32-lane
 * packed store、奇数列跨行 byte 配对、尾 block、persistent grid-stride 和
 * 非有限输入拒绝。
 *
 * @return 所有可执行测试通过时返回 0，否则返回 1。
 */
int run_nvfp4_cuda_tests() {
    int device_count = 0;
    const cudaError_t device_count_status = cudaGetDeviceCount(&device_count);
    if (device_count_status == cudaErrorNoDevice ||
        device_count_status == cudaErrorInsufficientDriver) {
        std::cout << "CUDA runtime 不可用，跳过 NVFP4 CUDA/CPU 对照测试。\n";
        return 0;
    }

    if (device_count_status != cudaSuccess) {
        std::cerr << "cudaGetDeviceCount 失败："
                  << cudaGetErrorString(device_count_status) << '\n';
        return 1;
    }

    if (device_count == 0) {
        std::cout << "未发现 CUDA 设备，跳过 NVFP4 CUDA/CPU 对照测试。\n";
        return 0;
    }

    int device_id = 0;
    cudaDeviceProp properties{};
    if (cudaGetDevice(&device_id) != cudaSuccess ||
        cudaGetDeviceProperties(&properties, device_id) != cudaSuccess ||
        properties.multiProcessorCount <= 0) {
        std::cerr << "无法查询当前 CUDA device 的有效 SM 数量。\n";
        return 1;
    }

    constexpr std::uint64_t kNumCols = 35U;
    const std::uint64_t num_rows = persistent_test_row_count(properties);
    const std::uint64_t element_count = num_rows * kNumCols;
    if (element_count > std::numeric_limits<std::size_t>::max()) {
        std::cerr << "NVFP4 CUDA 测试矩阵超出 host 容器范围。\n";
        return 1;
    }

    HostTensor input{
        .desc = TensorDesc{
            .num_rows = num_rows,
            .num_cols = kNumCols,
            .dtype = DType::kFloat32,
        },
        .values = make_input_values(static_cast<std::size_t>(element_count)),
    };
    QuantizationConfig nearest_config{
        .format = QuantFormat::kNvfp4,
        .block_size = kNvfp4BlockSize,
        .scale_mode = ScaleMode::kBlock,
        .rounding = RoundingMode::kNearest,
        .stochastic_seed = 0U,
    };
    if (!compare_with_cpu_reference(input, nearest_config, "nearest")) {
        return 1;
    }

    QuantizationConfig stochastic_config = nearest_config;
    stochastic_config.rounding = RoundingMode::kStochastic;
    stochastic_config.stochastic_seed = 0x9e3779b97f4a7c15ULL;
    if (!compare_with_cpu_reference(input, stochastic_config, "stochastic")) {
        return 1;
    }

    if (!rejects_nonfinite_input(input, nearest_config)) {
        std::cerr << "NVFP4 CUDA 未拒绝 NaN 输入。\n";
        return 1;
    }

    return 0;
}
