#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <limits>
#include <string_view>
#include <utility>
#include <vector>

#include "quant_dequant/quantize.hpp"

namespace {

using quant_dequant::CudaPipelineError;
using quant_dequant::DequantizationConfig;
using quant_dequant::DType;
using quant_dequant::HostTensor;
using quant_dequant::QuantFormat;
using quant_dequant::QuantizationConfig;
using quant_dequant::QuantizedTensor;
using quant_dequant::RoundingMode;
using quant_dequant::ScaleMode;
using quant_dequant::TensorDesc;
using quant_dequant::kMxfp8BlockSize;

/** 每个 CTA 中并行处理的 MXFP8 逻辑量化 block 数。 */
constexpr std::uint64_t kWarpsPerCta = 8U;

/** CUDA launcher 为每个 SM 期望常驻的 CTA 数量。 */
constexpr std::uint64_t kTargetCtasPerSm = 4U;

/**
 * @brief 返回测试应使用的持久化 kernel 工作行数。
 *
 * CUDA block-scale launcher 的一个 warp 处理一个 32 元素量化 block，且最多
 * 启动 `SM 数 * 4` 个 CTA，每个 CTA 有 8 个 warp。这里构造比该首轮容量多
 * 3 行的 35 列矩阵；每行有两个量化 block，因而无论实际 occupancy 是否低于
 * 目标值，至少一部分 warp 都必须通过 grid-stride 循环领取第二轮工作。同时
 * 35 列也覆盖了每行最后一个不足 32 元素的 tail block。
 *
 * @param properties 当前 CUDA device 的硬件属性。
 * @return 足以覆盖持久化循环与 tail block 的合法行数。
 */
[[nodiscard]] std::uint64_t persistent_test_row_count(
    const cudaDeviceProp& properties) {
    const std::uint64_t sm_count =
        static_cast<std::uint64_t>(properties.multiProcessorCount);
    return sm_count * kTargetCtasPerSm * kWarpsPerCta + 3U;
}

/**
 * @brief 构造包含多种幅值、零和 block 边界值的有限 FP32 测试输入。
 *
 * 值的生成不依赖随机数，因此 CPU reference 与 CUDA 的逐字节比较可稳定复现。
 * 其中较大的正负值会使不同 block 选择不同 E8M0 scale，小值覆盖 E4M3 的
 * normal/subnormal 编码区域；所有值保持有限，正常路径不应触发错误哨兵。
 *
 * @param element_count 所需 row-major FP32 元素数量。
 * @return 可直接装入 `HostTensor::values` 的确定性数据。
 */
[[nodiscard]] std::vector<float> make_finite_input_values(
    const std::size_t element_count) {
    std::vector<float> values(element_count, 0.0F);

    for (std::size_t index = 0U; index < values.size(); ++index) {
        const int bucket =
            static_cast<int>((index * 17U + 11U) % 97U) - 48;
        values[index] = static_cast<float>(bucket) * 0.125F;

        if (index % 113U == 0U) {
            values[index] = 0.0F;
        } else if (index % 257U == 0U) {
            values[index] = 448.0F;
        } else if (index % 521U == 0U) {
            values[index] = -896.0F;
        } else if (index % 347U == 0U) {
            values[index] = 0x1.0p-9F;
        }
    }

    return values;
}

/**
 * @brief 比较 CPU 和 CUDA 返回的 MXFP8 量化结果全部持久化字段。
 *
 * 若 payload 和 E8M0 local scale 都逐字节一致，二者代表完全相同的低精度
 * 张量；额外比较元数据和 `global_scale`，可发现 D2H 构造结果时的布局错误。
 *
 * @param cpu CPU reference 量化结果。
 * @param cuda CUDA pipeline D2H 量化结果。
 * @param case_name 用于失败诊断的舍入模式名称。
 * @return 所有持久化字段完全一致时为 true。
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

    if (!same_desc || cpu.payload != cuda.payload ||
        cpu.local_scales != cuda.local_scales ||
        cpu.global_scale != cuda.global_scale) {
        std::cerr << "MXFP8 CUDA " << case_name
                  << " 量化结果与 CPU reference 不一致。\n";
        return false;
    }

    return true;
}

/**
 * @brief 对同一输入执行一次 MXFP8 CUDA block-scale 量化到反量化的端到端对照。
 *
 * 此测试刻意将 `quantize_cuda()` D2H 返回的 `cuda_result` 直接传入
 * `dequantize_cuda()`，而不在两个 CUDA 阶段之间替换成 CPU reference 的低精度
 * 张量。这样同时覆盖 GPU 量化结果的 host 持久化布局、GPU 反量化输入 H2D、
 * E4M3/E8M0 解码 kernel 与最终 D2H。CPU reference 量化结果仅用于逐字节检查，
 * CPU reference 反量化结果则是最终 FP32 数值真值。
 *
 * @param input 有限 FP32 host 输入。
 * @param config 合法的 MXFP8 block-scale 或 tensor-scale 量化配置。
 * @param case_name 用于失败诊断的 scale 模式与舍入模式名称。
 * @return CPU/CUDA 低精度字节和端到端反量化 FP32 数值均一致时为 true。
 */
[[nodiscard]] bool compare_with_cpu_reference(
    const HostTensor& input,
    const QuantizationConfig& config,
    const std::string_view case_name) {
    QuantizedTensor cpu_result{};
    QuantizedTensor cuda_result{};
    try {
        cpu_result = quant_dequant::quantize_reference(input, config);
        cuda_result = quant_dequant::quantize_cuda(input, config);
    } catch (const std::exception& error) {
        std::cerr << "MXFP8 CUDA " << case_name
                  << " 路径意外失败：" << error.what() << '\n';
        return false;
    }

    if (!equal_quantized_results(cpu_result, cuda_result, case_name)) {
        return false;
    }

    DequantizationConfig dequantization_config{};
    dequantization_config.output_type = DType::kFloat32;

    try {
        const HostTensor cpu_dequantized =
            quant_dequant::dequantize_reference(cpu_result, dequantization_config);
        const HostTensor cuda_dequantized =
            quant_dequant::dequantize_cuda(cuda_result, dequantization_config);

        if (cpu_dequantized.desc.num_rows != cuda_dequantized.desc.num_rows ||
            cpu_dequantized.desc.num_cols != cuda_dequantized.desc.num_cols ||
            cpu_dequantized.desc.dtype != cuda_dequantized.desc.dtype ||
            cpu_dequantized.values != cuda_dequantized.values) {
            std::cerr << "MXFP8 CUDA " << case_name
                      << " GPU 量化→GPU 反量化结果与 CPU reference 不一致。\n";
            return false;
        }
    } catch (const std::exception& error) {
        std::cerr << "MXFP8 CUDA " << case_name
                  << " GPU 量化→GPU 反量化路径意外失败：" << error.what() << '\n';
        return false;
    }

    return true;
}

/**
 * @brief 验证 CUDA kernel 经由 E8M0 NaN 哨兵拒绝非有限输入。
 *
 * CPU reference 会在扫描 amax 时立即拒绝 NaN/Inf；CUDA 版本在 kernel 内由
 * warp ballot 发现非有限 lane、将对应 scale 写为 `0xff`，再由 D2H 阶段转为
 * `CudaPipelineError`。这个测试覆盖后者，确保不会返回带无意义 payload 的
 * `QuantizedTensor`。
 *
 * @param input 正常有限的输入，将复制后注入一个 NaN。
 * @param config 合法的 MXFP8 block-scale 配置。
 * @return CUDA 路径以包含 NaN/Inf 诊断的 CudaPipelineError 失败时为 true。
 */
[[nodiscard]] bool rejects_nonfinite_input(
    HostTensor input,
    const QuantizationConfig& config) {
    input.values.at(input.values.size() / 2U) =
        std::numeric_limits<float>::quiet_NaN();

    try {
        static_cast<void>(quant_dequant::quantize_cuda(input, config));
    } catch (const CudaPipelineError& error) {
        return std::string_view{error.what()}.find("NaN 或 Inf") !=
               std::string_view::npos;
    } catch (const std::exception& error) {
        std::cerr << "MXFP8 CUDA 非有限输入产生了错误异常类型："
                  << error.what() << '\n';
        return false;
    }

    std::cerr << "MXFP8 CUDA 没有拒绝 NaN 输入。\n";
    return false;
}

}  // namespace

/**
 * @brief 运行 MXFP8 CUDA tensor/block-scale 与 CPU reference 的端到端对照测试。
 *
 * 无 CUDA device 或当前机器缺少可用 driver 时，测试明确跳过并返回成功；其余
 * CUDA runtime 错误应被视为环境或实现失败。可用 GPU 上依次比较 nearest 与
 * stochastic rounding，覆盖 block-scale 的 persistent grid-stride、尾 block、
 * tensor-scale 的两阶段全局 amax reduction、量化 D2H、GPU 量化结果直接传入
 * GPU 反量化，以及两种模式的非有限输入错误哨兵。
 *
 * @return 所有可执行的 CUDA 对照项通过时返回 0，否则返回 1。
 */
int run_mxfp8_cuda_tests() {
    int device_count = 0;
    const cudaError_t device_count_status = cudaGetDeviceCount(&device_count);
    if (device_count_status == cudaErrorNoDevice ||
        device_count_status == cudaErrorInsufficientDriver) {
        std::cout << "CUDA runtime 不可用，跳过 MXFP8 CUDA/CPU 对照测试。\n";
        return 0;
    }

    if (device_count_status != cudaSuccess) {
        std::cerr << "cudaGetDeviceCount 失败："
                  << cudaGetErrorString(device_count_status) << '\n';
        return 1;
    }

    if (device_count == 0) {
        std::cout << "未发现 CUDA 设备，跳过 MXFP8 CUDA/CPU 对照测试。\n";
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
        std::cerr << "MXFP8 CUDA 测试矩阵超出 host 容器范围。\n";
        return 1;
    }

    HostTensor input{};
    input.desc = TensorDesc{
        num_rows,
        kNumCols,
        DType::kFloat32,
    };
    input.values = make_finite_input_values(
        static_cast<std::size_t>(element_count));

    QuantizationConfig nearest_config{};
    nearest_config.format = QuantFormat::kMxfp8;
    nearest_config.block_size = kMxfp8BlockSize;
    nearest_config.scale_mode = ScaleMode::kBlock;
    nearest_config.rounding = RoundingMode::kNearest;

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
        return 1;
    }

    // tensor mode 使用同一 E8M0 scale 覆盖整个矩阵。这里复用足以触发多轮
    // grid-stride 的测试矩阵，分别验证手写两阶段 amax reduction、scale[0]
    // 写入和独立 encode kernel 在 nearest/stochastic 下都与 CPU reference 一致。
    QuantizationConfig tensor_nearest_config = nearest_config;
    tensor_nearest_config.scale_mode = ScaleMode::kTensor;
    if (!compare_with_cpu_reference(
            input, tensor_nearest_config, "tensor-nearest")) {
        return 1;
    }

    QuantizationConfig tensor_stochastic_config = tensor_nearest_config;
    tensor_stochastic_config.rounding = RoundingMode::kStochastic;
    tensor_stochastic_config.stochastic_seed = 0xd1b54a32d192ed03ULL;
    if (!compare_with_cpu_reference(
            input, tensor_stochastic_config, "tensor-stochastic")) {
        return 1;
    }

    if (!rejects_nonfinite_input(input, tensor_nearest_config)) {
        return 1;
    }

    return 0;
}
