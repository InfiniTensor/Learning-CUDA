#include "cuda/nvfp4_dequantize.cuh"

#include <cuda_runtime_api.h>

#include <cstdint>
#include <string>

#include <thrust/device_ptr.h>

#include "formats/nvfp4_codec.cuh"
#include "pipeline/device_quantized_tensor.cuh"
#include "quant_dequant/quantize.hpp"

namespace quant_dequant::cuda {
namespace {

/** 单个 NVFP4 反量化 CTA 使用的线程数量。 */
inline constexpr int kThreadsPerCta = 256;

/** 每个 SM 的目标常驻 CTA 数量，用于限制持久化一维网格规模。 */
inline constexpr int kTargetCtasPerSm = 4;

static_assert(kThreadsPerCta > 0);

/**
 * @brief 将 CUDA runtime 返回值转换为带阶段说明的 pipeline 异常。
 *
 * @param status CUDA runtime API 的返回状态。
 * @param operation 失败的 CUDA 操作名称。
 * @throws CudaPipelineError status 不是 cudaSuccess 时抛出。
 */
void check_cuda(const cudaError_t status, const char* const operation) {
    if (status != cudaSuccess) {
        throw CudaPipelineError{
            "nvfp4 CUDA dequantize " + std::string{operation} +
            " 失败：" + cudaGetErrorString(status)};
    }
}

/**
 * @brief 验证 NVFP4 dequantize launcher 的输入量化张量与输出 FP32 buffer。
 *
 * @param input 包含 packed E2M1 payload、E4M3 local scale 与 FP32 global scale
 * 的只读 device 张量。
 * @param output 待写入的 device FP32 输出张量。
 * @throws CudaPipelineError 格式、形状、block 规则或 device buffer 不匹配时抛出。
 */
void validate_nvfp4_dequantize_launch(
    const pipeline::DeviceQuantizedTensor& input,
    const pipeline::DeviceDequantizationOutput& output) {
    if (!input.isConsistent() || !output.isConsistent()) {
        throw CudaPipelineError{
            "nvfp4 CUDA dequantize launcher 收到了不自洽的 device buffer。"};
    }

    if (input.desc.format != QuantFormat::kNvfp4 ||
        input.desc.block_size != kNvfp4BlockSize ||
        input.desc.scale_mode != ScaleMode::kBlock ||
        input.desc.scale_layout != ScaleLayout::kRowwise ||
        !input.global_scale.has_value() || input.global_scale->size() != 1U ||
        output.desc.num_rows != input.desc.source_desc.num_rows ||
        output.desc.num_cols != input.desc.source_desc.num_cols ||
        !is_supported_output_dtype(output.desc.dtype)) {
        throw CudaPipelineError{
            "nvfp4 CUDA dequantize launcher 收到了不匹配的输入描述或输出描述。"};
    }
}

/**
 * @brief 以一维 grid-stride 映射解码完整的 packed NVFP4 row-major payload。
 *
 * 每个线程处理一个或多个物理 payload byte。一个 byte 的低四位对应线性下标
 * `2 * payload_index`，高四位对应其后一项。两个元素分别恢复 `(row, column)`，
 * 并使用 `row * blocks_per_row + column / 16` 索引自己的 E4M3 local scale。
 * 因而奇数列矩阵的跨行 byte 不会误共享 local scale；最后一个奇数长度张量的
 * padding 高 nibble 也绝不会解码或写入输出。
 *
 * @param payload device 指针，指向 packed E2M1 byte payload，只读。
 * @param local_scales device 指针，指向 rowwise E4M3 scale code，只读。
 * @param global_scale device 指针，指向单个 FP32 global scale，只读。
 * @param output device 指针，指向连续 row-major FP32 结果，写入。
 * @param element_count 矩阵逻辑元素总数。
 * @param payload_byte_count 待读取的物理 packed payload byte 数。
 * @param num_cols 矩阵列数，用于从线性下标恢复 row/column。
 * @param blocks_per_row 每行的 16 元素 local-scale 数量。
 */
__global__ __launch_bounds__(kThreadsPerCta) void nvfp4DequantizeKernel(
    const std::uint8_t* __restrict__ payload,
    const std::uint8_t* __restrict__ local_scales,
    const float* __restrict__ global_scale,
    float* __restrict__ output,
    const std::uint64_t element_count,
    const std::uint64_t payload_byte_count,
    const std::uint64_t num_cols,
    const std::uint64_t blocks_per_row) {
    const std::uint64_t first_payload_index =
        static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::uint64_t payload_stride =
        static_cast<std::uint64_t>(gridDim.x) * blockDim.x;
    const float decode_global_scale = global_scale[0U];

    // payload、output 都按连续 row-major 次序访问。连续 warp 中每个线程读一个
    // 相邻 byte，且写相邻的两个 FP32 元素区间，保留自然的合并访存模式。
    for (std::uint64_t payload_index = first_payload_index;
         payload_index < payload_byte_count;
         payload_index += payload_stride) {
        const std::uint64_t even_index = payload_index * 2U;
        const std::uint64_t even_row = even_index / num_cols;
        const std::uint64_t even_column = even_index % num_cols;
        const std::uint64_t even_scale_index =
            even_row * blocks_per_row +
            even_column / static_cast<std::uint64_t>(kNvfp4BlockSize);
        const std::uint8_t packed_byte = payload[payload_index];

        output[even_index] = formats::decode_nvfp4_element(
            formats::unpack_e2m1_low_nibble(packed_byte),
            local_scales[even_scale_index],
            decode_global_scale);

        const std::uint64_t odd_index = even_index + 1U;
        if (odd_index < element_count) {
            const std::uint64_t odd_row = odd_index / num_cols;
            const std::uint64_t odd_column = odd_index % num_cols;
            const std::uint64_t odd_scale_index =
                odd_row * blocks_per_row +
                odd_column / static_cast<std::uint64_t>(kNvfp4BlockSize);
            output[odd_index] = formats::decode_nvfp4_element(
                formats::unpack_e2m1_high_nibble(packed_byte),
                local_scales[odd_scale_index],
                decode_global_scale);
        }
    }
}

}  // namespace

void launch_nvfp4_dequantize(
    const pipeline::DeviceQuantizedTensor& input,
    pipeline::DeviceDequantizationOutput& output,
    const cudaStream_t stream) {
    validate_nvfp4_dequantize_launch(input, output);

    const auto element_count = input.desc.elementCount();
    const auto blocks_per_row = input.desc.blocksPerRow();
    if (!element_count.has_value() || *element_count == 0U ||
        !blocks_per_row.has_value() || *blocks_per_row == 0U ||
        input.payload.empty()) {
        throw CudaPipelineError{
            "nvfp4 CUDA dequantize launcher 无法推导非空张量或 rowwise scale 布局。"};
    }

    int device_id = 0;
    check_cuda(cudaGetDevice(&device_id), "查询当前 device");

    cudaDeviceProp device_properties{};
    check_cuda(cudaGetDeviceProperties(&device_properties, device_id),
               "查询 device 属性");
    if (device_properties.multiProcessorCount <= 0 ||
        device_properties.maxGridSize[0] <= 0) {
        throw CudaPipelineError{
            "nvfp4 CUDA dequantize launcher 未发现有效的 SM 或 grid.x 上限。"};
    }

    const std::uint64_t payload_byte_count =
        static_cast<std::uint64_t>(input.payload.size());
    const std::uint64_t logical_cta_count =
        payload_byte_count / static_cast<std::uint64_t>(kThreadsPerCta) +
        (payload_byte_count % static_cast<std::uint64_t>(kThreadsPerCta) == 0U
             ? 0U
             : 1U);
    const std::uint64_t resident_cta_count =
        static_cast<std::uint64_t>(device_properties.multiProcessorCount) *
        static_cast<std::uint64_t>(kTargetCtasPerSm);
    const std::uint64_t desired_cta_count =
        logical_cta_count < resident_cta_count
            ? logical_cta_count
            : resident_cta_count;
    const std::uint64_t max_grid_x =
        static_cast<std::uint64_t>(device_properties.maxGridSize[0]);
    const std::uint64_t launch_cta_count =
        desired_cta_count < max_grid_x ? desired_cta_count : max_grid_x;
    if (launch_cta_count == 0U) {
        throw CudaPipelineError{
            "nvfp4 CUDA dequantize launcher 无法构造非空执行网格。"};
    }

    const dim3 grid{static_cast<unsigned int>(launch_cta_count), 1U, 1U};
    const dim3 block{static_cast<unsigned int>(kThreadsPerCta), 1U, 1U};
    const auto* const payload_ptr = thrust::raw_pointer_cast(input.payload.data());
    const auto* const local_scales_ptr =
        thrust::raw_pointer_cast(input.local_scales.data());
    const auto* const global_scale_ptr =
        thrust::raw_pointer_cast(input.global_scale->data());
    auto* const output_ptr = thrust::raw_pointer_cast(output.values.data());

    nvfp4DequantizeKernel<<<grid, block, 0U, stream>>>(
        payload_ptr,
        local_scales_ptr,
        global_scale_ptr,
        output_ptr,
        *element_count,
        payload_byte_count,
        input.desc.source_desc.num_cols,
        *blocks_per_row);

    // 仅检查参数/发射错误。kernel 的异步执行期错误会在 pipeline 随后的 Thrust
    // D2H 中报告；这里不主动同步，避免破坏上层的异步传输机会。
    check_cuda(cudaGetLastError(), "发射 dequantize kernel");
}

}  // namespace quant_dequant::cuda
