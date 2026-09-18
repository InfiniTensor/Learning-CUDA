#include "cuda/mxfp8_dequantize.cuh"

#include <cuda_runtime_api.h>

#include <cstdint>
#include <string>

#include <thrust/device_ptr.h>

#include "formats/mxfp8_codec.cuh"
#include "pipeline/device_quantized_tensor.cuh"
#include "quant_dequant/quantize.hpp"

namespace quant_dequant::cuda {
namespace {

/** 单个反量化 CTA 使用的线程数量。 */
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
            "mxfp8 CUDA dequantize " + std::string{operation} +
            " 失败：" + cudaGetErrorString(status)};
    }
}

/**
 * @brief 验证 MXFP8 dequantize launcher 的输入量化张量与输出 FP32 buffer。
 *
 * @param input 包含 E4M3 payload 与 E8M0 local scale 的只读 device 张量。
 * @param output 待写入的 device FP32 输出张量。
 * @throws CudaPipelineError 格式、形状、block 规则或 device buffer 不匹配时抛出。
 */
void validate_mxfp8_dequantize_launch(
    const pipeline::DeviceQuantizedTensor& input,
    const pipeline::DeviceDequantizationOutput& output) {
    if (!input.isConsistent() || !output.isConsistent()) {
        throw CudaPipelineError{
            "mxfp8 CUDA dequantize launcher 收到了不自洽的 device buffer。"};
    }

    if (input.desc.format != QuantFormat::kMxfp8 ||
        input.desc.block_size != kMxfp8BlockSize ||
        (input.desc.scale_mode != ScaleMode::kTensor &&
         input.desc.scale_mode != ScaleMode::kBlock) ||
        output.desc.num_rows != input.desc.source_desc.num_rows ||
        output.desc.num_cols != input.desc.source_desc.num_cols ||
        !is_supported_output_dtype(output.desc.dtype)) {
        throw CudaPipelineError{
            "mxfp8 CUDA dequantize launcher 收到了不匹配的输入描述或输出描述。"};
    }
}

/**
 * @brief 以一维 grid-stride 映射解码完整 row-major MXFP8 张量。
 *
 * 每个线程处理一个或多个线性元素下标。对于元素 `(row, column)`：
 *
 * - tensor-scale：读取唯一的 `local_scales[0]`；
 * - block-scale：读取 `local_scales[row * blocks_per_row + column / 32]`。
 *
 * 随后 `decode_mxfp8_element()` 完成 E4M3 与 E8M0 解码及乘法，结果写入 FP32
 * 输出。payload、scale 与 output 都按连续 row-major 线性布局访问，因此同一
 * warp 的相邻有效线程访问相邻地址，形成合并访存。
 *
 * @param payload device 指针，指向连续 E4M3 byte payload，只读。
 * @param local_scales device 指针，指向 E8M0 scale code，只读。
 * @param output device 指针，指向连续 row-major FP32 结果，写入。
 * @param element_count 矩阵逻辑元素总数。
 * @param num_cols 矩阵列数，用于从线性下标恢复 row/column。
 * @param blocks_per_row block-scale 时每行 32 元素 block 数；tensor-scale 时为 0。
 * @param scale_mode 决定每个元素的 local scale 索引公式。
 */
__global__ __launch_bounds__(kThreadsPerCta) void mxfp8DequantizeKernel(
    const std::uint8_t* __restrict__ payload,
    const std::uint8_t* __restrict__ local_scales,
    float* __restrict__ output,
    const std::uint64_t element_count,
    const std::uint64_t num_cols,
    const std::uint64_t blocks_per_row,
    const ScaleMode scale_mode) {
    const std::uint64_t first_linear_index =
        static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::uint64_t linear_stride =
        static_cast<std::uint64_t>(gridDim.x) * blockDim.x;

    // 大张量的逻辑线程数可超过持久化网格大小；每个线程按固定步长持续领取
    // 后续元素。没有跨元素依赖，因此既无需 shared memory 也无需同步。
    for (std::uint64_t linear_index = first_linear_index;
         linear_index < element_count;
         linear_index += linear_stride) {
        const std::uint64_t row = linear_index / num_cols;
        const std::uint64_t column = linear_index % num_cols;
        const std::uint64_t scale_index =
            scale_mode == ScaleMode::kTensor
                ? 0U
                : row * blocks_per_row +
                      column / static_cast<std::uint64_t>(kMxfp8BlockSize);

        output[linear_index] = formats::decode_mxfp8_element(
            payload[linear_index], local_scales[scale_index]);
    }
}

}  // namespace

void launch_mxfp8_dequantize(
    const pipeline::DeviceQuantizedTensor& input,
    pipeline::DeviceDequantizationOutput& output,
    const cudaStream_t stream) {
    validate_mxfp8_dequantize_launch(input, output);

    const auto element_count = input.desc.elementCount();
    if (!element_count.has_value() || *element_count == 0U) {
        throw CudaPipelineError{
            "mxfp8 CUDA dequantize launcher 无法推导非空元素数量。"};
    }

    // block-scale 唯一需要的格式布局信息是每行 scale 数量；tensor-scale 的
    // 唯一 scale 固定在下标 0，因此显式传入 0，kernel 不会读取该值。
    std::uint64_t blocks_per_row = 0U;
    if (input.desc.scale_mode == ScaleMode::kBlock) {
        const auto derived_blocks_per_row = input.desc.blocksPerRow();
        if (!derived_blocks_per_row.has_value() || *derived_blocks_per_row == 0U) {
            throw CudaPipelineError{
                "mxfp8 CUDA dequantize launcher 无法推导每行 block 数量。"};
        }
        blocks_per_row = *derived_blocks_per_row;
    }

    // 反量化无归约和跨 CTA 通信；网格只需覆盖足够线程后通过 grid-stride 完成
    // 全量元素。将 CTA 数限制到 SM * 4 能避免很大张量为每个小工作项发射 CTA。
    int device_id = 0;
    check_cuda(cudaGetDevice(&device_id), "查询当前 device");

    cudaDeviceProp device_properties{};
    check_cuda(cudaGetDeviceProperties(&device_properties, device_id),
               "查询 device 属性");
    if (device_properties.multiProcessorCount <= 0 ||
        device_properties.maxGridSize[0] <= 0) {
        throw CudaPipelineError{
            "mxfp8 CUDA dequantize launcher 未发现有效的 SM 或 grid.x 上限。"};
    }

    const std::uint64_t logical_cta_count =
        *element_count / static_cast<std::uint64_t>(kThreadsPerCta) +
        (*element_count % static_cast<std::uint64_t>(kThreadsPerCta) == 0U
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
            "mxfp8 CUDA dequantize launcher 无法构造非空执行网格。"};
    }

    const dim3 grid{static_cast<unsigned int>(launch_cta_count), 1U, 1U};
    const dim3 block{static_cast<unsigned int>(kThreadsPerCta), 1U, 1U};

    // DeviceQuantizedTensor / DeviceDequantizationOutput 管理实际所有权；这里
    // 只借用 raw pointer 传给 kernel，不触发额外 H2D/D2H 或隐式同步。
    const auto* const payload_ptr =
        thrust::raw_pointer_cast(input.payload.data());
    const auto* const local_scales_ptr =
        thrust::raw_pointer_cast(input.local_scales.data());
    auto* const output_ptr = thrust::raw_pointer_cast(output.values.data());

    mxfp8DequantizeKernel<<<grid, block, 0U, stream>>>(
        payload_ptr,
        local_scales_ptr,
        output_ptr,
        *element_count,
        input.desc.source_desc.num_cols,
        blocks_per_row,
        input.desc.scale_mode);

    // 这里只检查 launch 参数错误；异步执行期错误将在 pipeline 的 D2H 时由
    // Thrust 传输操作报告。
    check_cuda(cudaGetLastError(), "发射 dequantize kernel");
}

}  // namespace quant_dequant::cuda
