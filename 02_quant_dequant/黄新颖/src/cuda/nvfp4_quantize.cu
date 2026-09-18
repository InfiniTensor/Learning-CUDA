#include "cuda/nvfp4_quantize.cuh"

#include <cooperative_groups.h>

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>

#include <thrust/device_ptr.h>

#include "formats/nvfp4_codec.cuh"
#include "quant_dequant/quantize.hpp"
#include "pipeline/device_quantized_tensor.cuh"

namespace cg = cooperative_groups;

namespace quant_dequant::cuda {
namespace {

/** 单个 CUDA warp 的固定 lane 数量。 */
inline constexpr int kWarpSize = 32;

/** 一个 CTA 的固定 warp 数量。 */
inline constexpr int kWarpsPerCta = 8;

/** local-scale kernel 中每个 CTA 的 16 元素 Cooperative Group tile 数量。 */
inline constexpr int kNvfp4TilesPerCta = kWarpsPerCta * 2;

/** 所有 NVFP4 quantize kernel 的固定 CTA 线程数。 */
inline constexpr int kThreadsPerCta = kWarpSize * kWarpsPerCta;

/** persistent grid 希望每个 SM 同时承载的 CTA 数量上限。 */
inline constexpr int kTargetCtasPerSm = 4;

static_assert(kNvfp4BlockSize == static_cast<std::uint32_t>(kWarpSize / 2));
static_assert(kThreadsPerCta % kWarpSize == 0);

/**
 * @brief 将 CUDA runtime 状态转换为格式专用的公共 pipeline 异常。
 *
 * @param status CUDA runtime API 的返回值。
 * @param operation 失败操作的中文阶段说明。
 * @throws CudaPipelineError status 不是 cudaSuccess 时抛出。
 */
void check_cuda(const cudaError_t status, const char* const operation) {
    if (status != cudaSuccess) {
        throw CudaPipelineError{
            "nvfp4 CUDA " + std::string{operation} +
            " 失败：" + cudaGetErrorString(status)};
    }
}

/**
 * @brief 保存一个规约线程看见的有限 amax 与非有限值标记。
 *
 * @note NVFP4 的 global_scale 必须来自整张张量的有限 amax；若只做 float max，
 * NaN 可能被静默吞掉，因此必须和存在性标记一起规约。
 */
struct TensorAmaxState {
    /** 当前局部范围内有限元素的最大绝对值。 */
    float amax{0.0F};

    /** 当前局部范围是否出现 NaN、+Inf 或 -Inf。 */
    std::uint32_t has_nonfinite{0U};
};

/**
 * @brief 将一个输入 FP32 累计到全局 amax 规约状态。
 *
 * @param state 当前线程私有规约状态，原地更新。
 * @param value 刚从 global memory 加载的一个 FP32 输入。
 */
__device__ __forceinline__ void accumulate_tensor_amax(
    TensorAmaxState& state,
    const float value) {
    if (!formats::fp32::is_finite(value)) {
        state.has_nonfinite = 1U;
        return;
    }

    const float magnitude = formats::fp32::absolute_value(value);
    state.amax = state.amax > magnitude ? state.amax : magnitude;
}

/**
 * @brief 在一个完整 warp 内规约 NVFP4 全局 amax 状态。
 *
 * @param state 当前 lane 的局部状态。
 * @return lane 0 得到该 warp 的完整状态；其余 lane 的返回值是中间结果。
 */
__device__ __forceinline__ TensorAmaxState warp_reduce_tensor_amax(
    TensorAmaxState state) {
    constexpr unsigned int kAllWarpLanes = 0xffffffffU;
    for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
        const float neighbor_amax =
            __shfl_down_sync(kAllWarpLanes, state.amax, offset);
        const std::uint32_t neighbor_nonfinite =
            __shfl_down_sync(kAllWarpLanes, state.has_nonfinite, offset);
        state.amax = state.amax > neighbor_amax ? state.amax : neighbor_amax;
        state.has_nonfinite |= neighbor_nonfinite;
    }

    return state;
}

/**
 * @brief 以 warp shuffle 与 shared memory 规约一个固定 256-thread CTA。
 *
 * @param state 当前线程的局部规约状态。
 * @param shared_amax 长度为 8 的 shared 数组，用于每 warp 一项 amax。
 * @param shared_nonfinite 长度为 8 的 shared 数组，用于每 warp 一项标记。
 * @return thread 0 得到完整 CTA 结果；其他线程的结果无需使用。
 */
__device__ __forceinline__ TensorAmaxState block_reduce_tensor_amax(
    TensorAmaxState state,
    float* const shared_amax,
    std::uint32_t* const shared_nonfinite) {
    const std::uint32_t lane_id = threadIdx.x % kWarpSize;
    const std::uint32_t warp_id = threadIdx.x / kWarpSize;

    state = warp_reduce_tensor_amax(state);
    if (lane_id == 0U) {
        shared_amax[warp_id] = state.amax;
        shared_nonfinite[warp_id] = state.has_nonfinite;
    }

    // warp 0 读取每 warp partial 前，必须等待全部 lane 0 写入 shared memory。
    __syncthreads();

    if (warp_id == 0U) {
        const std::uint32_t warp_count =
            static_cast<std::uint32_t>(blockDim.x / kWarpSize);
        state = lane_id < warp_count
            ? TensorAmaxState{
                  shared_amax[lane_id],
                  shared_nonfinite[lane_id],
              }
            : TensorAmaxState{};
        state = warp_reduce_tensor_amax(state);
    }

    return state;
}

/**
 * @brief 第一阶段：以 float4 grid-stride 遍历输入，写每 CTA 的全局 amax partial。
 *
 * @param input 连续 row-major FP32 输入的 device 指针，只读。
 * @param partial_amax 长度等于 grid.x 的 device 输出，每 CTA 写一项。
 * @param partial_nonfinite 长度等于 grid.x 的 device 输出，每 CTA 写一项。
 * @param element_count 输入的逻辑元素数量。
 */
__global__ __launch_bounds__(kThreadsPerCta) void nvfp4TensorAmaxPartialKernel(
    const float* __restrict__ input,
    float* __restrict__ partial_amax,
    std::uint32_t* __restrict__ partial_nonfinite,
    const std::uint64_t element_count) {
    __shared__ float shared_amax[kWarpsPerCta];
    __shared__ std::uint32_t shared_nonfinite[kWarpsPerCta];

    TensorAmaxState state{};
    const auto* const input4 = reinterpret_cast<const float4*>(input);
    const std::uint64_t vector_count = element_count / 4U;
    const std::uint64_t first_vector_index =
        static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::uint64_t vector_stride =
        static_cast<std::uint64_t>(gridDim.x) * blockDim.x;

    for (std::uint64_t vector_index = first_vector_index;
         vector_index < vector_count;
         vector_index += vector_stride) {
        const float4 values = input4[vector_index];
        accumulate_tensor_amax(state, values.x);
        accumulate_tensor_amax(state, values.y);
        accumulate_tensor_amax(state, values.z);
        accumulate_tensor_amax(state, values.w);
    }

    const std::uint64_t tail_start = vector_count * 4U;
    for (std::uint64_t linear_index = tail_start + first_vector_index;
         linear_index < element_count;
         linear_index += vector_stride) {
        accumulate_tensor_amax(state, input[linear_index]);
    }

    state = block_reduce_tensor_amax(state, shared_amax, shared_nonfinite);
    if (threadIdx.x == 0U) {
        partial_amax[blockIdx.x] = state.amax;
        partial_nonfinite[blockIdx.x] = state.has_nonfinite;
    }
}

/**
 * @brief 第二阶段：规约全部 partial 并直接写 NVFP4 的唯一 FP32 global scale。
 *
 * @param partial_amax 第一阶段生成的 amax 数组，只读。
 * @param partial_nonfinite 第一阶段生成的非有限标记数组，只读。
 * @param global_scale 长度为 1 的 device 输出；thread 0 写 decode global scale。
 * @param partial_count 第一阶段 CTA 数量。
 */
__global__ __launch_bounds__(kThreadsPerCta) void nvfp4GlobalScaleFinalizeKernel(
    const float* __restrict__ partial_amax,
    const std::uint32_t* __restrict__ partial_nonfinite,
    float* __restrict__ global_scale,
    const std::uint64_t partial_count) {
    __shared__ float shared_amax[kWarpsPerCta];
    __shared__ std::uint32_t shared_nonfinite[kWarpsPerCta];

    TensorAmaxState state{};
    for (std::uint64_t partial_index = threadIdx.x;
         partial_index < partial_count;
         partial_index += blockDim.x) {
        const float partial_value = partial_amax[partial_index];
        state.amax = state.amax > partial_value ? state.amax : partial_value;
        state.has_nonfinite |= partial_nonfinite[partial_index];
    }

    state = block_reduce_tensor_amax(state, shared_amax, shared_nonfinite);
    if (threadIdx.x == 0U) {
        global_scale[0] = state.has_nonfinite != 0U
            ? formats::fp32::canonical_quiet_nan()
            : formats::compute_nvfp4_global_scale(state.amax);
    }
}

/**
 * @brief 使用 16-thread Cooperative Group tile 计算每个 rowwise block 的 E4M3 scale。
 *
 * 一个 256-thread CTA 含 16 个 `thread_block_tile<16>`，即每个物理 warp 同时
 * 覆盖两个 NVFP4 quantization block。每个 tile 的所有 lane 都参加 shuffle
 * 规约；尾 block 的无效 lane 贡献零且不访问越界输入。
 *
 * @param input 连续 row-major FP32 输入的 device 指针，只读。
 * @param local_scales 每个逻辑 16 元素 block 一项的 E4M3 device 输出。
 * @param global_scale 长度为 1 的 FP32 decode global scale，只读。
 * @param num_cols 输入矩阵列数。
 * @param blocks_per_row 每行的逻辑 NVFP4 block 数量。
 * @param total_blocks 整张矩阵的逻辑 block 总数。
 */
__global__ __launch_bounds__(kThreadsPerCta) void nvfp4LocalScaleKernel(
    const float* __restrict__ input,
    std::uint8_t* __restrict__ local_scales,
    const float* __restrict__ global_scale,
    const std::uint64_t num_cols,
    const std::uint64_t blocks_per_row,
    const std::uint64_t total_blocks) {
    const cg::thread_block cta = cg::this_thread_block();
    const cg::thread_block_tile<16> tile = cg::tiled_partition<16>(cta);
    const std::uint32_t lane_in_tile = tile.thread_rank();
    const std::uint64_t first_quant_block =
        static_cast<std::uint64_t>(blockIdx.x) * kNvfp4TilesPerCta +
        tile.meta_group_rank();
    const std::uint64_t quant_block_stride =
        static_cast<std::uint64_t>(gridDim.x) * kNvfp4TilesPerCta;
    const float tensor_global_scale = global_scale[0];

    for (std::uint64_t quant_block = first_quant_block;
         quant_block < total_blocks;
         quant_block += quant_block_stride) {
        const std::uint64_t row = quant_block / blocks_per_row;
        const std::uint64_t block_in_row = quant_block % blocks_per_row;
        const std::uint64_t first_column =
            block_in_row * static_cast<std::uint64_t>(kNvfp4BlockSize);
        const std::uint64_t valid_count = num_cols - first_column;
        const bool is_valid_lane = lane_in_tile < valid_count;
        const std::uint64_t linear_index =
            row * num_cols + first_column + lane_in_tile;
        const float value = is_valid_lane ? input[linear_index] : 0.0F;

        std::uint32_t has_nonfinite =
            is_valid_lane && !formats::fp32::is_finite(value) ? 1U : 0U;
        float block_amax = is_valid_lane
            ? formats::fp32::absolute_value(value)
            : 0.0F;

        // tile 是编译期固定的 16 lane 子组；无需手写 low/high half-warp mask。
        for (int offset = 8; offset > 0; offset /= 2) {
            const float neighbor_amax = tile.shfl_down(block_amax, offset);
            const std::uint32_t neighbor_nonfinite =
                tile.shfl_down(has_nonfinite, offset);
            block_amax = block_amax > neighbor_amax ? block_amax : neighbor_amax;
            has_nonfinite |= neighbor_nonfinite;
        }

        if (lane_in_tile == 0U) {
            local_scales[quant_block] = has_nonfinite != 0U
                ? formats::kE4M3CanonicalNaNCode
                : formats::compute_nvfp4_local_scale_code(
                      block_amax, tensor_global_scale);
        }
    }
}

/**
 * @brief 以一个完整 warp 处理连续 32 个元素，并由偶数 lane 独占 packed byte store。
 *
 * 每个 lane 先以自己所属 rowwise 16 元素 block 的 E4M3 scale 编码一个 E2M1
 * nibble。所有 32 lane 均执行 `shfl`，偶数 lane 取相邻奇数 lane 的 code 后写
 * 一个 byte；因此没有两个线程对同一 payload byte 做 read-modify-write。连续的
 * 32 个线性元素映射还覆盖了奇数列矩阵的跨行 byte 配对。
 *
 * @param input 连续 row-major FP32 输入的 device 指针，只读。
 * @param payload packed NVFP4 device 输出，每字节两个 E2M1 nibble。
 * @param local_scales rowwise E4M3 local scale 数组，只读。
 * @param global_scale 长度为 1 的 FP32 decode global scale，只读。
 * @param num_cols 输入矩阵列数。
 * @param blocks_per_row 每行的 NVFP4 logical block 数量。
 * @param element_count 逻辑输入元素总数。
 * @param rounding E2M1 元素编码使用的舍入模式。
 * @param stochastic_seed stochastic rounding 的确定性种子。
 */
__global__ __launch_bounds__(kThreadsPerCta) void nvfp4PackedEncodeKernel(
    const float* __restrict__ input,
    std::uint8_t* __restrict__ payload,
    const std::uint8_t* __restrict__ local_scales,
    const float* __restrict__ global_scale,
    const std::uint64_t num_cols,
    const std::uint64_t blocks_per_row,
    const std::uint64_t element_count,
    const RoundingMode rounding,
    const std::uint64_t stochastic_seed) {
    const cg::thread_block cta = cg::this_thread_block();
    const cg::thread_block_tile<32> warp = cg::tiled_partition<32>(cta);
    const std::uint32_t lane_in_warp = warp.thread_rank();
    const float tensor_global_scale = global_scale[0];

    if (!formats::fp32::is_finite(tensor_global_scale) ||
        tensor_global_scale <= 0.0F) {
        return;
    }

    const std::uint64_t first_warp =
        static_cast<std::uint64_t>(blockIdx.x) * kWarpsPerCta +
        warp.meta_group_rank();
    const std::uint64_t warp_stride =
        static_cast<std::uint64_t>(gridDim.x) * kWarpsPerCta;

    for (std::uint64_t logical_warp = first_warp;; logical_warp += warp_stride) {
        const std::uint64_t first_linear_index =
            logical_warp * static_cast<std::uint64_t>(kWarpSize);
        if (first_linear_index >= element_count) {
            break;
        }

        const std::uint64_t linear_index = first_linear_index + lane_in_warp;
        const bool is_valid_lane = linear_index < element_count;
        std::uint8_t own_code = 0x00U;
        if (is_valid_lane) {
            const std::uint64_t row = linear_index / num_cols;
            const std::uint64_t column = linear_index % num_cols;
            const std::uint64_t scale_index =
                row * blocks_per_row +
                column / static_cast<std::uint64_t>(kNvfp4BlockSize);
            const float uniform_random = rounding == RoundingMode::kStochastic
                ? formats::nvfp4_stochastic_uniform_for_element(
                      stochastic_seed, linear_index)
                : 0.0F;
            own_code = formats::encode_nvfp4_element(
                input[linear_index],
                local_scales[scale_index],
                tensor_global_scale,
                rounding,
                uniform_random);
        }

        // shfl 是一个 tile 集体操作：包括不写 payload 的奇数 lane 在内，全部
        // 线程都必须参与。奇数 lane 取自身只是为了给它一个合法 source rank；
        // 只有偶数 lane 使用从 lane+1 取得的结果并执行最终 packed store。
        const std::uint32_t partner_lane = (lane_in_warp & 1U) == 0U
            ? lane_in_warp + 1U
            : lane_in_warp;
        const std::uint8_t partner_code = warp.shfl(
            own_code, static_cast<int>(partner_lane));

        if ((lane_in_warp & 1U) == 0U) {
            payload[linear_index / 2U] = formats::pack_e2m1_nibbles(
                own_code, partner_code);
        }
    }
}

/**
 * @brief 验证 NVFP4 launcher 的输入、输出、工作区与配置完全匹配。
 *
 * @param input 已完成 H2D 的 device 输入。
 * @param config NVFP4 量化配置。
 * @param output 待写 device 量化结果。
 * @param workspace 全局 amax 规约工作区。
 * @throws CudaPipelineError 任一结构、格式或长度约束不满足时抛出。
 */
void validate_nvfp4_launch(
    const pipeline::DeviceQuantizationInput& input,
    const QuantizationConfig& config,
    const pipeline::DeviceQuantizedTensor& output,
    const pipeline::DeviceTensorQuantizationWorkspace& workspace) {
    if (!input.isConsistent() || !output.isConsistent() || !workspace.isConsistent()) {
        throw CudaPipelineError{
            "nvfp4 CUDA launcher 收到了不自洽的 device buffer 或规约工作区。"};
    }

    if (!config.isValid() || config.format != QuantFormat::kNvfp4 ||
        config.scale_mode != ScaleMode::kBlock ||
        config.block_size != kNvfp4BlockSize ||
        output.desc.format != QuantFormat::kNvfp4 ||
        output.desc.scale_mode != ScaleMode::kBlock ||
        output.desc.block_size != kNvfp4BlockSize ||
        output.desc.rounding != config.rounding ||
        output.desc.stochastic_seed != config.stochastic_seed ||
        output.desc.source_desc.num_rows != input.desc.num_rows ||
        output.desc.source_desc.num_cols != input.desc.num_cols ||
        output.desc.source_desc.dtype != input.desc.dtype ||
        !output.global_scale.has_value() || output.global_scale->size() != 1U) {
        throw CudaPipelineError{
            "nvfp4 CUDA launcher 收到了不匹配的输入、输出或 block-scale 配置。"};
    }
}

/**
 * @brief 查询 NVFP4 local-scale persistent kernel 的实际可驻留 CTA 数量。
 *
 * @return 以 CTA 为单位的 `SM 数 × min(4, occupancy)` 网格上限。
 * @throws CudaPipelineError 当前 device 或 occupancy 查询失败时抛出。
 */
[[nodiscard]] std::uint64_t resident_local_scale_cta_count() {
    int device_id = 0;
    check_cuda(cudaGetDevice(&device_id), "local-scale 查询当前 device");

    cudaDeviceProp properties{};
    check_cuda(cudaGetDeviceProperties(&properties, device_id),
               "local-scale 查询 device 属性");
    if (properties.multiProcessorCount <= 0) {
        throw CudaPipelineError{
            "nvfp4 CUDA local-scale launcher 未发现可用 SM。"};
    }

    int max_active_ctas_per_sm = 0;
    check_cuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                   &max_active_ctas_per_sm,
                   nvfp4LocalScaleKernel,
                   kThreadsPerCta,
                   0U),
               "查询 local-scale kernel occupancy");
    if (max_active_ctas_per_sm <= 0) {
        throw CudaPipelineError{
            "nvfp4 CUDA local-scale kernel 无法在当前 device 上驻留。"};
    }

    const int ctas_per_sm = max_active_ctas_per_sm < kTargetCtasPerSm
        ? max_active_ctas_per_sm
        : kTargetCtasPerSm;
    return static_cast<std::uint64_t>(properties.multiProcessorCount) *
           static_cast<std::uint64_t>(ctas_per_sm);
}

}  // namespace

std::size_t nvfp4_tensor_partial_count() {
    int device_id = 0;
    check_cuda(cudaGetDevice(&device_id), "global-amax 查询当前 device");

    cudaDeviceProp properties{};
    check_cuda(cudaGetDeviceProperties(&properties, device_id),
               "global-amax 查询 device 属性");
    if (properties.multiProcessorCount <= 0 || properties.maxGridSize[0] <= 0) {
        throw CudaPipelineError{
            "nvfp4 CUDA global-amax 未发现有效的 SM 或 grid.x 上限。"};
    }

    const std::uint64_t desired_count =
        static_cast<std::uint64_t>(properties.multiProcessorCount) *
        static_cast<std::uint64_t>(kTargetCtasPerSm);
    const std::uint64_t max_grid_x =
        static_cast<std::uint64_t>(properties.maxGridSize[0]);
    const std::uint64_t partial_count = desired_count < max_grid_x
        ? desired_count
        : max_grid_x;
    if (partial_count == 0U ||
        partial_count >
            static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
        throw CudaPipelineError{
            "nvfp4 CUDA global-amax 无法构造可表示的 partial 数量。"};
    }

    return static_cast<std::size_t>(partial_count);
}

void launch_nvfp4_block_quantize(
    const pipeline::DeviceQuantizationInput& input,
    const QuantizationConfig& config,
    pipeline::DeviceQuantizedTensor& output,
    pipeline::DeviceTensorQuantizationWorkspace& workspace,
    const cudaStream_t stream) {
    validate_nvfp4_launch(input, config, output, workspace);

    const auto element_count = input.desc.elementCount();
    const auto blocks_per_row = output.desc.blocksPerRow();
    if (!element_count.has_value() || !blocks_per_row.has_value() ||
        *element_count == 0U || *blocks_per_row == 0U ||
        output.local_scales.empty()) {
        throw CudaPipelineError{
            "nvfp4 CUDA launcher 无法推导非空输入和 rowwise local scale 布局。"};
    }

    const std::size_t partial_count = nvfp4_tensor_partial_count();
    if (workspace.partial_amax.size() != partial_count) {
        throw CudaPipelineError{
            "nvfp4 CUDA launcher 的 global-amax 工作区长度与执行网格不一致。"};
    }

    const std::uint64_t total_blocks =
        static_cast<std::uint64_t>(output.local_scales.size());
    const std::uint64_t logical_local_scale_ctas =
        total_blocks / static_cast<std::uint64_t>(kNvfp4TilesPerCta) +
        (total_blocks % static_cast<std::uint64_t>(kNvfp4TilesPerCta) == 0U
             ? 0U
             : 1U);
    const std::uint64_t resident_ctas = resident_local_scale_cta_count();
    const std::uint64_t local_scale_ctas = logical_local_scale_ctas < resident_ctas
        ? logical_local_scale_ctas
        : resident_ctas;
    if (local_scale_ctas == 0U ||
        local_scale_ctas > std::numeric_limits<unsigned int>::max() ||
        partial_count > std::numeric_limits<unsigned int>::max()) {
        throw CudaPipelineError{
            "nvfp4 CUDA launcher 无法构造可表示的 persistent grid。"};
    }

    const dim3 reduction_grid{static_cast<unsigned int>(partial_count), 1U, 1U};
    const dim3 local_scale_grid{
        static_cast<unsigned int>(local_scale_ctas), 1U, 1U};
    const dim3 cta_block{static_cast<unsigned int>(kThreadsPerCta), 1U, 1U};

    const auto* const input_ptr = thrust::raw_pointer_cast(input.values.data());
    auto* const payload_ptr = thrust::raw_pointer_cast(output.payload.data());
    auto* const local_scales_ptr =
        thrust::raw_pointer_cast(output.local_scales.data());
    auto* const global_scale_ptr =
        thrust::raw_pointer_cast(output.global_scale->data());
    auto* const partial_amax_ptr =
        thrust::raw_pointer_cast(workspace.partial_amax.data());
    auto* const partial_nonfinite_ptr =
        thrust::raw_pointer_cast(workspace.partial_nonfinite.data());

    // 第一、二阶段由 default stream 顺序保证：finalize 读取所有 partial，随后
    // local-scale 与 packed-encode kernel 都读取已经写好的 global_scale[0]。
    nvfp4TensorAmaxPartialKernel<<<reduction_grid, cta_block, 0U, stream>>>(
        input_ptr, partial_amax_ptr, partial_nonfinite_ptr, *element_count);
    check_cuda(cudaGetLastError(), "发射 global-amax partial kernel");

    nvfp4GlobalScaleFinalizeKernel<<<1U, cta_block, 0U, stream>>>(
        partial_amax_ptr,
        partial_nonfinite_ptr,
        global_scale_ptr,
        static_cast<std::uint64_t>(partial_count));
    check_cuda(cudaGetLastError(), "发射 global-scale finalize kernel");

    nvfp4LocalScaleKernel<<<local_scale_grid, cta_block, 0U, stream>>>(
        input_ptr,
        local_scales_ptr,
        global_scale_ptr,
        input.desc.num_cols,
        *blocks_per_row,
        total_blocks);
    check_cuda(cudaGetLastError(), "发射 local-scale tile kernel");

    // packed encode 复用全局规约的 SM×4 grid。每 warp 对应连续 32 个元素，
    // 相邻 lane 的两个 E2M1 nibble 最终仅由偶数 lane 写入一个 uint8_t。
    nvfp4PackedEncodeKernel<<<reduction_grid, cta_block, 0U, stream>>>(
        input_ptr,
        payload_ptr,
        local_scales_ptr,
        global_scale_ptr,
        input.desc.num_cols,
        *blocks_per_row,
        *element_count,
        config.rounding,
        config.stochastic_seed);
    check_cuda(cudaGetLastError(), "发射 packed E2M1 encode kernel");
}

}  // namespace quant_dequant::cuda
