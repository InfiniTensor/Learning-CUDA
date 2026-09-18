#include "cuda/mxfp8_quantize.cuh"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>

#include <thrust/device_ptr.h>

#include "formats/mxfp8_codec.cuh"
#include "quant_dequant/quantize.hpp"
#include "pipeline/device_quantized_tensor.cuh"

namespace quant_dequant::cuda {
namespace {

/** 单个 CUDA warp 中的固定线程数量。 */
inline constexpr int kWarpSize = 32;

/** 一个 CTA 同时处理的独立 MXFP8 量化 block 数量。 */
inline constexpr int kWarpsPerCta = 8;

/** block-scale kernel 的固定 CTA 线程数量。 */
inline constexpr int kThreadsPerCta = kWarpSize * kWarpsPerCta;

/** 每个 SM 的目标常驻 CTA 数量；实际值还受 occupancy 上限约束。 */
inline constexpr int kTargetCtasPerSm = 4;

static_assert(kMxfp8BlockSize == static_cast<std::uint32_t>(kWarpSize));
static_assert(kThreadsPerCta % kWarpSize == 0);

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
            "mxfp8 CUDA " + std::string{operation} +
            " 失败：" + cudaGetErrorString(status)};
    }
}

/**
 * @brief 检查 block-scale launcher 收到的 device 对象和配置是否彼此匹配。
 *
 * @param input 只读 device FP32 输入。
 * @param config 调用方传入的量化配置。
 * @param output 待写入的 MXFP8 device 输出。
 * @throws CudaPipelineError 任一对象不自洽或元数据不匹配时抛出。
 */
void validate_mxfp8_block_launch(
    const pipeline::DeviceQuantizationInput& input,
    const QuantizationConfig& config,
    const pipeline::DeviceQuantizedTensor& output) {
    if (!input.isConsistent() || !output.isConsistent()) {
        throw CudaPipelineError{
            "mxfp8 CUDA block-scale launcher 收到了不自洽的 device buffer。"};
    }

    if (!config.isValid() || config.format != QuantFormat::kMxfp8 ||
        config.scale_mode != ScaleMode::kBlock ||
        config.block_size != kMxfp8BlockSize ||
        output.desc.format != QuantFormat::kMxfp8 ||
        output.desc.scale_mode != ScaleMode::kBlock ||
        output.desc.block_size != kMxfp8BlockSize ||
        output.desc.rounding != config.rounding ||
        output.desc.stochastic_seed != config.stochastic_seed ||
        output.desc.source_desc.num_rows != input.desc.num_rows ||
        output.desc.source_desc.num_cols != input.desc.num_cols ||
        output.desc.source_desc.dtype != input.desc.dtype) {
        throw CudaPipelineError{
            "mxfp8 CUDA block-scale launcher 收到了不匹配的输入、输出或配置。"};
    }
}

/**
 * @brief 保存一个规约线程当前看到的有限最大绝对值和非有限输入标记。
 *
 * 单纯对 float 做最大值规约可能静默忽略 NaN；该状态把数值结果和 NaN/Inf
 * 存在性一起向上规约，确保 tensor-scale 与 block-scale 都能拒绝非有限输入。
 */
struct TensorAmaxState {
    /** 当前线程或子组内所有有限元素的最大绝对值。 */
    float amax{0.0F};

    /** 只要看到任意 NaN、+Inf 或 -Inf 就为非 0。 */
    std::uint32_t has_nonfinite{0U};
};

/**
 * @brief 将一个 FP32 输入值并入 tensor amax 规约状态。
 *
 * @param state 当前线程持有的局部规约状态，原地更新。
 * @param value 本线程通过标量或 float4 读取到的一个输入元素。
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
 * @brief 在一个 warp 内规约 amax 与非有限输入标记。
 *
 * 所有调用 lane 必须处于活动状态。`amax` 使用最大值，`has_nonfinite` 使用
 * 逻辑或；因此 lane 0 返回整个 warp 的完整状态。
 *
 * @param state 当前 lane 的局部规约状态。
 * @return 当前 lane 对应的 shuffle 规约中间状态；lane 0 为完整 warp 结果。
 */
__device__ __forceinline__ TensorAmaxState warp_reduce_tensor_amax(
    TensorAmaxState state) {
    constexpr unsigned int kAllWarpLanes = 0xffffffffU;

    for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
        const float neighbor_amax =
            __shfl_down_sync(kAllWarpLanes, state.amax, offset);
        const std::uint32_t neighbor_nonfinite = __shfl_down_sync(
            kAllWarpLanes, state.has_nonfinite, offset);
        state.amax = state.amax > neighbor_amax ? state.amax : neighbor_amax;
        state.has_nonfinite |= neighbor_nonfinite;
    }

    return state;
}

/**
 * @brief 组合 warp shuffle 和少量 shared memory，规约一个 256 线程 CTA。
 *
 * 每个 warp 先通过 shuffle 产生一项结果，由 lane 0 写入 shared memory；CTA
 * 的第一个 warp 再读取至多 8 项并做第二次 warp 规约。该结构直接借鉴给定 V7
 * reduction 的“warp reduce -> 每 warp 一项 shared -> warp 0 finalize”模式，
 * 但把求和替换为 amax 与非有限标记的联合规约。
 *
 * @param state 当前线程的局部规约状态。
 * @param shared_amax 长度至少为 8 的 shared 数组，保存每个 warp 的 amax。
 * @param shared_nonfinite 长度至少为 8 的 shared 数组，保存每个 warp 的标记。
 * @return thread 0 返回完整 CTA 结果；其他线程的返回值不应被使用。
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

    // warp 0 只有在所有 warp 的 lane 0 均写完 partial 后才能读取 shared memory。
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
 * @brief 第一阶段：使用 V7 风格 float4 grid-stride 加载生成每 CTA 的 amax partial。
 *
 * 每个 CTA 以 grid-stride 遍历多个连续 float4，寄存器中累计局部状态；随后采用
 * “warp shuffle + 8 项 shared memory + warp 0”规约，仅由 thread 0 写一项
 * `partial_amax[blockIdx.x]` 与一项 `partial_nonfinite[blockIdx.x]`。grid.x 固定
 * 为 `SM 数 * 4`，大输入由每个 CTA 的 grid-stride 循环继续处理。
 *
 * @param input device 指针，指向连续 row-major FP32 输入，只读且至少 4-byte 对齐。
 * @param partial_amax device 指针，长度为 grid.x，每 CTA 写一项有限最大绝对值。
 * @param partial_nonfinite device 指针，长度为 grid.x，每 CTA 写一项非有限标记。
 * @param element_count 输入逻辑元素数。
 */
__global__ __launch_bounds__(kThreadsPerCta) void mxfp8TensorAmaxPartialKernel(
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

    // float4 加载与 V7 一样让相邻线程读取相邻的 16-byte 向量；对于连续 FP32
    // device_vector，这同时减少循环次数并保持 warp 内合并全局内存访问。
    for (std::uint64_t vector_index = first_vector_index;
         vector_index < vector_count;
         vector_index += vector_stride) {
        const float4 values = input4[vector_index];
        accumulate_tensor_amax(state, values.x);
        accumulate_tensor_amax(state, values.y);
        accumulate_tensor_amax(state, values.z);
        accumulate_tensor_amax(state, values.w);
    }

    // vector_count * 4 之后最多还有 3 个元素。它们仍以与 V7 相同的 grid-stride
    // 映射由全部 CTA/线程共同处理，避免越过数组末尾的 float4 读取。
    const std::uint64_t tail_start = vector_count * 4U;
    for (std::uint64_t linear_index = tail_start + first_vector_index;
         linear_index < element_count;
         linear_index += vector_stride) {
        accumulate_tensor_amax(state, input[linear_index]);
    }

    state = block_reduce_tensor_amax(
        state, shared_amax, shared_nonfinite);
    if (threadIdx.x == 0U) {
        partial_amax[blockIdx.x] = state.amax;
        partial_nonfinite[blockIdx.x] = state.has_nonfinite;
    }
}

/**
 * @brief 第二阶段：单个 CTA 规约所有 partial，并直接写唯一的 E8M0 tensor scale。
 *
 * 每个线程用 block-stride 遍历若干 partial，因此即使第一阶段 CTA 数量大于 256，
 * 也无需第三层 reduction。完成 CTA 规约后 thread 0 直接调用
 * `compute_mxfp8_scale_code()` 写 `local_scales[0]`；这就是 tensor mode 省去
 * 独立 scale-finalize kernel 的关键。
 *
 * @param partial_amax 第一阶段输出的有限最大绝对值数组，只读。
 * @param partial_nonfinite 第一阶段输出的非有限标记数组，只读。
 * @param local_scales device 指针，长度为 1；thread 0 写 E8M0 scale 或 0xff 哨兵。
 * @param partial_count 第一阶段 CTA 与 partial 数量。
 */
__global__ __launch_bounds__(kThreadsPerCta) void mxfp8TensorScaleFinalizeKernel(
    const float* __restrict__ partial_amax,
    const std::uint32_t* __restrict__ partial_nonfinite,
    std::uint8_t* __restrict__ local_scales,
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

    state = block_reduce_tensor_amax(
        state, shared_amax, shared_nonfinite);
    if (threadIdx.x == 0U) {
        local_scales[0] = state.has_nonfinite != 0U
                              ? formats::kE8M0NaNCode
                              : formats::compute_mxfp8_scale_code(state.amax);
    }
}

/**
 * @brief 使用唯一的 tensor E8M0 scale 编码所有 row-major FP32 元素。
 *
 * scale-finalize kernel 与本 kernel 是同一 default stream 上顺序发射的不同
 * kernel，因此所有 CTA 都能安全读取已经写完的 `local_scales[0]`。若上一阶段
 * 发现 NaN/Inf 而写入 0xff，本 kernel 不写 payload，D2H 会据此拒绝结果。
 *
 * @param input device 指针，指向连续 row-major FP32 输入，只读。
 * @param payload device 指针，指向连续 E4M3 输出 payload，写入。
 * @param local_scales device 指针，长度为 1，读取唯一 E8M0 tensor scale。
 * @param element_count 输入和输出逻辑元素数。
 * @param rounding E4M3 编码时使用的舍入模式。
 * @param stochastic_seed stochastic rounding 的确定性种子。
 */
__global__ __launch_bounds__(kThreadsPerCta) void mxfp8TensorEncodeKernel(
    const float* __restrict__ input,
    std::uint8_t* __restrict__ payload,
    const std::uint8_t* __restrict__ local_scales,
    const std::uint64_t element_count,
    const RoundingMode rounding,
    const std::uint64_t stochastic_seed) {
    const std::uint8_t scale_code = local_scales[0];
    if (formats::is_e8m0_nan(scale_code)) {
        return;
    }

    const std::uint64_t first_linear_index =
        static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::uint64_t linear_stride =
        static_cast<std::uint64_t>(gridDim.x) * blockDim.x;

    for (std::uint64_t linear_index = first_linear_index;
         linear_index < element_count;
         linear_index += linear_stride) {
        const float uniform_random =
            rounding == RoundingMode::kStochastic
                ? formats::mxfp8_stochastic_uniform_for_element(
                      stochastic_seed, linear_index)
                : 0.0F;
        payload[linear_index] = formats::encode_mxfp8_element(
            input[linear_index], scale_code, rounding, uniform_random);
    }
}

/**
 * @brief 以固定 CTA 网格持久处理 MXFP8 rowwise 32 元素量化 block。
 *
 * 一个 warp 对应一个逻辑量化 block：lane 0--31 分别加载本 block 的一个元素，
 * 通过 warp shuffle 规约 `amax`，由 lane 0 写 E8M0 scale，再编码各自的 E4M3
 * payload。一个 CTA 放置 8 个互不依赖的 warp；CTA 数量固定为不超过
 * `SM 数 * 目标 CTA/SM` 的值。每个 warp 随后以 grid-stride 继续处理更多量化
 * block，因此大矩阵不需要发射与量化 block 数相同数量的 CTA。
 *
 * 尾 block 中超过实际列数的 lane 不访问内存、以 0 参与 amax 规约。发现 NaN
 * 或 Inf 时，lane 0 写 E8M0 NaN code (`0xff`) 作为错误哨兵；后续 D2H 阶段必须
 * 扫描该哨兵并按与 CPU reference 一致的规则报告错误。
 *
 * @param input device 指针，指向 row-major FP32 输入，只读。
 * @param payload device 指针，指向 row-major E4M3 输出 payload，写入。
 * @param local_scales device 指针，指向 rowwise E8M0 scale 数组，写入。
 * @param num_cols 输入矩阵列数。
 * @param blocks_per_row 每行的逻辑 32 元素量化 block 数。
 * @param total_blocks 全矩阵的逻辑量化 block 总数。
 * @param rounding E4M3 编码时使用的舍入模式。
 * @param stochastic_seed stochastic rounding 的确定性种子。
 */
__global__ __launch_bounds__(kThreadsPerCta) void mxfp8BlockQuantizeKernel(
    const float* __restrict__ input,
    std::uint8_t* __restrict__ payload,
    std::uint8_t* __restrict__ local_scales,
    const std::uint64_t num_cols,
    const std::uint64_t blocks_per_row,
    const std::uint64_t total_blocks,
    const RoundingMode rounding,
    const std::uint64_t stochastic_seed) {
    constexpr unsigned int kAllWarpLanes = 0xffffffffU;

    const std::uint32_t lane_id = threadIdx.x % kWarpSize;
    const std::uint32_t warp_id = threadIdx.x / kWarpSize;
    const std::uint64_t first_block =
        static_cast<std::uint64_t>(blockIdx.x) * kWarpsPerCta + warp_id;
    const std::uint64_t block_stride =
        static_cast<std::uint64_t>(gridDim.x) * kWarpsPerCta;

    // 固定数量 CTA 常驻时，每个 warp 以 grid-stride 反复领取同构的 32 元素
    // 工作项。这里无需 atomic work queue：每个量化 block 工作量相同，静态分配
    // 没有原子竞争且可保持相邻 lane 的合并访存。
    for (std::uint64_t quant_block = first_block;
         quant_block < total_blocks;
         quant_block += block_stride) {
        const std::uint64_t row = quant_block / blocks_per_row;
        const std::uint64_t block_in_row = quant_block % blocks_per_row;
        const std::uint64_t first_column =
            block_in_row * static_cast<std::uint64_t>(kMxfp8BlockSize);
        const std::uint64_t valid_element_count = num_cols - first_column;
        const bool is_valid_lane = lane_id < valid_element_count;
        const std::uint64_t linear_index =
            row * num_cols + first_column + lane_id;
        const float value = is_valid_lane ? input[linear_index] : 0.0F;

        // 所有 lane 都必须参与后续 shuffle。尾 block 的无效 lane 只贡献 +0，
        // 因此不会改变 amax，也绝不访问最后一个真实元素之后的地址。
        const unsigned int nonfinite_lanes = __ballot_sync(
            kAllWarpLanes,
            is_valid_lane && !formats::fp32::is_finite(value));
        if (nonfinite_lanes != 0U) {
            if (lane_id == 0U) {
                local_scales[quant_block] = formats::kE8M0NaNCode;
            }

            // 所有 lane 遵循相同条件分支，故可以安全继续处理下一个逻辑 block。
            // payload 不再有数值意义；D2H 会凭 0xff scale 哨兵拒绝整个量化结果。
            continue;
        }

        float warp_amax =
            is_valid_lane ? formats::fp32::absolute_value(value) : 0.0F;
        for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
            const float neighbor_amax =
                __shfl_down_sync(kAllWarpLanes, warp_amax, offset);
            warp_amax = warp_amax > neighbor_amax ? warp_amax : neighbor_amax;
        }

        std::uint8_t scale_code = 0x00U;
        if (lane_id == 0U) {
            scale_code = formats::compute_mxfp8_scale_code(warp_amax);
            local_scales[quant_block] = scale_code;
        }

        // lane 0 的 E8M0 code 是本逻辑量化 block 的唯一 scale；广播后每个
        // lane 都用它把自己的 FP32 value 编码为 E4M3。
        scale_code = static_cast<std::uint8_t>(__shfl_sync(
            kAllWarpLanes, static_cast<int>(scale_code), 0));

        if (is_valid_lane) {
            const float uniform_random =
                rounding == RoundingMode::kStochastic
                    ? formats::mxfp8_stochastic_uniform_for_element(
                          stochastic_seed, linear_index)
                    : 0.0F;
            payload[linear_index] = formats::encode_mxfp8_element(
                value, scale_code, rounding, uniform_random);
        }
    }
}

}  // namespace

void launch_mxfp8_block_quantize(
    const pipeline::DeviceQuantizationInput& input,
    const QuantizationConfig& config,
    pipeline::DeviceQuantizedTensor& output,
    const cudaStream_t stream) {
    // launcher 是 CUDA 后端的最后一道边界。即使某个内部调用绕开了 pipeline，
    // 也不能让错误格式、错误 scale_mode 或不匹配的 device buffer 进入 kernel。
    validate_mxfp8_block_launch(input, config, output);

    // block mode 中一行被切为 ceil(num_cols / 32) 个逻辑量化 block；每个
    // 量化 block 恰好对应 output.local_scales 中的一项 E8M0 scale。
    const auto blocks_per_row = output.desc.blocksPerRow();
    if (!blocks_per_row.has_value() || *blocks_per_row == 0U) {
        throw CudaPipelineError{
            "mxfp8 CUDA block-scale launcher 无法推导每行量化 block 数量。"};
    }

    // QuantizedTensorDesc 的一致性校验已保证：
    // total_blocks == num_rows * blocks_per_row。
    // 这里直接使用 device scale 数组长度，确保 kernel 写入范围与实际分配内存
    // 完全相同，而不是重新计算一份可能与输出描述脱节的总数。
    const std::uint64_t total_blocks =
        static_cast<std::uint64_t>(output.local_scales.size());
    if (total_blocks == 0U) {
        throw CudaPipelineError{
            "mxfp8 CUDA block-scale launcher 收到了空的 local scale 数组。"};
    }

    // DeviceQuantizationInput / DeviceQuantizedTensor 在“当前 CUDA device”上分配；
    // 必须查询同一个 device，才能得到与这些 buffer、kernel occupancy 对应的 SM
    // 数和资源上限。device_id 只是 CUDA runtime 中当前选中 GPU 的编号。
    int device_id = 0;
    check_cuda(cudaGetDevice(&device_id), "查询当前 device");

    // multiProcessorCount 就是当前 GPU 的 SM 数。不要在代码里写死 RTX 4060 的
    // SM 数；同一套 launcher 在其他 GPU 上应根据实际硬件自动调整网格。
    cudaDeviceProp device_properties{};
    check_cuda(cudaGetDeviceProperties(&device_properties, device_id),
               "查询 device 属性");
    if (device_properties.multiProcessorCount <= 0) {
        throw CudaPipelineError{
            "mxfp8 CUDA block-scale launcher 未发现可用 SM。"};
    }

    // `SM * 4` 是调度启发式，而非硬编码保证。实际 active CTA/SM 受每线程
    // register 数、每 CTA shared memory、CTA 线程数和架构上限影响。因此先询问
    // occupancy API：本 kernel、256 线程、0 字节动态 shared memory 时，一个 SM
    // 最多能同时驻留多少个 CTA。
    int max_active_ctas_per_sm = 0;
    check_cuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                   &max_active_ctas_per_sm,
                   mxfp8BlockQuantizeKernel,
                   kThreadsPerCta,
                   0U),
               "查询 kernel occupancy");
    if (max_active_ctas_per_sm <= 0) {
        throw CudaPipelineError{
            "mxfp8 CUDA block-scale kernel 无法在当前 device 上驻留。"};
    }

    // 目标是每个 SM 最多保留 4 个 CTA；若 occupancy 表明最多只能放 2 个，则
    // 必须降为 2，不能因为持久化调度而超出硬件资源限制。
    const int ctas_per_sm =
        max_active_ctas_per_sm < kTargetCtasPerSm
            ? max_active_ctas_per_sm
            : kTargetCtasPerSm;

    // resident_cta_count 是期望同时驻留的 CTA 总数，即：
    //
    //     SM 数 * 每个 SM 实际允许的目标 CTA 数。
    //
    // 例如 24 个 SM、每 SM 4 CTA 时为 96。它是持久化网格的“硬件侧上限”，
    // 而不是矩阵逻辑工作量。
    const std::uint64_t resident_cta_count =
        static_cast<std::uint64_t>(device_properties.multiProcessorCount) *
        static_cast<std::uint64_t>(ctas_per_sm);

    // 一个 CTA 固定有 kWarpsPerCta == 8 个 warp，且每个 warp 的首个工作项是
    // 一个逻辑量化 block。因此首轮覆盖全部 total_blocks 至少需要：
    //
    //     logical_cta_count = ceil(total_blocks / 8)
    //
    // 个 CTA。使用除法和余数写 ceil，避免 total_blocks + 7 的潜在 uint64 溢出。
    const std::uint64_t logical_cta_count =
        total_blocks / static_cast<std::uint64_t>(kWarpsPerCta) +
        (total_blocks % static_cast<std::uint64_t>(kWarpsPerCta) == 0U ? 0U : 1U);

    // 小矩阵不应该为了凑满 SM * 4 启动大量空 CTA；大矩阵则只启动可常驻的
    // resident_cta_count 个 CTA，剩余工作由 kernel 内的 grid-stride 循环领取。
    const std::uint64_t launch_cta_count =
        logical_cta_count < resident_cta_count
            ? logical_cta_count
            : resident_cta_count;

    // CUDA 的 dim3 名称很容易和“量化 block”混淆：
    // - grid.x：本次发射多少个 CUDA CTA；
    // - block.x：每个 CUDA CTA 的线程数，即 8 warp * 32 = 256；
    // - 一个 warp 才对应一个 MXFP8 的 32 元素逻辑量化 block。
    const dim3 grid{static_cast<unsigned int>(launch_cta_count), 1U, 1U};
    const dim3 block{static_cast<unsigned int>(kThreadsPerCta), 1U, 1U};

    // thrust::device_vector 是 host 侧的 RAII 所有者，kernel 不能按值接收它。
    // raw_pointer_cast 只取出 non-owning 的 raw device pointer，不触发 H2D/D2H；
    // H2D 已在 pipeline 构造 DeviceQuantizationInput 时完成。
    const auto* const input_ptr = thrust::raw_pointer_cast(input.values.data());
    auto* const payload_ptr = thrust::raw_pointer_cast(output.payload.data());
    auto* const local_scales_ptr =
        thrust::raw_pointer_cast(output.local_scales.data());

    // 将小型 metadata 与三个 device pointer 写入 kernel 参数区后异步发射。
    // kernel 内第一个任务编号为：
    //
    //     quant_block = blockIdx.x * 8 + warp_id
    //
    // 每处理完一个量化 block，warp 再加 gridDim.x * 8，从而持续覆盖剩余工作。
    mxfp8BlockQuantizeKernel<<<grid, block, 0U, stream>>>(
        input_ptr,
        payload_ptr,
        local_scales_ptr,
        input.desc.num_cols,
        *blocks_per_row,
        total_blocks,
        config.rounding,
        config.stochastic_seed);

    // cudaGetLastError 只检查“发射”本身，例如 grid/block 维度和参数是否有效；
    // 它不等待 kernel 完成。真正的异步执行错误将在后续 D2H 或显式同步时出现。
    check_cuda(cudaGetLastError(), "发射 persistent quantize kernel");
}

std::size_t mxfp8_tensor_partial_count() {
    // V7 reduction 采用 SM * 4 的固定 grid；它使每个 CTA 通过 grid-stride
    // 覆盖更多输入，且 partial 数量足够小，第二阶段一个 CTA 即可完成最终规约。
    int device_id = 0;
    check_cuda(cudaGetDevice(&device_id), "tensor-scale 查询当前 device");

    cudaDeviceProp device_properties{};
    check_cuda(cudaGetDeviceProperties(&device_properties, device_id),
               "tensor-scale 查询 device 属性");
    if (device_properties.multiProcessorCount <= 0 ||
        device_properties.maxGridSize[0] <= 0) {
        throw CudaPipelineError{
            "mxfp8 CUDA tensor-scale 未发现有效的 SM 或 grid.x 上限。"};
    }

    const std::uint64_t desired_count =
        static_cast<std::uint64_t>(device_properties.multiProcessorCount) *
        static_cast<std::uint64_t>(kTargetCtasPerSm);
    const std::uint64_t max_grid_x =
        static_cast<std::uint64_t>(device_properties.maxGridSize[0]);
    const std::uint64_t partial_count =
        desired_count < max_grid_x ? desired_count : max_grid_x;
    if (partial_count == 0U ||
        partial_count >
            static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
        throw CudaPipelineError{
            "mxfp8 CUDA tensor-scale 无法构造可表示的 partial 规约数量。"};
    }

    return static_cast<std::size_t>(partial_count);
}

void launch_mxfp8_tensor_quantize(
    const pipeline::DeviceQuantizationInput& input,
    const QuantizationConfig& config,
    pipeline::DeviceQuantizedTensor& output,
    pipeline::DeviceTensorQuantizationWorkspace& workspace,
    const cudaStream_t stream) {
    // 与 block-scale launcher 相同，格式专用边界必须独立验证。tensor mode 的
    // local_scales 长度应恰好为 1；workspace 的两条数组必须等长并且与第一阶段
    // persistent grid.x 一一对应。
    if (!input.isConsistent() || !output.isConsistent() ||
        !workspace.isConsistent()) {
        throw CudaPipelineError{
            "mxfp8 CUDA tensor-scale launcher 收到了不自洽的 device buffer。"};
    }

    if (!config.isValid() || config.format != QuantFormat::kMxfp8 ||
        config.scale_mode != ScaleMode::kTensor ||
        config.block_size != kMxfp8BlockSize ||
        output.desc.format != QuantFormat::kMxfp8 ||
        output.desc.scale_mode != ScaleMode::kTensor ||
        output.desc.block_size != kMxfp8BlockSize ||
        output.desc.rounding != config.rounding ||
        output.desc.stochastic_seed != config.stochastic_seed ||
        output.desc.source_desc.num_rows != input.desc.num_rows ||
        output.desc.source_desc.num_cols != input.desc.num_cols ||
        output.desc.source_desc.dtype != input.desc.dtype ||
        output.local_scales.size() != 1U) {
        throw CudaPipelineError{
            "mxfp8 CUDA tensor-scale launcher 收到了不匹配的输入、输出或配置。"};
    }

    const std::size_t expected_partial_count = mxfp8_tensor_partial_count();
    if (workspace.partial_amax.size() != expected_partial_count) {
        throw CudaPipelineError{
            "mxfp8 CUDA tensor-scale launcher 的 partial 工作区长度与执行网格不一致。"};
    }

    const auto element_count = input.desc.elementCount();
    if (!element_count.has_value() || *element_count == 0U) {
        throw CudaPipelineError{
            "mxfp8 CUDA tensor-scale launcher 无法推导非空输入元素数量。"};
    }

    // mxfp8_tensor_partial_count 已用 maxGridSize[0] 限制该数量；当前 CUDA
    // device 上的 unsigned dim3.x 可以安全承载它。第一阶段与编码阶段复用这
    // 个 CTA 数，使两次大规模 grid-stride 遍历均保持 SM * 4 的持久化调度。
    const dim3 persistent_grid{
        static_cast<unsigned int>(expected_partial_count), 1U, 1U};
    const dim3 reduction_block{
        static_cast<unsigned int>(kThreadsPerCta), 1U, 1U};

    const auto* const input_ptr = thrust::raw_pointer_cast(input.values.data());
    auto* const payload_ptr = thrust::raw_pointer_cast(output.payload.data());
    auto* const local_scales_ptr =
        thrust::raw_pointer_cast(output.local_scales.data());
    auto* const partial_amax_ptr =
        thrust::raw_pointer_cast(workspace.partial_amax.data());
    auto* const partial_nonfinite_ptr =
        thrust::raw_pointer_cast(workspace.partial_nonfinite.data());

    // 第一阶段以 V7 的 float4 + grid-stride 输入遍历写出每 CTA partial。该
    // kernel 不可能直接得出全局 scale，因为不同 CTA 间没有同步原语。
    mxfp8TensorAmaxPartialKernel<<<persistent_grid, reduction_block, 0U, stream>>>(
        input_ptr,
        partial_amax_ptr,
        partial_nonfinite_ptr,
        *element_count);
    check_cuda(cudaGetLastError(), "tensor-scale 发射 partial amax kernel");

    // 第二阶段只发射一个 CTA。每个线程可循环读取多个 partial，最后 thread 0
    // 直接写 local_scales[0]；后续 encode kernel 因为是另一次同 stream 发射，
    // 无需任何跨 CTA 或 host 同步即可安全读取该 scale。
    mxfp8TensorScaleFinalizeKernel<<<1U, reduction_block, 0U, stream>>>(
        partial_amax_ptr,
        partial_nonfinite_ptr,
        local_scales_ptr,
        static_cast<std::uint64_t>(expected_partial_count));
    check_cuda(cudaGetLastError(), "tensor-scale 发射 scale finalize kernel");

    mxfp8TensorEncodeKernel<<<persistent_grid, reduction_block, 0U, stream>>>(
        input_ptr,
        payload_ptr,
        local_scales_ptr,
        *element_count,
        config.rounding,
        config.stochastic_seed);
    check_cuda(cudaGetLastError(), "tensor-scale 发射 encode kernel");
}

}  // namespace quant_dequant::cuda
