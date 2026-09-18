#pragma once

#include <cstdint>

#include "mxfp8_codec.cuh"

namespace quant_dequant::formats {

/**
 * @brief E2M1 payload 中有效的四个 bit 掩码。
 *
 * NVFP4 的两个逻辑元素共用一个字节；单元素 codec 只处理其中一个 nibble，
 * 因此任何 E2M1 code 都必须限制在 bits `[3:0]`。
 */
inline constexpr std::uint8_t kE2M1NibbleMask = 0x0fU;

/**
 * @brief E2M1 正最大有限值的 magnitude code。
 *
 * `0b0111` 的 bit layout 为 `S=0, EE=3, M=1`，表示 `+6`。
 */
inline constexpr std::uint8_t kE2M1PositiveMaxCode = 0x07U;

/**
 * @brief E2M1 可表示的最大有限绝对值。
 */
inline constexpr float kE2M1MaxFinite = 6.0F;

/**
 * @brief E2M1 的最小 normal 正数。
 */
inline constexpr float kE2M1MinNormal = 1.0F;

/**
 * @brief E2M1 的最小 subnormal 正数。
 */
inline constexpr float kE2M1MinSubnormal = 0.5F;

/**
 * @brief 一个严格 NVFP4 rowwise block 的固定元素数量。
 */
inline constexpr std::uint32_t kNvfp4ElementsPerBlock = kNvfp4BlockSize;

/**
 * @brief 标准 NVFP4 decode 方向 global scale 的分母。
 *
 * 对 tensor amax $a_g$，NVFP4 写入的 global scale 是
 * $a_g / (448 \times 6) = a_g / 2688$。448 是 E4M3 的最大有限值，6 是
 * E2M1 的最大有限值。
 */
inline constexpr float kNvfp4GlobalScaleDenominator =
    kE4M3MaxFinite * kE2M1MaxFinite;

/**
 * @brief 为一个 NVFP4 元素生成可复现的 stochastic rounding 随机数。
 *
 * NVFP4 与 MXFP8 共用“配置 seed + row-major 线性下标”的无状态 SplitMix64
 * 映射；format 不参与随机状态，CPU reference 与 CUDA kernel 因而不会受执行
 * 顺序影响。这里保留 NVFP4 名称，避免调用方依赖另一格式的接口名称。
 *
 * @param stochastic_seed 配置文件指定的确定性随机种子。
 * @param linear_index 元素在完整 row-major 张量中的零起始下标。
 * @return 严格位于 `[0, 1)` 的 FP32 均匀随机数。
 */
[[nodiscard]] __host__ __device__ inline float
nvfp4_stochastic_uniform_for_element(
    const std::uint64_t stochastic_seed,
    const std::uint64_t linear_index) noexcept {
    return mxfp8_stochastic_uniform_for_element(stochastic_seed, linear_index);
}

namespace detail {

/**
 * @brief 将一个无符号 E2M1 magnitude code 解码为非负 FP32 幅值。
 *
 * 低三位的 bit layout 为 `EE M`。E=0、M=1 是 subnormal `0.5`；E>0
 * 使用隐含 leading 1。E2M1 的全部 8 个 magnitude code 都是有限数，
 * 不存在 Inf 或 NaN 保留码。
 *
 * @param magnitude_code E2M1 的低三位；高位会被忽略。
 * @return 对应的非负 E2M1 幅值。
 */
[[nodiscard]] __host__ __device__ inline float decode_e2m1_magnitude(
    const std::uint8_t magnitude_code) noexcept {
    switch (magnitude_code & kE2M1PositiveMaxCode) {
        case 0x00U:
            return 0.0F;

        case 0x01U:
            return kE2M1MinSubnormal;

        case 0x02U:
            return 1.0F;

        case 0x03U:
            return 1.5F;

        case 0x04U:
            return 2.0F;

        case 0x05U:
            return 3.0F;

        case 0x06U:
            return 4.0F;

        case 0x07U:
            return kE2M1MaxFinite;
    }

    // 输入已按低三位掩码收窄；此 return 只用于安静处理理论上不可达的路径。
    return 0.0F;
}

/**
 * @brief 找到不大于给定幅值的最大 E2M1 magnitude code。
 *
 * E2M1 只有八个有限正幅值，且相邻间距不均匀：`0.5, 0.5, 0.5, 1, 1, 2`。
 * 直接比较 codebook 边界比把 IEEE 位字段反推为指数和尾数更清楚，也避免在
 * subnormal 与 normal 边界引入特殊算式。stochastic rounding 以该下界和
 * 下一个 code 构造概率；RNE 则用这对端点判断最近值。
 *
 * 调用前要求 `magnitude` 非负、有限且严格小于 6。
 *
 * @param magnitude 待定位的 E2M1 非负幅值。
 * @return 最大的 code，使其解码值不大于 `magnitude`。
 */
[[nodiscard]] __host__ __device__ inline std::uint8_t
find_e2m1_lower_magnitude_code(const float magnitude) noexcept {
    if (magnitude < kE2M1MinSubnormal) {
        return 0x00U;
    }

    if (magnitude < 1.0F) {
        return 0x01U;
    }

    if (magnitude < 1.5F) {
        return 0x02U;
    }

    if (magnitude < 2.0F) {
        return 0x03U;
    }

    if (magnitude < 3.0F) {
        return 0x04U;
    }

    if (magnitude < 4.0F) {
        return 0x05U;
    }

    return 0x06U;
}

}  // namespace detail

/**
 * @brief 将一个 E2M1 nibble 解码为 FP32。
 *
 * E2M1 bit layout 是 `S EE M`。其指数 bias 为 1：E=0 时没有隐含 1，
 * `M=1` 表示 subnormal `0.5`；E>0 时，值为
 * $(1 + M/2) \times 2^{E-1}$。`+0` 与 `-0` 都是合法输入，解码时保留其
 * 符号位。不同于 IEEE FP32，E2M1 没有 Inf 或 NaN 编码。
 *
 * @param encoded 包含一个 E2M1 code 的低 nibble；高四位会被忽略。
 * @return 对应 FP32 值。
 */
[[nodiscard]] __host__ __device__ inline float decode_e2m1(
    const std::uint8_t encoded) noexcept {
    const bool negative = (encoded & 0x08U) != 0U;
    const float magnitude = detail::decode_e2m1_magnitude(encoded);

    if (magnitude == 0.0F) {
        return fp32::signed_zero(negative);
    }

    return negative ? -magnitude : magnitude;
}

/**
 * @brief 以 RNE（ties-to-even）和饱和规则将 FP32 编码为一个 E2M1 nibble。
 *
 * 有限输入超过 6 以及正负 Inf 均饱和到同符号的 `+/-6`。E2M1 没有 NaN
 * 编码，所以 NaN 规范化为正零 `0x0`；完整量化 pipeline 会在计算 scale 前
 * 拒绝非有限原始输入，这一回退只保证单元素 codec 的 host/device 行为确定。
 * 与项目的 MXFP8 写入约定一致，`-0` 也编码为正零。
 *
 * @param value 待编码的 FP32 值。
 * @return 低四位为 E2M1 `S EE M` 的 code，高四位恒为零。
 */
[[nodiscard]] __host__ __device__ inline std::uint8_t encode_e2m1_rne_sat(
    const float value) noexcept {
    const bool negative = fp32::has_negative_sign(value);
    const std::uint8_t sign_bit = negative ? 0x08U : 0x00U;

    if (!fp32::is_finite(value)) {
        const float magnitude = fp32::absolute_value(value);
        if (magnitude != magnitude) {
            return 0x00U;
        }

        return static_cast<std::uint8_t>(sign_bit | kE2M1PositiveMaxCode);
    }

    const float magnitude = fp32::absolute_value(value);
    if (magnitude == 0.0F) {
        return 0x00U;
    }

    if (magnitude >= kE2M1MaxFinite) {
        return static_cast<std::uint8_t>(sign_bit | kE2M1PositiveMaxCode);
    }

    const std::uint8_t lower_code =
        detail::find_e2m1_lower_magnitude_code(magnitude);
    const float lower_value = detail::decode_e2m1_magnitude(lower_code);
    if (magnitude == lower_value) {
        return static_cast<std::uint8_t>(sign_bit | lower_code);
    }

    const std::uint8_t upper_code =
        static_cast<std::uint8_t>(lower_code + 1U);
    const float upper_value = detail::decode_e2m1_magnitude(upper_code);
    const float midpoint = lower_value + (upper_value - lower_value) * 0.5F;

    if (magnitude < midpoint ||
        (magnitude == midpoint && (lower_code & 1U) == 0U)) {
        return static_cast<std::uint8_t>(sign_bit | lower_code);
    }

    return static_cast<std::uint8_t>(sign_bit | upper_code);
}

/**
 * @brief 以 stochastic rounding 和饱和规则将 FP32 编码为一个 E2M1 nibble。
 *
 * 若相邻可表示幅值为 $l$ 与 $h$，则以上界概率
 * $(|value|-l)/(h-l)$ 选择 $h$。调用方必须通过固定 seed 和全局线性下标生成
 * `uniform_random`，才能保证 CPU reference 和 CUDA kernel 的 payload 一致。
 *
 * @param value 待编码的 FP32 值。
 * @param uniform_random 候选均匀随机数；内部会将其收窄至 `[0, 1)`。
 * @return 低四位为 E2M1 `S EE M` 的 code，高四位恒为零。
 */
[[nodiscard]] __host__ __device__ inline std::uint8_t
encode_e2m1_stochastic_sat(const float value,
                           const float uniform_random) noexcept {
    if (!fp32::is_finite(value)) {
        return encode_e2m1_rne_sat(value);
    }

    const bool negative = fp32::has_negative_sign(value);
    const std::uint8_t sign_bit = negative ? 0x08U : 0x00U;
    const float magnitude = fp32::absolute_value(value);

    if (magnitude == 0.0F || magnitude >= kE2M1MaxFinite) {
        return encode_e2m1_rne_sat(value);
    }

    const std::uint8_t lower_code =
        detail::find_e2m1_lower_magnitude_code(magnitude);
    const float lower_value = detail::decode_e2m1_magnitude(lower_code);
    if (magnitude == lower_value) {
        return static_cast<std::uint8_t>(sign_bit | lower_code);
    }

    const std::uint8_t upper_code =
        static_cast<std::uint8_t>(lower_code + 1U);
    const float upper_value = detail::decode_e2m1_magnitude(upper_code);
    const float probability_of_upper =
        (magnitude - lower_value) / (upper_value - lower_value);

    const std::uint8_t selected_code =
        fp32::clamp_uniform_random(uniform_random) < probability_of_upper
            ? upper_code
            : lower_code;
    return static_cast<std::uint8_t>(sign_bit | selected_code);
}

/**
 * @brief 按配置选择 RNE 或 stochastic E2M1 元素编码。
 *
 * `RoundingMode::kUnknown` 应由上层配置校验拒绝；codec 为保持 host/device
 * 行为稳定，仍回退到 RNE。
 *
 * @param value 待编码的 FP32 值。
 * @param rounding_mode 元素舍入模式。
 * @param uniform_random stochastic rounding 使用的随机数；RNE 下忽略。
 * @return 一个高四位为零的 E2M1 nibble。
 */
[[nodiscard]] __host__ __device__ inline std::uint8_t encode_e2m1(
    const float value,
    const RoundingMode rounding_mode,
    const float uniform_random = 0.0F) noexcept {
    if (rounding_mode == RoundingMode::kStochastic) {
        return encode_e2m1_stochastic_sat(value, uniform_random);
    }

    return encode_e2m1_rne_sat(value);
}

/**
 * @brief 根据整张张量的 amax 计算 NVFP4 的 FP32 decode global scale。
 *
 * 对正有限 amax，返回 `amax / (448 * 6)`。全零张量返回 1.0，避免随后计算
 * `block_amax / (6 * global_scale)` 时出现除零。负 amax、NaN 与 Inf 都是
 * 调用方错误，返回 canonical FP32 quiet NaN 供上层检测。
 *
 * @param tensor_amax 整张输入张量的最大绝对值。
 * @return 有限正的 FP32 global scale、全零时的 1.0，或非法输入的 NaN。
 */
[[nodiscard]] __host__ __device__ inline float compute_nvfp4_global_scale(
    const float tensor_amax) noexcept {
    if (!fp32::is_finite(tensor_amax) ||
        (tensor_amax != 0.0F && fp32::has_negative_sign(tensor_amax))) {
        return fp32::canonical_quiet_nan();
    }

    if (tensor_amax == 0.0F) {
        return 1.0F;
    }

    return tensor_amax / kNvfp4GlobalScaleDenominator;
}

/**
 * @brief 根据一个 16 元素 block 的 amax 计算其 E4M3 local scale code。
 *
 * 严格 NVFP4 的 local decode scale 满足
 * $s_b=\operatorname{E4M3\_RNE\_SAT}(a_b/(6s_g))$。local scale 始终以
 * RNE 编码，和 E2M1 元素的 `rounding_mode` 无关。全零 block 写正零
 * `0x00`；非法 amax 或 global scale 返回 E4M3 canonical NaN `0x7f`。
 *
 * @param block_amax 当前 16 元素 block 的最大绝对值。
 * @param global_scale 本张量的 FP32 decode global scale。
 * @return 一个 E4M3 local scale byte，非法输入返回 `0x7f`。
 */
[[nodiscard]] __host__ __device__ inline std::uint8_t
compute_nvfp4_local_scale_code(const float block_amax,
                               const float global_scale) noexcept {
    if (!fp32::is_finite(block_amax) ||
        (block_amax != 0.0F && fp32::has_negative_sign(block_amax)) ||
        !fp32::is_finite(global_scale) || global_scale <= 0.0F) {
        return kE4M3CanonicalNaNCode;
    }

    if (block_amax == 0.0F) {
        return 0x00U;
    }

    return encode_e4m3_rne_sat(
        block_amax / (kE2M1MaxFinite * global_scale));
}

/**
 * @brief 将一个 FP32 元素编码为 NVFP4 E2M1 payload nibble。
 *
 * 先按 $z=value/(s_b s_g)$ 缩放，再按调用方选择的舍入策略编码为 E2M1。
 * 若 local scale 是 E4M3 NaN、任一 scale 非有限或 local scale 为零，则返回
 * 正零 nibble。最后一种情况对应 E4M3 local scale 下溢的 block；项目规范
 * 要求该 block 的所有元素都量化为零。
 *
 * @param value 待量化的 FP32 元素。
 * @param local_scale_code 当前 16 元素 block 的 E4M3 local scale code。
 * @param global_scale 本张量的 FP32 decode global scale。
 * @param rounding_mode E2M1 元素编码使用的舍入模式。
 * @param uniform_random stochastic rounding 使用的随机数；RNE 下忽略。
 * @return 一个高四位为零的 E2M1 nibble。
 */
[[nodiscard]] __host__ __device__ inline std::uint8_t encode_nvfp4_element(
    const float value,
    const std::uint8_t local_scale_code,
    const float global_scale,
    const RoundingMode rounding_mode,
    const float uniform_random = 0.0F) noexcept {
    const float local_scale = decode_e4m3(local_scale_code);
    const float combined_scale = local_scale * global_scale;

    if (!fp32::is_finite(local_scale) || !fp32::is_finite(global_scale) ||
        local_scale <= 0.0F || global_scale <= 0.0F ||
        !fp32::is_finite(combined_scale) || combined_scale <= 0.0F) {
        return 0x00U;
    }

    return encode_e2m1(value / combined_scale, rounding_mode, uniform_random);
}

/**
 * @brief 将一个 NVFP4 E2M1 payload nibble 反量化为 FP32。
 *
 * 数学过程为 $decode\_e2m1(payload) \times decode\_e4m3(local) \times global$。
 * local scale 的 E4M3 NaN 或 global FP32 NaN 会由 IEEE FP32 乘法自然传播。
 *
 * @param payload 一个 E2M1 nibble；高四位会被忽略。
 * @param local_scale_code 当前 16 元素 block 的 E4M3 local scale code。
 * @param global_scale 本张量的 FP32 decode global scale。
 * @return 反量化后的 FP32 值。
 */
[[nodiscard]] __host__ __device__ inline float decode_nvfp4_element(
    const std::uint8_t payload,
    const std::uint8_t local_scale_code,
    const float global_scale) noexcept {
    return decode_e2m1(payload) * decode_e4m3(local_scale_code) * global_scale;
}

/**
 * @brief 将两个 E2M1 nibble 按项目的 NVFP4 payload 顺序打包为一个字节。
 *
 * 线性偶数下标元素写入低 nibble，紧随其后的奇数下标元素写入高 nibble。
 * 最后仅剩一个逻辑元素时，调用方应将 `high_nibble` 传为零。
 *
 * @param low_nibble 偶数线性下标对应的 E2M1 code。
 * @param high_nibble 奇数线性下标对应的 E2M1 code。
 * @return 打包后的一个 payload 字节。
 */
[[nodiscard]] __host__ __device__ inline std::uint8_t pack_e2m1_nibbles(
    const std::uint8_t low_nibble,
    const std::uint8_t high_nibble) noexcept {
    return static_cast<std::uint8_t>(
        (low_nibble & kE2M1NibbleMask) |
        ((high_nibble & kE2M1NibbleMask) << 4U));
}

/**
 * @brief 取出 packed NVFP4 payload byte 的低 E2M1 nibble。
 *
 * @param packed_payload 一个含两个逻辑 E2M1 code 的 payload 字节。
 * @return 偶数线性下标元素的 E2M1 code。
 */
[[nodiscard]] __host__ __device__ inline std::uint8_t unpack_e2m1_low_nibble(
    const std::uint8_t packed_payload) noexcept {
    return packed_payload & kE2M1NibbleMask;
}

/**
 * @brief 取出 packed NVFP4 payload byte 的高 E2M1 nibble。
 *
 * @param packed_payload 一个含两个逻辑 E2M1 code 的 payload 字节。
 * @return 奇数线性下标元素的 E2M1 code。
 */
[[nodiscard]] __host__ __device__ inline std::uint8_t unpack_e2m1_high_nibble(
    const std::uint8_t packed_payload) noexcept {
    return static_cast<std::uint8_t>(packed_payload >> 4U);
}

}  // namespace quant_dequant::formats
