#pragma once

#include <cstdint>

#include "fp32_utils.cuh"
#include "quant_dequant/types.hpp"

namespace quant_dequant::formats {

/**
 * @brief E4M3 正最大有限值的 7-bit magnitude code。
 *
 * 完整 E4M3 byte 还会在 bit 7 放置符号。`0x7e` 的 bit 字段为
 * `0 1111 110`，即 E=15、M=6，对应 +448。
 */
inline constexpr std::uint8_t kE4M3PositiveMaxCode = 0x7eU;

/**
 * @brief E4M3 canonical NaN 的 7-bit magnitude code。
 *
 * `0x7f` 的 bit 字段为 `0 1111 111`。添加符号位后的 `0x7f` 与 `0xff`
 * 都按 E4M3 NaN 处理。
 */
inline constexpr std::uint8_t kE4M3CanonicalNaNCode = 0x7fU;

/**
 * @brief E8M0 NaN 的编码。
 *
 * 实际 bit 为 `0xff`，即八个指数 bit 全为 1。
 */
inline constexpr std::uint8_t kE8M0NaNCode = 0xffU;

/**
 * @brief E4M3 可表示的最大有限绝对值。
 */
inline constexpr float kE4M3MaxFinite = 448.0F;

/**
 * @brief E4M3 的最小 normal 正数，即 $2^{-6}$。
 */
inline constexpr float kE4M3MinNormal = 0x1.0p-6F;

/**
 * @brief E4M3 的最小 subnormal 正数，即 $2^{-9}$。
 */
inline constexpr float kE4M3MinSubnormal = 0x1.0p-9F;

/**
 * @brief 一个 MXFP8 rowwise block 的固定元素数量。
 */
inline constexpr std::uint32_t kMxfp8ElementsPerBlock = kMxfp8BlockSize;

/**
 * @brief 使用 SplitMix64 混合一个无状态随机数状态。
 *
 * stochastic rounding 必须按“量化配置 seed + 全局线性下标”独立生成随机数，
 * 不能依赖线程调度顺序。将该规则放入 host/device codec，确保 CPU reference
 * 与 CUDA kernel 对同一元素产生完全相同的随机 bit。
 *
 * 无符号整数溢出按模 $2^{64}$ 定义，因此这个过程在 host 与 device 上一致。
 *
 * @param state 待混合的 64-bit 状态。
 * @return 混合后的 64-bit 伪随机 bit。
 */
[[nodiscard]] __host__ __device__ inline std::uint64_t mxfp8_splitmix64(
    std::uint64_t state) noexcept {
    state += 0x9e3779b97f4a7c15ULL;
    state = (state ^ (state >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    state = (state ^ (state >> 27U)) * 0x94d049bb133111ebULL;
    return state ^ (state >> 31U);
}

/**
 * @brief 为一个 row-major 全局线性元素下标生成 `[0, 1)` 内的随机数。
 *
 * 仅提取 SplitMix64 输出的高 24 bit，再乘以 $2^{-24}$；结果严格小于 1.0F，
 * 可直接传给 `encode_mxfp8_element()` 的 stochastic rounding 路径。
 *
 * @param stochastic_seed 配置文件指定的确定性随机种子。
 * @param linear_index 元素在完整 row-major 张量中的零起始下标。
 * @return 位级可复现的 `[0, 1)` FP32 随机数。
 */
[[nodiscard]] __host__ __device__ inline float
mxfp8_stochastic_uniform_for_element(
    const std::uint64_t stochastic_seed,
    const std::uint64_t linear_index) noexcept {
    constexpr std::uint64_t kIndexStride = 0x9e3779b97f4a7c15ULL;
    constexpr float kInvTwoTo24 = 0x1.0p-24F;

    const std::uint64_t state =
        stochastic_seed + kIndexStride * (linear_index + 1U);
    const std::uint64_t random_bits = mxfp8_splitmix64(state);
    const std::uint32_t random_mantissa =
        static_cast<std::uint32_t>(random_bits >> 40U);

    return static_cast<float>(random_mantissa) * kInvTwoTo24;
}

namespace detail {

/**
 * @brief 找到小于等于给定幅值的最大 E4M3 有限 magnitude code。
 *
 * 完整 E4M3 的 bit layout 是 `S EEEE MMM`。本函数刻意忽略符号 `S`，
 * 只返回 `EEEE MMM`，因此结果是范围 `[0x00, 0x7e]` 内的 7-bit code。
 * 例如输入幅值为 1.30 时，返回 `0x3a`，它表示 +1.25；下一可表示值
 * `0x3b` 是 +1.375。stochastic rounding 需要这两个相邻端点来计算
 * 选择上界的概率，因此本函数返回下界，而不是最近值。
 *
 * 调用前要求 magnitude 位于 `[0, 448)`。该范围保证返回值是有限 E4M3：
 * `0x7f` 是 NaN，故不能作为普通上界。
 *
 * @param magnitude 非负、有限且严格小于 448 的 FP32 幅值。
 * @return 小于等于 magnitude 的最大 E4M3 magnitude code。
 */
[[nodiscard]] __host__ __device__ inline std::uint8_t
find_e4m3_lower_magnitude_code(const float magnitude) noexcept {
    if (magnitude < kE4M3MinNormal) {
        // subnormal 的间距为 2^-9。除以间距等于乘 2^9，即乘 512：
        // magnitude * 512 = magnitude / 2^-9。
        // 此分支保证结果落在 [0, 8)，故转换为 uint32 后不会溢出。
        const float subnormal_units = magnitude * 512.0F;

        // 对非负浮点数，float -> uint32 的转换截断小数部分，等价于 floor。
        // 因而 lower_mantissa 正是最大的 M，使 M * 2^-9 <= magnitude。
        const std::uint32_t lower_mantissa =
            static_cast<std::uint32_t>(subnormal_units);

        // E=0 的 subnormal code 是 `0000 MMM`，所以 M 本身就是 7-bit code。
        return static_cast<std::uint8_t>(lower_mantissa);
    }

    // 此时 magnitude 是 E4M3 normal。floor(log2(magnitude)) 返回真实指数 e，
    // 满足 2^e <= magnitude < 2^(e+1)。例如 magnitude=1.30 时 e=0；
    // magnitude=0.30 时 e=-2。它不是任何格式中直接存储的指数 bit。
    const int unbiased_exponent = fp32::floor_log2_positive(magnitude);

    // E4M3 的 4-bit 指数位不能直接存负数，故实际存储 E=e+bias=e+7。
    // 例如 e=-2 编码为 E=5；e=0 编码为 E=7。
    const std::uint32_t e4m3_biased_exponent =
        static_cast<std::uint32_t>(unbiased_exponent + 7);

    // 除以 2^e 后，normalized 位于 [1, 2)。减一得到 [0, 1) 的小数部分；
    // 乘 8 是因为 E4M3 只有 3 个 mantissa bit，可把该区间分成 8 格。
    const float normalized =
        magnitude / fp32::power_of_two(unbiased_exponent);
    const std::uint32_t lower_mantissa = static_cast<std::uint32_t>(
        (normalized - 1.0F) * 8.0F);

    // 拼接不含符号位的 `EEEE MMM`。lower_mantissa 向下截断，故结果正好是
    // 小于等于 magnitude 的同指数区间中最大 E4M3 值。
    return static_cast<std::uint8_t>(
        (e4m3_biased_exponent << 3U) | lower_mantissa);
}

}  // namespace detail

/**
 * @brief 判断 E4M3 原始 byte 是否表示 NaN。
 *
 * @param encoded 完整 E4M3 byte，包含符号位。
 * @return E=15 且 M=7 时返回 true。
 */
[[nodiscard]] __host__ __device__ inline bool is_e4m3_nan(
    const std::uint8_t encoded) noexcept {
    return (encoded & 0x7fU) == kE4M3CanonicalNaNCode;
}

/**
 * @brief 判断 E8M0 scale byte 是否表示 NaN。
 *
 * @param encoded E8M0 scale byte。
 * @return encoded 为 `0xff` 时返回 true。
 */
[[nodiscard]] __host__ __device__ inline bool is_e8m0_nan(
    const std::uint8_t encoded) noexcept {
    return encoded == kE8M0NaNCode;
}

/**
 * @brief 解码一个 E4M3 byte 为 FP32。
 *
 * E=0 时，E4M3 是 subnormal，幅值为 $M\times2^{-9}$；E>0 时，幅值为
 * $(8+M)\times2^{E-10}$。E=15、M=7 是 NaN，不存在 Inf 编码。
 *
 * @param encoded 完整 E4M3 byte，bit 7 为符号位。
 * @return 对应 FP32；E4M3 NaN 返回 bit 为 `0x7fc00000` 的 quiet NaN。
 */
[[nodiscard]] __host__ __device__ inline float decode_e4m3(
    const std::uint8_t encoded) noexcept {
    const bool negative = (encoded & 0x80U) != 0U;
    const std::uint8_t magnitude_code = encoded & 0x7fU;
    const std::uint8_t exponent_field = magnitude_code >> 3U;
    const std::uint8_t mantissa = magnitude_code & 0x07U;

    if (exponent_field == 0x0fU && mantissa == 0x07U) {
        return fp32::canonical_quiet_nan();
    }

    if (exponent_field == 0U && mantissa == 0U) {
        return fp32::signed_zero(negative);
    }

    const float magnitude = exponent_field == 0U
        ? static_cast<float>(mantissa) * kE4M3MinSubnormal
        : static_cast<float>(8U + mantissa) *
              fp32::power_of_two(static_cast<int>(exponent_field) - 10);

    return negative ? -magnitude : magnitude;
}

/**
 * @brief 以 RNE（ties-to-even）和饱和规则将 FP32 编码为 E4M3。
 *
 * 有限输入超过 448 以及 `+/-Inf` 均饱和到带相同符号的最大有限值
 * `+/-448`。NaN 编码为 canonical E4M3 NaN `0x7f`。
 *
 * @param value 待编码的 FP32 值。
 * @return 完整 E4M3 byte，bit 7 为符号位。
 */
[[nodiscard]] __host__ __device__ inline std::uint8_t encode_e4m3_rne_sat(
    const float value) noexcept {
    const bool negative = fp32::has_negative_sign(value);
    const std::uint8_t sign_bit = negative ? 0x80U : 0x00U;

    if (!fp32::is_finite(value)) {
        const float magnitude = fp32::absolute_value(value);

        if (magnitude != magnitude) {
            return kE4M3CanonicalNaNCode;
        }

        return static_cast<std::uint8_t>(sign_bit | kE4M3PositiveMaxCode);
    }

    const float magnitude = fp32::absolute_value(value);

    if (magnitude == 0.0F) {
        // 项目量化输出统一使用 +0（0x00），满足全零 block 的文件约定。
        // 解码器仍能正确读取外部文件中合法的 -0（0x80）。
        return 0x00U;
    }

    if (magnitude >= kE4M3MaxFinite) {
        return static_cast<std::uint8_t>(sign_bit | kE4M3PositiveMaxCode);
    }

    if (magnitude < kE4M3MinNormal) {
        const std::uint32_t rounded_units =
            fp32::round_to_nearest_even_nonnegative(magnitude * 512.0F);

        // code 8 正好是 E=1、M=0，即最小 normal，而不是 subnormal。
        return static_cast<std::uint8_t>(sign_bit | rounded_units);
    }

    int unbiased_exponent = fp32::floor_log2_positive(magnitude);
    const float normalized =
        magnitude / fp32::power_of_two(unbiased_exponent);
    std::uint32_t mantissa = fp32::round_to_nearest_even_nonnegative(
        (normalized - 1.0F) * 8.0F);

    // mantissa 从 7 向上舍入时进位到下一个 exponent。
    if (mantissa == 8U) {
        mantissa = 0U;
        ++unbiased_exponent;
    }

    const int exponent_field = unbiased_exponent + 7;

    if (exponent_field > 15 ||
        (exponent_field == 15 && mantissa >= 7U)) {
        // E=15、M=7 是 NaN，有限饱和必须停在 E=15、M=6。
        return static_cast<std::uint8_t>(sign_bit | kE4M3PositiveMaxCode);
    }

    return static_cast<std::uint8_t>(
        sign_bit |
        (static_cast<std::uint8_t>(exponent_field) << 3U) |
        static_cast<std::uint8_t>(mantissa));
}

/**
 * @brief 以 stochastic rounding 和饱和规则将 FP32 编码为 E4M3。
 *
 * 对相邻可表示幅值 $l$ 与 $h$，该函数以 `(abs(value)-l)/(h-l)` 的概率
 * 选择 $h$。调用方负责将固定 seed 和线性元素下标映射为
 * `uniform_random`，从而让 CPU/GPU 获得可复现的相同 payload。
 *
 * @param value 待编码的 FP32 值。
 * @param uniform_random 均匀随机数；正常调用应落在 `[0, 1)`。
 * @return 完整 E4M3 byte，bit 7 为符号位。
 */
[[nodiscard]] __host__ __device__ inline std::uint8_t
encode_e4m3_stochastic_sat(const float value,
                           const float uniform_random) noexcept {
    const bool negative = fp32::has_negative_sign(value);
    const std::uint8_t sign_bit = negative ? 0x80U : 0x00U;

    if (!fp32::is_finite(value)) {
        return encode_e4m3_rne_sat(value);
    }

    const float magnitude = fp32::absolute_value(value);

    if (magnitude == 0.0F || magnitude >= kE4M3MaxFinite) {
        return encode_e4m3_rne_sat(value);
    }

    const std::uint8_t lower_code =
        detail::find_e4m3_lower_magnitude_code(magnitude);
    const float lower_value = decode_e4m3(lower_code);

    if (magnitude == lower_value || lower_code == kE4M3PositiveMaxCode) {
        return static_cast<std::uint8_t>(sign_bit | lower_code);
    }

    const std::uint8_t upper_code =
        static_cast<std::uint8_t>(lower_code + 1U);
    const float upper_value = decode_e4m3(upper_code);
    const float probability_of_upper =
        (magnitude - lower_value) / (upper_value - lower_value);

    const std::uint8_t selected_code =
        fp32::clamp_uniform_random(uniform_random) < probability_of_upper
            ? upper_code
            : lower_code;

    return static_cast<std::uint8_t>(sign_bit | selected_code);
}

/**
 * @brief 按配置舍入方式将 FP32 编码为 E4M3。
 *
 * `RoundingMode::kUnknown` 理论上由配置校验拒绝。codec 不抛异常，为使
 * host/device 行为稳定，未知枚举值回退为 RNE。
 *
 * @param value 待编码的 FP32 值。
 * @param rounding_mode 元素舍入模式。
 * @param uniform_random stochastic rounding 使用的随机数；RNE 下忽略。
 * @return 完整 E4M3 byte。
 */
[[nodiscard]] __host__ __device__ inline std::uint8_t encode_e4m3(
    const float value,
    const RoundingMode rounding_mode,
    const float uniform_random = 0.0F) noexcept {
    if (rounding_mode == RoundingMode::kStochastic) {
        return encode_e4m3_stochastic_sat(value, uniform_random);
    }

    return encode_e4m3_rne_sat(value);
}

/**
 * @brief 解码 E8M0 scale byte 为 FP32。
 *
 * `encoded` 位于 `[0, 254]` 时表示 $2^{encoded-127}$。`0xff` 是 NaN，
 * 返回项目固定的 FP32 quiet NaN bit `0x7fc00000`。
 *
 * @param encoded E8M0 scale byte。
 * @return 正的 FP32 scale，或 NaN。
 */
[[nodiscard]] __host__ __device__ inline float decode_e8m0(
    const std::uint8_t encoded) noexcept {
    if (encoded == kE8M0NaNCode) {
        return fp32::canonical_quiet_nan();
    }

    return fp32::power_of_two(static_cast<int>(encoded) - 127);
}

/**
 * @brief 将非负 FP32 scale 向上取整为 E8M0。
 *
 * 返回 `clamp(ceil(log2(required_scale)) + 127, 0, 254)`。向上取整保证
 * scale 不会偏小；这是 MXFP8 block 最大值避免 E4M3 溢出的必要条件。
 * `+0` 与 `-0` 编码为 `0x00`，即最小有限 E8M0 scale $2^{-127}$；负
 * 非零值、NaN、Inf 均返回 E8M0 NaN `0xff`。
 *
 * @param required_scale 需要向上取整的非负 FP32 scale。
 * @return E8M0 scale byte，非法输入返回 `0xff`。
 */
[[nodiscard]] __host__ __device__ inline std::uint8_t
encode_e8m0_round_up(const float required_scale) noexcept {
    const float magnitude = fp32::absolute_value(required_scale);

    if (magnitude == 0.0F) {
        return 0x00U;
    }

    if (!fp32::is_finite(required_scale) ||
        fp32::has_negative_sign(required_scale)) {
        return kE8M0NaNCode;
    }

    const int floor_exponent = fp32::floor_log2_positive(required_scale);
    const int ceil_exponent = floor_exponent +
        (fp32::is_power_of_two_positive(required_scale) ? 0 : 1);

    int encoded_exponent = ceil_exponent + 127;

    if (encoded_exponent < 0) {
        encoded_exponent = 0;
    } else if (encoded_exponent > 254) {
        encoded_exponent = 254;
    }

    return static_cast<std::uint8_t>(encoded_exponent);
}

/**
 * @brief 由 block 或 tensor 的 amax 计算 MXFP8 E8M0 scale byte。
 *
 * 对 `scale_mode=block` 传入一个 rowwise 32 元素 block 的 amax；对
 * `scale_mode=tensor` 传入整张矩阵的 amax。两种模式都遵循
 * `round_up_to_e8m0(amax / 448)`。
 *
 * @param amax 输入范围的最大绝对值。
 * @return 对应 E8M0 scale byte；非法 amax 返回 `0xff`。
 */
[[nodiscard]] __host__ __device__ inline std::uint8_t
compute_mxfp8_scale_code(const float amax) noexcept {
    if (!fp32::is_finite(amax)) {
        return kE8M0NaNCode;
    }

    // FP32 的 +0 与 -0 数值相等；amax 为零时均应使用最小有限 scale。
    if (amax == 0.0F) {
        return 0x00U;
    }

    if (fp32::has_negative_sign(amax)) {
        return kE8M0NaNCode;
    }

    return encode_e8m0_round_up(amax / kE4M3MaxFinite);
}

/**
 * @brief 用 E8M0 scale 将一个 FP32 元素编码为 MXFP8 E4M3 payload。
 *
 * 数学过程是 $z=value/decode\_e8m0(scale\_code)$，再执行 E4M3 编码。
 * scale 为 E8M0 NaN 时，返回 E4M3 canonical NaN `0x7f`。
 *
 * @param value 待量化的 FP32 输入元素。
 * @param scale_code 当前 block 或 tensor 的 E8M0 scale byte。
 * @param rounding_mode E4M3 元素编码使用的舍入模式。
 * @param uniform_random stochastic rounding 使用的随机数；RNE 下忽略。
 * @return 一个 E4M3 payload byte。
 */
[[nodiscard]] __host__ __device__ inline std::uint8_t encode_mxfp8_element(
    const float value,
    const std::uint8_t scale_code,
    const RoundingMode rounding_mode,
    const float uniform_random = 0.0F) noexcept {
    const float scale = decode_e8m0(scale_code);

    if (scale != scale) {
        return kE4M3CanonicalNaNCode;
    }

    return encode_e4m3(value / scale, rounding_mode, uniform_random);
}

/**
 * @brief 将一个 MXFP8 E4M3 payload 反量化为 FP32。
 *
 * 数学过程是 $decode\_e4m3(payload)\times decode\_e8m0(scale\_code)$。
 * E4M3 或 E8M0 的 NaN 会遵循 IEEE FP32 乘法自然传播。
 *
 * @param payload E4M3 元素 byte。
 * @param scale_code 当前 block 或 tensor 的 E8M0 scale byte。
 * @return 反量化后的 FP32 值。
 */
[[nodiscard]] __host__ __device__ inline float decode_mxfp8_element(
    const std::uint8_t payload,
    const std::uint8_t scale_code) noexcept {
    return decode_e4m3(payload) * decode_e8m0(scale_code);
}

}  // namespace quant_dequant::formats
