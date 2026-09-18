#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <cstring>
#include <limits>

namespace quant_dequant::formats::fp32 {

static_assert(sizeof(float) == sizeof(std::uint32_t),
              "本项目的 FP32 codec 要求 float 占 32 bit。");
static_assert(std::numeric_limits<float>::is_iec559,
              "本项目的 FP32 codec 要求 IEEE 754 binary32 语义。");

/**
 * @brief IEEE 754 binary32 的符号位掩码。
 *
 * 实际 bit 为 `0x80000000`，即第 31 bit；置位表示负数，因而也可区分
 * `+0` 与 `-0`。
 */
inline constexpr std::uint32_t kSignMask = 0x80000000U;

/**
 * @brief IEEE 754 binary32 的指数位掩码。
 *
 * 实际 bit 为 `0x7f800000`，覆盖 bits `[30:23]`。指数全 1 表示 Inf 或
 * NaN；指数全 0 表示零或 subnormal。
 */
inline constexpr std::uint32_t kExponentMask = 0x7f800000U;

/**
 * @brief IEEE 754 binary32 的 fraction 位掩码。
 *
 * 实际 bit 为 `0x007fffff`，覆盖 bits `[22:0]`。
 */
inline constexpr std::uint32_t kFractionMask = 0x007fffffU;

/**
 * @brief IEEE 754 binary32 正无穷的 bit pattern。
 *
 * `std::numeric_limits<float>::infinity()` 表达“正无穷”的 C++ 语义；
 * 此常量则固定其 IEEE 754 binary32 表示为 `0x7f800000`。
 */
inline constexpr std::uint32_t kPositiveInfinityBits = 0x7f800000U;

/**
 * @brief 项目使用的 canonical quiet NaN 的 bit pattern。
 *
 * `std::numeric_limits<float>::quiet_NaN()` 只保证返回 NaN，并不承诺
 * payload 的具体 bit。codec 需要 CPU/GPU 一致的确定性结果，因此固定
 * 选用 `0x7fc00000`：指数为全 1，quiet bit（fraction bit 22）为 1。
 */
inline constexpr std::uint32_t kCanonicalQuietNaNBits = 0x7fc00000U;

/**
 * @brief FP32 的最大有限正数。
 *
 * 使用 `numeric_limits` 表达数值语义；在 IEEE 754 binary32 中，它的
 * 实际 bit 是 `0x7f7fffff`，即 E=254、fraction 全 1。
 */
inline constexpr float kMaxFinite = std::numeric_limits<float>::max();

/**
 * @brief FP32 的最小 normal 正数。
 *
 * 使用 `numeric_limits<float>::min()`，注意它不是最小正数，而是最小
 * normal 值 $2^{-126}$；实际 bit 是 `0x00800000`。
 */
inline constexpr float kMinNormal = std::numeric_limits<float>::min();

/**
 * @brief FP32 的最小 subnormal 正数。
 *
 * 使用 `numeric_limits<float>::denorm_min()`；其数值是 $2^{-149}$，
 * 实际 bit 是 `0x00000001`。
 */
inline constexpr float kMinSubnormal =
    std::numeric_limits<float>::denorm_min();

/**
 * @brief FP32 正无穷。
 *
 * 使用标准库表达返回值语义；等价的 IEEE 754 bit pattern 为
 * `kPositiveInfinityBits`。
 */
inline constexpr float kPositiveInfinity =
    std::numeric_limits<float>::infinity();

/**
 * @brief 小于 1 的最大可表示 FP32。
 *
 * 它是 $1-2^{-24}$，实际 bit 是 `0x3f7fffff`。stochastic rounding 用它
 * 将错误输入的 1.0 收敛至开区间右端，而不改变合法的 [0, 1) 输入。
 */
inline constexpr float kLargestBelowOne = 0x1.fffffep-1F;

/**
 * @brief 将 FP32 按位转换为 uint32，不改变任何 bit。
 *
 * device 编译时使用 CUDA intrinsic；host 编译时使用 memcpy，避免违反
 * strict aliasing 规则。两条路径都保留 NaN payload 与 -0 的符号位。
 *
 * @param value 待转换的 FP32 值。
 * @return 与 value 具有相同 IEEE 754 binary32 bit pattern 的整数。
 */
[[nodiscard]] __host__ __device__ inline std::uint32_t float_to_bits(
    const float value) noexcept {
#if defined(__CUDA_ARCH__)
    return __float_as_uint(value);
#else
    std::uint32_t bits = 0U;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
#endif
}

/**
 * @brief 将 uint32 按位转换为 FP32，不改变任何 bit。
 *
 * @param bits IEEE 754 binary32 原始 bit pattern。
 * @return 与 bits 具有相同 bit pattern 的 FP32 值。
 */
[[nodiscard]] __host__ __device__ inline float bits_to_float(
    const std::uint32_t bits) noexcept {
#if defined(__CUDA_ARCH__)
    return __uint_as_float(bits);
#else
    float value = 0.0F;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
#endif
}

/**
 * @brief 返回 FP32 是否为有限值。
 *
 * 这里不调用 `std::isfinite`，因为 codec 已经需要读取 exponent field，
 * 直接判断能在 host/device 上给出完全相同的位级语义。
 *
 * @param value 待判断的 FP32 值。
 * @return value 不是 NaN 且不是正负 Inf 时返回 true。
 */
[[nodiscard]] __host__ __device__ inline bool is_finite(
    const float value) noexcept {
    return (float_to_bits(value) & kExponentMask) != kExponentMask;
}

/**
 * @brief 返回 FP32 的符号位是否为 1。
 *
 * @param value 待检查的 FP32 值。
 * @return sign bit 为 1 时返回 true；-0 也会返回 true。
 */
[[nodiscard]] __host__ __device__ inline bool has_negative_sign(
    const float value) noexcept {
    return (float_to_bits(value) & kSignMask) != 0U;
}

/**
 * @brief 清除 FP32 符号位，得到非负幅值。
 *
 * @param value 任意 FP32 值。
 * @return value 的绝对值；NaN payload 保持不变。
 */
[[nodiscard]] __host__ __device__ inline float absolute_value(
    const float value) noexcept {
    return bits_to_float(float_to_bits(value) & ~kSignMask);
}

/**
 * @brief 构造项目固定的 canonical FP32 quiet NaN。
 *
 * @return bit pattern 为 `0x7fc00000` 的 quiet NaN。
 */
[[nodiscard]] __host__ __device__ inline float canonical_quiet_nan() noexcept {
    return bits_to_float(kCanonicalQuietNaNBits);
}

/**
 * @brief 构造带指定符号的 IEEE 754 零。
 *
 * @param negative 为 true 时构造 -0，否则构造 +0。
 * @return 对应符号的 FP32 零。
 */
[[nodiscard]] __host__ __device__ inline float signed_zero(
    const bool negative) noexcept {
    return bits_to_float(negative ? kSignMask : 0U);
}

/**
 * @brief 返回非零 uint32 的最高置位 bit 下标。
 *
 * @param value 非零无符号整数。
 * @return 最高置位 bit 的零起始下标。
 */
[[nodiscard]] __host__ __device__ inline int highest_set_bit_index(
    std::uint32_t value) noexcept {
    int index = 0;

    while (value > 1U) {
        value >>= 1U;
        ++index;
    }

    return index;
}

/**
 * @brief 计算正且有限 FP32 的 floor(log2(value))。
 *
 * normal 值的 unbiased exponent 是 exponent field 减 127。subnormal
 * 没有隐含 1，真实数值为 `fraction * 2^-149`，故需要寻找 fraction 的
 * 最高置位 bit。
 *
 * @param positive_value 必须为正且有限。
 * @return floor(log2(positive_value))。
 */
[[nodiscard]] __host__ __device__ inline int floor_log2_positive(
    const float positive_value) noexcept {
    const std::uint32_t bits = float_to_bits(positive_value);
    const std::uint32_t exponent_field = (bits >> 23U) & 0xffU;

    if (exponent_field != 0U) {
        return static_cast<int>(exponent_field) - 127;
    }

    const std::uint32_t fraction = bits & kFractionMask;
    return highest_set_bit_index(fraction) - 149;
}

/**
 * @brief 判断正且有限 FP32 是否恰为 2 的整数次幂。
 *
 * normal 数要求 fraction 为 0；subnormal 数要求 fraction 恰有一个置位
 * bit，例如 $2^{-127}$ 的 bit pattern 是 `0x00400000`。
 *
 * @param positive_value 必须为正且有限。
 * @return value 为 2^k 时返回 true。
 */
[[nodiscard]] __host__ __device__ inline bool is_power_of_two_positive(
    const float positive_value) noexcept {
    const std::uint32_t bits = float_to_bits(positive_value);
    const std::uint32_t exponent_field = (bits >> 23U) & 0xffU;
    const std::uint32_t fraction = bits & kFractionMask;

    if (exponent_field != 0U) {
        return fraction == 0U;
    }

    return fraction != 0U && (fraction & (fraction - 1U)) == 0U;
}

/**
 * @brief 构造范围 [-127, 127] 内的精确 FP32 幂 $2^{exponent}$。
 *
 * `numeric_limits<float>::infinity()` 用于表达越过 FP32 范围后的结果。
 * 对 E8M0 关键的 $2^{-127}$，必须显式构造 bit `0x00400000`：它是
 * subnormal，不能简单令 exponent field 为 0 后得到正确值。
 *
 * @param exponent 目标整数指数。
 * @return $2^{exponent}$；过小返回 +0，过大返回 +Inf。
 */
[[nodiscard]] __host__ __device__ inline float power_of_two(
    const int exponent) noexcept {
    constexpr std::uint32_t kMinE8M0ScaleBits = 0x00400000U;

    if (exponent < -127) {
        return 0.0F;
    }

    if (exponent == -127) {
        return bits_to_float(kMinE8M0ScaleBits);
    }

    if (exponent > 127) {
        return kPositiveInfinity;
    }

    const std::uint32_t exponent_field =
        static_cast<std::uint32_t>(exponent + 127);
    return bits_to_float(exponent_field << 23U);
}

/**
 * @brief 对非负且有限的较小值执行 RNE（ties-to-even）。
 *
 * 此函数适合格式 codec 的 mantissa 与 subnormal 单位数舍入。调用方要
 * 保证结果不会超出 uint32 范围；它不适合作为任意 FP32 的通用整数转换。
 *
 * @param nonnegative_value 待舍入的非负有限值。
 * @return 最近整数；恰好在中点时选择偶整数。
 */
[[nodiscard]] __host__ __device__ inline std::uint32_t
round_to_nearest_even_nonnegative(const float nonnegative_value) noexcept {
    const std::uint32_t truncated =
        static_cast<std::uint32_t>(nonnegative_value);
    const float fractional = nonnegative_value - static_cast<float>(truncated);

    if (fractional > 0.5F) {
        return truncated + 1U;
    }

    if (fractional == 0.5F && (truncated & 1U) != 0U) {
        return truncated + 1U;
    }

    return truncated;
}

/**
 * @brief 将随机数归一到 stochastic rounding 所需的 [0, 1) 区间。
 *
 * @param uniform_random 候选随机数。
 * @return 合法随机数保持不变；NaN、负数映射为 0，1 及更大值映射为
 *         `1-2^-24`。
 */
[[nodiscard]] __host__ __device__ inline float clamp_uniform_random(
    const float uniform_random) noexcept {
    if (!is_finite(uniform_random) || uniform_random <= 0.0F) {
        return 0.0F;
    }

    if (uniform_random >= 1.0F) {
        return kLargestBelowOne;
    }

    return uniform_random;
}

}  // namespace quant_dequant::formats::fp32
