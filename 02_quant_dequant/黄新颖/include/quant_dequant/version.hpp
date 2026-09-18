#pragma once

namespace quant_dequant {

/**
 * @brief 返回核心库的语义版本字符串。
 *
 * 返回值指向静态存储期字符串，不需要调用方释放。
 *
 * @return 核心库版本号。
 */
[[nodiscard]] const char* version() noexcept;

}  // namespace quant_dequant
