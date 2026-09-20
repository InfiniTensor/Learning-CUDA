#pragma once

// 模块接口：声明单值格式转换和FP4打包工具，供量化/恢复模块共享。
// 这里只描述可调用的接口；具体E4M3/E8M0与打包实现见src/formats.cpp。

#include <cstdint>

namespace luxorion {
// 单值格式工具：encode把浮点数变成存储编码，decode把编码还原成浮点数。
// E4M3用于MXFP8元素和NVFP4块scale，E8M0用于MXFP8块scale。
// E2M1单值编解码声明目前放在nvfp4.h，本文件仅声明它的pack/unpack工具。

enum class QuantFormat { Mxfp8, Nvfp4 };

// E4M3：S1 E4 M3，bias=7；最近邻 ties-to-even，有限输入饱和到 +/-448。
// 保留正负零；NaN/Inf 输入抛出 std::invalid_argument。
std::uint8_t encode_e4m3(float value);
// 全部 256 个字节均有定义；0x7F/0xFF 解码为 NaN。
float decode_e4m3(std::uint8_t code);

// E8M0：精确编码正的 2 的幂，指数范围 -127..127，bias=127。
// 零、负数、非有限值、非 2 的幂或超范围输入抛出 std::invalid_argument。
std::uint8_t encode_e8m0(float scale);
// 0..254 返回 2^(code-127)，255 返回 NaN；编码 0 不表示零。
float decode_e8m0(std::uint8_t code);

// 两个 E2M1 编码各须在 0..15；非法输入抛出 std::invalid_argument。
// first 放高四位，second 放低四位。
std::uint8_t pack_e2m1(std::uint8_t first, std::uint8_t second);
// 接受全部字节；通过两个不同的输出变量返回 first、second 编码。
void unpack_e2m1(std::uint8_t packed, std::uint8_t& first,
                 std::uint8_t& second);

}  // namespace luxorion
