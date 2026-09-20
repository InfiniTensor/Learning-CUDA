/*
 * 模块用途：提供低精度格式的基础转换能力，统一项目的舍入与编码规则。
 * 解决的问题：FP32数值不能直接作为低精度字节存储，必须按格式舍入并编码；
 *              恢复时又必须按同一格式解释这些比特。
 * 输入/输出：单个浮点数 <-> 单个编码；两个E2M1编码 <-> 一个打包字节。
 * 主流程位置：由MXFP8/NVFP4量化和反量化调用，也由演示入口用于解释日志。
 * 职责边界：只处理单值格式和打包；分块、amax与scale选择由上层CPU模块负责。
 * E2M1单值查表编解码目前在nvfp4_cpu.cpp中，本文件提供E4M3、E8M0与打包工具。
 */
#include "formats.h"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace luxorion {
// 本文件只负责「数值 <-> 格式编码」以及两个 FP4 编码的打包。
// 它不负责按块求最大值、选择 scale；这些策略在两个 *_cpu.cpp 中。
// uint8_t 在这里是存储比特的容器，不能把编码当作普通整数参与浮点运算。
namespace {
constexpr int kBias = 7;                 // E4M3：真实指数 = 指数字段 - 7。
constexpr double kMaxFinite = 448.0;     // E4M3 最大有限幅值，编码为 0x7E。
constexpr double kMinNormal = 0x1p-6;    // 十六进制浮点写法，表示 2^-6。
constexpr int kE8Bias = 127;             // E8M0：编码 = 真实指数 + 127。

// 仅用于非负、小范围数值；不依赖进程的浮点舍入模式。
// 用途：为E4M3编码统一执行最近邻舍入，避免各分支采用不同的中点规则。
// 输入：已换算到尾数/有效数字整数刻度的非负double；输出：舍入后的整数。
int round_nearest_even(double value) {
  // 将 value 拆成下方整数 lower 和小数部分 fraction。
  // 小于半格取 lower，大于半格取 lower+1；正好半格选择偶数。
  // lower & 1 检查整数最低位：1 表示奇数，此时上方整数才是偶数。
  // 例如 15.5 -> 16，16.5 -> 16；这里舍入的是缩放后的有效数字整数。
  const int lower = static_cast<int>(std::floor(value));
  const double fraction = value - lower;
  if (fraction > 0.5 || (fraction == 0.5 && (lower & 1))) {
    return lower + 1;
  }
  return lower;
}
}  // namespace

// 用途：模拟把一个FP32值转换为E4M3，产生真正可以保存的8位编码。
// 输入：有限浮点值；输出：一个字节，包含符号、指数和尾数。
// 调用位置：MXFP8保存缩放后的元素；NVFP4保存理想局部scale的近似值。
std::uint8_t encode_e4m3(float value) {
  // 本项目的编码接口只接收有限 FP32 输入，NaN/Inf 用异常拒绝。
  // 这是当前 reference 的输入约定，并非实现完整的特殊值转换接口。
  if (!std::isfinite(value)) {
    throw std::invalid_argument("E4M3 encoding requires a finite input");
  }
  const unsigned sign = std::signbit(value) ? 0x80U : 0U;
  // 8 位布局：S EEEE MMM。符号占 bit7，指数占 bit6..3，尾数占 bit2..0。
  // signbit 能识别 -0；value < 0 则不能，因此这里使用 signbit。
  // FP32 输入转 double 不改变其数值；下面只用 2 的幂进行缩放。
  const double magnitude = std::fabs(static_cast<double>(value));
  if (magnitude >= kMaxFinite) {
    // 饱和到最大有限值；按位或只补上符号，不改变幅值编码。
    return static_cast<std::uint8_t>(sign | 0x7EU);
  }

  if (magnitude < kMinNormal) {
    // 次正规数：value = M * 2^-9，尾数就是以 2^-9 为单位的刻度。
    // 舍入到 8 时，编码 0x08 恰好进入最小规格化数，自动处理边界。
    // ldexp(x, 9) 等价于 x * 2^9；例如 2^-10 乘 512 得 0.5，
    // ties-to-even 取尾数 0。幅值为零也走这里，最终仍保留原符号。
    const int mantissa = round_nearest_even(std::ldexp(magnitude, 9));
    return static_cast<std::uint8_t>(sign | static_cast<unsigned>(mantissa));
  }

  // frexp 返回 magnitude = fraction * 2^power，其中 fraction 在 [0.5,1)。
  int power = 0;
  std::frexp(magnitude, &power);
  int exponent = power - 1;
  // frexp 的小数范围是 [0.5,1)，而格式使用 [1,2) 的有效数字，
  // 所以将 fraction 翻倍，同时把指数减 1。
  // 规格化有效数字在 [1,2)，乘 8 后舍入得到整数有效数字 8..16。
  int significand = round_nearest_even(std::ldexp(magnitude, 3 - exponent));
  // significand 包含隐含的前导 1：8 对应 1.000，15 对应 1.111。
  // 例：15.5 的 exponent=3，缩放后仍为 15.5，舍入得到 16，进入下面进位分支。
  if (significand == 16) {
    // 1.111... 舍入成 10.000：指数加一，再恢复成 1.000。
    ++exponent;
    significand = 8;
  }
  const unsigned exponent_field = static_cast<unsigned>(exponent + kBias);
  const unsigned mantissa = static_cast<unsigned>(significand - 8);
  // 去掉隐含前导 1（整数形式为 8）后，mantissa 才是存储的三位尾数。
  // exponent_field << 3 为尾数留出三位；三个字段互不重叠，可用 | 拼接。
  // 15.5 最终 exponent=4，字段=11，尾数=0：0 | (11 << 3) | 0 = 0x58。
  return static_cast<std::uint8_t>(sign | (exponent_field << 3) | mantissa);
}

// 用途：解释一个E4M3字节表示的数值，供恢复计算或日志展示使用。
// 输入：8位编码；输出：可表示的float值，两个NaN编码返回NaN。
// 主流程关系：MXFP8恢复元素幅值；NVFP4得到实际保存的局部scale。
float decode_e4m3(std::uint8_t code) {
  // 右移使目标字段落到最低位，再用掩码保留所需位数。
  // 0xF=1111 保留四位指数，0x7=111 保留三位尾数。
  const unsigned exponent_field = (code >> 3) & 0xFU;
  const unsigned mantissa = code & 0x7U;
  if (exponent_field == 15 && mantissa == 7) {
    // 0x7F 和 0xFF 是 NaN；E4M3 的其他编码都是有限值，包括有符号零。
    return std::numeric_limits<float>::quiet_NaN();
  }
  // 指数字段为零：没有隐含前导 1，幅值 = M * 2^-9。
  // 指数字段非零：幅值 = (1 + M/8) * 2^(E-7)。
  // 例如 0x79：E=15、M=1，得到 1.125 * 2^8 = 288。
  const double magnitude = exponent_field == 0
      ? std::ldexp(static_cast<double>(mantissa), -9)
      : std::ldexp(1.0 + mantissa / 8.0,
                   static_cast<int>(exponent_field) - kBias);
  // copysign 把符号位贴回幅值，也能正确恢复编码 0x80 所表示的 -0。
  return std::copysign(static_cast<float>(magnitude),
                       (code & 0x80U) ? -1.0F : 1.0F);
}
// 用途：将MXFP8已选好的scale压缩成一个字节，以便一块只保存一份scale。
// 输入：范围内精确的正2的幂；输出：带127偏移的指数编码。
// 主流程关系：由MXFP8的select_scale调用，不承担向上选择scale的策略。
std::uint8_t encode_e8m0(float scale) {
  // 这里只编码「已经选好的」scale，不会把任意正数自动舍入成 2 的幂。
  // 例如 2 可直接编码，1.3 会报错；MXFP8 select_scale 负责把需求向上取幂。
  if (!std::isfinite(scale) || scale <= 0.0F) {
    throw std::invalid_argument("E8M0 encoding requires a positive finite scale");
  }
  int power = 0;
  const float fraction = std::frexp(scale, &power);
  // 正的 2 的幂恰好拆成 0.5 * 2^power；直接检查，不使用 log2。
  if (fraction != 0.5F) {
    throw std::invalid_argument("E8M0 encoding requires an exact power of two");
  }
  const int exponent = power - 1;
  if (exponent < -127 || exponent > 127) {
    throw std::invalid_argument("E8M0 exponent must be in -127..127");
  }
  // 例：scale=2=2^1 -> 编码 1+127=128=0x80；scale=1 -> 0x7F。
  return static_cast<std::uint8_t>(exponent + kE8Bias);
}

// 用途：将MXFP8块scale编码恢复为实际乘除用的数值。
// 输入：E8M0编码；输出：float scale，0xFF返回NaN。
// 主流程关系：量化时用于除法，反量化时用于乘法，两端使用相同scale。
float decode_e8m0(std::uint8_t code) {
  // E8M0 没有符号位和尾数；整个字节保存带偏移的指数。
  // 特别注意编码 0 表示 2^-127，而不是数值零；只有 255 保留给 NaN。
  if (code == 0xFFU) {
    return std::numeric_limits<float>::quiet_NaN();
  }
  return std::ldexp(1.0F, static_cast<int>(code) - kE8Bias);
}
// 用途：让两个四位元素共用一个字节，落实NVFP4的紧凑元素存储。
// 输入：两个合法E2M1编码；输出：高四位为first、低四位为second的字节。
// 调用位置：NVFP4量化在完成两个元素编码后调用。
std::uint8_t pack_e2m1(std::uint8_t first, std::uint8_t second) {
  // 这里只拼接两个已有的四位编码，不进行量化，也不解释其浮点含义。
  if (first > 0xFU || second > 0xFU) {
    throw std::invalid_argument("E2M1 packing requires two codes in 0..15");
  }
  // first=0x7、second=0xD：0111 << 4 与 1101 拼成 01111101，即 0x7D。
  // 本项目约定第一个元素在高四位，第二个元素在低四位。
  return static_cast<std::uint8_t>((first << 4) | second);
}

// 用途：从存储字节中取回两个元素的编码，供后续E2M1解码。
// 输入：打包字节；输出：通过first/second引用返回两个四位编码。
// 调用位置：NVFP4反量化和日志展示；它本身还没有恢复浮点值。
void unpack_e2m1(std::uint8_t packed, std::uint8_t& first,
                 std::uint8_t& second) {
  // 引用参数用于把两个结果交给调用者；调用时需传两个不同变量。
  // packed=0x7D 时，右移取出 0x7，直接掩码取出 0xD。
  first = static_cast<std::uint8_t>((packed >> 4) & 0xFU);
  second = static_cast<std::uint8_t>(packed & 0xFU);
}
}  // namespace luxorion
