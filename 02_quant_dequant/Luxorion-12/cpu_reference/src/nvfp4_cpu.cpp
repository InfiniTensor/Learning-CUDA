/*
 * 模块用途：在CPU上模拟NVFP4的两级缩放、四位元素量化、打包及反量化。
 * 解决的问题：E2M1元素范围和精度较小，需要全局与块scale配合表示输入；
 *              四位编码还需按两元素一字节存储，并明确尾块有效长度。
 * 输入：非空、有限的连续FP32数组。
 * 量化输出：NVFP4Result，保存有效长度、packed字节、E4M3局部scale编码、FP32全局scale。
 * 反量化输出：仅包含有效元素的FP32近似值数组。
 * 主流程：input -> 两级scale/元素编码/打包 -> NVFP4Result -> 解包/解码/恢复 -> metrics。
 * 依赖：formats.cpp提供E4M3和pack/unpack；本文件还实现E2M1单值查表转换。
 * 存储约定：每16元素块固定8字节，尾块补零；这也是未来CUDA必须遵守的布局。
 */
#include "nvfp4.h"
#include "formats.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace luxorion {
// 本文件分两层：先用小查表实现单个 E2M1 数的编解码，再实现 NVFP4 分块流程。
// NVFP4 一份输入共用一个 FP32 global scale，每 16 个元素共用一个 E4M3 local scale。
// 单个元素缩放后的值保存为 E2M1 编码；两个四位编码拼成一个字节。

namespace {
constexpr std::array<float, 8> kMagnitude = {
    0.0F, 0.5F, 1.0F, 1.5F, 2.0F, 3.0F, 4.0F, 6.0F};
// kMagnitude 的下标 0..7 就是去掉符号后的三位编码，而不是浮点值本身。
// 例如下标 5（二进制 101）表示幅值 3；第四位单独记录正负号。
constexpr std::size_t kBlockSize = 16;
constexpr std::size_t kBytesPerBlock = 8;

// 用途：统一NVFP4存储尺寸，决定局部scale数量和固定8字节块的数量。
// 输入：有效元素数；输出：16元素量化块数，尾块也计作完整存储块。
std::size_t block_count(std::size_t size) {
  // 整数除法加尾块标记，得到 ceil(size/16)；N=19 时有两个块。
  return size / kBlockSize + (size % kBlockSize != 0);
}

// 用途：把全局与局部两层scale合成一个实际用于元素乘除的FP32系数。
// 输入：global浮点数与局部scale编码；输出：global*decode(local)的FP32数值。
// 调用位置：量化和反量化共用，防止两条路径采用不同精度或运算顺序。
float effective_scale(float global, std::uint8_t block_code) {
  // block_code 是存储编码，必须先解码成 local scale，才能与 global 相乘。
  // 返回 FP32 乘积；量化与恢复共用此函数，固定计算顺序和精度。
  // input/(global*local) 与 (input/global)/local 在浮点舍入下不保证相同。
  const float scale = global * decode_e4m3(block_code);
  if (scale == 0.0F) {
    // 两个 scale 各自非零，乘积仍可能因 FP32 下溢而变成零；不能拿它做除数。
    throw std::underflow_error("NVFP4 effective scale underflows to zero");
  }
  if (!std::isfinite(scale)) {
    throw std::overflow_error("NVFP4 effective scale exceeds the FP32 range");
  }
  return scale;
}
}  // namespace

// 用途：将一个缩放后的元素近似为E2M1可表示值，并给出四位存储编码。
// 输入：有限浮点数；输出：0..15编码，最近邻/ties-to-even且有限幅值饱和到6。
// 调用位置：NVFP4量化先完成scale除法，再调用此函数；之后才执行pack。
std::uint8_t encode_e2m1(float value) {
  // 输入是单个待编码浮点数；输出在 0..15，只有最低四位有效。
  // 这是 NVFP4 元素编码工具，也可以独立测试，不负责 scale 的选择。
  if (!std::isfinite(value)) {
    throw std::invalid_argument("E2M1 encoding requires a finite input");
  }
  const bool negative = std::signbit(value);
  const std::uint8_t sign = negative ? 0x8U : 0U;
  // signbit 保留 -0；0x8=1000 将符号放在四位编码的最高位 bit3。
  const float magnitude = std::fabs(value);
  // 先处理饱和，避免大数与不同候选的距离因有限精度变得相同。
  if (magnitude >= kMagnitude.back()) {
    return static_cast<std::uint8_t>(sign | 0x7U);
  }
  std::size_t best = 0;
  double best_distance = std::numeric_limits<double>::infinity();
  // 遍历八个正幅值候选，寻找绝对距离最小者。
  // best 保存的是候选下标（编码），best_distance 保存距离，两者含义不同。
  // infinity 让第一个候选总能成为初始 best。
  for (std::size_t i = 0; i < kMagnitude.size(); ++i) {
    // 输入仍是 FP32；用 double 比较距离，避免中点邻值被额外舍入成平局。
    const double distance =
        std::fabs(static_cast<double>(magnitude) - kMagnitude[i]);
    // 表下标就是幅值编码；偶数下标的最低位为 0。
    // 条件分两部分：距离更小就更新；距离相同时，仅用偶数编码替换奇数编码。
    // 例：2.5 与 2（编码4）和 3（编码5）同距，选择编码4 -> 数值2。
    // ties-to-even 指保留编码的最低位为0，不是要求浮点数的整数部分为偶数。
    if (distance < best_distance ||
        (distance == best_distance && i % 2 == 0 && best % 2 != 0)) {
      best = i;
      best_distance = distance;
    }
  }
  // 幅值占低三位、符号占第四位，按位或将两者拼接。
  // 例：-3 的幅值编码为5，5 | 8 = 13 = 0xD。
  return static_cast<std::uint8_t>(best | sign);
}

// 用途：查回四位编码表示的有符号浮点数，是元素恢复的单值基础工具。
// 输入：0..15编码；输出：一个E2M1可表示float值，尚未乘scale。
// 调用位置：NVFP4反量化先unpack，再decode，最后乘有效scale。
float decode_e2m1(std::uint8_t code) {
  // 掩码 0x7=0111 提取幅值表下标，0x8=1000 检查符号。
  // 例：0xD -> 下标5 -> 幅值3 -> -3；编码0x8则恢复为 -0。
  if (code > 0xFU) {
    throw std::invalid_argument("E2M1 decoding requires a code in 0..15");
  }
  const float magnitude = kMagnitude[code & 0x7U];
  return (code & 0x8U) ? -magnitude : magnitude;
}
// 用途：生成可独立保存和恢复的NVFP4表示，作为GPU量化的对照结果。
// 输入：有限FP32数组；输出：NVFP4Result四项字段，包括尾块零填充。
// 步骤：求全局amax/scale -> 按16分块求局部amax/scale -> 元素缩放/编码 -> 打包。
// 主流程关系：这里产出编码负载；误差要在反量化恢复出数值后才能计算。
NVFP4Result nvfp4_quantize_cpu(const std::vector<float>& input) {
  // 输入是连续 FP32 数组；先遍历整个数组，检查输入并计算全局最大绝对值。
  // 再逐块计算局部最大绝对值。global_amax 与 block_amax 的范围不同。
  if (input.empty()) {
    throw std::invalid_argument("NVFP4 quantization requires a nonempty input");
  }
  float global_amax = 0.0F;
  for (float value : input) {
    if (!std::isfinite(value)) {
      throw std::invalid_argument("NVFP4 quantization requires finite inputs");
    }
    global_amax = std::max(global_amax, std::fabs(value));
  }

  NVFP4Result result;
  result.element_count = input.size();
  // 2688 = E4M3 最大有限 scale 448 × E2M1 最大幅值 6。
  // global_scale 将全局动态范围分配到这两层表示范围；全零输入约定取1。
  // 例如 global_amax=2688 时 global_scale=1，便于手工检查后面的局部 scale。
  result.global_scale = global_amax == 0.0F ? 1.0F
      : static_cast<float>(static_cast<double>(global_amax) / 2688.0);
  if (result.global_scale == 0.0F) {
    throw std::underflow_error("NVFP4 global scale underflows to zero");
  }
  const std::size_t blocks = block_count(input.size());
  // element_count 保存有效长度，不能从 packed_data 长度推回原始长度，
  // 因为最后一块也固定存8字节。N=19与N=32的 packed_data 都是16字节。
  result.block_scales.resize(blocks);
  // 初始化完整存储为零；尾块只写有效元素，剩余填充保持为零。
  result.packed_data.resize(blocks * kBytesPerBlock, 0);
  for (std::size_t b = 0; b < blocks; ++b) {
    const std::size_t begin = b * kBlockSize;
    const std::size_t end = begin + std::min(kBlockSize, input.size() - begin);
    // 第 b 块处理 [16*b,end)。这里没有额外的拷贝或重排，只通过下标访问。
    float block_amax = 0.0F;
    for (std::size_t i = begin; i < end; ++i) {
      block_amax = std::max(block_amax, std::fabs(input[i]));
    }
    // 先用高精度计算 ideal，再转 FP32 后编码，顺序作为数值约定。
    // ideal = (block_amax / 6) / global_scale，是理想的局部 scale。
    // 但它未必能被 E4M3 精确表示；encode 后的实际 scale 才用于元素量化。
    // 例如 global=1、block_amax=6：ideal=1，编码0x38，实际local=1。
    // 全零块的 local 取1，此时 effective=global，仍能稳定编码零。
    const float ideal = block_amax == 0.0F ? 1.0F
        : static_cast<float>((static_cast<double>(block_amax) / 6.0)
                             / result.global_scale);
    std::uint8_t block_code = encode_e4m3(ideal);
    // 非零块的局部 scale 舍入为零时，使用最小正 E4M3 scale。
    // 编码1表示 2^-9；这是一项防止分母为零的项目策略，不表示数值1。
    if (block_amax != 0.0F && block_code == 0) block_code = 1;
    result.block_scales[b] = block_code;
    const float scale = effective_scale(result.global_scale, block_code);
    // 每轮处理相邻两个元素，从而只写一次 packed 字节。
    // begin 是16的倍数，所以 i 总为偶数，第一个元素对应高四位。
    for (std::size_t i = begin; i < end; i += 2) {
      const auto first = encode_e2m1(input[i] / scale);
      const auto second = i + 1 < end ? encode_e2m1(input[i + 1] / scale)
                                    : std::uint8_t{0};
      // 若有效长度为奇数，最后一个字节只有高四位有效，低四位主动写0。
      // i=16时写byte8；i=18时写byte9；块内字节关系也可写成 b*8+(i-b*16)/2。
      result.packed_data[i / 2] = pack_e2m1(first, second);
    }
  }
  return result;
}

// 用途：从NVFP4保存的四项信息恢复有效元素，验证scale、打包和尾块布局。
// 输入：NVFP4Result；输出：element_count个FP32近似值，不需要原始数组。
// 步骤：检查尺寸/scale/填充 -> 合成有效scale -> 解包 -> E2M1解码 -> 相乘。
// 主流程关系：输出用于误差计算，并作为未来GPU反量化的正确性基准。
std::vector<float> nvfp4_dequantize_cpu(const NVFP4Result& q) {
  // 恢复前校验外部结果的结构、scale和填充；这些检查不重新做量化。
  const std::size_t blocks = block_count(q.element_count);
  if (q.element_count == 0 || q.block_scales.size() != blocks
      || q.packed_data.size() != blocks * kBytesPerBlock) {
    throw std::invalid_argument("NVFP4 element count and payload sizes are inconsistent");
  }
  if (!std::isfinite(q.global_scale) || q.global_scale <= 0.0F) {
    throw std::invalid_argument("NVFP4 global scale must be positive and finite");
  }
  for (auto code : q.block_scales) {
    // 合法正 E4M3 scale 的编码是 1..126；排除零、负数和 NaN。
    if (code == 0 || code > 0x7EU) {
      throw std::invalid_argument("NVFP4 block scales must be positive and finite");
    }
  }
  // 检查奇数尾元素之后的低四位，以及后面的完整填充字节。
  const std::size_t used_bytes = q.element_count / 2 + (q.element_count % 2 != 0);
  // used_bytes=ceil(N/2)，只计包含有效元素的字节，不等于完整存储字节数。
  // N=19时 used_bytes=10：byte9低四位为填充，byte10..15整字节为填充。
  if (q.element_count % 2 != 0 && (q.packed_data[used_bytes - 1] & 0xFU) != 0) {
    throw std::invalid_argument("NVFP4 tail padding must be zero");
  }
  for (std::size_t byte = used_bytes; byte < q.packed_data.size(); ++byte) {
    if (q.packed_data[byte] != 0) {
      throw std::invalid_argument("NVFP4 tail padding must be zero");
    }
  }

  std::vector<float> output(q.element_count);
  // 仅为有效元素分配输出，因此补零元素不参与恢复结果或误差统计。
  for (std::size_t b = 0; b < blocks; ++b) {
    const float scale = effective_scale(q.global_scale, q.block_scales[b]);
    const std::size_t begin = b * kBlockSize;
    const std::size_t end = begin + std::min(kBlockSize, q.element_count - begin);
    for (std::size_t i = begin; i < end; i += 2) {
      std::uint8_t first = 0;
      std::uint8_t second = 0;
      unpack_e2m1(q.packed_data[i / 2], first, second);
      // first/second 仍是四位编码：先查表解码，再乘两级scale的FP32乘积。
      // 例如 byte8=0x7D -> 编码7和13 -> 数值6和-3；effective=1则原样恢复。
      const float restored_first = decode_e2m1(first) * scale;
      if (!std::isfinite(restored_first)) {
        throw std::overflow_error("NVFP4 reconstruction exceeds the FP32 range");
      }
      output[i] = restored_first;
      if (i + 1 < end) {
        // 奇数尾元素的低四位虽然是零，也不能写到 output[N]，否则越界。
        const float restored_second = decode_e2m1(second) * scale;
        if (!std::isfinite(restored_second)) {
          throw std::overflow_error("NVFP4 reconstruction exceeds the FP32 range");
        }
        output[i + 1] = restored_second;
      }
    }
  }
  return output;
}
}  // namespace luxorion
