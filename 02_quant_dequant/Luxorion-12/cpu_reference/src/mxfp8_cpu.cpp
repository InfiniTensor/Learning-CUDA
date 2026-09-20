/*
 * 模块用途：在CPU上模拟完整的MXFP8量化/反量化，作为未来CUDA实现的正确性基准。
 * 解决的问题：用每块共享的scale覆盖输入动态范围，再以E4M3近似保存各元素。
 * 输入：非空、有限的连续FP32数组；形状的展平与文件读取由调用者负责。
 * 量化输出：MXFP8Result，包含每元素E4M3编码和每32元素块的E8M0 scale编码。
 * 反量化输出：与原输入等长的FP32近似值数组，供指标模块比较误差。
 * 主流程：input -> quantize -> MXFP8Result -> dequantize -> restored -> metrics。
 * 依赖：formats.cpp负责单值格式转换；本模块负责分块、scale策略与数组处理。
 */
#include "mxfp8.h"
#include "formats.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace luxorion {
// 输入是已经展开成连续数组的 FP32 数值；这里按连续下标分块，
// 不读取文件，也不做矩阵转置或重新排列。块可能跨越矩阵的行边界。
// 流程：块内 amax -> E8M0 scale -> E4M3 元素编码；恢复时执行相反过程。
namespace {
constexpr std::size_t kBlockSize = 32;

// 用途：根据有效长度计算所需scale数量，量化分配和反量化校验共用这一规则。
// 输入：元素数；输出：32元素量化块的数量（包含不足32元素的尾块）。
std::size_t block_count(std::size_t size) {
  // 等价于 ceil(size / 32)，但这里执行整数除法，并为非空尾块多计一块。
  // 33 个元素有 2 块；避免使用 size+31，防止该加法对极大 size 溢出。
  return size / kBlockSize + (size % kBlockSize != 0);
}

// 用途：为一块选择实际可存储的scale，使缩放后的最大幅值不超过E4M3范围。
// 输入：本块最大绝对值amax；输出：E8M0编码，而不是float scale。
// 调用位置：quantize先求amax，再调用此函数，最后decode该编码进行元素量化。
std::uint8_t select_scale(float amax) {
  // 全零块不需要缩放，选 1 避免除零，并给出稳定、可复现的编码。
  if (amax == 0.0F) return encode_e8m0(1.0F);

  // double 避免 FP32 极小输入除以 448 后先下溢为零。
  const double required = static_cast<double>(amax) / 448.0;
  // 理想需求满足 amax/scale <= 448。实际 scale 只能是 E8M0 的 2 的幂，
  // 因此选择不小于 required 的最小可表示幂，而不是最近的幂。
  // 例如 amax=600，required≈1.3393，选择 scale=2，其编码为 0x80。
  // 若需求比最小 scale 还小，返回编码 0（实际数值 2^-127）。
  if (required <= std::ldexp(1.0, -127)) return 0;

  int power = 0;
  const double fraction = std::frexp(required, &power);
  // required 恰好是 2 的幂时保留它；否则选择更大的那个幂。
  const int exponent = fraction == 0.5 ? power - 1 : power;
  // required=1 时 frexp 得 0.5*2^1，选 2^0；required=1.3 时
  // frexp 得 0.65*2^1，选 2^1。这样在幂的边界处不会额外放大一倍。
  return encode_e8m0(std::ldexp(1.0F, exponent));
}
}  // namespace

// 用途：生成MXFP8低精度表示，模拟量化造成的精度损失与实际负载布局。
// 输入：有限FP32数组；输出：元素编码data与块scale编码scales。
// 步骤：检查输入 -> 按32分块 -> 求amax/选scale -> 缩放 -> E4M3编码。
// 主流程关系：输出可交给CPU反量化、未来GPU反量化或后续二进制写入模块。
MXFP8Result mxfp8_quantize_cpu(const std::vector<float>& input) {
  // 输入检查在计算之前完成；异常表示调用者需修正数据，不能忽略后继续计算。
  if (input.empty()) {
    throw std::invalid_argument("MXFP8 quantization requires a nonempty input");
  }
  for (float value : input) {
    if (!std::isfinite(value)) {
      throw std::invalid_argument("MXFP8 quantization requires finite inputs");
    }
  }

  MXFP8Result result;
  result.data.resize(input.size());
  result.scales.resize(block_count(input.size()));
  // data 与 input 一一对应，一元素一字节；scales 一块一字节。
  // 例如 N=33：data 有 33 字节、scales 有 2 字节，尾块只保存一个元素。
  for (std::size_t b = 0; b < result.scales.size(); ++b) {
    const std::size_t begin = b * kBlockSize;
    const std::size_t end = begin + std::min(kBlockSize, input.size() - begin);
    // 区间为 [begin,end)：begin 包含、end 不包含。
    // b=1、N=33 时只处理 [32,33)，不会访问不存在的 input[33]。
    float amax = 0.0F;
    for (std::size_t i = begin; i < end; ++i) {
      amax = std::max(amax, std::fabs(input[i]));
    }
    result.scales[b] = select_scale(amax);
    // amax 是本块最大绝对值；负数也需要按其幅值决定动态范围。
    // 使用实际保存的 scale，而不是理想 required。
    const float scale = decode_e8m0(result.scales[b]);
    // 编码和恢复必须使用同一个「实际保存的」scale。若用 required 量化、
    // 用舍入后的 scale 恢复，两边单位不同，即使元素编码正确也会得到错误结果。
    for (std::size_t i = begin; i < end; ++i) {
      const float normalized = input[i] / scale;
      // normalized 是仍为 FP32 的缩放后数值；encode 才执行低精度舍入。
      // 600 / 2 = 300 -> 编码 0x79，解码幅值为 288。
      result.data[i] = encode_e4m3(normalized);
    }
  }
  return result;
}

// 用途：仅凭保存的低精度表示恢复近似FP32数值，验证量化结果是否可独立使用。
// 输入：MXFP8Result；输出：与data等长的恢复数组，不需要原始输入。
// 步骤：校验结果 -> 查元素所属块 -> 解码元素和scale -> 相乘恢复。
// 主流程关系：恢复数组交给metrics，与原始输入比较；不能保证恢复为原值。
std::vector<float> mxfp8_dequantize_cpu(const MXFP8Result& q) {
  // 结果也可能来自外部文件或其他实现，因此恢复前必须校验结构与编码，
  // 不能假定输入一定由上面的 quantize 函数生成。
  if (q.data.empty() || q.scales.size() != block_count(q.data.size())) {
    throw std::invalid_argument("MXFP8 data length and scale count are inconsistent");
  }
  for (auto code : q.scales) {
    if (code == 0xFFU) throw std::invalid_argument("MXFP8 NaN scale is not supported");
  }
  for (auto code : q.data) {
    // 清掉符号位后检查 NaN 幅值编码，同时覆盖 0x7F 和 0xFF。
    if ((code & 0x7FU) == 0x7FU) {
      throw std::invalid_argument("MXFP8 NaN element is not supported");
    }
  }

  std::vector<float> output(q.data.size());
  for (std::size_t i = 0; i < output.size(); ++i) {
    const std::size_t block_id = i / kBlockSize;
    // 0..31 -> scale[0]，32..63 -> scale[1]；这一对应关系也会用于 CUDA。
    const float scale = decode_e8m0(q.scales[block_id]);
    const float restored = decode_e4m3(q.data[i]) * scale;
    // 恢复为 FP32：例如 decode(0x79)=288，乘 scale=2，得到 576。
    // FP32 极大输入在低精度舍入后可能恢复溢出，因此这里显式检查结果。
    if (!std::isfinite(restored)) {
      throw std::overflow_error("MXFP8 reconstruction exceeds the FP32 range");
    }
    output[i] = restored;
  }
  return output;
}
}  // namespace luxorion
