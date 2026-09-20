// =============================================================================
//  config.hpp -- 量化配置(对应题目要求的「量化参数文件」)
// =============================================================================
#pragma once

#include <cstdint>
#include <string>

#include "lpq/formats.hpp"

namespace lpq {

enum class QuantFormat : int {
    MxFP8 = 0,  // MXFP8 : 32 元素共享一个 E8M0(2 的整数次幂)缩放因子, 元素 e4m3 / e5m2
    NVFP4 = 1,  // NVFP4 : 16 元素一组,E4M3 局部缩放 + FP32 全张量缩放, 元素 e2m1
    MxFP4 = 2,  // MXFP4 : 32 元素共享一个 E8M0 缩放因子, 元素 e2m1
};

// 块缩放因子的选取规则(即「4/6 自适应块缩放」的可选档位).
//
// 只有 NVFP4 采用「E4M3 块缩放 + FP32 全局缩放」的两级结构, 块缩放因子有 3 位尾数,
// 足以表达非 2 的幂的 1.5 倍扩张; MX 家族的缩放因子是 E8M0(只有 8 位指数), 表达不了
// 4/6 需要的 1.5 倍, 因此自适应规则只对 NVFP4 生效, MX 格式只接受 Static6.
enum class ScaleRule : int {
    Static6 = 0,  // 标准 NVFP4:每块把 amax 映射到 E2M1 的最大值 6.0
    Static4 = 1,  // 每块把 amax 映射到 4.0(即 4/6 里的「4」候选, 固定选它)
    AbsMax = 2,   // 在 4 / 6 两个候选里, 取「最大绝对误差」更小的那个
    Mae = 3,      //  ... 「平均绝对误差」更小的那个
    Mse = 4,      //  ... 「均方误差」更小的那个(4/6 论文的默认规则)
};

enum class ScaleMode : int {
    Tensor = 0,  // per-tensor:整个张量共用一个缩放因子
    Block = 1,   // block-wise:每 block_size 个元素一个缩放因子
};

enum class OutputType : int { FP16 = 0, BF16 = 1, FP32 = 2 };
enum class InputType : int { FP32 = 0, FP16 = 1 };

struct QuantConfig {
    QuantFormat format = QuantFormat::MxFP8;
    ScaleMode scale_mode = ScaleMode::Block;
    OutputType output_type = OutputType::FP16;
    Rounding rounding = Rounding::Nearest;
    int block_size = 32;  // mxfp8 / mxfp4 默认 32,nvfp4 默认 16
    bool block_size_explicit = false;
    bool mxfp8_e5m2 = false;                    // mxfp8 元素格式:false=e4m3(默认),true=e5m2
    ScaleRule scale_rule = ScaleRule::Static6;  // 仅 NVFP4 生效, 见 ScaleRule
    // 参数文件里 elem_format 被显式写成了什么:-1 未写, 0=e4m3, 1=e5m2, 2=e2m1.
    // 只用于在 apply_defaults 里做「格式与元素格式是否自洽」的检查.
    int elem_format_explicit = -1;
    std::string target_gpu = "unknown";  // 仅用于报告说明
    uint64_t seed = 20260915ull;         // 随机舍入的随机种子

    // 元素/缩放因子的位宽与编码
    int element_bits() const { return format == QuantFormat::MxFP8 ? 8 : 4; }
    // MX 家族(OCP Microscaling):每 block_size 个元素共享一个 E8M0 缩放因子
    bool is_mx() const { return format != QuantFormat::NVFP4; }
    // 4bit 元素:每字节打包 2 个元素
    bool is_4bit() const { return element_bits() == 4; }
    // 是否启用 4/6 自适应块缩放(需要两级缩放才能表达 1.5 倍扩张, 故仅 NVFP4)
    bool scale_rule_adaptive() const {
        return format == QuantFormat::NVFP4 && scale_rule != ScaleRule::Static6 &&
               scale_rule != ScaleRule::Static4;
    }
    // 依据格式补齐默认 block_size, 并做格式自洽性检查
    void apply_defaults();
};

// 解析量化参数文件;缺省项使用默认值
QuantConfig parse_quant_config(const std::string& path);
// 解析形如 "key = value" 的行(供测试直接使用)
void parse_quant_config_line(QuantConfig& cfg, const std::string& line);
// 输出为可读文本(写入日志)
std::string quant_config_to_string(const QuantConfig& cfg);

const char* to_string(QuantFormat v);
const char* to_string(ScaleRule v);
const char* to_string(ScaleMode v);
const char* to_string(OutputType v);
const char* to_string(Rounding v);
const char* to_string(InputType v);

OutputType output_type_from_string(const std::string& s);
InputType input_type_from_string(const std::string& s);

}  // namespace lpq