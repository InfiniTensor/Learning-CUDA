// =============================================================================
//  config.cu -- 量化参数文件的解析与格式化
// =============================================================================
#include "lpq/config.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>
#include <vector>

#include "lpq/common.hpp"

namespace lpq {
namespace {

std::string trim(const std::string& s) {
    size_t b = 0, e = s.size();
    while (b < e && std::isspace((unsigned char)s[b]))
        ++b;
    while (e > b && std::isspace((unsigned char)s[e - 1]))
        --e;
    return s.substr(b, e - b);
}

// 去掉首尾成对的引号,并统一转小写
std::string normalize_value(const std::string& s) {
    std::string v = trim(s);
    if (v.size() >= 2 &&
        ((v.front() == '"' && v.back() == '"') || (v.front() == '\'' && v.back() == '\''))) {
        v = v.substr(1, v.size() - 2);
    }
    v = trim(v);
    std::transform(
        v.begin(), v.end(), v.begin(), [](unsigned char c) { return (char)std::tolower(c); });
    return v;
}

}  // namespace

const char* to_string(QuantFormat v) {
    switch (v) {
        case QuantFormat::MxFP8:
            return "mxfp8";
        case QuantFormat::MxFP4:
            return "mxfp4";
        default:
            return "nvfp4";
    }
}
const char* to_string(ScaleRule v) {
    switch (v) {
        case ScaleRule::Static6:
            return "static_6";
        case ScaleRule::Static4:
            return "static_4";
        case ScaleRule::AbsMax:
            return "abs_max";
        case ScaleRule::Mae:
            return "mae";
        default:
            return "mse";
    }
}
const char* to_string(ScaleMode v) {
    return v == ScaleMode::Tensor ? "tensor" : "block";
}
const char* to_string(OutputType v) {
    switch (v) {
        case OutputType::FP16:
            return "fp16";
        case OutputType::BF16:
            return "bf16";
        default:
            return "fp32";
    }
}
const char* to_string(Rounding v) {
    return v == Rounding::Nearest ? "nearest" : "stochastic";
}
const char* to_string(InputType v) {
    return v == InputType::FP32 ? "fp32" : "fp16";
}

OutputType output_type_from_string(const std::string& s) {
    const std::string v = normalize_value(s);
    if (v == "fp16" || v == "float16" || v == "half")
        return OutputType::FP16;
    if (v == "bf16" || v == "bfloat16")
        return OutputType::BF16;
    if (v == "fp32" || v == "float32" || v == "float")
        return OutputType::FP32;
    throw Error(
        "unknown output_type: '" + s + "' (expected fp16 / bf16 / fp32)", __FILE__, __LINE__);
}

InputType input_type_from_string(const std::string& s) {
    const std::string v = normalize_value(s);
    if (v == "fp32" || v == "float32" || v == "float")
        return InputType::FP32;
    if (v == "fp16" || v == "float16" || v == "half")
        return InputType::FP16;
    throw Error("unknown input dtype: '" + s + "' (expected fp32 / fp16)", __FILE__, __LINE__);
}

void QuantConfig::apply_defaults() {
    if (!block_size_explicit || block_size <= 0) {
        block_size = is_mx() ? 32 : 16;
    }
    if (block_size <= 0)
        block_size = 32;

    // elem_format 与 format 必须自洽: e2m1 <-> mxfp4, e4m3 / e5m2 <-> mxfp8
    if (elem_format_explicit >= 0) {
        const char* given =
            elem_format_explicit == 2 ? "e2m1" : (elem_format_explicit == 1 ? "e5m2" : "e4m3");
        if (format == QuantFormat::MxFP4 && elem_format_explicit != 2) {
            throw Error(std::string("mxfp4 的元素格式固定为 e2m1, 不能写成 elem_format = ") + given,
                __FILE__, __LINE__);
        }
        if (format == QuantFormat::MxFP8 && elem_format_explicit == 2) {
            throw Error(
                "mxfp8 的元素格式只能是 e4m3 / e5m2; e2m1 对应 format = mxfp4", __FILE__, __LINE__);
        }
        if (format == QuantFormat::NVFP4) {
            throw Error("nvfp4 的元素格式固定为 e2m1, 不需要写 elem_format", __FILE__, __LINE__);
        }
    }

    // 4/6 自适应块缩放要靠 E4M3 块缩放因子的尾数去表达 1.5 倍扩张, 而 MX 家族的
    // 缩放因子是 E8M0(只有 8 位指数, 只能表示 2 的整数次幂), 因此只对 nvfp4 开放.
    if (scale_rule != ScaleRule::Static6 && format != QuantFormat::NVFP4) {
        throw Error(
            "scale_rule 只有 nvfp4 支持 (MX 的 E8M0 缩放因子表达不了 4/6 需要的"
            " 1.5 倍扩张)",
            __FILE__, __LINE__);
    }
}

void parse_quant_config_line(QuantConfig& cfg, const std::string& raw_line) {
    std::string line = raw_line;
    // 去掉注释
    size_t hash = line.find('#');
    if (hash != std::string::npos)
        line = line.substr(0, hash);
    line = trim(line);
    if (line.empty())
        return;

    size_t eq = line.find('=');
    if (eq == std::string::npos) {
        throw Error("config line is missing '=' : '" + raw_line + "'", __FILE__, __LINE__);
    }
    const std::string key = normalize_value(line.substr(0, eq));
    const std::string val = normalize_value(line.substr(eq + 1));

    if (key == "format") {
        if (val == "mxfp8" || val == "mx_fp8") {
            cfg.format = QuantFormat::MxFP8;
        } else if (val == "mxfp4" || val == "mx_fp4") {
            cfg.format = QuantFormat::MxFP4;
        } else if (val == "nvfp4" || val == "nv_fp4") {
            cfg.format = QuantFormat::NVFP4;
        } else {
            throw Error("unknown format: '" + val + "' (expected mxfp8 / mxfp4 / nvfp4)", __FILE__,
                __LINE__);
        }
    } else if (key == "scale_rule") {
        if (val == "static_6" || val == "static6") {
            cfg.scale_rule = ScaleRule::Static6;
        } else if (val == "static_4" || val == "static4") {
            cfg.scale_rule = ScaleRule::Static4;
        } else if (val == "abs_max" || val == "absmax") {
            cfg.scale_rule = ScaleRule::AbsMax;
        } else if (val == "mae") {
            cfg.scale_rule = ScaleRule::Mae;
        } else if (val == "mse" || val == "4/6" || val == "four_over_six") {
            cfg.scale_rule = ScaleRule::Mse;
        } else {
            throw Error("unknown scale_rule: '" + val +
                            "' (expected static_6 / static_4 / abs_max / mae / mse)",
                __FILE__, __LINE__);
        }
    } else if (key == "block_size") {
        cfg.block_size = std::stoi(val);
        cfg.block_size_explicit = true;
    } else if (key == "scale_mode") {
        if (val == "tensor") {
            cfg.scale_mode = ScaleMode::Tensor;
        } else if (val == "block" || val == "blockwise" || val == "block_wise") {
            cfg.scale_mode = ScaleMode::Block;
        } else {
            throw Error(
                "unknown scale_mode: '" + val + "' (expected tensor / block)", __FILE__, __LINE__);
        }
    } else if (key == "output_type") {
        cfg.output_type = output_type_from_string(val);
    } else if (key == "rounding") {
        if (val == "nearest" || val == "rne") {
            cfg.rounding = Rounding::Nearest;
        } else if (val == "stochastic" || val == "sr") {
            cfg.rounding = Rounding::Stochastic;
        } else {
            throw Error("unknown rounding: '" + val + "' (expected nearest / stochastic)", __FILE__,
                __LINE__);
        }
    } else if (key == "target_gpu") {
        cfg.target_gpu = trim(line.substr(eq + 1)).empty() ? "unknown" : trim(line.substr(eq + 1));
    } else if (key == "elem_format" || key == "element_format") {
        if (val == "e4m3") {
            cfg.mxfp8_e5m2 = false;
            cfg.elem_format_explicit = 0;
        } else if (val == "e5m2") {
            cfg.mxfp8_e5m2 = true;
            cfg.elem_format_explicit = 1;
        } else if (val == "e2m1") {
            cfg.elem_format_explicit = 2;
        } else {
            throw Error("unknown elem_format: '" + val + "' (expected e4m3 / e5m2 / e2m1)",
                __FILE__, __LINE__);
        }
    } else if (key == "seed") {
        cfg.seed = (uint64_t)std::stoull(val);
    } else {
        throw Error("unknown field in config file: '" + key + "'", __FILE__, __LINE__);
    }
}

QuantConfig parse_quant_config(const std::string& path) {
    std::ifstream ifs(path);
    if (!ifs)
        throw Error("cannot open config file: " + path, __FILE__, __LINE__);
    QuantConfig cfg;
    std::string line;
    while (std::getline(ifs, line))
        parse_quant_config_line(cfg, line);
    cfg.apply_defaults();
    return cfg;
}

std::string quant_config_to_string(const QuantConfig& cfg) {
    std::ostringstream os;
    os << "format = " << to_string(cfg.format) << "\n";
    if (cfg.format == QuantFormat::MxFP8) {
        os << "elem_format = " << (cfg.mxfp8_e5m2 ? "e5m2" : "e4m3") << "\n";
    } else if (cfg.format == QuantFormat::MxFP4) {
        os << "elem_format = e2m1\n";
    } else {
        os << "scale_rule = " << to_string(cfg.scale_rule) << "\n";
    }
    os << "block_size = " << cfg.block_size << "\n";
    os << "scale_mode = " << to_string(cfg.scale_mode) << "\n";
    os << "output_type = " << to_string(cfg.output_type) << "\n";
    os << "rounding = " << to_string(cfg.rounding) << "\n";
    os << "target_gpu = " << cfg.target_gpu << "\n";
    os << "seed = " << cfg.seed << "\n";
    return os.str();
}

}  // namespace lpq