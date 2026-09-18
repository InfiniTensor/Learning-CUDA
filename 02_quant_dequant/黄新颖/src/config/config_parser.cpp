#include "quant_dequant/config.hpp"

#include <charconv>
#include <cctype>
#include <fstream>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>
#include <type_traits>
#include <utility>

namespace quant_dequant {
namespace {

/**
 * @brief 删除字符串两端的 ASCII 空白字符，但不分配新内存。
 *
 * @param value 待裁剪的字符串视图。
 * @return 指向原字符串有效子区间的视图。
 */
[[nodiscard]] std::string_view trim_ascii_whitespace(std::string_view value) noexcept {
    const auto is_whitespace = [](const unsigned char character) noexcept {
        return std::isspace(character) != 0;
    };

    while (!value.empty() && is_whitespace(static_cast<unsigned char>(value.front()))) {
        value.remove_prefix(1U);
    }

    while (!value.empty() && is_whitespace(static_cast<unsigned char>(value.back()))) {
        value.remove_suffix(1U);
    }

    return value;
}

/**
 * @brief 找到引号外的行尾注释起点。
 *
 * 当前配置格式不支持字符串转义；因此双引号只能成对出现。未闭合的引号
 * 会在后续值解析阶段报告为带行号的 ConfigError。
 *
 * @param line 单行配置文本。
 * @return 删除引号外 `#` 注释后的字符串视图。
 */
[[nodiscard]] std::string_view remove_trailing_comment(std::string_view line) noexcept {
    bool inside_quotes = false;

    for (std::size_t index = 0U; index < line.size(); ++index) {
        const char character = line[index];

        if (character == '"') {
            inside_quotes = !inside_quotes;
            continue;
        }

        if (character == '#' && !inside_quotes) {
            return line.substr(0U, index);
        }
    }

    return line;
}

/**
 * @brief 找到引号外唯一有效的 key/value 分隔符。
 *
 * @param line 已移除行尾注释的配置行。
 * @return 分隔符位置；不存在时返回空值。
 */
[[nodiscard]] std::optional<std::size_t> find_assignment_separator(
    const std::string_view line) noexcept {
    bool inside_quotes = false;

    for (std::size_t index = 0U; index < line.size(); ++index) {
        const char character = line[index];

        if (character == '"') {
            inside_quotes = !inside_quotes;
            continue;
        }

        if (character == '=' && !inside_quotes) {
            return index;
        }
    }

    return std::nullopt;
}

/**
 * @brief 为 ConfigError 生成统一的异常消息。
 *
 * @param file_path 配置文件路径。
 * @param line_number 1-based 行号；0 表示无特定行。
 * @param detail 具体错误说明。
 * @return 可传给 std::runtime_error 的完整错误文本。
 */
[[nodiscard]] std::string make_error_message(const std::filesystem::path& file_path,
                                              const std::size_t line_number,
                                              const std::string_view detail) {
    std::string message{"配置文件 \""};
    message += file_path.string();
    message += "\"";

    if (line_number != 0U) {
        message += " 第 ";
        message += std::to_string(line_number);
        message += " 行";
    }

    message += "：";
    message += detail;
    return message;
}

/**
 * @brief 抛出带文件路径和行号的配置错误。
 *
 * @param file_path 配置文件路径。
 * @param line_number 1-based 行号；0 表示无特定行。
 * @param detail 具体错误说明。
 */
[[noreturn]] void throw_config_error(const std::filesystem::path& file_path,
                                     const std::size_t line_number,
                                     std::string detail) {
    throw ConfigError(file_path, line_number, std::move(detail));
}

/**
 * @brief 解析可带双引号的标量字符串。
 *
 * 双引号用于保留 `#`、空格等普通字符；第一版不支持转义序列或嵌套引号。
 *
 * @param value 已裁剪空白的原始 value。
 * @param file_path 配置文件路径，用于报错。
 * @param line_number 发生解析的 1-based 行号。
 * @return 去除外层双引号后的非拥有字符串视图。
 * @throws ConfigError value 为空、引号不配对或包含未支持的内嵌引号时抛出。
 */
[[nodiscard]] std::string_view parse_scalar_value(
    const std::string_view value, const std::filesystem::path& file_path,
    const std::size_t line_number) {
    if (value.empty()) {
        throw_config_error(file_path, line_number, "key 缺少 value。");
    }

    const bool starts_with_quote = value.front() == '"';
    const bool ends_with_quote = value.back() == '"';

    if (starts_with_quote != ends_with_quote) {
        throw_config_error(file_path, line_number, "字符串 value 的双引号不配对。");
    }

    if (!starts_with_quote) {
        if (value.find('"') != std::string_view::npos) {
            throw_config_error(file_path, line_number,
                               "未加引号的 value 不能包含双引号。");
        }
        return value;
    }

    if (value.size() < 2U) {
        throw_config_error(file_path, line_number, "字符串 value 不能为空的单个双引号。");
    }

    const std::string_view unquoted_value = value.substr(1U, value.size() - 2U);
    if (unquoted_value.find('"') != std::string_view::npos) {
        throw_config_error(file_path, line_number,
                           "第一版配置格式不支持字符串内的双引号或转义序列。");
    }

    return unquoted_value;
}

/**
 * @brief 将十进制无符号整数字符串转换为指定整数类型。
 *
 * @tparam IntegerType 目标无符号整数类型。
 * @param value 已去除引号和两端空白的十进制字符串。
 * @param field_name 当前字段名，用于生成诊断。
 * @param file_path 配置文件路径。
 * @param line_number 发生解析的 1-based 行号。
 * @return 成功解析的整数。
 * @throws ConfigError value 非十进制整数、超出范围或为空时抛出。
 */
template <typename IntegerType>
[[nodiscard]] IntegerType parse_unsigned_integer(
    const std::string_view value, const std::string_view field_name,
    const std::filesystem::path& file_path, const std::size_t line_number) {
    static_assert(std::is_unsigned_v<IntegerType>);

    IntegerType result{};
    const auto [end_ptr, error_code] =
        std::from_chars(value.data(), value.data() + value.size(), result, 10);

    if (error_code != std::errc{} || end_ptr != value.data() + value.size()) {
        throw_config_error(file_path, line_number,
                           "字段 \"" + std::string{field_name} +
                               "\" 必须是范围内的十进制无符号整数。");
    }

    return result;
}

/**
 * @brief 将配置文本映射到 QuantFormat 枚举。
 *
 * @param value 已去除引号的文本。
 * @param file_path 配置文件路径。
 * @param line_number 发生解析的 1-based 行号。
 * @return 合法的量化格式。
 */
[[nodiscard]] QuantFormat parse_quant_format(const std::string_view value,
                                              const std::filesystem::path& file_path,
                                              const std::size_t line_number) {
    if (value == "mxfp8") {
        return QuantFormat::kMxfp8;
    }

    if (value == "nvfp4") {
        return QuantFormat::kNvfp4;
    }

    throw_config_error(file_path, line_number,
                       "字段 \"format\" 只能是 \"mxfp8\" 或 \"nvfp4\"。");
}

/**
 * @brief 将配置文本映射到 ScaleMode 枚举。
 *
 * @param value 已去除引号的文本。
 * @param file_path 配置文件路径。
 * @param line_number 发生解析的 1-based 行号。
 * @return 合法的 scale mode。
 */
[[nodiscard]] ScaleMode parse_scale_mode(const std::string_view value,
                                         const std::filesystem::path& file_path,
                                         const std::size_t line_number) {
    if (value == "tensor") {
        return ScaleMode::kTensor;
    }

    if (value == "block") {
        return ScaleMode::kBlock;
    }

    throw_config_error(file_path, line_number,
                       "字段 \"scale_mode\" 只能是 \"tensor\" 或 \"block\"。");
}

/**
 * @brief 将配置文本映射到反量化输出 DType。
 *
 * @param value 已去除引号的文本。
 * @param file_path 配置文件路径。
 * @param line_number 发生解析的 1-based 行号。
 * @return 合法的反量化输出类型。
 */
[[nodiscard]] DType parse_output_type(const std::string_view value,
                                      const std::filesystem::path& file_path,
                                      const std::size_t line_number) {
    if (value == "fp16") {
        return DType::kFloat16;
    }

    if (value == "bf16") {
        return DType::kBFloat16;
    }

    if (value == "fp32") {
        return DType::kFloat32;
    }

    throw_config_error(file_path, line_number,
                       "字段 \"output_type\" 只能是 \"fp16\"、\"bf16\" 或 \"fp32\"。");
}

/**
 * @brief 将配置文本映射到 RoundingMode 枚举。
 *
 * @param value 已去除引号的文本。
 * @param file_path 配置文件路径。
 * @param line_number 发生解析的 1-based 行号。
 * @return 合法的舍入模式。
 */
[[nodiscard]] RoundingMode parse_rounding_mode(
    const std::string_view value, const std::filesystem::path& file_path,
    const std::size_t line_number) {
    if (value == "nearest") {
        return RoundingMode::kNearest;
    }

    if (value == "stochastic") {
        return RoundingMode::kStochastic;
    }

    throw_config_error(file_path, line_number,
                       "字段 \"rounding\" 只能是 \"nearest\" 或 \"stochastic\"。");
}

/**
 * @brief 记录配置字段是否出现及其 1-based 行号。
 *
 * 同一个平面 `key = value` 文件可被投影为量化、反量化或完整 app 配置；
 * 因此“是否必填”不属于这个记录结构，而由调用方选择的配置入口决定。
 */
struct ConfigFieldLines {
    std::size_t format{0U};
    std::size_t block_size{0U};
    std::size_t scale_mode{0U};
    std::size_t output_type{0U};
    std::size_t rounding{0U};
    std::size_t target_gpu{0U};
    std::size_t stochastic_seed{0U};
};

/**
 * @brief 保存一次语法解析的全部已知字段及其出现位置。
 *
 * 该类型只存在于 parser 内部。它允许文本文件只写量化字段或只写反量化字段，
 * 再由不同的 public loader 施加各自的必填字段和语义约束。
 */
struct ParsedConfigDocument {
    QuantizationConfig quantization{};
    DequantizationConfig dequantization{};
    ReportConfig report{};
    ConfigFieldLines field_lines{};
};

/**
 * @brief 在字段首次出现时记录行号，重复时抛出异常。
 *
 * @param line_slot 对应字段的行号槽位。
 * @param key 当前 key。
 * @param line_number 当前 1-based 行号。
 * @param file_path 配置文件路径。
 */
void mark_field_seen(std::size_t& line_slot, const std::string_view key,
                     const std::size_t line_number,
                     const std::filesystem::path& file_path) {
    if (line_slot != 0U) {
        throw_config_error(file_path, line_number,
                           "字段 \"" + std::string{key} + "\" 重复；首次出现于第 " +
                               std::to_string(line_slot) + " 行。");
    }

    line_slot = line_number;
}

/**
 * @brief 确认一个按当前 loader 必填的字段确实出现在文件中。
 *
 * @param line_number 对应字段的出现行号；0 表示字段未出现。
 * @param field_name 字段名称。
 * @param file_path 配置文件路径。
 */
void require_field(const std::size_t line_number,
                   const std::string_view field_name,
                   const std::filesystem::path& file_path) {
    if (line_number == 0U) {
        throw_config_error(file_path, 0U,
                           "缺少必填字段 \"" + std::string{field_name} + "\"。");
    }
}

/**
 * @brief 验证单方向量化配置的必填字段及数值语义。
 *
 * @param document 已完成语法解析的配置文档。
 * @param file_path 配置文件路径。
 */
void validate_quantization_config(const ParsedConfigDocument& document,
                                  const std::filesystem::path& file_path) {
    const ConfigFieldLines& lines = document.field_lines;
    const QuantizationConfig& config = document.quantization;

    require_field(lines.format, "format", file_path);
    require_field(lines.block_size, "block_size", file_path);
    require_field(lines.scale_mode, "scale_mode", file_path);
    require_field(lines.rounding, "rounding", file_path);

    if (!is_valid_block_size(config.format, config.block_size)) {
        throw_config_error(file_path, lines.block_size,
                           "字段 \"block_size\" 与 \"format\" 不匹配。");
    }

    if (!is_valid_scale_mode(config.format, config.scale_mode)) {
        throw_config_error(
            file_path, lines.scale_mode,
            "NVFP4 仅支持 scale_mode = \"block\"：每 16 个元素保存一个 E4M3 "
            "local scale，并与 FP32 global_scale 共同构成分层缩放。");
    }

    if (config.rounding == RoundingMode::kNearest &&
        config.stochastic_seed != 0U) {
        throw_config_error(file_path, lines.stochastic_seed,
                           "nearest 舍入不允许非零 \"stochastic_seed\"。");
    }

    if (!config.isValid()) {
        throw_config_error(file_path, 0U, "量化配置未通过最终语义校验。");
    }
}

/**
 * @brief 验证单方向反量化配置的必填字段及数值语义。
 *
 * @param document 已完成语法解析的配置文档。
 * @param file_path 配置文件路径。
 */
void validate_dequantization_config(const ParsedConfigDocument& document,
                                    const std::filesystem::path& file_path) {
    require_field(document.field_lines.output_type, "output_type", file_path);

    if (!document.dequantization.isValid()) {
        throw_config_error(file_path, document.field_lines.output_type,
                           "反量化输出类型不受支持。");
    }
}

/**
 * @brief 验证完整 app 配置额外需要的报告字段。
 *
 * @param document 已完成语法解析的配置文档。
 * @param file_path 配置文件路径。
 */
void validate_app_config(const ParsedConfigDocument& document,
                         const std::filesystem::path& file_path) {
    validate_quantization_config(document, file_path);
    validate_dequantization_config(document, file_path);
    require_field(document.field_lines.target_gpu, "target_gpu", file_path);

    const AppConfig app_config{
        .quantization = document.quantization,
        .dequantization = document.dequantization,
        .report = document.report,
    };
    if (!app_config.isValid()) {
        throw_config_error(file_path, 0U, "完整 app 配置未通过最终语义校验。");
    }
}

/**
 * @brief 解析平面 `key = value` 文件中的全部已知字段，不预设运行方向。
 *
 * 语法、标量类型、重复 key 与未知 key 始终在这里验证；量化、反量化和 app
 * 三类 loader 各自的必填字段判断则延后到对应的语义校验函数。
 *
 * @param config_path 待解析的配置文件路径。
 * @return 已完成语法解析的内部文档。
 * @throws ConfigError 文件无法打开、I/O 失败或文本语法非法时抛出。
 */
[[nodiscard]] ParsedConfigDocument parse_config_document(
    const std::filesystem::path& config_path) {
    std::ifstream input_stream{config_path};
    if (!input_stream.is_open()) {
        throw_config_error(config_path, 0U, "无法打开文件。");
    }

    ParsedConfigDocument document{};
    std::string line{};
    std::size_t line_number = 0U;

    while (std::getline(input_stream, line)) {
        ++line_number;

        if (line_number == 1U && line.starts_with("\xEF\xBB\xBF")) {
            line.erase(0U, 3U);
        }

        const std::string_view uncommented_line = remove_trailing_comment(line);
        const std::string_view content = trim_ascii_whitespace(uncommented_line);
        if (content.empty()) {
            continue;
        }

        const auto separator_position = find_assignment_separator(content);
        if (!separator_position.has_value()) {
            throw_config_error(config_path, line_number,
                               "配置行必须使用 \"key = value\" 形式。");
        }

        const std::string_view key =
            trim_ascii_whitespace(content.substr(0U, *separator_position));
        const std::string_view raw_value =
            trim_ascii_whitespace(content.substr(*separator_position + 1U));
        if (key.empty()) {
            throw_config_error(config_path, line_number, "配置行缺少 key。");
        }

        if (key.find_first_of(" \t\r\n") != std::string_view::npos) {
            throw_config_error(config_path, line_number, "key 不能包含空白字符。");
        }

        const std::string_view value =
            parse_scalar_value(raw_value, config_path, line_number);
        ConfigFieldLines& field_lines = document.field_lines;

        if (key == "format") {
            mark_field_seen(field_lines.format, key, line_number, config_path);
            document.quantization.format =
                parse_quant_format(value, config_path, line_number);
            continue;
        }

        if (key == "block_size") {
            mark_field_seen(field_lines.block_size, key, line_number, config_path);
            document.quantization.block_size = parse_unsigned_integer<std::uint32_t>(
                value, key, config_path, line_number);
            continue;
        }

        if (key == "scale_mode") {
            mark_field_seen(field_lines.scale_mode, key, line_number, config_path);
            document.quantization.scale_mode =
                parse_scale_mode(value, config_path, line_number);
            continue;
        }

        if (key == "rounding") {
            mark_field_seen(field_lines.rounding, key, line_number, config_path);
            document.quantization.rounding =
                parse_rounding_mode(value, config_path, line_number);
            continue;
        }

        if (key == "stochastic_seed") {
            mark_field_seen(field_lines.stochastic_seed, key, line_number, config_path);
            document.quantization.stochastic_seed =
                parse_unsigned_integer<std::uint64_t>(value, key, config_path, line_number);
            continue;
        }

        if (key == "output_type") {
            mark_field_seen(field_lines.output_type, key, line_number, config_path);
            document.dequantization.output_type =
                parse_output_type(value, config_path, line_number);
            continue;
        }

        if (key == "target_gpu") {
            mark_field_seen(field_lines.target_gpu, key, line_number, config_path);
            if (value.empty()) {
                throw_config_error(config_path, line_number,
                                   "字段 \"target_gpu\" 不能为空字符串。");
            }
            document.report.target_gpu = value;
            continue;
        }

        throw_config_error(config_path, line_number,
                           "未知配置字段 \"" + std::string{key} + "\"。");
    }

    if (!input_stream.eof()) {
        throw_config_error(config_path, line_number, "读取文件时发生 I/O 错误。");
    }

    return document;
}

}  // namespace

ConfigError::ConfigError(std::filesystem::path file_path, const std::size_t line_number,
                         std::string detail)
    : std::runtime_error(make_error_message(file_path, line_number, detail)),
      mFilePath(std::move(file_path)),
      mLineNumber(line_number) {}

const std::filesystem::path& ConfigError::filePath() const noexcept {
    return mFilePath;
}

std::size_t ConfigError::lineNumber() const noexcept {
    return mLineNumber;
}

QuantizationConfig load_quantization_config(
    const std::filesystem::path& config_path) {
    const ParsedConfigDocument document = parse_config_document(config_path);
    validate_quantization_config(document, config_path);
    return document.quantization;
}

DequantizationConfig load_dequantization_config(
    const std::filesystem::path& config_path) {
    const ParsedConfigDocument document = parse_config_document(config_path);
    validate_dequantization_config(document, config_path);
    return document.dequantization;
}

AppConfig load_app_config(const std::filesystem::path& config_path) {
    const ParsedConfigDocument document = parse_config_document(config_path);
    validate_app_config(document, config_path);

    return AppConfig{
        .quantization = document.quantization,
        .dequantization = document.dequantization,
        .report = document.report,
    };
}

}  // namespace quant_dequant
