// 解析 INI 配置和 CLI 参数，尽早报告可定位的输入错误。
#include "pricer/config.hpp"

#include <cerrno>
#include <charconv>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string_view>
#include <unordered_map>
#include <unordered_set>

namespace pricer {
namespace {

struct Entry {
    std::string value;
    std::size_t line;
};

using Entries = std::unordered_map<std::string, Entry>;

std::string trim(std::string_view text) {
    const auto first = text.find_first_not_of(" \t\r\n");
    if (first == std::string_view::npos) {
        return {};
    }
    const auto last = text.find_last_not_of(" \t\r\n");
    return std::string(text.substr(first, last - first + 1));
}

const char *code_name(ErrorCode code) noexcept {
    switch (code) {
    case ErrorCode::ConfigInvalid:
        return "CONFIG_INVALID";
    case ErrorCode::FileRead:
        return "FILE_READ";
    case ErrorCode::UnsupportedFeature:
        return "UNSUPPORTED_FEATURE";
    }
    return "UNKNOWN";
}

[[noreturn]] void fail(ErrorCode code, const std::filesystem::path &file,
                       std::optional<std::size_t> line,
                       std::optional<std::string> field,
                       const std::string &reason) {
    throw ConfigError(code, file, std::move(line), std::move(field), reason);
}

std::string strip_comment(const std::string &line,
                          const std::filesystem::path &file,
                          std::size_t line_number) {
    bool quoted = false;
    for (std::size_t index = 0; index < line.size(); ++index) {
        if (line[index] == '"') {
            quoted = !quoted;
        } else if (line[index] == '#' && !quoted) {
            return line.substr(0, index);
        }
    }
    if (quoted) {
        fail(ErrorCode::ConfigInvalid, file, line_number, std::nullopt,
             "unmatched double quote");
    }
    return line;
}

Entries parse_ini(const std::filesystem::path &path,
                  const std::unordered_set<std::string> &allowed_keys) {
    std::ifstream stream(path);
    if (!stream) {
        fail(ErrorCode::FileRead, path, std::nullopt, std::nullopt,
             "could not open file");
    }

    Entries entries;
    std::string raw_line;
    std::size_t line_number = 0;
    while (std::getline(stream, raw_line)) {
        ++line_number;
        const auto line = trim(strip_comment(raw_line, path, line_number));
        if (line.empty()) {
            continue;
        }

        const auto equals = line.find('=');
        if (equals == std::string::npos) {
            fail(ErrorCode::ConfigInvalid, path, line_number, std::nullopt,
                 "expected key = value");
        }
        const auto key = trim(std::string_view(line).substr(0, equals));
        auto value = trim(std::string_view(line).substr(equals + 1));
        if (key.empty()) {
            fail(ErrorCode::ConfigInvalid, path, line_number, std::nullopt,
                 "empty key");
        }
        if (allowed_keys.find(key) == allowed_keys.end()) {
            fail(ErrorCode::ConfigInvalid, path, line_number, key,
                 "unknown key");
        }
        if (entries.find(key) != entries.end()) {
            fail(ErrorCode::ConfigInvalid, path, line_number, key,
                 "duplicate key");
        }
        if (!value.empty() && value.front() == '"' && value.back() == '"' &&
            value.size() >= 2) {
            value = value.substr(1, value.size() - 2);
        } else if (value.find('"') != std::string::npos) {
            fail(ErrorCode::ConfigInvalid, path, line_number, key,
                 "unmatched double quote");
        }
        entries.emplace(key, Entry{value, line_number});
    }
    if (stream.bad()) {
        fail(ErrorCode::FileRead, path, std::nullopt, std::nullopt,
             "could not read file");
    }
    return entries;
}

const Entry &required(const Entries &entries, const char *key,
                      const std::filesystem::path &path) {
    const auto entry = entries.find(key);
    if (entry == entries.end()) {
        fail(ErrorCode::ConfigInvalid, path, std::nullopt, key,
             "missing required key");
    }
    return entry->second;
}

std::uint64_t parse_unsigned(const Entry &entry, const char *field,
                             const std::filesystem::path &path) {
    std::uint64_t value = 0;
    const auto begin = entry.value.data();
    const auto end = begin + entry.value.size();
    const auto result = std::from_chars(begin, end, value);
    if (entry.value.empty() || entry.value.front() == '-' ||
        result.ec != std::errc() || result.ptr != end) {
        fail(ErrorCode::ConfigInvalid, path, entry.line, field,
             "expected integer, got '" + entry.value + "'");
    }
    return value;
}

std::uint32_t parse_uint32(const Entry &entry, const char *field,
                           const std::filesystem::path &path) {
    const auto value = parse_unsigned(entry, field, path);
    if (value > UINT32_MAX) {
        fail(ErrorCode::ConfigInvalid, path, entry.line, field,
             "expected integer <= 4294967295, got '" + entry.value + "'");
    }
    return static_cast<std::uint32_t>(value);
}

double parse_double(const Entry &entry, const char *field,
                    const std::filesystem::path &path) {
    errno = 0;
    char *end = nullptr;
    const auto value = std::strtod(entry.value.c_str(), &end);
    if (entry.value.empty() ||
        end != entry.value.c_str() + entry.value.size() || errno == ERANGE ||
        !std::isfinite(value)) {
        fail(ErrorCode::ConfigInvalid, path, entry.line, field,
             "expected finite number, got '" + entry.value + "'");
    }
    return value;
}

OptionType parse_option_type(const Entry &entry,
                             const std::filesystem::path &path) {
    if (entry.value == "european_call") {
        return OptionType::EuropeanCall;
    }
    if (entry.value == "asian_call") {
        return OptionType::AsianArithmeticCall;
    }
    if (entry.value == "european_put" || entry.value == "barrier_call") {
        fail(ErrorCode::UnsupportedFeature, path, entry.line, "option_type",
             "option type '" + entry.value + "' is not supported in P0");
    }
    fail(ErrorCode::ConfigInvalid, path, entry.line, "option_type",
         "invalid option type '" + entry.value + "'");
}

Precision parse_precision(const Entry &entry,
                          const std::filesystem::path &path) {
    if (entry.value == "fp32") {
        return Precision::Fp32;
    }
    if (entry.value == "fp64") {
        return Precision::Fp64;
    }
    fail(ErrorCode::ConfigInvalid, path, entry.line, "precision",
         "invalid precision '" + entry.value + "'");
}

VarianceReduction parse_variance_reduction(const Entry &entry,
                                           const std::filesystem::path &path) {
    if (entry.value == "none") {
        return VarianceReduction::None;
    }
    if (entry.value == "antithetic" || entry.value == "control_variate") {
        fail(ErrorCode::UnsupportedFeature, path, entry.line,
             "variance_reduction",
             "variance reduction '" + entry.value + "' is not supported in P0");
    }
    fail(ErrorCode::ConfigInvalid, path, entry.line, "variance_reduction",
         "invalid variance reduction '" + entry.value + "'");
}

RngType parse_rng(const Entry &entry, const std::filesystem::path &path) {
    if (entry.value == "curand_philox") {
        return RngType::CurandPhilox;
    }
    fail(ErrorCode::ConfigInvalid, path, entry.line, "rng",
         "invalid rng '" + entry.value + "'");
}

void validate_positive(double value, const Entry &entry, const char *field,
                       const std::filesystem::path &path) {
    if (value <= 0.0) {
        fail(ErrorCode::ConfigInvalid, path, entry.line, field,
             "expected number > 0, got '" + entry.value + "'");
    }
}

} // namespace

ConfigError::ConfigError(ErrorCode code, std::filesystem::path source_file,
                         std::optional<std::size_t> line,
                         std::optional<std::string> field, std::string reason)
    : std::runtime_error(reason), code_(code),
      source_file_(std::move(source_file)), line_(std::move(line)),
      field_(std::move(field)), reason_(std::move(reason)) {}

ErrorCode ConfigError::code() const noexcept { return code_; }

const std::filesystem::path &ConfigError::source_file() const noexcept {
    return source_file_;
}

const std::optional<std::size_t> &ConfigError::line() const noexcept {
    return line_;
}

const std::optional<std::string> &ConfigError::field() const noexcept {
    return field_;
}

const std::string &ConfigError::reason() const noexcept { return reason_; }

std::string ConfigError::render() const {
    std::ostringstream stream;
    stream << "ERROR [" << code_name(code_) << "] "
           << source_file_.filename().string();
    if (line_) {
        stream << ':' << *line_;
    }
    if (field_) {
        stream << " field '" << *field_ << "'";
    }
    stream << ": " << reason_;
    return stream.str();
}

int exit_code_for(ErrorCode code) noexcept {
    switch (code) {
    case ErrorCode::ConfigInvalid:
        return 2;
    case ErrorCode::FileRead:
        return 3;
    case ErrorCode::UnsupportedFeature:
        return 6;
    }
    return 1;
}

RunConfig load_run_config(const std::filesystem::path &option_file,
                          const std::filesystem::path &simulation_file,
                          std::filesystem::path output_dir, bool run_cpu,
                          bool run_gpu) {
    const Entries option_entries = parse_ini(
        option_file, {"option_type", "spot", "strike", "risk_free_rate",
                      "volatility", "maturity", "barrier"});
    const Entries simulation_entries =
        parse_ini(simulation_file, {"num_paths", "num_steps", "seed", "rng",
                                    "variance_reduction", "precision",
                                    "block_size", "batch_size"});

    const auto &option_type_entry =
        required(option_entries, "option_type", option_file);
    const auto &spot_entry = required(option_entries, "spot", option_file);
    const auto &strike_entry = required(option_entries, "strike", option_file);
    const auto &rate_entry =
        required(option_entries, "risk_free_rate", option_file);
    const auto &volatility_entry =
        required(option_entries, "volatility", option_file);
    const auto &maturity_entry =
        required(option_entries, "maturity", option_file);

    OptionParams option{
        parse_option_type(option_type_entry, option_file),
        parse_double(spot_entry, "spot", option_file),
        parse_double(strike_entry, "strike", option_file),
        parse_double(rate_entry, "risk_free_rate", option_file),
        parse_double(volatility_entry, "volatility", option_file),
        parse_double(maturity_entry, "maturity", option_file),
        std::nullopt};
    if (const auto barrier = option_entries.find("barrier");
        barrier != option_entries.end()) {
        fail(ErrorCode::ConfigInvalid, option_file, barrier->second.line,
             "barrier", "barrier is not supported for P0 option types");
    }

    validate_positive(option.spot, spot_entry, "spot", option_file);
    validate_positive(option.strike, strike_entry, "strike", option_file);
    validate_positive(option.maturity, maturity_entry, "maturity", option_file);
    if (option.volatility < 0.0) {
        fail(ErrorCode::ConfigInvalid, option_file, volatility_entry.line,
             "volatility",
             "expected number >= 0, got '" + volatility_entry.value + "'");
    }

    const auto &num_paths_entry =
        required(simulation_entries, "num_paths", simulation_file);
    const auto &num_steps_entry =
        required(simulation_entries, "num_steps", simulation_file);
    const auto &seed_entry =
        required(simulation_entries, "seed", simulation_file);
    const auto &rng_entry =
        required(simulation_entries, "rng", simulation_file);
    const auto &variance_entry =
        required(simulation_entries, "variance_reduction", simulation_file);
    const auto &precision_entry =
        required(simulation_entries, "precision", simulation_file);
    const auto &block_size_entry =
        required(simulation_entries, "block_size", simulation_file);
    const auto &batch_size_entry =
        required(simulation_entries, "batch_size", simulation_file);

    SimulationParams simulation{
        parse_unsigned(num_paths_entry, "num_paths", simulation_file),
        parse_uint32(num_steps_entry, "num_steps", simulation_file),
        parse_unsigned(seed_entry, "seed", simulation_file),
        parse_rng(rng_entry, simulation_file),
        parse_precision(precision_entry, simulation_file),
        parse_variance_reduction(variance_entry, simulation_file),
        parse_uint32(block_size_entry, "block_size", simulation_file),
        parse_unsigned(batch_size_entry, "batch_size", simulation_file)};
    if (simulation.num_paths == 0) {
        fail(ErrorCode::ConfigInvalid, simulation_file, num_paths_entry.line,
             "num_paths",
             "expected integer >= 1, got '" + num_paths_entry.value + "'");
    }
    if (option.type == OptionType::AsianArithmeticCall &&
        simulation.num_steps == 0) {
        fail(ErrorCode::ConfigInvalid, simulation_file, num_steps_entry.line,
             "num_steps",
             "expected integer >= 1, got '" + num_steps_entry.value + "'");
    }
    if (simulation.block_size == 0) {
        fail(ErrorCode::ConfigInvalid, simulation_file, block_size_entry.line,
             "block_size",
             "expected integer >= 1, got '" + block_size_entry.value + "'");
    }

    return RunConfig{option, simulation, std::move(output_dir), run_cpu,
                     run_gpu};
}

} // namespace pricer
