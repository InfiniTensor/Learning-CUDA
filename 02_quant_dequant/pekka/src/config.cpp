#include "qd/config.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace qd {
namespace {

std::string trim(std::string value) {
  const auto non_space = [](unsigned char character) {
    return !std::isspace(character);
  };
  value.erase(value.begin(), std::find_if(value.begin(), value.end(), non_space));
  value.erase(std::find_if(value.rbegin(), value.rend(), non_space).base(),
              value.end());
  return value;
}

std::string unquote(const std::string &value) {
  if (value.size() >= 2 && value.front() == '"' && value.back() == '"') {
    return value.substr(1, value.size() - 2);
  }
  return value;
}

} // namespace

const char *to_string(DataType value) {
  switch (value) {
  case DataType::kFp16: return "fp16";
  case DataType::kFp32: return "fp32";
  case DataType::kBf16: return "bf16";
  }
  return "unknown";
}

const char *to_string(Format value) {
  return value == Format::kMxfp8 ? "mxfp8" : "nvfp4";
}

const char *to_string(ScaleMode value) {
  return value == ScaleMode::kBlock ? "block" : "tensor";
}

const char *to_string(Rounding value) {
  return value == Rounding::kNearest ? "nearest" : "stochastic";
}

Config read_config(const std::filesystem::path &path) {
  std::ifstream input(path);
  if (!input) throw std::runtime_error("cannot open config: " + path.string());

  std::unordered_map<std::string, std::string> fields;
  std::string line;
  std::size_t line_number = 0;
  while (std::getline(input, line)) {
    ++line_number;
    const auto comment = line.find('#');
    if (comment != std::string::npos) line.erase(comment);
    line = trim(line);
    if (line.empty()) continue;
    const auto equals = line.find('=');
    if (equals == std::string::npos) {
      throw std::runtime_error("invalid config line " + std::to_string(line_number));
    }
    fields[trim(line.substr(0, equals))] = unquote(trim(line.substr(equals + 1)));
  }

  const auto required = [&fields](const char *key) -> const std::string & {
    const auto found = fields.find(key);
    if (found == fields.end()) throw std::runtime_error(std::string("missing config field: ") + key);
    return found->second;
  };

  Config config;
  const std::string &format = required("format");
  if (format == "mxfp8") config.format = Format::kMxfp8;
  else if (format == "nvfp4") config.format = Format::kNvfp4;
  else throw std::runtime_error("format must be mxfp8 or nvfp4");

  config.block_size = static_cast<std::uint32_t>(std::stoul(required("block_size")));
  const std::string &mode = required("scale_mode");
  if (mode == "block") config.scale_mode = ScaleMode::kBlock;
  else if (mode == "tensor") config.scale_mode = ScaleMode::kTensor;
  else throw std::runtime_error("scale_mode must be block or tensor");

  const std::string &output = required("output_type");
  if (output == "fp16") config.output_type = DataType::kFp16;
  else if (output == "bf16") config.output_type = DataType::kBf16;
  else if (output == "fp32") config.output_type = DataType::kFp32;
  else throw std::runtime_error("output_type must be fp16, bf16, or fp32");

  const std::string &rounding = required("rounding");
  if (rounding == "nearest") config.rounding = Rounding::kNearest;
  else if (rounding == "stochastic") config.rounding = Rounding::kStochastic;
  else throw std::runtime_error("rounding must be nearest or stochastic");

  if (fields.count("target_gpu")) config.target_gpu = fields.at("target_gpu");
  if (fields.count("seed")) config.seed = std::stoull(fields.at("seed"));
  validate_config(config);
  return config;
}

void validate_config(const Config &config) {
  if (config.block_size == 0) throw std::runtime_error("block_size must be positive");
  if (config.format == Format::kMxfp8 && config.block_size != 32) {
    throw std::runtime_error("mxfp8 v1 requires block_size = 32");
  }
  if (config.format == Format::kNvfp4 && config.block_size != 16) {
    throw std::runtime_error("nvfp4 v1 requires block_size = 16");
  }
}

} // namespace qd
