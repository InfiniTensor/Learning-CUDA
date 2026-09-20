#include "quant_config.h"

#include <array>
#include <cctype>
#include <fstream>
#include <set>
#include <stdexcept>

namespace luxorion {
namespace {

std::string trim(std::string value) {
  while (!value.empty() &&
         std::isspace(static_cast<unsigned char>(value.front())))
    value.erase(value.begin());
  while (!value.empty() &&
         std::isspace(static_cast<unsigned char>(value.back())))
    value.pop_back();
  return value;
}

std::string quoted(const std::string &value, const std::string &key) {
  if (value.size() < 2 || value.front() != '"' || value.back() != '"') {
    throw std::invalid_argument(key + " must be a quoted string");
  }
  return value.substr(1, value.size() - 2);
}

} // namespace

QuantConfig read_quant_config(const std::string &path) {
  std::ifstream file(path);
  if (!file)
    throw std::runtime_error("cannot open quantization config: " + path);
  QuantConfig config;
  std::set<std::string> seen;
  std::string line;
  std::size_t line_number = 0;
  while (std::getline(file, line)) {
    ++line_number;
    const auto comment = line.find('#');
    if (comment != std::string::npos)
      line.erase(comment);
    line = trim(line);
    if (line.empty())
      continue;
    const auto equal = line.find('=');
    if (equal == std::string::npos ||
        line.find('=', equal + 1) != std::string::npos) {
      throw std::invalid_argument("invalid config assignment at line " +
                                  std::to_string(line_number));
    }
    const std::string key = trim(line.substr(0, equal));
    const std::string value = trim(line.substr(equal + 1));
    if (!seen.insert(key).second)
      throw std::invalid_argument("duplicate config field: " + key);
    if (key == "format") {
      const auto text = quoted(value, key);
      if (text == "mxfp8")
        config.format = QuantFormat::Mxfp8;
      else if (text == "nvfp4")
        config.format = QuantFormat::Nvfp4;
      else
        throw std::invalid_argument("format must be mxfp8 or nvfp4");
    } else if (key == "block_size") {
      std::size_t parsed = 0;
      const unsigned long number = std::stoul(value, &parsed);
      if (parsed != value.size() || number == 0)
        throw std::invalid_argument("block_size must be positive");
      config.block_size = static_cast<std::size_t>(number);
    } else if (key == "scale_mode") {
      const auto text = quoted(value, key);
      if (text == "tensor")
        config.scale_mode = ScaleMode::Tensor;
      else if (text == "block")
        config.scale_mode = ScaleMode::Block;
      else
        throw std::invalid_argument("scale_mode must be tensor or block");
    } else if (key == "output_type") {
      const auto text = quoted(value, key);
      if (text == "fp16")
        config.output_type = OutputType::FP16;
      else if (text == "bf16")
        config.output_type = OutputType::BF16;
      else if (text == "fp32")
        config.output_type = OutputType::FP32;
      else
        throw std::invalid_argument("output_type must be fp16, bf16, or fp32");
    } else if (key == "rounding") {
      const auto text = quoted(value, key);
      if (text == "nearest")
        config.rounding = RoundingMode::Nearest;
      else if (text == "stochastic")
        config.rounding = RoundingMode::Stochastic;
      else
        throw std::invalid_argument("rounding must be nearest or stochastic");
    } else if (key == "target_gpu") {
      config.target_gpu = quoted(value, key);
      if (config.target_gpu.empty())
        throw std::invalid_argument("target_gpu must not be empty");
    } else if (key == "seed") {
      std::size_t parsed = 0;
      const unsigned long number = std::stoul(value, &parsed);
      if (parsed != value.size() || number > 0xFFFFFFFFUL) {
        throw std::invalid_argument("seed must be a uint32 value");
      }
      config.seed = static_cast<std::uint32_t>(number);
    } else {
      throw std::invalid_argument("unknown config field: " + key);
    }
  }
  const std::array<const char *, 6> required = {"format",     "block_size",
                                                "scale_mode", "output_type",
                                                "rounding",   "target_gpu"};
  for (const char *key : required) {
    if (seen.count(key) == 0)
      throw std::invalid_argument(std::string("missing config field: ") + key);
  }
  const std::size_t expected = config.format == QuantFormat::Mxfp8 ? 32U : 16U;
  if (config.block_size != expected) {
    throw std::invalid_argument(
        "block_size does not match the selected format");
  }
  return config;
}

} // namespace luxorion
