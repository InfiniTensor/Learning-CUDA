#include "nbody/config.hpp"

#include <cmath>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace nbody {
namespace {

std::string trim(const std::string& text) {
    const auto begin = text.find_first_not_of(" \t\r\n");
    if (begin == std::string::npos) return {};
    const auto end = text.find_last_not_of(" \t\r\n");
    return text.substr(begin, end - begin + 1);
}

double parse_double(const std::string& value, const std::string& key) {
    std::size_t used = 0;
    double parsed = 0.0;
    try {
        parsed = std::stod(value, &used);
    } catch (const std::exception&) {
        throw std::runtime_error("配置项 " + key + " 不是有效浮点数: " + value);
    }
    if (used != value.size() || !std::isfinite(parsed)) {
        throw std::runtime_error("配置项 " + key + " 不是有限浮点数: " + value);
    }
    return parsed;
}

int parse_int(const std::string& value, const std::string& key) {
    std::size_t used = 0;
    long long parsed = 0;
    try {
        parsed = std::stoll(value, &used);
    } catch (const std::exception&) {
        throw std::runtime_error("配置项 " + key + " 不是有效整数: " + value);
    }
    if (used != value.size() || parsed < std::numeric_limits<int>::min() ||
        parsed > std::numeric_limits<int>::max()) {
        throw std::runtime_error("配置项 " + key + " 超出 int 范围: " + value);
    }
    return static_cast<int>(parsed);
}

std::string unquote(std::string value) {
    if (value.size() >= 2 &&
        ((value.front() == '"' && value.back() == '"') ||
         (value.front() == '\'' && value.back() == '\''))) {
        return value.substr(1, value.size() - 2);
    }
    return value;
}

}  // namespace

Config read_config(const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("无法打开配置文件: " + path.string());

    const std::unordered_set<std::string> allowed = {
        "dt", "num_steps", "record_interval", "G", "softening", "integrator"};
    std::unordered_map<std::string, std::string> values;
    std::string line;
    int line_number = 0;
    while (std::getline(input, line)) {
        ++line_number;
        const auto comment = line.find('#');
        if (comment != std::string::npos) line.erase(comment);
        line = trim(line);
        if (line.empty()) continue;
        const auto equal = line.find('=');
        if (equal == std::string::npos || line.find('=', equal + 1) != std::string::npos) {
            throw std::runtime_error("配置文件第 " + std::to_string(line_number) +
                                     " 行必须是 key = value");
        }
        const std::string key = trim(line.substr(0, equal));
        const std::string value = trim(line.substr(equal + 1));
        if (!allowed.count(key)) {
            throw std::runtime_error("配置文件第 " + std::to_string(line_number) +
                                     " 行包含未知键: " + key);
        }
        if (value.empty()) {
            throw std::runtime_error("配置项 " + key + " 没有值");
        }
        if (!values.emplace(key, value).second) {
            throw std::runtime_error("配置项重复: " + key);
        }
    }
    for (const auto& key : allowed) {
        if (!values.count(key)) throw std::runtime_error("缺少配置项: " + key);
    }

    Config config;
    config.dt = parse_double(values.at("dt"), "dt");
    config.num_steps = parse_int(values.at("num_steps"), "num_steps");
    config.record_interval = parse_int(values.at("record_interval"), "record_interval");
    config.gravitational_constant = parse_double(values.at("G"), "G");
    config.softening = parse_double(values.at("softening"), "softening");
    const std::string integrator = unquote(trim(values.at("integrator")));
    if (integrator == "euler") {
        config.integrator = Integrator::Euler;
    } else if (integrator == "leapfrog") {
        config.integrator = Integrator::Leapfrog;
    } else {
        throw std::runtime_error("integrator 只支持 euler 或 leapfrog");
    }

    if (!(config.dt > 0.0)) throw std::runtime_error("dt 必须大于 0");
    if (config.num_steps < 0) throw std::runtime_error("num_steps 不能为负数");
    if (config.record_interval < 1) throw std::runtime_error("record_interval 必须至少为 1");
    if (!(config.gravitational_constant > 0.0)) throw std::runtime_error("G 必须大于 0");
    if (!(config.softening > 0.0)) throw std::runtime_error("softening 必须大于 0");
    return config;
}

std::string to_string(Integrator integrator) {
    return integrator == Integrator::Euler ? "euler" : "leapfrog";
}

}  // namespace nbody

