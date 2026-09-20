// sim_params.cpp —— 参数文件解析实现
#include "sim_params.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace nbody {
namespace {

void Trim(std::string *s) {
  auto not_space = [](unsigned char c) { return !std::isspace(c); };
  s->erase(s->begin(), std::find_if(s->begin(), s->end(), not_space));
  s->erase(std::find_if(s->rbegin(), s->rend(), not_space).base(), s->end());
}

// 去掉行尾注释：支持 # 与 //
void StripComment(std::string *s) {
  const std::size_t hash = s->find('#');
  if (hash != std::string::npos)
    s->erase(hash);
  const std::size_t slashes = s->find("//");
  if (slashes != std::string::npos)
    s->erase(slashes);
}

void StripQuotes(std::string *s) {
  if (s->size() >= 2 && (s->front() == '"' || s->front() == '\'') &&
      s->back() == s->front()) {
    *s = s->substr(1, s->size() - 2);
  }
}

std::string ToLower(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return s;
}

} // namespace

const char *IntegratorName(Integrator integ) {
  return integ == Integrator::kEuler ? "euler" : "leapfrog";
}

SimParams ParseSimParams(const std::string &path) {
  std::ifstream in(path);
  if (!in)
    throw std::runtime_error("cannot open params file: " + path);

  SimParams p;
  std::string line;
  int lineno = 0;
  while (std::getline(in, line)) {
    ++lineno;
    StripComment(&line);
    Trim(&line);
    if (line.empty())
      continue;

    const std::size_t eq = line.find('=');
    if (eq == std::string::npos) {
      throw std::runtime_error("params line " + std::to_string(lineno) +
                               ": expected `key = value`, got: " + line);
    }
    std::string key = line.substr(0, eq);
    std::string value = line.substr(eq + 1);
    Trim(&key);
    Trim(&value);
    StripQuotes(&value);
    key = ToLower(key);

    auto as_double = [&]() {
      try {
        return std::stod(value);
      } catch (const std::exception &) {
        throw std::runtime_error("params line " + std::to_string(lineno) +
                                 ": `" + key +
                                 "` expects a number, got: " + value);
      }
    };
    auto as_int = [&]() {
      try {
        return std::stoi(value);
      } catch (const std::exception &) {
        throw std::runtime_error("params line " + std::to_string(lineno) +
                                 ": `" + key +
                                 "` expects an integer, got: " + value);
      }
    };

    if (key == "dt") {
      p.dt = static_cast<real>(as_double());
    } else if (key == "num_steps") {
      p.num_steps = as_int();
    } else if (key == "record_interval") {
      p.record_interval = as_int();
    } else if (key == "g") {
      p.G = static_cast<real>(as_double());
    } else if (key == "softening") {
      p.softening = static_cast<real>(as_double());
    } else if (key == "integrator") {
      const std::string v = ToLower(value);
      if (v == "euler") {
        p.integrator = Integrator::kEuler;
      } else if (v == "leapfrog") {
        p.integrator = Integrator::kLeapfrog;
      } else {
        throw std::runtime_error(
            "params line " + std::to_string(lineno) +
            ": integrator must be euler|leapfrog, got: " + value);
      }
    } else if (key == "format" || key == "output_format") {
      const std::string v = ToLower(value);
      if (v == "binary" || v == "bin") {
        p.format = TrajectoryFormat::kBinary;
      } else if (v == "csv") {
        p.format = TrajectoryFormat::kCsv;
      } else {
        throw std::runtime_error("params line " + std::to_string(lineno) +
                                 ": format must be binary|csv, got: " + value);
      }
    } else {
      // 不静默忽略：拼错的键（例如 `timestep` 而非 `dt`）会导致
      // "我明明设了却没生效"这类极难排查的问题。
      throw std::runtime_error("params line " + std::to_string(lineno) +
                               ": unknown key `" + key + "`");
    }
  }

  if (p.dt <= real(0))
    throw std::runtime_error("dt must be positive");
  if (p.num_steps <= 0)
    throw std::runtime_error("num_steps must be positive");
  if (p.record_interval <= 0)
    throw std::runtime_error("record_interval must be positive");
  if (p.softening < real(0))
    throw std::runtime_error("softening must be non-negative");
  return p;
}

} // namespace nbody
