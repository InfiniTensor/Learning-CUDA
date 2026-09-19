#include "nbody.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>

namespace {
int parse_int(const std::string &s) {
  std::size_t end = 0;
  int value = std::stoi(s, &end);
  if (end != s.size())
    throw std::runtime_error("Invalid integer: " + s);
  return value;
}
std::string json_string(const std::string &value) {
  std::ostringstream out;
  out << '"';
  for (unsigned char c : value) {
    if (c == '"' || c == '\\')
      out << '\\' << c;
    else if (c < 32)
      out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << int(c)
          << std::dec;
    else
      out << c;
  }
  out << '"';
  return out.str();
}
void write_diag(std::ostream &out, const nb::Diagnostics &a,
                const nb::Diagnostics &b) {
  const double e0 = a.kinetic + a.potential, e1 = b.kinetic + b.potential;
  double dp = 0, dl = 0;
  for (int k = 0; k < 3; ++k) {
    dp += std::pow(b.momentum[k] - a.momentum[k], 2);
    dl += std::pow(b.angular[k] - a.angular[k], 2);
  }
  out << "{\"initial_energy\":" << e0 << ",\"final_energy\":" << e1
      << ",\"relative_energy_error\":"
      << std::abs(e1 - e0) / std::max(std::abs(e0), 1e-30)
      << ",\"momentum_drift_norm\":" << std::sqrt(dp)
      << ",\"normalized_momentum_error\":"
      << std::sqrt(dp) / std::max(a.momentum_scale, 1e-30)
      << ",\"angular_momentum_drift_norm\":" << std::sqrt(dl)
      << ",\"initial_momentum\":[" << a.momentum[0] << ',' << a.momentum[1]
      << ',' << a.momentum[2] << "],\"final_momentum\":[" << b.momentum[0]
      << ',' << b.momentum[1] << ',' << b.momentum[2] << "]}";
}
} // namespace
int main(int argc, char **argv) {
  try {
    using Clock = std::chrono::steady_clock;
    const auto total_begin = Clock::now();
    const std::set<std::string> flags{"--no-trajectory", "--no-diagnostics",
                                      "--compare-cpu"};
    const std::set<std::string> values{
        "--input",   "--config", "--output",     "--log",        "--final",
        "--backend", "--kernel", "--block-size", "--cpu-threads"};
    std::map<std::string, std::string> args;
    for (int i = 1; i < argc; ++i) {
      std::string key = argv[i];
      if (key == "--help") {
        std::cout << "nbody --input particles.txt --config config.txt "
                     "[--output trajectory.bin]\n"
                     "  [--backend cuda|cpu] [--kernel tiled|naive] "
                     "[--block-size 128|256]\n"
                     "  [--cpu-threads N] [--compare-cpu] [--no-trajectory] "
                     "[--no-diagnostics]\n"
                     "  [--log metrics.json] [--final final.txt]\n";
        return 0;
      }
      if (args.count(key))
        throw std::runtime_error("Duplicate argument: " + key);
      if (flags.count(key))
        args[key] = "true";
      else if (values.count(key) && i + 1 < argc)
        args[key] = argv[++i];
      else
        throw std::runtime_error("Unknown or incomplete argument: " + key);
    }
    if (!args.count("--input") || !args.count("--config"))
      throw std::runtime_error(
          "--input and --config are required (see --help)");
    auto get = [&](const std::string &key, const std::string &fallback) {
      return args.count(key) ? args.at(key) : fallback;
    };
    nb::Options o;
    o.backend = get("--backend", "cuda");
    o.kernel = get("--kernel", "tiled");
    o.block_size = parse_int(get("--block-size", "128"));
    o.cpu_threads = parse_int(get("--cpu-threads", "1"));
    if ((o.backend != "cuda" && o.backend != "cpu") ||
        (o.kernel != "tiled" && o.kernel != "naive") ||
        (o.block_size != 128 && o.block_size != 256) || o.cpu_threads < 1 ||
        o.cpu_threads > 1024)
      throw std::runtime_error(
          "Invalid backend, kernel, block size or thread count");
    if (args.count("--compare-cpu") &&
        (o.backend != "cuda" || !args.count("--no-trajectory")))
      throw std::runtime_error("--compare-cpu requires --backend cuda "
                               "--no-trajectory for equal timing conditions");
    auto c = nb::read_config(args.at("--config"));
    auto s = nb::read_particles(args.at("--input"));
    auto initial = s;
    // Reject aliasing output/input paths before opening any output with
    // truncation.
    std::set<std::filesystem::path> paths;
    for (const auto &key : {"--input", "--config"})
      paths.insert(std::filesystem::weakly_canonical(args.at(key)));
    std::vector<std::string> outputs{get("--log", "output/metrics.json")};
    if (!args.count("--no-trajectory")) {
      outputs.push_back(get("--output", "output/trajectory.bin"));
      outputs.push_back(outputs.back() + ".json");
    }
    if (args.count("--final"))
      outputs.push_back(args.at("--final"));
    for (const auto &path : outputs)
      if (!paths.insert(std::filesystem::weakly_canonical(path)).second)
        throw std::runtime_error("Input/output paths must be distinct");
    nb::Diagnostics before;
    if (!args.count("--no-diagnostics"))
      before = nb::diagnostics(s, c);
    std::unique_ptr<nb::Trajectory> trajectory;
    nb::Record record;
    if (!args.count("--no-trajectory")) {
      trajectory = std::make_unique<nb::Trajectory>(
          get("--output", "output/trajectory.bin"),
          static_cast<int>(s.position.size()), c);
      record = [&](int step, const nb::Vec4 *p) {
        trajectory->record(step, p);
      };
    }
    auto timing = o.backend == "cuda" ? nb::run_cuda(s, c, o, record)
                                      : nb::run_cpu(s, c, o, record);
    for (std::size_t i = 0; i < s.position.size(); ++i) {
      auto p = s.position[i], v = s.velocity[i];
      if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z) ||
          !std::isfinite(v.x) || !std::isfinite(v.y) || !std::isfinite(v.z))
        throw std::runtime_error(
            "Non-finite result; reduce dt or increase softening");
    }
    double flush_ms = 0;
    if (trajectory) {
      auto t = Clock::now();
      trajectory->finish();
      flush_ms =
          std::chrono::duration<double, std::milli>(Clock::now() - t).count();
    }
    if (args.count("--final"))
      nb::write_particles(args.at("--final"), s);
    nb::Diagnostics after;
    if (!args.count("--no-diagnostics"))
      after = nb::diagnostics(s, c);
    nb::Timing cpu;
    double max_error = 0;
    if (args.count("--compare-cpu")) {
      cpu = nb::run_cpu(initial, c, o, {});
      for (std::size_t i = 0; i < s.position.size(); ++i) {
        auto a = s.position[i], b = initial.position[i];
        max_error = std::max({max_error, std::abs(double(a.x) - b.x),
                              std::abs(double(a.y) - b.y),
                              std::abs(double(a.z) - b.z)});
      }
    }
    const double total_ms =
        std::chrono::duration<double, std::milli>(Clock::now() - total_begin)
            .count();
    std::ostringstream log;
    log << std::setprecision(12) << "{\n  \"particles\":" << s.position.size()
        << ",\"steps\":" << c.steps << ",\"dt\":" << c.dt << ",\"G\":" << c.G
        << ",\"softening\":" << c.softening
        << ",\"integrator\":" << json_string(c.integrator)
        << ",\n  \"backend\":" << json_string(o.backend)
        << ",\"kernel\":" << json_string(o.kernel)
        << ",\"block_size\":" << o.block_size
        << ",\"cpu_threads\":" << o.cpu_threads
        << ",\"device\":" << json_string(timing.device)
        << ",\n  \"compute_ms\":" << timing.compute_ms
        << ",\"simulation_wall_ms\":" << timing.wall_ms
        << ",\"trajectory_flush_ms\":" << flush_ms
        << ",\"total_ms\":" << total_ms
        << ",\"mean_step_ms\":" << (c.steps ? timing.compute_ms / c.steps : 0)
        << ",\"particle_steps_per_second\":"
        << (c.steps ? s.position.size() * double(c.steps) * 1000 /
                          std::max(timing.compute_ms, 1e-30)
                    : 0)
        << ",\"device_allocated_bytes\":" << timing.device_bytes
        << ",\"trajectory_records\":"
        << (trajectory ? nb::recording_steps(c).size() : 0)
        << ",\n  \"diagnostics\":";
    if (args.count("--no-diagnostics"))
      log << "null";
    else
      write_diag(log, before, after);
    log << ",\n  \"cpu_comparison\":";
    if (!args.count("--compare-cpu"))
      log << "null";
    else
      log << "{\"compute_ms\":" << cpu.compute_ms
          << ",\"simulation_wall_ms\":" << cpu.wall_ms
          << ",\"compute_speedup\":"
          << cpu.compute_ms / std::max(timing.compute_ms, 1e-30)
          << ",\"wall_speedup\":"
          << cpu.wall_ms / std::max(timing.wall_ms, 1e-30)
          << ",\"max_position_abs_error\":" << max_error << '}';
    log << "\n}\n";
    auto log_path = get("--log", "output/metrics.json");
    nb::make_parent(log_path);
    std::ofstream out(log_path);
    out << log.str();
    out.close();
    if (!out)
      throw std::runtime_error("Cannot write performance log");
    std::cout << log.str();
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "error: " << e.what() << '\n';
    return 1;
  }
}
