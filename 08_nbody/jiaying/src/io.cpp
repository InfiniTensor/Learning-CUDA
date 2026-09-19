#include "nbody.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <sys/mman.h>
#include <unistd.h>

namespace nb {
namespace {
std::string trim(std::string s) {
  auto first = s.find_first_not_of(" \t\r\n\"");
  if (first == std::string::npos)
    return {};
  return s.substr(first, s.find_last_not_of(" \t\r\n\"") - first + 1);
}
double number(const std::string &s) {
  std::size_t end = 0;
  double value = std::stod(s, &end);
  if (end != s.size() || !std::isfinite(value))
    throw std::runtime_error("Invalid number: " + s);
  return value;
}
int integer(const std::string &s) {
  double v = number(s);
  if (v < 0 || v > 1000000000 || v != std::floor(v))
    throw std::runtime_error("Invalid integer: " + s);
  return static_cast<int>(v);
}
} // namespace
void make_parent(const std::string &path) {
  auto parent = std::filesystem::path(path).parent_path();
  if (!parent.empty())
    std::filesystem::create_directories(parent);
}
Config read_config(const std::string &path) {
  std::ifstream in(path);
  if (!in)
    throw std::runtime_error("Cannot read config: " + path);
  Config c;
  std::set<std::string> seen;
  std::string line;
  while (std::getline(in, line)) {
    line = trim(line.substr(0, line.find('#')));
    if (line.empty())
      continue;
    auto eq = line.find('=');
    if (eq == std::string::npos)
      throw std::runtime_error("Expected key = value: " + line);
    auto key = trim(line.substr(0, eq)), value = trim(line.substr(eq + 1));
    if (!seen.insert(key).second)
      throw std::runtime_error("Duplicate key: " + key);
    if (key == "dt")
      c.dt = static_cast<float>(number(value));
    else if (key == "G")
      c.G = static_cast<float>(number(value));
    else if (key == "softening")
      c.softening = static_cast<float>(number(value));
    else if (key == "num_steps")
      c.steps = integer(value);
    else if (key == "record_interval")
      c.record_interval = integer(value);
    else if (key == "integrator")
      c.integrator = value;
    else
      throw std::runtime_error("Unknown config key: " + key);
  }
  if (!std::isfinite(c.dt) || c.dt <= 0 || !std::isfinite(c.G) || c.G < 0 ||
      !std::isfinite(c.softening) || c.softening <= 0 ||
      !std::isnormal(c.softening * c.softening) || c.record_interval < 1 ||
      (c.integrator != "leapfrog" && c.integrator != "euler"))
    throw std::runtime_error(
        "Require finite dt > 0, G >= 0, representable softening^2 > 0, "
        "record_interval > 0 and euler/leapfrog");
  return c;
}
State read_particles(const std::string &path) {
  std::ifstream in(path);
  if (!in)
    throw std::runtime_error("Cannot read particles: " + path);
  State s;
  std::string line;
  int line_number = 0;
  while (std::getline(in, line)) {
    ++line_number;
    line = trim(line.substr(0, line.find('#')));
    if (line.empty())
      continue;
    std::istringstream row(line);
    float a[7];
    for (float &value : a) {
      if (!(row >> value) || !std::isfinite(value))
        throw std::runtime_error("Invalid particle on line " +
                                 std::to_string(line_number));
    }
    std::string extra;
    if (row >> extra || a[6] <= 0)
      throw std::runtime_error(
          "Expected seven values and positive mass on line " +
          std::to_string(line_number));
    s.position.push_back({a[0], a[1], a[2], a[6]});
    s.velocity.push_back({a[3], a[4], a[5], 0});
  }
  if (s.position.empty() || s.position.size() > 1000000)
    throw std::runtime_error("Particle count must be in [1, 1000000]");
  return s;
}
void write_particles(const std::string &path, const State &s) {
  make_parent(path);
  std::ofstream out(path);
  out << std::setprecision(9);
  for (std::size_t i = 0; i < s.position.size(); ++i) {
    auto p = s.position[i], v = s.velocity[i];
    out << p.x << ' ' << p.y << ' ' << p.z << ' ' << v.x << ' ' << v.y << ' '
        << v.z << ' ' << p.w << '\n';
  }
  out.close();
  if (!out)
    throw std::runtime_error("Cannot write final state: " + path);
}
std::vector<int> recording_steps(const Config &c) {
  std::vector<int> result{0};
  for (std::int64_t step = c.record_interval; step <= c.steps;
       step += c.record_interval)
    result.push_back(static_cast<int>(step));
  if (result.back() != c.steps)
    result.push_back(c.steps);
  return result;
}
Trajectory::Trajectory(const std::string &path, int particles, const Config &c)
    : particles_(particles), steps_(recording_steps(c)) {
  static_assert(sizeof(float) == 4 && sizeof(std::int32_t) == 4);
  const std::uint16_t endian = 1;
  if (*reinterpret_cast<const unsigned char *>(&endian) != 1)
    throw std::runtime_error("Little-endian host required");
  bytes_ = 8 + static_cast<std::size_t>(particles) * steps_.size() * 3 *
                   sizeof(float);
  make_parent(path);
  fd_ = open(path.c_str(), O_CREAT | O_TRUNC | O_RDWR, 0644);
  if (fd_ < 0)
    throw std::runtime_error("Cannot open trajectory: " + path);
  if (ftruncate(fd_, static_cast<off_t>(bytes_)) != 0) {
    close(fd_);
    fd_ = -1;
    throw std::runtime_error("Cannot size trajectory file");
  }
  mapping_ = mmap(nullptr, bytes_, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
  if (mapping_ == MAP_FAILED) {
    mapping_ = nullptr;
    close(fd_);
    fd_ = -1;
    throw std::runtime_error("Cannot map trajectory");
  }
  std::int32_t header[2] = {particles,
                            static_cast<std::int32_t>(steps_.size())};
  std::memcpy(mapping_, header, sizeof(header));
  std::ofstream meta(path + ".json");
  meta << std::setprecision(9)
       << "{\"format\":\"int32 P,R; float32 positions[P,R,3]; "
          "little-endian\",\"dt\":"
       << c.dt << ",\"steps\":[";
  for (std::size_t k = 0; k < steps_.size(); ++k)
    meta << (k ? "," : "") << steps_[k];
  meta << "]}\n";
  meta.close();
  if (!meta) {
    munmap(mapping_, bytes_);
    mapping_ = nullptr;
    close(fd_);
    fd_ = -1;
    throw std::runtime_error("Cannot write trajectory metadata");
  }
}
Trajectory::~Trajectory() {
  if (mapping_)
    munmap(mapping_, bytes_);
  if (fd_ >= 0)
    close(fd_);
}
void Trajectory::record(int step, const Vec4 *p) {
  if (index_ >= static_cast<int>(steps_.size()) || steps_[index_] != step)
    throw std::runtime_error("Recording step mismatch");
  auto *data = reinterpret_cast<float *>(static_cast<char *>(mapping_) + 8);
  for (int i = 0; i < particles_; ++i) {
    const auto offset =
        (static_cast<std::size_t>(i) * steps_.size() + index_) * 3;
    if (!std::isfinite(p[i].x) || !std::isfinite(p[i].y) ||
        !std::isfinite(p[i].z))
      throw std::runtime_error(
          "Non-finite trajectory; reduce dt or increase softening");
    data[offset] = p[i].x;
    data[offset + 1] = p[i].y;
    data[offset + 2] = p[i].z;
  }
  ++index_;
}
void Trajectory::finish() {
  if (index_ != static_cast<int>(steps_.size()))
    throw std::runtime_error("Incomplete trajectory");
  if (msync(mapping_, bytes_, MS_SYNC) != 0)
    throw std::runtime_error("Cannot flush trajectory");
}
} // namespace nb
