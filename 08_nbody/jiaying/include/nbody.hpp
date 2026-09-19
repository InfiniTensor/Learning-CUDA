#pragma once
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace nb {
// Identical layout to CUDA float4, including alignment and mass in position.w.
struct alignas(16) Vec4 {
  float x, y, z, w;
};
struct State {
  std::vector<Vec4> position, velocity;
};
struct Config {
  float dt = 0.001f, G = 1.0f, softening = 0.01f;
  int steps = 1000, record_interval = 10;
  std::string integrator = "leapfrog";
};
struct Options {
  std::string backend = "cuda", kernel = "tiled";
  int block_size = 128, cpu_threads = 1;
};
struct Timing {
  double compute_ms = 0, wall_ms = 0;
  std::uint64_t device_bytes = 0;
  std::string device = "CPU";
};
struct Diagnostics {
  double kinetic = 0, potential = 0, momentum[3] = {}, angular[3] = {};
  double com[3] = {}, mass = 0, momentum_scale = 0;
};
using Record = std::function<void(int, const Vec4 *)>;
Config read_config(const std::string &path);
State read_particles(const std::string &path);
void write_particles(const std::string &path, const State &state);
void make_parent(const std::string &path);
Diagnostics diagnostics(const State &state, const Config &config);
Timing run_cpu(State &state, const Config &config, const Options &options,
               const Record &record);
Timing run_cuda(State &state, const Config &config, const Options &options,
                const Record &record);
std::vector<int> recording_steps(const Config &config);

// A mapped particle-major file avoids retaining and transposing all frames in
// RAM.
class Trajectory {
public:
  Trajectory(const std::string &path, int particles, const Config &config);
  ~Trajectory();
  Trajectory(const Trajectory &) = delete;
  Trajectory &operator=(const Trajectory &) = delete;
  void record(int step, const Vec4 *positions);
  void finish();

private:
  int fd_ = -1, particles_ = 0, index_ = 0;
  std::size_t bytes_ = 0;
  void *mapping_ = nullptr;
  std::vector<int> steps_;
};
} // namespace nb
