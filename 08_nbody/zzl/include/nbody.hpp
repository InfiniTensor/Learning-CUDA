#pragma once
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <string>
#include <vector>

namespace nbody {
struct alignas(16) Vec4 {
    float x = 0, y = 0, z = 0, w = 0;
};
struct State {
    std::vector<Vec4> position; // xyz position, w mass
    std::vector<Vec4> velocity;
};
struct Config {
    float dt = 1e-3f;
    int num_steps = 1000;
    int record_interval = 100;
    float G = 1;
    float softening = 1e-4f;
    std::string integrator = "leapfrog";
};
struct Options {
    std::string backend = "cuda";
    std::string kernel = "tiled";
    int block_size = 256;
    int cpu_threads = 0;
    std::string cuda_execution = "auto";
};
struct Timing {
    double compute_seconds = 0;
    double host_compute_seconds = 0;
    double graph_setup_seconds = 0;
    int graph_batch_steps = 0;
    int graph_launches = 0;
    int direct_steps = 0;
    double transfer_seconds = 0;
    double backend_wall_seconds = 0;
    std::size_t device_bytes = 0;
    std::string device = "CPU";
    int cpu_threads = 1;
};
struct Diagnostics {
    double mass = 0;
    double momentum[3] = {};
    double com[3] = {};
    double angular[3] = {};
    double kinetic = 0;
    double potential = 0;
    double energy = 0;
    double momentum_scale = 0;
};
using Recorder = std::function<void(int, const std::vector<Vec4> &)>;
State read_particles(const std::filesystem::path &path);
Config read_config(const std::filesystem::path &path);
// Saved frames are the initial state, every record_interval, and the final state.
// Both are closed forms so that a long record history costs no host memory.
int record_count(const Config &config);
int record_step_at(int index, const Config &config);
Diagnostics diagnose(const State &state, const Config &config, bool energy);
Timing run_cpu(State &state, const Config &config, const Options &options, const Recorder &record);
#ifdef NBODY_HAS_CUDA
Timing run_cuda(State &state, const Config &config, const Options &options, const Recorder &record);
std::string cuda_device_info();
#endif
void write_state(const std::filesystem::path &path, const State &state);
std::string json_string(const std::string &value);

// Frames are spooled to disk, then transposed in bounded chunks into particle-major order.
class TrajectoryWriter {
  public:
    TrajectoryWriter(const std::filesystem::path &path, int particles, const Config &config);
    ~TrajectoryWriter();
    void append(int step, const std::vector<Vec4> &positions);
    void finish();

  private:
    std::filesystem::path path_, spool_path_, temporary_path_;
    std::ofstream spool_;
    int particles_;
    Config config_;
    int records_ = 0;
    std::size_t recorded_ = 0;
    std::vector<float> frame_;
};
} // namespace nbody
