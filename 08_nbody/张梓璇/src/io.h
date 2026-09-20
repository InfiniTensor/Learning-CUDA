#pragma once

#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

struct Particle {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
    float vx = 0.0f;
    float vy = 0.0f;
    float vz = 0.0f;
    float mass = 1.0f;
};

struct SimConfig {
    float dt = 1.0e-3f;
    int num_steps = 1000;
    int record_interval = 100;
    float G = 1.0f;
    float softening = 1.0e-4f;
    std::string integrator = "leapfrog";
    std::string kernel = "tiled";
    int block_size = 256;
    int source_chunks = 2;
    bool cuda_graph = false;
};

struct ConservationValues {
    double px = 0.0;
    double py = 0.0;
    double pz = 0.0;
    double kinetic = 0.0;
    double potential = 0.0;
    double total_energy = 0.0;
};

struct PerformanceData {
    double input_read_ms = 0.0;
    double simulation_wall_ms = 0.0;
    float h2d_ms = 0.0f;
    float gpu_kernel_ms = 0.0f;
    float d2h_ms = 0.0f;
    double trajectory_write_ms = 0.0;
    double cpu_simulation_ms = -1.0;
    std::size_t gpu_allocated_bytes = 0;
    std::size_t gpu_free_bytes_after_alloc = 0;
    std::size_t gpu_total_bytes = 0;
    double max_position_error = -1.0;
    double rms_position_error = -1.0;
    double max_velocity_error = -1.0;
    double rms_velocity_error = -1.0;
    bool conservation_available = true;
};

std::vector<Particle> read_particles(const std::string& path);
SimConfig read_config(const std::string& path);
int expected_record_count(const SimConfig& config);
ConservationValues compute_conservation(const std::vector<Particle>& particles,
                                        float G,
                                        float softening);

class TrajectoryWriter {
public:
    TrajectoryWriter(const std::string& path, int particle_count, int record_count);
    void write_record(const std::vector<float>& x,
                      const std::vector<float>& y,
                      const std::vector<float>& z);
    void close();
    ~TrajectoryWriter();

private:
    std::ofstream file_;
    int particle_count_ = 0;
    int expected_records_ = 0;
    int written_records_ = 0;
};

void write_performance_log(const std::string& path,
                           int particle_count,
                           const SimConfig& config,
                           const std::string& kernel_name,
                           const PerformanceData& perf,
                           const ConservationValues& initial,
                           const ConservationValues& final);
