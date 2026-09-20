#include "io.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace {

std::string trim(const std::string& text) {
    const auto first = text.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) {
        return "";
    }
    const auto last = text.find_last_not_of(" \t\r\n");
    return text.substr(first, last - first + 1);
}

std::string strip_quotes(std::string value) {
    value = trim(value);
    if (value.size() >= 2 &&
        ((value.front() == '"' && value.back() == '"') ||
         (value.front() == '\'' && value.back() == '\''))) {
        return value.substr(1, value.size() - 2);
    }
    return value;
}

void ensure_parent_directory(const std::string& path) {
    const std::filesystem::path output_path(path);
    if (output_path.has_parent_path()) {
        std::filesystem::create_directories(output_path.parent_path());
    }
}

}  // namespace

std::vector<Particle> read_particles(const std::string& path) {
    std::ifstream file(path);
    if (!file) {
        throw std::runtime_error("无法打开粒子文件: " + path);
    }

    std::vector<Particle> particles;
    std::string line;
    int line_number = 0;
    while (std::getline(file, line)) {
        ++line_number;
        const auto comment = line.find('#');
        if (comment != std::string::npos) {
            line.erase(comment);
        }
        line = trim(line);
        if (line.empty()) {
            continue;
        }

        Particle p;
        std::istringstream input(line);
        if (!(input >> p.x >> p.y >> p.z >> p.vx >> p.vy >> p.vz >> p.mass)) {
            throw std::runtime_error("粒子文件第 " + std::to_string(line_number) +
                                     " 行格式错误，应为 7 个数");
        }
        std::string extra;
        if (input >> extra) {
            throw std::runtime_error("粒子文件第 " + std::to_string(line_number) +
                                     " 行有多余内容");
        }
        if (!std::isfinite(p.mass) || p.mass <= 0.0f) {
            throw std::runtime_error("粒子文件第 " + std::to_string(line_number) +
                                     " 行的质量必须是正数");
        }
        particles.push_back(p);
    }

    if (particles.empty()) {
        throw std::runtime_error("粒子文件为空: " + path);
    }
    return particles;
}

SimConfig read_config(const std::string& path) {
    std::ifstream file(path);
    if (!file) {
        throw std::runtime_error("无法打开配置文件: " + path);
    }

    SimConfig config;
    std::string line;
    int line_number = 0;
    while (std::getline(file, line)) {
        ++line_number;
        const auto comment = line.find('#');
        if (comment != std::string::npos) {
            line.erase(comment);
        }
        line = trim(line);
        if (line.empty()) {
            continue;
        }

        const auto equal = line.find('=');
        if (equal == std::string::npos) {
            throw std::runtime_error("配置文件第 " + std::to_string(line_number) +
                                     " 行缺少 '='");
        }
        const std::string key = trim(line.substr(0, equal));
        const std::string value = strip_quotes(line.substr(equal + 1));
        try {
            if (key == "dt") config.dt = std::stof(value);
            else if (key == "num_steps") config.num_steps = std::stoi(value);
            else if (key == "record_interval") config.record_interval = std::stoi(value);
            else if (key == "G") config.G = std::stof(value);
            else if (key == "softening") config.softening = std::stof(value);
            else if (key == "integrator") config.integrator = value;
            else if (key == "kernel") config.kernel = value;
            else if (key == "block_size") config.block_size = std::stoi(value);
            else if (key == "source_chunks") config.source_chunks = std::stoi(value);
            else if (key == "cuda_graph") {
                if (value == "true" || value == "1") config.cuda_graph = true;
                else if (value == "false" || value == "0") config.cuda_graph = false;
                else throw std::runtime_error("cuda_graph 只支持 true/false 或 1/0");
            }
            else throw std::runtime_error("未知配置项: " + key);
        } catch (const std::invalid_argument&) {
            throw std::runtime_error("配置文件第 " + std::to_string(line_number) +
                                     " 行的值无法解析: " + value);
        } catch (const std::out_of_range&) {
            throw std::runtime_error("配置文件第 " + std::to_string(line_number) +
                                     " 行的值超出范围: " + value);
        }
    }

    if (!(config.dt > 0.0f) || config.num_steps < 0 || config.record_interval <= 0 ||
        !(config.G > 0.0f) || config.softening < 0.0f) {
        throw std::runtime_error("配置值不合法：dt/G 必须为正，num_steps 不能为负，"
                                 "record_interval 必须为正，softening 不能为负");
    }
    if (config.integrator != "euler" && config.integrator != "leapfrog") {
        throw std::runtime_error("integrator 只支持 euler 或 leapfrog");
    }
    if (config.kernel != "baseline" && config.kernel != "tiled" &&
        config.kernel != "tiled-float4" && config.kernel != "split-source") {
        throw std::runtime_error(
            "kernel 只支持 baseline、tiled、tiled-float4 或 split-source");
    }
    if (config.block_size <= 0 || config.block_size > 1024) {
        throw std::runtime_error("block_size 必须在 1 到 1024 之间");
    }
    if (config.source_chunks <= 0 || config.source_chunks > 64) {
        throw std::runtime_error("source_chunks 必须在 1 到 64 之间");
    }
    return config;
}

int expected_record_count(const SimConfig& config) {
    // 第 0 帧是初始位置，之后只保存能被 record_interval 整除的步数。
    return 1 + config.num_steps / config.record_interval;
}

ConservationValues compute_conservation(const std::vector<Particle>& particles,
                                        float G,
                                        float softening) {
    ConservationValues values;
    const double softening2 = static_cast<double>(softening) * softening;
    for (std::size_t i = 0; i < particles.size(); ++i) {
        const auto& p = particles[i];
        values.px += static_cast<double>(p.mass) * p.vx;
        values.py += static_cast<double>(p.mass) * p.vy;
        values.pz += static_cast<double>(p.mass) * p.vz;
        values.kinetic += 0.5 * static_cast<double>(p.mass) *
                          (static_cast<double>(p.vx) * p.vx +
                           static_cast<double>(p.vy) * p.vy +
                           static_cast<double>(p.vz) * p.vz);

        // 势能只算 j > i，这样同一对粒子不会被重复统计。
        for (std::size_t j = i + 1; j < particles.size(); ++j) {
            const double dx = static_cast<double>(particles[j].x) - p.x;
            const double dy = static_cast<double>(particles[j].y) - p.y;
            const double dz = static_cast<double>(particles[j].z) - p.z;
            const double distance = std::sqrt(dx * dx + dy * dy + dz * dz + softening2);
            values.potential -= static_cast<double>(G) * p.mass * particles[j].mass / distance;
        }
    }
    values.total_energy = values.kinetic + values.potential;
    return values;
}

TrajectoryWriter::TrajectoryWriter(const std::string& path,
                                   int particle_count,
                                   int record_count)
    : particle_count_(particle_count), expected_records_(record_count) {
    ensure_parent_directory(path);
    file_.open(path, std::ios::binary | std::ios::trunc);
    if (!file_) {
        throw std::runtime_error("无法创建轨迹文件: " + path);
    }
    const std::int32_t count = particle_count;
    const std::int32_t records = record_count;
    file_.write(reinterpret_cast<const char*>(&count), sizeof(count));
    file_.write(reinterpret_cast<const char*>(&records), sizeof(records));
}

void TrajectoryWriter::write_record(const std::vector<float>& x,
                                    const std::vector<float>& y,
                                    const std::vector<float>& z) {
    if (static_cast<int>(x.size()) != particle_count_ || x.size() != y.size() || x.size() != z.size()) {
        throw std::runtime_error("写轨迹时数组长度与粒子数不一致");
    }
    if (written_records_ >= expected_records_) {
        throw std::runtime_error("写入的轨迹帧数超过文件头中的 record_count");
    }
    for (int i = 0; i < particle_count_; ++i) {
        const float xyz[3] = {x[i], y[i], z[i]};
        file_.write(reinterpret_cast<const char*>(xyz), sizeof(xyz));
    }
    if (!file_) {
        throw std::runtime_error("写入轨迹文件失败");
    }
    ++written_records_;
}

void TrajectoryWriter::close() {
    if (file_.is_open()) {
        if (written_records_ != expected_records_) {
            throw std::runtime_error("轨迹帧数不匹配：预计 " + std::to_string(expected_records_) +
                                     "，实际 " + std::to_string(written_records_));
        }
        file_.close();
    }
}

TrajectoryWriter::~TrajectoryWriter() {
    // 析构函数里不抛异常，正常路径由 close() 检查帧数。
    if (file_.is_open()) {
        file_.close();
    }
}

void write_performance_log(const std::string& path,
                           int particle_count,
                           const SimConfig& config,
                           const std::string& kernel_name,
                           const PerformanceData& perf,
                           const ConservationValues& initial,
                           const ConservationValues& final) {
    ensure_parent_directory(path);
    std::ofstream out(path, std::ios::trunc);
    if (!out) {
        throw std::runtime_error("无法创建性能日志: " + path);
    }

    const double particle_steps = static_cast<double>(particle_count) * config.num_steps;
    const double particle_steps_per_sec = perf.gpu_kernel_ms > 0.0f
        ? particle_steps / (perf.gpu_kernel_ms / 1000.0)
        : 0.0;
    const double avg_step_ms = config.num_steps > 0
        ? perf.gpu_kernel_ms / config.num_steps
        : 0.0;
    const double speedup = perf.cpu_simulation_ms >= 0.0 && perf.gpu_kernel_ms > 0.0f
        ? perf.cpu_simulation_ms / perf.gpu_kernel_ms
        : -1.0;
    const double momentum_error = std::sqrt(
        (final.px - initial.px) * (final.px - initial.px) +
        (final.py - initial.py) * (final.py - initial.py) +
        (final.pz - initial.pz) * (final.pz - initial.pz));
    const double relative_energy_error = std::abs(initial.total_energy) > 0.0
        ? std::abs((final.total_energy - initial.total_energy) / initial.total_energy)
        : std::abs(final.total_energy - initial.total_energy);

    out << std::setprecision(10) << std::fixed;
    out << "particle_count = " << particle_count << '\n';
    out << "num_steps = " << config.num_steps << '\n';
    out << "record_interval = " << config.record_interval << '\n';
    out << "integrator = " << config.integrator << '\n';
    out << "kernel = " << kernel_name << '\n';
    out << "block_size = " << config.block_size << '\n';
    out << "source_chunks = " << config.source_chunks << '\n';
    out << "cuda_graph = " << (config.cuda_graph ? "on" : "off") << '\n';
    out << "input_read_ms = " << perf.input_read_ms << '\n';
    out << "total_simulation_wall_ms = " << perf.simulation_wall_ms << '\n';
    out << "host_to_device_ms = " << perf.h2d_ms << '\n';
    out << "gpu_simulation_ms = " << perf.gpu_kernel_ms << '\n';
    out << "average_time_per_step_ms = " << avg_step_ms << '\n';
    out << "particle_steps_per_sec = " << particle_steps_per_sec << '\n';
    out << "device_to_host_ms = " << perf.d2h_ms << '\n';
    out << "trajectory_write_ms = " << perf.trajectory_write_ms << '\n';
    out << "gpu_allocated_bytes = " << perf.gpu_allocated_bytes << '\n';
    out << "gpu_free_bytes_after_alloc = " << perf.gpu_free_bytes_after_alloc << '\n';
    out << "gpu_total_bytes = " << perf.gpu_total_bytes << '\n';
    if (perf.cpu_simulation_ms >= 0.0) {
        out << "cpu_simulation_ms = " << perf.cpu_simulation_ms << '\n';
        out << "cpu_gpu_speedup = " << speedup << '\n';
        out << "max_position_error = " << perf.max_position_error << '\n';
        out << "rms_position_error = " << perf.rms_position_error << '\n';
        out << "max_velocity_error = " << perf.max_velocity_error << '\n';
        out << "rms_velocity_error = " << perf.rms_velocity_error << '\n';
    } else {
        out << "cpu_simulation_ms = skipped\n";
        out << "cpu_gpu_speedup = unavailable\n";
        out << "max_position_error = unavailable\n";
        out << "rms_position_error = unavailable\n";
        out << "max_velocity_error = unavailable\n";
        out << "rms_velocity_error = unavailable\n";
    }
    if (perf.conservation_available) {
        out << "initial_momentum = " << initial.px << ' ' << initial.py << ' ' << initial.pz << '\n';
        out << "final_momentum = " << final.px << ' ' << final.py << ' ' << final.pz << '\n';
        out << "momentum_error = " << momentum_error << '\n';
        out << "initial_energy = " << initial.total_energy << '\n';
        out << "final_energy = " << final.total_energy << '\n';
        out << "relative_energy_error = " << relative_energy_error << '\n';
    } else {
        out << "initial_momentum = unavailable\n";
        out << "final_momentum = unavailable\n";
        out << "momentum_error = unavailable\n";
        out << "initial_energy = unavailable\n";
        out << "final_energy = unavailable\n";
        out << "relative_energy_error = unavailable\n";
    }
}
