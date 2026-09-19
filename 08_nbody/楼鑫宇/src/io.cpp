#include "nbody/io.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

namespace nbody {
namespace {

std::string trim(const std::string& text) {
    const auto begin = text.find_first_not_of(" \t\r\n");
    if (begin == std::string::npos) return {};
    const auto end = text.find_last_not_of(" \t\r\n");
    return text.substr(begin, end - begin + 1);
}

bool finite_particle(const Particle& particle) {
    return std::isfinite(particle.position.x) && std::isfinite(particle.position.y) &&
           std::isfinite(particle.position.z) && std::isfinite(particle.velocity.x) &&
           std::isfinite(particle.velocity.y) && std::isfinite(particle.velocity.z) &&
           std::isfinite(particle.mass);
}

void write_u32_le(std::ostream& output, std::uint32_t value) {
    for (int shift = 0; shift < 32; shift += 8) {
        output.put(static_cast<char>((value >> shift) & 0xffu));
    }
}

void write_f32_le(std::ostream& output, float value) {
    static_assert(sizeof(float) == sizeof(std::uint32_t), "需要 32 位 float");
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    write_u32_le(output, bits);
}

double vector_norm(const Vec3& value) {
    return std::sqrt(dot(value, value));
}

}  // namespace

std::vector<Particle> read_particles(const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("无法打开粒子文件: " + path.string());
    std::vector<Particle> particles;
    std::string line;
    int line_number = 0;
    while (std::getline(input, line)) {
        ++line_number;
        const auto comment = line.find('#');
        if (comment != std::string::npos) line.erase(comment);
        line = trim(line);
        if (line.empty()) continue;
        Particle particle;
        std::istringstream parser(line);
        if (!(parser >> particle.position.x >> particle.position.y >> particle.position.z >>
              particle.velocity.x >> particle.velocity.y >> particle.velocity.z >>
              particle.mass)) {
            throw std::runtime_error("粒子文件第 " + std::to_string(line_number) +
                                     " 行必须包含 7 个数值");
        }
        std::string extra;
        if (parser >> extra) {
            throw std::runtime_error("粒子文件第 " + std::to_string(line_number) +
                                     " 行包含多余字段");
        }
        if (!finite_particle(particle)) {
            throw std::runtime_error("粒子文件第 " + std::to_string(line_number) +
                                     " 行包含非有限数值");
        }
        if (!(particle.mass > 0.0)) {
            throw std::runtime_error("粒子文件第 " + std::to_string(line_number) +
                                     " 行质量必须大于 0");
        }
        particles.push_back(particle);
    }
    if (particles.empty()) throw std::runtime_error("粒子文件没有有效数据");
    return particles;
}

void write_run_outputs(const std::filesystem::path& output_dir,
                       const std::filesystem::path& input_path,
                       const Config& config,
                       const CpuRunResult& result,
                       double pre_output_wall_ms,
                       const std::string& backend,
                       const std::string& precision,
                       std::optional<double> theta) {
    const std::size_t particle_count = result.particles.size();
    const std::size_t record_count = result.recorded_steps.size();
    if (particle_count > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()) ||
        record_count > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::runtime_error("轨迹头部超出 int32 范围");
    }
    if (result.record_enabled &&
        result.trajectory.size() != particle_count * record_count * 3) {
        throw std::runtime_error("内部轨迹长度不一致");
    }
    if (result.record_enabled) {
        std::ofstream output(output_dir / "trajectory.bin", std::ios::binary);
        if (!output) throw std::runtime_error("无法创建 trajectory.bin");
        write_u32_le(output, static_cast<std::uint32_t>(particle_count));
        write_u32_le(output, static_cast<std::uint32_t>(record_count));
        for (float value : result.trajectory) write_f32_le(output, value);
        if (!output) throw std::runtime_error("写入 trajectory.bin 失败");
    }
    {
        std::ofstream output(output_dir / "final_state.csv");
        if (!output) throw std::runtime_error("无法创建 final_state.csv");
        output << "particle_id,x,y,z,vx,vy,vz,mass\n" << std::setprecision(17);
        for (std::size_t i = 0; i < particle_count; ++i) {
            const auto& p = result.particles[i];
            output << i << ',' << p.position.x << ',' << p.position.y << ',' << p.position.z
                   << ',' << p.velocity.x << ',' << p.velocity.y << ',' << p.velocity.z
                   << ',' << p.mass << '\n';
        }
    }
    if (result.diagnostics_enabled) {
        std::ofstream output(output_dir / "diagnostics.csv");
        if (!output) throw std::runtime_error("无法创建 diagnostics.csv");
        output << "stage,kinetic,potential,total_energy,px,py,pz,lx,ly,lz,cmx,cmy,cmz\n"
               << std::setprecision(17);
        const auto write_row = [&output](const char* stage, const Diagnostics& d) {
            output << stage << ',' << d.kinetic << ',' << d.potential << ','
                   << d.total_energy << ',' << d.momentum.x << ',' << d.momentum.y
                   << ',' << d.momentum.z << ',' << d.angular_momentum.x << ','
                   << d.angular_momentum.y << ',' << d.angular_momentum.z << ','
                   << d.center_of_mass.x << ',' << d.center_of_mass.y << ','
                   << d.center_of_mass.z << '\n';
        };
        write_row("initial", result.initial_diagnostics);
        write_row("final", result.final_diagnostics);
    }
    double relative_energy_error = 0.0;
    double momentum_error = 0.0;
    if (result.diagnostics_enabled) {
        const double energy_scale = std::max(
            result.initial_diagnostics.kinetic +
                std::abs(result.initial_diagnostics.potential),
            1e-30);
        relative_energy_error = std::abs(
            result.final_diagnostics.total_energy -
            result.initial_diagnostics.total_energy) /
            energy_scale;
        momentum_error = vector_norm(
            result.final_diagnostics.momentum -
            result.initial_diagnostics.momentum);
    }
    const double particle_steps_per_second =
        config.num_steps == 0 || result.simulation_ms <= 0.0 ? 0.0 :
        static_cast<double>(particle_count) * config.num_steps /
        (result.simulation_ms / 1000.0);
    {
        std::ofstream output(output_dir / "performance.json");
        if (!output) throw std::runtime_error("无法创建 performance.json");
        output << std::setprecision(17) << "{\n"
               << "  \"schema_version\": 1,\n"
               << "  \"backend\": \"" << backend << "\",\n"
               << "  \"precision\": \"" << precision << "\",\n"
               << "  \"theta\": ";
        if (theta) output << *theta;
        else output << "null";
        output << ",\n"
               << "  \"record_enabled\": "
               << (result.record_enabled ? "true" : "false") << ",\n"
               << "  \"diagnostics_mode\": \""
               << (result.diagnostics_enabled ? "final" : "off") << "\",\n"
               << "  \"particle_count\": " << particle_count << ",\n"
               << "  \"num_steps\": " << config.num_steps << ",\n"
               << "  \"force_evaluations\": " << result.force_evaluations << ",\n"
               << "  \"force_total_ms\": " << result.force_ms << ",\n"
               << "  \"simulation_wall_ms\": " << result.simulation_ms << ",\n"
               << "  \"pre_output_wall_ms\": " << pre_output_wall_ms << ",\n"
               << "  \"particle_steps_per_sec\": " << particle_steps_per_second << ",\n";
        if (result.diagnostics_enabled) {
            output << "  \"relative_energy_error\": " << relative_energy_error
                   << ",\n"
                   << "  \"absolute_momentum_error\": " << momentum_error << "\n";
        } else {
            output << "  \"relative_energy_error\": null,\n"
                   << "  \"absolute_momentum_error\": null\n";
        }
        output << "}\n";
    }
    {
        std::ofstream output(output_dir / "metadata.json");
        if (!output) throw std::runtime_error("无法创建 metadata.json");
        output << std::setprecision(17) << "{\n"
               << "  \"schema_version\": 1,\n"
               << "  \"status\": \"complete\",\n"
               << "  \"backend\": \"" << backend << "\",\n"
               << "  \"precision\": \"" << precision << "\",\n"
               << "  \"theta\": ";
        if (theta) output << *theta;
        else output << "null";
        output << ",\n"
               << "  \"record_enabled\": "
               << (result.record_enabled ? "true" : "false") << ",\n"
               << "  \"diagnostics_mode\": \""
               << (result.diagnostics_enabled ? "final" : "off") << "\",\n"
               << "  \"input_file\": \"" << input_path.filename().string() << "\",\n"
               << "  \"particle_count\": " << particle_count << ",\n"
               << "  \"record_count\": " << record_count << ",\n"
               << "  \"recorded_steps\": [";
        for (std::size_t i = 0; i < record_count; ++i) {
            if (i) output << ',';
            output << result.recorded_steps[i];
        }
        output << "],\n  \"times\": [";
        for (std::size_t i = 0; i < record_count; ++i) {
            if (i) output << ',';
            output << result.recorded_steps[i] * config.dt;
        }
        output << "],\n  \"config\": {\n"
               << "    \"dt\": " << config.dt << ",\n"
               << "    \"num_steps\": " << config.num_steps << ",\n"
               << "    \"record_interval\": " << config.record_interval << ",\n"
               << "    \"G\": " << config.gravitational_constant << ",\n"
               << "    \"softening\": " << config.softening << ",\n"
               << "    \"integrator\": \"" << to_string(config.integrator) << "\"\n"
               << "  },\n  \"trajectory\": ";
        if (result.record_enabled) {
            output << "{\n"
                   << "    \"file\": \"trajectory.bin\",\n"
                   << "    \"dtype\": \"<f4\",\n"
                   << "    \"layout\": \"particle_record_xyz\",\n"
                   << "    \"header\": \"<i4,<i4\"\n"
                   << "  }\n";
        } else {
            output << "null\n";
        }
        output << "}\n";
    }
}

}  // namespace nbody
