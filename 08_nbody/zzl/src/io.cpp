#include "nbody.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>

namespace nbody {
namespace {
std::string trim(const std::string &s) {
    const auto first = s.find_first_not_of(" \t\r\n");
    return first == std::string::npos ? ""
                                      : s.substr(first, s.find_last_not_of(" \t\r\n") - first + 1);
}
std::string clean_line(std::string line) {
    if (line.compare(0, 3, "\xEF\xBB\xBF") == 0)
        line.erase(0, 3);
    return trim(line.substr(0, line.find('#')));
}
double number(const std::string &s) {
    std::size_t consumed = 0;
    const double value = std::stod(s, &consumed);
    if (consumed != s.size() || !std::isfinite(value))
        throw std::runtime_error("Invalid finite number: " + s);
    return value;
}
float real(const std::string &s) {
    const double value = number(s);
    const float result = static_cast<float>(value);
    if (!std::isfinite(result) || (value != 0 && result == 0))
        throw std::runtime_error("Number outside float range: " + s);
    return result;
}
int integer(const std::string &s) {
    const double value = number(s);
    if (value < 0 || value > std::numeric_limits<int>::max() - 1 || std::floor(value) != value)
        throw std::runtime_error("Expected nonnegative integer: " + s);
    return static_cast<int>(value);
}
void parents(const std::filesystem::path &path) {
    if (!path.parent_path().empty())
        std::filesystem::create_directories(path.parent_path());
}
void write_i32(std::ostream &out, std::uint32_t value) {
    char bytes[4];
    for (unsigned i = 0; i < 4; ++i)
        bytes[i] = static_cast<char>((value >> (i * 8)) & 255);
    out.write(bytes, 4);
}
void write_f32(std::ostream &out, const std::vector<float> &data) {
    static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559);
    const std::uint16_t endian = 1;
    if (*reinterpret_cast<const unsigned char *>(&endian) == 1)
        out.write(reinterpret_cast<const char *>(data.data()),
                  static_cast<std::streamsize>(data.size() * 4));
    else
        for (float value : data) {
            std::uint32_t bits;
            std::memcpy(&bits, &value, 4);
            write_i32(out, bits);
        }
}
} // namespace

State read_particles(const std::filesystem::path &path) {
    std::ifstream file(path);
    if (!file)
        throw std::runtime_error("Cannot open particles: " + path.string());
    State state;
    std::string line;
    int line_number = 0;
    double total_mass = 0;
    while (std::getline(file, line)) {
        ++line_number;
        line = clean_line(line);
        if (line.empty())
            continue;
        try {
            std::istringstream row(line);
            std::vector<float> values;
            std::string token;
            while (row >> token)
                values.push_back(real(token));
            if (values.size() != 7 || values[6] < 0)
                throw std::runtime_error("Expected x y z vx vy vz mass, mass >= 0");
            state.position.push_back({values[0], values[1], values[2], values[6]});
            state.velocity.push_back({values[3], values[4], values[5], 0});
            total_mass += values[6];
        } catch (const std::exception &e) {
            throw std::runtime_error(path.string() + ":" + std::to_string(line_number) + ": " +
                                     e.what());
        }
    }
    if (file.bad())
        throw std::runtime_error("Error reading particle file");
    if (state.position.empty() || total_mass <= 0)
        throw std::runtime_error("Need at least one particle and positive total mass");
    if (state.position.size() > static_cast<std::size_t>(std::numeric_limits<int>::max() - 512))
        throw std::runtime_error("Too many particles for 32-bit kernel indexing");
    return state;
}

Config read_config(const std::filesystem::path &path) {
    std::ifstream file(path);
    if (!file)
        throw std::runtime_error("Cannot open config: " + path.string());
    Config c;
    std::set<std::string> seen;
    std::string line;
    while (std::getline(file, line)) {
        line = clean_line(line);
        if (line.empty())
            continue;
        const auto equal = line.find('=');
        if (equal == std::string::npos)
            throw std::runtime_error("Config line needs '=': " + line);
        const auto key = trim(line.substr(0, equal));
        auto value = trim(line.substr(equal + 1));
        if (!seen.insert(key).second)
            throw std::runtime_error("Duplicate config key: " + key);
        if (key == "dt")
            c.dt = real(value);
        else if (key == "num_steps")
            c.num_steps = integer(value);
        else if (key == "record_interval")
            c.record_interval = integer(value);
        else if (key == "G")
            c.G = real(value);
        else if (key == "softening")
            c.softening = real(value);
        else if (key == "integrator") {
            if (value.size() >= 2 && ((value.front() == '"' && value.back() == '"') ||
                                      (value.front() == '\'' && value.back() == '\'')))
                value = value.substr(1, value.size() - 2);
            c.integrator = value;
        } else
            throw std::runtime_error("Unknown config key: " + key);
    }
    if (file.bad())
        throw std::runtime_error("Error reading config file");
    const float eps2 = c.softening * c.softening;
    if (c.dt <= 0 || c.G < 0 || c.softening <= 0 || !std::isnormal(eps2) ||
        c.record_interval <= 0 || (c.integrator != "euler" && c.integrator != "leapfrog"))
        throw std::runtime_error("Require dt > 0, G >= 0, positive representable softening^2, "
                                 "record_interval > 0, integrator = euler or leapfrog");
    return c;
}

int record_count(const Config &c) {
    // The initial state, one frame per interval, and the final state if it is not already one.
    const int intervals = c.num_steps / c.record_interval;
    return intervals + 1 + (c.num_steps % c.record_interval != 0 ? 1 : 0);
}

int record_step_at(int index, const Config &c) {
    if (index <= 0)
        return 0;
    const std::int64_t step = std::int64_t(index) * c.record_interval;
    return static_cast<int>(std::min<std::int64_t>(step, c.num_steps));
}

TrajectoryWriter::TrajectoryWriter(const std::filesystem::path &path, int particles,
                                   const Config &c)
    : path_(path), spool_path_(path.string() + ".frames.tmp"),
      temporary_path_(path.string() + ".tmp"), particles_(particles), config_(c),
      records_(record_count(c)) {
    if (std::filesystem::exists(path_) || std::filesystem::exists(path_.string() + ".json") ||
        std::filesystem::exists(spool_path_) || std::filesystem::exists(temporary_path_))
        throw std::runtime_error("Trajectory path already exists; choose a new output path");
    parents(path_);
    spool_.exceptions(std::ios::failbit | std::ios::badbit);
    spool_.open(spool_path_, std::ios::binary);
    frame_.resize(static_cast<std::size_t>(particles_) * 3);
}
TrajectoryWriter::~TrajectoryWriter() {
    try {
        if (spool_.is_open())
            spool_.close();
        std::error_code ignored;
        std::filesystem::remove(spool_path_, ignored);
        std::filesystem::remove(temporary_path_, ignored);
    } catch (...) {
        // Destructors must not mask a preceding I/O or CUDA failure.
    }
}
void TrajectoryWriter::append(int step, const std::vector<Vec4> &positions) {
    if (recorded_ >= static_cast<std::size_t>(records_) ||
        step != record_step_at(static_cast<int>(recorded_), config_) ||
        positions.size() != static_cast<std::size_t>(particles_))
        throw std::runtime_error("Unexpected trajectory frame");
    for (std::size_t i = 0; i < positions.size(); ++i) {
        const auto &p = positions[i];
        if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z))
            throw std::runtime_error("Non-finite trajectory: reduce dt or rescale units");
        frame_[3 * i] = p.x;
        frame_[3 * i + 1] = p.y;
        frame_[3 * i + 2] = p.z;
    }
    // Internal temporary format is native endian, frame-major.
    spool_.write(reinterpret_cast<const char *>(frame_.data()),
                 static_cast<std::streamsize>(frame_.size() * 4));
    ++recorded_;
}
void TrajectoryWriter::finish() {
    if (recorded_ != static_cast<std::size_t>(records_))
        throw std::runtime_error("Trajectory is incomplete");
    spool_.close();
    std::ifstream input(spool_path_, std::ios::binary);
    input.exceptions(std::ios::failbit | std::ios::badbit);
    std::ofstream output(temporary_path_, std::ios::binary);
    output.exceptions(std::ios::failbit | std::ios::badbit);
    write_i32(output, particles_);
    write_i32(output, static_cast<std::uint32_t>(recorded_));
    // 32 MiB destination budget reduces frame-spool seeks for large histories.
    // A single particle may exceed the budget if its history alone is larger.
    const auto chunk = std::max<std::size_t>(
        1, std::min<std::size_t>(particles_, (32 * 1024 * 1024) / (recorded_ * 12)));
    for (std::size_t base = 0; base < static_cast<std::size_t>(particles_); base += chunk) {
        const auto count = std::min(chunk, static_cast<std::size_t>(particles_) - base);
        std::vector<float> tile(count * recorded_ * 3), row(count * 3);
        for (std::size_t frame = 0; frame < recorded_; ++frame) {
            const auto offset = (frame * particles_ + base) * 12;
            // A whole-particle chunk consumes the spool sequentially from byte
            // zero; avoid a redundant filesystem seek before each frame.
            if (count != static_cast<std::size_t>(particles_))
                input.seekg(static_cast<std::streamoff>(offset));
            input.read(reinterpret_cast<char *>(row.data()),
                       static_cast<std::streamsize>(row.size() * 4));
            for (std::size_t i = 0; i < count; ++i)
                std::copy_n(row.data() + 3 * i, 3, tile.data() + (i * recorded_ + frame) * 3);
        }
        write_f32(output, tile);
    }
    output.close();
    // Publish the payload before its sidecar. An interrupted write then leaves a complete
    // trajectory without metadata, which the reader accepts, instead of a sidecar that
    // occupies the output path with no trajectory beside it.
    std::filesystem::rename(temporary_path_, path_);
    std::ofstream metadata(path_.string() + ".json");
    metadata.exceptions(std::ios::failbit | std::ios::badbit);
    metadata << std::setprecision(10) << "{\n  \"format\":\"nbody-particle-major-v1\",\n"
             << "  \"endianness\":\"little\",\n  \"dtype\":\"float32\",\n"
             << "  \"particles\":" << particles_ << ",\n  \"records\":" << recorded_
             << ",\n  \"dt\":" << config_.dt << ",\n  \"steps\":[";
    for (int i = 0; i < records_; ++i)
        metadata << (i ? "," : "") << record_step_at(i, config_);
    metadata << "]\n}\n";
    metadata.close();
}

void write_state(const std::filesystem::path &path, const State &s) {
    if (std::filesystem::exists(path))
        throw std::runtime_error("State output already exists");
    parents(path);
    std::ofstream out(path);
    out.exceptions(std::ios::failbit | std::ios::badbit);
    out << std::setprecision(9) << "particle_id,x,y,z,vx,vy,vz,mass\n";
    for (std::size_t i = 0; i < s.position.size(); ++i) {
        const auto &p = s.position[i];
        const auto &v = s.velocity[i];
        out << i << ',' << p.x << ',' << p.y << ',' << p.z << ',' << v.x << ',' << v.y << ',' << v.z
            << ',' << p.w << '\n';
    }
}
std::string json_string(const std::string &s) {
    std::ostringstream result;
    result << '"';
    for (const unsigned char ch : s) {
        if (ch == '\\' || ch == '"')
            result << '\\' << ch;
        else if (ch < 32)
            result << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                   << static_cast<int>(ch);
        else
            result << ch;
    }
    result << '"';
    return result.str();
}
} // namespace nbody
