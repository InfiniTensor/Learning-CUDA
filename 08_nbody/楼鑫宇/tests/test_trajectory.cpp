#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

#include "nbody/io.hpp"

namespace {

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

std::uint32_t read_u32_le(const std::vector<unsigned char>& bytes,
                          std::size_t offset) {
    return static_cast<std::uint32_t>(bytes[offset]) |
           (static_cast<std::uint32_t>(bytes[offset + 1]) << 8) |
           (static_cast<std::uint32_t>(bytes[offset + 2]) << 16) |
           (static_cast<std::uint32_t>(bytes[offset + 3]) << 24);
}

float read_f32_le(const std::vector<unsigned char>& bytes, std::size_t offset) {
    const std::uint32_t bits = read_u32_le(bytes, offset);
    float value = 0.0f;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

class TempDirectory {
public:
    TempDirectory() {
        const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
        path_ = std::filesystem::temp_directory_path() /
                ("nbody_trajectory_test_" + std::to_string(stamp));
        std::filesystem::create_directories(path_);
    }

    ~TempDirectory() {
        std::error_code error;
        std::filesystem::remove_all(path_, error);
    }

    const std::filesystem::path& path() const { return path_; }

private:
    std::filesystem::path path_;
};

}  // namespace

int main() {
    try {
        TempDirectory temporary;
        nbody::Config config;
        config.dt = 0.5;
        config.num_steps = 2;
        config.record_interval = 1;
        config.gravitational_constant = 1.0;
        config.softening = 0.01;
        config.integrator = nbody::Integrator::Leapfrog;

        nbody::CpuRunResult result;
        result.particles = {
            {{20, 21, 22}, {}, 1},
            {{120, 121, 122}, {}, 2},
        };
        result.recorded_steps = {0, 1, 2};
        result.trajectory = {
            0, 1, 2, 10, 11, 12, 20, 21, 22,
            100, 101, 102, 110, 111, 112, 120, 121, 122,
        };

        nbody::write_run_outputs(temporary.path(), "sentinel.txt", config,
                                 result, 0.0);

        std::ifstream input(temporary.path() / "trajectory.bin", std::ios::binary);
        require(static_cast<bool>(input), "无法读取轨迹测试文件");
        const std::vector<unsigned char> bytes{
            std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
        require(bytes.size() == 8 + 2 * 3 * 3 * sizeof(float),
                "轨迹文件字节数错误");
        require(read_u32_le(bytes, 0) == 2, "轨迹粒子数头部错误");
        require(read_u32_le(bytes, 4) == 3, "轨迹记录数头部错误");

        const std::vector<float> expected = {
            0, 1, 2, 10, 11, 12, 20, 21, 22,
            100, 101, 102, 110, 111, 112, 120, 121, 122,
        };
        for (std::size_t i = 0; i < expected.size(); ++i) {
            require(read_f32_le(bytes, 8 + i * sizeof(float)) == expected[i],
                    "轨迹不是约定的 [P,R,3] 粒子优先布局");
        }

        std::ifstream metadata_input(temporary.path() / "metadata.json");
        const std::string metadata{std::istreambuf_iterator<char>(metadata_input),
                                   std::istreambuf_iterator<char>()};
        require(metadata.find("\"recorded_steps\": [0,1,2]") != std::string::npos,
                "metadata 记录步错误");
        require(metadata.find("\"layout\": \"particle_record_xyz\"") !=
                    std::string::npos,
                "metadata 轨迹布局错误");

        std::cout << "Trajectory sentinel round-trip test passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Trajectory test failed: " << error.what() << '\n';
        return 1;
    }
}
