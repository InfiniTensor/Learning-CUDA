#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>

#include "nbody/config.hpp"
#include "nbody/io.hpp"

namespace {

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void write_text(const std::filesystem::path& path, const std::string& text) {
    std::ofstream output(path);
    if (!output) throw std::runtime_error("无法创建测试文件");
    output << text;
}

void expect_error(const std::function<void()>& action, const char* message) {
    try {
        action();
    } catch (const std::exception&) {
        return;
    }
    throw std::runtime_error(message);
}

class TempDirectory {
public:
    TempDirectory() {
        const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
        path_ = std::filesystem::temp_directory_path() /
                ("nbody_io_test_" + std::to_string(stamp));
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
        const auto valid_config = temporary.path() / "valid.cfg";
        write_text(valid_config,
                   "# parser smoke test\n"
                   "dt = 0.001\n"
                   "num_steps = 1000\n"
                   "record_interval = 10 # inline comment\n"
                   "G = 1.0\n"
                   "softening = 0.01\n"
                   "integrator = \"leapfrog\"\n");
        const nbody::Config config = nbody::read_config(valid_config);
        require(std::abs(config.dt - 0.001) < 1e-15, "dt 解析错误");
        require(config.num_steps == 1000, "num_steps 解析错误");
        require(config.record_interval == 10, "record_interval 解析错误");
        require(config.integrator == nbody::Integrator::Leapfrog,
                "integrator 解析错误");

        const auto invalid_config = temporary.path() / "invalid.cfg";
        write_text(invalid_config,
                   "dt = -0.001\nnum_steps = 1\nrecord_interval = 1\n"
                   "G = 1\nsoftening = 0.01\nintegrator = leapfrog\n");
        expect_error([&] { (void)nbody::read_config(invalid_config); },
                     "非法 dt 应被拒绝");

        const auto unknown_key = temporary.path() / "unknown.cfg";
        write_text(unknown_key,
                   "dt = 0.001\nnum_steps = 1\nrecord_interval = 1\n"
                   "G = 1\nsoftening = 0.01\nintegrator = leapfrog\n"
                   "typo = 7\n");
        expect_error([&] { (void)nbody::read_config(unknown_key); },
                     "未知配置键应被拒绝");

        const auto valid_particles = temporary.path() / "valid_particles.txt";
        write_text(valid_particles,
                   "# x y z vx vy vz mass\n"
                   "-0.5 0 0 0 -0.7 0 1\n"
                   "0.5 0 0 0 0.7 0 2 # second particle\n");
        const auto particles = nbody::read_particles(valid_particles);
        require(particles.size() == 2, "粒子数量解析错误");
        require(particles[0].position.x == -0.5 && particles[1].mass == 2.0,
                "粒子字段解析错误");

        const auto missing_field = temporary.path() / "missing_field.txt";
        write_text(missing_field, "0 0 0 0 0 1\n");
        expect_error([&] { (void)nbody::read_particles(missing_field); },
                     "不足七列的粒子行应被拒绝");

        const auto extra_field = temporary.path() / "extra_field.txt";
        write_text(extra_field, "0 0 0 0 0 0 1 extra\n");
        expect_error([&] { (void)nbody::read_particles(extra_field); },
                     "多余粒子字段应被拒绝");

        const auto invalid_mass = temporary.path() / "invalid_mass.txt";
        write_text(invalid_mass, "0 0 0 0 0 0 0\n");
        expect_error([&] { (void)nbody::read_particles(invalid_mass); },
                     "非正质量应被拒绝");

        std::cout << "I/O parsing tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "I/O parsing test failed: " << error.what() << '\n';
        return 1;
    }
}
