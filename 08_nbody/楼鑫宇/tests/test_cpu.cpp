#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "nbody/cpu_solver.hpp"

namespace {

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

bool near(double lhs, double rhs, double tolerance = 1.0e-12) {
    return std::abs(lhs - rhs) <= tolerance;
}

}  // namespace

int main() {
    try {
        std::vector<nbody::Particle> particle = {{{1, 2, 3}, {0.5, -1, 2}, 1}};
        const auto acceleration = nbody::compute_accelerations(particle, 1.0, 1e-4);
        require(acceleration.size() == 1 && nbody::norm(acceleration[0]) == 0.0,
                "单粒子加速度应为零");

        nbody::Config motion_config;
        motion_config.num_steps = 4;
        motion_config.dt = 0.25;
        motion_config.record_interval = 3;
        motion_config.integrator = nbody::Integrator::Leapfrog;
        const auto motion = nbody::simulate_cpu(particle, motion_config);
        require(near(motion.particles[0].position.x, 1.5) &&
                    near(motion.particles[0].position.y, 1.0) &&
                    near(motion.particles[0].position.z, 5.0),
                "单粒子应保持匀速直线运动");
        require(motion.recorded_steps == std::vector<int>({0, 3, 4}),
                "记录步应包含初始、间隔和末状态");

        nbody::RunOptions compute_only;
        compute_only.record_enabled = false;
        compute_only.diagnostics_enabled = false;
        const auto unrecorded =
            nbody::simulate_cpu(particle, motion_config, compute_only);
        require(unrecorded.recorded_steps.empty() &&
                    unrecorded.trajectory.empty(),
                "record=off 不应保留轨迹");
        require(!unrecorded.record_enabled &&
                    !unrecorded.diagnostics_enabled,
                "运行选项没有写入结果");

        std::vector<nbody::Particle> pair(2);
        pair[0] = {{-1, 0, 0}, {}, 1};
        pair[1] = {{1, 0, 0}, {}, 1};
        const auto pair_acceleration = nbody::compute_accelerations(pair, 1.0, 0.0);
        require(near(pair_acceleration[0].x, 0.25) &&
                    near(pair_acceleration[1].x, -0.25),
                "二体加速度应大小相等、方向相反");
        require(near(pair_acceleration[0].y, 0.0) &&
                    near(pair_acceleration[1].z, 0.0),
                "二体加速度应位于连线上");

        pair[1].position = pair[0].position;
        const auto overlap_acceleration = nbody::compute_accelerations(pair, 1.0, 0.1);
        require(nbody::norm(overlap_acceleration[0]) == 0.0 &&
                    nbody::norm(overlap_acceleration[1]) == 0.0,
                "重合粒子在软化模型下不应产生非有限加速度");

        require(nbody::make_recorded_steps(4, 3) == std::vector<int>({0, 3, 4}),
                "记录步生成失败");
        std::cout << "CPU core tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "CPU core test failed: " << error.what() << '\n';
        return 1;
    }
}
