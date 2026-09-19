#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "nbody/cpu_solver.hpp"

namespace {

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

double position_error(double dt, double final_time) {
    constexpr double softening = 1.0e-4;
    const double speed = std::sqrt(0.5 / std::pow(1.0 + softening * softening, 1.5));
    const double angular_speed = 2.0 * speed;
    const std::vector<nbody::Particle> initial = {
        {{-0.5, 0, 0}, {0, -speed, 0}, 1},
        {{0.5, 0, 0}, {0, speed, 0}, 1},
    };

    nbody::Config config;
    config.dt = dt;
    config.num_steps = static_cast<int>(std::llround(final_time / dt));
    config.record_interval = config.num_steps;
    config.gravitational_constant = 1.0;
    config.softening = softening;
    config.integrator = nbody::Integrator::Leapfrog;

    const auto result = nbody::simulate_cpu(initial, config);
    const double angle = angular_speed * final_time;
    const nbody::Vec3 exact{-0.5 * std::cos(angle), -0.5 * std::sin(angle), 0};
    return nbody::norm(result.particles[0].position - exact);
}

}  // namespace

int main() {
    try {
        constexpr double final_time = 10.0;
        const std::array<double, 3> errors = {
            position_error(1.0e-3, final_time),
            position_error(5.0e-4, final_time),
            position_error(2.5e-4, final_time),
        };
        require(errors[0] > errors[1] && errors[1] > errors[2],
                "减小 dt 后位置误差没有单调下降");

        const double first_ratio = errors[0] / errors[1];
        const double second_ratio = errors[1] / errors[2];
        require(first_ratio > 3.7 && first_ratio < 4.3,
                "第一次减半 dt 未呈现 Leapfrog 二阶收敛");
        require(second_ratio > 3.7 && second_ratio < 4.3,
                "第二次减半 dt 未呈现 Leapfrog 二阶收敛");

        std::cout << "Leapfrog dt convergence passed: errors="
                  << errors[0] << ',' << errors[1] << ',' << errors[2]
                  << " ratios=" << first_ratio << ',' << second_ratio << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Convergence test failed: " << error.what() << '\n';
        return 1;
    }
}
