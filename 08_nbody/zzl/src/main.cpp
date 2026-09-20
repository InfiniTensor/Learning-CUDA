#include "nbody.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cwctype>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>

namespace {
using Clock = std::chrono::steady_clock;
const char *help = R"(N-body gravity simulator
Usage: nbody --input particles.txt --config simulation.cfg [options]
  --backend cuda|cpu          Default: cuda (no silent CPU fallback)
  --output trajectory.bin    Default: trajectory.bin, never overwritten
  --log performance.json     Default: performance.json, never overwritten
  --final-state state.csv    Optional positions, velocities and masses
  --no-trajectory            Benchmark without trajectory disk I/O
  --compare-cpu              Measure a full identical CPU simulation as baseline
  --cpu-threads N            OpenMP threads (0 = runtime default)
  --kernel tiled|naive        CUDA force kernel; default tiled
  --block-size 128|256|512    Default: 256
  --cuda-execution auto|direct|graph  Auto batches long simulations with N <= 256
  --diagnostics auto|full|momentum  auto computes energy for N <= 8192
  --device-info              Print CUDA device JSON and exit
  --help                     Show this message
Config: dt, num_steps, record_interval, G, softening, integrator
)";
double length(const double *v) { return std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]); }
std::string array(const double *v) {
    std::ostringstream out;
    out << std::setprecision(17) << '[' << v[0] << ',' << v[1] << ',' << v[2] << ']';
    return out.str();
}
int parse_int(const std::string &s) {
    std::size_t used;
    const auto value = std::stoi(s, &used);
    if (used != s.size() || value < 0)
        throw std::runtime_error("Invalid nonnegative integer: " + s);
    return value;
}
} // namespace

int main(int argc, char **argv) {
    using namespace nbody;
    try {
        const auto program_start = Clock::now();
        std::map<std::string, std::string> args;
        const std::set<std::string> flags{"--no-trajectory", "--compare-cpu", "--help",
                                          "--device-info"};
        const std::set<std::string> values{
            "--input",       "--config",      "--backend", "--output",     "--log",
            "--final-state", "--cpu-threads", "--kernel",  "--block-size", "--diagnostics",
            "--cuda-execution"};
        for (int i = 1; i < argc; ++i) {
            const std::string key = argv[i];
            if (args.count(key))
                throw std::runtime_error("Duplicate option: " + key);
            if (flags.count(key))
                args[key] = "true";
            else if (values.count(key)) {
                // A value option consumes the next token, so an option placed where its
                // value belongs must be reported as missing. Otherwise "--output --no-trajectory"
                // would silently write a file named "--no-trajectory" and drop the flag.
                if (i + 1 >= argc || argv[i + 1][0] == '\0' ||
                    (argv[i + 1][0] == '-' && argv[i + 1][1] == '-'))
                    throw std::runtime_error("Missing value for option: " + key);
                args[key] = argv[++i];
            } else
                throw std::runtime_error("Unknown or incomplete option: " + key);
        }
        if (args.count("--help") || argc == 1) {
            std::cout << help;
            return 0;
        }
        if (args.count("--device-info")) {
#ifdef NBODY_HAS_CUDA
            std::cout << cuda_device_info() << '\n';
            return 0;
#else
            throw std::runtime_error("CUDA backend not compiled; configure NBODY_ENABLE_CUDA=ON");
#endif
        }
        if (!args.count("--input") || !args.count("--config"))
            throw std::runtime_error("Both --input and --config are required");
        Options options;
        if (args.count("--backend"))
            options.backend = args.at("--backend");
        if (args.count("--kernel"))
            options.kernel = args.at("--kernel");
        if (args.count("--block-size"))
            options.block_size = parse_int(args.at("--block-size"));
        if (args.count("--cpu-threads"))
            options.cpu_threads = parse_int(args.at("--cpu-threads"));
        if (args.count("--cuda-execution"))
            options.cuda_execution = args.at("--cuda-execution");
        if (options.cuda_execution != "auto" && options.cuda_execution != "direct" &&
            options.cuda_execution != "graph")
            throw std::runtime_error("Invalid CUDA execution mode");
        if (options.backend == "cpu" && args.count("--cuda-execution"))
            throw std::runtime_error("--cuda-execution requires --backend cuda");
        if ((options.backend != "cuda" && options.backend != "cpu") ||
            (options.kernel != "tiled" && options.kernel != "naive") ||
            (options.block_size != 128 && options.block_size != 256 && options.block_size != 512))
            throw std::runtime_error("Invalid backend, kernel or block size");
        const bool compare = args.count("--compare-cpu") != 0;
        if (compare && options.backend != "cuda")
            throw std::runtime_error("--compare-cpu requires --backend cuda");
        const bool trajectory = !args.count("--no-trajectory");
        const std::filesystem::path output =
            args.count("--output") ? args.at("--output") : "trajectory.bin";
        const std::filesystem::path log =
            args.count("--log") ? args.at("--log") : "performance.json";
        std::set<std::filesystem::path> destinations;
        const auto input_path = std::filesystem::absolute(args.at("--input")).lexically_normal();
        const auto config_path = std::filesystem::absolute(args.at("--config")).lexically_normal();
        auto check_destination = [&](const std::filesystem::path &p) {
            const auto absolute = std::filesystem::absolute(p).lexically_normal();
            auto key = std::filesystem::weakly_canonical(absolute);
#ifdef _WIN32
            // Windows paths are case insensitive; resolve parent aliases before comparing outputs.
            auto native = key.native();
            std::transform(native.begin(), native.end(), native.begin(),
                           [](wchar_t ch) { return static_cast<wchar_t>(std::towlower(ch)); });
            key = native;
#endif
            if (std::filesystem::exists(p) || absolute == input_path || absolute == config_path ||
                !destinations.insert(key).second)
                throw std::runtime_error("Output already exists or paths collide: " + p.string());
        };
        check_destination(log);
        if (trajectory) {
            check_destination(output);
            check_destination(output.string() + ".json");
            check_destination(output.string() + ".frames.tmp");
            check_destination(output.string() + ".tmp");
        }
        if (args.count("--final-state"))
            check_destination(args.at("--final-state"));
        const auto config = read_config(config_path);
        const auto initial = read_particles(input_path);
        const auto n = initial.position.size();
        const auto mode = args.count("--diagnostics") ? args.at("--diagnostics") : "auto";
        if (mode != "auto" && mode != "full" && mode != "momentum")
            throw std::runtime_error("Invalid diagnostics mode");
        const bool energy = mode == "full" || (mode == "auto" && n <= 8192);
        const auto d0 = diagnose(initial, config, energy);
        std::unique_ptr<TrajectoryWriter> writer;
        double trajectory_spool_seconds = 0;
        auto record_frame = [&](int step, const std::vector<Vec4> &p) {
            const auto started = Clock::now();
            writer->append(step, p);
            trajectory_spool_seconds +=
                std::chrono::duration<double>(Clock::now() - started).count();
        };
        if (trajectory) {
            writer = std::make_unique<TrajectoryWriter>(output, static_cast<int>(n), config);
            record_frame(0, initial.position);
        }
        Recorder record;
        if (writer)
            record = record_frame;
        auto state = initial;
        Timing timing;
        double warmup_seconds = 0;
        if (options.backend == "cpu") {
            timing = run_cpu(state, config, options, record);
        } else {
#ifdef NBODY_HAS_CUDA
            // Context creation and two warm-up steps are excluded from measured compute time.
            const auto warmup_start = Clock::now();
            auto warm_state = initial;
            auto warm_config = config;
            warm_config.num_steps = 2;
            auto warm_options = options;
            warm_options.cuda_execution = "direct";
            run_cuda(warm_state, warm_config, warm_options, {});
            warmup_seconds = std::chrono::duration<double>(Clock::now() - warmup_start).count();
            timing = run_cuda(state, config, options, record);
#else
            throw std::runtime_error(
                "CUDA backend not compiled; use --backend cpu or rebuild with CUDA");
#endif
        }
        const auto d1 = diagnose(state, config, energy);
        double cpu_seconds = 0, cpu_wall_seconds = 0;
        int cpu_threads = timing.cpu_threads;
        if (compare) {
            std::cerr << "Measuring full CPU baseline: " << n << " particles x " << config.num_steps
                      << " steps\n";
            auto reference = initial;
            const auto baseline = run_cpu(reference, config, options, {});
            diagnose(reference, config, false);
            cpu_seconds = baseline.compute_seconds;
            cpu_wall_seconds = baseline.backend_wall_seconds;
            cpu_threads = baseline.cpu_threads;
        }
        const auto io_start = Clock::now();
        double trajectory_finalize_seconds = 0, final_state_write_seconds = 0;
        if (writer) {
            const auto started = Clock::now();
            writer->finish();
            trajectory_finalize_seconds =
                std::chrono::duration<double>(Clock::now() - started).count();
        }
        if (args.count("--final-state")) {
            const auto started = Clock::now();
            write_state(args.at("--final-state"), state);
            final_state_write_seconds =
                std::chrono::duration<double>(Clock::now() - started).count();
        }
        const auto final_io_seconds =
            std::chrono::duration<double>(Clock::now() - io_start).count();
        double dp[3], com_residual[3];
        for (int i = 0; i < 3; ++i) {
            dp[i] = d1.momentum[i] - d0.momentum[i];
            com_residual[i] = d1.com[i] - d0.com[i] -
                              double(config.dt) * config.num_steps * d0.momentum[i] / d0.mass;
        }
        const auto total_seconds =
            std::chrono::duration<double>(Clock::now() - program_start).count();
        if (!log.parent_path().empty())
            std::filesystem::create_directories(log.parent_path());
        std::ofstream out(log);
        out.exceptions(std::ios::failbit | std::ios::badbit);
        out << std::setprecision(17) << "{\n"
            << "  \"backend\":" << json_string(options.backend) << ",\n"
            << "  \"device\":" << json_string(timing.device) << ",\n"
            << "  \"kernel\":" << json_string(options.kernel) << ",\n"
            << "  \"block_size\":" << options.block_size << ",\n"
            << "  \"particles\":" << n << ",\n  \"num_steps\":" << config.num_steps << ",\n"
            << "  \"dt\":" << config.dt << ",\n  \"G\":" << config.G << ",\n"
            << "  \"softening\":" << config.softening << ",\n"
            << "  \"integrator\":" << json_string(config.integrator) << ",\n"
            << "  \"record_interval\":" << config.record_interval << ",\n"
            << "  \"records\":" << (trajectory ? record_count(config) : 0) << ",\n"
            << "  \"simulation_seconds\":" << timing.compute_seconds << ",\n"
            << "  \"simulation_timer\":"
            << json_string(options.backend == "cuda" ? "cuda_event" : "steady_clock") << ",\n"
            << "  \"simulation_host_seconds\":" << timing.host_compute_seconds << ",\n"
            << "  \"cuda_execution\":"
            << json_string(options.backend == "cpu" ? "not_applicable" :
                           (timing.graph_batch_steps ? "graph" : "direct")) << ",\n"
            << "  \"graph_setup_seconds\":" << timing.graph_setup_seconds << ",\n"
            << "  \"graph_batch_steps\":" << timing.graph_batch_steps << ",\n"
            << "  \"graph_launches\":" << timing.graph_launches << ",\n"
            << "  \"direct_steps\":" << timing.direct_steps << ",\n"
            << "  \"average_step_ms\":"
            << (config.num_steps ? timing.compute_seconds * 1000 / config.num_steps : 0) << ",\n"
            << "  \"particle_steps_per_second\":"
            << (timing.compute_seconds > 0 ? n * double(config.num_steps) / timing.compute_seconds
                                           : 0)
            << ",\n"
            << "  \"device_allocation_bytes\":" << timing.device_bytes << ",\n"
            << "  \"transfer_seconds\":" << timing.transfer_seconds << ",\n"
            << "  \"backend_wall_seconds\":" << timing.backend_wall_seconds << ",\n"
            << "  \"warmup_seconds\":" << warmup_seconds << ",\n"
            << "  \"final_output_seconds\":" << final_io_seconds << ",\n"
            << "  \"trajectory_spool_seconds\":" << trajectory_spool_seconds << ",\n"
            << "  \"trajectory_finalize_seconds\":" << trajectory_finalize_seconds << ",\n"
            << "  \"final_state_write_seconds\":" << final_state_write_seconds << ",\n"
            << "  \"total_wall_seconds\":" << total_seconds << ",\n"
            << "  \"cpu_threads\":" << cpu_threads << ",\n"
            << "  \"cpu_simulation_seconds\":";
        if (compare)
            out << cpu_seconds;
        else
            out << "null";
        out << ",\n  \"cpu_backend_wall_seconds\":";
        if (compare)
            out << cpu_wall_seconds;
        else
            out << "null";
        out << ",\n  \"speedup_vs_cpu\":";
        if (compare && timing.compute_seconds > 0 && config.num_steps > 0)
            out << cpu_seconds / timing.compute_seconds;
        else
            out << "null";
        out << ",\n  \"speedup_vs_cpu_host\":";
        if (compare && timing.host_compute_seconds > 0 && config.num_steps > 0)
            out << cpu_seconds / timing.host_compute_seconds;
        else
            out << "null";
        out << ",\n  \"energy_computed\":" << (energy ? "true" : "false")
            << ",\n  \"initial_energy\":";
        if (energy)
            out << d0.energy;
        else
            out << "null";
        out << ",\n  \"final_energy\":";
        if (energy)
            out << d1.energy;
        else
            out << "null";
        out << ",\n  \"relative_energy_error\":";
        if (energy && std::abs(d0.energy) > 1e-30)
            out << std::abs((d1.energy - d0.energy) / d0.energy);
        else
            out << "null";
        out << ",\n  \"initial_momentum\":" << array(d0.momentum)
            << ",\n  \"final_momentum\":" << array(d1.momentum)
            << ",\n  \"absolute_momentum_error\":" << length(dp)
            << ",\n  \"momentum_normalization_scale\":" << d0.momentum_scale
            << ",\n  \"normalized_momentum_error\":";
        if (d0.momentum_scale > 1e-30)
            out << length(dp) / d0.momentum_scale;
        else
            out << "null";
        out << ",\n  \"initial_angular_momentum\":" << array(d0.angular)
            << ",\n  \"final_angular_momentum\":" << array(d1.angular)
            << ",\n  \"center_of_mass_ballistic_error\":" << length(com_residual) << "\n}\n";
        out.close();
        std::cout << std::fixed << std::setprecision(6) << options.backend << ": " << n
                  << " particles, " << config.num_steps << " steps, " << timing.compute_seconds
                  << " s compute";
        if (compare && timing.compute_seconds > 0)
            std::cout << ", speedup " << cpu_seconds / timing.compute_seconds << "x";
        std::cout << "\nLog: " << log.string() << '\n';
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }
}
