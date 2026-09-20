#include "engine.hpp"
#include <chrono>
#include <iostream>
#include <stdexcept>
#include <string>

int main(int argc, char **argv) {
  using namespace pricing;
  using Clock = std::chrono::steady_clock;
  try {
    const auto start = Clock::now();
    std::string option_path, simulation_path, prefix = "results/pricing",
                                              backend = "both";
    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i];
      if (arg == "--help") {
        std::cout
            << "option_pricing --option OPTION.cfg --simulation SIMULATION.cfg "
               "[--backend cpu|gpu|both] [--output PREFIX]\n";
        return 0;
      }
      if (i + 1 == argc)
        throw std::runtime_error("Missing argument for " + arg);
      const std::string value = argv[++i];
      if (arg == "--option")
        option_path = value;
      else if (arg == "--simulation")
        simulation_path = value;
      else if (arg == "--output")
        prefix = value;
      else if (arg == "--backend")
        backend = value;
      else
        throw std::runtime_error("Unknown argument: " + arg);
    }
    if (option_path.empty() || simulation_path.empty())
      throw std::runtime_error(
          "--option and --simulation are required; use --help");
    if (backend != "cpu" && backend != "gpu" && backend != "both")
      throw std::runtime_error("backend must be cpu, gpu or both");
#ifndef OPTION_HAS_CUDA
    if (backend != "cpu")
      throw std::runtime_error(
          "This executable was built without CUDA; use --backend cpu");
#endif
    const Option o = read_option(option_path);
    const Simulation s = read_simulation(simulation_path);
    const double io_ms =
        std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    Run cpu, gpu;
    const bool use_cpu = backend != "gpu", use_gpu = backend != "cpu";
    if (use_cpu)
      cpu = run_cpu(o, s);
#ifdef OPTION_HAS_CUDA
    if (use_gpu)
      gpu = run_gpu(o, s);
#endif
    const auto e = estimate(o, s, use_gpu ? gpu : cpu);
    const double process_ms =
        std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    write_outputs(prefix, o, s, use_cpu ? &cpu : nullptr,
                  use_gpu ? &gpu : nullptr, io_ms, process_ms);
    std::cout.precision(10);
    std::cout << option_name(o.kind) << " price=" << e.price
              << " se=" << e.standard_error << " ci95=[" << e.low << ","
              << e.high << "]";
    if (e.has_reference)
      std::cout << " BlackScholes=" << e.reference;
    if (use_gpu)
      std::cout << " gpu_ms=" << gpu.total_ms;
    if (use_cpu)
      std::cout << " cpu_ms=" << cpu.total_ms;
    if (use_cpu && use_gpu)
      std::cout << " speedup=" << cpu.total_ms / gpu.total_ms;
    std::cout << "\nOutputs: " << prefix << ".json, " << prefix
              << "_performance.json\n";
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "error: " << e.what() << '\n';
    return 1;
  }
}
