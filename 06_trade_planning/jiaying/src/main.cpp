#include "flow.hpp"
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>

namespace {
std::string json_string(const std::string &value) {
  std::ostringstream out;
  out << '"';
  for (unsigned char c : value) {
    if (c == '"' || c == '\\')
      out << '\\' << c;
    else if (c < 32)
      out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << int(c);
    else
      out << c;
  }
  out << '"';
  return out.str();
}
void write_file(const std::string &path, const std::string &contents) {
  const auto parent = std::filesystem::path(path).parent_path();
  if (!parent.empty())
    std::filesystem::create_directories(parent);
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  out << contents;
  out.close();
  if (!out)
    throw std::runtime_error("Failed to write: " + path);
}
int integer(const std::map<std::string, std::string> &args,
            const std::string &key, int fallback) {
  if (!args.count(key))
    return fallback;
  size_t length = 0;
  const int value = std::stoi(args.at(key), &length);
  if (length != args.at(key).size())
    throw std::runtime_error("Invalid integer: " + key);
  return value;
}
} // namespace

int main(int argc, char **argv) {
  try {
    const auto process_start = trade::Clock::now();
    std::map<std::string, std::string> args;
    bool verify = false, graphs = true;
    for (int i = 1; i < argc; ++i) {
      const std::string key = argv[i];
      if (key == "--help") {
        std::cout
            << "trade_flow --graph graph.bin --queries queries.txt --output "
               "results.txt --log metrics.json\n"
               "  --engine cuda|cpu|cpu-pr (default cuda)\n"
               "  --check                 CPU Dinic equality and residual "
               "min-cut certificate\n"
               "  --schedule warp|thread  GPU work granularity (default "
               "thread)\n"
               "  --batch-rounds 8 --global-interval 64 --max-rounds 1000000\n"
               "  --no-graphs             launch the same kernels without CUDA "
               "Graphs\n";
        return 0;
      }
      if (key == "--check") {
        verify = true;
        continue;
      }
      if (key == "--no-graphs") {
        graphs = false;
        continue;
      }
      if (key != "--graph" && key != "--queries" && key != "--output" &&
          key != "--log" && key != "--engine" && key != "--schedule" &&
          key != "--batch-rounds" && key != "--global-interval" &&
          key != "--max-rounds")
        throw std::runtime_error("Unknown option: " + key);
      if (++i == argc || args.count(key))
        throw std::runtime_error("Missing value or duplicate: " + key);
      args[key] = argv[i];
    }
    for (const auto *key : {"--graph", "--queries", "--output", "--log"})
      if (!args.count(key))
        throw std::runtime_error(std::string("Required option: ") + key);
    std::vector<std::filesystem::path> paths;
    for (const auto *key : {"--graph", "--queries", "--output", "--log"}) {
      const auto p = std::filesystem::weakly_canonical(args.at(key));
      for (const auto &previous : paths)
        if (p == previous ||
            (std::filesystem::exists(p) && std::filesystem::exists(previous) &&
             std::filesystem::equivalent(p, previous)))
          throw std::runtime_error(
              "Input, output and log paths must be distinct");
      paths.push_back(p);
    }
    const std::string engine =
        args.count("--engine") ? args.at("--engine") : "cuda";
    if (engine != "cuda" && engine != "cpu" && engine != "cpu-pr")
      throw std::runtime_error("Engine must be cuda, cpu or cpu-pr");
    const std::string schedule =
        args.count("--schedule") ? args.at("--schedule") : "thread";
    if (schedule != "warp" && schedule != "thread")
      throw std::runtime_error("Invalid schedule");
    trade::GpuOptions options;
    options.warp = schedule == "warp";
    options.graphs = graphs;
    options.batch_rounds =
        integer(args, "--batch-rounds", options.batch_rounds);
    options.global_interval =
        integer(args, "--global-interval", options.global_interval);
    options.max_rounds = integer(args, "--max-rounds", options.max_rounds);

    const auto io_start = trade::Clock::now();
    const auto input = trade::read_graph(args.at("--graph"));
    const auto queries = trade::read_queries(args.at("--queries"), input.n);
    const double input_ms = trade::elapsed_ms(io_start);
    const auto preprocess_start = trade::Clock::now();
    const trade::ResidualGraph graph(input);
    std::unique_ptr<trade::GpuSolver> gpu;
    std::unique_ptr<trade::CpuSolver> cpu;
    if (engine == "cuda")
      gpu = std::make_unique<trade::GpuSolver>(graph, options);
    else
      cpu = std::make_unique<trade::CpuSolver>(graph);
    const double preprocess_ms = trade::elapsed_ms(preprocess_start);
    // Optional validation is deliberately outside all solver timings.
    const auto checker_start = trade::Clock::now();
    std::unique_ptr<trade::CpuSolver> checker;
    if (verify && gpu)
      checker = std::make_unique<trade::CpuSolver>(graph);
    double validation_ms = trade::elapsed_ms(checker_start);
    std::vector<trade::QueryResult> results;
    double total_ms = 0, startup_ttfq_ms = 0;
    const auto query_start = trade::Clock::now();
    for (const auto &[s, t] : queries) {
      auto result =
          gpu ? gpu->solve(s, t) : cpu->solve(s, t, engine == "cpu-pr");
      if (results.empty())
        startup_ttfq_ms = trade::elapsed_ms(process_start);
      total_ms += result.ms;
      if (checker) {
        const auto validation_start = trade::Clock::now();
        const auto reference = checker->solve(s, t);
        if (result.flow != reference.flow)
          throw std::runtime_error("CUDA/CPU maxflow mismatch for " +
                                   std::to_string(s) + " -> " +
                                   std::to_string(t));
        gpu->verify(s, t, result.flow);
        validation_ms += trade::elapsed_ms(validation_start);
      }
      results.push_back(result);
    }
    const double query_wall_ms = trade::elapsed_ms(query_start);
    std::ostringstream output;
    for (size_t i = 0; i < queries.size(); ++i)
      output << queries[i].first << ' ' << queries[i].second << ' '
             << results[i].flow << '\n';
    const auto output_start = trade::Clock::now();
    write_file(args.at("--output"), output.str());
    const double output_ms = trade::elapsed_ms(output_start);
    std::ostringstream log;
    log << std::fixed << std::setprecision(6)
        << "{\n  \"schema_version\": 1,\n  \"engine\": " << json_string(engine)
        << ",\n  \"device\": "
        << json_string(gpu ? gpu->device_name() : "CPU single thread")
        << ",\n  \"nodes\": " << input.n << ",\n  \"edges\": " << input.m
        << ",\n  \"query_count\": " << queries.size()
        << ",\n  \"input_read_ms\": " << input_ms
        << ",\n  \"preprocess_ms\": " << preprocess_ms
        << ",\n  \"first_query_ms\": " << results.front().ms
        << ",\n  \"ttfq_ms\": " << preprocess_ms + results.front().ms
        << ",\n  \"startup_ttfq_ms\": " << startup_ttfq_ms
        << ",\n  \"total_query_ms\": " << total_ms
        << ",\n  \"tpq_ms\": " << total_ms / queries.size()
        << ",\n  \"query_phase_wall_ms\": " << query_wall_ms
        << ",\n  \"validation_ms\": " << validation_ms
        << ",\n  \"output_write_ms\": " << output_ms
        << ",\n  \"process_until_log_ms\": " << trade::elapsed_ms(process_start)
        << ",\n  \"device_buffer_bytes\": " << (gpu ? gpu->device_bytes() : 0)
        << ",\n  \"checked\": " << (checker ? "true" : "false")
        << ",\n  \"schedule\": " << json_string(schedule)
        << ",\n  \"cuda_graphs\": " << (graphs ? "true" : "false")
        << ",\n  \"batch_rounds\": " << options.batch_rounds
        << ",\n  \"global_interval\": " << options.global_interval
        << ",\n  \"timing_scope\": {"
           "\"preprocess\": \"residual CSR construction, solver allocation, "
           "CUDA context, one topology/capacity upload, graph capture/upload; "
           "excludes file input\","
           "\"query\": \"wall clock: residual reset, GPU kernels, global BFS, "
           "host control, scalar/active status copies and result "
           "synchronization; excludes validation/output\","
           "\"ttfq\": \"preprocess + first_query\","
           "\"total_query\": \"sum of per-query wall times, excluding optional "
           "checks\"},\n  \"queries\": [\n";
    for (size_t i = 0; i < queries.size(); ++i) {
      const auto &r = results[i];
      log << "    {\"source\": " << queries[i].first
          << ", \"target\": " << queries[i].second
          << ", \"maxflow\": " << r.flow << ", \"ms\": " << r.ms
          << ", \"rounds\": " << r.rounds
          << ", \"global_relabels\": " << r.global_relabels << '}'
          << (i + 1 == queries.size() ? "\n" : ",\n");
    }
    log << "  ]\n}\n";
    write_file(args.at("--log"), log.str());
    std::cout << engine << ": N=" << input.n << " M=" << input.m
              << " queries=" << queries.size() << " total_ms=" << total_ms
              << " TPQ_ms=" << total_ms / queries.size()
              << " checked=" << (checker ? "yes" : "no") << '\n';
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "error: " << e.what() << '\n';
    return 1;
  }
}
