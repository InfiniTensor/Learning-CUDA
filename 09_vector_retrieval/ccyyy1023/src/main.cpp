#include "gvs/io.hpp"
#include "gvs/search.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

class Args {
public:
    Args(int argc, char** argv, int start = 2) {
        for (int i = start; i < argc; ++i) {
            const std::string key = argv[i];
            if (key.rfind("--", 0) != 0 || i + 1 >= argc)
                throw std::invalid_argument("expected --key value pairs");
            values_[key.substr(2)] = argv[++i];
        }
    }
    std::string get(const std::string& key, const std::string& fallback = "") const {
        auto it = values_.find(key);
        return it == values_.end() ? fallback : it->second;
    }
    std::string require(const std::string& key) const {
        const auto value = get(key);
        if (value.empty()) throw std::invalid_argument("missing --" + key);
        return value;
    }
    std::size_t number(const std::string& key, std::size_t fallback) const {
        const auto value = get(key);
        return value.empty() ? fallback : static_cast<std::size_t>(std::stoull(value));
    }
private:
    std::map<std::string, std::string> values_;
};

gvs::Metric parse_metric(const std::string& value) {
    if (value == "l2") return gvs::Metric::L2;
    if (value == "inner_product") return gvs::Metric::InnerProduct;
    throw std::invalid_argument("metric must be l2 or inner_product");
}

double percentile(std::vector<double> values, double p) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    const std::size_t index = static_cast<std::size_t>(
        std::ceil(p * static_cast<double>(values.size()))) - 1;
    return values[std::min(index, values.size() - 1)];
}

double mean_rank_score_error(const gvs::SearchResult& truth,
                             const gvs::SearchResult& candidate) {
    if (truth.queries != candidate.queries || truth.k != candidate.k)
        throw std::invalid_argument("quality result shapes differ");
    double sum = 0.0;
    for (std::size_t i = 0; i < truth.scores.size(); ++i)
        sum += std::abs(static_cast<double>(truth.scores[i]) - candidate.scores[i]);
    return truth.scores.empty() ? 0.0 : sum / truth.scores.size();
}

std::string json_escape(const std::string& value) {
    std::string out;
    for (char c : value) {
        if (c == '"' || c == '\\') out.push_back('\\');
        out.push_back(c);
    }
    return out;
}

void command_generate(const Args& args) {
    const std::size_t n = args.number("n", 10000);
    const std::size_t nq = args.number("nq", 100);
    const std::size_t dim = args.number("dim", 128);
    const auto metric = parse_metric(args.get("metric", "l2"));
    const std::uint64_t seed = args.number("seed", 42);
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    gvs::VectorFile database{n, dim, metric, std::vector<float>(n * dim)};
    gvs::VectorFile queries{nq, dim, metric, std::vector<float>(nq * dim)};
    for (float& value : database.values) value = normal(rng);
    for (float& value : queries.values) value = normal(rng);
    gvs::write_vector_file(args.require("db"), database, true);
    gvs::write_vector_file(args.require("queries"), queries, false);
    std::cout << "generated n=" << n << " nq=" << nq << " dim=" << dim << '\n';
}

void command_build(const Args& args) {
    const auto database = gvs::read_vector_file(args.require("db"), true);
    const auto start = Clock::now();
    gvs::IvfFlatIndex index(database.dim, args.number("nlist", 256), database.metric);
    index.train(database.values, database.count, args.number("iterations", 12),
                args.number("seed", 42));
    index.add(database.values, database.count);
    index.save(args.require("index"));
    const double ms = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    std::cout << "index_build_ms=" << ms << '\n';
}

void command_search(const Args& args) {
    const auto database = gvs::read_vector_file(args.require("db"), true);
    const auto queries = gvs::read_vector_file(args.require("queries"), false);
    if (database.dim != queries.dim) throw std::runtime_error("database/query dim mismatch");
    const auto parameters = gvs::read_parameters(args.require("params"));
    gvs::SearchConfig config{parameters.top_k, parameters.batch_size};
    gvs::GpuStats stats;
    gvs::SearchResult result;
    if (parameters.search_mode == "exact") {
        result = gvs::gpu_exact_search(database.values, queries.values,
                                       database.count, queries.count, database.dim,
                                       database.metric, config, &stats);
    } else {
        auto index = gvs::IvfFlatIndex::load(args.require("index"));
        result = index.search(queries.values, queries.count, parameters.nprobe,
                              config, &stats);
    }
    gvs::write_search_result(args.require("output"), result);
    const double qps = stats.elapsed_ms == 0.0 ? 0.0 : queries.count * 1000.0 / stats.elapsed_ms;
    std::cout << "mode=" << parameters.search_mode << " query_ms=" << stats.elapsed_ms
              << " qps=" << qps << " workspace_mib="
              << stats.peak_workspace_bytes / (1024.0 * 1024.0) << '\n';
}

void command_benchmark(const Args& args) {
    const auto database = gvs::read_vector_file(args.require("db"), true);
    const auto queries = gvs::read_vector_file(args.require("queries"), false);
    if (database.dim != queries.dim) throw std::runtime_error("database/query dim mismatch");
    const std::size_t k = args.number("k", 10);
    const std::size_t batch = args.number("batch", 128);
    const std::size_t nprobe = args.number("nprobe", 16);
    const std::size_t cpu_limit = std::min(args.number("cpu-query-limit", 20), queries.count);
    const std::size_t latency_limit = std::min(args.number("latency-query-limit", 100), queries.count);
    const gvs::SearchConfig config{k, batch};

    gvs::GpuStats exact_stats;
    const auto exact = gvs::gpu_exact_search(database.values, queries.values,
                                             database.count, queries.count, database.dim,
                                             database.metric, config, &exact_stats);

    std::vector<float> cpu_queries(queries.values.begin(),
                                   queries.values.begin() + cpu_limit * database.dim);
    const auto cpu_begin = Clock::now();
    const auto cpu = gvs::cpu_exact_search(database.values, cpu_queries,
                                           database.count, cpu_limit, database.dim,
                                           database.metric, k);
    const double cpu_ms = std::chrono::duration<double, std::milli>(
        Clock::now() - cpu_begin).count();
    gvs::SearchResult exact_prefix{cpu_limit, exact.k,
        std::vector<int>(exact.ids.begin(), exact.ids.begin() + cpu_limit * exact.k),
        std::vector<float>(exact.scores.begin(), exact.scores.begin() + cpu_limit * exact.k)};
    const double exact_recall = gvs::recall_at_k(cpu, exact_prefix);

    double build_ms = 0.0;
    const std::string index_path = args.require("index");
    if (!std::filesystem::exists(index_path) || args.get("rebuild", "false") == "true") {
        const auto build_begin = Clock::now();
        gvs::IvfFlatIndex built(database.dim, args.number("nlist", 256), database.metric);
        built.train(database.values, database.count, args.number("iterations", 12),
                    args.number("seed", 42));
        built.add(database.values, database.count);
        built.save(index_path);
        build_ms = std::chrono::duration<double, std::milli>(
            Clock::now() - build_begin).count();
    }
    auto index = gvs::IvfFlatIndex::load(index_path);
    gvs::GpuStats ivf_stats;
    const auto approximate = index.search(queries.values, queries.count, nprobe,
                                          config, &ivf_stats);
    const double ivf_recall = gvs::recall_at_k(exact, approximate);
    const double score_error = mean_rank_score_error(exact, approximate);

    std::vector<double> exact_latencies;
    exact_latencies.reserve(latency_limit);
    for (std::size_t q = 0; q < latency_limit; ++q) {
        std::vector<float> one(queries.values.begin() + q * database.dim,
                               queries.values.begin() + (q + 1) * database.dim);
        gvs::GpuStats one_stats;
        gvs::gpu_exact_search(database.values, one, database.count, 1, database.dim,
                              database.metric, gvs::SearchConfig{k, 1}, &one_stats);
        exact_latencies.push_back(one_stats.elapsed_ms);
    }

    const double exact_qps = exact_stats.elapsed_ms == 0.0 ? 0.0
        : queries.count * 1000.0 / exact_stats.elapsed_ms;
    const double ivf_qps = ivf_stats.elapsed_ms == 0.0 ? 0.0
        : queries.count * 1000.0 / ivf_stats.elapsed_ms;
    const double cpu_per_query = cpu_limit == 0 ? 0.0 : cpu_ms / cpu_limit;
    const double gpu_per_query = queries.count == 0 ? 0.0 : exact_stats.elapsed_ms / queries.count;
    const double speedup = gpu_per_query == 0.0 ? 0.0 : cpu_per_query / gpu_per_query;

    std::ofstream log(args.require("log"));
    if (!log) throw std::runtime_error("cannot open benchmark log");
    log << std::fixed << std::setprecision(6)
        << "{\n"
        << "  \"status\": \"measured\",\n"
        << "  \"device\": \"" << json_escape(exact_stats.device_name) << "\",\n"
        << "  \"dataset\": {\"n\": " << database.count << ", \"nq\": "
        << queries.count << ", \"dim\": " << database.dim << ", \"k\": " << k << "},\n"
        << "  \"exact\": {\"query_ms\": " << exact_stats.elapsed_ms
        << ", \"qps\": " << exact_qps << ", \"p50_ms\": "
        << percentile(exact_latencies, 0.50) << ", \"p99_ms\": "
        << percentile(exact_latencies, 0.99) << ", \"workspace_mib\": "
        << exact_stats.peak_workspace_bytes / (1024.0 * 1024.0)
        << ", \"cpu_gpu_recall_at_k\": " << exact_recall
        << ", \"speedup_vs_cpu\": " << speedup << "},\n"
        << "  \"cpu\": {\"queries_measured\": " << cpu_limit
        << ", \"query_ms\": " << cpu_ms << "},\n"
        << "  \"ivf_flat\": {\"index_build_ms\": " << build_ms
        << ", \"query_ms\": " << ivf_stats.elapsed_ms << ", \"qps\": "
        << ivf_qps << ", \"nlist\": " << index.nlist() << ", \"nprobe\": "
        << nprobe << ", \"recall_at_k\": " << ivf_recall
        << ", \"mean_rank_score_error\": " << score_error
        << ", \"workspace_mib\": "
        << ivf_stats.peak_workspace_bytes / (1024.0 * 1024.0) << "}\n"
        << "}\n";
    std::cout << "benchmark written to " << args.require("log") << '\n';
}

void usage() {
    std::cout
        << "GPU Vector Search\n"
        << "  generate --db DB --queries Q --n N --nq NQ --dim D --metric l2|inner_product\n"
        << "  build --db DB --index INDEX --nlist C [--iterations I]\n"
        << "  search --db DB --queries Q --params CONFIG --index INDEX --output OUT\n"
        << "  benchmark --db DB --queries Q --index INDEX --log JSON [--k K --batch B]\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 2) { usage(); return 1; }
        const std::string command = argv[1];
        const Args args(argc, argv);
        if (command == "generate") command_generate(args);
        else if (command == "build") command_build(args);
        else if (command == "search") command_search(args);
        else if (command == "benchmark") command_benchmark(args);
        else { usage(); return 1; }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 2;
    }
}

