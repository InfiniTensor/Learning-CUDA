#include "lp.hpp"
#include <filesystem>
#include <iostream>
#include <map>
#include <set>
#include <stdexcept>

namespace {
void help() {
    std::cout << "Portable MXFP8 / NVFP4 software simulation\n"
                 "  low_precision run --input tensor.bin --config params.toml\n"
                 "    --weights weights.bin --output restored.bin --log metrics.json\n"
                 "    [--backend cuda|cpu] [--warmup 5] [--iterations 50] [--cpu-repeats 3]\n"
                 "  low_precision decode --weights weights.bin --output restored.bin\n"
                 "    [--output-type fp32|fp16|bf16] [--backend cuda|cpu]\n";
}
int count(const std::map<std::string, std::string> &args, const std::string &key, int fallback,
          int minimum) {
    auto it = args.find(key);
    if (it == args.end())
        return fallback;
    size_t used;
    const auto n = std::stoll(it->second, &used);
    if (used != it->second.size() || n < minimum || n > 100000)
        throw std::runtime_error("Invalid count: " + key);
    return int(n);
}
void ensure_distinct(const std::vector<std::string> &paths) {
    std::set<std::filesystem::path> seen;
    for (const auto &p : paths) {
        auto path = std::filesystem::weakly_canonical(std::filesystem::absolute(p));
        if (!seen.insert(path).second)
            throw std::runtime_error("Input/output file paths must be distinct");
    }
}
} // namespace
int main(int argc, char **argv) {
    try {
        const double start = lp::now_ms();
        if (argc < 2 || std::string(argv[1]) == "--help") {
            help();
            return argc < 2 ? 1 : 0;
        }
        const std::string command = argv[1];
        if (command != "run" && command != "decode")
            throw std::runtime_error("Unknown command");
        const std::set<std::string> allowed =
            command == "run" ? std::set<std::string>{"--input",  "--config",     "--weights",
                                                     "--output", "--log",        "--backend",
                                                     "--warmup", "--iterations", "--cpu-repeats"}
                             : std::set<std::string>{"--weights", "--output", "--output-type",
                                                     "--backend", "--warmup", "--iterations"};
        std::map<std::string, std::string> args;
        for (int i = 2; i < argc; i += 2) {
            const std::string key = argv[i];
            if (i + 1 == argc || !allowed.count(key) || !args.emplace(key, argv[i + 1]).second)
                throw std::runtime_error("Invalid, duplicate, or incomplete option: " + key);
        }
        auto required = [&](const std::string &key) {
            auto it = args.find(key);
            if (it == args.end())
                throw std::runtime_error("Missing " + key);
            return it->second;
        };
        const std::string backend = args.count("--backend") ? args.at("--backend") : "cuda";
        if (backend != "cuda" && backend != "cpu")
            throw std::runtime_error("Invalid backend");
#ifndef LP_ENABLE_CUDA
        if (backend == "cuda")
            throw std::runtime_error("CUDA was disabled at build time; select --backend cpu");
#endif
        const int warmup = count(args, "--warmup", 5, 0),
                  repeats = count(args, "--iterations", 50, 1);
        const auto weights_path = required("--weights"), output_path = required("--output");
        if (command == "decode") {
            ensure_distinct({weights_path, output_path});
            auto w = lp::read_weights(weights_path);
            auto type = args.count("--output-type") ? lp::parse_dtype(args.at("--output-type"))
                                                    : w.config.output_type;
            auto out = lp::make_output(w, type);
            lp::Timings timing;
#ifdef LP_ENABLE_CUDA
            if (backend == "cuda")
                timing = lp::decode_cuda(w, out, warmup, repeats);
            else
#endif
            {
                double d0 = lp::now_ms();
                lp::dequantize_cpu(w, out);
                timing.dequant_ms = lp::elapsed_ms(d0);
            }
            lp::write_tensor(output_path, out);
            std::cout << "Decoded " << w.rows << 'x' << w.cols << " to " << lp::dtype_name(type)
                      << "; " << timing.dequant_ms << " ms\n";
            return 0;
        }
        const auto input_path = required("--input"), config_path = required("--config"),
                   log_path = required("--log");
        ensure_distinct({input_path, config_path, weights_path, output_path, log_path});
        double io0 = lp::now_ms();
        auto config = lp::read_config(config_path);
        auto input = lp::read_tensor(input_path);
        double io = lp::elapsed_ms(io0);
        auto w = lp::make_weights(input, config);
        auto out = lp::make_output(w, config.output_type);
        auto cpu = lp::run_cpu(input, w, out, count(args, "--cpu-repeats", 3, 1));
        lp::Timings timing = cpu;
#ifdef LP_ENABLE_CUDA
        if (backend == "cuda") {
            // Byte-exact parity includes scale bytes, padded nibbles, and output conversion.
            auto gpu_w = lp::make_weights(input, config);
            auto gpu_out = lp::make_output(gpu_w, config.output_type);
            timing = lp::run_cuda(input, gpu_w, gpu_out, warmup, repeats);
            if (w.data != gpu_w.data || w.scales != gpu_w.scales ||
                lp::float_bits(w.global_scale) != lp::float_bits(gpu_w.global_scale) ||
                out.data != gpu_out.data)
                throw std::runtime_error("CPU/CUDA byte-exact validation failed");
            w = std::move(gpu_w);
            out = std::move(gpu_out);
        }
#endif
        auto error = lp::measure_error(input, out);
        io0 = lp::now_ms();
        lp::write_weights(weights_path, w);
        lp::write_tensor(output_path, out);
        io += lp::elapsed_ms(io0);
        lp::write_log(log_path, input, w, error, timing, cpu, io, lp::elapsed_ms(start), warmup,
                      repeats, backend);
        std::cout << "OK " << input.rows << 'x' << input.cols << "; " << timing.device
                  << "; MAE=" << error.mae << "; MSE=" << error.mse << "; quant=" << timing.quant_ms
                  << " ms; dequant=" << timing.dequant_ms << " ms\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }
}
