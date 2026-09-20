#include "project.hpp"
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
using namespace lp;
using Clock = std::chrono::steady_clock;
static double ms(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}
int main(int argc, char **argv) {
    try {
        if (argc < 2)
            throw std::runtime_error(
                "usage: lp generate|quantize|dequantize|verify|benchmark --key value");
        std::string cmd = argv[1];
        std::map<std::string, std::string> args;
        for (int i = 2; i < argc; i += 2) {
            if (i + 1 == argc || std::string(argv[i]).rfind("--", 0) != 0 ||
                !args.emplace(std::string(argv[i]).substr(2), argv[i + 1]).second)
                throw std::runtime_error("invalid arguments");
        }
        auto get = [&](std::string k, std::string def = "") {
            auto it = args.find(k);
            if (it == args.end())
                return def;
            auto v = it->second;
            args.erase(it);
            return v;
        };
        auto number = [&](std::string k, std::string def) {
            std::string s = get(k, def);
            if (s.empty() || s.find_first_not_of("0123456789") != std::string::npos)
                throw std::runtime_error("invalid integer: " + k);
            return std::stoull(s);
        };
        auto finish = [&]() {
            if (!args.empty())
                throw std::runtime_error("inapplicable/unknown argument: " + args.begin()->first);
        };
        if (cmd == "generate") {
            auto path = get("output");
            auto r = number("rows", "128"), c = number("cols", "128"), seed = number("seed", "42");
            auto dt = get("dtype", "fp32"), dist = get("distribution", "normal");
            finish();
            if (dt != "fp32" && dt != "fp16")
                throw std::runtime_error("invalid dtype");
            save_tensor(path, generate(r, c, dt == "fp16", dist, seed));
            std::cout << "generated " << path << '\n';
        } else if (cmd == "quantize") {
            auto in = get("input"), out = get("output"), cfg = get("config"),
                 backend = get("backend", "cuda"), variant = get("kernel_variant", "baseline");
            finish();
            if (backend != "cpu" && backend != "cuda")
                throw std::runtime_error("invalid backend");
            if (variant != "baseline" && variant != "optimized")
                throw std::runtime_error("invalid kernel variant");
            auto c = read_config(cfg);
            auto x = load_tensor(in);
            auto p = backend == "cpu" ? quant_cpu(x, c) : quant_gpu(x, c, variant);
            save_packed(out, p);
            std::cout << "quantized " << out << '\n';
        } else if (cmd == "dequantize") {
            auto in = get("input"), out = get("output"), backend = get("backend", "cuda");
            finish();
            if (backend != "cpu" && backend != "cuda")
                throw std::runtime_error("invalid backend");
            auto p = load_packed(in);
            save_tensor(out, backend == "cpu" ? dequant_cpu(p) : dequant_gpu(p));
            std::cout << "dequantized " << out << '\n';
        } else if (cmd == "verify") {
            auto in = get("input"), packed = get("packed"), backend = get("backend", "cuda");
            finish();
            if (backend != "cpu" && backend != "cuda")
                throw std::runtime_error("invalid backend");
            auto x = load_tensor(in);
            auto p = load_packed(packed);
            equal(quant_cpu(x, p.c), p);
            auto y = dequant_cpu(p);
            if (backend == "cuda")
                equal(y, dequant_gpu(p));
            auto e = error(x, y);
            std::cout << "PASS max=" << e.max << " mae=" << e.mae << " mse=" << e.mse << '\n';
        } else if (cmd == "benchmark") {
            auto in = get("input"), cfg = get("config"), csv = get("csv"),
                 variant = get("kernel_variant", "baseline"), work = get("workdir", ".");
            auto repeat = number("repeat", "10"), warmup = number("warmup", "3");
            finish();
            if (repeat == 0 || repeat > 10000 || warmup > 10000)
                throw std::runtime_error("invalid repeat/warmup");
            auto x = load_tensor(in);
            auto c = read_config(cfg);
            std::filesystem::create_directories(work);
            auto cp = quant_cpu(x, c);
            auto cy = dequant_cpu(cp);
            for (uint64_t i = 0; i < warmup; ++i)
                quant_gpu(x, c, variant);
            std::ofstream out(csv);
            if (!out)
                throw std::runtime_error("CSV open failed");
            out << "iteration,n,input_dtype,format,mode,round,out,variant,scale_ms,encode_ms,quant_"
                   "ms,dequant_ms,h2d_ms,d2h_ms,process_ms,cpu_quant_ms,cpu_dequant_ms,e2e_ms,cpu_"
                   "e2e_ms,max_abs,mae,mse,payload_ratio,file_ratio,dequant_GBs,kernel_speedup,"
                   "process_speedup\n";
            out << std::setprecision(12);
            for (uint64_t i = 0; i < repeat; ++i) {
                auto a = Clock::now();
                auto p0 = quant_cpu(x, c);
                auto b = Clock::now();
                auto y0 = dequant_cpu(p0);
                auto d = Clock::now();
                Timing t;
                Tensor y;
                auto process_start = Clock::now();
                auto p = quant_gpu(x, c, variant, &t, &y);
                // 包含主机/设备分配、传输、同步与释放，与 CPU API 返回边界一致。
                t.process = ms(process_start, Clock::now());
                equal(p0, p);
                equal(y0, y);
                auto e = error(x, y);
                auto start = Clock::now();
                auto diskx = load_tensor(in);
                auto diskp = quant_gpu(diskx, c, variant);
                auto qp = work + "/bench.lp", yp = work + "/bench.tensor";
                save_packed(qp, diskp);
                auto reloaded = load_packed(qp);
                auto disky = dequant_gpu(reloaded);
                save_tensor(yp, disky);
                double e2e = ms(start, Clock::now());
                start = Clock::now();
                auto cx = load_tensor(in);
                save_packed(work + "/cpu.lp", quant_cpu(cx, c));
                save_tensor(work + "/cpu.tensor", dequant_cpu(load_packed(work + "/cpu.lp")));
                double ce2e = ms(start, Clock::now());
                double stored = p.data.size() + p.scales.size() + (c.format == 2 ? 4 : 0);
                double ratio = x.data.size() / stored;
                double fr = double(48 + x.data.size()) / (112 + p.data.size() + p.scales.size());
                double cq = ms(a, b), cd = ms(b, d);
                double traffic =
                    p.data.size() + p.scales.size() + y.data.size() + (c.format == 2 ? 4 : 0);
                out << i << ',' << x.size() << ',' << x.dtype << ',' << c.format << ',' << c.mode
                    << ',' << c.round << ',' << c.out << ',' << variant << ',' << t.scale << ','
                    << t.encode << ',' << t.quant << ',' << t.dequant << ',' << t.h2d << ','
                    << t.d2h << ',' << t.process << ',' << cq << ',' << cd << ',' << e2e << ','
                    << ce2e << ',' << e.max << ',' << e.mae << ',' << e.mse << ',' << ratio << ','
                    << fr << ',' << traffic / (t.dequant * 1e6) << ','
                    << (cq + cd) / (t.quant + t.dequant) << ',' << (cq + cd) / t.process << '\n';
            }
            out.close();
            if (!out)
                throw std::runtime_error("CSV write failed");
            std::cout << "benchmark " << csv << '\n';
        } else
            throw std::runtime_error("unknown command");
    } catch (std::exception &e) {
        std::cerr << "error: " << e.what() << '\n';
        return 1;
    }
}
