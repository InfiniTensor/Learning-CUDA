#include "project.hpp"
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
using namespace lp;
void require(bool b, const char *s) {
    if (!b)
        throw std::runtime_error(s);
}
void rejects(const std::function<void()> &f) {
    bool failed = false;
    try {
        f();
    } catch (const std::exception &) {
        failed = true;
    }
    require(failed, "expected rejection");
}
int main(int argc, char **argv) {
    try {
        bool gpu = argc > 1 && std::string(argv[1]) == "gpu";
        std::filesystem::create_directories("test-artifacts");
        int cases = 0;
        for (int f = 0; f < 3; ++f)
            for (int dtype = 0; dtype < 2; ++dtype)
                for (int out = 0; out < 3; ++out)
                    for (int mode = 0; mode < 2; ++mode)
                        for (int round = 0; round < 2; ++round) {
                            Config c;
                            c.format = f;
                            c.block = f == 2 ? 16 : 32;
                            c.mode = mode;
                            c.out = out;
                            c.round = round;
                            for (auto shape : std::vector<std::pair<int, int>>{{1, 1},
                                                                               {1, 15},
                                                                               {1, 16},
                                                                               {1, 17},
                                                                               {1, 31},
                                                                               {1, 32},
                                                                               {1, 33},
                                                                               {3, 17}})
                                for (const std::string dist : {"uniform", "normal", "outliers"}) {
                                    auto x = generate(shape.first, shape.second, dtype, dist, 42);
                                    auto p = quant_cpu(x, c);
                                    auto y = dequant_cpu(p);
                                    save_tensor("test-artifacts/x.tensor", x);
                                    equal(x, load_tensor("test-artifacts/x.tensor"));
                                    save_packed("test-artifacts/p.lp", p);
                                    auto loaded = load_packed("test-artifacts/p.lp");
                                    equal(p, loaded);
                                    equal(y, dequant_cpu(loaded));
                                    if (gpu) {
                                        Tensor gy;
                                        auto gp = quant_gpu(x, c, "baseline", nullptr, &gy);
                                        equal(p, gp);
                                        equal(y, gy);
                                        equal(y, dequant_gpu(loaded));
                                        Tensor oy;
                                        equal(p, quant_gpu(x, c, "optimized", nullptr, &oy));
                                        equal(y, oy);
                                    }
                                    ++cases;
                                }
                            for (int pattern = 0; pattern < 4; ++pattern) {
                                Tensor x{3, 17, dtype, {}};
                                x.data.resize(51 * (dtype ? 2 : 4));
                                for (int i = 0; i < 51; ++i) {
                                    double v = pattern == 0   ? (i % 2 ? -0.0 : 0.0)
                                               : pattern == 1 ? 3.0
                                               : pattern == 2 ? (i % 2 ? -0.25 : 0.25)
                                                              : (i == 0 ? 60000. : 1e-7);
                                    store_value(x.data.data(), i, dtype, output_bits(v, dtype));
                                }
                                auto p = quant_cpu(x, c);
                                if (gpu)
                                    equal(p, quant_gpu(x, c, "baseline"));
                            }
                        }
        for (int f = 0; f < 3; ++f)
            for (int mode = 0; mode < 2; ++mode) {
                Config e;
                e.format = f;
                e.block = f == 2 ? 16 : 32;
                e.out = 0;
                e.mode = mode;
                Tensor t{1, 7, 0, {}};
                t.data.resize(28);
                float values[] = {0.0f, -0.0f, FLT_MAX, -FLT_MAX, 0x1p-149f, -0x1p-149f, 0x1p-126f};
                for (int i = 0; i < 7; ++i)
                    store_value(t.data.data(), i, 0, bits(values[i]));
                auto q = quant_cpu(t, e);
                if (gpu) {
                    Tensor y;
                    equal(q, quant_gpu(t, e, "optimized", nullptr, &y));
                    equal(dequant_cpu(q), y);
                }
            }
        // 手算字节布局的独立测试，不通过 quant_cpu 生成期待结果。
        Packed manual;
        manual.rows = 1;
        manual.cols = 3;
        manual.c.format = 2;
        manual.c.block = 16;
        manual.c.out = 0;
        manual.global = 1;
        manual.scales = {56};
        manual.data = {0x32, 0x07};
        auto manual_y = dequant_cpu(manual);
        require(input_value(manual_y.data.data(), 0, 0) == 1 &&
                    input_value(manual_y.data.data(), 1, 0) == 1.5 &&
                    input_value(manual_y.data.data(), 2, 0) == 6,
                "manual packed layout");
        if (gpu)
            equal(manual_y, dequant_gpu(manual));
        Config c;
        c.format = 2;
        c.block = 16;
        auto x = generate(3, 17, 0, "normal", 42);
        auto p = quant_cpu(x, c);
        require(p.data.size() == 26, "packed size");
        save_packed("test-artifacts/good.lp", p);
        std::ifstream in("test-artifacts/good.lp", std::ios::binary);
        std::vector<char> bytes((std::istreambuf_iterator<char>(in)), {});
        in.close();
        for (size_t len = 0; len < bytes.size(); ++len) {
            std::ofstream o("test-artifacts/bad.lp", std::ios::binary);
            o.write(bytes.data(), len);
            o.close();
            rejects([] { load_packed("test-artifacts/bad.lp"); });
        }
        for (int offset :
             {0, 8, 12, 16, 24, 32, 36, 40, 44, 48, 52, 64, 68, 72, 76, 80, 88, 96, 104, 112}) {
            auto b = bytes;
            for (int j = 0; j < (offset == 112 ? 1 : 4); ++j)
                b[offset + j] = char(255);
            std::ofstream o("test-artifacts/bad.lp", std::ios::binary);
            o.write(b.data(), b.size());
            o.close();
            rejects([] { load_packed("test-artifacts/bad.lp"); });
        }
        for (auto config : {"format=nvfp4\nfp8_type=e4m3", "format=bad", "block_size=17",
                            "rounding=bad", "seed=-1", "seed=42x", "output_type=bad",
                            "scale_mode=bad", "unknown=1", "seed=1\nseed=2"}) {
            std::ofstream o("test-artifacts/bad.toml");
            o << config;
            o.close();
            rejects([] { read_config("test-artifacts/bad.toml"); });
        }
        rejects([] { checked_size(0, 1); });
        rejects([] { checked_size(INT64_MAX, INT64_MAX); });
        for (float v : {INFINITY, -INFINITY, NAN}) {
            store_value(x.data.data(), 0, 0, bits(v));
            rejects([&] { quant_cpu(x, c); });
            if (gpu)
                rejects([&] { quant_gpu(x, c, "baseline"); });
        }
        std::cout << "PASS " << cases
                  << " full configuration/distribution/shape cases; scales, files, corruption, "
                     "invalid config; backend="
                  << (gpu ? "RTX GPU" : "CPU") << '\n';
    } catch (std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
