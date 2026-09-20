#pragma once
#include "numeric.hpp"
#include <stdexcept>
#include <string>
#include <vector>
namespace lp {
struct Config {
    int format = 0, block = 32, mode = 0, out = 1, round = 0;
    uint64_t seed = 42;
    std::string target = "RTX3090";
};
struct Tensor {
    uint64_t rows = 0, cols = 0;
    int dtype = 0;
    std::vector<uint8_t> data;
    uint64_t size() const {
        return rows * cols;
    }
};
struct Packed {
    uint64_t rows = 0, cols = 0;
    int input_dtype = 0;
    Config c;
    float global = 1;
    std::vector<uint8_t> scales, data;
    uint64_t size() const {
        return rows * cols;
    }
};
struct Timing {
    double scale = 0, encode = 0, quant = 0, dequant = 0, h2d = 0, d2h = 0, process = 0;
};
uint64_t checked_size(uint64_t r, uint64_t c);
void validate(const Config &c);
Config read_config(const std::string &path);
Tensor generate(uint64_t r, uint64_t c, int dtype, const std::string &distribution, uint64_t seed);
Packed quant_cpu(const Tensor &, const Config &);
Tensor dequant_cpu(const Packed &);
Packed quant_gpu(const Tensor &, const Config &, const std::string &variant, Timing *t = nullptr,
                 Tensor *decoded = nullptr);
Tensor dequant_gpu(const Packed &, Timing *t = nullptr);
void save_tensor(const std::string &, const Tensor &);
Tensor load_tensor(const std::string &);
void save_packed(const std::string &, const Packed &);
Packed load_packed(const std::string &);
void validate_packed(const Packed &);
void equal(const Packed &, const Packed &);
void equal(const Tensor &, const Tensor &);
struct Error {
    double max = 0, mae = 0, mse = 0;
};
Error error(const Tensor &, const Tensor &);
} // namespace lp
