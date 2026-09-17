#include "lp.hpp"
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>

namespace lp {
namespace {
std::string trim(std::string s) {
    const auto first = s.find_first_not_of(" \t\r\n");
    if (first == std::string::npos)
        return "";
    return s.substr(first, s.find_last_not_of(" \t\r\n") - first + 1);
}
uint64_t integer(const std::string &s) {
    if (s.empty() || s.find_first_not_of("0123456789") != std::string::npos)
        throw std::runtime_error("Expected unsigned integer: " + s);
    size_t pos = 0;
    auto n = std::stoull(s, &pos);
    if (pos != s.size())
        throw std::runtime_error("Invalid integer");
    return n;
}
void put(std::ostream &f, uint64_t x, int bytes) {
    for (int i = 0; i < bytes; ++i)
        f.put(static_cast<char>(x >> (i * 8)));
}
uint64_t get(std::istream &f, int bytes) {
    uint64_t x = 0;
    for (int i = 0; i < bytes; ++i) {
        int c = f.get();
        if (c == EOF)
            throw std::runtime_error("Truncated binary header");
        x |= uint64_t(static_cast<uint8_t>(c)) << (8 * i);
    }
    return x;
}
void magic(std::istream &f, const char *expected) {
    char m[8];
    f.read(m, 8);
    if (!f || std::memcmp(m, expected, 8))
        throw std::runtime_error("Invalid binary magic/version");
}
uint64_t file_size(std::ifstream &f) {
    if (!f)
        throw std::runtime_error("Cannot open input file");
    f.seekg(0, std::ios::end);
    auto n = f.tellg();
    f.seekg(0);
    if (n < 0)
        throw std::runtime_error("Cannot determine file size");
    return uint64_t(n);
}
void check_little_endian() {
    uint32_t x = 1;
    if (*reinterpret_cast<uint8_t *>(&x) != 1)
        throw std::runtime_error("This build requires a little-endian host");
}
void check_written(const std::ofstream &f) {
    if (!f)
        throw std::runtime_error("Output write failed");
}
std::string json_string(const std::string &s) {
    std::ostringstream o;
    o << '"';
    for (unsigned char c : s) {
        if (c == '"' || c == '\\')
            o << '\\' << c;
        else if (c < 32)
            o << "\\u" << std::hex << std::setw(4) << std::setfill('0') << unsigned(c);
        else
            o << c;
    }
    o << '"';
    return o.str();
}
std::string number(double n) {
    if (!std::isfinite(n))
        return "null";
    std::ostringstream o;
    o << std::setprecision(17) << n;
    return o.str();
}
} // namespace
Config read_config(const std::string &path) {
    std::ifstream f(path);
    if (!f)
        throw std::runtime_error("Cannot open config: " + path);
    std::map<std::string, std::string> fields;
    std::string line;
    while (std::getline(f, line)) {
        bool quoted = false;
        size_t comment = line.size();
        for (size_t i = 0; i < line.size(); ++i) {
            if (line[i] == '"')
                quoted = !quoted;
            if (line[i] == '#' && !quoted) {
                comment = i;
                break;
            }
        }
        line = trim(line.substr(0, comment));
        if (line.empty())
            continue;
        auto eq = line.find('=');
        if (eq == std::string::npos)
            throw std::runtime_error("Expected key = value in config");
        auto key = trim(line.substr(0, eq)), value = trim(line.substr(eq + 1));
        if (value.size() >= 2 && value.front() == '"' && value.back() == '"')
            value = value.substr(1, value.size() - 2);
        if (!fields.emplace(key, value).second)
            throw std::runtime_error("Duplicate config key: " + key);
    }
    Config c;
    auto fmt = fields.find("format");
    if (fmt != fields.end()) {
        if (fmt->second == "nvfp4") {
            c.format = Format::NVFP4;
            c.block_size = 16;
        } else if (fmt->second != "mxfp8")
            throw std::runtime_error("Invalid format");
    }
    for (auto &[k, v] : fields) {
        if (k == "format")
            continue;
        if (k == "block_size") {
            auto n = integer(v);
            if (n > 1024)
                throw std::runtime_error("block_size too large");
            c.block_size = static_cast<uint32_t>(n);
        } else if (k == "scale_mode") {
            if (v != "tensor" && v != "block")
                throw std::runtime_error("Invalid scale_mode");
            c.tensor_scale = v == "tensor";
        } else if (k == "rounding") {
            if (v != "nearest" && v != "stochastic")
                throw std::runtime_error("Invalid rounding");
            c.stochastic = v == "stochastic";
        } else if (k == "output_type")
            c.output_type = parse_dtype(v);
        else if (k == "seed")
            c.seed = integer(v);
        else if (k == "target_gpu")
            c.target_gpu = v;
        else
            throw std::runtime_error("Unknown config key: " + k);
    }
    validate_config(c);
    return c;
}
Tensor read_tensor(const std::string &path, bool input_only) {
    check_little_endian();
    std::ifstream f(path, std::ios::binary);
    const auto length = file_size(f);
    magic(f, "LPTENS1\0");
    Tensor t;
    t.rows = get(f, 8);
    t.cols = get(f, 8);
    uint64_t dtype = get(f, 4);
    if (dtype > (input_only ? 1U : 2U) || get(f, 4) != 0)
        throw std::runtime_error("Invalid tensor dtype or flags");
    t.dtype = static_cast<DType>(dtype);
    auto size = checked_size(t.rows, t.cols, dtype_bytes(t.dtype));
    if (length < 32 || size != length - 32)
        throw std::runtime_error("Tensor payload length mismatch");
    t.data.resize(size);
    f.read(reinterpret_cast<char *>(t.data.data()), size);
    if (!f)
        throw std::runtime_error("Truncated tensor payload");
    if (input_only)
        for (uint64_t i = 0; i < t.size(); ++i)
            if (!std::isfinite(tensor_value(t, i)))
                throw std::runtime_error("Input contains NaN/Inf at element " + std::to_string(i));
    return t;
}
void write_tensor(const std::string &path, const Tensor &t) {
    check_little_endian();
    std::ofstream f(path, std::ios::binary);
    f.write("LPTENS1\0", 8);
    put(f, t.rows, 8);
    put(f, t.cols, 8);
    put(f, uint32_t(t.dtype), 4);
    put(f, 0, 4);
    f.write(reinterpret_cast<const char *>(t.data.data()), t.data.size());
    f.close();
    check_written(f);
}
void write_weights(const std::string &path, const Weights &w) {
    std::ofstream f(path, std::ios::binary);
    f.write("LPWGHT1\0", 8);
    put(f, 1, 4);
    put(f, 96, 4);
    put(f, w.rows, 8);
    put(f, w.cols, 8);
    put(f, uint32_t(w.config.format), 4);
    put(f, w.config.block_size, 4);
    put(f, w.config.tensor_scale, 4);
    put(f, uint32_t(w.config.output_type), 4);
    put(f, w.config.stochastic, 4);
    put(f, uint32_t(w.input_type), 4);
    put(f, w.config.seed, 8);
    put(f, w.scales.size(), 8);
    put(f, w.data.size(), 8);
    put(f, float_bits(w.global_scale), 4);
    put(f, 0, 4);
    put(f, 0, 8);
    f.write(reinterpret_cast<const char *>(w.data.data()), w.data.size());
    f.write(reinterpret_cast<const char *>(w.scales.data()), w.scales.size());
    f.close();
    check_written(f);
}
Weights read_weights(const std::string &path) {
    std::ifstream f(path, std::ios::binary);
    const auto length = file_size(f);
    magic(f, "LPWGHT1\0");
    if (get(f, 4) != 1 || get(f, 4) != 96)
        throw std::runtime_error("Unsupported weight header");
    Weights w;
    w.rows = get(f, 8);
    w.cols = get(f, 8);
    w.config.format = static_cast<Format>(get(f, 4));
    w.config.block_size = get(f, 4);
    auto mode = get(f, 4);
    w.config.output_type = static_cast<DType>(get(f, 4));
    auto rounding = get(f, 4), input = get(f, 4);
    if (mode > 1 || rounding > 1 || input > 1)
        throw std::runtime_error("Invalid weight metadata");
    w.config.tensor_scale = mode;
    w.config.stochastic = rounding;
    w.input_type = static_cast<DType>(input);
    w.config.seed = get(f, 8);
    auto scales = get(f, 8), data = get(f, 8);
    w.global_scale = bits_float(get(f, 4));
    if (get(f, 4) || get(f, 8))
        throw std::runtime_error("Unsupported weight flags");
    validate_config(w.config);
    checked_size(w.rows, w.cols, 4);
    const auto expected_data = checked_size(w.rows, w.row_bytes(), 1);
    const auto expected_scales =
        w.config.tensor_scale ? 1 : checked_size(w.rows, w.blocks_per_row(), 1);
    if (data != expected_data || scales != expected_scales || length < 96 || data > length - 96 ||
        scales != length - 96 - data)
        throw std::runtime_error("Weight payload length mismatch");
    if (!std::isfinite(w.global_scale) || w.global_scale <= 0 ||
        (w.config.format == Format::MXFP8 && w.global_scale != 1))
        throw std::runtime_error("Invalid global scale");
    w.data.resize(data);
    w.scales.resize(scales);
    f.read(reinterpret_cast<char *>(w.data.data()), data);
    f.read(reinterpret_cast<char *>(w.scales.data()), scales);
    if (!f)
        throw std::runtime_error("Truncated weight payload");
    const bool nv = w.config.format == Format::NVFP4;
    for (auto s : w.scales)
        if (nv ? s > 126 : s == 255)
            throw std::runtime_error("Invalid scale encoding");
    if (!nv) {
        for (auto v : w.data)
            if ((v & 127) == 127)
                throw std::runtime_error("NaN weight code is not supported");
    } else if (w.cols & 1) {
        for (uint64_t r = 0; r < w.rows; ++r)
            if (w.data[(r + 1) * w.row_bytes() - 1] & 0xf0)
                throw std::runtime_error("Nonzero padding nibble");
    }
    return w;
}
void write_log(const std::string &path, const Tensor &t, const Weights &w, const Metrics &m,
               const Timings &g, const Timings &c, double io, double total, int warmup, int repeats,
               const std::string &backend) {
    std::ofstream f(path);
    if (!f)
        throw std::runtime_error("Cannot open log file");
    f << "{\n";
    auto item = [&](const std::string &key, const std::string &value) {
        f << "  " << json_string(key) << ": " << value << ",\n";
    };
    auto num = [&](const std::string &key, double n) { item(key, number(n)); };
    item("backend", json_string(backend));
    item("device", json_string(g.device));
    item("compute_capability", json_string(g.compute_capability));
    num("cuda_runtime_version", g.runtime_version);
    num("cuda_driver_version", g.driver_version);
    item("target_gpu_note", json_string(w.config.target_gpu));
    item("format", json_string(w.config.format == Format::MXFP8 ? "mxfp8" : "nvfp4"));
    item("scale_mode", json_string(w.config.tensor_scale ? "tensor" : "block"));
    item("rounding", json_string(w.config.stochastic ? "stochastic" : "nearest"));
    item("seed", std::to_string(w.config.seed));
    num("rows", t.rows);
    num("cols", t.cols);
    num("block_size", w.config.block_size);
    item("input_type", json_string(dtype_name(t.dtype)));
    item("output_type", json_string(dtype_name(w.config.output_type)));
    num("global_scale", w.global_scale);
    num("max_abs_error", m.max_abs);
    num("mae", m.mae);
    num("mse", m.mse);
    num("rmse", m.rmse);
    num("nonfinite_output_count", m.nonfinite_output);
    num("input_payload_bytes", t.data.size());
    num("packed_data_bytes", w.data.size());
    num("scale_bytes", w.scales.size());
    num("weight_file_bytes", 96.0 + w.data.size() + w.scales.size());
    double payload =
        double(w.data.size() + w.scales.size()) + (w.config.format == Format::NVFP4 ? 4 : 0);
    num("compression_payload", t.data.size() / payload);
    num("compression_file", (32.0 + t.data.size()) / (96.0 + w.data.size() + w.scales.size()));
    num("quant_kernel_ms", backend == "cuda" ? g.quant_ms : 0);
    num("dequant_kernel_ms", backend == "cuda" ? g.dequant_ms : 0);
    num("quant_logical_bytes", g.quant_bytes);
    num("dequant_logical_bytes", g.dequant_bytes);
    num("quant_effective_gbps", g.quant_ms > 0 ? g.quant_bytes / (g.quant_ms * 1e6) : 0);
    num("dequant_effective_gbps", g.dequant_ms > 0 ? g.dequant_bytes / (g.dequant_ms * 1e6) : 0);
    num("cpu_quant_ms", c.quant_ms);
    num("cpu_dequant_ms", c.dequant_ms);
    num("gpu_transfer_pipeline_ms", backend == "cuda" ? g.transfer_pipeline_ms : 0);
    num("resident_speedup",
        backend == "cuda" ? (c.quant_ms + c.dequant_ms) / (g.quant_ms + g.dequant_ms) : 1);
    num("transfer_pipeline_speedup",
        backend == "cuda" ? c.transfer_pipeline_ms / g.transfer_pipeline_ms : 1);
    num("file_io_ms", io);
    num("process_total_ms", total);
    num("gpu_warmup", warmup);
    num("gpu_repeats", repeats);
    f << "  \"software_simulation\": true\n}\n";
    f.close();
    check_written(f);
}
} // namespace lp
