#include "lp.hpp"
#include <algorithm>
#include <chrono>
#include <limits>
#include <stdexcept>

namespace lp {
size_t dtype_bytes(DType type) { return type == DType::FP32 ? 4 : 2; }
std::string dtype_name(DType type) {
    return type == DType::FP32 ? "fp32" : type == DType::FP16 ? "fp16" : "bf16";
}
DType parse_dtype(const std::string &s) {
    if (s == "fp32")
        return DType::FP32;
    if (s == "fp16")
        return DType::FP16;
    if (s == "bf16")
        return DType::BF16;
    throw std::runtime_error("Invalid dtype: " + s);
}
void validate_config(const Config &c) {
    if (uint32_t(c.format) > 1 || uint32_t(c.output_type) > 2)
        throw std::runtime_error("Invalid format or output type");
    if (c.block_size < 2 || c.block_size > 1024 || (c.block_size & (c.block_size - 1)))
        throw std::runtime_error("block_size must be a power of two in [2,1024]");
}
size_t checked_size(uint64_t rows, uint64_t cols, size_t bytes) {
    const auto limit = uint64_t(std::numeric_limits<int64_t>::max());
    if (!rows || !cols || rows > limit || cols > limit || rows > limit / cols ||
        rows * cols > std::numeric_limits<size_t>::max() / bytes)
        throw std::runtime_error("Invalid or overflowing tensor dimensions");
    return static_cast<size_t>(rows * cols * bytes);
}
Weights make_weights(const Tensor &t, const Config &c) {
    validate_config(c);
    checked_size(t.rows, t.cols, dtype_bytes(t.dtype));
    Weights w;
    w.rows = t.rows;
    w.cols = t.cols;
    w.config = c;
    w.input_type = t.dtype;
    w.data.resize(checked_size(w.rows, w.row_bytes(), 1));
    w.scales.resize(c.tensor_scale ? 1 : checked_size(w.rows, w.blocks_per_row(), 1));
    return w;
}
Tensor make_output(const Weights &w, DType type) {
    Tensor out{w.rows, w.cols, type, {}};
    out.data.resize(checked_size(w.rows, w.cols, dtype_bytes(type)));
    return out;
}
float tensor_value(const Tensor &t, uint64_t i) {
    if (t.dtype == DType::FP32) {
        float x;
        std::memcpy(&x, t.data.data() + i * 4, 4);
        return x;
    }
    uint16_t h;
    std::memcpy(&h, t.data.data() + i * 2, 2);
    return t.dtype == DType::FP16 ? from_half(h) : from_bfloat(h);
}
static void store_value(Tensor &t, uint64_t i, float x) {
    if (t.dtype == DType::FP32)
        std::memcpy(t.data.data() + i * 4, &x, 4);
    else {
        uint16_t h = t.dtype == DType::FP16 ? to_half(x) : to_bfloat(x);
        std::memcpy(t.data.data() + i * 2, &h, 2);
    }
}
void quantize_cpu(const Tensor &t, Weights &w) {
    const Config &c = w.config;
    const bool nv = c.format == Format::NVFP4;
    float amax = 0;
    if (nv || c.tensor_scale)
        for (uint64_t i = 0; i < t.size(); ++i)
            amax = std::max(amax, std::fabs(tensor_value(t, i)));
    w.global_scale = nv ? nv_global(amax) : 1;
    const uint64_t groups = w.scales.size();
    for (uint64_t g = 0; g < groups; ++g) {
        uint64_t begin = 0, end = t.size();
        float local = amax;
        if (!c.tensor_scale) {
            const uint64_t row = g / w.blocks_per_row();
            const uint64_t col = (g % w.blocks_per_row()) * c.block_size;
            begin = row * t.cols + col;
            end = begin + std::min<uint64_t>(c.block_size, t.cols - col);
            local = 0;
            for (uint64_t i = begin; i < end; ++i)
                local = std::max(local, std::fabs(tensor_value(t, i)));
        }
        w.scales[g] = nv ? nv_scale(local, w.global_scale) : mx_scale(local);
        const float scale = nv ? decode_e4m3(w.scales[g]) : decode_e8m0(w.scales[g]);
        for (uint64_t i = begin; i < end; ++i) {
            float x = tensor_value(t, i);
            // Divide in stages to avoid overflow/underflow in an intermediate product.
            float y = nv ? (scale == 0 ? std::copysign(0.0f, x) : (x / w.global_scale) / scale)
                         : x / scale;
            const float u = c.stochastic ? random_unit(i, c.seed) : 0;
            const uint8_t code =
                nv ? encode_e2m1(y, c.stochastic, u) : encode_e4m3(y, c.stochastic, u);
            if (!nv)
                w.data[i] = code;
            else {
                const uint64_t row = i / t.cols, col = i % t.cols;
                uint8_t &packed = w.data[row * w.row_bytes() + col / 2];
                if ((col & 1) == 0)
                    packed = code;
                else
                    packed |= code << 4;
            }
        }
    }
}
void dequantize_cpu(const Weights &w, Tensor &out) {
    const bool nv = w.config.format == Format::NVFP4;
    for (uint64_t r = 0; r < w.rows; ++r) {
        for (uint64_t col = 0; col < w.cols; ++col) {
            const uint64_t g =
                w.config.tensor_scale ? 0 : r * w.blocks_per_row() + col / w.config.block_size;
            const float scale = nv ? decode_e4m3(w.scales[g]) : decode_e8m0(w.scales[g]);
            const uint8_t code = w.data[r * w.row_bytes() + (nv ? col / 2 : col)];
            const float value =
                nv ? decode_e2m1((code >> ((col & 1) * 4)) & 15) : decode_e4m3(code);
            store_value(out, r * w.cols + col, (value * scale) * w.global_scale);
        }
    }
}
double now_ms() {
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}
double elapsed_ms(double start) { return now_ms() - start; }
Timings run_cpu(const Tensor &t, Weights &w, Tensor &out, int repeats) {
    Timings timings;
    quantize_cpu(t, w);
    dequantize_cpu(w, out);
    const double q0 = now_ms();
    for (int k = 0; k < repeats; ++k)
        quantize_cpu(t, w);
    timings.quant_ms = elapsed_ms(q0) / repeats;
    const double d0 = now_ms();
    for (int k = 0; k < repeats; ++k)
        dequantize_cpu(w, out);
    timings.dequant_ms = elapsed_ms(d0) / repeats;
    timings.transfer_pipeline_ms = timings.quant_ms + timings.dequant_ms;
    return timings;
}
Metrics measure_error(const Tensor &input, const Tensor &output) {
    Metrics m;
    long double sum = 0, squares = 0;
    for (uint64_t i = 0; i < input.size(); ++i) {
        double y = tensor_value(output, i);
        if (!std::isfinite(y)) {
            ++m.nonfinite_output;
            continue;
        }
        double d = std::abs(double(tensor_value(input, i)) - y);
        m.max_abs = std::max(m.max_abs, d);
        sum += d;
        squares += static_cast<long double>(d) * d;
    }
    if (m.nonfinite_output)
        m.max_abs = m.mae = m.mse = m.rmse = std::numeric_limits<double>::infinity();
    else {
        m.mae = double(sum / input.size());
        m.mse = double(squares / input.size());
        m.rmse = std::sqrt(m.mse);
    }
    return m;
}
} // namespace lp
