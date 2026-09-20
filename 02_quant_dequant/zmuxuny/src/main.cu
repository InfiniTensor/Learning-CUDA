#include "../include/host.hpp"

using namespace lp;

// Scalar CPU baseline for the same format, scales and packed layout.
Packed cpu_quantize(const Tensor &x, Layout p) {
  Packed q{p, 1, std::vector<uint8_t>(p.groups()), std::vector<uint8_t>(p.bytes(), 0)};
  float amax = 0;
  if (p.fmt == NVFP4 || p.tensor)
    for (size_t i = 0; i < p.count(); ++i)
      amax = std::max(amax, fabsf(load(x.data.data(), i, p.dtype)));
  if (p.fmt == NVFP4)
    q.global = global_scale(amax);
  size_t gpr = (p.cols + p.block - 1) / p.block;
  for (size_t g = 0; g < p.groups(); ++g) {
    float a = amax;
    if (!p.tensor) {
      a = 0;
      size_t r = g / gpr, start = (g % gpr) * p.block;
      for (size_t c = start; c < std::min(start + p.block, p.cols); ++c)
        a = std::max(a, fabsf(load(x.data.data(), r * p.cols + c, p.dtype)));
    }
    q.scales[g] = p.fmt == MXFP8 ? mx_scale(a) : encode8((a / q.global) / 6.0f);
  }
  for (size_t r = 0; r < p.rows; ++r)
    for (size_t c = 0; c < p.cols; ++c) {
      size_t i = r * p.cols + c, g = p.tensor ? 0 : r * gpr + c / p.block;
      float s = scale_value(q.scales[g], p.fmt), v = load(x.data.data(), i, p.dtype);
      if (p.fmt == NVFP4)
        v /= q.global;
      v = s == 0 ? 0 : v / s;
      uint8_t code = p.fmt == MXFP8 ? encode8(v, p.stochastic, uniform(i, p.seed))
                                    : encode4(v, p.stochastic, uniform(i, p.seed));
      if (p.fmt == MXFP8)
        q.data[i] = code;
      else
        q.data[r * ((p.cols + 1) / 2) + c / 2] |= code << ((c & 1) * 4);
    }
  return q;
}

int main(int argc, char **argv) try {
  auto o = arguments(argc, argv);
  auto mode = get(o, "mode", "quantize");
  int repeats = int(positive(get(o, "repeats", "50")));
  if (mode == "dequantize") {
    auto q = read_packed(required(o, "packed"));
    int out_type = dtype_id(get(o, "output_type", "fp32"));
    Device data(q.data.size()), scales(q.scales.size()),
        out(q.p.count() * width(out_type));
    data.upload(q.data.data(), q.data.size());
    scales.upload(q.scales.data(), q.scales.size());
    double ms = elapsed(
        [&] {
          dequantize(data.as<uint8_t>(), scales.as<uint8_t>(), q.global, out.ptr,
                     out_type, q.p);
        },
        repeats);
    Tensor y{q.p.rows, q.p.cols, out_type,
             std::vector<uint8_t>(q.p.count() * width(out_type))};
    out.download(y.data.data(), y.data.size());
    write_tensor(required(o, "output"), y);
    log_json(required(o, "log"), {{"dequant_ms", ms}},
             {{"gpu", gpu_name()},
            {"platform", compute_platform()},
              {"fp8_encoding", fp8_encoding_backend()},
              {"mode", mode}});
    return 0;
  }
  if (mode != "quantize")
    throw std::runtime_error("mode must be quantize or dequantize");
  auto x = read_tensor(required(o, "input"));
  if (x.dtype == BF16)
    throw std::runtime_error("quantization input must be fp32 or fp16");
  auto cfg = config(required(o, "config"));
  auto p = layout(x, cfg);
  int out_type = dtype_id(get(cfg, "output_type", "fp16"));
  Device in(x.data.size()), data(p.bytes()), scales(p.groups()), amax(sizeof(float)),
      out(p.count() * width(out_type));
  in.upload(x.data.data(), x.data.size());
  check(cudaMemset(amax.ptr, 0, sizeof(float)));
  double qms = elapsed(
      [&] {
        quantize(in.ptr, data.as<uint8_t>(), scales.as<uint8_t>(), amax.as<float>(), p);
      },
      repeats);
  float max_value = 0;
  amax.download(&max_value, sizeof(float));
  float global = p.fmt == NVFP4 ? global_scale(max_value) : 1;
  double dms = elapsed(
      [&] {
        dequantize(data.as<uint8_t>(), scales.as<uint8_t>(), global, out.ptr, out_type,
                   p);
      },
      repeats);
  Tensor y{x.rows, x.cols, out_type, std::vector<uint8_t>(p.count() * width(out_type))};
  // End-to-end timing includes transfers and all scale kernels; excludes file I/O and
  // allocation.
  double e2e = elapsed(
      [&] {
        check(cudaMemcpyAsync(in.ptr, x.data.data(), x.data.size(),
                              cudaMemcpyHostToDevice));
        quantize(in.ptr, data.as<uint8_t>(), scales.as<uint8_t>(), amax.as<float>(), p);
        dequantize(data.as<uint8_t>(), scales.as<uint8_t>(), global, out.ptr, out_type,
                   p);
        check(cudaMemcpyAsync(y.data.data(), out.ptr, y.data.size(),
                              cudaMemcpyDeviceToHost));
      },
      repeats);
  Packed q{p, global, std::vector<uint8_t>(p.groups()),
           std::vector<uint8_t>(p.bytes())};
  data.download(q.data.data(), q.data.size());
  scales.download(q.scales.data(), q.scales.size());
  out.download(y.data.data(), y.data.size());
  auto start = std::chrono::steady_clock::now();
  auto cpu = cpu_quantize(x, p);
  Tensor cy{x.rows, x.cols, out_type, std::vector<uint8_t>(y.data.size())};
  size_t gpr = (p.cols + p.block - 1) / p.block;
  for (size_t i = 0; i < p.count(); ++i) {
    size_t r = i / p.cols, c = i % p.cols, g = p.tensor ? 0 : r * gpr + c / p.block;
    uint8_t code =
        p.fmt == MXFP8
            ? cpu.data[i]
            : (cpu.data[r * ((p.cols + 1) / 2) + c / 2] >> ((c & 1) * 4)) & 15;
    float v = p.fmt == MXFP8 ? fp8_value(code) : fp4_value(code);
    save(cy.data.data(), i, (v * scale_value(cpu.scales[g], p.fmt)) * cpu.global,
         out_type);
  }
  double cpu_ms = std::chrono::duration<double, std::milli>(
                      std::chrono::steady_clock::now() - start)
                      .count();
  if (cpu.data != q.data || cpu.scales != q.scales || cpu.global != q.global ||
      cy.data != y.data)
    throw std::runtime_error("CPU/GPU packed or dequantized result mismatch");
  auto metrics = errors(x, y);
  metrics.insert(
      {{"rows", double(p.rows)},
       {"cols", double(p.cols)},
       {"quant_ms", qms},
       {"dequant_ms", dms},
       {"gpu_with_transfers_ms", e2e},
       {"cpu_quant_dequant_ms", cpu_ms},
       {"speedup_with_transfers", cpu_ms / e2e},
       {"quant_effective_gbps", (x.data.size() + p.bytes() + p.groups()) / qms / 1e6},
       {"dequant_effective_gbps", (y.data.size() + p.bytes() + p.groups()) / dms / 1e6},
       {"compression_payload", double(x.data.size()) / (p.bytes() + p.groups() + 4)},
       {"compression_file", double(x.data.size() + 32) / (p.bytes() + p.groups() + 56)},
       {"packed_data_bytes", double(p.bytes())},
       {"scale_bytes", double(p.groups())},
       {"repeats", double(repeats)}});
  write_packed(required(o, "packed"), q);
  write_tensor(required(o, "output"), y);
  log_json(required(o, "log"), metrics,
           {{"gpu", gpu_name()},
            {"platform", compute_platform()},
            {"fp8_encoding", fp8_encoding_backend()},
            {"format", get(cfg, "format", "mxfp8")},
            {"rounding", get(cfg, "rounding", "nearest")},
            {"scale_mode", get(cfg, "scale_mode", "block")},
            {"target_gpu", get(cfg, "target_gpu", "unspecified")},
            {"bandwidth_definition",
             "logical input+output+scales, not measured DRAM traffic"}});
  std::cout << "quant=" << qms << " ms dequant=" << dms << " ms CPU/GPU exact match\n";
  return 0;
} catch (const std::exception &e) {
  std::cerr << "error: " << e.what() << '\n';
  return 1;
}
