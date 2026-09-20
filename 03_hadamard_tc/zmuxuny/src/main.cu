#include "../include/host.hpp"
#if defined(LP_ASCEND)
#include "../ascend/hadamard.hpp"
#else
#include "hadamard.cuh"
#include "tensor_core.cuh"
#endif

using namespace lp;

int main(int argc, char **argv) try {
  auto o = arguments(argc, argv);
  auto x = read_tensor(required(o, "input"));
  if (x.dtype == FP32)
    throw std::runtime_error("Hadamard input must be fp16 or bf16");
  int d = int(x.cols);
  if (x.cols > 1024 || (d & (d - 1)))
    throw std::runtime_error("head_dim must be a power of two in [1,1024]");
  size_t batch = positive(get(o, "batch", "1")),
         seq = positive(get(o, "seq", std::to_string(x.rows))),
         heads = positive(get(o, "heads", "1"));
  if (batch > x.rows || seq > x.rows / batch || heads != x.rows / (batch * seq) ||
      batch * seq * heads != x.rows)
    throw std::runtime_error("batch*seq*heads must equal input rows");
  bool norm = get(o, "normalize", "1") == "1",
       signs = get(o, "random_sign", "0") == "1";
  uint32_t sign_seed = uint32_t(std::stoul(get(o, "sign_seed", "7")));
  int repeats = int(positive(get(o, "repeats", "100")));
  Options cfg{{"format", get(o, "format", "mxfp8")},
              {"rounding", get(o, "rounding", "nearest")},
              {"seed", get(o, "seed", "42")}};
  Layout p = layout(x, cfg);
  Device in(x.data.size()), out(x.data.size()), data(p.bytes()), scales(p.groups()),
      amax(sizeof(float));
  Device fdata(p.bytes()), fscales(p.groups()), fmax(sizeof(float));
  in.upload(x.data.data(), x.data.size());
  check(cudaMemset(amax.ptr, 0, 4));
  check(cudaMemset(fmax.ptr, 0, 4));
  auto transform = [&] {
    launch_had<0>(in.ptr, out.ptr, x.rows, d, x.dtype, norm, signs, sign_seed, nullptr,
                  nullptr, nullptr, p);
  };
  double hms = elapsed(transform, repeats);
  Tensor y{x.rows, x.cols, x.dtype, std::vector<uint8_t>(x.data.size())};
  out.download(y.data.data(), y.data.size());
  double separate = elapsed(
      [&] {
        transform();
        quantize(out.ptr, data.as<uint8_t>(), scales.as<uint8_t>(), amax.as<float>(),
                 p);
      },
      repeats);
  double fused = elapsed(
      [&] {
        fused_had(in.ptr, x.rows, d, x.dtype, norm, signs, sign_seed,
                  fdata.as<uint8_t>(), fscales.as<uint8_t>(), fmax.as<float>(), p);
      },
      repeats);
  float ma = 0, mf = 0;
  amax.download(&ma, 4);
  fmax.download(&mf, 4);
  Packed q{p, p.fmt == NVFP4 ? global_scale(ma) : 1, std::vector<uint8_t>(p.groups()),
           std::vector<uint8_t>(p.bytes())};
  Packed fq{p, p.fmt == NVFP4 ? global_scale(mf) : 1, std::vector<uint8_t>(p.groups()),
            std::vector<uint8_t>(p.bytes())};
  data.download(q.data.data(), q.data.size());
  scales.download(q.scales.data(), q.scales.size());
  fdata.download(fq.data.data(), fq.data.size());
  fscales.download(fq.scales.data(), fq.scales.size());
  if (q.global != fq.global || q.data != fq.data || q.scales != fq.scales)
    throw std::runtime_error("fused/unfused packed result mismatch");

  // Float64 CPU transform is deliberately independent of warp scheduling.
  Tensor ref{x.rows, x.cols, x.dtype, std::vector<uint8_t>(x.data.size())};
  auto start = std::chrono::steady_clock::now();
  std::vector<double> row(d);
  for (size_t r = 0; r < x.rows; ++r) {
    for (int c = 0; c < d; ++c) {
      row[c] = load(x.data.data(), r * d + c, x.dtype);
      if (signs && uniform(c, sign_seed) < 0.5f)
        row[c] = -row[c];
    }
    for (int s = 1; s < d; s *= 2)
      for (int base = 0; base < d; base += 2 * s)
        for (int j = 0; j < s; ++j) {
          double a = row[base + j], b = row[base + j + s];
          row[base + j] = a + b;
          row[base + j + s] = a - b;
        }
    for (int c = 0; c < d; ++c)
      save(ref.data.data(), r * d + c,
           float(row[c] * (norm ? 1.0 / std::sqrt(double(d)) : 1.0)), x.dtype);
  }
  double cpu_ms = std::chrono::duration<double, std::milli>(
                      std::chrono::steady_clock::now() - start)
                      .count();
  auto metrics = errors(ref, y);
  double tolerance = x.dtype == FP16 ? 1e-2 : 5e-2;
  if (metrics["max_abs_error"] >= tolerance)
    throw std::runtime_error("Hadamard exceeds absolute error tolerance");
  double transfer_ms = elapsed(
      [&] {
        check(cudaMemcpyAsync(in.ptr, x.data.data(), x.data.size(),
                              cudaMemcpyHostToDevice));
        transform();
        check(cudaMemcpyAsync(y.data.data(), out.ptr, y.data.size(),
                              cudaMemcpyDeviceToHost));
      },
      repeats);
  metrics.insert({{"batch", double(batch)},
                  {"seq", double(seq)},
                  {"heads", double(heads)},
                  {"head_dim", double(d)},
                  {"hadamard_ms", hms},
                  {"unfused_ms", separate},
                  {"fused_ms", fused},
                  {"fusion_speedup", separate / fused},
                  {"cpu_hadamard_ms", cpu_ms},
                  {"gpu_with_transfers_ms", transfer_ms},
                  {"speedup_with_transfers", cpu_ms / transfer_ms},
                  {"hadamard_effective_gbps", 2 * x.data.size() / hms / 1e6},
                  {"packed_equal", 1},
                  {"repeats", double(repeats)}});
#if !defined(LP_ASCEND)
  if (x.dtype == FP16 && d >= 16 && get(o, "tensor_core", "1") == "1") {
    std::vector<__half> h(d * d);
    for (int r = 0; r < d; ++r)
      for (int c = 0; c < d; ++c)
        h[r * d + c] =
            __float2half_rn(__builtin_parity(unsigned(r & c)) ? -1.0f : 1.0f);
    Device dh(h.size() * 2), ty(x.data.size());
    dh.upload(h.data(), h.size() * 2);
    double tc = elapsed(
        [&] {
          hadamard_tc<<<dim3((x.rows + 15) / 16, (d + 16 * DENSE_WAVES - 1) / (16 * DENSE_WAVES)), DENSE_WAVES * MATRIX_LANES>>>(
              in.as<__half>(), dh.as<__half>(), ty.as<__half>(), x.rows, d, norm, signs,
              sign_seed);
        },
        repeats);
    Tensor t{x.rows, x.cols, x.dtype, std::vector<uint8_t>(x.data.size())};
    ty.download(t.data.data(), t.data.size());
    double e = errors(ref, t)["max_abs_error"];
    if (e >= tolerance)
      throw std::runtime_error("Tensor Core exceeds absolute error tolerance");
    metrics["tensor_core_ms"] = tc;
    metrics["tensor_core_max_abs_error"] = e;
    metrics["tensor_core_vs_butterfly_speedup"] = hms / tc;
    if (!get(o, "tc_output").empty())
      write_tensor(get(o, "tc_output"), t);
  }
#endif
#if defined(LP_ASCEND) || defined(__MACACC__) || defined(__ILUVATAR__) || defined(__MUSACC__) || LP_CUDA_ARCH >= 80
  if (d >= 16 && get(o, "tensor_core", "1") == "1") {
    Device ty(x.data.size());
    double tc = elapsed(
        [&] {
          launch_had_mma(in.ptr, ty.ptr, x.rows, d, x.dtype, norm, signs, sign_seed);
        },
        repeats);
    Tensor t{x.rows, x.cols, x.dtype, std::vector<uint8_t>(x.data.size())};
    ty.download(t.data.data(), t.data.size());
    double e = errors(ref, t)["max_abs_error"];
    if (e >= tolerance)
      throw std::runtime_error("factorized MMA exceeds absolute error tolerance");
    double tc_separate = elapsed(
        [&] {
          launch_had_mma(in.ptr, ty.ptr, x.rows, d, x.dtype, norm, signs, sign_seed);
          quantize(ty.ptr, data.as<uint8_t>(), scales.as<uint8_t>(), amax.as<float>(),
                   p);
        },
        repeats);
    double tc_fused = elapsed(
        [&] {
          fused_had_mma(in.ptr, x.rows, d, x.dtype, norm, signs, sign_seed,
                        fdata.as<uint8_t>(), fscales.as<uint8_t>(), fmax.as<float>(),
                        p);
        },
        repeats);
    Packed tq = q, tfq = fq;
    amax.download(&ma, 4);
    fmax.download(&mf, 4);
    tq.global = p.fmt == NVFP4 ? global_scale(ma) : 1;
    tfq.global = p.fmt == NVFP4 ? global_scale(mf) : 1;
    data.download(tq.data.data(), tq.data.size());
    scales.download(tq.scales.data(), tq.scales.size());
    fdata.download(tfq.data.data(), tfq.data.size());
    fscales.download(tfq.scales.data(), tfq.scales.size());
    if (tq.global != tfq.global || tq.data != tfq.data || tq.scales != tfq.scales)
      throw std::runtime_error("factorized MMA fused/unfused packed mismatch");
    if (!get(o, "factorized_tc_packed").empty())
      write_packed(get(o, "factorized_tc_packed"), tfq);
    if (p.fmt == NVFP4 && get(o, "materialized_compare", "0") == "1") {
      double partial = elapsed(
          [&] {
            materialized_had_mma(in.ptr, ty.ptr, x.rows, d, x.dtype, norm, signs,
                                 sign_seed, fdata.as<uint8_t>(), fscales.as<uint8_t>(),
                                 fmax.as<float>(), p);
          },
          repeats);
      Packed pq = tq;
      fmax.download(&mf, 4);
      pq.global = global_scale(mf);
      fdata.download(pq.data.data(), pq.data.size());
      fscales.download(pq.scales.data(), pq.scales.size());
      if (pq.global != tq.global || pq.data != tq.data || pq.scales != tq.scales)
        throw std::runtime_error("materialized MMA packed mismatch");
      if (!get(o, "factorized_tc_materialized_packed").empty())
        write_packed(get(o, "factorized_tc_materialized_packed"), pq);
      metrics["factorized_tc_materialized_ms"] = partial;
      metrics["factorized_tc_materialized_speedup"] = tc_separate / partial;
    }
    metrics["factorized_tc_unfused_ms"] = tc_separate;
    metrics["factorized_tc_fused_ms"] = tc_fused;
    metrics["factorized_tc_fusion_speedup"] = tc_separate / tc_fused;
    metrics["factorized_tc_fused_vs_butterfly_speedup"] = fused / tc_fused;
    metrics["factorized_tc_packed_equal"] = 1;
    metrics["factorized_tc_ms"] = tc;
    metrics["factorized_tc_max_abs_error"] = e;
    metrics["factorized_tc_vs_butterfly_speedup"] = hms / tc;
    if (!get(o, "factorized_tc_output").empty())
      write_tensor(get(o, "factorized_tc_output"), t);
  }
#endif
  write_tensor(required(o, "output"), y);
  write_packed(required(o, "packed"), fq);
  if (!get(o, "unfused_packed").empty())
    write_packed(get(o, "unfused_packed"), q);
  log_json(required(o, "log"), metrics,
           {{"gpu", gpu_name()},
            {"platform", compute_platform()},
            {"fp8_encoding", fp8_encoding_backend()},
            {"format", get(cfg, "format")},
            {"dtype", x.dtype == FP16 ? "fp16" : "bf16"},
            {"normalize", norm ? "true" : "false"},
            {"random_sign", signs ? "true" : "false"},
            {"rounding", get(cfg, "rounding")}});
  std::cout << "Hadamard=" << hms << " ms fused=" << fused << " ms unfused=" << separate
            << " ms; packed equal\n";
  return 0;
} catch (const std::exception &e) {
  std::cerr << "error: " << e.what() << '\n';
  return 1;
}
