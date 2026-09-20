// CLI for the MXFP8 / NVFP4 software quantization & dequantization pipeline.
//
//   lowp run     --input t.bin --config q.txt [--out-weight w.bin]
//                 [--out-dq out.bin] [--log log.txt]
//       end-to-end: quantize + dequantize + error/perf log
//   lowp quant   --input t.bin --config q.txt --out-weight w.bin
//   lowp dequant --weight w.bin --config q.txt --out-dq out.bin [--log log.txt]
#include "fp_formats.h"
#include "quant_kernels.cuh"
#include "tensor_io.h"

#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#define CUDA_CHECK(call)                                                    \
  do {                                                                      \
    cudaError_t e_ = (call);                                                \
    if (e_ != cudaSuccess) {                                                \
      std::cerr << "CUDA error " << cudaGetErrorString(e_) << " at "        \
                << __FILE__ << ":" << __LINE__ << std::endl;                \
      return 1;                                                             \
    }                                                                       \
  } while (0)

// RunResult-returning variant of the CUDA check macro.
#define PIPE_CHECK(call)                                                 \
  do {                                                                   \
    cudaError_t e_ = (call);                                             \
    if (e_ != cudaSuccess) {                                             \
      std::cerr << "CUDA error " << cudaGetErrorString(e_) << " at "     \
                << __FILE__ << ":" << __LINE__ << std::endl;             \
      return {};                                                         \
    }                                                                    \
  } while (0)

namespace {

struct RunResult {
  ErrorMetrics metrics;
  float quant_ms;
  float dequant_ms;
  double compression;
  double quant_bw;    // GB/s
  double dequant_bw;  // GB/s
  uint64_t n;
};

// Runs quantize + dequantize on device for a host fp32 tensor.
RunResult run_pipeline(const std::vector<float>& host, const QuantConfig& cfg,
                       QuantizedBuffer* out_q, std::vector<uint8_t>* host_dq) {
  const uint64_t n = host.size();
  const uint64_t packed_bytes =
      cfg.format == QuantFormat::NVFP4 ? (n + 1) / 2 : n;
  const uint64_t nblocks = (n + cfg.block_size - 1) / cfg.block_size;
  const bool block_mode = cfg.scale_mode == ScaleMode::BLOCK;

  float* d_in;
  uint8_t *d_packed, *d_scales = nullptr;
  void* d_out;
  PIPE_CHECK(cudaMalloc(&d_in, n * 4));
  PIPE_CHECK(cudaMalloc(&d_packed, packed_bytes));
  if (block_mode) PIPE_CHECK(cudaMalloc(&d_scales, nblocks));
  const size_t out_bytes = n * output_type_size(cfg.output_type);
  PIPE_CHECK(cudaMalloc(&d_out, out_bytes));
  PIPE_CHECK(cudaMemcpy(d_in, host.data(), n * 4, cudaMemcpyHostToDevice));

  StageTiming tq, td;
  launch_quantize(d_in, n, cfg, d_packed, d_scales, nullptr, nullptr, &tq);

  float tensor_scale = 1.0f, global_scale = 1.0f;
  {
    // recover the fp32 scales the same way launch_quantize computes them
    const float amax = reduce_amax(d_in, n);
    if (cfg.scale_mode == ScaleMode::TENSOR)
      tensor_scale = amax > 0 ? amax / (cfg.format == QuantFormat::NVFP4
                                            ? 6.0f : 448.0f) : 1.0f;
    else if (cfg.format == QuantFormat::NVFP4)
      global_scale = amax > 2688.0f ? amax / 2688.0f : 1.0f;
  }
  launch_dequantize(d_packed, n, d_scales, tensor_scale, global_scale, cfg,
                    d_out, &td);

  ErrorMetrics m = compute_metrics(d_in, d_out, n, cfg.output_type);

  RunResult r;
  r.metrics = m;
  r.quant_ms = tq.gpu_ms;
  r.dequant_ms = td.gpu_ms;
  r.n = n;
  const double in_bytes = (double)n * 4;
  const double total_q_bytes =
      (double)packed_bytes + (block_mode ? (double)nblocks : 4.0);
  r.compression =
      total_q_bytes > 0 ? in_bytes / total_q_bytes : 0.0;
  r.quant_bw = (in_bytes + total_q_bytes) / (tq.gpu_ms * 1e-3) / 1e9;
  r.dequant_bw =
      (in_bytes + total_q_bytes) / (td.gpu_ms * 1e-3) / 1e9;

  if (out_q) {
    out_q->format = cfg.format;
    out_q->scale_mode = cfg.scale_mode;
    out_q->block_size = (uint32_t)cfg.block_size;
    out_q->num_elems = n;
    out_q->data.resize(packed_bytes);
    PIPE_CHECK(cudaMemcpy(out_q->data.data(), d_packed, packed_bytes,
                          cudaMemcpyDeviceToHost));
    if (block_mode) {
      out_q->block_scales.resize(nblocks);
      PIPE_CHECK(cudaMemcpy(out_q->block_scales.data(), d_scales, nblocks,
                            cudaMemcpyDeviceToHost));
    }
    out_q->tensor_scale = tensor_scale;
    out_q->global_scale = global_scale;
  }
  if (host_dq) {
    host_dq->resize(out_bytes);
    PIPE_CHECK(cudaMemcpy(host_dq->data(), d_out, out_bytes,
                          cudaMemcpyDeviceToHost));
  }

  cudaFree(d_in);
  cudaFree(d_packed);
  if (d_scales) cudaFree(d_scales);
  cudaFree(d_out);
  return r;
}

void write_log(const std::string& path, const QuantConfig& cfg,
               const TensorHeader& h, const RunResult& r,
               const char* distribution) {
  std::ostringstream os;
  os << std::fixed << std::setprecision(6);
  os << "# low-precision quantization report\n";
  os << "target_gpu = \"" << cfg.target_gpu << "\"\n";
  os << "distribution = \"" << (distribution ? distribution : "unknown")
     << "\"\n";
  os << "format = \"" << format_name(cfg.format) << "\"\n";
  os << "block_size = " << cfg.block_size << "\n";
  os << "scale_mode = \"" << (cfg.scale_mode == ScaleMode::BLOCK ? "block"
                                                                 : "tensor")
     << "\"\n";
  os << "rounding = \"" << (cfg.rounding == Rounding::STOCHASTIC
                                ? "stochastic" : "nearest") << "\"\n";
  os << "input_dtype = \"" << (h.dtype == InputDtype::FP32 ? "fp32" : "fp16")
     << "\"\n";
  os << "output_type = \"" << output_type_name(cfg.output_type) << "\"\n";
  os << "rows = " << h.num_rows << "\n";
  os << "cols = " << h.num_cols << "\n";
  os << "num_elements = " << r.n << "\n";
  os << "max_abs_error = " << r.metrics.max_abs << "\n";
  os << "mae = " << r.metrics.mae << "\n";
  os << "mse = " << r.metrics.mse << "\n";
  os << "compression_ratio = " << std::setprecision(4) << r.compression
     << "\n";
  os << "quantize_kernel_ms = " << std::setprecision(4) << r.quant_ms << "\n";
  os << "dequantize_kernel_ms = " << std::setprecision(4) << r.dequant_ms
     << "\n";
  os << "quantize_bandwidth_gbps = " << r.quant_bw << "\n";
  os << "dequantize_bandwidth_gbps = " << r.dequant_bw << "\n";
  if (path.empty()) {
    std::cout << os.str();
  } else {
    std::ofstream f(path);
    f << os.str();
  }
}

std::string get_arg(int argc, char** argv, const std::string& key,
                    const std::string& def = "") {
  for (int i = 1; i + 1 < argc; ++i)
    if (key == argv[i]) return argv[i + 1];
  return def;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "usage: lowp <run|quant|dequant> [args]\n"
              << "  run     --input T.bin --config Q.txt [--out-weight W.bin] "
                 "[--out-dq O.bin] [--log L.txt] [--dist random]\n"
              << "  quant   --input T.bin --config Q.txt --out-weight W.bin\n"
              << "  dequant --weight W.bin --config Q.txt --out-dq O.bin "
                 "[--log L.txt]\n";
    return 1;
  }
  const std::string mode = argv[1];

  if (mode == "run" || mode == "quant") {
    QuantConfig cfg = parse_config(get_arg(argc, argv, "--config"));
    TensorData t = read_tensor(get_arg(argc, argv, "--input"));
    const std::string wpath = get_arg(argc, argv, "--out-weight");
    const std::string opath = get_arg(argc, argv, "--out-dq");
    QuantizedBuffer q;
    std::vector<uint8_t> dq;
    RunResult r =
        run_pipeline(t.values, cfg, wpath.empty() ? nullptr : &q,
                     opath.empty() ? nullptr : &dq);

    if (!wpath.empty())
      save_weights(wpath, q, t.header.num_rows, t.header.num_cols);
    if (!opath.empty())
      write_dequant_tensor(opath, t.header.num_rows, t.header.num_cols,
                           cfg.output_type, dq.data());
    if (mode == "run") {
      write_log(get_arg(argc, argv, "--log"), cfg, t.header, r,
                get_arg(argc, argv, "--dist").c_str());
    }
    return 0;
  }
  if (mode == "dequant") {
    QuantConfig cfg = parse_config(get_arg(argc, argv, "--config"));
    uint64_t rows, cols;
    QuantizedBuffer q = load_weights(get_arg(argc, argv, "--weight"), &rows, &cols);
    // header overrides what the config file can't express
    cfg.format = q.format;
    cfg.scale_mode = q.scale_mode;
    cfg.block_size = q.block_size;
    const uint64_t n = q.num_elems;

    const float* d_in_dummy = nullptr;
    (void)d_in_dummy;
    uint8_t *d_packed, *d_scales = nullptr;
    void* d_out;
    CUDA_CHECK(cudaMalloc(&d_packed, q.data.size()));
    CUDA_CHECK(cudaMemcpy(d_packed, q.data.data(), q.data.size(),
                          cudaMemcpyHostToDevice));
    if (!q.block_scales.empty()) {
      CUDA_CHECK(cudaMalloc(&d_scales, q.block_scales.size()));
      CUDA_CHECK(cudaMemcpy(d_scales, q.block_scales.data(),
                            q.block_scales.size(), cudaMemcpyHostToDevice));
    }
    const size_t out_bytes = n * output_type_size(cfg.output_type);
    CUDA_CHECK(cudaMalloc(&d_out, out_bytes));

    StageTiming td;
    launch_dequantize(d_packed, n, d_scales, q.tensor_scale, q.global_scale,
                      cfg, d_out, &td);
    std::vector<uint8_t> dq(out_bytes);
    CUDA_CHECK(cudaMemcpy(dq.data(), d_out, out_bytes, cudaMemcpyDeviceToHost));
    std::string opath = get_arg(argc, argv, "--out-dq");
    if (!opath.empty())
      write_dequant_tensor(opath, rows, cols, cfg.output_type, dq.data());

    const double in_bytes = (double)n * 4;
    const double total_q_bytes =
        (double)q.data.size() + (q.block_scales.empty() ? 4.0 : (double)q.block_scales.size());
    std::ostringstream os;
    os << std::fixed << std::setprecision(6)
       << "dequantize_kernel_ms = " << td.gpu_ms << "\n"
       << "dequantize_bandwidth_gbps = "
       << (in_bytes + total_q_bytes) / (td.gpu_ms * 1e-3) / 1e9 << "\n";
    std::string lpath = get_arg(argc, argv, "--log");
    if (lpath.empty())
      std::cout << os.str();
    else {
      std::ofstream f(lpath);
      f << os.str();
    }
    cudaFree(d_packed);
    if (d_scales) cudaFree(d_scales);
    cudaFree(d_out);
    return 0;
  }
  std::cerr << "unknown mode: " << mode << "\n";
  return 1;
}
