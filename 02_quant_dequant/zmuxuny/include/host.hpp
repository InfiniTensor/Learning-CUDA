#pragma once
#if defined(LP_ASCEND)
#include "../ascend/backend.hpp"
#else
#include "gpu.cuh"
#endif
#include <algorithm>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace lp {
inline void check(cudaError_t e) {
  if (e != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(e));
}
struct Device {
  void *ptr = nullptr;
  explicit Device(size_t bytes) { check(cudaMalloc(&ptr, bytes)); }
  ~Device() { cudaFree(ptr); }
  Device(const Device &) = delete;
  Device &operator=(const Device &) = delete;
  template <class T> T *as() { return static_cast<T *>(ptr); }
  void upload(const void *x, size_t n) {
    check(cudaMemcpy(ptr, x, n, cudaMemcpyHostToDevice));
  }
  void download(void *x, size_t n) {
    check(cudaMemcpy(x, ptr, n, cudaMemcpyDeviceToHost));
  }
};
template <class F> double elapsed(F f, int repeats) {
  for (int i = 0; i < 3; ++i)
    f();
  check(cudaGetLastError());
  check(cudaDeviceSynchronize());
  cudaEvent_t a, b;
  check(cudaEventCreate(&a));
  check(cudaEventCreate(&b));
  check(cudaEventRecord(a));
  for (int i = 0; i < repeats; ++i)
    f();
  check(cudaEventRecord(b));
  check(cudaEventSynchronize(b));
  float ms;
  check(cudaEventElapsedTime(&ms, a, b));
  check(cudaEventDestroy(a));
  check(cudaEventDestroy(b));
  check(cudaGetLastError());
  return double(ms) / repeats;
}
inline int width(int dtype) { return dtype == FP32 ? 4 : 2; }
inline int dtype_id(const std::string &s) {
  if (s == "fp32")
    return FP32;
  if (s == "fp16")
    return FP16;
  if (s == "bf16")
    return BF16;
  throw std::runtime_error("dtype must be fp32, fp16 or bf16");
}
inline std::string trim(std::string x) {
  auto a = x.find_first_not_of(" \t\r\n\"");
  return a == std::string::npos ? ""
                                : x.substr(a, x.find_last_not_of(" \t\r\n\"") - a + 1);
}
using Options = std::map<std::string, std::string>;
inline Options arguments(int argc, char **argv) {
  Options o;
  for (int i = 1; i < argc; i += 2) {
    std::string key(argv[i]);
    if (key.rfind("--", 0) != 0 || i + 1 == argc)
      throw std::runtime_error("expected --key value pairs");
    o[key.substr(2)] = argv[i + 1];
  }
  return o;
}
inline std::string get(const Options &o, const std::string &k,
                       const std::string &d = "") {
  auto it = o.find(k);
  return it == o.end() ? d : it->second;
}
inline std::string required(const Options &o, const std::string &k) {
  auto s = get(o, k);
  if (s.empty())
    throw std::runtime_error("missing --" + k);
  return s;
}
inline size_t positive(const std::string &s) {
  size_t used;
  long long v = std::stoll(s, &used);
  if (v <= 0 || used != s.size())
    throw std::runtime_error("expected positive integer: " + s);
  return size_t(v);
}
inline Options config(const std::string &path) {
  std::ifstream f(path);
  if (!f)
    throw std::runtime_error("cannot open config: " + path);
  Options o;
  std::string line;
  while (std::getline(f, line)) {
    line = trim(line.substr(0, line.find('#')));
    if (line.empty())
      continue;
    auto eq = line.find('=');
    if (eq == std::string::npos)
      throw std::runtime_error("expected key = value");
    o[trim(line.substr(0, eq))] = trim(line.substr(eq + 1));
  }
  return o;
}
template <class T> void read(std::istream &f, T &v) {
  f.read(reinterpret_cast<char *>(&v), sizeof(v));
}
template <class T> void write(std::ostream &f, T v) {
  f.write(reinterpret_cast<const char *>(&v), sizeof(v));
}
struct Tensor {
  size_t rows, cols;
  int dtype;
  std::vector<uint8_t> data;
};
inline void valid_shape(uint64_t rows, uint64_t cols) {
  if (!rows || !cols || rows > (size_t(1) << 34) / cols)
    throw std::runtime_error("invalid tensor shape");
}
inline Tensor read_tensor(const std::string &path) {
  std::ifstream f(path, std::ios::binary);
  char magic[8];
  uint64_t r = 0, c = 0;
  uint32_t d = 9, reserved;
  f.read(magic, 8);
  read(f, r);
  read(f, c);
  read(f, d);
  read(f, reserved);
  if (!f || std::memcmp(magic, "LPTENS1\0", 8) || d > 2)
    throw std::runtime_error("invalid tensor header");
  valid_shape(r, c);
  Tensor t{size_t(r), size_t(c), int(d), std::vector<uint8_t>(r * c * width(d))};
  f.read(reinterpret_cast<char *>(t.data.data()), t.data.size());
  if (!f || f.peek() != EOF)
    throw std::runtime_error("tensor payload size mismatch");
  for (size_t i = 0; i < r * c; ++i)
    if (!std::isfinite(load(t.data.data(), i, d)))
      throw std::runtime_error("input must contain finite values");
  return t;
}
inline void write_tensor(const std::string &path, const Tensor &t) {
  std::ofstream f(path, std::ios::binary);
  f.write("LPTENS1\0", 8);
  write(f, uint64_t(t.rows));
  write(f, uint64_t(t.cols));
  write(f, uint32_t(t.dtype));
  write(f, uint32_t(0));
  f.write(reinterpret_cast<const char *>(t.data.data()), t.data.size());
  if (!f)
    throw std::runtime_error("cannot write tensor: " + path);
}
struct Packed {
  Layout p;
  float global;
  std::vector<uint8_t> scales, data;
};
inline void write_packed(const std::string &path, const Packed &q) {
  std::ofstream f(path, std::ios::binary);
  f.write("LPPACK1\0", 8);
  write(f, uint64_t(q.p.rows));
  write(f, uint64_t(q.p.cols));
  write(f, uint32_t(q.p.fmt));
  write(f, uint32_t(q.p.block));
  write(f, uint32_t(q.p.tensor));
  write(f, q.global);
  write(f, uint64_t(q.scales.size()));
  write(f, uint64_t(q.data.size()));
  f.write(reinterpret_cast<const char *>(q.scales.data()), q.scales.size());
  f.write(reinterpret_cast<const char *>(q.data.data()), q.data.size());
  if (!f)
    throw std::runtime_error("cannot write packed file: " + path);
}
inline Packed read_packed(const std::string &path) {
  std::ifstream f(path, std::ios::binary);
  char magic[8];
  uint64_t r = 0, c = 0, ns = 0, nb = 0;
  uint32_t fmt = 9, block = 0, tensor = 0;
  float g = 0;
  f.read(magic, 8);
  read(f, r);
  read(f, c);
  read(f, fmt);
  read(f, block);
  read(f, tensor);
  read(f, g);
  read(f, ns);
  read(f, nb);
  if (!f || std::memcmp(magic, "LPPACK1\0", 8) || fmt > 1 || block < 16 ||
      block > 1024 || block % 16 || tensor > 1 || !std::isfinite(g) || g <= 0)
    throw std::runtime_error("invalid packed header");
  valid_shape(r, c);
  Layout p{size_t(r), size_t(c), FP32, int(fmt), int(block), bool(tensor), false, 0};
  if (ns != p.groups() || nb != p.bytes())
    throw std::runtime_error("packed layout size mismatch");
  Packed q{p, g, std::vector<uint8_t>(ns), std::vector<uint8_t>(nb)};
  f.read(reinterpret_cast<char *>(q.scales.data()), ns);
  f.read(reinterpret_cast<char *>(q.data.data()), nb);
  if (!f || f.peek() != EOF)
    throw std::runtime_error("packed payload size mismatch");
  for (auto s : q.scales)
    if ((fmt == MXFP8 && s == 255) || (fmt == NVFP4 && s > 126))
      throw std::runtime_error("invalid scale code");
  if (fmt == MXFP8)
    for (auto v : q.data)
      if ((v & 127) == 127)
        throw std::runtime_error("NaN FP8 payload unsupported");
  return q;
}
inline Layout layout(const Tensor &t, const Options &o) {
  auto fmt = get(o, "format", "mxfp8"), mode = get(o, "scale_mode", "block"),
       rounding = get(o, "rounding", "nearest");
  if (fmt != "mxfp8" && fmt != "nvfp4")
    throw std::runtime_error("format must be mxfp8 or nvfp4");
  if (mode != "block" && mode != "tensor")
    throw std::runtime_error("scale_mode must be block or tensor");
  if (rounding != "nearest" && rounding != "stochastic")
    throw std::runtime_error("rounding must be nearest or stochastic");
  size_t b = positive(get(o, "block_size", fmt == "mxfp8" ? "32" : "16"));
  if (b < 16 || b > 1024 || b % 16)
    throw std::runtime_error("block_size must be a multiple of 16 in [16,1024]");
  return {t.rows,
          t.cols,
          t.dtype,
          fmt == "mxfp8" ? MXFP8 : NVFP4,
          int(b),
          mode == "tensor",
          rounding == "stochastic",
          uint32_t(std::stoul(get(o, "seed", "42")))};
}
inline void log_json(const std::string &path,
                     const std::map<std::string, double> &metrics,
                     const std::map<std::string, std::string> &labels) {
  std::ofstream f(path);
  f << std::setprecision(12) << "{\n";
  bool first = true;
  for (auto &v : labels) {
    if (!first)
      f << ",\n";
    first = false;
    f << "  \"" << v.first << "\": " << std::quoted(v.second);
  }
  for (auto &v : metrics) {
    if (!std::isfinite(v.second))
      throw std::runtime_error("nonfinite metric: " + v.first);
    if (!first)
      f << ",\n";
    first = false;
    f << "  \"" << v.first << "\": " << v.second;
  }
  f << "\n}\n";
  if (!f)
    throw std::runtime_error("cannot write log: " + path);
}
inline std::map<std::string, double> errors(const Tensor &x, const Tensor &y) {
  double mae = 0, mse = 0, mx = 0;
  for (size_t i = 0; i < x.rows * x.cols; ++i) {
    double d =
        double(load(x.data.data(), i, x.dtype)) - load(y.data.data(), i, y.dtype);
    mae += fabs(d);
    mse += d * d;
    mx = std::max(mx, fabs(d));
  }
  return {{"max_abs_error", mx},
          {"mae", mae / (x.rows * x.cols)},
          {"mse", mse / (x.rows * x.cols)}};
}
inline std::string gpu_name() {
  cudaDeviceProp prop;
  check(cudaGetDeviceProperties(&prop, 0));
  return prop.name;
}
} // namespace lp
