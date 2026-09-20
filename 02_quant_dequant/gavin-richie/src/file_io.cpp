#include "tensor_io.h"
#include "quant_kernels.cuh"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace {
float fp16_bits_to_fp32(uint16_t h) {  uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
  uint32_t e = (h >> 10) & 0x1fu;
  uint32_t m = h & 0x3ffu;
  uint32_t bits;
  if (e == 0) {
    if (m == 0) bits = sign;
    else {
      // subnormal
      e = 127 - 15 + 1;
      while ((m & 0x400u) == 0) { m <<= 1; --e; }
      m &= 0x3ffu;
      bits = sign | (e << 23) | (m << 13);
    }
  } else if (e == 0x1f) {
    bits = sign | 0x7f800000u | (m ? 0x400000u : 0u);
  } else {
    bits = sign | ((e + 127 - 15) << 23) | (m << 13);
  }
  float v;
  std::memcpy(&v, &bits, 4);
  return v;
}

void read_exact(std::ifstream& f, void* dst, size_t n, const char* what) {
  f.read(static_cast<char*>(dst), (std::streamsize)n);
  if ((size_t)f.gcount() != n)
    throw std::runtime_error(std::string("truncated file while reading ") +
                             what);
}

}  // namespace

TensorData read_tensor(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open tensor file: " + path);
  int64_t rows, cols;
  char dtype[4];
  read_exact(f, &rows, 8, "num_rows");
  read_exact(f, &cols, 8, "num_cols");
  read_exact(f, dtype, 4, "dtype");
  TensorData t;
  t.header.num_rows = rows;
  t.header.num_cols = cols;
  if (std::string(dtype, 4) == "fp32") {
    t.header.dtype = InputDtype::FP32;
    t.values.resize((size_t)rows * cols);
    read_exact(f, t.values.data(), t.values.size() * 4, "fp32 data");
  } else if (std::string(dtype, 4) == "fp16") {
    t.header.dtype = InputDtype::FP16;
    std::vector<uint16_t> raw((size_t)rows * cols);
    read_exact(f, raw.data(), raw.size() * 2, "fp16 data");
    t.values.resize(raw.size());
    for (size_t i = 0; i < raw.size(); ++i) t.values[i] = fp16_bits_to_fp32(raw[i]);
  } else {
    throw std::runtime_error("unknown dtype in tensor header");
  }
  return t;
}

void write_dequant_tensor(const std::string& path, uint64_t rows,
                          uint64_t cols, OutputType type,
                          const void* raw_row_major) {
  std::ofstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot create tensor file: " + path);
  const size_t n = (size_t)rows * cols;
  f.write((const char*)&rows, 8);
  f.write((const char*)&cols, 8);
  const char* dt = output_type_name(type);
  f.write(dt, 4);
  f.write((const char*)raw_row_major,
          (std::streamsize)(n * output_type_size(type)));
}

const char* output_type_name(OutputType t) {
  switch (t) {
    case OutputType::FP16: return "fp16";
    case OutputType::BF16: return "bf16";
    case OutputType::FP32: return "fp32";
  }
  return "?";
}

size_t output_type_size(OutputType t) {
  switch (t) {
    case OutputType::FP16: return 2;
    case OutputType::BF16: return 2;
    case OutputType::FP32: return 4;
  }
  return 0;
}

// ---------------------------------------------------------------------------
// quant config file: python-style `key = "value"` / `key = 32` lines
// ---------------------------------------------------------------------------

static std::string strip(const std::string& s) {
  size_t a = s.find_first_not_of(" \t\r\n");
  size_t b = s.find_last_not_of(" \t\r\n");
  return a == std::string::npos ? "" : s.substr(a, b - a + 1);
}

QuantConfig parse_config(const std::string& path) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("cannot open config file: " + path);
  QuantConfig cfg;
  std::string line;
  while (std::getline(f, line)) {
    size_t eq = line.find('=');
    if (eq == std::string::npos) continue;
    std::string key = strip(line.substr(0, eq));
    std::string val = strip(line.substr(eq + 1));
    if (!val.empty() && val.front() == '"' && val.back() == '"')
      val = val.substr(1, val.size() - 2);
    if (key == "format") {
      if (val == "mxfp8") cfg.format = QuantFormat::MXFP8_E4M3;
      else if (val == "mxfp8-e5m2") cfg.format = QuantFormat::MXFP8_E5M2;
      else if (val == "nvfp4") cfg.format = QuantFormat::NVFP4;
      else throw std::runtime_error("unknown format: " + val);
    } else if (key == "block_size") {
      cfg.block_size = std::stoi(val);
    } else if (key == "scale_mode") {
      cfg.scale_mode = val == "block" ? ScaleMode::BLOCK : ScaleMode::TENSOR;
    } else if (key == "output_type") {
      if (val == "fp16") cfg.output_type = OutputType::FP16;
      else if (val == "bf16") cfg.output_type = OutputType::BF16;
      else if (val == "fp32") cfg.output_type = OutputType::FP32;
      else throw std::runtime_error("unknown output_type: " + val);
    } else if (key == "rounding") {
      cfg.rounding =
          val == "stochastic" ? Rounding::STOCHASTIC : Rounding::NEAREST;
    } else if (key == "target_gpu") {
      cfg.target_gpu = val;
    }
  }
  return cfg;
}

const char* format_name(QuantFormat f) {
  switch (f) {
    case QuantFormat::MXFP8_E4M3: return "mxfp8-e4m3";
    case QuantFormat::MXFP8_E5M2: return "mxfp8-e5m2";
    case QuantFormat::NVFP4: return "nvfp4";
  }
  return "?";
}

int bits_per_element(QuantFormat f) {
  return f == QuantFormat::NVFP4 ? 4 : 8;
}

// ---------------------------------------------------------------------------
// low-precision weight file I/O
// ---------------------------------------------------------------------------

void save_weights(const std::string& path, const QuantizedBuffer& q,
                  uint64_t rows, uint64_t cols) {
  std::ofstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot create weight file: " + path);
  WeightFileHeader h{};
  std::memcpy(h.magic, "QWHT", 4);
  h.version = 1;
  h.format = (uint32_t)q.format;
  h.scale_mode = (uint32_t)q.scale_mode;
  h.block_size = q.block_size;
  h.num_rows = rows;
  h.num_cols = cols;
  h.num_packed_bytes = q.data.size();
  h.num_scales = q.block_scales.size();
  h.scale_dtype = q.format == QuantFormat::NVFP4 ? 1 : 0;
  h.has_global_scale =
      (q.format == QuantFormat::NVFP4 && q.scale_mode == ScaleMode::BLOCK) ? 1
                                                                           : 0;
  f.write((const char*)&h, sizeof(h));
  f.write((const char*)q.data.data(), q.data.size());
  if (!q.block_scales.empty())
    f.write((const char*)q.block_scales.data(), q.block_scales.size());
  // fp32 scales: per-tensor scale always; nvfp4 additionally global scale
  f.write((const char*)&q.tensor_scale, 4);
  if (h.has_global_scale) f.write((const char*)&q.global_scale, 4);
}

QuantizedBuffer load_weights(const std::string& path, uint64_t* rows,
                             uint64_t* cols) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open weight file: " + path);
  WeightFileHeader h;
  read_exact(f, &h, sizeof(h), "weight header");
  if (std::memcmp(h.magic, "QWHT", 4) != 0 || h.version != 1)
    throw std::runtime_error("bad weight file magic/version");
  QuantizedBuffer q;
  q.format = (QuantFormat)h.format;
  q.scale_mode = (ScaleMode)h.scale_mode;
  q.block_size = h.block_size;
  q.num_elems = h.num_rows * h.num_cols;
  q.data.resize(h.num_packed_bytes);
  read_exact(f, q.data.data(), q.data.size(), "packed data");
  if (h.num_scales) {
    q.block_scales.resize(h.num_scales);
    read_exact(f, q.block_scales.data(), q.block_scales.size(), "scales");
  }
  read_exact(f, &q.tensor_scale, 4, "tensor scale");
  if (h.has_global_scale) read_exact(f, &q.global_scale, 4, "global scale");
  *rows = h.num_rows;
  *cols = h.num_cols;
  return q;
}
