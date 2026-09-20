// File I/O round-trip tests: tensor file, config file, weight file.
#include "fp_formats.h"
#include "quant_kernels.cuh"
#include "tensor_io.h"

#include <cuda_fp16.h>

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <random>
#include <vector>

static int failures = 0;
#define CHECK(cond, msg)                                          \
  do {                                                            \
    if (!(cond)) {                                                \
      fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, msg); \
      ++failures;                                                 \
    }                                                             \
  } while (0)

int main() {
  const std::string dir = "/tmp/lowp_io_test";
  system(("mkdir -p " + dir).c_str());

  // ---- tensor file (fp32 & fp16) ------------------------------------------
  {
    std::vector<float> v(100);
    std::mt19937 rng(1);
    std::normal_distribution<float> d(0, 1);
    for (auto& x : v) x = d(rng);

    for (const char* dt : {"fp32", "fp16"}) {
      const std::string p = dir + "/t.bin";
      std::ofstream f(p, std::ios::binary);
      int64_t r = 10, c = 10;
      f.write((const char*)&r, 8);
      f.write((const char*)&c, 8);
      f.write(dt, 4);
      if (std::string(dt) == "fp32") {
        f.write((const char*)v.data(), 400);
      } else {
        __half h[100];
        for (int i = 0; i < 100; ++i) h[i] = __float2half(v[i]);
        f.write((const char*)h, 200);
      }
      f.close();

      TensorData t = read_tensor(p);
      CHECK(t.header.num_rows == 10 && t.header.num_cols == 10,
            "tensor dims");
      CHECK(t.header.dtype == (std::string(dt) == "fp32" ? InputDtype::FP32
                                                         : InputDtype::FP16),
            "tensor dtype");
      CHECK(t.values.size() == 100, "tensor size");
      for (size_t i = 0; i < 100; ++i) {
        if (std::string(dt) == "fp32") {
          CHECK(t.values[i] == v[i], "fp32 roundtrip");
        } else {
          // fp16 storage rounds; compare through the same conversion
          CHECK(t.values[i] == __half2float(__float2half(v[i])),
                "fp16 roundtrip");
        }
      }
    }
  }

  // ---- weight file roundtrip ----------------------------------------------
  {
    QuantizedBuffer q;
    q.format = QuantFormat::NVFP4;
    q.scale_mode = ScaleMode::BLOCK;
    q.block_size = 16;
    q.num_elems = 33;
    q.data.resize(17);         // 33 elements -> 17 bytes (4-bit packed)
    q.block_scales = {1, 2, 3};
    q.tensor_scale = 0.25f;
    q.global_scale = 8.0f;
    for (size_t i = 0; i < q.data.size(); ++i) q.data[i] = (uint8_t)(i * 7);
    save_weights(dir + "/w.bin", q, 3, 11);
    uint64_t rows, cols;
    QuantizedBuffer r = load_weights(dir + "/w.bin", &rows, &cols);
    CHECK(rows == 3 && cols == 11, "weight dims");
    CHECK(r.format == q.format && r.scale_mode == q.scale_mode &&
              r.block_size == q.block_size,
          "weight meta");
    CHECK(r.data == q.data, "weight packed data");
    CHECK(r.block_scales == q.block_scales, "weight scales");
    CHECK(r.tensor_scale == q.tensor_scale, "tensor scale");
    CHECK(r.global_scale == q.global_scale, "global scale");

    // 4-bit packing invariant: 2 elements per byte, not one uint8 each
    CHECK(q.data.size() == (q.num_elems + 1) / 2, "4bit packing density");
  }

  // ---- config file ----------------------------------------------------------
  {
    std::ofstream f(dir + "/q.txt");
    f << "format = \"nvfp4\"\nblock_size = 16\nscale_mode = \"block\"\n"
      << "output_type = \"bf16\"\nrounding = \"stochastic\"\n"
      << "target_gpu = \"4060\"\n";
    f.close();
    QuantConfig c = parse_config(dir + "/q.txt");
    CHECK(c.format == QuantFormat::NVFP4, "cfg format");
    CHECK(c.block_size == 16, "cfg block");
    CHECK(c.scale_mode == ScaleMode::BLOCK, "cfg scale");
    CHECK(c.output_type == OutputType::BF16, "cfg out");
    CHECK(c.rounding == Rounding::STOCHASTIC, "cfg rounding");
    CHECK(c.target_gpu == "4060", "cfg gpu");
  }

  if (failures) {
    fprintf(stderr, "%d failure(s)\n", failures);
    return 1;
  }
  printf("test_io: all passed\n");
  return 0;
}
