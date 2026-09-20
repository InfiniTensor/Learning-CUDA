// GPU round-trip tests: quantize -> dequantize through the CUDA kernels and
// compare against (a) a host software reference for exactness of the data
// path and (b) sanity error bounds vs the original fp32 values.
#include "fp_formats.h"
#include "quant_kernels.cuh"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
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

static float dequant_at(const std::vector<float>& in, const QuantConfig& cfg,
                        size_t i);

// Host reference of the full pipeline (must match kernel decisions).
struct HostRef {
  std::vector<float> dq;
};

static HostRef host_reference(const std::vector<float>& in,
                              const QuantConfig& cfg) {
  using namespace lp;
  HostRef r;
  const size_t n = in.size();
  const size_t bs = cfg.block_size;
  float amax_t = 0;
  for (float v : in) amax_t = fmaxf(amax_t, fabsf(v));
  float tensor_scale = 1, global_scale = 1;
  if (cfg.scale_mode == ScaleMode::TENSOR)
    tensor_scale = amax_t > 0 ? amax_t / (cfg.format == QuantFormat::NVFP4
                                              ? 6.0f : 448.0f) : 1.0f;
  else if (cfg.format == QuantFormat::NVFP4)
    global_scale = amax_t > 2688.0f ? amax_t / 2688.0f : 1.0f;

  r.dq.resize(n);
  for (size_t b = 0; b * bs < n; ++b) {
    const size_t s = b * bs, e = std::min(s + bs, n);
    float scale;
    if (cfg.scale_mode == ScaleMode::BLOCK) {
      float amax = 0;
      for (size_t i = s; i < e; ++i) amax = fmaxf(amax, fabsf(in[i]));
      if (cfg.format == QuantFormat::NVFP4) {
        float sb = amax == 0 ? 1.0f
                             : fp8_to_fp32(fp32_to_fp8(amax / global_scale / 6.0f, true), true);
        scale = global_scale * sb;
      } else {
        const float fmax = cfg.format == QuantFormat::MXFP8_E4M3 ? 448.0f
                                                                 : 57344.0f;
        const uint8_t code = amax == 0 ? 127 : fp32_to_e8m0_ceil(amax / fmax);
        scale = e8m0_to_fp32(code);
      }
    } else {
      scale = tensor_scale;
    }
    for (size_t i = s; i < e; ++i) {
      if (cfg.format == QuantFormat::NVFP4)
        r.dq[i] = e2m1_to_fp32(fp32_to_e2m1(in[i] / scale)) * scale;
      else
        r.dq[i] = fp8_to_fp32(
                      fp32_to_fp8(in[i] / scale,
                                  cfg.format == QuantFormat::MXFP8_E4M3),
                      cfg.format == QuantFormat::MXFP8_E4M3) * scale;
    }
  }
  return r;
}

// Run the GPU pipeline and return fp32 dequantized values.
static std::vector<float> gpu_pipeline(const std::vector<float>& in,
                                       const QuantConfig& cfg) {
  const uint64_t n = in.size();
  float* d_in;
  uint8_t *d_packed, *d_scales = nullptr;
  float* d_out;
  cudaMalloc(&d_in, n * 4);
  cudaMalloc(&d_packed, cfg.format == QuantFormat::NVFP4 ? (n + 1) / 2 : n);
  const uint64_t nblocks = (n + cfg.block_size - 1) / cfg.block_size;
  if (cfg.scale_mode == ScaleMode::BLOCK) cudaMalloc(&d_scales, nblocks);
  cudaMalloc(&d_out, n * 4);
  cudaMemcpy(d_in, in.data(), n * 4, cudaMemcpyHostToDevice);

  QuantConfig cfg32 = cfg;
  cfg32.output_type = OutputType::FP32;
  launch_quantize(d_in, n, cfg32, d_packed, d_scales, nullptr, nullptr,
                  nullptr);
  float tensor_scale = 1, global_scale = 1;
  const float amax = reduce_amax(d_in, n);
  if (cfg.scale_mode == ScaleMode::TENSOR)
    tensor_scale = amax > 0 ? amax / (cfg.format == QuantFormat::NVFP4
                                          ? 6.0f : 448.0f) : 1.0f;
  else if (cfg.format == QuantFormat::NVFP4)
    global_scale = amax > 2688.0f ? amax / 2688.0f : 1.0f;
  launch_dequantize(d_packed, n, d_scales, tensor_scale, global_scale, cfg32,
                    d_out, nullptr);

  std::vector<float> out(n);
  cudaMemcpy(out.data(), d_out, n * 4, cudaMemcpyDeviceToHost);
  cudaFree(d_in);
  cudaFree(d_packed);
  if (d_scales) cudaFree(d_scales);
  cudaFree(d_out);
  return out;
}

static void test_config(const char* name, const QuantConfig& cfg,
                        std::vector<float> in, float rel_tol) {
  const HostRef ref = host_reference(in, cfg);
  const std::vector<float> got = gpu_pipeline(in, cfg);
  CHECK(got.size() == in.size(), name);
  const bool exact = cfg.rounding == Rounding::NEAREST;
  float max_rel = 0;
  for (size_t i = 0; i < in.size(); ++i) {
    if (exact) {
      // GPU result must match the host reference bit-for-bit (same rounding
      // decisions; both run in fp32). Stochastic rounding is excluded: it
      // deliberately perturbs the input.
      CHECK(got[i] == ref.dq[i], name);
      if (got[i] != ref.dq[i]) {
        fprintf(stderr, "  mismatch at %zu: gpu=%f ref=%f\n", i, got[i],
                ref.dq[i]);
        break;
      }
    }
    const float denom = fabsf(in[i]) + 1e-6f;
    max_rel = fmaxf(max_rel, fabsf(in[i] - got[i]) / denom);
  }
  CHECK(max_rel <= rel_tol, name);
  printf("  %-34s max_rel_err=%.4f (bound %.2f)\n", name, max_rel, rel_tol);
}

static float dequant_at(const std::vector<float>&, const QuantConfig&,
                        size_t) {
  return 0;  // unused
}

int main() {
  std::mt19937 rng(42);
  std::normal_distribution<float> norm(0.0f, 1.0f);
  std::uniform_real_distribution<float> uni(-100.0f, 100.0f);

  std::vector<float> normal(4096), uniform(4096), outlier(4096);
  for (size_t i = 0; i < 4096; ++i) {
    normal[i] = norm(rng);
    uniform[i] = uni(rng);
    outlier[i] = norm(rng) * 0.01f;
  }
  // inject outliers into a few positions
  for (int k : {7, 100, 511, 1000, 2048, 3000, 4090})
    outlier[(size_t)k] = 500.0f * ((k % 2) ? 1 : -1);

  QuantConfig mxfp8 = [] {
    QuantConfig c; c.format = QuantFormat::MXFP8_E4M3; return c;
  }();
  QuantConfig mxfp8_t = mxfp8; mxfp8_t.scale_mode = ScaleMode::TENSOR;
  QuantConfig mxfp8_e5 = mxfp8; mxfp8_e5.format = QuantFormat::MXFP8_E5M2;
  QuantConfig nvfp4 = [] {
    QuantConfig c; c.format = QuantFormat::NVFP4; c.block_size = 16; return c;
  }();
  QuantConfig nvfp4_t = nvfp4; nvfp4_t.scale_mode = ScaleMode::TENSOR;
  QuantConfig nvfp4_s = nvfp4; nvfp4_s.rounding = Rounding::STOCHASTIC;

  printf("test_roundtrip:\n");
  test_config("mxfp8-e4m3 block/normal", mxfp8, normal, 0.2f);
  test_config("mxfp8-e4m3 tensor/uniform", mxfp8_t, uniform, 0.5f);
  test_config("mxfp8-e5m2 block/outlier", mxfp8_e5, outlier, 0.5f);
  test_config("nvfp4 block/normal", nvfp4, normal, 1.001f);
  test_config("nvfp4 tensor/normal", nvfp4_t, normal, 1.001f);
  test_config("nvfp4 block/outlier", nvfp4, outlier, 1.001f);
  test_config("nvfp4 block/uniform", nvfp4, uniform, 1.001f);
  test_config("nvfp4 stochastic/normal", nvfp4_s, normal, 20.0f);
  // odd-sized tail block (4097 elements -> ragged nvfp4 block)
  std::vector<float> odd(4097);
  for (auto& v : odd) v = norm(rng);
  test_config("nvfp4 block/odd-tail", nvfp4, odd, 1.001f);

  if (failures) {
    fprintf(stderr, "%d failure(s)\n", failures);
    return 1;
  }
  printf("all passed\n");
  return 0;
}
