// 硬件/库对照路径实现（训练营加分项）。
// 与 src/low_precision.cu 的纯软件位操作路径对照：
// 1. cvt 路径：E4M3 编解码改用 SM 原生转换指令（sm_89+）；
// 2. gemm 路径：调用 cuBLASLt 原生 FP8/FP4 块缩放 GEMM（Blackwell Tensor Core），
//    与「软件反量化 + cublasGemmEx FP16 GEMM」流水线对照。
#include "hw_path.cuh"

#ifdef LPQ_WITH_HW_PATH

#include <cuda_fp16.h>
#include <cuda_fp8.h>

#ifdef LPQ_HW_CUBLAS
#include <cublasLt.h>
#include <cublas_v2.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

namespace lpq::hw {
namespace {

constexpr int kThreads = 256;

// cvt 指令需要 sm_89+；低架构编译时 kernel 体留空，host 侧在入口检查设备能力。
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 890)
#define LPQ_HW_CVT_KERNEL 1
#endif

__global__ void mx_encode_cvt_kernel(const float* input,
                                     const std::uint8_t* scales,
                                     std::uint8_t* packed, std::size_t count,
                                     int block_size, bool tensor_mode) {
#ifdef LPQ_HW_CVT_KERNEL
  // 与软件 mx_quantize_kernel 相同的 grid-stride 结构和 scale 读取，唯一的
  // 区别是元素编码用一条 cvt.rn.satfinite.e4m3 指令代替二分查找。
  for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
       i += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
    const std::size_t scale_index = tensor_mode ? 0 : i / block_size;
    const float scale = decode_e8m0(scales[scale_index]);
    float value = input[i] / scale;
    // 软件编码器把 -0.0 视为非负数输出 0x00；硬件指令保留符号位输出 0x80。
    // 两者解码后数值相等，但为了逐字节等价，这里把 -0.0 归一化为 +0.0。
    if (value == 0.0f) value = 0.0f;
    packed[i] = __nv_cvt_float_to_fp8(value, __NV_SATFINITE, __NV_E4M3);
  }
#endif
}

__global__ void mx_decode_cvt_kernel(const std::uint8_t* packed,
                                     const std::uint8_t* scales, float* output,
                                     std::size_t count, int block_size,
                                     bool tensor_mode) {
#ifdef LPQ_HW_CVT_KERNEL
  for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
       i += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
    const std::size_t scale_index = tensor_mode ? 0 : i / block_size;
    // E4M3 的有限值在 half 中都精确可表示，half->float 也是精确转换，
    // 因此乘 scale 的结果与软件 decode_e4m3 路径位级一致。
    const __half_raw raw = __nv_cvt_fp8_to_halfraw(packed[i], __NV_E4M3);
    output[i] = __half2float(__half(raw)) * decode_e8m0(scales[scale_index]);
  }
#endif
}

inline int launch_blocks(std::size_t work) {
  const std::size_t blocks = (work + kThreads - 1) / kThreads;
  return static_cast<int>(blocks > 4096 ? 4096 : (blocks == 0 ? 1 : blocks));
}

}  // namespace

bool cvt_supported(const cudaDeviceProp& device) {
  // cvt.rn.satfinite.e4m3x2.f32 自 sm_89（Ada）起可用。
  return device.major > 8 || (device.major == 8 && device.minor >= 9);
}

cudaError_t launch_mx_encode_cvt(const float* input, const std::uint8_t* scales,
                                 std::uint8_t* packed, std::size_t count,
                                 int block_size, bool tensor_mode,
                                 cudaStream_t stream) {
  if (count == 0 || block_size <= 0 || input == nullptr || scales == nullptr ||
      packed == nullptr)
    return cudaErrorInvalidValue;
  mx_encode_cvt_kernel<<<launch_blocks(count), kThreads, 0, stream>>>(
      input, scales, packed, count, block_size, tensor_mode);
  return cudaGetLastError();
}

cudaError_t launch_mx_decode_cvt(const std::uint8_t* packed,
                                 const std::uint8_t* scales, float* output,
                                 std::size_t count, int block_size,
                                 bool tensor_mode, cudaStream_t stream) {
  if (count == 0 || block_size <= 0 || packed == nullptr || scales == nullptr ||
      output == nullptr)
    return cudaErrorInvalidValue;
  mx_decode_cvt_kernel<<<launch_blocks(count), kThreads, 0, stream>>>(
      packed, scales, output, count, block_size, tensor_mode);
  return cudaGetLastError();
}

#ifdef LPQ_HW_CUBLAS

namespace {

#define CUBLAS_CHECK(call)                                        \
  do {                                                            \
    const cublasStatus_t status__ = (call);                       \
    if (status__ != CUBLAS_STATUS_SUCCESS)                        \
      return cudaErrorUnknown;                                    \
  } while (0)

// 简单的 RAII device buffer。
class DeviceBuf {
 public:
  explicit DeviceBuf(std::size_t bytes) {
    if (bytes > 0 && cudaMalloc(&data_, bytes) != cudaSuccess) data_ = nullptr;
  }
  ~DeviceBuf() { if (data_) cudaFree(data_); }
  DeviceBuf(const DeviceBuf&) = delete;
  DeviceBuf& operator=(const DeviceBuf&) = delete;
  DeviceBuf(DeviceBuf&& other) noexcept : data_(other.data_) {
    other.data_ = nullptr;
  }
  DeviceBuf& operator=(DeviceBuf&& other) noexcept {
    if (this != &other) {
      if (data_) cudaFree(data_);
      data_ = other.data_;
      other.data_ = nullptr;
    }
    return *this;
  }
  void* get() const { return data_; }
  bool ok() const { return data_ != nullptr; }

 private:
  void* data_ = nullptr;
};

// cuBLASLt 原生低精度 GEMM 的上下文：descriptor、布局和已选中的算法。
// 计算 Y = X·Wᵀ（推理线性层形态），行大序 X(M×K)、W(N×K) 都按 K 连续存储，
// 在列大序的 cuBLASLt 中等价为 Cᵀ(N×M) = opA(A)·opB(B)，其中
// A = W 的缓冲区（存成 (K×N) 列大序，opA=T）、B = X 的缓冲区（opB=N）。
struct LtGemm {
  cublasLtHandle_t handle = nullptr;
  cublasLtMatmulDesc_t desc = nullptr;
  cublasLtMatrixLayout_t a = nullptr;
  cublasLtMatrixLayout_t b = nullptr;
  cublasLtMatrixLayout_t c = nullptr;
  cublasLtMatmulHeuristicResult_t heuristic{};
  cublasLtMatmulPreference_t pref = nullptr;
  bool ready = false;

  ~LtGemm() {
    if (pref) cublasLtMatmulPreferenceDestroy(pref);
    if (a) cublasLtMatrixLayoutDestroy(a);
    if (b) cublasLtMatrixLayoutDestroy(b);
    if (c) cublasLtMatrixLayoutDestroy(c);
    if (desc) cublasLtMatmulDescDestroy(desc);
    if (handle) cublasLtDestroy(handle);
  }
};

// 初始化 FP8/FP4 块缩放 GEMM；返回 false 表示 cuBLAS 不支持该组合。
bool setup_lt_gemm(LtGemm* g, cudaDataType ab_type,
                   cublasLtMatmulMatrixScale_t scale_mode, int m, int n, int k,
                   const void* a_scale, const void* b_scale,
                   std::size_t workspace_bytes) {
  if (cublasLtCreate(&g->handle) != CUBLAS_STATUS_SUCCESS) return false;
  if (cublasLtMatmulDescCreate(&g->desc, CUBLAS_COMPUTE_32F, CUDA_R_32F) !=
      CUBLAS_STATUS_SUCCESS)
    return false;
  const cublasOperation_t op_t = CUBLAS_OP_T;
  const cublasOperation_t op_n = CUBLAS_OP_N;
  cublasLtMatmulDescSetAttribute(g->desc, CUBLASLT_MATMUL_DESC_TRANSA, &op_t,
                                 sizeof(op_t));
  cublasLtMatmulDescSetAttribute(g->desc, CUBLASLT_MATMUL_DESC_TRANSB, &op_n,
                                 sizeof(op_n));
  // 块缩放：scale 缓冲区布局与数据矩阵一致（行大序 W/X 的每 32/16 个连续元素
  // 一个 scale，即对应列大序存储下沿 innermost 维的分块）。
  cublasLtMatmulDescSetAttribute(g->desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
                                 &a_scale, sizeof(a_scale));
  cublasLtMatmulDescSetAttribute(g->desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
                                 &b_scale, sizeof(b_scale));
  const int mode = static_cast<int>(scale_mode);
  cublasLtMatmulDescSetAttribute(g->desc, CUBLASLT_MATMUL_DESC_A_SCALE_MODE,
                                 &mode, sizeof(mode));
  cublasLtMatmulDescSetAttribute(g->desc, CUBLASLT_MATMUL_DESC_B_SCALE_MODE,
                                 &mode, sizeof(mode));
  // 布局按缓冲区在列大序下的存储形状描述：W(N×K) 行大序 = (K×N) 列大序。
  if (cublasLtMatrixLayoutCreate(&g->a, ab_type, k, n, k) !=
          CUBLAS_STATUS_SUCCESS ||
      cublasLtMatrixLayoutCreate(&g->b, ab_type, k, m, k) !=
          CUBLAS_STATUS_SUCCESS ||
      cublasLtMatrixLayoutCreate(&g->c, CUDA_R_16F, n, m, n) !=
          CUBLAS_STATUS_SUCCESS)
    return false;
  if (cublasLtMatmulPreferenceCreate(&g->pref) != CUBLAS_STATUS_SUCCESS)
    return false;
  cublasLtMatmulPreferenceSetAttribute(
      g->pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_bytes,
      sizeof(workspace_bytes));
  int found = 0;
  const cublasStatus_t st = cublasLtMatmulAlgoGetHeuristic(
      g->handle, g->desc, g->a, g->b, g->c, g->c, g->pref, 1, &g->heuristic,
      &found);
  g->ready = st == CUBLAS_STATUS_SUCCESS && found > 0;
  return g->ready;
}

cudaError_t run_lt_gemm(LtGemm* g, const void* a_data, const void* b_data,
                        void* c_data, float alpha, void* workspace,
                        std::size_t workspace_bytes, cudaStream_t stream) {
  const float beta = 0.0f;
  const cublasStatus_t st = cublasLtMatmul(
      g->handle, g->desc, &alpha, a_data, g->a, b_data, g->b, &beta, c_data,
      g->c, c_data, g->c, &g->heuristic.algo, workspace, workspace_bytes,
      stream);
  return st == CUBLAS_STATUS_SUCCESS ? cudaSuccess : cudaErrorUnknown;
}

// 软件基线：FP16 GEMM（cublasGemmEx），与 LtGemm 相同的 Y = X·Wᵀ 布局。
cudaError_t run_fp16_gemm(cublasHandle_t handle, const __half* w,
                          const __half* x, __half* c, int m, int n, int k,
                          cudaStream_t /*unused*/) {
  const float alpha = 1.0f;
  const float beta = 0.0f;
  // Cᵀ(N×M) = W(N×K)·Xᵀ(K×M)：cublas 列大序下同布局等价式。
  const cublasStatus_t st = cublasGemmEx(
      handle, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, &alpha, w, CUDA_R_16F, k, x,
      CUDA_R_16F, k, &beta, c, CUDA_R_16F, n, CUBLAS_COMPUTE_32F,
      CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  return st == CUBLAS_STATUS_SUCCESS ? cudaSuccess : cudaErrorUnknown;
}

// 事件计时工具：对 fn 计时 iterations 次取平均。
template <typename F>
cudaError_t time_loop(F&& fn, int warmup, int iterations, float* avg_ms) {
  for (int i = 0; i < warmup; ++i) fn();
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) return err;
  cudaEvent_t start, stop;
  if (cudaEventCreate(&start) != cudaSuccess) return cudaErrorUnknown;
  if (cudaEventCreate(&stop) != cudaSuccess) return cudaErrorUnknown;
  cudaEventRecord(start);
  for (int i = 0; i < iterations; ++i) fn();
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float total = 0.0f;
  cudaEventElapsedTime(&total, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  *avg_ms = total / iterations;
  return cudaGetLastError();
}

// 用软件路径把 side×side 的 FP32 输入量化成指定格式，返回量化缓冲区。
struct Quantized {
  DeviceBuf packed{0};
  DeviceBuf scales{0};
  DeviceBuf global{0};
  DeviceBuf scratch{0};
  std::size_t packed_size = 0;
  std::size_t scales_size = 0;
};

cudaError_t quantize_matrix(const float* d_input, std::size_t count,
                            const QuantConfig& config, Quantized* out) {
  out->packed_size = packed_bytes(count, config.format);
  out->scales_size = scale_count(count, config);
  out->packed = DeviceBuf(out->packed_size);
  out->scales = DeviceBuf(out->scales_size);
  out->global = DeviceBuf(sizeof(float));
  out->scratch = DeviceBuf(sizeof(float));
  if (!out->packed.ok() || !out->scales.ok() || !out->global.ok() ||
      !out->scratch.ok())
    return cudaErrorMemoryAllocation;
  return launch_quantize(d_input, InputType::kFp32,
                         static_cast<std::uint8_t*>(out->packed.get()),
                         static_cast<std::uint8_t*>(out->scales.get()),
                         static_cast<float*>(out->global.get()),
                         static_cast<float*>(out->scratch.get()), count, config,
                         nullptr);
}

// cuBLASLt 的 1D 块缩放要求 scale 按 128(outer)×4(inner) tile 布局存放
// （见 cuBLAS 文档 "1D Block Scaling Factors Layout"），与本项目的自然
// row-major (rows × k_blocks) 数组不同。这里在 host 侧做一次重排；该步骤属于
// 一次性 setup，不计入 GEMM 计时。
// natural[o * k_blocks + i] 表示第 o 行、第 i 个 K 向分块的 scale。
inline std::size_t round_up(std::size_t v, std::size_t m) {
  return (v + m - 1) / m * m;
}

std::vector<std::uint8_t> swizzle_scales(const std::vector<std::uint8_t>& natural,
                                         int rows, int k_blocks) {
  const int sf_inner_dim = static_cast<int>(round_up(k_blocks, 4));
  const int outer_padded = static_cast<int>(round_up(rows, 128));
  std::vector<std::uint8_t> out(
      static_cast<std::size_t>(sf_inner_dim) * outer_padded, 0);
  for (int o = 0; o < rows; ++o) {
    for (int i = 0; i < k_blocks; ++i) {
      const int tile_o = o / 128;
      const int tile_i = i / 4;
      const std::size_t in_tile = static_cast<std::size_t>(o % 32) * 16 +
                                  ((o % 128) / 32) * 4 + (i % 4);
      const std::size_t offset =
          (static_cast<std::size_t>(tile_i) * 4 +
           static_cast<std::size_t>(tile_o) * sf_inner_dim) * 128 +
          in_tile;
      out[offset] = natural[static_cast<std::size_t>(o) * k_blocks + i];
    }
  }
  return out;
}

// 把量化得到的自然布局 scale 拷贝回 host、重排成 tile 布局并重新上传。
cudaError_t prepare_swizzled_scales(const Quantized& q, int rows, int k_blocks,
                                    DeviceBuf* out) {
  std::vector<std::uint8_t> natural(q.scales_size);
  cudaError_t err = cudaMemcpy(natural.data(), q.scales.get(), q.scales_size,
                               cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) return err;
  const auto tiled = swizzle_scales(natural, rows, k_blocks);
  *out = DeviceBuf(tiled.size());
  if (!out->ok()) return cudaErrorMemoryAllocation;
  return cudaMemcpy(out->get(), tiled.data(), tiled.size(),
                    cudaMemcpyHostToDevice);
}

// 256³ 小规模自检：与 CPU double 参考对比，返回相对误差 ||C-Cref||/||Cref||。
// d_input/d_packed/d_scales 是已完成的量化结果（side×side）。
cudaError_t gemm_self_check(LtGemm* g, cublasHandle_t blas, int side,
                            Format format, const Quantized& qx,
                            void* workspace, std::size_t workspace_bytes,
                            double* hw_rel, double* sw_rel) {
  const std::size_t count = static_cast<std::size_t>(side) * side;
  DeviceBuf d_x16(count * sizeof(__half));
  DeviceBuf d_w16(count * sizeof(__half));
  DeviceBuf d_c_hw(count * sizeof(__half));
  DeviceBuf d_c_sw(count * sizeof(__half));
  if (!d_x16.ok() || !d_w16.ok() || !d_c_hw.ok() || !d_c_sw.ok())
    return cudaErrorMemoryAllocation;
  QuantConfig dq_config;
  dq_config.format = format;
  dq_config.block_size = format == Format::kMxfp8 ? 32 : 16;
  cudaError_t err = launch_dequantize(
      static_cast<const std::uint8_t*>(qx.packed.get()),
      static_cast<const std::uint8_t*>(qx.scales.get()),
      static_cast<const float*>(qx.global.get()), d_x16.get(), OutputType::kFp16,
      count, dq_config, nullptr);
  if (err != cudaSuccess) return err;
  err = cudaMemcpy(d_w16.get(), d_x16.get(), count * sizeof(__half),
                   cudaMemcpyDeviceToDevice);
  if (err != cudaSuccess) return err;
  err = run_fp16_gemm(blas, static_cast<const __half*>(d_w16.get()),
                      static_cast<const __half*>(d_x16.get()),
                      static_cast<__half*>(d_c_sw.get()), side, side, side,
                      nullptr);
  if (err != cudaSuccess) return err;
  // NVFP4 的 FP32 全局 scale 折进 alpha：D = (gs_x·gs_w)·op(A)·op(B)。
  float gs = 1.0f;
  err = cudaMemcpy(&gs, qx.global.get(), sizeof(float), cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) return err;
  const float alpha = format == Format::kNvfp4 ? gs * gs : 1.0f;
  err = run_lt_gemm(g, qx.packed.get(), qx.packed.get(), d_c_hw.get(), alpha,
                    workspace, workspace_bytes, nullptr);
  if (err != cudaSuccess) return err;
  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) return err;

  std::vector<__half> c_hw(count), c_sw(count);
  cudaMemcpy(c_hw.data(), d_c_hw.get(), count * sizeof(__half),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(c_sw.data(), d_c_sw.get(), count * sizeof(__half),
             cudaMemcpyDeviceToHost);
  // CPU 参考直接从量化文件内容反量化到 double 再乘加，与两条 GPU 路径解耦。
  std::vector<std::uint8_t> packed(qx.packed_size), scales(qx.scales_size);
  cudaMemcpy(packed.data(), qx.packed.get(), qx.packed_size,
             cudaMemcpyDeviceToHost);
  cudaMemcpy(scales.data(), qx.scales.get(), qx.scales_size,
             cudaMemcpyDeviceToHost);
  auto dequant = [&](std::size_t i) {
    const std::size_t block = i / dq_config.block_size;
    if (format == Format::kMxfp8)
      return static_cast<double>(decode_e4m3(packed[i])) *
             decode_e8m0(scales[block]);
    const std::uint8_t byte = packed[i / 2];
    const std::uint8_t nibble =
        (i & 1) == 0 ? (byte & 0x0f) : ((byte >> 4) & 0x0f);
    return static_cast<double>(decode_e2m1(nibble)) *
           decode_e4m3(scales[block]) * gs;
  };
  long double hw_err = 0.0L, sw_err = 0.0L, ref_norm = 0.0L;
  for (int i = 0; i < side; ++i) {
    for (int j = 0; j < side; ++j) {
      double ref = 0.0;
      for (int kk = 0; kk < side; ++kk)
        ref += dequant(static_cast<std::size_t>(i) * side + kk) *
               dequant(static_cast<std::size_t>(j) * side + kk);
      const std::size_t idx = static_cast<std::size_t>(i) * side + j;
      hw_err += std::pow(__half2float(c_hw[idx]) - ref, 2);
      sw_err += std::pow(__half2float(c_sw[idx]) - ref, 2);
      ref_norm += ref * ref;
    }
  }
  *hw_rel = std::sqrt(static_cast<double>(hw_err / ref_norm));
  *sw_rel = std::sqrt(static_cast<double>(sw_err / ref_norm));
  return cudaSuccess;
}

}  // namespace

cudaError_t run_gemm_comparison(const float* d_input, int side, Format format,
                                int warmup, int iterations,
                                GemmBenchResult* result) {
  const std::size_t count = static_cast<std::size_t>(side) * side;
  QuantConfig config;
  config.format = format;
  config.block_size = format == Format::kMxfp8 ? 32 : 16;
  result->side = side;

  // 两种对照共用同一份软件量化结果；W 与 X 使用同一矩阵以控制显存占用。
  Quantized qx;
  cudaError_t err = quantize_matrix(d_input, count, config, &qx);
  if (err != cudaSuccess) return err;
  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) return err;

  const std::size_t workspace_bytes = 32u << 20;
  DeviceBuf workspace(workspace_bytes);
  if (!workspace.ok()) return cudaErrorMemoryAllocation;

  // 运行时探测 cuBLAS 是否支持该类型/缩放组合；不支持则如实返回。
  const cudaDataType ab_type =
      format == Format::kMxfp8 ? CUDA_R_8F_E4M3 : CUDA_R_4F_E2M1;
  const cublasLtMatmulMatrixScale_t scale_mode =
      format == Format::kMxfp8 ? CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0
                               : CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
  // 把自然 row-major 的 scale 重排为 cuBLASLt 要求的 128×4 tile 布局。
  DeviceBuf tiled_scales(0);
  err = prepare_swizzled_scales(qx, side, side / config.block_size,
                                &tiled_scales);
  if (err != cudaSuccess) return err;
  LtGemm lt;
  result->hw_supported =
      setup_lt_gemm(&lt, ab_type, scale_mode, side, side, side,
                    tiled_scales.get(), tiled_scales.get(), workspace_bytes);
  result->hw_impl = format == Format::kMxfp8
                        ? "cuBLASLt FP8 E4M3 + VEC32_UE8M0 block scale"
                        : "cuBLASLt FP4 E2M1 + VEC16_UE4M3 block scale";
  if (!result->hw_supported) return cudaSuccess;

  cublasHandle_t blas;
  if (cublasCreate(&blas) != CUBLAS_STATUS_SUCCESS) return cudaErrorUnknown;

  // 小规模数值自检（256³，与 CPU double 参考对比）。
  {
    DeviceBuf small_input(static_cast<std::size_t>(256) * 256 * sizeof(float));
    // 取输入矩阵左上角 256×256 的确定性切片。
    std::vector<float> host(256 * 256);
    for (int r = 0; r < 256; ++r) {
      std::vector<float> row(256);
      cudaMemcpy(row.data(), d_input + static_cast<std::size_t>(r) * side,
                 256 * sizeof(float), cudaMemcpyDeviceToHost);
      std::memcpy(host.data() + static_cast<std::size_t>(r) * 256, row.data(),
                  256 * sizeof(float));
    }
    cudaMemcpy(small_input.get(), host.data(), host.size() * sizeof(float),
               cudaMemcpyHostToDevice);
    Quantized qs;
    err = quantize_matrix(static_cast<const float*>(small_input.get()),
                          host.size(), config, &qs);
    if (err == cudaSuccess) err = cudaDeviceSynchronize();
    if (err == cudaSuccess) {
      DeviceBuf small_scales(0);
      err = prepare_swizzled_scales(qs, 256, 256 / config.block_size,
                                    &small_scales);
      if (err == cudaSuccess) {
        LtGemm lt_small;
        if (setup_lt_gemm(&lt_small, ab_type, scale_mode, 256, 256, 256,
                          small_scales.get(), small_scales.get(),
                          workspace_bytes)) {
          err = gemm_self_check(&lt_small, blas, 256, format, qs,
                                workspace.get(), workspace_bytes,
                                &result->self_check_hw_rel,
                                &result->self_check_sw_rel);
        }
      }
    }
    if (err != cudaSuccess) {
      cublasDestroy(blas);
      return err;
    }
  }

  // 基准尺寸的软件基线缓冲区。
  DeviceBuf d_x16(count * sizeof(__half));
  DeviceBuf d_w16(count * sizeof(__half));
  DeviceBuf d_c_hw(count * sizeof(__half));
  DeviceBuf d_c_sw(count * sizeof(__half));
  if (!d_x16.ok() || !d_w16.ok() || !d_c_hw.ok() || !d_c_sw.ok()) {
    cublasDestroy(blas);
    return cudaErrorMemoryAllocation;
  }

  float gs = 1.0f;
  cudaMemcpy(&gs, qx.global.get(), sizeof(float), cudaMemcpyDeviceToHost);
  const float alpha = format == Format::kNvfp4 ? gs * gs : 1.0f;

  QuantConfig dq_config = config;
  auto dequant_both = [&] {
    launch_dequantize(static_cast<const std::uint8_t*>(qx.packed.get()),
                      static_cast<const std::uint8_t*>(qx.scales.get()),
                      static_cast<const float*>(qx.global.get()), d_x16.get(),
                      OutputType::kFp16, count, dq_config, nullptr);
    launch_dequantize(static_cast<const std::uint8_t*>(qx.packed.get()),
                      static_cast<const std::uint8_t*>(qx.scales.get()),
                      static_cast<const float*>(qx.global.get()), d_w16.get(),
                      OutputType::kFp16, count, dq_config, nullptr);
  };
  auto hw_gemm = [&] {
    run_lt_gemm(&lt, qx.packed.get(), qx.packed.get(), d_c_hw.get(), alpha,
                workspace.get(), workspace_bytes, nullptr);
  };
  auto sw_gemm = [&] {
    run_fp16_gemm(blas, static_cast<const __half*>(d_w16.get()),
                  static_cast<const __half*>(d_x16.get()),
                  static_cast<__half*>(d_c_sw.get()), side, side, side,
                  nullptr);
  };

  float ms = 0.0f;
  err = time_loop(hw_gemm, warmup, iterations, &ms);
  if (err == cudaSuccess) result->hw_gemm_ms = ms;
  if (err == cudaSuccess) err = time_loop(dequant_both, warmup, iterations, &ms);
  if (err == cudaSuccess) {
    result->sw_dequant_ms = ms;
    err = time_loop(sw_gemm, warmup, iterations, &ms);
  }
  if (err == cudaSuccess) result->sw_gemm_ms = ms;
  if (err != cudaSuccess) {
    cublasDestroy(blas);
    return err;
  }

  const double flops = 2.0 * side * side * static_cast<double>(side);
  result->hw_tflops = flops / (result->hw_gemm_ms * 1.0e-3) * 1.0e-12;
  result->sw_tflops = flops / (result->sw_gemm_ms * 1.0e-3) * 1.0e-12;

  // 基准尺寸上两条路径输出的直接对比（X/W 相同，C 应为同一矩阵）。
  hw_gemm();
  sw_gemm();
  err = cudaDeviceSynchronize();
  if (err == cudaSuccess) {
    std::vector<__half> c_hw(count), c_sw(count);
    cudaMemcpy(c_hw.data(), d_c_hw.get(), count * sizeof(__half),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(c_sw.data(), d_c_sw.get(), count * sizeof(__half),
               cudaMemcpyDeviceToHost);
    double max_abs = 0.0, max_ref = 0.0;
    for (std::size_t i = 0; i < count; ++i) {
      const double diff = std::abs(
          static_cast<double>(__half2float(c_hw[i])) - __half2float(c_sw[i]));
      max_abs = std::max(max_abs, diff);
      max_ref =
          std::max(max_ref, std::abs(static_cast<double>(__half2float(c_sw[i]))));
    }
    result->bench_hw_vs_sw_max_abs = max_abs;
    result->bench_hw_vs_sw_max_rel = max_ref > 0.0 ? max_abs / max_ref : 0.0;
  }
  cublasDestroy(blas);
  return cudaSuccess;
}

#endif  // LPQ_HW_CUBLAS

}  // namespace lpq::hw

#endif  // LPQ_WITH_HW_PATH
