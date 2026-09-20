// ============================================================================
// 选题二「MXFP8 / NVFP4 低精度模拟与反量化」—— 库实现（GPU 侧）
// ============================================================================
// 本文件是纯库实现：不含 main、不含命令行解析、不产出可执行文件。
// 使用方是 test/test_all.cpp；公开接口见 include/quant_gpu.h。
//
// 硬件约束（题目硬性要求）：
//   - 不依赖 Hopper/Blackwell/Ampere+ 特性；无 FP8/FP4 Tensor Core；
//   - grid-stride + warp shuffle + atomicMax(ull) 都是 sm_35 即有的原语。
// ============================================================================

// 本文件是唯一含 kernel 的 TU，也是唯一「定义」__constant__ 设备表的 TU；
// 其它 TU 只看到 extern 声明（见 include/quant_tables.h 顶部说明）。
#define QC_DEVICE_TABLES

#include "quant_gpu.h"

#include <fstream>
#include <iomanip>
#include <sstream>

#define KLAUNCH(k, grid, block, ...) k<<<grid, block>>>(__VA_ARGS__)

// ============================================================================
// 1. Kernels
// ============================================================================

// 全局 amax。fabs 结果非负，非负 double 的位模式可用无符号整数比较大小
// （IEEE754 阶码在高位、单调），因此归约用 ULL max / atomicMax。
__global__ void k_amax(const float *__restrict__ w, int64_t n,
                       unsigned long long *__restrict__ out) {
    __shared__ unsigned long long sdata[256];
    double a = 0.0;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride)
        a = fmax(a, fabs((double)w[i]));
    sdata[threadIdx.x] = (unsigned long long)__double_as_longlong(a);
    __syncthreads();
    for (unsigned s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s && sdata[threadIdx.x + s] > sdata[threadIdx.x])
            sdata[threadIdx.x] = sdata[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0)
        atomicMax(out, sdata[0]);
}

// MXFP8 block 量化：每 warp 负责一个 32 元素块（grid-stride）。
// 相邻线程读相邻元素 → 合并访问；块内 amax 用 warp shuffle 归约（两遍法的
// 两遍合并进一个 kernel：32 个元素正好放进一个 warp 的寄存器）。
__global__ void k_mxfp8_quant_block(const float *__restrict__ w, int64_t n, int p, int fmt_id,
                                    int mode, uint64_t seed, uint8_t *__restrict__ scales,
                                    uint8_t *__restrict__ codes) {
    int64_t num_groups = (n + 31) / 32;
    int64_t warps_total = ((int64_t)gridDim.x * blockDim.x) >> 5;
    int64_t warp_id = ((int64_t)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    for (int64_t g = warp_id; g < num_groups; g += warps_total) {
        int64_t i = g * 32 + lane;
        double v = i < n ? (double)w[i] : 0.0; // 尾块补零
        double a = fabs(v);
#pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            a = fmax(a, __shfl_xor_sync(0xffffffffu, a, off));
        // 归约后所有 lane 持有相同 amax → 各自计算指数/缩放，无需广播
        long long e = mx_scale_exponent(a, p);
        if (lane == 0)
            scales[g] = (uint8_t)(e + 127);
        double scale = ldexp(1.0, (int)e);
        codes[i] = fmt_encode(fmt_id, v / scale, mode, elem_rng(seed, i));
    }
}

// MXFP8 tensor 量化：整张量一个 E8M0 缩放（scale 由主机从 amax 算好传入）。
__global__ void k_mxfp8_quant_tensor(const float *__restrict__ w, int64_t n, int fmt_id, int mode,
                                     uint64_t seed, double scale, uint8_t *__restrict__ codes) {
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride)
        codes[i] = fmt_encode(fmt_id, (double)w[i] / scale, mode, elem_rng(seed, i));
}

// NVFP4 block 量化：每 half-warp(16 lane) 负责一个 16 元素块。
// 流程：块 amax → ideal = amax/(6·gs) → 向上取整到 E4M3 码点
// → 逐元素编码 E2M1 → 偶数 lane 合成打包字节（低 4bit = 偶数下标元素）。
__global__ void k_nvfp4_quant_block(const float *__restrict__ w, int64_t n, float gs_f, int mode,
                                    uint64_t seed, uint8_t *__restrict__ scales,
                                    uint8_t *__restrict__ packed) {
    int64_t num_groups = (n + 15) / 16;
    int64_t hws_total = ((int64_t)gridDim.x * blockDim.x) >> 4;
    int64_t hw = ((int64_t)blockIdx.x * blockDim.x + threadIdx.x) >> 4;
    int lane = threadIdx.x & 15;
    unsigned mask = ((threadIdx.x & 31) < 16) ? 0xFFFFu : 0xFFFF0000u;
    double gs = (double)gs_f;
    for (int64_t g = hw; g < num_groups; g += hws_total) {
        int64_t i = g * 16 + lane;
        double v = i < n ? (double)w[i] : 0.0;
        double a = fabs(v);
#pragma unroll
        for (int off = 8; off > 0; off >>= 1)
            a = fmax(a, __shfl_xor_sync(mask, a, off));
        double ideal = a / (6.0 * gs);
        int s_idx = round_up_idx(e4m3_mags(), e4m3_mags_k(), ideal);
        if (lane == 0)
            scales[g] = (uint8_t)s_idx;
        double denom = e4m3_mags()[s_idx] * gs;
        if (!(denom > 0.0))
            denom = 1.0; // 零块防 0/0
        uint8_t code = fmt_encode(FMT_E2M1, v / denom, mode, elem_rng(seed, i));
        // __shfl 不支持 8bit 类型 → 转 unsigned 再 shuffle（取相邻奇数 lane 的码点）
        unsigned other = __shfl_xor_sync(mask, (unsigned)code, 1);
        if ((lane & 1) == 0)
            packed[g * 8 + (lane >> 1)] = (uint8_t)(code | (other << 4));
    }
}

// NVFP4 tensor 量化：每线程 2 个元素 → 1 个打包字节。
__global__ void k_nvfp4_quant_tensor(const float *__restrict__ w, int64_t n, double gs, int mode,
                                     uint64_t seed, uint8_t *__restrict__ packed) {
    int64_t nbytes = (n + 1) / 2;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for (int64_t b = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; b < nbytes; b += stride) {
        int64_t i0 = b * 2, i1 = b * 2 + 1;
        double v0 = i0 < n ? (double)w[i0] : 0.0;
        double v1 = i1 < n ? (double)w[i1] : 0.0;
        uint8_t c0 = fmt_encode(FMT_E2M1, v0 / gs, mode, elem_rng(seed, i0));
        uint8_t c1 = fmt_encode(FMT_E2M1, v1 / gs, mode, elem_rng(seed, i1));
        packed[b] = (uint8_t)(c0 | (c1 << 4));
    }
}

// 输出写入：out_type ∈ {OUT_FP32, OUT_FP16, OUT_BF16}；fp16/bf16 存位模式
__device__ inline void write_out(void *out, int64_t i, double r, int out_type) {
    switch (out_type) {
    case OUT_FP32:
        ((float *)out)[i] = (float)r;
        break;
    case OUT_FP16:
        ((uint16_t *)out)[i] = f64_to_f16_bits(r);
        break;
    default:
        ((uint16_t *)out)[i] = f32_to_bf16_bits((float)r);
        break;
    }
}

// MXFP8 反量化：逐元素查表 × E8M0 缩放。tensor_mode 时 scale 恒为 scales[0]。
__global__ void k_mxfp8_dequant(const uint8_t *__restrict__ codes,
                                const uint8_t *__restrict__ scales, int bs, int tensor_mode,
                                int64_t n, int fmt_id, int out_type, void *__restrict__ out) {
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        int64_t sb_idx = tensor_mode ? 0 : i / bs;
        double scale = ldexp(1.0, (int)__ldg(&scales[sb_idx]) - 127);
        write_out(out, i, fmt_decode(fmt_id, __ldg(&codes[i])) * scale, out_type);
    }
}

// NVFP4 反量化：逐元素解包（低 4bit = 偶数下标）→ E2M1 查表
// × s_block(E4M3) × s_global(FP32)。乘法顺序固定为 (v × s_b) × gs，与主机端一致。
__global__ void k_nvfp4_dequant(const uint8_t *__restrict__ packed,
                                const uint8_t *__restrict__ scales, int bs, int tensor_mode,
                                int64_t n, float gs_f, int out_type, void *__restrict__ out) {
    double gs = (double)gs_f;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        uint8_t byte = __ldg(&packed[i >> 1]);
        uint8_t code = (i & 1) ? (uint8_t)(byte >> 4) : (uint8_t)(byte & 0x0F);
        int64_t sb_idx = tensor_mode ? 0 : i / bs;
        double s_b = fmt_decode(FMT_E4M3, __ldg(&scales[sb_idx]));
        double t = fmt_decode(FMT_E2M1, code) * s_b;
        write_out(out, i, t * gs, out_type);
    }
}

// ============================================================================
// 2. CUDA 基础设施（缓冲 RAII / 计时 / 网格尺寸）
// ============================================================================

struct DevBuf {
    void *p = nullptr;
    void alloc(size_t bytes) { CUDA_CHECK(cudaMalloc(&p, bytes)); }
    ~DevBuf() {
        if (p)
            cudaFree(p);
    }
};

struct Timer {
    cudaEvent_t s = nullptr, e = nullptr;
    Timer() {
        CUDA_CHECK(cudaEventCreate(&s));
        CUDA_CHECK(cudaEventCreate(&e));
    }
    ~Timer() {
        cudaEventDestroy(s);
        cudaEventDestroy(e);
    }
    void begin() { CUDA_CHECK(cudaEventRecord(s)); }
    float end_ms() { // record + sync + elapsed：测的是纯 GPU 时间
        CUDA_CHECK(cudaEventRecord(e));
        CUDA_CHECK(cudaEventSynchronize(e));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, s, e));
        return ms;
    }
};

static int g_sm_count = 0;

void gpu_init(int device) {
    CUDA_CHECK(cudaSetDevice(device));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    g_sm_count = prop.multiProcessorCount;
}

// 1D 网格：256 线程/块，块数封顶 SM 数 × 16（kernel 内部 grid-stride 兜底）
static dim3 grid_1d(int64_t threads_needed) {
    int64_t blocks = (threads_needed + 255) / 256;
    int64_t cap = (int64_t)g_sm_count * 16;
    if (blocks > cap)
        blocks = cap;
    if (blocks < 1)
        blocks = 1;
    if (blocks > 0x7FFFFFFFLL)
        blocks = 0x7FFFFFFFLL;
    return dim3((unsigned)blocks, 1, 1);
}

// ============================================================================
// 3. GPU 量化驱动（结果 + 计时 + 与主机端实现的逐字节对拍）
// ============================================================================

GpuQuantResult quantize_gpu(const std::vector<float> &w, const Config &cfg, int repeat,
                                   bool verify) {
    int64_t n = (int64_t)w.size();
    int mode = cfg.rounding == "stochastic" ? MODE_STOCHASTIC : MODE_NEAREST;
    int elem_id = cfg.elem_format == "e5m2" ? FMT_E5M2 : FMT_E4M3;
    int p = elem_id == FMT_E4M3 ? 3 : 2;
    uint64_t seed = (uint64_t)cfg.seed;

    GpuQuantResult res;
    Quantized &q = res.q;
    q.format = cfg.format;
    q.elem_id = cfg.format == "mxfp8" ? elem_id : FMT_E2M1;
    q.scale_mode = cfg.scale_mode;
    q.num_elems = n;

    DevBuf d_w, d_scales, d_packed, d_amax;
    d_w.alloc(4 * (size_t)n);
    CUDA_CHECK(cudaMemcpy(d_w.p, w.data(), 4 * (size_t)n, cudaMemcpyHostToDevice));

    Timer t;
    double amax_ms = 0.0;
    double amax = 0.0; // 全局 amax（tensor 模式 / nvfp4 需要）

    auto run_amax = [&]() {
        unsigned long long zero = 0;
        CUDA_CHECK(cudaMemcpy(d_amax.p, &zero, 8, cudaMemcpyHostToDevice));
        KLAUNCH(k_amax, grid_1d(n), dim3(256), (const float *)d_w.p, n,
                (unsigned long long *)d_amax.p);
        unsigned long long bits = 0;
        CUDA_CHECK(cudaMemcpy(&bits, d_amax.p, 8, cudaMemcpyDeviceToHost));
        std::memcpy(&amax, &bits, 8);
        CUDA_CHECK(cudaGetLastError());
    };

    // 计时辅助：warmup 一次 + repeat 次取最小
    auto timed = [&](int64_t grid_threads, auto &&launch) {
        launch(grid_1d(grid_threads));
        CUDA_CHECK(cudaDeviceSynchronize());
        double best = 1e300;
        for (int r = 0; r < repeat; r++) {
            t.begin();
            launch(grid_1d(grid_threads));
            float ms = t.end_ms();
            if (ms < best)
                best = ms;
        }
        return best;
    };

    if (cfg.format == "mxfp8") {
        if (cfg.scale_mode == "block") {
            int64_t nblk = (n + cfg.block_size - 1) / cfg.block_size;
            int64_t padded = nblk * cfg.block_size;
            q.block_size = cfg.block_size;
            q.num_blocks = nblk;
            q.global_scale = 1.0f;
            q.scale_bytes.resize((size_t)nblk);
            q.packed.resize((size_t)padded);
            d_scales.alloc((size_t)nblk);
            d_packed.alloc((size_t)padded);
            res.quantize_ms = timed(nblk * 32, [&](dim3 g) {
                KLAUNCH(k_mxfp8_quant_block, g, dim3(256), (const float *)d_w.p, n, p, elem_id,
                        mode, seed, (uint8_t *)d_scales.p, (uint8_t *)d_packed.p);
            });
        } else { // tensor
            d_amax.alloc(8);
            t.begin();
            run_amax();
            amax_ms = t.end_ms();
            long long e = mx_scale_exponent(amax, p);
            q.num_blocks = 1;
            q.global_scale = 1.0f;
            q.scale_bytes.assign(1, (uint8_t)(e + 127));
            double scale = ldexp(1.0, (int)e);
            q.packed.resize((size_t)n);
            d_packed.alloc((size_t)n);
            res.quantize_ms = timed(n, [&](dim3 g) {
                KLAUNCH(k_mxfp8_quant_tensor, g, dim3(256), (const float *)d_w.p, n, elem_id, mode,
                        seed, scale, (uint8_t *)d_packed.p);
            });
        }
    } else { // nvfp4
        d_amax.alloc(8);
        t.begin();
        run_amax();
        amax_ms = t.end_ms();
        if (cfg.scale_mode == "block") {
            int64_t nblk = (n + cfg.block_size - 1) / cfg.block_size;
            int64_t padded = nblk * cfg.block_size;
            q.global_scale = amax == 0.0 ? 1.0f : (float)(amax / (E4M3_MAX * E2M1_MAX));
            q.block_size = cfg.block_size;
            q.num_blocks = nblk;
            q.scale_bytes.resize((size_t)nblk);
            q.packed.resize((size_t)(padded / 2));
            d_scales.alloc((size_t)nblk);
            d_packed.alloc((size_t)(padded / 2));
            float gs = q.global_scale;
            res.quantize_ms = timed(nblk * 16, [&](dim3 g) {
                KLAUNCH(k_nvfp4_quant_block, g, dim3(256), (const float *)d_w.p, n, gs, mode, seed,
                        (uint8_t *)d_scales.p, (uint8_t *)d_packed.p);
            });
        } else { // tensor
            q.global_scale = amax == 0.0 ? 1.0f : (float)(amax / E2M1_MAX);
            q.num_blocks = 1;
            q.scale_bytes.assign(1, 0x38);
            q.packed.resize((size_t)((n + 1) / 2));
            d_packed.alloc((size_t)((n + 1) / 2));
            double gs = q.global_scale;
            res.quantize_ms = timed((n + 1) / 2, [&](dim3 g) {
                KLAUNCH(k_nvfp4_quant_tensor, g, dim3(256), (const float *)d_w.p, n, gs, mode, seed,
                        (uint8_t *)d_packed.p);
            });
        }
    }
    res.quantize_ms += amax_ms;

    // 拷回量化结果。
    // scale 只在 block 模式下由设备产生（tensor 模式的缩放因子是主机算好后直接填进
    // q.scale_bytes 的，设备侧没有对应缓冲），所以判据必须是「设备缓冲是否分配」，
    // 而不是「scale_bytes 是否非空」—— 后者在 tensor 模式下成立，但 d_scales.p 是
    // 空指针，cudaMemcpy 会以 invalid argument 失败。
    if (d_scales.p)
        CUDA_CHECK(cudaMemcpy(q.scale_bytes.data(), d_scales.p, q.scale_bytes.size(),
                              cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(q.packed.data(), d_packed.p, q.packed.size(), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaGetLastError());

    // verify：与主机端实现逐字节对拍（nearest 与 stochastic 均应一致，
    // 两侧共用同一套逐元素确定性随机数 splitmix64(seed, idx)）
    if (verify) {
        Quantized hq = quantize_host(w, cfg);
        res.verify_match = hq.packed == q.packed && hq.scale_bytes == q.scale_bytes &&
                           hq.num_blocks == q.num_blocks && hq.global_scale == q.global_scale;
    }
    return res;
}

// ============================================================================
// 4. GPU 反量化驱动（计时 + 与主机端实现对拍）
// ============================================================================

GpuDequantResult dequant_gpu(const Quantized &q, int oti, int repeat, bool verify) {
    int64_t n = q.num_elems;
    size_t obytes = out_type_bytes(oti);
    int tensor_mode = q.scale_bytes.size() == 1 ? 1 : 0;
    int bs = q.block_size;

    DevBuf d_packed, d_scales, d_out;
    d_packed.alloc(q.packed.size());
    d_scales.alloc(q.scale_bytes.size());
    d_out.alloc(obytes * (size_t)n);
    CUDA_CHECK(cudaMemcpy(d_packed.p, q.packed.data(), q.packed.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(
        cudaMemcpy(d_scales.p, q.scale_bytes.data(), q.scale_bytes.size(), cudaMemcpyHostToDevice));

    Timer t;
    auto launch = [&](dim3 g) {
        if (q.format == "mxfp8") {
            KLAUNCH(k_mxfp8_dequant, g, dim3(256), (const uint8_t *)d_packed.p,
                    (const uint8_t *)d_scales.p, bs, tensor_mode, n, q.elem_id, oti, d_out.p);
        } else {
            KLAUNCH(k_nvfp4_dequant, g, dim3(256), (const uint8_t *)d_packed.p,
                    (const uint8_t *)d_scales.p, bs, tensor_mode, n, q.global_scale, oti, d_out.p);
        }
    };
    launch(grid_1d(n));
    CUDA_CHECK(cudaDeviceSynchronize());
    GpuDequantResult res;
    double best = 1e300;
    for (int r = 0; r < repeat; r++) {
        t.begin();
        launch(grid_1d(n));
        float ms = t.end_ms();
        if (ms < best)
            best = ms;
    }
    res.dequant_ms = best;
    CUDA_CHECK(cudaGetLastError());

    // 拷回并解码成 FP32（位模式 → float 为精确操作）
    res.out.resize((size_t)n);
    std::vector<uint8_t> raw(obytes * (size_t)n);
    CUDA_CHECK(cudaMemcpy(raw.data(), d_out.p, raw.size(), cudaMemcpyDeviceToHost));
    if (oti == OUT_FP32) {
        std::memcpy(res.out.data(), raw.data(), raw.size());
    } else {
        for (int64_t i = 0; i < n; i++) {
            uint16_t h = (uint16_t)(raw[2 * i] | (raw[2 * i + 1] << 8));
            res.out[(size_t)i] = oti == OUT_FP16 ? f16_bits_to_f32(h) : bf16_bits_to_f32(h);
        }
    }

    if (verify) {
        std::vector<float> hout = cast_output(dequantize_host(q), oti);
        res.verify_match = hout == res.out;
    }
    return res;
}

// ============================================================================
// 5. 端到端：单个（格式 × 矩阵）组合
// ============================================================================

static std::string fmt_g(double v) {
    std::ostringstream os;
    os << std::setprecision(17) << v;
    return os.str();
}

int run_one(const std::string &input_path, const Config &cfg, const std::string &outdir, int repeat,
            bool verify, RunMetrics *row) {
    if (g_sm_count <= 0)
        gpu_init(); // 网格尺寸依赖 SM 数，未初始化时兜底

    TensorFile tf = read_tensor(input_path);
    qc_validate_finite(tf.data);
    std::vector<double> w(tf.data.begin(), tf.data.end());

    GpuQuantResult gq = quantize_gpu(tf.data, cfg, repeat, verify);
    GpuDequantResult gd = dequant_gpu(gq.q, out_type_id_of(cfg), repeat, verify);

    // 输出目录: outdir/<输入名去扩展名>/
    std::string stem = input_path;
    size_t slash = stem.find_last_of('/');
    if (slash != std::string::npos)
        stem = stem.substr(slash + 1);
    size_t dot = stem.find_last_of('.');
    if (dot != std::string::npos)
        stem = stem.substr(0, dot);
    std::string odir = outdir + "/" + stem;
    qc_mkdir(odir);

    write_quantized(odir + "/weights.bin", gq.q, tf.rows, tf.cols);
    write_tensor(odir + "/dequant.bin", tf.rows, tf.cols, cfg.output_type, gd.out);

    ErrStats st = error_stats(w, gd.out);
    int src_bits = tf.dtype == "fp32" ? 32 : 16;
    int gs_bits = gq.q.format == "nvfp4" ? 32 : 0;
    CompStats cp = compression_stats(src_bits, gq.q.num_elems, gq.q.packed.size(),
                                     gq.q.scale_bytes.size(), gs_bits);
    // 有效带宽（题目口径）：kernel 实际搬运的字节 / 时间
    double quant_bytes = 4.0 * (double)gq.q.num_elems + (double)gq.q.packed.size() +
                         (double)gq.q.scale_bytes.size();
    double dequant_bytes = (double)gq.q.packed.size() + (double)gq.q.scale_bytes.size() +
                           (double)gq.q.num_elems * (double)out_type_bytes(out_type_id_of(cfg));
    double quant_gbps = quant_bytes / (gq.quantize_ms * 1e-3) / 1e9;
    double dequant_gbps = dequant_bytes / (gd.dequant_ms * 1e-3) / 1e9;

    {
        std::ofstream f(odir + "/metrics.json");
        f << std::setprecision(17);
        f << "{\n"
          << "  \"input\": \"" << stem << "\",\n"
          << "  \"input_dtype\": \"" << tf.dtype << "\",\n"
          << "  \"num_elements\": " << gq.q.num_elems << ",\n"
          << "  \"config\": {\"format\": \"" << cfg.format
          << "\", \"block_size\": " << cfg.block_size << ", \"scale_mode\": \"" << cfg.scale_mode
          << "\", \"output_type\": \"" << cfg.output_type << "\", \"rounding\": \"" << cfg.rounding
          << "\", \"target_gpu\": \"" << cfg.target_gpu << "\", \"elem_format\": \""
          << cfg.elem_format << "\", \"seed\": " << cfg.seed << "},\n"
          << "  \"error\": {\"max_abs_err\": " << fmt_g(st.max_abs) << ", \"mae\": " << fmt_g(st.mae)
          << ", \"mse\": " << fmt_g(st.mse) << ", \"rmse\": " << fmt_g(st.rmse) << "},\n"
          << "  \"compression\": {\"src_bits_per_elem\": " << cp.src_bits
          << ", \"dst_bits_per_elem\": " << fmt_g(cp.dst_bits)
          << ", \"compression_ratio\": " << fmt_g(cp.ratio) << "},\n"
          << "  \"verify\": {\"quantize_bytes_match\": " << (gq.verify_match ? "true" : "false")
          << ", \"dequant_bytes_match\": " << (gd.verify_match ? "true" : "false") << "},\n"
          << "  \"quantize_kernel_ms\": " << fmt_g(gq.quantize_ms) << ",\n"
          << "  \"dequantize_kernel_ms\": " << fmt_g(gd.dequant_ms) << ",\n"
          << "  \"effective_bandwidth_gbps\": " << fmt_g(dequant_gbps) << ",\n"
          << "  \"quantize_bandwidth_gbps\": " << fmt_g(quant_gbps) << ",\n"
          << "  \"repeat\": " << repeat << ", \"timing\": \"min\"\n"
          << "}\n";
    }

    {
        std::ostringstream log;
        log << std::string(62, '=') << "\n"
            << "输入      : " << stem << " (" << tf.rows << "x" << tf.cols << ", " << tf.dtype
            << ", " << gq.q.num_elems << " 元素)\n"
            << "格式      : " << gq.q.format << " (元素 " << cfg.elem_format << ", 块 "
            << (gq.q.scale_mode == "block" ? std::to_string(gq.q.block_size) : std::string("-"))
            << ", " << gq.q.scale_mode << "-scale, " << cfg.rounding << " 舍入)\n"
            << "输出类型  : " << cfg.output_type << "\n"
            << std::string(62, '-') << "\n"
            << "最大绝对误差 : " << fmt_g(st.max_abs) << "\n"
            << "MAE          : " << fmt_g(st.mae) << "\n"
            << "MSE          : " << fmt_g(st.mse) << "\n"
            << "RMSE         : " << fmt_g(st.rmse) << "\n"
            << "压缩后位/元素 : " << fmt_g(cp.dst_bits) << " bit"
            << (gq.q.format == "nvfp4"
                    ? "（理论 " + fmt_g(4.0 + 8.0 / std::max(gq.q.block_size, 1)) + "）"
                    : "")
            << "\n"
            << "压缩率       : " << fmt_g(cp.ratio) << "x\n"
            << std::string(62, '-') << "\n"
            << "量化 kernel   : " << fmt_g(gq.quantize_ms) << " ms (" << fmt_g(quant_gbps)
            << " GB/s)\n"
            << "反量化 kernel : " << fmt_g(gd.dequant_ms) << " ms (" << fmt_g(dequant_gbps)
            << " GB/s)\n"
            << "验证         : 量化 " << (gq.verify_match ? "一致" : "不一致") << " / 反量化 "
            << (gd.verify_match ? "一致" : "不一致") << (verify ? "" : "（verify 关闭）") << "\n"
            << std::string(62, '=') << "\n";
        std::ofstream f(odir + "/metrics.log");
        f << log.str();
    }

    if (row) {
        row->format = gq.q.format;
        row->matrix = stem;
        row->scale_mode = gq.q.scale_mode;
        row->output_type = cfg.output_type;
        row->in_dtype = tf.dtype;
        row->rounding = cfg.rounding;
        row->num_elems = gq.q.num_elems;
        row->rows = (int)tf.rows;
        row->cols = (int)tf.cols;
        row->quantize_ms = gq.quantize_ms;
        row->dequant_ms = gd.dequant_ms;
        row->quant_gbps = quant_gbps;
        row->dequant_gbps = dequant_gbps;
        row->max_abs = st.max_abs;
        row->mae = st.mae;
        row->mse = st.mse;
        row->rmse = st.rmse;
        row->compression_ratio = cp.ratio;
        row->dst_bits = cp.dst_bits;
        row->verify_quant = gq.verify_match;
        row->verify_dequant = gd.verify_match;
        row->verify_disabled = !verify;
        row->repeat = repeat;
    }

    if (verify && (!gq.verify_match || !gd.verify_match))
        return -1;
    return 0;
}
