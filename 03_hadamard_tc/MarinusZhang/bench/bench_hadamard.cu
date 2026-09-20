// Benchmark driver: kernel time in ms plus effective bandwidth in GB/s.
//
// Three things get measured:
//   * table 1 - the two memory regimes (L2-resident, and one working set past L2);
//   * table 2 - every head_dim the project supports, so the support boundaries are
//     measured rather than asserted;
//   * a copy of each shape's own footprint, which is the read+write ceiling this
//     machine reaches at that size. Hardware counters are unavailable in this
//     container (see docs/report.md section 4), so this is what the kernel rows are
//     held against: every row is charged the bytes it really moves, so the copy and
//     the transform rows (4 B/elem) are directly comparable, a fused FP8 row (3 B/elem)
//     is at the memory limit as soon as it reports the same bandwidth, and no fused row
//     can finish in less than three quarters of the copy's time.
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <string>
#include <vector>

#include "hadamard/common.h"
#include "hadamard/cuda_utils.h"
#include "hadamard/hadamard.h"

namespace {

struct BenchCase {
    int batch;
    int seq_len;
    int num_heads;
    int head_dim;
};

// Table 1. Traffic (one read plus one write) stays well below the 72 MiB L2 of the
// target GPU for the first five, so across the timed iterations they measure L2-bound
// bandwidth; the last one is past L2 and measures HBM.
const BenchCase kRegimeCases[] = {
    {1, 128, 12, 64},   // 0.4 MiB
    {2, 256, 16, 128},  // 4 MiB
    {1, 512, 8, 256},   // 4 MiB
    {4, 512, 16, 128},  // 16 MiB
    {1, 4096, 12, 64},  // 12 MiB
    // 8 * 4096 * 16 * 128 elements = 256 MiB of traffic, i.e. larger than L2: this is
    // the row that shows what the kernel achieves against HBM (see the report).
    {8, 4096, 16, 128},
};

// Table 2. Every head_dim the project supports, at a row count that keeps the working
// set inside L2 and at the degenerate rows = 1. The L2 row count is chosen so the
// measurement is not sitting on the ~2 us launch floor: 4 Mi elements is 8 MiB in and
// 8 MiB out, which fits in the 72 MiB L2 with room to spare but still takes long enough
// for the bandwidth column to mean something. This is where the support boundaries
// become visible: d = 8 has no Tensor Core K tile, so both Tensor Core entries fall
// back to the butterfly, and d = 1024 has no fused Tensor Core path at all. The rows = 1
// shapes are latency probes - one row is 512 B at d = 256, so their GB/s column is not
// meaningful - and exist to show that the per-call cost grows with d on the Tensor Core
// path and stays flat on the butterfly.
constexpr long kSweepElems = 4 * 1024 * 1024;
constexpr int kSweepDims[] = {8, 16, 32, 64, 128, 256, 512, 1024};

// The CPU reference this benchmark times is the O(d log2 d) butterfly (ref_fwht_fp32),
// not the explicit matmul, and it runs at about 3.4 ns per element regardless of d. So
// what bounds a row is the element count and not d^2: the limit below puts one call of
// the largest shape at roughly a quarter of a second, a second of wall time for the
// kCpuIters iterations. It is set so that the 256 MiB shape keeps its CPU column, which
// is the number the previous milestone reported. The two shape tables above are
// unchanged, which keeps the table comparable across milestones.
constexpr long kCpuElemLimit = 64 * 1024 * 1024;

constexpr int kGpuIters = 50;
constexpr int kCpuIters = 5;

std::vector<BenchCase> make_sweep_cases() {
    std::vector<BenchCase> cases;
    for (int d : kSweepDims) {
        cases.push_back({1, 1, 1, d});
        cases.push_back({1, static_cast<int>(kSweepElems / d), 1, d});
    }
    return cases;
}

std::string case_name(const BenchCase& c) {
    return "[" + std::to_string(c.batch) + "," + std::to_string(c.seq_len) + "," +
           std::to_string(c.num_heads) + "," + std::to_string(c.head_dim) + "]";
}

// Vector copy of a byte range, used only to establish the machine's read+write ceiling
// for a footprint. The tail is at most 15 bytes and is handled by one thread.
__global__ void copy_kernel(const unsigned char* __restrict__ src, unsigned char* __restrict__ dst,
                            long bytes) {
    const long vecs = bytes / 16;
    const long stride = static_cast<long>(gridDim.x) * blockDim.x;
    const uint4* const s = reinterpret_cast<const uint4*>(src);
    uint4* const d = reinterpret_cast<uint4*>(dst);
    for (long i = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x; i < vecs; i += stride) {
        d[i] = s[i];
    }
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        for (long i = vecs * 16; i < bytes; ++i) {
            dst[i] = src[i];
        }
    }
}

void print_header(const char* title) {
    std::printf("\n== %s ==\n\n", title);
    std::printf("%-24s %-5s %-16s %10s %10s\n", "shape (B,S,H,D)", "dtype", "impl", "ms", "GB/s");
    std::printf("--------------------------------------------------------------------------\n");
}

double bench_cpu_fwht(const std::vector<float>& x, std::vector<float>& y, long rows, int d) {
    const auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < kCpuIters; ++i) {
        hadamard::ref_fwht_fp32(x.data(), y.data(), rows, d);
    }
    const auto t1 = std::chrono::high_resolution_clock::now();
    const double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return total_ms / kCpuIters;
}

void bench_gpu_implementation(const char* impl_name,
                              bool (*launch)(const void*, void*, const hadamard::Shape&,
                                             hadamard::DType, cudaStream_t),
                              const BenchCase& c, hadamard::DType dtype) {
    using namespace hadamard;
    const Shape shape{c.batch, c.seq_len, c.num_heads, c.head_dim};
    const long n = shape.elems();
    const size_t bytes = static_cast<size_t>(n) * sizeof(uint16_t) * 2;  // read + write

    void* d_x = nullptr;
    void* d_y = nullptr;
    HW_CUDA_CHECK(cudaMalloc(&d_x, static_cast<size_t>(n) * sizeof(uint16_t)));
    HW_CUDA_CHECK(cudaMalloc(&d_y, static_cast<size_t>(n) * sizeof(uint16_t)));
    HW_CUDA_CHECK(cudaMemset(d_x, 0, static_cast<size_t>(n) * sizeof(uint16_t)));

    const bool available = launch(d_x, d_y, shape, dtype, nullptr);
    if (!available) {
        std::printf("%-24s %-5s %-16s %10s %10s\n", case_name(c).c_str(), dtype_name(dtype),
                    impl_name, "n/a", "n/a");
        HW_CUDA_CHECK(cudaFree(d_x));
        HW_CUDA_CHECK(cudaFree(d_y));
        return;
    }

    CudaEventTimer timer;
    launch(d_x, d_y, shape, dtype, nullptr);  // warm-up
    timer.start();
    for (int i = 0; i < kGpuIters; ++i) {
        launch(d_x, d_y, shape, dtype, nullptr);
    }
    timer.stop();
    const float ms = timer.elapsed_ms() / kGpuIters;

    std::printf("%-24s %-5s %-16s %10.4f %10.1f\n", case_name(c).c_str(), dtype_name(dtype),
                impl_name, static_cast<double>(ms), effective_bandwidth_gbps(bytes, ms));
    HW_CUDA_CHECK(cudaFree(d_x));
    HW_CUDA_CHECK(cudaFree(d_y));
}

template <typename Fn>
double time_gpu(Fn&& fn) {
    hadamard::CudaEventTimer timer;
    fn();  // warm-up
    timer.start();
    for (int i = 0; i < kGpuIters; ++i) {
        fn();
    }
    timer.stop();
    return timer.elapsed_ms() / kGpuIters;
}

// Read+write ceiling at this shape's footprint, charged 2 B/elem each way so it lines
// up with the transform rows. Returns the ms so the caller can print derived limits.
double bench_copy_ceiling(const BenchCase& c) {
    using namespace hadamard;
    const Shape shape{c.batch, c.seq_len, c.num_heads, c.head_dim};
    const size_t bytes = static_cast<size_t>(shape.elems()) * sizeof(uint16_t);

    void* d_x = nullptr;
    void* d_y = nullptr;
    HW_CUDA_CHECK(cudaMalloc(&d_x, bytes));
    HW_CUDA_CHECK(cudaMalloc(&d_y, bytes));
    HW_CUDA_CHECK(cudaMemset(d_x, 0, bytes));

    constexpr int kThreads = 256;
    int sm_count = 0;
    HW_CUDA_CHECK(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0));
    const long vecs = static_cast<long>(bytes / 16);
    const long needed = std::max<long>(1, (vecs + kThreads - 1) / kThreads);
    // A grid-stride loop wants enough blocks to fill the machine and no more; more blocks
    // than work would just add launch overhead to a 2 us kernel.
    const long blocks = std::min<long>(needed, static_cast<long>(sm_count) * 32);

    const double ms = time_gpu([&] {
        copy_kernel<<<static_cast<unsigned>(blocks), kThreads>>>(
            static_cast<const unsigned char*>(d_x), static_cast<unsigned char*>(d_y),
            static_cast<long>(bytes));
    });

    std::printf("%-24s %-5s %-16s %10.4f %10.1f\n", case_name(c).c_str(), "fp16", "copy ceiling",
                ms, effective_bandwidth_gbps(2 * bytes, static_cast<float>(ms)));
    HW_CUDA_CHECK(cudaFree(d_x));
    HW_CUDA_CHECK(cudaFree(d_y));
    return ms;
}

// FP8 rows. The byte model is the point of these rows: a fused kernel reads the
// activation once and writes the codes, a two-stage pipeline pays for the transform's
// own read/write as well. Per-row scales add 4 B per row, i.e. 4/d B per element.
size_t fp8_code_bytes(long n, long rows) {
    return static_cast<size_t>(n) + static_cast<size_t>(rows) * sizeof(float);
}

void bench_gpu_fp8(const char* impl_name,
                   bool (*launch)(const void*, uint8_t*, float*, const hadamard::Shape&,
                                  hadamard::DType, cudaStream_t),
                   const BenchCase& c, hadamard::DType dtype) {
    using namespace hadamard;
    const Shape shape{c.batch, c.seq_len, c.num_heads, c.head_dim};
    const long n = shape.elems();
    const long rows = shape.rows();
    const size_t bytes = static_cast<size_t>(n) * sizeof(uint16_t) + fp8_code_bytes(n, rows);

    void* d_x = nullptr;
    uint8_t* d_q = nullptr;
    float* d_s = nullptr;
    HW_CUDA_CHECK(cudaMalloc(&d_x, static_cast<size_t>(n) * sizeof(uint16_t)));
    HW_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_q), static_cast<size_t>(n)));
    HW_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_s), rows * sizeof(float)));
    HW_CUDA_CHECK(cudaMemset(d_x, 0, static_cast<size_t>(n) * sizeof(uint16_t)));

    if (!launch(d_x, d_q, d_s, shape, dtype, nullptr)) {
        std::printf("%-24s %-5s %-16s %10s %10s\n", case_name(c).c_str(), dtype_name(dtype),
                    impl_name, "n/a", "n/a");
    } else {
        const double ms = time_gpu([&] { launch(d_x, d_q, d_s, shape, dtype, nullptr); });
        std::printf("%-24s %-5s %-16s %10.4f %10.1f\n", case_name(c).c_str(), dtype_name(dtype),
                    impl_name, ms, effective_bandwidth_gbps(bytes, static_cast<float>(ms)));
    }
    HW_CUDA_CHECK(cudaFree(d_x));
    HW_CUDA_CHECK(cudaFree(d_q));
    HW_CUDA_CHECK(cudaFree(d_s));
}

void bench_gpu_fp8_two_stage(const char* impl_name,
                             bool (*xform)(const void*, void*, const hadamard::Shape&,
                                           hadamard::DType, cudaStream_t),
                             bool (*quant)(const void*, uint8_t*, float*, const hadamard::Shape&,
                                           hadamard::DType, cudaStream_t),
                             const BenchCase& c, hadamard::DType dtype) {
    using namespace hadamard;
    const Shape shape{c.batch, c.seq_len, c.num_heads, c.head_dim};
    const long n = shape.elems();
    const long rows = shape.rows();
    // Transform: read + write an activation tensor. Quantizer: read it back and write
    // the codes. Both stages are timed together, and both are charged to the byte model.
    const size_t bytes = 3 * static_cast<size_t>(n) * sizeof(uint16_t) + fp8_code_bytes(n, rows);

    void* d_x = nullptr;
    void* d_y = nullptr;
    uint8_t* d_q = nullptr;
    float* d_s = nullptr;
    HW_CUDA_CHECK(cudaMalloc(&d_x, static_cast<size_t>(n) * sizeof(uint16_t)));
    HW_CUDA_CHECK(cudaMalloc(&d_y, static_cast<size_t>(n) * sizeof(uint16_t)));
    HW_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_q), static_cast<size_t>(n)));
    HW_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_s), rows * sizeof(float)));
    HW_CUDA_CHECK(cudaMemset(d_x, 0, static_cast<size_t>(n) * sizeof(uint16_t)));

    const bool available =
        xform(d_x, d_y, shape, dtype, nullptr) && quant(d_y, d_q, d_s, shape, dtype, nullptr);
    if (!available) {
        std::printf("%-24s %-5s %-16s %10s %10s\n", case_name(c).c_str(), dtype_name(dtype),
                    impl_name, "n/a", "n/a");
    } else {
        const double ms = time_gpu([&] {
            xform(d_x, d_y, shape, dtype, nullptr);
            quant(d_y, d_q, d_s, shape, dtype, nullptr);
        });
        std::printf("%-24s %-5s %-16s %10.4f %10.1f\n", case_name(c).c_str(), dtype_name(dtype),
                    impl_name, ms, effective_bandwidth_gbps(bytes, static_cast<float>(ms)));
    }
    HW_CUDA_CHECK(cudaFree(d_x));
    HW_CUDA_CHECK(cudaFree(d_y));
    HW_CUDA_CHECK(cudaFree(d_q));
    HW_CUDA_CHECK(cudaFree(d_s));
}

// One shape: the copy ceiling first, then the transform rows, then the quantized rows.
// The ceiling comes first so it can be held against the rows that follow it.
void run_case(const BenchCase& c, bool with_bf16) {
    using namespace hadamard;
    const Shape shape{c.batch, c.seq_len, c.num_heads, c.head_dim};

    bench_copy_ceiling(c);

    if (shape.elems() <= kCpuElemLimit) {
        const long n = shape.elems();
        const std::vector<float> x = random_host_data(n, 0xF00Du + c.head_dim);
        std::vector<float> y(static_cast<size_t>(n));
        const double cpu_ms = bench_cpu_fwht(x, y, shape.rows(), shape.head_dim);
        const double cpu_bytes_gbps = effective_bandwidth_gbps(
            static_cast<size_t>(n) * sizeof(float) * 2, static_cast<float>(cpu_ms));
        std::printf("%-24s %-5s %-16s %10.4f %10.1f\n", case_name(c).c_str(), "fp32",
                    "cpu reference", cpu_ms, cpu_bytes_gbps);
    }

    bench_gpu_implementation("fwht baseline", &launch_fwht_baseline, c, DType::kFp16);
    bench_gpu_implementation("tensor core", &launch_hadamard_tc, c, DType::kFp16);
    if (with_bf16) {
        bench_gpu_implementation("fwht baseline", &launch_fwht_baseline, c, DType::kBFloat16);
        bench_gpu_implementation("tensor core", &launch_hadamard_tc, c, DType::kBFloat16);
    }

    // FP8 rows run on fp16 activations only: bf16 traffic is identical (2 B/elem), and
    // the interesting question is whether fusing beats doing it in two passes.
    bench_gpu_fp8("fp8 quantize only", &launch_quantize_fp8, c, DType::kFp16);
    bench_gpu_fp8("fwht+fp8 fused", &launch_fwht_baseline_fp8, c, DType::kFp16);
    bench_gpu_fp8("tc+fp8 fused", &launch_hadamard_tc_fp8, c, DType::kFp16);
    bench_gpu_fp8_two_stage("tc->fp8 2-stage", &launch_hadamard_tc, &launch_quantize_fp8, c,
                            DType::kFp16);
    std::printf("\n");
}

}  // namespace

int main() {
    using namespace hadamard;

    const DeviceInfo info = query_device(0);
    std::printf("device: %s (sm_%d%d, %d SMs)\n", info.name.c_str(), info.major, info.minor,
                info.sm_count);
    std::printf("iters : gpu=%d, cpu=%d\n", kGpuIters, kCpuIters);

    print_header("table 1: memory regimes");
    for (const BenchCase& c : kRegimeCases) {
        run_case(c, true);
    }

    print_header("table 2: head_dim sweep (fp16, working set inside L2)");
    for (const BenchCase& c : make_sweep_cases()) {
        run_case(c, false);
    }

    std::printf("notes\n");
    std::printf("  GB/s counts the traffic each row is expected to move: 4 B/elem for the copy\n");
    std::printf("  and the transform rows, 7 B/elem for the two-stage FP8 rows, 3 B/elem plus\n");
    std::printf("  4 B/row for the fused ones. The rows are therefore comparable to each other,\n");
    std::printf("  and on the DRAM-bound shape they all land at the same 925-952 GB/s: that is\n");
    std::printf("  this machine's sustained rate. The copy of the shape's own footprint comes\n");
    std::printf("  first as the reference for the 4 B/elem rows; a fused FP8 row moves three\n");
    std::printf("  quarters of those bytes, so its time cannot be below three quarters of the\n");
    std::printf("  copy's time, and reporting the copy's bandwidth means it is at the memory\n");
    std::printf("  limit, not slow. In table 1 the first five shapes stay inside the 72 MiB L2\n");
    std::printf("  across the timed iterations (GB/s above the HBM peak is normal there); the\n");
    std::printf(
        "  last one is DRAM-bound. Table 2 keeps the working set inside L2 on purpose: it\n");
    std::printf(
        "  is about coverage against head_dim, where d = 8 falls back to the butterfly for\n");
    std::printf(
        "  both Tensor Core entries and d = 1024 has no fused Tensor Core path, which is\n");
    std::printf("  why those rows read n/a; its rows = 1 shapes are latency probes whose GB/s\n");
    std::printf("  column is not meaningful. The CPU reference is only measured while a row has\n");
    std::printf("  at most %ld elements, its cost being linear in the element count.\n",
                kCpuElemLimit);
    return 0;
}
