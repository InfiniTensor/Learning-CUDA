// Pure-C++ benchmark: our Hadamard kernels vs fast-hadamard-transform (fht).
//
// No Python / torch / virtualenv. Everything is compiled with nvcc and run as a
// single binary. Correctness is checked against an fp32 in-place FWHT reference
// (identical to y = x @ H with Sylvester H), and gated on max|ours - fht|.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <curand.h>

#include "cuda_check.h"
#include "dtype.h"
#include "fht/fast_hadamard_transform.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <set>
#include <sstream>
#include <string>
#include <vector>

// ---- kernel entry points (explicitly instantiated in their .cu files) ----
template <typename T> void hadamard_v1(const T*, T*, int, int, cudaStream_t);
template <typename T> void hadamard_v2(const T*, T*, int, int, cudaStream_t);
template <typename T> void hadamard_v3(const T*, T*, int, int, cudaStream_t);
template <typename T> void hadamard_v4(const T*, T*, int, int, cudaStream_t);
template <typename T> void hadamard_hybrid(const T*, T*, int, int, cudaStream_t);

template <DType dtype> void run_fht(void*, void*, uint32_t, uint32_t, cudaStream_t);

template <typename T> void fast_hadamard_transform_cuda(HadamardParamsBase&, cudaStream_t);

// ---- dtype traits ----
template <typename T> __device__ __forceinline__ T cvt_from_float(float f);
template <> __device__ __forceinline__ __half cvt_from_float<__half>(float f) { return __float2half(f); }
template <> __device__ __forceinline__ __nv_bfloat16 cvt_from_float<__nv_bfloat16>(float f) { return __float2bfloat16(f); }

template <typename T> inline float widen_to_float(T v);
template <> inline float widen_to_float<__half>(__half v) { return __half2float(v); }
template <> inline float widen_to_float<__nv_bfloat16>(__nv_bfloat16 v) { return __bfloat162float(v); }

template <typename T> struct DTypeInfo;
template <> struct DTypeInfo<__half> {
    static const char* name() { return "fp16"; }
    static constexpr double eps = 1.0 / 1024.0;        // 2^-10
    static constexpr double max_err_vs_fht = 1e-2;
    static constexpr DType hada_dtype = DType::Half;
};
template <> struct DTypeInfo<__nv_bfloat16> {
    static const char* name() { return "bf16"; }
    static constexpr double eps = 1.0 / 128.0;         // 2^-7
    static constexpr double max_err_vs_fht = 5e-2;
    static constexpr DType hada_dtype = DType::BFloat16;
};

// ---- CLI options ----
struct Options {
    std::string impl = "2";     // 1|2|3|4|hybrid
    std::string rows = "small";
    std::string dims = "small";
    std::string dtype = "all";  // fp16|bf16|all
    bool no_check = false;
    bool no_bench = false;
    bool no_warmup = false;
    int iters = 100;
    double warmup_ms = 200.0;
    unsigned seed = 0;
};

enum class Impl { V1, V2, V3, V4, Hybrid };

Impl parse_impl(const std::string& s) {
    if (s == "1") return Impl::V1;
    if (s == "2") return Impl::V2;
    if (s == "3") return Impl::V3;
    if (s == "4") return Impl::V4;
    if (s == "hybrid") return Impl::Hybrid;
    std::fprintf(stderr,
                 "unknown --impl '%s' (use 1|2|3|4|hybrid)\n",
                 s.c_str());
    std::exit(1);
}

const std::vector<int> ROWS_SMALL  = {1, 4, 16, 32, 64, 128, 256, 512};
const std::vector<int> ROWS_MIDDLE = {1024, 2048, 4096, 8192};
const std::vector<int> ROWS_LARGE  = {16384, 32768, 65536};
const std::vector<int> DIMS_SMALL  = {2, 4, 8, 16, 32, 64, 128, 256};
const std::vector<int> DIMS_MIDDLE = {512, 1024, 2048, 4096, 8192};
const std::vector<int> DIMS_LARGE  = {16384, 32768, 65536};

// Resolve a "small,middle,1024" spec (for either rows or dims) into a list.
std::vector<int> resolve(const std::string& spec, bool rows) {
    const std::vector<int>* small  = rows ? &ROWS_SMALL  : &DIMS_SMALL;
    const std::vector<int>* middle = rows ? &ROWS_MIDDLE : &DIMS_MIDDLE;
    const std::vector<int>* large  = rows ? &ROWS_LARGE  : &DIMS_LARGE;

    std::vector<int> out;
    std::set<int> seen;
    std::stringstream ss(spec);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        if (tok.empty()) continue;
        const std::vector<int>* v = nullptr;
        if (tok == "small")  v = small;
        else if (tok == "middle") v = middle;
        else if (tok == "large")  v = large;
        else if (tok == "all") {
            for (int x : *small)  { if (seen.insert(x).second) out.push_back(x); }
            for (int x : *middle) { if (seen.insert(x).second) out.push_back(x); }
            for (int x : *large)  { if (seen.insert(x).second) out.push_back(x); }
            continue;
        } else if (tok.find_first_not_of("0123456789") == std::string::npos) {
            int x = std::atoi(tok.c_str());
            if (seen.insert(x).second) out.push_back(x);
            continue;
        } else {
            std::fprintf(stderr, "unknown level/value '%s'; use small/middle/large, 'all', or an int\n", tok.c_str());
            std::exit(1);
        }
        for (int x : *v) if (seen.insert(x).second) out.push_back(x);
    }
    return out;
}

// ---- reference FWHT (fp32, in-place, natural/Sylvester ordering) ----
void fwht_row(float* x, int n) {
    for (int s = 1; s < n; s <<= 1) {
        for (int i = 0; i < n; i += 2 * s) {
            for (int j = 0; j < s; ++j) {
                float a = x[i + j], b = x[i + j + s];
                x[i + j] = a + b;
                x[i + j + s] = a - b;
            }
        }
    }
}

// ---- device helpers ----
struct CudaBuffer {
    void* ptr = nullptr;
    size_t bytes = 0;
    ~CudaBuffer() { if (ptr) cudaFree(ptr); }
    void alloc(size_t b) {
        if (bytes >= b && ptr) return;
        if (ptr) cudaFree(ptr);
        CUDA_CHECK(cudaMalloc(&ptr, b));
        bytes = b;
    }
};

template <typename T>
__global__ void quantize_kernel(const float* __restrict__ in, T* __restrict__ out, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = cvt_from_float<T>(in[i]);
}

__global__ void warmup_kernel(const float* __restrict__ a, float* __restrict__ b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) b[i] = a[i] * 0.999f + 1.0f;
}

// Generate rows*cols standard-normal fp16/bf16 values in `d_x` (via fp32 scratch).
template <typename T>
void generate_input(T* d_x, float* d_f32, size_t numel, unsigned seed) {
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, seed);
    curandGenerateNormal(gen, d_f32, numel, 0.0f, 1.0f);
    curandDestroyGenerator(gen);

    int threads = 256;
    size_t blocks = (numel + threads - 1) / threads;
    quantize_kernel<T><<<blocks, threads>>>(d_f32, d_x, numel);
    CUDA_LAUNCH_CHECK();
}

void warmup_gpu(double seconds) {
    int n = 2048 * 2048;
    float *a, *b;
    CUDA_CHECK(cudaMalloc(&a, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&b, n * sizeof(float)));
    CUDA_CHECK(cudaMemset(a, 0, n * sizeof(float)));
    auto t0 = std::chrono::steady_clock::now();
    while (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < seconds) {
        for (int i = 0; i < 100; ++i)
            warmup_kernel<<<(n + 255) / 256, 256>>>(a, b, n);
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    CUDA_CHECK(cudaFree(a));
    CUDA_CHECK(cudaFree(b));
}

float time_launches(const std::function<void()>& fn, int iters, double warmup_ms) {
    cudaEvent_t start, end;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&end));

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(start));
    float elapsed = 0.0f;
    while (true) {
        fn();
        CUDA_CHECK(cudaEventRecord(end));
        CUDA_CHECK(cudaEventSynchronize(end));
        CUDA_CHECK(cudaEventElapsedTime(&elapsed, start, end));
        if (elapsed >= warmup_ms) break;
    }

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i) fn();
    CUDA_CHECK(cudaEventRecord(end));
    CUDA_CHECK(cudaEventSynchronize(end));
    float total = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total, start, end));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(end));
    return total / iters;
}

// ---- launchers ----
template <typename T>
void launch_ours(Impl impl, const T* x, T* out, int rows, int cols,
                 cudaStream_t stream) {
    switch (impl) {
        case Impl::V1: hadamard_v1<T>(x, out, rows, cols, stream); break;
        case Impl::V2: hadamard_v2<T>(x, out, rows, cols, stream); break;
        case Impl::V3: hadamard_v3<T>(x, out, rows, cols, stream); break;
        case Impl::V4: hadamard_v4<T>(x, out, rows, cols, stream); break;
        // The tensor-core hybrid path starts at 256 elements. Keep the CLI
        // usable with the benchmark's standard small dimension set by using
        // the fp32 v4 path below that boundary.
        case Impl::Hybrid:
            if (cols < 256) hadamard_v4<T>(x, out, rows, cols, stream);
            else hadamard_hybrid<T>(x, out, rows, cols, stream);
            break;
    }
}

template <typename T>
void launch_hada(const T* x, T* out, int rows, int cols, cudaStream_t stream) {
    run_fht<DTypeInfo<T>::hada_dtype>((void*)x, (void*)out, (uint32_t)((size_t)rows * cols), (uint32_t)cols, stream);
}

template <typename T>
void launch_fht(const T* x, T* out, int rows, int cols, cudaStream_t stream) {
    HadamardParamsBase p;
    std::memset(&p, 0, sizeof(p));
    p.batch = rows;
    p.dim = cols;
    p.log_N = 0;
    while ((1 << p.log_N) < cols) ++p.log_N;
    p.x_batch_stride = cols;
    p.out_batch_stride = cols;
    p.scale = 1.0f;
    p.x_ptr = (void*)x;
    p.out_ptr = (void*)out;
    fast_hadamard_transform_cuda<T>(p, stream);
}

bool hada_supported(int rows, int cols) {
    return ((size_t)rows * cols) % 256 == 0 && cols >= 2 && cols <= 32768 && (cols & (cols - 1)) == 0;
}

bool fht_supported(int cols) {
    return cols >= 2 && cols <= 32768 && (cols & (cols - 1)) == 0;
}

template <typename T>
std::vector<float> copy_out_to_float(const T* d, size_t n) {
    std::vector<T> h(n);
    CUDA_CHECK(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost));
    std::vector<float> f(n);
    for (size_t i = 0; i < n; ++i) f[i] = widen_to_float(h[i]);
    return f;
}

float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    float m = 0.0f;
    size_t n = std::min(a.size(), b.size());
    for (size_t i = 0; i < n; ++i) m = std::max(m, std::fabs(a[i] - b[i]));
    return m;
}

// ---- correctness pass for one shape ----
template <typename T>
bool check_one(Impl impl, int rows, int cols, unsigned seed) {
    size_t numel = (size_t)rows * cols;

    CudaBuffer d_f32, d_x, d_out_ours, d_out_fht, d_out_hada;
    d_f32.alloc(numel * sizeof(float));
    d_x.alloc(numel * sizeof(T));
    d_out_ours.alloc(numel * sizeof(T));
    d_out_hada.alloc(numel * sizeof(T));

    generate_input<T>((T*)d_x.ptr, (float*)d_f32.ptr, numel, seed);

    // fht only supports dim <= 32768; pad 2/4 up to 8 (zero padding gives the
    // same H_n result after slicing, matching fht's Python wrapper).
    bool fht_ok = fht_supported(cols);
    int fht_dim = (cols < 8) ? 8 : cols;
    CudaBuffer d_x_pad, d_out_fht_pad;
    const void* fht_in = d_x.ptr;
    void* fht_out = nullptr;
    if (fht_ok) {
        if (fht_dim == cols) {
            d_out_fht.alloc(numel * sizeof(T));
            fht_out = d_out_fht.ptr;
        } else {
            d_x_pad.alloc((size_t)rows * fht_dim * sizeof(T));
            d_out_fht_pad.alloc((size_t)rows * fht_dim * sizeof(T));
            d_out_fht.alloc(numel * sizeof(T));
            CUDA_CHECK(cudaMemset(d_x_pad.ptr, 0, (size_t)rows * fht_dim * sizeof(T)));
            CUDA_CHECK(cudaMemcpy2D(d_x_pad.ptr, fht_dim * sizeof(T), d_x.ptr, cols * sizeof(T),
                                    cols * sizeof(T), rows, cudaMemcpyDeviceToDevice));
            fht_in = d_x_pad.ptr;
            fht_out = d_out_fht_pad.ptr;
        }
    }

    // fp32 reference from the *quantized* input (== x.float() in the old script).
    std::vector<float> ref = copy_out_to_float((const T*)d_x.ptr, numel);
    for (int r = 0; r < rows; ++r) fwht_row(ref.data() + (size_t)r * cols, cols);

    launch_ours<T>(impl, (const T*)d_x.ptr, (T*)d_out_ours.ptr,
                   rows, cols, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    if (fht_ok) {
        launch_fht<T>((const T*)fht_in, (T*)fht_out, rows, fht_dim, 0);
        CUDA_CHECK(cudaDeviceSynchronize());
        if (fht_dim != cols) {
            CUDA_CHECK(cudaMemcpy2D(d_out_fht.ptr, cols * sizeof(T), d_out_fht_pad.ptr, fht_dim * sizeof(T),
                                    cols * sizeof(T), rows, cudaMemcpyDeviceToDevice));
        }
    }
    bool run_hada = hada_supported(rows, cols);
    if (run_hada) {
        launch_hada<T>((const T*)d_x.ptr, (T*)d_out_hada.ptr, rows, cols, 0);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    std::vector<float> ours = copy_out_to_float((const T*)d_out_ours.ptr, numel);
    std::vector<float> theirs, hada;
    if (fht_ok) theirs = copy_out_to_float((const T*)d_out_fht.ptr, numel);
    if (run_hada) hada = copy_out_to_float((const T*)d_out_hada.ptr, numel);

    double tol = DTypeInfo<T>::max_err_vs_fht;
    bool passed = true;

    std::printf("  dtype=%s, rows=%d, cols=%d\n", DTypeInfo<T>::name(), rows, cols);
    std::printf("    max|ours  - ref| = %.4e\n", max_abs_diff(ours, ref));
    if (fht_ok) std::printf("    max|fht   - ref| = %.4e\n", max_abs_diff(theirs, ref));
    if (run_hada) std::printf("    max|hada  - ref| = %.4e\n", max_abs_diff(hada, ref));

    if (fht_ok) {
        double e = max_abs_diff(ours, theirs);
        size_t ours_gt = 0, ours_lt = 0, unequal = 0, first_unequal = ours.size();
        for (size_t i = 0; i < ours.size(); ++i) {
            if (ours[i] > theirs[i]) { ++ours_gt; ++unequal; first_unequal = std::min(first_unequal, i); }
            else if (ours[i] < theirs[i]) { ++ours_lt; ++unequal; first_unequal = std::min(first_unequal, i); }
        }
        bool ok = e <= tol;
        std::printf("    max|ours  - fht| = %.4e  (requirement <= %.0e) [%s]\n",
                    e, tol, ok ? "PASS" : "FAIL");
        std::printf("    unequal=%zu (ours>fht: %zu, ours<fht: %zu)\n",
                    unequal, ours_gt, ours_lt);
        if (first_unequal != ours.size())
            std::printf("    first unequal index=%zu: ours=%+.9g, fht=%+.9g, ref=%+.9g\n",
                        first_unequal, ours[first_unequal], theirs[first_unequal], ref[first_unequal]);
        if (!ok) {
            std::fprintf(stderr, "%s: max|ours - fht| = %.4e exceeds requirement %.0e\n",
                         DTypeInfo<T>::name(), e, tol);
            passed = false;
        }
    } else {
        std::printf("    (fht n/a for cols=%d; skipping ours-vs-fht gate)\n", cols);
    }

    if (run_hada) {
        double e = max_abs_diff(hada, theirs);
        double htol = DTypeInfo<T>::max_err_vs_fht;
        bool ok = e <= htol;
        std::printf("    max|hada  - fht| = %.4e  (requirement <= %.0e) [%s]\n",
                    e, htol, ok ? "PASS" : "FAIL");
        if (!ok) {
            std::fprintf(stderr, "%s: max|hadacore - fht| = %.4e exceeds requirement %.0e\n",
                         DTypeInfo<T>::name(), e, htol);
            passed = false;
        }
    }
    return passed;
}

// ---- timing pass for one shape ----
template <typename T>
void bench_one(Impl impl, int rows, int cols, int iters, double warmup_ms, unsigned seed) {
    size_t numel = (size_t)rows * cols;

    CudaBuffer d_f32, d_x, d_out_ours, d_out_hada;
    d_f32.alloc(numel * sizeof(float));
    d_x.alloc(numel * sizeof(T));
    d_out_ours.alloc(numel * sizeof(T));
    d_out_hada.alloc(numel * sizeof(T));

    generate_input<T>((T*)d_x.ptr, (float*)d_f32.ptr, numel, seed);

    bool fht_ok = fht_supported(cols);
    int fht_dim = (cols < 8) ? 8 : cols;
    CudaBuffer d_x_pad, d_out_fht, d_out_fht_pad;
    const void* fht_in = d_x.ptr;
    void* fht_out = nullptr;
    if (fht_ok) {
        if (fht_dim == cols) {
            d_out_fht.alloc(numel * sizeof(T));
            fht_out = d_out_fht.ptr;
        } else {
            d_x_pad.alloc((size_t)rows * fht_dim * sizeof(T));
            d_out_fht_pad.alloc((size_t)rows * fht_dim * sizeof(T));
            d_out_fht.alloc(numel * sizeof(T));
            CUDA_CHECK(cudaMemset(d_x_pad.ptr, 0, (size_t)rows * fht_dim * sizeof(T)));
            CUDA_CHECK(cudaMemcpy2D(d_x_pad.ptr, fht_dim * sizeof(T), d_x.ptr, cols * sizeof(T),
                                    cols * sizeof(T), rows, cudaMemcpyDeviceToDevice));
            fht_in = d_x_pad.ptr;
            fht_out = d_out_fht_pad.ptr;
        }
    }

    bool run_hada = hada_supported(rows, cols);

    float ours = time_launches([&] { launch_ours<T>(impl, (const T*)d_x.ptr, (T*)d_out_ours.ptr, rows, cols, 0); },
                               iters, warmup_ms);
    float theirs = fht_ok
        ? time_launches([&] { launch_fht<T>((const T*)fht_in, (T*)fht_out, rows, fht_dim, 0); }, iters, warmup_ms)
        : 0.0f;
    float hada = run_hada
        ? time_launches([&] { launch_hada<T>((const T*)d_x.ptr, (T*)d_out_hada.ptr, rows, cols, 0); }, iters, warmup_ms)
        : 0.0f;

    double flops = (double)rows * cols * std::log2((double)cols);
    auto gflops = [&](float ms) { return flops / (ms * 1e-3) / 1e9; };

    double speedup = fht_ok ? theirs / ours : 0.0;
    double speedup_hada = (run_hada && fht_ok) ? theirs / hada : 0.0;

    std::printf("  dtype=%s, rows=%6d, cols=%5d | ours %8.3f ms (%7.1f GFLOP/s) | ",
                DTypeInfo<T>::name(), rows, cols, ours, gflops(ours));
    if (fht_ok) {
        std::printf("fht  %8.3f ms (%7.1f GFLOP/s) | speedup %6.2fx",
                    theirs, gflops(theirs), speedup);
    } else {
        std::printf("fht  n/a");
    }
    if (run_hada) {
        std::printf(" | hada %8.3f ms (%7.1f GFLOP/s) | hada/fht %6.2fx",
                    hada, gflops(hada), speedup_hada);
    } else {
        std::printf(" | hada  n/a");
    }
    std::printf("\n");
}

// ---- driver ----
template <typename T>
int check_dtype(Impl impl, const std::vector<int>& rows, const std::vector<int>& dims, unsigned seed) {
    int failures = 0;
    for (int cols : dims)
        for (int r : rows)
            if (!check_one<T>(impl, r, cols, seed)) ++failures;
    return failures;
}

template <typename T>
void bench_dtype(Impl impl, const std::vector<int>& rows, const std::vector<int>& dims,
                 int iters, double warmup_ms, unsigned seed) {
    for (int r : rows)
        for (int cols : dims)
            bench_one<T>(impl, r, cols, iters, warmup_ms, seed);
}

int main(int argc, char** argv) {
    Options opt;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&](const char* flag) -> std::string {
            if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", flag); std::exit(1); }
            return argv[++i];
        };
        if (a == "--impl") opt.impl = next(a.c_str());
        else if (a == "--rows") opt.rows = next(a.c_str());
        else if (a == "--dims") opt.dims = next(a.c_str());
        else if (a == "--dtype") opt.dtype = next(a.c_str());
        else if (a == "--iters") opt.iters = std::atoi(next(a.c_str()).c_str());
        else if (a == "--warmup-ms") opt.warmup_ms = std::atof(next(a.c_str()).c_str());
        else if (a == "--seed") opt.seed = (unsigned)std::strtoul(next(a.c_str()).c_str(), nullptr, 10);
        else if (a == "--no-check") opt.no_check = true;
        else if (a == "--no-bench") opt.no_bench = true;
        else if (a == "--no-warmup") opt.no_warmup = true;
        else { std::fprintf(stderr, "unknown arg '%s'\n", a.c_str()); std::exit(1); }
    }

    if (opt.dtype != "fp16" && opt.dtype != "bf16" && opt.dtype != "all") {
        std::fprintf(stderr, "--dtype must be fp16|bf16|all\n");
        std::exit(1);
    }

    Impl impl = parse_impl(opt.impl);
    std::vector<int> rows = resolve(opt.rows, true);
    std::vector<int> dims = resolve(opt.dims, false);

    const int MAX_CHECK_DIM = 8192;
    std::vector<int> check_dims, skipped;
    for (int d : dims) (d <= MAX_CHECK_DIM ? check_dims : skipped).push_back(d);

    int check_failures = 0;
    if (!opt.no_check) {
        std::printf("== correctness (vs fp32 FWHT reference) ==\n");
        if (opt.dtype == "fp16" || opt.dtype == "all")
            check_failures += check_dtype<__half>(impl, rows, check_dims, opt.seed);
        if (opt.dtype == "bf16" || opt.dtype == "all")
            check_failures += check_dtype<__nv_bfloat16>(impl, rows, check_dims, opt.seed);
        if (!skipped.empty()) {
            std::printf("  (skipped dims {");
            for (size_t i = 0; i < skipped.size(); ++i) std::printf("%s%d", i ? "," : "", skipped[i]);
            std::printf("}: reference capped at %d)\n", MAX_CHECK_DIM);
        }
        std::printf("== correctness summary: %d shape(s) failed ==\n", check_failures);
    }

    if (!opt.no_bench) {
        std::printf("== timing ==\n");
        if (!opt.no_warmup) warmup_gpu(1.0);
        if (opt.dtype == "fp16" || opt.dtype == "all")
            bench_dtype<__half>(impl, rows, dims, opt.iters, opt.warmup_ms, opt.seed);
        if (opt.dtype == "bf16" || opt.dtype == "all")
            bench_dtype<__nv_bfloat16>(impl, rows, dims, opt.iters, opt.warmup_ms, opt.seed);
    }
    return check_failures == 0 ? 0 : 1;
}
