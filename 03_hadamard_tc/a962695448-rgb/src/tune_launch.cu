// 独立发射配置实验；不改变产品接口和默认派发。
#include "kernels.cuh"
#include "reference.hpp"
#include <array>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <type_traits>

namespace {
constexpr std::array<int, 4> threads{32, 64, 128, 256};
constexpr int captured_calls = 64, replays = 20, groups = 5, warmup = 5;
std::string context;
void check(cudaError_t error, const char* call) {
    if (error != cudaSuccess)
        throw std::runtime_error(context + " " + call + ": " + cudaGetErrorString(error));
}
#define CHECK(call) check((call), #call)
void cleanup(cudaError_t error) noexcept {
    if (error != cudaSuccess) std::cerr << "cleanup: " << cudaGetErrorString(error) << '\n';
}
template<class T> struct Buffer {
    T* ptr = nullptr;
    std::size_t count;
    explicit Buffer(std::size_t n) : count(n) {
        CHECK(cudaMalloc(reinterpret_cast<void**>(&ptr), count * sizeof(T)));
    }
    ~Buffer() { cleanup(cudaFree(ptr)); }
    Buffer(const Buffer&) = delete;
    std::vector<T> read() const {
        std::vector<T> host(count);
        CHECK(cudaMemcpy(host.data(), ptr, count * sizeof(T), cudaMemcpyDeviceToHost));
        return host;
    }
};
struct Stream {
    cudaStream_t handle{};
    Stream() { CHECK(cudaStreamCreateWithFlags(&handle, cudaStreamNonBlocking)); }
    ~Stream() { cleanup(cudaStreamDestroy(handle)); }
};
struct Event {
    cudaEvent_t handle{};
    Event() { CHECK(cudaEventCreate(&handle)); }
    ~Event() { cleanup(cudaEventDestroy(handle)); }
};
struct Graph {
    cudaGraph_t graph{};
    cudaGraphExec_t exec{};
    ~Graph() {
        if (exec) cleanup(cudaGraphExecDestroy(exec));
        if (graph) cleanup(cudaGraphDestroy(graph));
    }
};
template<class T> void equal(const std::vector<T>& actual, const std::vector<T>& expected,
                             const char* name) {
    if (actual.size() != expected.size()) throw std::runtime_error(context + " size mismatch");
    if (std::memcmp(actual.data(), expected.data(), actual.size() * sizeof(T)) == 0) return;
    std::size_t i = 0;
    while (std::memcmp(&actual[i], &expected[i], sizeof(T)) == 0) ++i;
    std::ostringstream message;
    message << context << " exact comparison failed: " << name << " first_index=" << i;
    for (std::size_t j = i > 1 ? i - 2 : 0; j < std::min(i + 3, actual.size()); ++j) {
        std::uint64_t a = 0, b = 0;
        std::memcpy(&a, &actual[j], sizeof(T)); std::memcpy(&b, &expected[j], sizeof(T));
        message << " [" << j << ": actual=" << static_cast<float>(actual[j])
                << " expected=" << static_cast<float>(expected[j]) << " bits=0x" << std::hex
                << a << "/0x" << b << std::dec << ']';
    }
    throw std::runtime_error(message.str());
}

template<class T, int N, bool Quantize>
void run(int rows, float scale, const cudaDeviceProp& gpu, int runtime) {
    const char* dtype = std::is_same_v<T, __half> ? "fp16" : "bf16";
    const std::string case_name = std::string(dtype) + " rows=" + std::to_string(rows) +
        " N=" + std::to_string(N) + " scale=" + std::to_string(scale) +
        " mode=" + (Quantize ? "fused_int4" : "transform");
    context = case_name + " baseline threads=128";
    const unsigned seed = 20260905u + rows + N;
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> uniform(-1.0f, 1.0f);
    std::vector<T> input(static_cast<std::size_t>(rows) * N);
    for (std::size_t i = 0; i < input.size(); ++i) {
        float x = uniform(rng);
        if (i % 257 == 0) x *= 8;
        if (i < N && rows > 1) x = 0; // 全零行覆盖量化 scale=1。
        input[i] = hadamard::as_storage<T>(x);
    }
    Buffer<T> x(input.size()), y(input.size());
    Buffer<std::uint8_t> packed(static_cast<std::size_t>(rows) * ((N + 1) / 2));
    Buffer<float> scales(rows);
    Stream stream;
    // pageable H2D 的 host 返回不保证设备复制完成；必须和 nonblocking kernel 同 stream。
    CHECK(cudaMemcpyAsync(x.ptr, input.data(), input.size() * sizeof(T),
                          cudaMemcpyHostToDevice, stream.handle));
    auto launch = [&](int block_threads) {
        hadamard::launch_warp<T, N, true, Quantize>(
            x.ptr, y.ptr, packed.ptr, scales.ptr, rows, scale, block_threads, stream.handle);
        CHECK(cudaGetLastError());
    };
    // 取得分步变换结果，以其实际舍入后的值验证融合量化。
    hadamard::launch_warp<T, N, true, false>(
        x.ptr, y.ptr, nullptr, nullptr, rows, scale, 128, stream.handle);
    CHECK(cudaGetLastError());
    CHECK(cudaStreamSynchronize(stream.handle));
    const auto transformed = y.read();
    std::vector<float> first_rows(std::min(rows, 4) * N);
    for (std::size_t i = 0; i < first_rows.size(); ++i)
        first_rows[i] = hadamard::as_float(input[i]);
    const auto dense = hadamard::dense_reference(first_rows, N, scale);
    const double tolerance = std::is_same_v<T, __half> ? 1e-2 : 5e-2;
    for (std::size_t i = 0; i < dense.size(); ++i) {
        const auto rounded = hadamard::as_storage<T>(static_cast<float>(dense[i]));
        const double error = std::abs(static_cast<double>(hadamard::as_float(transformed[i])) -
                                      hadamard::as_float(rounded));
        if (!(error < tolerance))
            throw std::runtime_error(context + " dense reference failed index=" + std::to_string(i) +
                " actual=" + std::to_string(hadamard::as_float(transformed[i])) +
                " expected=" + std::to_string(hadamard::as_float(rounded)));
    }
    launch(128);
    CHECK(cudaStreamSynchronize(stream.handle));
    std::vector<T> baseline_y;
    std::vector<std::uint8_t> baseline_packed;
    std::vector<float> baseline_scales;
    if constexpr (Quantize) {
        baseline_packed = packed.read();
        baseline_scales = scales.read();
        std::vector<float> rounded(transformed.size());
        for (std::size_t i = 0; i < rounded.size(); ++i)
            rounded[i] = hadamard::as_float(transformed[i]);
        const auto cpu = hadamard::quantize_int4(rounded, N);
        equal(baseline_packed, cpu.packed, "baseline packed vs CPU quantization");
        equal(baseline_scales, cpu.scales, "baseline scales vs CPU quantization");
    } else baseline_y = transformed;
    auto validate = [&](const char* phase, int block_threads) {
        context = case_name + " phase=" + phase + " threads=" + std::to_string(block_threads);
        CHECK(cudaStreamSynchronize(stream.handle));
        if constexpr (Quantize) {
            equal(packed.read(), baseline_packed, "packed");
            equal(scales.read(), baseline_scales, "scales");
        } else equal(y.read(), baseline_y, "transform output");
    };
    // 先验证每个配置的全部元素；写入哨兵防止复用旧输出掩盖漏写。
    for (int block_threads : threads) {
        CHECK(cudaMemsetAsync(y.ptr, 0xa5, y.count * sizeof(T), stream.handle));
        CHECK(cudaMemsetAsync(packed.ptr, 0xa5, packed.count, stream.handle));
        CHECK(cudaMemsetAsync(scales.ptr, 0xa5, scales.count * sizeof(float), stream.handle));
        launch(block_threads);
        validate("direct", block_threads);
    }
    std::array<Graph, 4> graphs;
    for (int i = 0; i < 4; ++i) {
        CHECK(cudaStreamBeginCapture(stream.handle, cudaStreamCaptureModeThreadLocal));
        for (int call = 0; call < captured_calls; ++call) launch(threads[i]);
        CHECK(cudaStreamEndCapture(stream.handle, &graphs[i].graph));
        CHECK(cudaGraphInstantiate(&graphs[i].exec, graphs[i].graph, nullptr, nullptr, 0));
        CHECK(cudaGraphLaunch(graphs[i].exec, stream.handle));
        validate("graph_before_timing", threads[i]);
    }
    for (int w = 0; w < warmup; ++w)
        for (int i = 0; i < 4; ++i) CHECK(cudaGraphLaunch(graphs[(i + w) % 4].exec, stream.handle));
    CHECK(cudaStreamSynchronize(stream.handle));
    Event start, stop;
    std::array<std::array<double, 4>, groups> samples{};
    std::array<std::array<int, 4>, groups> order{};
    for (int g = 0; g < groups; ++g) {
        for (int position = 0; position < 4; ++position) {
            // 每组轮换起点并反转方向，避免所有组都按同一线程数顺序执行。
            const int i = (g + (g % 2 ? -position : position) + 4) % 4;
            order[g][i] = position;
            CHECK(cudaEventRecord(start.handle, stream.handle));
            for (int repeat = 0; repeat < replays; ++repeat)
                CHECK(cudaGraphLaunch(graphs[i].exec, stream.handle));
            CHECK(cudaEventRecord(stop.handle, stream.handle));
            CHECK(cudaEventSynchronize(stop.handle));
            float elapsed_ms = 0;
            CHECK(cudaEventElapsedTime(&elapsed_ms, start.handle, stop.handle));
            samples[g][i] = elapsed_ms * 1000.0 / (replays * captured_calls);
        }
    }
    // 计时后仍验证捕获路径；所有检查与分配/复制均在 event 区间外。
    for (int i = 0; i < 4; ++i) {
        CHECK(cudaGraphLaunch(graphs[i].exec, stream.handle));
        validate("graph_after_timing", threads[i]);
    }
    for (int g = 0; g < groups; ++g)
        for (int i = 0; i < 4; ++i)
            std::cout << '"' << gpu.name << "\"," << gpu.major * 10 + gpu.minor << ','
                      << runtime << ',' << seed << ',' << (Quantize ? "fused_int4" : "transform")
                      << ',' << rows << ',' << N << ',' << dtype << ',' << scale << ',' << threads[i]
                      << ',' << g << ',' << order[g][i] << ',' << captured_calls << ',' << replays
                      << ',' << samples[g][i] << ",PASS\n";
    std::cout.flush();
}
template<class T, int N> void cases(const cudaDeviceProp& gpu, int runtime) {
    for (int rows : {1, 17, 4096, 16384})
        for (float scale : {1.0f, 1.0f / std::sqrt(static_cast<float>(N))}) {
            run<T, N, false>(rows, scale, gpu, runtime);
            run<T, N, true>(rows, scale, gpu, runtime);
        }
}
} // namespace

int main(int argc, char**) {
    try {
        if (argc != 1) throw std::invalid_argument("Usage: tune_launch > new-results.csv");
        int device = 0, runtime = 0;
        CHECK(cudaGetDevice(&device));
        cudaDeviceProp gpu{};
        CHECK(cudaGetDeviceProperties(&gpu, device));
        CHECK(cudaRuntimeGetVersion(&runtime));
        if (gpu.major < 8) throw std::runtime_error("This experiment requires sm80 or newer");
        std::cerr << "Timing: fixed input/output buffers, 64 serial kernels per graph, 20 replays, "
                     "5 groups; per-call amortized graph interval, not isolated kernel latency.\n";
        std::cout << std::setprecision(10)
                  << "gpu,sm,cuda_runtime,seed,mode,rows,dim,dtype,scale,threads,group,order,"
                     "captured_calls,replays,mean_us,check_status\n";
        cases<__half, 16>(gpu, runtime); cases<__half, 64>(gpu, runtime); cases<__half, 256>(gpu, runtime);
        cases<__nv_bfloat16, 16>(gpu, runtime); cases<__nv_bfloat16, 64>(gpu, runtime);
        cases<__nv_bfloat16, 256>(gpu, runtime);
        CHECK(cudaDeviceSynchronize());
        std::cerr << "PASS: 96 shape/dtype/scale/mode cases; 384 launch configurations; 1920 raw samples.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
