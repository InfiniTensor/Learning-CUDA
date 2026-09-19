#include "lp.hpp"
#include <algorithm>
#include <cuda_runtime.h>
#include <stdexcept>

namespace lp {
namespace {
void check(cudaError_t e, const char *operation) {
    if (e != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(e));
}
#define CUDA_CHECK(call) check((call), #call)
template <typename T> struct DeviceBuffer {
    T *ptr = nullptr;
    explicit DeviceBuffer(size_t n) { CUDA_CHECK(cudaMalloc(&ptr, n * sizeof(T))); }
    ~DeviceBuffer() { cudaFree(ptr); }
    DeviceBuffer(const DeviceBuffer &) = delete;
    DeviceBuffer &operator=(const DeviceBuffer &) = delete;
};
struct Event {
    cudaEvent_t event{};
    Event() { CUDA_CHECK(cudaEventCreate(&event)); }
    ~Event() { cudaEventDestroy(event); }
};
template <typename Fn> double benchmark(Fn fn, int warmup, int repeats) {
    for (int k = 0; k < warmup; ++k)
        fn();
    Event begin, end;
    CUDA_CHECK(cudaEventRecord(begin.event));
    for (int k = 0; k < repeats; ++k)
        fn();
    CUDA_CHECK(cudaEventRecord(end.event));
    CUDA_CHECK(cudaEventSynchronize(end.event));
    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, begin.event, end.event));
    return ms / repeats;
}
template <typename T> __device__ float load(const T *p, uint64_t i) {
    if constexpr (sizeof(T) == 4)
        return p[i];
    else
        return from_half(p[i]);
}
template <int Out> __device__ void store(void *p, uint64_t i, float x) {
    if constexpr (Out == 0)
        static_cast<float *>(p)[i] = x;
    else
        static_cast<uint16_t *>(p)[i] = Out == 1 ? to_half(x) : to_bfloat(x);
}
__device__ float cta_max(float value) {
    __shared__ float shared[8];
    for (int delta = 16; delta; delta /= 2)
        value = fmaxf(value, __shfl_down_sync(0xffffffffU, value, delta));
    if ((threadIdx.x & 31) == 0)
        shared[threadIdx.x / 32] = value;
    __syncthreads();
    value = threadIdx.x < 8 ? shared[threadIdx.x] : 0;
    if (threadIdx.x < 32)
        for (int delta = 16; delta; delta /= 2)
            value = fmaxf(value, __shfl_down_sync(0xffffffffU, value, delta));
    return value;
}
template <typename T> __global__ void reduce_amax(const T *input, uint64_t n, float *partial) {
    float amax = 0;
    for (uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += uint64_t(blockDim.x) * gridDim.x)
        amax = fmaxf(amax, fabsf(load(input, i)));
    amax = cta_max(amax);
    if (threadIdx.x == 0)
        partial[blockIdx.x] = amax;
}
template <bool NV>
__global__ void finish_amax(const float *partial, int n, float *global, uint8_t *scales,
                            bool tensor) {
    float amax = 0;
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        amax = fmaxf(amax, partial[i]);
    amax = cta_max(amax);
    if (threadIdx.x == 0) {
        const float g = NV ? nv_global(amax) : 1;
        *global = g;
        if (tensor)
            scales[0] = NV ? nv_scale(amax, g) : mx_scale(amax);
    }
}
template <bool NV>
__device__ uint8_t quant_code(float x, float scale, float global, bool stochastic, uint64_t index,
                              uint64_t seed) {
    float y;
    if constexpr (NV)
        y = scale == 0 ? copysignf(0.0f, x) : (x / global) / scale;
    else
        y = x / scale;
    const float u = stochastic ? random_unit(index, seed) : 0;
    return NV ? encode_e2m1(y, stochastic, u) : encode_e4m3(y, stochastic, u);
}

// One subgroup per scale block. NVFP4 pairs exchange codes in registers; only
// even lanes write, so no byte is written by two threads (including odd rows).
template <typename T, bool NV>
__global__ void quantize_blocks(const T *input, uint8_t *data, uint8_t *scales,
                                const float *global_ptr, uint64_t rows, uint64_t cols,
                                unsigned block_size, bool stochastic, uint64_t seed) {
    const unsigned width = block_size < 32 ? block_size : 32;
    const unsigned lane = threadIdx.x % width;
    const uint64_t bpr = (cols + block_size - 1) / block_size;
    const uint64_t groups = rows * bpr;
    const uint64_t stride = uint64_t(gridDim.x) * (256 / width);
    const float global = NV ? *global_ptr : 1;
    for (uint64_t base = uint64_t(blockIdx.x) * (256 / width); base < groups; base += stride) {
        const uint64_t group = base + threadIdx.x / width;
        const unsigned mask = __ballot_sync(0xffffffffU, group < groups);
        if (group < groups) {
            const uint64_t row = group / bpr, col0 = (group % bpr) * block_size;
            float amax = 0;
            // Default sizes keep the input in a register across the reduction.
            float first = col0 + lane < cols ? load(input, row * cols + col0 + lane) : 0;
            amax = fabsf(first);
            for (unsigned j = lane + width; j < block_size && col0 + j < cols; j += width)
                amax = fmaxf(amax, fabsf(load(input, row * cols + col0 + j)));
            for (int delta = width / 2; delta; delta /= 2)
                amax = fmaxf(amax, __shfl_down_sync(mask, amax, delta, width));
            amax = __shfl_sync(mask, amax, 0, width);
            const uint8_t sc = NV ? nv_scale(amax, global) : mx_scale(amax);
            if (lane == 0)
                scales[group] = sc;
            const float scale = NV ? decode_e4m3(sc) : decode_e8m0(sc);
            for (unsigned j = lane; j < block_size; j += width) {
                const uint64_t col = col0 + j;
                float x = j == lane ? first : (col < cols ? load(input, row * cols + col) : 0);
                unsigned code = col < cols ? quant_code<NV>(x, scale, global, stochastic,
                                                            row * cols + col, seed)
                                           : 0;
                if constexpr (NV) {
                    const unsigned next = __shfl_down_sync(mask, code, 1, width);
                    if ((lane & 1) == 0 && col < cols)
                        data[row * ((cols + 1) / 2) + col / 2] = code | (next << 4);
                } else if (col < cols)
                    data[row * cols + col] = code;
            }
        }
    }
}
template <typename T, bool NV>
__global__ void quantize_tensor(const T *input, uint8_t *data, const uint8_t *scales,
                                const float *global_ptr, uint64_t rows, uint64_t cols,
                                bool stochastic, uint64_t seed) {
    const uint64_t pitch = NV ? (cols + 1) / 2 : cols;
    const float scale = NV ? decode_e4m3(scales[0]) : decode_e8m0(scales[0]);
    const float global = NV ? *global_ptr : 1;
    for (uint64_t p = uint64_t(blockIdx.x) * 256 + threadIdx.x; p < rows * pitch;
         p += uint64_t(gridDim.x) * 256) {
        if constexpr (!NV)
            data[p] = quant_code<false>(load(input, p), scale, global, stochastic, p, seed);
        else {
            const uint64_t row = p / pitch, col = (p % pitch) * 2, i = row * cols + col;
            uint8_t code = quant_code<true>(load(input, i), scale, global, stochastic, i, seed);
            if (col + 1 < cols)
                code |= quant_code<true>(load(input, i + 1), scale, global, stochastic, i + 1, seed)
                        << 4;
            data[p] = code;
        }
    }
}
template <bool NV, int Out>
__global__ void dequantize(const uint8_t *data, const uint8_t *scales, const float *global_ptr,
                           void *output, uint64_t rows, uint64_t cols, unsigned block_size,
                           bool tensor) {
    const uint64_t pitch = NV ? (cols + 1) / 2 : cols;
    const uint64_t bpr = (cols + block_size - 1) / block_size;
    const float global = NV ? *global_ptr : 1;
    for (uint64_t p = uint64_t(blockIdx.x) * 256 + threadIdx.x; p < rows * pitch;
         p += uint64_t(gridDim.x) * 256) {
        const uint64_t row = p / pitch, col = (p % pitch) * (NV ? 2 : 1);
        const uint64_t group = tensor ? 0 : row * bpr + col / block_size;
        const float scale = NV ? decode_e4m3(scales[group]) : decode_e8m0(scales[group]);
        const uint8_t packed = data[p]; // Exactly one packed byte load for two FP4 values.
        const float x = NV ? decode_e2m1(packed & 15) : decode_e4m3(packed);
        store<Out>(output, row * cols + col, (x * scale) * global);
        if constexpr (NV)
            if (col + 1 < cols)
                store<Out>(output, row * cols + col + 1,
                           (decode_e2m1(packed >> 4) * scale) * global);
    }
}
unsigned grid_for(uint64_t work) {
    return static_cast<unsigned>(std::min<uint64_t>((work + 255) / 256, 4096));
}
struct Buffers {
    DeviceBuffer<uint8_t> input, data, scales, output;
    DeviceBuffer<float> global, partial;
    Buffers(size_t in, const Weights &w, const Tensor &out)
        : input(std::max<size_t>(in, 1)), data(w.data.size()), scales(w.scales.size()),
          output(out.data.size()), global(1), partial(4096) {}
};
Timings device_info() {
    Timings t;
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    if (prop.major < 6)
        throw std::runtime_error("This build targets compute capability 6.0 or later");
    t.device = prop.name;
    t.compute_capability = std::to_string(prop.major) + "." + std::to_string(prop.minor);
    CUDA_CHECK(cudaRuntimeGetVersion(&t.runtime_version));
    CUDA_CHECK(cudaDriverGetVersion(&t.driver_version));
    return t;
}
template <typename T, bool NV> void launch_quant(const Tensor &in, const Weights &w, Buffers &b) {
    const T *input = reinterpret_cast<const T *>(b.input.ptr);
    const auto &c = w.config;
    if (NV || c.tensor_scale) {
        const unsigned blocks = grid_for(in.size());
        reduce_amax<<<blocks, 256>>>(input, in.size(), b.partial.ptr);
        finish_amax<NV>
            <<<1, 256>>>(b.partial.ptr, blocks, b.global.ptr, b.scales.ptr, c.tensor_scale);
    }
    if (c.tensor_scale)
        quantize_tensor<T, NV><<<grid_for(w.data.size()), 256>>>(
            input, b.data.ptr, b.scales.ptr, b.global.ptr, w.rows, w.cols, c.stochastic, c.seed);
    else {
        const unsigned width = std::min(c.block_size, 32U);
        quantize_blocks<T, NV><<<grid_for(w.scales.size() * width), 256>>>(
            input, b.data.ptr, b.scales.ptr, b.global.ptr, w.rows, w.cols, c.block_size,
            c.stochastic, c.seed);
    }
    CUDA_CHECK(cudaGetLastError());
}
void launch_quant(const Tensor &in, const Weights &w, Buffers &b) {
    if (in.dtype == DType::FP32) {
        if (w.config.format == Format::NVFP4)
            launch_quant<float, true>(in, w, b);
        else
            launch_quant<float, false>(in, w, b);
    } else {
        if (w.config.format == Format::NVFP4)
            launch_quant<uint16_t, true>(in, w, b);
        else
            launch_quant<uint16_t, false>(in, w, b);
    }
}
template <bool NV> void launch_decode_format(const Weights &w, const Tensor &out, Buffers &b) {
#define DECODE(T)                                                                                  \
    dequantize<NV, T><<<grid_for(w.data.size()), 256>>>(                                           \
        b.data.ptr, b.scales.ptr, b.global.ptr, b.output.ptr, w.rows, w.cols, w.config.block_size, \
        w.config.tensor_scale)
    if (out.dtype == DType::FP32) {
        DECODE(0);
    } else if (out.dtype == DType::FP16) {
        DECODE(1);
    } else {
        DECODE(2);
    }
#undef DECODE
    CUDA_CHECK(cudaGetLastError());
}
void launch_decode(const Weights &w, const Tensor &out, Buffers &b) {
    if (w.config.format == Format::NVFP4)
        launch_decode_format<true>(w, out, b);
    else
        launch_decode_format<false>(w, out, b);
}
void download(Weights &w, Tensor &out, Buffers &b) {
    CUDA_CHECK(cudaMemcpy(w.data.data(), b.data.ptr, w.data.size(), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(w.scales.data(), b.scales.ptr, w.scales.size(), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&w.global_scale, b.global.ptr, 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(out.data.data(), b.output.ptr, out.data.size(), cudaMemcpyDeviceToHost));
}
} // namespace
Timings run_cuda(const Tensor &input, Weights &w, Tensor &out, int warmup, int repeats) {
    Timings t = device_info();
    Buffers b(input.data.size(), w, out);
    const float one = 1;
    CUDA_CHECK(cudaMemcpy(b.global.ptr, &one, 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(
        cudaMemcpy(b.input.ptr, input.data.data(), input.data.size(), cudaMemcpyHostToDevice));
    t.quant_ms = benchmark([&] { launch_quant(input, w, b); }, warmup, repeats);
    t.dequant_ms = benchmark([&] { launch_decode(w, out, b); }, warmup, repeats);
    // Wall-clock pipeline includes synchronous pageable H2D and every output D2H.
    const int pipeline_repeats = std::min(repeats, 5);
    const double start = now_ms();
    for (int k = 0; k < pipeline_repeats; ++k) {
        CUDA_CHECK(
            cudaMemcpy(b.input.ptr, input.data.data(), input.data.size(), cudaMemcpyHostToDevice));
        launch_quant(input, w, b);
        launch_decode(w, out, b);
        download(w, out, b);
    }
    t.transfer_pipeline_ms = elapsed_ms(start) / pipeline_repeats;
    t.quant_bytes = double(input.data.size()) + w.data.size() + w.scales.size();
    if (w.config.block_size > 32 && !w.config.tensor_scale) {
        const uint64_t full = w.cols / w.config.block_size;
        const uint64_t tail = w.cols % w.config.block_size;
        const uint64_t reread = full * (w.config.block_size - 32) + (tail > 32 ? tail - 32 : 0);
        t.quant_bytes += double(w.rows * reread) * dtype_bytes(input.dtype);
    }
    if (w.config.format == Format::NVFP4 || w.config.tensor_scale)
        t.quant_bytes += double(input.data.size()) + 8.0 * grid_for(input.size()) + 4;
    t.dequant_bytes = double(w.data.size()) + w.scales.size() + out.data.size() +
                      (w.config.format == Format::NVFP4 ? 4 : 0);
    return t;
}
Timings decode_cuda(const Weights &w, Tensor &out, int warmup, int repeats) {
    Timings t = device_info();
    Buffers b(0, w, out);
    CUDA_CHECK(cudaMemcpy(b.data.ptr, w.data.data(), w.data.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(b.scales.ptr, w.scales.data(), w.scales.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(b.global.ptr, &w.global_scale, 4, cudaMemcpyHostToDevice));
    t.dequant_ms = benchmark([&] { launch_decode(w, out, b); }, warmup, repeats);
    CUDA_CHECK(cudaMemcpy(out.data.data(), b.output.ptr, out.data.size(), cudaMemcpyDeviceToHost));
    return t;
}
} // namespace lp
