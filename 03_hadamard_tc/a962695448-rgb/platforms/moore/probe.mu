#include <musa_runtime.h>
#include <musa_fp16.h>
#include <musa_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CHECK(call) do { musaError_t e = (call); if (e != musaSuccess) { \
    std::fprintf(stderr, "%s: %s\n", #call, musaGetErrorString(e)); return 2; } } while (0)

__global__ void probe(int* out, float* numeric) {
    const int tid = threadIdx.x;
    out[tid * 22] = warpSize;
    int slot = 1;
    for (int width = 32; width <= 32; width *= 2) {
        for (int step = 1; step < width; step *= 2) {
            out[tid * 22 + slot++] = __shfl_xor_sync(0xffffffffu, tid, step, width);
        }
    }
    float x = (tid - 128) * 0.125f;
    numeric[tid * 3] = __half2float(__float2half_rn(x));
    numeric[tid * 3 + 1] = __bfloat162float(__float2bfloat16_rn(x));
    const float scale = 1.0f / 7.0f;
    numeric[tid * 3 + 2] = 0.5f / scale;
}

int main() {
    musaDeviceProp p{};
    CHECK(musaSetDevice(0));
    CHECK(musaGetDeviceProperties(&p, 0));
    std::printf("DEVICE name=%s arch=%d.%d warp=%d memory=%zu max_threads=%d sm=%d\n",
                p.name, p.major, p.minor, p.warpSize, p.totalGlobalMem,
                p.maxThreadsPerBlock, p.multiProcessorCount);
    int runtime = 0, driver = 0;
    CHECK(musaRuntimeGetVersion(&runtime)); CHECK(musaDriverGetVersion(&driver));
    std::printf("VERSIONS runtime=%d driver=%d half_bytes=%zu bf16_bytes=%zu\n",
                runtime, driver, sizeof(__half), sizeof(__mt_bfloat16));
    if (p.warpSize != 128) { std::puts("UNEXPECTED_WARP_SIZE"); return 3; }
    int* d = nullptr; float* f = nullptr;
    CHECK(musaMalloc(reinterpret_cast<void**>(&d), 256 * 22 * sizeof(int)));
    CHECK(musaMalloc(reinterpret_cast<void**>(&f), 256 * 3 * sizeof(float)));
    probe<<<1, 256>>>(d, f);
    CHECK(musaGetLastError()); CHECK(musaDeviceSynchronize());
    std::vector<int> out(256 * 22);
    std::vector<float> numeric(256 * 3);
    CHECK(musaMemcpy(out.data(), d, out.size()*sizeof(int), musaMemcpyDeviceToHost));
    CHECK(musaMemcpy(numeric.data(), f, numeric.size()*sizeof(float), musaMemcpyDeviceToHost));
    int bad = 0, checks = 0, numeric_bad = 0;
    int width_bad[3]{};
    std::printf("BUILTIN_WARP first=%d last=%d\n",out[0],out[255*22]);
    for (int tid = 0; tid < 256; ++tid) {
        if (out[tid * 22] != 32) ++bad;
        int slot = 1;
        for (int width = 32; width <= 32; width *= 2) {
            for (int step = 1; step < width; step *= 2) {
                const int expected = (tid / width) * width + ((tid % width) ^ step);
                const int actual = out[tid * 22 + slot++];
                if (actual != expected) {
                    if (bad < 8) std::printf("SHUFFLE_FAIL tid=%d width=%d step=%d got=%d expected=%d\n", tid,width,step,actual,expected);
                    ++bad;
                    ++width_bad[width == 32 ? 0 : (width == 64 ? 1 : 2)];
                }
                ++checks;
            }
        }
        const float expected = (tid - 128) * 0.125f;
        if (numeric[tid*3] != expected || numeric[tid*3+1] != expected) { ++bad; ++numeric_bad; }
    }
    std::printf("RESULT shuffle_checks=%d failures=%d division_probe=%.9g\n",checks,bad,numeric[2]);
    std::printf("SUPPORTED_COUNTS width32_failures=%d numeric_failures=%d\n",width_bad[0],numeric_bad);
    CHECK(musaFree(d)); CHECK(musaFree(f));
    return bad ? 1 : 0;
}
