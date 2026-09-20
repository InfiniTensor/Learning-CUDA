#include <cstdio>
#include <cuda_runtime.h>
#include <stdexcept>
#define CHECK(x)                                                                                   \
    do {                                                                                           \
        auto e = (x);                                                                              \
        if (e != cudaSuccess)                                                                      \
            throw std::runtime_error(cudaGetErrorString(e));                                       \
    } while (0)
__global__ void probe(int *x) {
    x[threadIdx.x] = int(threadIdx.x) * 3 + 1;
}
int main() {
    try {
        int n = 0;
        CHECK(cudaGetDeviceCount(&n));
        std::printf("device_count=%d\n", n);
        if (!n)
            return 1;
        cudaDeviceProp p{};
        CHECK(cudaGetDeviceProperties(&p, 0));
        std::printf("device=%s cc=%d.%d memory=%zu\n", p.name, p.major, p.minor, p.totalGlobalMem);
        int *d = nullptr;
        CHECK(cudaMalloc(&d, 128 * sizeof(int)));
        probe<<<1, 128>>>(d);
        CHECK(cudaGetLastError());
        CHECK(cudaDeviceSynchronize());
        int h[128];
        CHECK(cudaMemcpy(h, d, sizeof(h), cudaMemcpyDeviceToHost));
        CHECK(cudaFree(d));
        for (int i = 0; i < 128; ++i)
            if (h[i] != i * 3 + 1)
                return 2;
        puts("allocation/kernel/synchronize/result PASS");
    } catch (const std::exception &e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
