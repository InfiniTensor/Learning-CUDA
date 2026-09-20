// M0 probe: verifies the toolchain, the device, and the Tensor Core path.
//
// This is the executable the M0 deliverable asks for: it must build against
// CUDA 12.8 for sm_89 and run cleanly on the RTX 4090.
#include <cuda_runtime.h>

#include <cstdio>
#include <exception>
#include <initializer_list>

#include "hadamard/common.h"
#include "hadamard/cuda_utils.h"
#include "hadamard/hadamard.h"

int main() {
    using namespace hadamard;
    int failures = 0;

    try {
        const DeviceInfo info = query_device(0);
        std::printf("device         : %s (sm_%d%d, %d SMs)\n", info.name.c_str(), info.major,
                    info.minor, info.sm_count);
        std::printf("driver/runtime : %d / %d\n", info.driver_version, info.runtime_version);
        std::printf("global memory  : %.2f GiB\n",
                    static_cast<double>(info.global_mem_bytes) / (1024.0 * 1024.0 * 1024.0));
        std::printf("scope          : head_dim 64/128/256 (powers of two), fp16 and bf16\n\n");

        int* d_out = nullptr;
        HW_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_out), sizeof(int)));
        launch_hello(d_out, 0xC0DA);
        int host_out = 0;
        HW_CUDA_CHECK(cudaMemcpy(&host_out, d_out, sizeof(int), cudaMemcpyDeviceToHost));
        HW_CUDA_CHECK(cudaFree(d_out));
        const bool hello_ok = (host_out == 0xC0DA);
        std::printf("[%s] hello kernel round-trip (got 0x%04X)\n", hello_ok ? "PASS" : "FAIL",
                    static_cast<unsigned>(host_out));
        failures += hello_ok ? 0 : 1;

        for (DType dtype : {DType::kFp16, DType::kBFloat16}) {
            bool ok = false;
            if (launch_wmma_identity_probe(dtype, &ok)) {
                std::printf("[%s] WMMA 16x16x16 identity probe (%s)\n", ok ? "PASS" : "FAIL",
                            dtype_name(dtype));
                failures += ok ? 0 : 1;
            } else {
                std::printf("[SKIP] WMMA probe (%s) unsupported on this device\n",
                            dtype_name(dtype));
            }
        }

        std::printf("\nM0 probe       : %s\n", failures == 0 ? "OK" : "FAILED");
    } catch (const std::exception& e) {
        std::fprintf(stderr, "fatal: %s\n", e.what());
        return 2;
    }

    return failures == 0 ? 0 : 1;
}
