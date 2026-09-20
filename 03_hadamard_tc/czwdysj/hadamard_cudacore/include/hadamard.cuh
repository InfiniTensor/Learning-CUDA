#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace hadamard {

enum class KernelVersion : int {
    kV0 = 0,  // Global-memory, one launch per FWHT stage.
    kV1 = 1,  // One block per row, FP32 shared-memory butterflies.
    kV2 = 2,  // Vectorized I/O, warp shuffles, then shared-memory butterflies.
};

enum class DType : int {
    kFloat32 = 0,
    kFloat16 = 1,
    kBFloat16 = 2,
};

const char* kernel_name(KernelVersion version);
const char* dtype_name(DType dtype);
std::size_t dtype_size(DType dtype);
bool is_power_of_two(int value);

std::size_t workspace_bytes(KernelVersion version, std::int64_t batch,
                            int head_dim, DType dtype);

// Every version computes x @ H(head_dim) / sqrt(head_dim) on the final,
// contiguous dimension. Input and output must not alias.
cudaError_t launch(KernelVersion version, const void* input, void* output,
                   void* workspace, std::int64_t batch, int head_dim,
                   DType dtype, cudaStream_t stream = nullptr);

cudaError_t launch_hadamard_v0(const void* input, void* output,
                               void* workspace, std::int64_t batch,
                               int head_dim, DType dtype,
                               cudaStream_t stream);
cudaError_t launch_hadamard_v1(const void* input, void* output,
                               std::int64_t batch, int head_dim, DType dtype,
                               cudaStream_t stream);
cudaError_t launch_hadamard_v2(const void* input, void* output,
                               std::int64_t batch, int head_dim, DType dtype,
                               cudaStream_t stream);

}  // namespace hadamard

// Stable, version-neutral C ABI used by the Python benchmark through ctypes.
// Version values are the enum values above.
extern "C" {
std::size_t hadamard_workspace_bytes(int version, std::int64_t batch,
                                     int head_dim, int dtype);
int hadamard_launch(int version, const void* input, void* output,
                    void* workspace, std::int64_t batch, int head_dim,
                    int dtype, void* stream);
const char* hadamard_cuda_error_string(int error);
}
