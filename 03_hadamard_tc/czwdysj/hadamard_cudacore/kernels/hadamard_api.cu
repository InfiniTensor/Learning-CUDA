#include "include/hadamard.cuh"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace hadamard {

const char* kernel_name(KernelVersion version) {
    switch (version) {
        case KernelVersion::kV0: return "hadamard_v0";
        case KernelVersion::kV1: return "hadamard_v1";
        case KernelVersion::kV2: return "hadamard_v2";
    }
    return "unknown";
}

const char* dtype_name(DType dtype) {
    switch (dtype) {
        case DType::kFloat32: return "FP32";
        case DType::kFloat16: return "FP16";
        case DType::kBFloat16: return "BF16";
    }
    return "UNKNOWN";
}

std::size_t dtype_size(DType dtype) { return dtype == DType::kFloat32 ? 4 : 2; }

bool is_power_of_two(int value) { return value > 0 && (value & (value - 1)) == 0; }

std::size_t workspace_bytes(KernelVersion version, std::int64_t batch,
                            int head_dim, DType dtype) {
    if (version != KernelVersion::kV0 || batch <= 0 || head_dim <= 0) return 0;
    return static_cast<std::size_t>(batch) * head_dim * dtype_size(dtype);
}

cudaError_t launch(KernelVersion version, const void* input, void* output,
                   void* workspace, std::int64_t batch, int head_dim,
                   DType dtype, cudaStream_t stream) {
    if (input == nullptr || output == nullptr || input == output || batch <= 0 ||
        batch > static_cast<std::int64_t>(0x7fffffffU) || head_dim < 32 ||
        head_dim > 8192 || !is_power_of_two(head_dim)) return cudaErrorInvalidValue;
    int device = 0;
    cudaError_t error = cudaGetDevice(&device);
    if (error != cudaSuccess) return error;
    int max_shared = 0;
    error = cudaDeviceGetAttribute(&max_shared, cudaDevAttrMaxSharedMemoryPerBlock, device);
    if (error != cudaSuccess) return error;
    if (version != KernelVersion::kV0 && head_dim * static_cast<int>(sizeof(float)) > max_shared)
        return cudaErrorNotSupported;
    switch (version) {
        case KernelVersion::kV0: return launch_hadamard_v0(input, output, workspace, batch, head_dim, dtype, stream);
        case KernelVersion::kV1: return launch_hadamard_v1(input, output, batch, head_dim, dtype, stream);
        case KernelVersion::kV2: return launch_hadamard_v2(input, output, batch, head_dim, dtype, stream);
    }
    return cudaErrorInvalidValue;
}

}  // namespace hadamard

extern "C" std::size_t hadamard_workspace_bytes(int version, std::int64_t batch,
                                                  int head_dim, int dtype) {
    return hadamard::workspace_bytes(static_cast<hadamard::KernelVersion>(version), batch,
                                     head_dim, static_cast<hadamard::DType>(dtype));
}

extern "C" int hadamard_launch(int version, const void* input, void* output,
                                void* workspace, std::int64_t batch, int head_dim,
                                int dtype, void* stream) {
    return static_cast<int>(hadamard::launch(
        static_cast<hadamard::KernelVersion>(version), input, output, workspace,
        batch, head_dim, static_cast<hadamard::DType>(dtype),
        reinterpret_cast<cudaStream_t>(stream)));
}

extern "C" const char* hadamard_cuda_error_string(int error) {
    return cudaGetErrorString(static_cast<cudaError_t>(error));
}
