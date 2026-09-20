#pragma once
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

namespace fused_hq {
enum class Version : int { kV0 = 0, kV1 = 1, kV2 = 2, kSeparate = 3 };
enum class DType : int { kFloat32 = 0, kFloat16 = 1, kBFloat16 = 2 };

const char* version_name(Version version);
const char* dtype_name(DType dtype);
std::size_t workspace_bytes(Version version, std::int64_t batch, int dim);
cudaError_t launch(Version version, const void* input, std::int8_t* quantized,
                   float* amax, float* scale, void* workspace,
                   std::int64_t batch, int dim, DType dtype,
                   cudaStream_t stream = nullptr);

cudaError_t launch_v0(const void*, std::int8_t*, float*, float*, std::int64_t, int, DType, cudaStream_t);
cudaError_t launch_v1(const void*, std::int8_t*, float*, float*, std::int64_t, int, DType, cudaStream_t);
cudaError_t launch_v2(const void*, std::int8_t*, float*, float*, std::int64_t, int, DType, cudaStream_t);
cudaError_t launch_separate(const void*, std::int8_t*, float*, float*, void*, std::int64_t, int, DType, cudaStream_t);
}  // namespace fused_hq
