#include "include/fused_hadamard_quant.cuh"
namespace fused_hq {
const char* version_name(Version v) {
    switch (v) { case Version::kV0: return "fused_v0"; case Version::kV1: return "fused_v1";
                 case Version::kV2: return "fused_v2"; case Version::kSeparate: return "separate"; }
    return "unknown";
}
const char* dtype_name(DType t) {
    switch (t) { case DType::kFloat32: return "fp32"; case DType::kFloat16: return "fp16"; case DType::kBFloat16: return "bf16"; }
    return "unknown";
}
std::size_t workspace_bytes(Version v, std::int64_t batch, int dim) {
    return v == Version::kSeparate ? static_cast<std::size_t>(batch) * dim * sizeof(float) : 0;
}
cudaError_t launch(Version v, const void* input, std::int8_t* output, float* amax, float* scale, void* workspace,
                   std::int64_t batch, int dim, DType dtype, cudaStream_t stream) {
    if (!input || !output || !amax || !scale || batch <= 0 || dim < 32 || dim > 8192 || (dim & (dim - 1)))
        return cudaErrorInvalidValue;
    if (v == Version::kSeparate && !workspace) return cudaErrorInvalidValue;
    switch (v) {
        case Version::kV0: return launch_v0(input, output, amax, scale, batch, dim, dtype, stream);
        case Version::kV1: return launch_v1(input, output, amax, scale, batch, dim, dtype, stream);
        case Version::kV2: return launch_v2(input, output, amax, scale, batch, dim, dtype, stream);
        case Version::kSeparate: return launch_separate(input, output, amax, scale, workspace, batch, dim, dtype, stream);
    }
    return cudaErrorInvalidValue;
}
}  // namespace fused_hq

extern "C" std::size_t fused_hq_workspace_bytes(int version, std::int64_t batch, int dim) {
    return fused_hq::workspace_bytes(static_cast<fused_hq::Version>(version), batch, dim);
}
extern "C" int fused_hq_launch(int version, const void* input, std::int8_t* output,
                                float* amax, float* scale, void* workspace,
                                std::int64_t batch, int dim, int dtype, void* stream) {
    return static_cast<int>(fused_hq::launch(static_cast<fused_hq::Version>(version), input, output,
        amax, scale, workspace, batch, dim, static_cast<fused_hq::DType>(dtype),
        reinterpret_cast<cudaStream_t>(stream)));
}
extern "C" const char* fused_hq_cuda_error_string(int error) {
    return cudaGetErrorString(static_cast<cudaError_t>(error));
}
