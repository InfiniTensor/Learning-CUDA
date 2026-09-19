// Iluvatar uses the shared CUDA-compatible implementation until a platform-
// specific kernel is needed. Keeping this entry separate lets CMake select
// the backend without copying the quantization logic.
#include "../nvidia/cuda_quant.cu"
