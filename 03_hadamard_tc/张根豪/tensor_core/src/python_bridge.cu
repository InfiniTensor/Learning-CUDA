// Narrow C ABI for measurements on PyTorch's current CUDA stream.
// No Python/Torch ABI dependency; callers own device buffers and their lifetime.
#include "hadamard.cuh"
#include <climits>
#ifdef HADAMARD_BRIDGE_TC
#include "hadamard_tc.cuh"
#endif
#ifdef HADAMARD_BRIDGE_MMA
extern "C" int hadamard_mma_run(const void*, void*, int, int, int, int, int, cudaStream_t);
#endif

extern "C" int hadamard_run(const void* x, void* y, float* scales,
                            int tokens, int dim, int dtype, int normalize,
                            int mode, cudaStream_t stream) {
  if (!x || !y || tokens <= 0 || dim < 64 || dim > 1024 ||
      (dim & (dim - 1)) || tokens > INT_MAX / dim ||
      dtype < 0 || dtype > 1 || normalize < 0 || normalize > 1) return -10;
  const DataType dt = dtype == 0 ? DataType::FP16 : DataType::BF16;
  switch (mode) {
    case 0: return launch_hadamard(x, y, 1, 1, tokens, dim, dt, normalize, stream);
    case 1: return launch_hadamard_optimized(x, y, 1, 1, tokens, dim, dt, normalize, stream);
    case 2: return launch_hadamard_warp(x, y, 1, 1, tokens, dim, dt, normalize, stream);
    case 3:
      if (!scales) return -10;
      return launch_quantize_int4(x, static_cast<unsigned char*>(y), scales, tokens, dim, dt, stream);
    case 4:
      if (!scales) return -10;
      return launch_hadamard_fused_quant_int4(x, static_cast<unsigned char*>(y), scales,
                                            1, 1, tokens, dim, dt, normalize, stream);
    case 5:
      if (!scales) return -10;
      return launch_hadamard_fused_quant_int4_warp(x, static_cast<unsigned char*>(y), scales,
                                                 1, 1, tokens, dim, dt, normalize, stream);
#ifdef HADAMARD_BRIDGE_TC
    case 6: case 7:
      return launch_hadamard_tc(x, y, 1, 1, tokens, dim, dt, normalize,
                                mode == 6 ? TcMode::Fast : TcMode::Split, stream);
#endif
#ifdef HADAMARD_BRIDGE_MMA
    case 8: case 9: return hadamard_mma_run(x,y,tokens,dim,dtype,normalize,mode==9,stream);
#endif
    default: return -10;
  }
}
