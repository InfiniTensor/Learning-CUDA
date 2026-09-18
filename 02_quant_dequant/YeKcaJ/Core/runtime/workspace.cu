// 唯一包含正式 kernel 定义的编译单元；按格式和选项选择执行路径。
#include "runtime/workspace.cuh"
#include "kernels/reduce.cuh"
#ifdef LP_BACKEND_MUSA
#include "backends/musa/quantize.cuh"
#else
#include "kernels/mxfp8_quantize.cuh"
#include "kernels/nvfp4_quantize.cuh"
#endif
#include "kernels/fallback_quantize.cuh"
#include "kernels/dequantize.cuh"

namespace pipeline {

void Workspace::launch_quant(bool baseline) {
  if (!n) return;

#ifdef LP_BACKEND_MUSA
  if (!options.tensor && !options.stochastic && !baseline) {
    if (nvfp4) {
      maximum<<<partials, 256>>>(input.ptr, scratch.ptr, n);
      finalize_max<<<1, 256>>>(scratch.ptr, state.ptr, partials);
    }
    musa_quantize_vectorized_kernel<<<static_cast<unsigned>(ceil_div(n, 1024)), 256>>>(
        input.ptr, data.ptr, scales.ptr, state.ptr, n);
    return;
  }
#else
  if (!nvfp4 && !options.tensor && !options.stochastic && !baseline) {
    // 256 threads / 8 lanes per group = 32 groups = 1024 elements per block。
    mxfp8_quantize_vectorized_kernel<<<static_cast<unsigned>(ceil_div(n, 1024)), 256>>>(
        input.ptr, data.ptr, scales.ptr, n);
    return;
  }

  // NVFP4 block+nearest：全局归约仍是必要前置（global_scale 依赖全张量最大值），
  // 之后的 scale 计算与编码合并为一个 kernel，input 由读 3 遍降为读 2 遍。
  if (nvfp4 && !options.tensor && !options.stochastic && !baseline) {
    maximum<<<partials, 256>>>(input.ptr, scratch.ptr, n);
    finalize_max<<<1, 256>>>(scratch.ptr, state.ptr, partials);
    nvfp4_quantize_fused_kernel<<<static_cast<unsigned>(ceil_div(data_size(n), 256)), 256>>>(
        input.ptr, data.ptr, scales.ptr, state.ptr, n);
    return;
  }

#endif
  // MXFP8 block 模式只需局部最大值，跳过全张量归约。
  if (nvfp4 || options.tensor) {
    maximum<<<partials, 256>>>(input.ptr, scratch.ptr, n);
    finalize_max<<<1, 256>>>(scratch.ptr, state.ptr, partials);
  }

  if (options.tensor || baseline)
    build_scales<<<static_cast<unsigned>(groups), 32>>>(
        input.ptr, state.ptr, scales.ptr, n, options.tensor);
  else
    build_block_scales<<<static_cast<unsigned>(ceil_div(n, 256)), 256>>>(
        input.ptr, state.ptr, scales.ptr, n);

  // 按输出字节分配线程，因此 NVFP4 的线程数约为元素数的一半。
  const unsigned grid = static_cast<unsigned>(ceil_div(data_size(n), 256));
  if (baseline)
    quantize_kernel<true><<<grid, 256>>>(
        input.ptr, data.ptr, scales.ptr, state.ptr, n, group,
        options.stochastic, options.seed);
  else
    quantize_kernel<false><<<grid, 256>>>(
        input.ptr, data.ptr, scales.ptr, state.ptr, n, group,
        options.stochastic, options.seed);
}

template <typename T>
void Workspace::launch_dequant() {
  if (n)
    dequantize_kernel<<<static_cast<unsigned>(ceil_div(n, 256)), 256>>>(
        data.ptr, scales.ptr, state.ptr, reinterpret_cast<T*>(output.ptr), n, group);
}

// 三种目标类型在此实例化，调用方无需包含 kernel 实现。
template void Workspace::launch_dequant<float>();
template void Workspace::launch_dequant<__half>();
template void Workspace::launch_dequant<__nv_bfloat16>();

}  // namespace pipeline
