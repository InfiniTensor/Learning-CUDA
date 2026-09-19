# Iluvatar Backend

The Iluvatar CoreX 4.x toolchain compiles the CUDA-compatible kernel source in
`../nvidia/cuda_quant.cu` with `clang++ -x ivcore`. The public API and packed
MXFP8/NVFP4 format are shared with the NVIDIA backend; this directory records
the backend contract without duplicating the kernel implementation.
