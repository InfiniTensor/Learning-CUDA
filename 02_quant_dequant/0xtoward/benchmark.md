# Kernel benchmark

NVIDIA GeForce RTX 4070 Laptop GPU

skill CUDA events; prewarm1, warmup5, 31 samples, discard1, 256MiB cache thrash before each sample

FP32 input/output; row-wise groups; MX uses FLOOR. NV tensor amax is precomputed.
All compared packed bytes, scales and FP32 reconstructions match exactly.

| Shape | Format | Implementation | Quant median (ms) | Dequant median (ms) | Quant GB/s | Dequant GB/s |
|---|---|---|---:|---:|---:|---:|
| 1024x3584 | mxfp8 | manual_cuda | 0.1956 | 0.1311 | 94.4 | 140.9 |
| 1024x3584 | mxfp8 | triton | 0.0922 | 0.0758 | 200.4 | 243.7 |
| 1024x3584 | mxfp8 | cuda_math_api | 0.1403 | 0.1014 | 131.6 | 182.1 |
| 1024x3584 | nvfp4 | manual_cuda | 0.3400 | 0.1423 | 49.3 | 117.6 |
| 1024x3584 | nvfp4 | triton | 0.1004 | 0.0676 | 166.9 | 247.8 |
| 1024x3584 | nvfp4+4over6 | manual_cuda | 0.6502 | 0.1423 | 25.8 | 117.6 |
| 1024x3584 | nvfp4+4over6 | triton | 0.1321 | 0.0686 | 126.8 | 244.1 |
| 4096x4096 | mxfp8 | manual_cuda | 0.7352 | 0.4280 | 114.8 | 197.2 |
| 4096x4096 | mxfp8 | triton | 0.3850 | 0.3389 | 219.2 | 249.1 |
| 4096x4096 | mxfp8 | cuda_math_api | 0.5847 | 0.3963 | 144.4 | 213.0 |
| 4096x4096 | nvfp4 | manual_cuda | 1.5247 | 0.6001 | 50.2 | 127.6 |
| 4096x4096 | nvfp4 | triton | 0.3891 | 0.3082 | 196.7 | 248.3 |
| 4096x4096 | nvfp4+4over6 | manual_cuda | 3.3679 | 0.6001 | 22.7 | 127.6 |
| 4096x4096 | nvfp4+4over6 | triton | 0.5509 | 0.3082 | 138.9 | 248.3 |

Full min/max/std/p20/p80: [benchmark.json](results/benchmark.json).
Display activity causes occasional outliers; median and p20/p80 are retained rather than reporting only the best sample.

## CPU reference (512 x 1024)

NumPy LUT/searchsorted includes allocation. CUDA is the preallocated quantization kernel, not Python wall time.

| Format | CPU reference (ms) | CUDA kernel (ms) |
|---|---:|---:|
| mxfp8 | 48.384 | 0.0297 |
| nvfp4 | 30.262 | 0.0553 |
