# MXFP8 第一层性能分析

条件：RTX 3060 Laptop，4M FP32 输入，`--benchmark 4194304 20`。Nsight Systems 采集包含 benchmark 内部对照、默认路径和反量化调用，因此 kernel summary 不能直接当作单次默认量化耗时。

当前向量化 kernel：

| kernel | 实例数 | 中位耗时 |
|---|---:|---:|
| `mxfp8_quantize_vectorized_kernel` | 47 | 0.069114 ms |
| `dequantize_kernel<float>` | 23 | 0.118025 ms |
| `dequantize_kernel<__half>` | 23 | 0.111729 ms |
| `dequantize_kernel<__nv_bfloat16>` | 23 | 0.117398 ms |

说明：占比最高的 `quantize_kernel<(bool)1>` 是 benchmark 的枚举 baseline，不是当前默认向量化路径。当前默认量化 kernel 的主要 GPU 时间约为 0.069 ms；Host API 中 `cudaMalloc`、`cudaMemcpy` 和同步合计占绝大多数端到端时间，后续端到端优化应优先考虑 workspace 复用和减少拷贝。

完整原始统计：[STATS.txt](STATS.txt)。
