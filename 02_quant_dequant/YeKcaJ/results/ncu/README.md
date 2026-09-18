# Nsight Compute 报告

本目录保存 GPU 硬件计数器的原始报告和分析结论，供写报告时引用。

| 文件 | 实测采集的 kernel | Duration |
|---|---|---:|
| `mxfp8-test.ncu-rep` | `mxfp8_quantize_vectorized_kernel` | 71.14 us |
| `nvfp4-test.ncu-rep` | `maximum` | 61.31 us |
| `nvfp4-maximum.ncu-rep` | `maximum` | 60.83 us |
| `nvfp4-maximum-1024.ncu-rep` | `nvfp4_quantize_fused_kernel` | 106.59 us |

分析结论：

- [mxfp8-test-analysis.md](mxfp8-test-analysis.md)：MXFP8 向量化 kernel，
  DRAM 90.28%、SM 81.95%、占用率 88.92%、寄存器 25、无溢出。
- [nvfp4-test-analysis.md](nvfp4-test-analysis.md)：NVFP4 两个 kernel 的解读，
  融合量化计算受限（SM 84.46%、DRAM 56.08%），`maximum` 访存受限（DRAM 88.01%、SM 32.50%）。

文件名沿用当时的命名（`nvfp4-maximum*`），实际采集的 kernel 以上表为准。
查看方式：`ncu --import <文件>`，或用 `ncu-ui` 图形界面打开。

`.ncu-rep` 是二进制报告，体积大（合计约 154 MB）且可用同样命令重现，因此被
`.gitignore` 排除、不入版本库；分析文档入库。这些报告不作为正式性能结论：
`ncu` 会为重放 kernel 改变时钟和缓存状态，正式耗时以 CUDA event benchmark 为准。
采集条件见 [性能测试固定协议](../../docs/BENCHMARK_PROTOCOL.md)，
nsys 时间线见 [records/09-nsys-mxfp8](../../records/09-nsys-mxfp8/SUMMARY.md)。
