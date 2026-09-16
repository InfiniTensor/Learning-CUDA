# 历史实验结果索引

这些文件来自结构重构前的已有实验。整理时只改变文件位置，原始 CSV、环境、日志和 JSON 内容保持原样；其中旧路径与旧命令是历史记录。删除的中间构建日志、重复图像和旧校验清单保存在仓库外备份。

| 目录 | 数据与说明 |
|---|---|
| [rtx4090d](rtx4090d/) | 2026-09-07，CUDA 12.8；`benchmark_legacy.csv` 为 kernel min / e2e mean，缺 stddev；另含 CPU 对照、质量、nsys 摘要及原始综合日志 |
| [rtx3060_laptop](rtx3060_laptop/) | legacy 基准，以及 2026-09-15 的配对质量补充实验；`benchmark_quality.csv` 为 warmup=3/repeat=5；环境为当时未提交工作树 |
| [moore_s4000_musa5.1](moore_s4000_musa5.1/) | 2026-09-15，MUSA 5.1；base 1080p/4K 与 RGB all 为 repeat=10；完整矩阵 repeat=3；CPU base 仅一次，不能写成三次 |
| [metax_c500_maca3.0](metax_c500_maca3.0/) | 2026-09-15，MACA 3.0，C500 25% sGPU；base repeat=10、矩阵 repeat=3、CPU repeat=3；含原生 mcTracer JSON |
| [iluvatar_mrv100_corex4.4.0](iluvatar_mrv100_corex4.4.0/) | 2026-09-15/16，CoreX 4.4.0；base repeat=10、矩阵 repeat=3、CPU repeat=3；上传工作树快照的来源记录仍保留 |

`SHA256SUMS.txt` 以本目录为起点覆盖当前保留的原始证据文件。这个清单只验证整理后的文件完整性，不代表重新运行了实验；旧归档哈希仍属于当时的原始归档。

Linux 下复核：

```bash
cd assets/results
sha256sum -c SHA256SUMS.txt
```

新实验默认写入 `output/results/<platform>/`，待检查环境、参数、结果与最终 commit 一致后，再选择需要的证据纳入本目录。完整复现命令见 [README](../../README.md)。
