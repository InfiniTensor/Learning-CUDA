# Outputs

本项目源码已在两套平台完成构建与运行（实际数字见 REPORT.md）：

- **NVIDIA**：远程 RTX 4090，CMake + nvcc（CUDA 12.0/12.8，驱动 570）。本目录
  保存 NVIDIA 平台报告、实验汇总与代表性性能/质量日志；
- **天数智芯 CoreX**：MR-V100（IX-ML 4.4.0），`Makefile.corex` 构建。对应日志与
  21 组 nprobe×batch 扫描汇总见仓库根目录 `outputs_corex/`（本目录未重复存放）。

```text
REPORT.md              详细总结报告（含实测 QPS/P50/P99/recall）
PROFILING.md           Nsight Systems 实测 kernel 占比与分析
experiment_summary.csv nprobe × batch_size 扫描汇总
exact_perf.log         exact 1000 query 性能（1e6×128）
ivf_flat_perf.log      IVF-Flat 性能（1e6×128）
ivf_flat_quality.log   IVF-Flat recall
ivf_pq16_perf.log      IVF-PQ16 + rerank 性能
ivf_pq16_quality.log   IVF-PQ16 + rerank recall
retrieval_results/     exact / IVF-Flat / IVF-PQ16 的 Top-K 结果文件
indexes/               IVF-Flat 索引、IVF-PQ16 索引
sweep/                 21 组 nprobe × batch_size 的 result/perf/quality
```

NVIDIA 平台本次最终源码回归（300k×128）的 perf/quality/result/index 产物见
[nvidia_300k_regression/](nvidia_300k_regression/)，对应 REPORT.md §8.3 表 3b。
如需重新实验：

```bash
python python/gen_dataset.py --out data --n 1000000 --dim 128 --nq 1000 --top-k 100
python python/run_experiments.py --vsearch ./build/vsearch
```

NVIDIA 使用 `build/vsearch`，CoreX 使用 `build_corex/vsearch`；脚本会重新生成
数据并覆盖对应日志。
