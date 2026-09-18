# docs：题目与规范文档

放置**题目原文**和**跨目录的规范性文档**。这里不放源码，也不放实验数据。

---

## 目录内容

| 文件 | 作用 |
|---|---|
| `2026夏季训练营 CUDA 方向项目.pdf` | 题目原文，选题二「MXFP8 / NVFP4 低精度模拟与反量化」 |
| `BENCHMARK_PROTOCOL.md` | 性能测试的固定协议：冻结输入、统计口径、加速比规则 |
| `MUSA_OPTIMIZATION_LOG.md` | 摩尔线程平台的优化记录 |
| `PLATFORM_ADAPTATION.md` | 国产平台适配进度、验收标准与远程操作流程 |

---

## 各文档用途

### BENCHMARK_PROTOCOL.md

正式性能结论的唯一口径来源。规定了：

- 只用 `input/benchmark-v1/` 的冻结 FP32 输入，并校验 SHA256
- 固定 `block + nearest`、1M/4M/16M、预热 3 次、测量 20 次
- 只比较 `quant_optimized / resident_gpu` 的中位数与 P95
- **加速比只有两列**：相对上一轮、相对本平台本格式的优化前基线

还说明了冻结输入不在版本库里时如何用 `--benchmark-export` 重建。

### PLATFORM_ADAPTATION.md

记录国产平台适配的当前状态与验收标准：

- 已完成：摩尔线程 MUSA（MTT S4000）
- 未适配：沐曦 MXMACA、华为昇腾

每个平台的验收标准包括 packed data 逐字节一致、scale 逐字节一致、三种输出一致、1M/4M/16M 性能数据与误差指标。

### MUSA_OPTIMIZATION_LOG.md

摩尔线程平台的两轮优化数据，与 CUDA 的 `OPTIMIZATION_LOG.md` 分开记录，两者的性能结论不能互推。

---

## 相关文档

- 仓库总览与操作步骤见 [根 README](../README.md)
- 题目第 4 页要求的三类输出、三类矩阵误差统计及生成命令见 [output/README.md](../output/README.md)；固定评估输入见 [input/README.md](../input/README.md)
- CUDA 优化轮次见 [OPTIMIZATION_LOG.md](../OPTIMIZATION_LOG.md)
- 实验过程证据见 [records/README.md](../records/README.md)
