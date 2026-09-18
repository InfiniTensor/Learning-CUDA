# NVFP4 float2 输入加载：负结果（不纳入正式轮次）

本目录是一次**已放弃的试验**记录，不是第 7 次优化。保留它是因为项目要求保留负结果，
不能把试过的失败方案当作成功优化。

## 试了什么

`kernels/nvfp4_quantize.cuh` 的 NVFP4 融合量化改为按 `float2` 加载输入。
`environment.json` 的 `source_sha256` 显示**只有这一个文件**与第 4 次
（`records/07-mxfp8-vectorized/`）不同，其余源码逐字节相同。

## 结果

| 元素数 | 第 4 次 median | 本次 float2 median | 变化 |
|---:|---:|---:|---:|
| 1M | 0.048928 | 0.050176 | 慢约 2.6% |
| 4M | 0.143360 | 0.141824 | 快约 1.1% |
| 16M | 0.522144 | 0.519088 | 快约 0.6% |

收益不稳定、且在 1M 上反而变慢，因此没有保留。当时的结论是：不建议继续投入
`float2` 输入加载，实测收益约为 ±1% 且不稳定。

## 重要：这份源码已丢失

`environment.json` 记录该版本的 `kernels/nvfp4_quantize.cuh` SHA256 为
`9d768efabe262dabb11c6a500786f5d553efbdbebb0a60e6be405ea5403bb228`。
该版本**从未提交**，也不在任何 git 对象中，无法恢复。当前代码是第 4 次的实现
（`b3b5397c…`），即本试验已完全回退。

因此本目录只能用于说明“测过且放弃”，**不能**用它重建那次实验。

## 数据

`RESULTS.md`、`summary.json`、`*.jsonl` 均由 `Core/tools/benchmark.py` 生成，
内容是该次 `--backend cuda` 的完整输出。本 README 不参与自动生成，可独立编辑。
