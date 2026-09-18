# 摩尔线程第 1 次优化证据

量化速度表与结论见 [摩尔优化日志](../../docs/MUSA_OPTIMIZATION_LOG.md)。环境：S4000 / MUSA 5.1.0 / mp_22，日期 2026-09-15。固定 FP32 正态输入、seed=20260909、block+nearest；预热 3 次、测量 20 次，使用 MUSA event。

| 目录/文件 | 内容 |
|---|---|
| `baseline/` | 本轮 shared 适配版重新测量的优化前基线 |
| `step1/benchmark/` | shuffle 归约 + 一次 shared scale 广播，标量读写 |
| `step2/benchmark/` | float4 加载 + MXFP8 uint32/NVFP4 uint16 合并写出 |
| `final-benchmark/` | 清理未调用的步骤 1 kernel 后，最终交付版本的性能 |
| `recheck/` | 按数据规模交替执行 before/step2 的复测，含二进制哈希 |
| `before/`、`step1/`、`step2/` 中的 `Core-source.tar.gz` | 对应阶段的完整 Core 源码快照（归档不提交 Git）；可在新目录解包后用 README 的 MUSA 命令构建 |
| `step1/ctest-*-failed.log` | 两个纯 shuffle 广播候选失败证据，未纳入正式路径 |
| `final-ctest.log`、`final-ctest-details.log` | 最终 MUSA 10/10 CTest 通过，包含尾部哨兵和编码边界 |
| `cuda-ctest.log` | CUDA 9/9 回归通过 |
| `shuffle-test.log` | 4/8/16/32 宽度、52 组输入、39,936 个线程位置的独立验证 |
| `evaluation-summary.json`、`evaluation/summary.json` | 144 组 CPU 量化/反量化全部一致，无非有限输出，含逐组误差及输入哈希 |
| `SOURCE_SHA256SUMS.txt`、`SHA256SUMS.txt` | 最终 Core 源码和本目录证据的完整性清单 |

各 benchmark 目录都包含环境、测量时源码哈希、原始 JSONL、summary.json、RESULTS.md。复测用的二进制只保存在远程同名 before/step1/step2 目录，未回传本地；`recheck.py` 使用这些留存二进制。输入/输出二进制保留在远程 evaluation，未回传。

量化的 P95 使用 nearest-rank，中位数取中间两样本均值。`quant_enumeration` 只是内部编码对照；本次优化前后始终比较 `quant_optimized / resident_gpu`。NVFP4 包含全局两级归约，CPU 校验、文件和传输不在该计时内。

正式代码仅保留向量化融合 kernel，步骤 1 版本在源码快照中。CPU reference、golden、scale/舍入规则未改；FP16 输入仍先展开为 FP32，因此也会使用该路径。完整厂商内存/竞争检查尚未完成，哨兵只检查覆盖用例的越界写。
