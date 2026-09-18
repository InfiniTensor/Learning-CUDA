# MUSA 首轮适配证据（2026-09-15）

平台：摩尔线程 MTT S4000，MUSA SDK/驱动 5.1.0。本地源码与远程 `/data/Learning-CUDA/02_quant_dequant/YeKcaJ` 同步；来源分支 `mxfp8-YeKcaJ`，基础提交 `58099b85eb6ecf8012bbe4906216cdf9ba5500d1`，本轮适配尚未提交 Git。

| 文件 | 内容 |
|---|---|
| `ctest-final.log`、`ctest-details.log` | 最终 MUSA 构建的 9 项测试及各项输出，包含写入尾部哨兵检查 |
| `cuda-ctest.log` | 本地 CUDA 9/9 回归通过 |
| `build-final.log` | MUSA 增量编译日志 |
| `evaluation-summary.json` | 144 组全部通过，无非有限输出 |
| `evaluation/summary.json` | 各组误差、压缩率、输出类型、CPU 对照、输入哈希；绝对路径指向远程原始产物 |
| `benchmark/RESULTS.md`、`summary.json`、`*.jsonl` | 1M/4M/16M，预热 3 次、测量 20 次的性能与原始记录 |
| `benchmark/environment.json` | 测量时设备、编译器、CPU 和源码 SHA256 |
| `numeric_probe.cu` | 次正规数问题的诊断探针，仅作证据，不参与正式编译 |
| `SOURCE_SHA256SUMS.txt` | 最终同步的 Core 源码、配置、构建定义哈希 |
| `SHA256SUMS.txt` | 本目录证据文件完整性清单（不含清单本身） |

基准测量后只追加边界测试、修复 CUDA 专属尾部写入并整理说明，MUSA 计算路径未改；测量时源码哈希与最终源码哈希分别保留。原报告将 event 写成 CUDA 的文字已更正为 MUSA，原始 JSONL 数值未改。

本轮仅回传日志、汇总与环境信息。评估产生的输入、权重和张量保留在远程 `evaluation/`；避免把可重建二进制堆进仓库。

边界验证限制：MUSA 已做输出哨兵检查，尚未运行厂商内存/竞争检查工具。本地 CUDA compute-sanitizer 启动失败（`cuda-memcheck-unavailable.log`），不能将本轮测试写成 sanitizer 通过；CUDA 的 CTest 和哨兵检查实际通过。

复现命令见 [MUSA 使用说明](../../Core/backends/musa/README.md)，结论见 [平台适配记录](../../docs/PLATFORM_ADAPTATION.md)。
