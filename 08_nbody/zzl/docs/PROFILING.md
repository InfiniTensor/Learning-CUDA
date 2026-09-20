# CUDA 性能分析与复现

## 1. 实验条件与结果

性能分析使用 WSL Ubuntu 24.04、NVIDIA GeForce RTX 5060 Laptop GPU、驱动 582.05
和 Nsight Systems 2026.5.1。输入包含 4096 个粒子，积分运行 5 步，使用 direct
提交模式并关闭轨迹输出。tiled 与 naive 使用相同输入和配置。
上述驱动与工具版本属于本次 WSL 采集。开发工作区另存有早期 Windows 构建的平台记录，
其驱动和 CUDA 运行时版本与本次不同，两者平台不同，不用于同一次采集的对照。

两份 SQLite 导出文件均包含 16 条 CUDA 内核记录和 57 条 CUDA Runtime 记录。
其中求力内核的统计如下。

| 求力内核 | 调用次数 | 平均耗时 / μs |
|---|---:|---:|
| tiled，线程块大小 256 | 9 | 196.991 |
| naive | 9 | 533.190 |

naive 与 tiled 的平均耗时之比为 2.707。该结果表明，在本次短任务中，共享内存
分块实现的求力耗时较低。统计包含预热和正式计算调用，且分析工具会影响程序执行，
因此该比值用于描述本次求力内核的相对耗时。完整进程性能见 [实验报告](../REPORT.md)。
CUDA API 汇总包含初始化及分析工具开销，求力内核比较采用独立的内核时间统计。

归档数据为 `evidence/profiling-nsys.json`，原始统计为
`evidence/nsys-2026-stats.txt` 和 `evidence/nsys-2026-naive-stats.txt`。
其中 profiling-nsys.json 由该次采集的 `summary.json` 与两份统计报告汇总而成，
字段命名与脚本直接输出的汇总不同；汇总时逐项核对两份报告的调用次数与平均耗时，
不一致则拒绝写出。这些文件对应上述实验批次，并记录被测程序与输入文件的 SHA-256；
后续复测结果应单独保存。

## 2. 复现步骤

在 Linux/WSL 中准备可用的 Nsight Systems，并在项目根目录执行以下命令。
将 `/path/to/nsys` 替换为实际程序路径。Nsight Systems 作为外部分析工具单独准备。

```bash
chmod +x bin/nbody
python scripts/profile_nsys.py \
  --nsys /path/to/nsys \
  --executable bin/nbody \
  --input outputs/submission-benchmark/cluster_4096.txt \
  --config evidence/profile_inputs/five_steps.cfg \
  --output-dir outputs/nsys-run-01
```

程序路径按来源选择：交付包解压后为 `bin/nbody`，需先赋予执行权限；在源码树中构建时改为 `build/nbody`。
`--input` 指向的 4096 粒子初态由 README 中的 benchmark 命令生成，或直接取自交付包；
每次采集使用新的输出目录。
脚本分别运行 tiled 和 naive，导出 SQLite，检查 CUDA 记录后生成 `summary.json`。
原始采集文件、SQLite 和日志保存在同一输出目录内。

汇总文件记录平均值、中位数、样本数和耗时比，并保存程序、工具、输入、配置及两份
分析脚本的 SHA-256。每份 SQLite 也单独记录哈希。采集命令及工作目录用于复查运行条件。

以下情况会拒绝生成对比报告：CUDA 数据缺失、求力内核与指定类型不符、混入另一种
求力内核、时间戳无效、两种内核样本数不同，或采集前后输入文件哈希不同。
这些检查核对采集数据的一致性；时间线完整性还需结合工具日志和运行条件判断，
文件哈希校验的覆盖范围为采集开始与结束两个时点。

工具输出保存在 `.profile.log` 和 `.export.log` 中。命令失败时，错误信息会指出
日志位置。独立检查已有 SQLite 可运行：

```bash
python scripts/verify_nsys.py outputs/nsys-run-01/tiled.sqlite
python tests/test_verify_nsys.py
```

## 3. 工具适用情况

本报告采用 Nsight Systems 2026.5.1 的有效 CUDA 时间线作为内核分析依据。
2024.6.2 的早期采集仅作为兼容性排查记录保存。

Nsight Compute 2025.1.1 在本机初始化阶段记录了 `LibraryNotLoaded` 错误。
因此，本报告讨论运行时间与调用分布；占用率、内存吞吐率及 warp stall 指标
属于后续分析的内容。Windows Compute Sanitizer 的历史日志记录了检查过程中的
平台错误，相关记录见 `evidence/memcheck.txt`、`evidence/synccheck.txt` 和
`evidence/ncu.txt`。数值验收与工具检查分别按各自证据评价。
