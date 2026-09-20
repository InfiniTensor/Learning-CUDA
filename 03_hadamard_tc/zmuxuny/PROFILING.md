# 性能分析与工具检查

题目将 ncu 和/或 nsys 的使用与分析列为加分项，未把 Sanitizer 列为必需项。本页汇总实际取得的证据，完整性能与计时定义见 [总结报告](REPORT.md)。

## 结果与证据

| 平台 / 实验版本 | 性能分析 | 内存与并发检查 | 记录 |
|---|---|---|---|
| RTX 4090 D / tuning | Nsight Systems 时间线、SASS 与静态资源；NCU 因 `ERR_NVGPUCTRPERM` 无计数器结果 | 软件及原生 FP8 两路共 36 次 memcheck/racecheck/synccheck 通过 | [软件](results/4090d/tuning/software/profiling.json)、[原生](results/4090d/tuning/native/profiling.json) |
| C500 | mcTracer 时间线 | 实验镜像无可用设备端 Sanitizer | [采样汇总](results/c500/profile/summary.json) |
| MR-V100 | ixsys 时间线、ixkn 硬件计数器 | 12 次 memcheck/racecheck/initcheck 通过 | [时间线](results/iluvatar/profile/summary_trace.json)、[计数器](results/iluvatar/profile/summary_counters.json)、[检查](results/iluvatar/profile/summary_sanitizer.json) |
| S4000 | MUPTI 取得活动记录但起止时间为零，状态为 `UNAVAILABLE_TIMESTAMPS`；性能用未插桩 event | 未取得设备端 Sanitizer 结果 | [采样状态](results/musa/profile/summary.json) |
| Ascend 910B2 | 4 组 msprof 任务时间与 PipeUtilization 计数器 | 4 组基础 memcheck 通过；全量源码插桩未成功，race/init/sync 未通过验证 | [采样](results/ascend/profile/summary.json)、[检查](results/ascend/sanitize/summary.json) |

记录对应各次实验的二进制与环境，覆盖范围以日志用例为准。NVIDIA 的结果不代表国产后端已通过相同检查。

## 分析结论

MR-V100 在 8192×1024 的独立/融合变换中均记录到 32,768 条矩阵指令，与 H16 tile 数相符；这支持原生矩阵路径确实执行，而是否更快仍由完整流程事件计时判断。计数器与解读见 [MR-V100 报告](REPORT_ILUVATAR.md)。

Ascend 的小输入采样中 Cube MAC 利用率约 0.4%，只能证明该采样场景利用率低，不能据此推断所有尺寸。矩阵路径还要经过 AIV 与 GM 交接；同机性能表包含这些开销。S4000 原生矩阵路径在大输入上慢于蝶形，故不能以使用矩阵指令替代性能对照。

Profiler 下的 kernel 均值用于分析组成；加速比使用未插桩、预热后的完整流程事件计时。逻辑有效带宽不能替代 DRAM 计数器，验证程序中各路径的总调用占比也不能直接当作单次业务耗时占比。

## 复现入口

先按 [README](README.md) 设置对应 SDK 环境并构建，再在本题目录执行：

| 平台 | 命令 |
|---|---|
| NVIDIA | `python3 tests/profile_native.py --output results/review-profile/software` |
| NVIDIA 原生 FP8 对照 | `python3 tests/profile_native.py --binary build/hadamard_native --output results/review-profile/native` |
| C500 | `python3 tests/profile_metax.py` |
| MR-V100 | `python3 tests/profile_iluvatar.py --mode trace`，另以 `counters` / `sanitizer` 运行 |
| S4000 | `python3 tests/profile_musa.py` |
| Ascend | `python3 tests/profile_ascend.py --tool profile`，另以 `--tool sanitize` 运行 |

这些脚本默认结果目录及用例见各平台报告。复测归档版本须使用报告关联源码；当前分支的运行应作为新的实验记录。服务器镜像、驱动和权限会影响可用工具。

## NVIDIA kernel 分析

采样输入为 8192×128 FP16，开启随机符号。以下为分解 Tensor Core kernel 的带插桩平均时间，单位 μs；性能结论使用报告中的无插桩 CUDA event 测量。

| 阶段 | 软件 FP8 编码 | 原生 FP8 对照 |
| --- | --- | --- |
| 单独 Hadamard，MXFP8 试验 | 2.985 | 2.998 |
| Hadamard + MXFP8 融合输出 | 4.024 | 3.467 |
| NVFP4 变换后 amax 预遍历 | 2.638 | 2.648 |
| Hadamard + NVFP4 融合输出 | 5.023 | 4.964 |

原生 FP8 转换主要减少 MXFP8 融合输出中的编码计算，单独变换与 NVFP4 的 amax 预遍历基本不变。NVFP4 仍需两次变换和清零，E2M1 数据编码也是软件实现，因此 E4M3 scale 的硬件转换只改善其中一小部分。单独变换记录 46 次调用，来自单独测试和非融合测试；融合输出及 amax 各 23 次，包括 3 次预热和 20 次重复。表中 kernel 平均时间之和不含全部启动间隔与清零，不能代替完整融合延迟。

软件融合在 D=128 使用 32–37 个寄存器、D=1024 使用 56–64 个寄存器，无 local-memory 分配和栈帧；软件编码优化收益主要来自指令与访问方式变化，并没有把静态寄存器数量下降当作前提。NVFP4 的中间结果保留对照在无插桩基准与 Sanitizer 中单独运行，默认时间线保持重算式融合。


原始工具命令、退出码与二进制指纹见上表记录链接。工具内时间只用于解释 kernel 组成，总优化收益采用总结报告中的完整流程事件计时。
