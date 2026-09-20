# 性能分析与工具检查

题目将 ncu 和/或 nsys 的使用与分析列为加分项，未把 Sanitizer 列为必需项。本页汇总实际取得的证据，完整性能与计时定义见 [总结报告](REPORT.md)。

## 结果与证据

| 平台 / 实验版本 | 性能分析 | 内存与并发检查 | 记录 |
|---|---|---|---|
| RTX 4090 D / tuning | Nsight Systems 时间线、SASS 与静态资源；NCU 因 `ERR_NVGPUCTRPERM` 无计数器结果 | 软件及原生 FP8 两路共 48 次 memcheck/racecheck/synccheck 通过 | [软件](results/4090d/tuning/software/profiling.json)、[原生](results/4090d/tuning/native/profiling.json) |
| C500 | mcTracer 时间线 | 实验镜像无可用设备端 Sanitizer | [采样汇总](results/c500/profile/summary.json) |
| MR-V100 | ixsys 时间线、ixkn 硬件计数器 | 12 次 memcheck/racecheck/initcheck 通过 | [时间线](results/iluvatar/profile/summary_trace.json)、[计数器](results/iluvatar/profile/summary_counters.json)、[检查](results/iluvatar/profile/summary_sanitizer.json) |
| S4000 | MUPTI 取得活动记录但起止时间为零，状态为 `UNAVAILABLE_TIMESTAMPS`；性能用未插桩 event | 未取得设备端 Sanitizer 结果 | [采样状态](results/musa/profile/summary.json) |
| Ascend 910B2 | 4 组 msprof 任务时间与 PipeUtilization 计数器 | 4 组基础 memcheck 通过；全量源码插桩未成功，race/init/sync 未通过验证 | [采样](results/ascend/profile/summary.json)、[检查](results/ascend/sanitize/summary.json) |

记录对应各次实验的二进制与环境，覆盖范围以日志用例为准。NVIDIA 的结果不代表国产后端已通过相同检查。

## 分析结论

4090 D 的 1024×1024 FP16 采样中，软件 MXFP8 量化平均 2.913 μs，原生 E4M3 对照 2.650 μs；NVFP4 全局 amax 为 1.911 μs，局部缩放与量化为 3.886 μs。时间线确认 MXFP8 的局部缩放已合入量化，而 NVFP4 仍有全局归约成本。两路反量化实现相同，耗时接近。具体调用次数和资源占用见 [逐 kernel 分析](#nvidia-kernel-分析)。

Ascend 的软件编码以标量操作为主，msprof 的 Scalar 管线占用约 98%，说明仅有向量 amax 仍不足以消除编码成本。显式预取反量化候选在实测中更慢，最终保留直接 GM 读取、批量写出，见 [候选比较](REPORT_ASCEND.md)。

Profiler 下的 kernel 均值用于分析组成；加速比使用未插桩、预热后的完整流程事件计时。逻辑有效带宽不能替代 DRAM 计数器，验证程序中各路径的总调用占比也不能直接当作单次业务耗时占比。

## 复现入口

先按 [README](README.md) 设置对应 SDK 环境并构建，再在本题目录执行：

| 平台 | 命令 |
|---|---|
| NVIDIA | `python3 tests/profile_native.py --output results/review-profile/software` |
| NVIDIA 原生 FP8 对照 | `python3 tests/profile_native.py --binary build/quantize_native --output results/review-profile/native` |
| C500 | `python3 tests/profile_metax.py` |
| MR-V100 | `python3 tests/profile_iluvatar.py --mode trace`，另以 `counters` / `sanitizer` 运行 |
| S4000 | `python3 tests/profile_musa.py` |
| Ascend | `python3 tests/profile_ascend.py --tool profile`，另以 `--tool sanitize` 运行 |

这些脚本默认结果目录及用例见各平台报告。复测归档版本须使用报告关联源码；当前分支的运行应作为新的实验记录。服务器镜像、驱动和权限会影响可用工具。

## NVIDIA kernel 分析

采样输入为 1024×1024 FP16；沿用配置文件，MXFP8 反量化输出 FP16，NVFP4 输出 FP32。下表为带插桩的 kernel 平均时间，单位 μs；完整性能比较使用报告中无插桩的 CUDA event 数据。

| kernel | 软件版 | 原生 FP8 对照 |
| --- | --- | --- |
| MXFP8 量化 | 2.913 | 2.650 |
| MXFP8 反量化至 FP16 | 2.194 | 2.191 |
| NVFP4 全局 amax | 1.911 | 1.903 |
| NVFP4 局部缩放与量化 | 3.886 | 3.782 |
| NVFP4 反量化至 FP32 | 3.001 | 2.996 |

MXFP8 时间线中量化只保留一次向量 kernel；NVFP4 仍需清零、向量 amax 和量化。原生 E4M3 转换主要减少 MXFP8 编码时间，NVFP4 的数据编码与全局归约没有因此消失，收益较小。反量化两路采用同一实现，时间接近。每个主要 kernel 记录 46 次调用，来自单项测试和包含传输测试的各 3 次预热 + 20 次重复；不能把整个验证程序的总占比解释为一次量化的占比，也不能把上表 kernel 时间之和当作完整流程延迟。

软件量化实例使用 24 个（MXFP8）或 32 个（NVFP4）寄存器，向量反量化使用 16–22 个；两者无共享内存、无 local-memory 分配和栈帧。向量 amax 使用每 CTA 128 字节共享内存、22–25 个寄存器。静态资源说明这次向量化没有出现寄存器溢出对应的 local-memory 分配，实际 occupancy 仍需硬件计数器测量。


原始工具命令、退出码与二进制指纹见上表记录链接。工具内时间只用于解释 kernel 组成，总优化收益采用总结报告中的完整流程事件计时。
