# Tensor Core kernel 的 NCU 硬件计数器（Slurm job 17083350，H200 `gh115`）

主 sweep 所在的 L40S 分区上 performance counter 被 DCGM 占用，因此硬件计数器在
Hopper 节点单独采集。由 `sbatch tensor_core/slurm/tc_ncu_h200.slurm` 生成。

| 文件 | 内容 |
|---|---|
| `environment.txt` | H200、driver 580.82.07、CUDA 13.0.88、Nsight Compute 2025.3.1 |
| `status.txt` | `ncu_exit_status=0`，三个 kernel 全部采集成功 |
| `ncu_summary.md` / `ncu_summary.csv` | 用于结构诊断的关键指标（由 `scripts/summarize_tc_ncu.py` 提取） |
| `ncu/<target>.ncu-rep` | 原始报告，可用 `ncu-ui` 打开 |
| `ncu/<target>_raw.csv` | `--page raw` 全量导出（610 列） |
| `ncu/<target>.log` | 采集日志 |

配置：FP16、`head_dim=128`、16384 tokens，section 为 SpeedOfLight、
MemoryWorkloadAnalysis、Occupancy、WarpStateStats、SchedulerStats。

口径说明：这些计数器采自 H200，而端到端性能表采自 L40S。两者实验目的不同 ——
计数器用于解释 kernel 的结构差异（stall 构成、shared memory 压力、occupancy），
端到端时间用于性能结论。两组数字不混算，也不用 NCU 的 duration 替代无 profiler
的 CUDA Event 计时（NCU 会串行化并引入采集开销）。
