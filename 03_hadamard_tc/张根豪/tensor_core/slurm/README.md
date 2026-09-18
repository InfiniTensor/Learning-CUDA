# Slurm 入口

这些是该集群的可复现模板；换环境时需要调整 account、partition、ROOT、容器和
Python/Dao 路径。新 NCU 工作请用 Hopper 专用入口，不复制历史恢复脚本。

| 任务 | 入口 |
|---|---|
| H200/H100 PTX 计数器、同卡计时与宽写回复测 | `close_hopper_gaps.slurm` |
| 保护式 TC 与选择性 INT4 的输出、压力、sanitizer、性能验证 | `guard_and_selective_quant.slurm` |
| 残差表示/累加精度对照 | `mma_precision_followup.slurm` |
| 主项目和实验分支构建验证，不占 GPU | `check_gap_builds.slurm` |
| 参考库性能 / warp 参数 / PTX 首轮实验 | `library_performance.slurm` / `warp_tuning.slurm` / `mma_experiment.slurm` |
| 原始 WMMA 审计、验收和 profile | `tc_*.slurm`，参数说明见各文件 |

`archive/finish_hopper_gaps.slurm` 只保留作业 17291971 的历史恢复流程，依赖特定
节点和 scratch 产物，不应作为新的任务入口。运行中的 shell 文件不要原地编辑。
