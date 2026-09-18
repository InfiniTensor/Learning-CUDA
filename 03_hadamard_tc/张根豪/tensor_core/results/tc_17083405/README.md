# Tensor Core 对照实验产物（Slurm job 17083405）

由 `sbatch tensor_core/slurm/tc_bench.slurm` 一次生成。

| 文件 | 内容 |
|---|---|
| `environment.txt` | GPU、driver、nvcc、检测到的 `CUDA_ARCH` |
| `status.txt` | 扫描退出码（0 = 全部配置通过） |
| `sweep.log` | 全矩阵逐配置的正确性与计时原始输出 |
| `tc_results.csv` | 机器可读结果：五条变换路径 + 五条融合路径 |
| `tc_summary.md` | 由 `scripts/summarize_tc.py` 生成的摘要表 |
| `figures/tc_performance.png` | 时间与误差四联图 |
| `nsys_tc_stats_*.csv` | Nsight Systems kernel/API 汇总 |
| `nsys_summary/` | launch 形状、寄存器、shared memory、resource-limit occupancy |
| `ncu/` | Nsight Compute 尝试的原始日志（本节点 counter 被 DCGM 占用） |

规模：131072 tokens、`normalize=true`、20 warmup + 100 CUDA Event 迭代；
另含 6 个 `normalize=false` 配置和 1 个 token 数不整除每 tile token 数的
尾部回落配置。`sweep_exit_code=0` 表示全部配置的精度判据与融合逐位一致性都通过。
