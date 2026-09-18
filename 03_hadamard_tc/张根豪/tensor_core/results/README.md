# 实验数据索引

仓库保留支撑报告的 CSV、JSON、非空关键日志与图表，不携带大体积 profiler 二进制、
原始失败张量或每配置的重复控制台日志。归档与恢复方法见
[清理说明](../../docs/repository_cleanup.md)。历史失败状态不删改成成功。

| 结果目录 | 用途 |
|---|---|
| [guard_quant_17294196](guard_quant_17294196/summary.md) | 最新保护式 TC 验收、压力测试、sanitizer、选择性 INT4 性能 |
| [hopper_gaps_17292654](hopper_gaps_17292654/summary.md) | H200 同卡 PTX/WMMA/warp 与三个独立量化复测进程 |
| [hopper_gaps_17292654/precision_summary.md](hopper_gaps_17292654/precision_summary.md) | 三种残差算法的通过率及成本 |
| [hopper_gaps_17291971](hopper_gaps_17291971/ncu_summary.csv) | 新 PTX 的唯一原始 30 项 NCU 数据及重采日志 |
| [gap_builds_17294647](gap_builds_17294647/status.txt) | CMake、Make、实验分支三套构建检查 |
| [verify_17251073/final_library](verify_17251073/final_library/summary.md) | 原始 TC 参数优化与实际 Dao 验收总表 |
| [verify_17251073/final](verify_17251073/final/summary.md) | 对应独立 FP32 参考验收，不与 Dao 报告混淆 |
| [verify_17251286](verify_17251286/environment.txt) | Dao 全矩阵验收源数据 |
| [ncu_17251063](ncu_17251063/environment.txt) | 旧 WMMA 参数的 H100 40 项计数器 |
| [libperf_17289704](libperf_17289704/benchmark/summary.md) | 首轮实际参考库同边界性能 |
| [warp_tune_17290172](warp_tune_17290172/benchmark/summary.md) | warp/融合 block 与宽写回参数扫描 |
| [precision_17290602](precision_17290602/diagnosis/stages.csv) | WMMA 失败用例的逐阶段数值定位 |
| [mma_17290837](mma_17290837/summary.md) | 首轮 L40S 显式 PTX 实验及 NCU 失败记录 |

其余目录保留历史基线、首次审计及参考库安装证据。`verify_17251073/progress` 与
`summary` 中间汇总已归档，最终两种参考的报告仍保留。

## 重建最新图表

在安装了 numpy/matplotlib 的 Python 环境中，从仓库根目录执行；不需要 GPU：

```bash
python tensor_core/scripts/plot_hopper_ncu.py tensor_core/results/hopper_gaps_17291971 \
  --dram-peak-gbs 4800 --fp32-peak-tflops 67
python tensor_core/scripts/summarize_hopper.py tensor_core/results/hopper_gaps_17292654 \
  --profile-dir tensor_core/results/hopper_gaps_17291971 \
  --dram-peak-gbs 4800 --fp32-peak-tflops 67
python tensor_core/scripts/summarize_precision_followup.py tensor_core/results/hopper_gaps_17292654
python tensor_core/scripts/summarize_guard.py tensor_core/results/guard_quant_17294196
```

`hopper_gaps_17292654/ncu` 原本是前一作业的逐字复制，清理后统一读取原始目录。
清理没有重新测量或改动性能数值，峰值参数仍必须匹配实际 GPU。
