# 报告图表

图表来自本项目历史结果。`results/` 保存相应 CSV、环境、验证日志与 MetaX 原生 trace。性能图中的 kernel min / mean 以图下注释和原始 CSV 为准。

| 图表 | 数据 |
|---|---|
| [RTX 4090 D 性能](performance_4090.png) | rtx4090d/benchmark_legacy.csv；kernel min，e2e mean |
| [MTT S4000 性能](performance_moore_s4000.png) | moore_s4000_musa5.1/benchmark_1080p_rgb_base.csv、benchmark_4k_rgb_base.csv |
| [MetaX C500 性能](performance_metax_c500.png) | metax_c500_maca3.0/benchmark_base.csv |
| [Iluvatar MR-V100 性能](performance_iluvatar_mrv100.png) | iluvatar_mrv100_corex4.4.0/benchmark_base.csv |
| [RTX 3060 质量与延迟](quality_latency_tradeoff.png) | rtx3060_laptop/quality_tradeoff.csv、benchmark_quality.csv |
| [MetaX 质量与延迟](quality_latency_metax_c500.png) | metax_c500_maca3.0/quality_tradeoff.csv、benchmark_base.csv、benchmark_matrix.csv；V2 |
| [Iluvatar 质量与延迟](quality_latency_iluvatar_mrv100.png) | iluvatar_mrv100_corex4.4.0/quality_tradeoff.csv、benchmark_base.csv、benchmark_matrix.csv；V2 |
| [合成配对图](quality_triptych.png) | test_images/ 中的 1080p clean/noisy/V2 输出 |
| [nsys 摘要](nsys_kernel_summary.png) | rtx4090d/nsys_summary_legacy.csv |
| [MetaX 时间线](mctracer_timeline_metax_c500.png) | metax_c500_maca3.0/mctracer/nlm_1080_v1-3213.json |

重复的跨平台三联图及旧验收卡已从提交中移除；对应原始版本保留在仓库外备份。图表不用于证明所有平台在当前重构后均重新运行成功。
