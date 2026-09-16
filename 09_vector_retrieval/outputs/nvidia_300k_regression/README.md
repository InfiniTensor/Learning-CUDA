# NVIDIA 300k 回归数据（2026-09-07）

对应报告 §8.3 表 3b：用包含双平台适配的最终源码在 NVIDIA RTX 4090 上重新生成
数据并完成 exact / IVF-Flat / IVF-PQ 三次完整 bench。

数据规模：`N=300000, D=128, nq=1000, topK=100, nlist=1024, nprobe=16, pq_m=16`
（`python/gen_dataset.py --clusters 100 --vector-scale 0.10
--query-scale 0.02`，修复后文件头格式）。

```text
exact_perf.log          独立 exact bench 的 perf（nq=200 CPU 参考行 + 全程行）
exact_quality.log       exact 模式质量
ivf_flat_perf.log       IVF-Flat bench（内含 exact 子集对比行）
ivf_flat_quality.log    IVF-Flat recall@100=0.985830
ivf_pq_perf.log         IVF-PQ16 + rerank bench
ivf_pq_quality.log      IVF-PQ recall@100=0.012280
ivf_flat_result.txt / ivf_pq_result.txt  Top-100 检索结果（1000 query）
params.txt              实际运行参数
indexes/                IVF-Flat 与 IVF-PQ 构建索引
```

说明：

- 表 3b 的 build_ms（ivf_flat=97、ivf_pq=213）来自 bench 运行日志
  `index built: build_ms=...`，perf.log 的 build_ms 字段在 bench 模式下为 0
  （检索阶段单独计时，索引构建在 perf 行采样前完成）。
- exact 表行以独立 `exact_perf.log` 为准（98.6 ms / 2029 QPS / 87.1×）；
  IVF bench 内嵌的 exact 子集对比行（~95.5 ms / 90.8×）用于 recall 与 speedup
  对照，两者存在正常的运行间波动。
- 原始 `vectors.bin` / `queries.bin` 数据文件不随报告分发；如需复现，用
  `python/gen_dataset.py`（参数见上文）重新生成即可。
