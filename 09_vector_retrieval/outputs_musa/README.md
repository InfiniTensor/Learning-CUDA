# 实验产物（摩尔线程 MUSA 平台）

本目录保存摩尔线程 MUSA 平台（MUSA 5.1.0，`mcc -x musa` + `mublas`）的实测日志与
扫描汇总，对应报告 §7.6 的表 9-10。四个平台产物的对应关系：

| 目录 | 平台 | 对应报告章节 |
| --- | --- | --- |
| `outputs/` | NVIDIA RTX 4090 | §7.3 表 1-4 |
| `outputs_corex/` | 天数智芯 MR-V100 | §7.4 表 5-6 |
| `outputs_maca/` | 沐曦 MetaX MXC500 | §7.5 表 7-8 |
| `outputs_musa/`（本目录） | 摩尔线程 MUSA | §7.6 表 9-10 |

四平台横向对比见报告 §7.7 表 11。

## 目录结构

```text
outputs_musa/
├── README.md                  本说明
├── experiment_summary.csv     nprobe(1,2,4,8,16,32,64) × batch(32,128,512) 汇总（表 10）
├── exact_perf.log             GPU exact（200 query 子集对比 CPU + 1000 query 性能）
├── exact_quality.log          exact 模式质量
├── ivf_flat_perf.log          IVF-Flat 性能（表 9）
├── ivf_flat_quality.log       IVF-Flat recall@100 = 0.991390
├── ivf_pq16_perf.log          IVF-PQ16 + rerank 性能（表 9）
├── ivf_pq16_quality.log       IVF-PQ recall@100 ≈ 0.012（量化限制，分析见 §7.8）
└── sweep/                     7 组 nprobe（batch=128）的 perf/quality 日志
```

数据规模：`N=300000, D=128, nq=1000, topK=100, nlist=1024, nprobe=16, pq_m=16`，
与其他三个平台使用相同生成参数与随机种子，因此 IVF-Flat 的 recall 数值与
`outputs_corex/`、`outputs_maca/` 一致。

## 构建与复现

```bash
make -f Makefile.musa -j
export LD_LIBRARY_PATH=/usr/local/musa/lib:/usr/local/musa/lib64
./build_musa/test_host && ./build_musa/test_gpu

python3 python/gen_dataset.py --out data_musa --n 300000 --dim 128 --nq 1000 \
    --top-k 100 --metric l2 --clusters 100 --vector-scale 0.10 \
    --query-scale 0.02 --nlist 1024 --nprobe 16
python3 python/run_experiments.py --vsearch ./build_musa/vsearch \
    --data data_musa --mode ivf_flat --out-dir outputs_musa
```

MUSA 的 API 名称映射（`cuda*`/`cublas*` → `musa*`/`mublas*`）与设备能力探测结果见
报告 §3.3、§7.2、§7.6；探测程序见 [tools/probe/](../tools/probe/README.md)。
