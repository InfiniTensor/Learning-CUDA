# 实验产物（天数智芯 CoreX 平台）

本目录保存天数智芯 Iluvatar CoreX（MR-V100，IX-ML 4.4.0）平台的实测日志与扫描
汇总，对应报告 §7.4 的表 5-6。四个平台产物的对应关系：

| 目录 | 平台 | 对应报告章节 |
| --- | --- | --- |
| `outputs/` | NVIDIA RTX 4090 | §7.3 表 1-4 |
| `outputs_corex/`（本目录） | 天数智芯 MR-V100 | §7.4 表 5-6 |
| `outputs_maca/` | 沐曦 MetaX MXC500 | §7.5 表 7-8 |
| `outputs_musa/` | 摩尔线程 MUSA | §7.6 表 9-10 |

## 目录结构

```text
outputs_corex/
├── README.md                  本说明
├── experiment_summary.csv     nprobe(1,2,4,8,16,32,64) × batch(32,128,512) 汇总（表 6）
├── exact_perf.log             GPU exact（200 query 子集对比 CPU + 1000 query 性能）
├── exact_quality.log          exact 模式质量
├── ivf_flat_perf.log          IVF-Flat 性能（表 5）
├── ivf_flat_quality.log       IVF-Flat recall@100 = 0.991390
├── ivf_pq16_perf.log          IVF-PQ16 + rerank 性能（表 5）
├── ivf_pq16_quality.log       IVF-PQ recall@100 ≈ 0.012（量化限制，分析见 §7.8）
└── sweep/                     21 组 nprobe × batch 的 perf/quality 日志
```

数据规模：`N=300000, D=128, nq=1000, topK=100, nlist=1024, nprobe=16, pq_m=16`。
本目录只收录汇总与日志；`result.txt` 与 `.idx` 体积较大，按下方命令重新生成。

## 构建与复现

```bash
make -f Makefile.corex -j
export LD_LIBRARY_PATH=/usr/local/corex/lib64:/usr/local/corex/lib
./build_corex/test_host && ./build_corex/test_gpu

python python/gen_dataset.py --out data_corex --n 300000 --dim 128 --nq 1000 \
    --top-k 100 --metric l2 --clusters 100 --vector-scale 0.10 \
    --query-scale 0.02 --nlist 1024 --nprobe 16
python python/run_experiments.py --vsearch ./build_corex/vsearch \
    --data data_corex --mode ivf_flat --out-dir outputs_corex
```

该平台的设备特性处理（32 位原子计数、float 距离累计）见报告 §3.3 与 §7.2；
探测程序见 [tools/probe/](../tools/probe/README.md)。
