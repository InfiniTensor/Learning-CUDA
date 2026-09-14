# 实验产物（沐曦 MetaX 平台）

本目录保存沐曦 MetaX MXC500（MACA 3.5.3，cu-bridge）平台的实测日志与扫描汇总，
对应报告 §7.5 的表 7-8。三个平台产物的对应关系：

| 目录 | 平台 | 对应报告章节 |
| --- | --- | --- |
| `outputs/` | NVIDIA RTX 4090 | §7.3 表 1-4 |
| `outputs_corex/` | 天数智芯 MR-V100 | §7.4 表 5-6 |
| `outputs_maca/`（本目录） | 沐曦 MetaX MXC500 | §7.5 表 7-8 |

## 目录结构

```text
outputs_maca/
├── README.md                  本说明
├── experiment_summary.csv     nprobe(1,2,4,8,16,32,64) × batch(32,128,512) 汇总（表 8）
├── exact_perf.log             GPU exact（200 query 子集对比 CPU + 1000 query 性能）
├── exact_quality.log          exact 模式质量
├── ivf_flat_perf.log          IVF-Flat 性能（表 7）
├── ivf_flat_quality.log       IVF-Flat recall@100 = 0.991390
├── ivf_pq16_perf.log          IVF-PQ16 + rerank 性能（表 7）
├── ivf_pq16_quality.log       IVF-PQ recall@100 ≈ 0.012（量化限制，分析见 §7.7）
└── sweep/                     7 组 nprobe（batch=128）的 perf/quality 日志
```

数据规模：`N=300000, D=128, nq=1000, topK=100, nlist=1024, nprobe=16, pq_m=16`，
与另两个平台使用相同生成参数与随机种子，因此 IVF-Flat 的 recall 数值与
`outputs_corex/` 一致。

## 构建与复现

```bash
make -f Makefile.maca -j
export LD_LIBRARY_PATH=/opt/maca/lib:/opt/maca/tools/cu-bridge/lib:/opt/maca/lib64
./build_maca/test_host && ./build_maca/test_gpu

# 该平台的 conda 环境自带 numpy，用其运行数据生成与扫描脚本
/opt/conda/bin/python python/gen_dataset.py --out data_maca --n 300000 --dim 128 \
    --nq 1000 --top-k 100 --metric l2 --clusters 100 --vector-scale 0.10 \
    --query-scale 0.02 --nlist 1024 --nprobe 16
/opt/conda/bin/python python/run_experiments.py \
    --vsearch ./build_maca/vsearch --data data_maca --mode ivf_flat \
    --out-dir outputs_maca
```

该平台的设备能力与差异处理见报告 §3.3 与 §7.2；探测程序见
[tools/probe/](../tools/probe/README.md)。
