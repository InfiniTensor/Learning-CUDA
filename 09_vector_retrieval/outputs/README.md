# 实验产物（NVIDIA 平台）

本目录保存 NVIDIA（RTX 4090）平台的实验日志与结果样例，对应报告 §7.3 的表 1-4。
另外三个平台的产物分别在：

| 目录 | 平台 | 对应报告章节 |
| --- | --- | --- |
| `outputs/`（本目录） | NVIDIA RTX 4090 | §7.3 表 1-4 |
| `outputs_corex/` | 天数智芯 MR-V100 | §7.4 表 5-6 |
| `outputs_maca/` | 沐曦 MetaX MXC500 | §7.5 表 7-8 |
| `outputs_musa/` | 摩尔线程 MUSA | §7.6 表 9-10 |

四平台横向对比见报告 §7.7 表 11；内核耗时剖析见 §8 与
[PROFILING.md](PROFILING.md)。

## 目录结构

```text
outputs/
├── README.md                  本说明
├── PROFILING.md               Nsight Systems kernel 占比分析（报告 §8）
├── experiment_summary.csv     1e6 数据组的 nprobe × batch 扫描汇总（表 2）
├── exact_perf.log             exact bench 性能
├── ivf_flat_perf.log          IVF-Flat 性能
├── ivf_flat_quality.log       IVF-Flat 召回率与距离误差（表 1）
├── ivf_pq16_perf.log          IVF-PQ16 + rerank 性能
├── ivf_pq16_quality.log       IVF-PQ16 + rerank 质量（表 3）
├── sweep/                     21 组 nprobe(1~64) × batch(32/128/512) 的
│                              result / perf / quality 三件套
├── retrieval_results/         exact / IVF-Flat / IVF-PQ16 的 Top-K 结果样例
├── indexes/                   IVF-Flat 与 IVF-PQ16 索引文件
└── nvidia_300k_regression/    300k 数据组的复核产物（表 4）
```

## 数据组对照

| 位置 | 数据规模 | 参数 | 对应报告 |
| --- | --- | --- | --- |
| 根目录日志、`sweep/`、`indexes/`、`retrieval_results/` | 1e6×128，nq=1000 | nlist=4096，nprobe=16，topK=100，pq_m=16 | §7.3 表 1-3 |
| `nvidia_300k_regression/` | 300k×128，nq=1000 | nlist=1024，nprobe=16，topK=100（与另两个平台同参数同种子） | §7.3 表 4 |

## 复现

```bash
# 构建（NVIDIA）
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# 生成 1e6 数据组并扫描
python python/gen_dataset.py --out data \
    --n 1000000 --dim 128 --nq 1000 --top-k 100 --metric l2
python python/run_experiments.py --vsearch ./build/vsearch \
    --data data --mode ivf_flat \
    --nprobe-list 1,2,4,8,16,32,64 --batch-list 32,128,512
```

其他平台的构建与复现命令见各自目录的 README（`outputs_corex/`、
`outputs_maca/`）。
