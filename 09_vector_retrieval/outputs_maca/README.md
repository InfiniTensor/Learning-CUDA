# 沐曦 MetaX（MACA）输出

与 `outputs/`（NVIDIA）、`outputs_corex/`（天数智芯）对应，本目录保存沐曦 MetaX
MXC500（MACA 3.5.3，cu-bridge）实测日志与扫描汇总，供报告 §11.4 引用。

```text
experiment_summary.csv nprobe(1,2,4,8,16,32,64) × batch(32,128,512) 汇总
exact_perf.log         GPU exact（200 query 子集 CPU 对比 + 1000 query 性能）
exact_quality.log      exact 模式质量
ivf_flat_perf.log      IVF-Flat（300k×128，nprobe=16）
ivf_flat_quality.log   IVF-Flat recall@100 = 0.991390
ivf_pq16_perf.log      IVF-PQ16 + rerank
ivf_pq16_quality.log   IVF-PQ recall@100 ≈ 0.012（ADC 量化限制）
sweep/                 7 组 nprobe（batch=128）的 perf/quality 日志
```

数据规模：`N=300000, D=128, nq=1000, topK=100`（与 NVIDIA / CoreX 相同的生成参数
与随机种子，因此 recall 数值与 CoreX 逐位一致）。

构建与复现：

```bash
make -f Makefile.maca -j
export LD_LIBRARY_PATH=/opt/maca/lib:/opt/maca/tools/cu-bridge/lib:/opt/maca/lib64
./build_maca/test_host && ./build_maca/test_gpu

# 沐曦自带 conda 环境含 numpy，用其运行数据生成与扫描脚本
/opt/conda/bin/python python/gen_dataset.py --out data_maca --n 300000 --dim 128 \
    --nq 1000 --top-k 100 --metric l2 --clusters 100 --vector-scale 0.10 \
    --query-scale 0.02 --nlist 1024 --nprobe 16
/opt/conda/bin/python python/run_experiments.py \
    --vsearch ./build_maca/vsearch --data data_maca --mode ivf_flat \
    --out-dir outputs_maca
```

说明：本平台**无需任何源码改动**——沐曦设备端的 64 位 `atomicAdd` 与 double 累加
均正常，因此引擎沿用 NVIDIA 代码路径（详见报告 §11.4.2）。
