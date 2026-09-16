# CoreX（天数智芯）输出

与 `outputs/`（NVIDIA RTX 4090）对应，本目录保存天数智芯 Iluvatar CoreX
（MR-V100，IX-ML 4.4.0）实测日志与扫描汇总，供报告 §11 引用。

```text
experiment_summary.csv nprobe(1,2,4,8,16,32,64) × batch(32,128,512) 汇总
exact_perf.log         GPU exact（200 query 子集 CPU 对比 + 1000 query 性能）
exact_quality.log      exact 模式质量
ivf_flat_perf.log      IVF-Flat（300k×128，nprobe=16）
ivf_flat_quality.log   IVF-Flat recall@100（0.986）
ivf_pq16_perf.log      IVF-PQ16 + rerank
ivf_pq16_quality.log   IVF-PQ16 recall@100（ADC 量化限制，约 0.012）
sweep/                 21 组 nprobe × batch 的 perf/quality 日志
```

数据规模：`N=300000, D=128, nq=1000, topK=100`（修复后文件头格式）。本目录
收录汇总与日志；完整 result.txt 与 .idx 如需复现，可按下方命令重新生成。

构建与复现：

```bash
make -f Makefile.corex -j
export LD_LIBRARY_PATH=/usr/local/corex/lib64:/usr/local/corex/lib
python python/gen_dataset.py --out data_c2 --n 300000 --dim 128 --nq 1000 \
    --top-k 100 --metric l2 --clusters 100 --vector-scale 0.10 \
    --query-scale 0.02 --nlist 1024 --nprobe 16
python python/run_experiments.py --vsearch ./build_corex/vsearch \
    --data data_c2 --mode ivf_flat --out-dir outputs_c2
```
