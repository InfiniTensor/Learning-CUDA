# GPU Profiling：Nsight Systems 实测分析

## 1. 环境与工作负载

- GPU：NVIDIA RTX 4090 24 GB（sm_89）
- 系统：Ubuntu 24.04
- 驱动：570.124（用户态 libcuda 已修正为同版本）
- CUDA Toolkit：12.8；Nsight Systems：2024.6.2
- 工作负载：`N=10^6, D=128, nq=1000, topK=100, nlist=4096, nprobe=16`

## 2. 采集方法

```bash
# exact 检索
nsys profile --force-overwrite=true -o prof/nsys_exact -t cuda,osrt \
  ./build/vsearch search --vectors=data_ann/vectors.bin \
  --queries=data_ann/queries.bin --params=data_ann/params.txt \
  --search_mode=exact --batch_size=128

# IVF-Flat 检索
nsys profile --force-overwrite=true -o prof/nsys_ivf_flat -t cuda,osrt \
  ./build/vsearch search --vectors=data_ann/vectors.bin \
  --queries=data_ann/queries.bin --params=data_ann/params.txt \
  --search_mode=ivf_flat --nprobe=16 --batch_size=128 \
  --index=outputs/ann_flat/index.idx

# IVF-PQ16 + rerank 检索
nsys profile --force-overwrite=true -o prof/nsys_pq -t cuda,osrt \
  ./build/vsearch search --vectors=data_ann/vectors.bin \
  --queries=data_ann/queries.bin --params=data_ann/params.txt \
  --search_mode=ivf_pq --pq_m=16 --pq_rerank=256 --nprobe=16 \
  --batch_size=128 --index=outputs/ann_pq/index_pq.idx

# 查看 kernel/内存操作汇总
nsys stats --report cuda_gpu_sum prof/nsys_exact.nsys-rep
nsys stats --report cuda_gpu_sum prof/nsys_ivf_flat.nsys-rep
nsys stats --report cuda_gpu_sum prof/nsys_pq.nsys-rep
```

采集得到的 `.nsys-rep` 报告即上述命令中 `-o prof/nsys_*` 指定的文件。

## 3. exact 检索 kernel 占比

| Kernel | GPU 时间占比 | 平均耗时 |
| --- | ---: | ---: |
| `exactKeysKernel`（全量打分） | 74.9% | 49.2 ms |
| CUB segmented radix sort（两级内核） | 23.1% | ~1.3-1.6 ms/instance |
| H2D 查询/键上传等 | 1.9% | 24 μs avg |
| `decodeTopKeysKernel` | <0.1% | 1.6 μs |

分析：exact 的主要成本是“读取 512 MB 向量库并计算 1e6×128 的点积”；分段排序占
约 1/4 时间，属于精确检索的固定开销。

## 4. IVF-Flat 阶段（不包含 exact baseline）

| Kernel | 总耗时（1000 query） | 平均/instance | 说明 |
| --- | ---: | ---: | --- |
| `ivfCenterKeysKernel` | 2.57 ms | 0.32 ms | 128 query 对 4096 中心打分 |
| `ivfFlatKeysKernel` | 23.37 ms | 2.92 ms | 候选向量精确重打分 |
| `gatherCandidatesKernel` | 84 μs | 10.5 μs | 倒排桶拼接 |
| probe/count/scan/decode 等 | <1 ms 合计 | 微秒级 | 候选管理和解码 |
| CUB 分段排序 | 约 22.5% | - | 中心排序 + 候选 Top-K |

分析：IVF-Flat 中每次 batch（128 query）需要重算约 3900/query 个候选，
`ivfFlatKeysKernel` 平均仅 2.9 ms；倒排拼接、扫描、解码开销都很小。进一步优化应
减少排序数据量或使用阈值式 Top-K。

## 5. IVF-PQ16 + rerank 阶段

| Kernel | 平均/instance | 说明 |
| --- | ---: | --- |
| `ivfCenterKeysKernel` | 0.32 ms | query 到中心 |
| `pqTableKernel` | 5.4 μs | 128 query × 16×256 ADC 表 |
| `pqPackKeysKernel` | 0.126 ms | ADC 候选打分 |
| CUB segmented sort（ADC 粗排） | 数十 μs~1.4 ms | 与数据量相关 |
| `rerankPrepKernel` | 0.33 ms | 取 ADC top-256 |
| `ivfFlatKeysKernel`（精排） | 0.08 ms | 只重算 top-256 |
| `decodeVariableTopKernel` | 1.2 μs | 结果写出 |

分析：PQ 表构建极便宜；ADC 打分只有 0.126 ms/batch。rerank 把原始向量重算限制到
每 query 256 个，因此 recall 与 IVF-Flat 相当，搜索总耗时也未增加。

## 6. 结论与后续优化建议

1. `exactKeysKernel` 是 full-scan 主瓶颈，后续可用 float4 向量化、每线程多行合并
   减少索引/调度开销，或使用 fp16/tensor core 累计内积。
2. 精确 Top-K 的分段 radix sort 约占 1/4；候选集更大时可改“粗阈值 + 紧凑选择”
   降低排序带宽。
3. IVF 的 gather/scan 仅数微秒，说明倒排结构设计合理，不需要微优化。
4. nsys 时间线适合定位“哪个 kernel 占时间”；若需要 SM/DRAM 占用率等硬件计数器，
   可在具备硬件计数权限的宿主环境用厂商计数器工具补测 `exactKeysKernel` 的
   访存吞吐，以确认是否已达 DRAM 带宽上限。
