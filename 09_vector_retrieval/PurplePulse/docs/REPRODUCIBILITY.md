# WarpFusion 正式实验复现记录

## 2026-09-11 第九阶段工程化入口

统一入口为：

```bash
mkdir -p results/real/reproducibility/logs
nohup python3 scripts/reproduce.py --project-root "$PWD" --resume \
  > results/real/reproducibility/nohup.log 2>&1 &
```

默认流水线包含 CMake/Ninja Release 构建、完整 CTest、SIFT1M 与 Yahoo MiniLM
384 维数据准备、WarpFusion/FAISS Exact 与 IVF-Flat 对照、两套图表、技术报告
PDF 和发布校验。`--dataset` 可只选择一套数据，`--skip-download` 与
`--skip-index-build` 用于复用已核验产物，`--dry-run` 输出完整绝对路径计划而不
执行。每个阶段写独立日志，完成一步后原子更新
`results/real/reproducibility/checkpoint.json`；中断后 `--resume` 不重复已经成功
的步骤。

发布校验器：

```bash
python3 scripts/validate_release.py \
  --project-root "$PWD" \
  --report output/pdf/WarpFusion_Technical_Report.pdf \
  --json-output evidence/2026-09-11/stage9/release-validation.json
```

它检查两套公开数据的 10 个正式 CSV、Stage 8 覆盖矩阵、架构/性能 SVG、核心
文档标记和 PDF 完整性。它不会启动长实验，因此适合作为复现流水线的最后一道
fail-closed 检查。

干净 Linux + CUDA 验证使用 CUDA 12.8、GCC 13.3、RTX 4090 D 和全新构建目录，
并以 257 vectors、65 维、13 queries 的 FP32/L2 数据运行 Exact CPU/GPU 与
IVF CPU/GPU。IVF 使用 nlist=16、nprobe=16，确保最小查询同时校验索引读写和
全 probe CPU/GPU 等价。CTest 18/18 通过；Exact 与 IVF 各输出 130 行，候选
集合相同，平均绝对分数误差 3.039×10⁻⁶，最大误差 9.6×10⁻⁶，无近似并列换位。

远端 `nohup` 任务结果位于
`/data/purplepulse-work/PurplePulse/results/stage9_clean_20260911_v2/`，阶段状态、
单项日志、摘要和输出校验和均分阶段落盘。仓库仅保留小型证据：
`evidence/2026-09-11/stage9/`。首次配置因非登录 shell 的 `PATH` 不含
`/usr/local/cuda/bin` 而 fail-closed；修正 PATH 后使用新的 `-v2` 目录重跑，
没有删除或复用失败构建，也没有重复已经完成的 SIFT1M 或 Yahoo MiniLM 长实验。

## 2026-09-10 第八阶段稳定性矩阵

新增 `stability_tests`，使用包含重复向量与全零 query 的确定性、FP16 可精确
表示数据。Exact 主矩阵覆盖 64/128/384/768 维、batch
1/8/32/40/64/128、K=1/10/50/100、FP32/FP16 和 L2/IP/cosine，共 576 组；
另用 65 维覆盖非整 warp 和每种 `batch+1` query 的满 batch + 尾 batch。

IVF 测试人工构造 8 个桶：4 个空桶，最大桶包含 125/128 个向量。在 65 维、
K=1/10/50/100、FP32/FP16、三种 metric 下分别运行 warp 与 bucket-major，
共 48 组。所有 GPU 结果均按 query/rank 比较 ID 与 score；相同分数验证较小
ID 优先，而不是只比较 recall 或结果数量。

远端任务与结果：

- `stage8-stability-build-20260910`：nohup 构建和完整 CTest，状态 0，14/14。
- `stage8-stability-sanitizer-20260910`：先落盘覆盖摘要，再运行
  `compute-sanitizer --tool memcheck`，状态 0、0 errors。
- 结果目录：`/data/purplepulse-work/PurplePulse/results/stage8_stability/`；仓库
  保存 `../results/stage8_stability/coverage.csv`，大日志继续只留在 `/data`。

保守的 Exact/IVF 实现选择、GPU 适用范围和 fail-closed 回退规则见
`STABILITY_MATRIX.md`。FP16、非 batch64、64/768 维或其他 GPU 虽已验证正确性，
但没有被错误标记为已有正式性能调优；这些组合默认回退到完整 warp-query heap。

## 2026-09-10 第七阶段 Exact fused 剖析

新增 `--profile-breakdown 1 --profile-repeat N` 和
`scripts/benchmark_exact_breakdown.py`。剖析器使用与 fused 路径相同的网格、
query cache 和距离归约运行一个 distance-only kernel；该 kernel 每个 warp
仅写一个 checksum，以阻止编译器删除计算。最终归并由 CUDA event 直接计时，
局部 Top-K 则定义为 `fused kernel - distance-only kernel`。因此它是可复现的
differential estimate，不应解释为三个硬件计数器区间的严格可加时间线。

正式运行使用 RTX 4090 D、batch=64、预热 5 次、查询重复 20 次、剖析重复
20 次。正式基线先完成候选集合与分数校验；随后同一基线在 A/B 复测中使用加严
后的驱动，按 query、rank、候选 ID 和分数与既有 Exact 参考逐项比较。结果如下。

| 数据集 | K | QPS | distance-only | 局部 Top-K 估算 | 最终归并 | 占比（距离 / 局部 / 归并） |
|---|---:|---:|---:|---:|---:|---:|
| Synthetic-1M | 10 | 6743 | 144.94 ms | 2.38 ms | 0.19 ms | 98.26% / 1.61% / 0.13% |
| Synthetic-1M | 50 | 6083 | 145.53 ms | 17.13 ms | 0.83 ms | 89.02% / 10.48% / 0.51% |
| Synthetic-1M | 100 | 5484 | 146.29 ms | 33.08 ms | 1.86 ms | 80.72% / 18.25% / 1.03% |
| Yahoo MiniLM 384 | 10 | 3525 | 279.28 ms | 3.40 ms | 0.19 ms | 98.73% / 1.20% / 0.07% |
| Yahoo MiniLM 384 | 50 | 3366 | 280.78 ms | 13.61 ms | 0.88 ms | 95.09% / 4.61% / 0.30% |
| Yahoo MiniLM 384 | 100 | 3172 | 281.07 ms | 30.70 ms | 2.00 ms | 89.58% / 9.79% / 0.64% |

完整 CSV 位于 `../results/stage7_exact_profile/`。这组结果把后续优先级指向
数据库读取与跨 query 数据复用；只优化最终 merge 不可能达到阶段目标。

### 已否决：cosine 范数预计算

第一项 A/B 尝试在建库时预计算数据库范数，并在每批预计算 query 范数，避免
cosine fused kernel 重复归约范数。基线和候选版本在同一任务中顺序运行，均
预热 5 次、重复 20 次，并通过逐 query、逐 rank 的 ID/score 校验。

| 数据集 | K | 基线 QPS | 候选 QPS | 候选 / 基线 |
|---|---:|---:|---:|---:|
| Synthetic-1M | 10 | 6741 | 6628 | 0.983× |
| Synthetic-1M | 50 | 6067 | 5993 | 0.988× |
| Synthetic-1M | 100 | 5478 | 5425 | 0.990× |
| Yahoo MiniLM 384 | 10 | 3497 | 3407 | 0.974× |
| Yahoo MiniLM 384 | 50 | 3341 | 3330 | 0.997× |
| Yahoo MiniLM 384 | 100 | 3160 | 3240 | 1.025× |

该方案只在 Yahoo K=100 提升 2.5%，Yahoo K=10 下降 2.6%，Synthetic 全部
下降 1%--2%，因此未进入默认代码。候选版本源码已撤回，基线与候选 CSV、
二进制/源码校验和、恢复后的当前源码校验和及可重放脚本保存在
`../results/stage7_exact_profile/cosine_norm_ab/` 与
`../evidence/2026-09-10/stage7/`。撤回后在远端重新构建，CTest 12/12 通过。

### 已保留：跨 query 数据库读取复用

`exact_query_tile=2/4/8` 让一个 fused block 对同一向量 chunk 同时计算多个
query，每个数据库分量只从全局显存读取一次，再用于 tile 内所有 query。
`exact_fused_blocks=N` 可固定 `grid.x`；默认值 0 使用并行度启发式。初版直接
减少 `grid.y` 导致并行度下降，修订版按 query tile 放大 `grid.x`，使总 block
数维持在约 1024。两版 smoke 均保留在结果目录，避免只报告成功版本。

正式项统一预热 5 次、重复 20 次，驱动在每个 variant 后 checkpoint CSV，并
按 query、rank、ID 和 score 与第六阶段 Exact 参考逐项校验。

| 数据集 | K | 基线 QPS | 最佳配置 | 最佳 QPS | 加速 |
|---|---:|---:|---|---:|---:|
| Synthetic-1M | 10 | 6739 | tile4 / 64 blocks | 9250 | 1.373× |
| SIFT1M | 10 | 6940 | tile4 / 64 blocks | 9228 | 1.330× |
| Yahoo MiniLM 384 | 10 | 3577 | tile8 / 160 blocks | 6747 | 1.887× |
| Yahoo MiniLM 384 | 50 | 3363 | tile4 / 64 blocks | 4074 | 1.211× |

Synthetic/SIFT K=50 的最佳策略仍是 baseline；tile2 分别为 0.986×/0.962×，
tile4 为 0.917×/0.890×。K=100 smoke 中 tile2/tile4 在 Synthetic 仅为
0.684×/0.567×，Yahoo 为 0.712×/0.698×，因此没有继续正式长跑。K=10 的
最终归并仍低于 0.3 ms；Yahoo 最佳配置相对 baseline 只增加约 1.06 MiB
显式 buffer。完整 CSV 位于 `../results/stage7_exact_query_tile/`，源码与配置
校验和位于 `../evidence/2026-09-10/stage7/query_tile_source.sha256`。

优化后的 K=10 相对 FAISS GPU Flat 差距为 Synthetic 2.30 倍、SIFT1M
2.24 倍、Yahoo 3.15 倍，尚未达到 2 倍阶段门槛。后续应改用 block 级数据库
tile 保持 query 并行度，或针对 K=50/100 比较专用选择结构，而不是继续扩大
同一 warp 内的 query tile。

### 已否决：block 级数据库 tile

`exact_fused_layout=block_tile` 让一个 block 的 8 个 warp 各自负责一个 query，
并通过共享内存复用 8 个数据库向量，目的是避免 query-tile 中单 warp 串行维护
多个 query heap。筛选统一使用 batch=64、K=10、预热 1 次、重复 3 次，扫描
`grid.x=64/96/128/160/192`；每个 variant 均与既有 Exact 参考逐 query、逐 rank
比较 ID 和 score。CSV 在每个 variant 完成后落盘。

| 数据集 | baseline QPS | warp-query 最优 | block-tile 最优 | block / warp |
|---|---:|---:|---:|---:|
| Synthetic-1M | 6929 | 9265（tile4/64） | 9050（tile8/128） | 0.977× |
| Yahoo MiniLM 384 | 3620 | 6989（tile8/160） | 4794（tile8/128） | 0.686× |

block-tile 虽分别比未 tiled baseline 快 30.6% 和 32.5%，但没有超过已经接纳的
warp-query；Yahoo 还退化 31.4%。同步与共享内存中转成本抵消了保留 query
并行度的收益，因此不再运行 SIFT 或 20 次正式长跑。该路径保留为显式 opt-in
负实验，默认 `exact_fused_layout=warp_query` 不变；原始筛选 CSV 位于
`../results/stage7_exact_block_tile/smoke/`。

### 选择结构裁决与阶段验收

当前 Exact fused 并非单线程 heap：每个 block 的 8 个 warp 分别维护局部 heap，
块内先合并成 `K` 项，再由第二个 kernel 归并各 chunk，已经是分层
Warp/Block Top-K。实际运行过的候选物化对照是旧 matrix 路径：它先写回
`batch × N` score；K=10 使用每线程候选缓冲后归并，K=50/100 使用 block
反复选择。Synthetic-1M K=10/50/100
分别只有 1775/716/444 QPS，而对应 fused 为 5564/5123/4701 QPS，慢
3.13--10.59 倍且显式 buffer 多约 33%。

对全量 block radix/select 的裁决属于设计边界，不冒充实测：若只保留少量局部
候选，它仍需先使用 heap/selection 得到这些候选；若要对全部精确 score 做
radix/select，则必须物化 `batch × N` score，或为每个 radix pass 重算距离。
前者回到已否决 matrix 路径，后者破坏单遍扫描并放大已经占 80.72%--98.73%
的主瓶颈，因此没有为“列齐算法名称”而实现不可接纳路径。第七阶段最终未达到
2 倍硬目标，但满足计划规定的失败分支：保存可复现 profile、严格正确性结果、
被否决方案、默认回退和原因。

## 2026-09-10 第六阶段公开数据

第六阶段使用 SIFT1M 与 VIBE `yahoo-minilm-384-normalized`。数据准备、校验、
转换和完整基准入口分别为 `scripts/prepare_real_datasets.py` 与
`scripts/run_real_benchmarks.py`。公开源与固定校验和如下：

| 数据集 | 公开源 | 规模 | 校验和 |
|---|---|---:|---|
| SIFT1M | `ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz` | 1M × 128 | MD5 `b23d1b3b2ee8469d819b61ca900ef0ed` |
| yahoo-minilm-384-normalized | `https://huggingface.co/datasets/vector-index-bench/vibe` | 677,305 × 384 | SHA-256 `8b0519850249d7ea4aea258eb43ae837dca1bb2f7a85ee0a5145afdb699c4136` |

正式矩阵统一使用 nlist=1024、训练样本 100,000、15 次迭代、seed 2026、
batch=64，以及 K=10/50/100、nprobe=32/64/128/160/192/224。完整命令见
README 的“第六阶段公开真实数据”。每套结果目录已包含 `build.csv`、
`exact_gpu.csv`、三份有效 IVF CSV、FAISS GPU CSV、公开 ground truth 验证、
分区 Top-K 回退判定、环境快照、原始输出和 recall-QPS/latency 图。

### 正式结果摘要

正式运行使用 RTX 4090 D、CUDA 12.8.61、`batch=64`、预热 1 次和重复 5 次。
下表固定 `nprobe=160`；两套系统使用相同的 nlist、训练样本、迭代次数、K、
nprobe 和 batch，但各自训练聚类中心。

| 数据集 | K | WarpFusion recall / QPS | FAISS GPU recall / QPS | WF / FAISS |
|---|---:|---:|---:|---:|
| SIFT1M | 10 | 0.999870 / 24579.1 | 0.999120 / 13616.6 | 1.81× |
| SIFT1M | 50 | 0.999706 / 19236.8 | 0.999516 / 13442.7 | 1.43× |
| SIFT1M | 100 | 0.999565 / 17362.2 | 0.999457 / 13102.4 | 1.33× |
| Yahoo MiniLM 384 | 10 | 0.993900 / 15733.2 | 0.995300 / 6216.2 | 2.53× |
| Yahoo MiniLM 384 | 50 | 0.991520 / 14949.7 | 0.991280 / 6157.2 | 2.43× |
| Yahoo MiniLM 384 | 100 | 0.989230 / 14196.2 | 0.989080 / 6059.8 | 2.34× |

WarpFusion/FAISS IVF 建库时间分别为 SIFT1M 25.47/31.69 秒、Yahoo
87.54/77.89 秒。WarpFusion Exact K=10/50/100 吞吐为 SIFT1M
7038/6337/5622 QPS、Yahoo 3610/3471/3286 QPS。Exact 与公开 ground truth
存在极少数同分 ID 选择差异，但两套数据的 score-equivalent query fraction
均为 1.0；SIFT 的 rank/report score error 为 0，Yahoo 的最大报告分数误差为
`2.98e-7`。

K=50 的分区路径在 SIFT1M 和 Yahoo 上分别损失 0.000326 和 0.000980 recall，
超过预设 `1e-4` 阈值，`ivf_k50.csv` 因此自动复制完整 heap 结果。K=100 在
两套数据的全部六个 nprobe 点上均保持零 recall 损失，正式结果保留分区路径；
`nprobe=160` 下相对完整 heap 分别快 1.38 倍和 1.72 倍。提交仓库保存
`results/real/` 下的 CSV、JSON 和 SVG；原始 Top-K 输出与日志保留在实验机
`/data/purplepulse-work/PurplePulse/results/real/`。

## 2026-09-08 增量实验

新 GPU 实例仍为 RTX 4090 D / CUDA 12.8，但从空工作目录重新同步源码、配置
并构建。CTest 8/8 通过，Compute Sanitizer 报告 0 errors。环境、测试与数据
校验和位于 `../evidence/2026-09-08/`。

本轮增加六组结果：K=50 batch 调优、缓存感知桶任务调度、K=100 单 warp
正式 A/B、K=100 分区局部 Top-K、IVF 多线程建库和 MNIST 真实数据。原始 CSV 位于
`../results/formal_4090d_20260908/`。百万数据并行建库：

```bash
./build/ivf_build \
  --database data/million/inner_fp32_db.bin \
  --output data/million/inner_fp32_nlist256.ivf \
  --nlist 256 --iterations 15 --training-samples 100000 --threads 16
```

K=50 在 nprobe=160 下先以 10 次重复扫描 batch=8/32/36/40/44/48/56/64/
80/96/128，再对 batch=40 和 batch=64 各预热 1 次、正式重复 100 次：

```bash
./build/ivf_search \
  --index data/million/inner_fp32_nlist256.ivf \
  --queries data/million/inner_fp32_q.bin \
  --params configs/ivf_flat_million_k50_warp_compact_batch40.conf \
  --output results/k50_batch40.txt \
  --backend gpu --warmup 1 --repeat 100

./build/ivf_search \
  --index data/million/inner_fp32_nlist256.ivf \
  --queries data/million/inner_fp32_q.bin \
  --params configs/ivf_flat_million_k50_batch64.conf \
  --output results/k50_batch64.txt \
  --backend gpu --warmup 1 --repeat 100

python3 scripts/evaluate_recall.py \
  --exact results/new_gpu/exact_k50.txt \
  --approximate results/k50_batch40.txt
```

batch=40 的正式结果为 6019.91 QPS、run P50/P99=166.303/168.097 ms、
recall@50=0.92482；同实例 batch=64 对照为 5737.20 QPS，严格 A/B 提升
4.93%。筛选表、warp 对照和原始日志分别位于
`../results/formal_4090d_20260908/` 与 `../evidence/2026-09-08/`。

缓存感知调度使用相同索引、K、batch、nprobe 和输出语义，仅改变 probe block
的执行顺序。正式 A/B 命令为：

```bash
./build/ivf_search \
  --index data/million/inner_fp32_nlist256.ivf \
  --queries data/million/inner_fp32_q.bin \
  --params configs/ivf_flat_million_k50_warp_compact_batch40.conf \
  --output results/k50_probe_major.txt \
  --backend gpu --warmup 5 --repeat 100

./build/ivf_search \
  --index data/million/inner_fp32_nlist256.ivf \
  --queries data/million/inner_fp32_q.bin \
  --params configs/ivf_flat_million_k50_bucket_major.conf \
  --output results/k50_bucket_major.txt \
  --backend gpu --warmup 5 --repeat 100

cmp results/k50_probe_major.txt results/k50_bucket_major.txt
```

K=50 的正式结果为 6009.22 与 6964.16 QPS，提升 15.89%；两份输出完全
一致。nprobe=32/64/96/128/160/192/224 的扫描结果位于
`k50_bucket_major_sweep.csv`。K=100 单独启用 bucket-major 提升 2.00%；加入
heap 根节点寄存器缓存、空穴下沉和单 warp 直写后，同轮 100 次 A/B 从
5298.37 提高到 5425.61 QPS，再与 bucket-major 组合达到 5491.76 QPS，累计
提升 3.65%。三条路径的输出逐字一致，该组合构成后续分区实验的完整 heap
基线。

K=50 的分区候选筛选包含 2×32、4×20、4×16 与 8×8。未接纳的 2×32、4×20
和 8×8 分别取得 7.37%、9.51% 与 2.88% 的增益；4×16 是唯一超过 10% 接纳线的
配置。正式命令为：

```bash
./build/ivf_search \
  --index data/million/inner_fp32_nlist256.ivf \
  --queries data/million/inner_fp32_q.bin \
  --params configs/ivf_flat_million_k50_partitioned_4x16.conf \
  --output results/k50_partitioned_4x16.txt \
  --backend gpu --warmup 5 --repeat 100
```

同一二进制的完整 heap 基线为 7165.23 QPS，4×16 为 7934.10 QPS，提升
10.73%；输出逐字一致，recall@50 均为 0.92482。七个 nprobe 点的增益范围为
6.04% 至 22.86%，均保持输出一致。原始 CSV、输出和 memcheck 日志位于
`../results/formal_4090d_20260908/k50_partitioned_tight/`；2×32 与 4×20 记录
位于 `../results/formal_4090d_20260908/k50_partitioned/`。

K=100 的分区局部 Top-K 使用 4 个 warp、每个 warp 40 项局部候选。正式复测
保持同一二进制、索引、`batch=64` 和 `nprobe=160`：

```bash
./build/ivf_search \
  --index data/million/inner_fp32_nlist256.ivf \
  --queries data/million/inner_fp32_q.bin \
  --params configs/ivf_flat_million_k100_partitioned.conf \
  --output results/k100_partitioned.txt \
  --backend gpu --warmup 5 --repeat 100
```

完整 heap 基线为 5503.11 QPS，分区路径为 7204.54 QPS，提升 30.92%；两份
输出逐字一致，recall@100 均为 0.91758。`nprobe=128/160/192/224` 的 10 次
扫描分别提升 26.49%/31.28%/34.32%/36.80%。原始 CSV、结果文件与 memcheck
日志位于 `../results/formal_4090d_20260908/k100_partitioned/`。

三项未采用的 K=100 选择或排序实验记录在
`results/formal_4090d_20260908/k100_selection_negative.csv`。分布式寄存器
Top-K 重复 100 次，仅提升 0.43%；buffer-8 快速重复 10 次后退化 22.68%，
因此按停止规则未进入 100 次阶段。原始摘要、资源用量和回退校验见
`evidence/2026-09-08/k100_selection_negative.log`。
warp bitonic sort 也在 10 次快速 A/B 中退化 18.72%，输出逐字一致，因而
同样按停止规则终止。

K=50 的 FAISS 同批次补测固定 batch=40、nprobe=160、预热 5 次、重复 100 次。
隔离环境使用 Python 3.12.3、FAISS GPU 1.14.1 和 NumPy 2.2.6，并复用主机
CUDA 12.8 的 runtime/cuBLAS。FAISS Flat 与 WarpFusion Exact 的候选 ID
完全一致。FAISS IVF-Flat 为 8328.11 QPS、0.92312 recall@50；WarpFusion
最终 4×16 路径为 7934.10 QPS、0.92482 recall@50，FAISS 吞吐高 4.97%。完整命令见
`../evidence/2026-09-08/commands.md`，原始 CSV 为
`../results/formal_4090d_20260908/faiss_gpu_batch40.csv`。

MNIST 归档来自 Keras 公共数据地址
`https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz`。转换、
建库与扫描命令为：

```bash
python3 scripts/import_mnist.py \
  --input data/mnist/mnist.npz \
  --database data/mnist/database.bin \
  --queries data/mnist/queries.bin

./build/ivf_build \
  --database data/mnist/database.bin --output data/mnist/nlist256.ivf \
  --nlist 256 --iterations 15 --training-samples 60000 --threads 16

python3 scripts/benchmark_ivf_sweep.py \
  --search-binary ./build/ivf_search \
  --index data/mnist/nlist256.ivf --queries data/mnist/queries.bin \
  --params configs/ivf_flat_mnist.conf \
  --exact results/mnist_exact_k10.txt \
  --output-dir results/mnist_ivf --csv results/mnist_ivf.csv \
  --nlist 256 --nprobes 8,16,32,64,96,128 \
  --training-samples 60000 --training-iterations 15 \
  --warmup 2 --repeat 10
```

下文保留 2026-09-04 的主 FAISS 对照口径。

本文固定本轮正式结果的数据、训练、延迟和环境口径。公开结果文件不包含原始
数据与索引；二者通过生成命令和 SHA-256 校验和审计。

## 1. 环境

- GPU：NVIDIA GeForce RTX 4090 D，24,564 MiB，compute capability 8.9。
- Driver：570.124.06。
- OS：Ubuntu 24.04.1 LTS。
- CUDA Toolkit / NVCC：12.8 / 12.8.61。
- CMake / Ninja / GCC：3.31.4 / 1.11.1 / 13.3.0。
- FAISS 专用 venv：FAISS 1.14.1 GPU，NumPy 2.2.6，Python 3.12.3；CPU
  对照固定 1 线程。系统 Python 不含 FAISS，不用于 FAISS 正式实验。
- 构建类型：Release，`CMAKE_CUDA_ARCHITECTURES=native`。

GPU 机上的部署目录不是 Git checkout。正式实验前，核心源码和本地基准提交
`fea41f6` 的 SHA-256 逐文件一致；随后新增的评测字段不改变检索 kernel。
完整环境输出、测试与 profiler 摘要保存在 `../evidence/2026-09-04/`。

## 2. 数据与索引

```bash
python3 scripts/generate_data.py \
  --database data/million/inner_fp32_db.bin \
  --queries data/million/inner_fp32_q.bin \
  --num-vectors 1000000 \
  --num-queries 1000 \
  --dim 128 \
  --dtype fp32 \
  --metric inner_product \
  --seed 2026
```

| 文件 | SHA-256 |
|---|---|
| `inner_fp32_db.bin` | `d4dde1418ea7626e5226e594e592f9ae5ffbe4c412c326395d96388f95df9800` |
| `inner_fp32_q.bin` | `a58ebe43fce865b71a2a651791286fa0bd1567348e07d6c0601b69beaebceb45` |
| `inner_fp32_nlist256_spherical_100k15.ivf` | `f5bd87fcc72d37164fa931436e5bfd293dfb7f953d09cb28169a77c7cdadc415` |
| FAISS Flat index | `760e3a94f24c5becfa21e89a57f67395013d5ad5375c20eb3f5d5cfb4c3720a1` |
| FAISS IVF-Flat index（100k/15/seed 2026） | `a66a63bb28ed9e70ae83c042f57a8ca79de3ee01a6414377452a7610738431b1` |

WarpFusion 正式索引的构建命令为：

```bash
./build/ivf_build \
  --database data/million/inner_fp32_db.bin \
  --output data/million/inner_fp32_nlist256_spherical_100k15.ivf \
  --nlist 256 \
  --iterations 15 \
  --training-samples 100000
```

构建耗时 67,678 ms，桶大小 min/avg/max 为
3734/3906.25/4091，无空桶，索引大小 520,133,160 bytes。

两种实现都使用 100,000 个训练向量和 15 次迭代，但不强制共享中心：

- WarpFusion 从全库等间距、确定性抽取训练样本，初始中心也等间距选择；
  当前实现没有单独的训练随机种子。
- FAISS 用 NumPy `default_rng(2026)` 无放回抽样，并把聚类 seed 设为 2026。

所以这是相同训练预算和检索参数下的端到端系统比较，不是相同桶扫描比较。

## 3. 正式查询命令

Exact 的 K=10/50/100 均使用 batch=64、预热 5 次、正式重复 100 次：

```bash
for k in 10 50 100; do
  ./build/vector_search \
    --database data/million/inner_fp32_db.bin \
    --queries data/million/inner_fp32_q.bin \
    --params configs/exact_million_k${k}_fused_batch64.conf \
    --backend gpu \
    --output results/formal_100k15/exact_k${k}.txt \
    --warmup 5 --repeat 100
done
```

IVF 对 nprobe=128/160/192/224 扫描。K=10 的示例为：

```bash
python3 scripts/benchmark_ivf_sweep.py \
  --search-binary ./build/ivf_search \
  --index data/million/inner_fp32_nlist256_spherical_100k15.ivf \
  --queries data/million/inner_fp32_q.bin \
  --params configs/ivf_flat_million_batch64.conf \
  --exact results/formal_100k15/exact_k10.txt \
  --output-dir results/formal_100k15/ivf_k10 \
  --csv results/formal_100k15/ivf_k10.csv \
  --nlist 256 --nprobes 128,160,192,224 \
  --training-samples 100000 --training-iterations 15 \
  --warmup 5 --repeat 100
```

K=50/100 分别替换为对应的 batch64 配置和 Exact 文件。FAISS 使用同一数据、
K、batch、nlist、nprobe、训练预算、预热和重复次数；完整命令保存在
`../evidence/2026-09-04/commands.md`。

## 4. 延迟与质量定义

- `run P50/P99`：一次处理全部 1000 queries 的端到端稳定查询时间。
- `batch P50/P99`：每个 batch 的 GPU 时间，从 query H2D 开始，到 Top-K
  ID/score D2H 完成为止；包含 H2D、所有检索 kernel 和 D2H。
- 稳定查询不包含数据库/索引的一次性加载、一次性 H2D 和引擎初始化；这些
  时间另列，不能混入常驻 QPS。
- GPU 正式项每个 run 有 16 个 batch；100 次重复共产生 1600 个 batch 延迟
  样本。单线程 CPU Flat 重复 20 次，共 320 个 batch 延迟样本。
- `recall@K` 是返回 ID 集合与 Exact Top-K 的交集比例。
- `mean/max absolute score error` 按名次比较近似 Top-K 与 Exact Top-K 的
  分数绝对差；它衡量漏召回造成的分数质量损失，不等同于同 ID 数值误差。
  Exact kernel A/B 的分数误差则按相同候选 ID 比较。

## 5. 自动验证

```bash
ctest --test-dir build --output-on-failure
compute-sanitizer --tool memcheck --error-exitcode 99 ./build/unit_tests
```

本轮 CTest 8/8 通过，`compute-sanitizer` 报告 0 errors。Nsight Systems 的
Exact K10 代表性采样显示 fused 距离+局部 Top-K kernel 占 GPU kernel 时间
约 99.9%；文本摘要已入库，原始 `.nsys-rep` 因体积和机器相关性不提交。

新增评测字段和两处样本数日志上传后，GPU 端已再次完成增量编译，CTest 仍为
8/8。新版 IVF 和 FAISS 脚本分别完成百万数据 smoke test，确认训练元数据、
run/batch 样本数与平均/最大分数误差均能写入 CSV；对应日志文件名包含
`after_instrumentation`。

为排除构建目录、数据文件和历史结果造成的隐式依赖，另将源码排除
`build/`、`data/`、`results/`、`evidence/` 和 `.git/` 后解压到新目录，从零
执行 CMake、编译和 CTest，8/8 通过。输出见
`../evidence/2026-09-08/submit_audit_ctest.log`。
