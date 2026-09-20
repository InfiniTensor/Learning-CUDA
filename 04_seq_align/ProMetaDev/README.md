# GPU 生命序列比对 —— 选题四（生命序列比对）

基于 CUDA 的 **seed-and-extend** 读段比对程序：把海量 FASTQ reads 快速比对到 FASTA 参考基因组，
定位最佳匹配位置并给出比对得分，无法比对者标记为 `unknown_origin`。

- 参考基因组规模：实测 **10⁸ bp**（题目要求 ≥100 条 × 10⁸ bp，见「局限」一节说明）
- reads 规模：实测 **10⁵ 条 × 150 bp**
- 打分参数（按题目要求）：匹配 **+2**、错配 **−1**、空位 **−1**（线性空位罚分）

## 目录结构

```
proj_seqalign/
├── include/
│   ├── types.h        # 参考基因组 / read / 命中 / 配置
│   ├── dna.h          # 碱基编码与 k-mer 哈希（host/device 共用）
│   ├── io.h           # FASTA / FASTQ / 结果读写
│   └── align.h        # 各比对器接口
├── src/
│   ├── main.cpp       # CLI: cpu / cpuseed / gpu / selftest
│   ├── io.cpp         # FASTA / FASTQ 解析、结果输出
│   ├── cpu_ref.cpp    # CPU 穷举参考（正确性基准）
│   ├── cpu_seeded.cpp # CPU 版同算法实现（性能基准）
│   └── align_gpu.cu   # GPU: k-mer 索引 + 种子枚举 + 带状 DP 验证
├── tools/
│   └── gen_data.py    # 合成 FASTA/FASTQ/真值生成器
├── tests/
│   ├── run_tests.sh   # 端到端测试
│   └── benchmark.sh   # 多规模性能基准
├── CMakeLists.txt
├── build.sh
└── README.md
```

## 构建

```bash
bash build.sh [SM_ARCH]      # 默认 90；同时生成 PTX 以兼容更新架构
```

产物：`build/seqalign`

## 使用

```bash
# 自检
./build/seqalign selftest

# GPU 比对
./build/seqalign gpu <ref.fa> <reads.fq> <out.txt> [options]

# CPU 穷举（正确性基准，代价正比于参考长度，仅适合小参考）
./build/seqalign cpu <ref.fa> <reads.fq> <out.txt> [options]

# CPU 同算法实现（性能基准）
./build/seqalign cpuseed <ref.fa> <reads.fq> <out.txt> [options]
```

### 选项

| 选项 | 默认 | 说明 |
|---|---|---|
| `--k <int>` | 15 | 种子长度（1..16） |
| `--band <int>` | 8 | 带状 DP 半宽（4/8/16），即允许的最大净空位数 |
| `--seed-step <int>` | 8 | read 上相邻种子间隔 |
| `--max-cand <int>` | 64 | 每条 read 最多验证的候选数 |
| `--min-ratio <float>` | 0.7 | 判定阈值：`score ≥ ratio × match × read_len` 才算已比对 |
| `--index-bits <int>` | 0 | 索引桶位数，0 = 按参考规模自动选取 |
| `--truth <file>` | — | 真值文件，用于统计准确率 |
| `--perf <file>` | — | 性能日志输出路径 |

### 输入 / 输出

**FASTA 参考**（多条序列，序列可跨行）：
```
>chr1
ACGTACGT...
>chr2
GCTAGCTA...
```

**FASTQ reads**（每 4 行一条）：
```
@read1
ATCGATCG
+
IIIIIIII
```

**结果**（每行 `<read名> <参考序列名> <起始位置> <得分>`；未比对为 `<read名> unknown_origin`）：
```
read1 chr1 0 8
read2 chr2 0 8
read3 unknown_origin
```

## 算法设计

### 流水线

```
参考基因组 ──► ① k-mer 索引（GPU 计数排序 + 散列分桶，常驻显存）
reads ───────► ② 种子枚举（GPU：每 (read, seed) 一线程查表 → 候选锚点）
              ─► ③ 候选去重（每 read 一线程：插入排序 + 相邻去重）
              ─► ④ 带状 DP 验证（每候选一线程：窗口内自由起始的 fitting 比对）
              ─► ⑤ 每 read 归约（取最高分，同分取最左）
```

索引只与参考有关，构建一次即可服务任意批次的 reads。

### ① k-mer 索引

- A/C/G/T 分别编码为 0/1/2/3，k ≤ 16 时可把 k-mer 压进一个 32 位整数。
- 含 `N` 的 k-mer 不进入索引；参考序列之间用 64 个 `N` 填充，天然阻止 k-mer 与比对窗口跨越序列边界。
- 用**计数排序**建桶：先统计每个哈希桶的条目数 → 前缀和 → 散列写入 `(code, pos)`。
  桶内额外存 `code`，查表时用它过滤哈希碰撞，**不产生假阳性**。

### ② 种子枚举

read 上每隔 `seed_step` 取一个 k-mer 作为种子（150bp read、step=8 时约 17 个种子），
查索引得到该 k-mer 在参考上的全部出现位置 `p`，候选锚点 `q = p - off`。
每条 read 的候选用原子追加写入定长容量区（默认 64 个）。

### ③④ 带状 DP：窗口内自由起始的 fitting 比对

给定锚点 `q`，窗口取 `ref[q-B, q+L+B)`（长度 `L+2B+1`），要求**整条 read 被完整比对到窗口内的某一段**，
起始位置在窗口内自由。用对角偏移 `e = j - i - B ∈ [-B, B]` 作列下标（`i` 消耗的 read 碱基数、
`j` 消耗的参考碱基数），递推为

```
D[i][e] = max( D[i-1][e]   + s(read[i-1], ref[q+i+e-1]),   // 匹配/错配
               D[i-1][e+1] + gap,                          // 参考中出现空位
               D[i][e-1]   + gap )                         // read 中出现空位
```

一线程负责一个候选，两行 DP 缓冲驻留寄存器（`B=8` 时每行 17 个 int32）。

**关键实现细节：把「得分」和「起始位置」打包进同一个 int32**：

```
packed = (score + L) * (2B+1) + (2B - t)      t = 起始列编号 ∈ [0, 2B]
```

这样只需一个 DP 数组，且取 `packed` 最大天然实现了「得分最高，同分取最左」的确定性规则
（`score + L ≥ 0` 保证整数除法可安全解码）。

### 关于「unknown_origin」的判定

题目给定的打分中 **错配(−1) 与空位(−1) 同分**，这会让 DP 主动用空位去「挑」更匹配的参考位置：
随机 read 也能刷到满分的 ~50%（实测 150bp 随机 read 得分约 138/300）。
因此单靠 DP 最优值无法区分真假，本实现用两道闸门：

1. **种子**：随机 read 几乎不可能在参考上找到精确 k-mer（10⁸ bp 参考、k=15 时，
   一个随机 15-mer 命中的期望次数约 10⁻³），天然把随机 read 挡在验证之外；
2. **阈值**：`score ≥ min_ratio × match × L`（默认 0.7），用于拒绝个别能刷出高分的读段。

自带来源的 read（3% 替换率）得分约为满分的 95%，与随机 read 的 ~50% 有充分间隔。

## 性能

RTX 5070 Ti Laptop（sm_120，JIT 运行 sm_90 PTX），CPU 为 32 核。
`bg` 表示参考碱基数，`align` 为纯比对阶段（不含索引构建与文件解析）。

| 规模 (参考bp / reads) | 索引条目 | GPU 索引 | GPU 比对 | CPU 索引 | CPU 比对 | 比对加速比 |
|---|---|---|---|---|---|---|
| 50 k / 5 k | 15 k | 1.11 ms | 0.56 ms | 0.86 ms | 8.94 ms | 16.0× |
| 200 k / 20 k | 1.6 M | 6.23 ms | 3.80 ms | 32.2 ms | 16.5 ms | 4.3× |
| 1 M / 50 k | 20 M | 82.6 ms | 11.2 ms | 1134 ms | 60.6 ms | 5.4× |
| **5 M×20 = 10⁸ / 10⁵** | **100 M** | **459 ms** | **30.5 ms** | **10 630 ms** | **232 ms** | **7.6×** |

目标规模下：
- GPU 比对吞吐 **≈ 3.3 × 10⁶ reads/s（≈ 500 Mbp/s）**
- 端到端（索引 + 比对）：GPU 490 ms vs CPU 10 862 ms → **22.2×**
- 索引构建：GPU 459 ms vs CPU（单线程）10 630 ms → **23.2×**

## 正确性验证

| 验证项 | 结果 |
|---|---|
| 引擎自检（精确片段 / 突变片段 / 随机片段，CPU vs GPU） | PASS |
| 小规模 GPU vs **CPU 穷举**（4 组 k/band 组合，逐行 diff） | 完全一致 |
| 中等规模（1.6 Mbp × 2000 reads）真值召回 | 有来源 1600/1600 被比对；随机 400/400 判为 unknown_origin |
| 目标规模（10⁸ bp × 10⁵ reads）真值召回 | 有来源 80000/80000；随机 20000/20000 |
| **目标规模 GPU vs CPU 同算法实现逐字节 diff** | **完全一致** |
| 结果确定性（重复运行） | 逐字节一致 |

起点完全一致率 96.8~97.0%，得分完全一致率 86.6~88.8%（差异来自 DP 合法地用空位换取更高分，
真值文件记录的是"采样位置 + 无空位理论分"，并非 DP 最优值）。

## 局限

1. **未在题目要求的最大规模上实测**：题目要求 ≥100 条 × 10⁸ bp（合计 10¹⁰ bp）。
   本机 12 GB 显存与磁盘无法承载（10¹⁰ bp 的 k-mer 索引约 120 GB），
   实测做到 10⁸ bp。程序结构对规模是线性的，扩到 10¹⁰ bp 需要
   ① minimizer / 双索引降低索引体积，② 参考分片流式处理。
2. **种子敏感度**：低同源度（>15% 突变）或高 indel 的 read 可能因找不到精确种子而漏比。
   放宽 `--k` / `--seed-step` 可提升敏感度，代价是候选数与耗时上升。
3. **阈值依赖数据**：`min_ratio` 需要按测序质量调整，理想做法是像 BLAST 那样
   用参考规模做统计显著性（E-value）标定。
4. **未使用 FASTQ 质量分**（题目第 4 行，属加分项）。
5. **未做国产平台适配**。
6. **`ncu` / `nsys` 未采集**：WSL2 不透传 CUPTI。此外本机 GPU 为 sm_120，
   而 CUDA 12.2 不支持 sm_120 原生编译，只能由驱动 JIT 运行 sm_90 PTX，
   首次内核启动有一次性编译开销（已在计时前预热剔除）。

## 复现

```bash
bash build.sh 90
./build/seqalign selftest
bash tests/run_tests.sh 90

# 生成目标规模数据（10⁸ bp 参考 + 10⁵ reads，约 20 秒）
python3 tools/gen_data.py --ref-seqs 20 --ref-len 5000000 --reads 100000 \
    --read-len 150 --sub-rate 0.03 --random-frac 0.2 --seed 2026 \
    --out-ref tests/tmp/target.fa --out-reads tests/tmp/target.fq \
    --out-truth tests/tmp/target.truth

./build/seqalign gpu tests/tmp/target.fa tests/tmp/target.fq out.txt \
    --k 15 --band 8 --seed-step 8 --truth tests/tmp/target.truth

bash tests/benchmark.sh 90
```
