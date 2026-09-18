# FastHadamard-CUDA

在 `[batch_size, seq_len, num_heads, head_dim]` 激活张量的**最后一维**上做快速
Walsh-Hadamard 变换（FHT），并可与逐 token 对称 INT4 量化融合成单个 kernel。
支持 FP16 / BF16，核心 `head_dim` 为 64/128/256，另实例化 32/512/1024。

用途背景：激活中的少量异常值会放大量化 scale，挤压普通值的有效量化区间。正交
Hadamard 旋转把能量扩散到多个通道，降低单通道动态范围，同时保持网络等价性 ——
这是 QuaRot / SpinQuant 一类 4-bit 量化方案的前置算子。

> **完整技术报告：[`docs/final_report.md`](docs/final_report.md)。**
---

## 1. 实现了哪些路径

同一个数学定义，五条变换路径 + 五条融合量化路径，可在**同一进程、同一 stream、
同一份输入**上互相对照：

| 路径 | 关键结构 | 状态 |
|---|---|---|
| `baseline` | 一 block 一 token，全程 shared memory 蝶形 | 严格验收通过；保留为正确性锚点 |
| `optimized`| `half2`/`bfloat162` packed I/O + warp shuffle + 多 token/block | 严格验收通过 |
| `warp`（warp-per-token） | 每线程 `d/32` 元素，**0 shared memory、0 barrier** | 严格验收通过，**推荐默认** |
| `tc_fast` / `tc_split` | Kronecker 分解 + 两/三次 `m16n16k16` WMMA | 已实现，**严格验收有保留失败** |
| `ptx_fast` / `ptx_split` | 显式 `mma.sync` + warp shuffle，消除 shared 中转 | 实验路径，**未接入默认分发** |

所有路径的蝶形均在 **FP32** 中完成，只在最终写回时舍入到 FP16/BF16；支持归一化
`Hx/sqrt(d)` 与非归一化两种语义。接口按加法方式演进，旧路径数值语义从未改动，
因此任意两条路径都可做严格逐位回归。

## 2. 快速开始

```bash
# 构建（两种方式等价；CUDA_ARCH / CMAKE_CUDA_ARCHITECTURES 按实际 GPU 指定）
make -j2
# 或
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=86 && cmake --build build -j

# 小规模冒烟（含 CPU 参考核对）
make smoke

# 单配置计时与正确性
./build/hadamard_bench \
  --batch 4 --seq 1024 --heads 32 --head_dim 128 \
  --dtype fp16 --normalize true --warmup 5 --iters 20 --check true

# baseline / optimized / unfused INT4 / fused INT4 的 A/B 与三类一致性检查
./build/hadamard_advanced_bench \
  --batch 4 --seq 1024 --heads 32 --head_dim 128 \
  --dtype fp16 --normalize true --warmup 20 --iters 100
```

Tensor Core 分支需要 **SM ≥ 80**（BF16 WMMA 的硬件要求）：

```bash
make -C tensor_core CUDA_ARCH=89 -j4        # L40S=89, A100=80, H100/H200=90
./tensor_core/build/hadamard_tc_bench \
  --batch 4 --seq 1024 --heads 32 --head_dim 256 \
  --dtype fp16 --normalize true --warmup 20 --iters 100
```

### 验证

```bash
bash scripts/run_tests.sh            # FP16/BF16 × 64/128/256 × 多档规模
bash scripts/run_advanced.sh         # 优化 + 融合 INT4 的完整矩阵、CSV、可视化
make -C tensor_core sweep            # warp/TC 全矩阵回归扫描（不替代外部验收）

# 严格参考库验收（需要装有 CUDA PyTorch + fast_hadamard_transform 的解释器）
python tensor_core/scripts/validate_tc.py \
  --bin tensor_core/build/hadamard_tc_bench \
  --output-dir tensor_core/results/library_audit --reference library
```

### Profiler

```bash
bash scripts/profile.sh nsys
bash scripts/profile.sh ncu          # 需要 GPU performance-counter 权限
```

## 3. 怎么读本项目的数字

这套结果横跨五种 GPU、三种计时方法和两类数值参考。

**状态词分四级，不可互相替代**：已实现（能编译）→ 已测试（跑通并有记录）→
严格验收通过（逐配置通过且失败数为 0）→ 已确认收益（多次重复测量的中位数稳定
更快）。

**三种计时口径互不混算**：CUDA Event（无 profiler，正式性能数字）、
CUDA Graph 回放摊销（与外部库的同边界对照）、NCU duration（仅用于 profiler 内部
结构对比）。NCU 会串行化 kernel 并引入采集开销。

**验收判据是固定绝对误差**：参考为 Dao-AILab `fast_hadamard_transform` v1.1.0 的
**全张量低精度输出**，同输入、同 dtype、同 scale，FP16 `< 0.01`、BF16 `< 0.05`，
比较全部元素不采样。相对误差、FP64 参考、token-peak ULP **仅作诊断，不参与判定**
—— 尤其 token-peak ULP 不是逐元素 ULP，`≤ 0.5` 不能证明正确舍入。

**跨 GPU 的数字不混算**，每张表都标注硬件与作业号。

## 4. 主要结果

### 4.1 正确性（严格 132 配置矩阵）

矩阵构成：FP16/BF16 × d=64/128/256/512/1024 × normalize 开关 × tokens=1/5/2048
（60 个），加 FP16/BF16 × d=64/128/256 × normalize 开关 × 六种输入分布（72 个）。
两套参考独立执行、结论一致（独立 FP32 蝶形 + Dao v1.1.0 参考库）：

| 路径 | 严格通过 / 总数 | 最大绝对误差 |
|---|---:|---:|
| `baseline` / `optimized` / `warp`（各自） | **132/132** | **0** |
| `tc_fast` | 96/132 | 1 |
| `tc_split` | 121/132 | 0.5 |
| unfused / `fused_opt` / `fused_warp` INT4（各自） | 132/132 | packed bytes 与 scale 逐位一致 |
| `fused_tc_fast` / `fused_tc_split` INT4（各自） | 108/108 | 与各自未融合算法逐位一致 |

**非 Tensor Core 路径全部通过；Tensor Core 路径有保留失败，因此不作默认。**
TC 的失败主要集中在非归一化输入（split 的 11 个失败全部如此），但这只是观察到的
分布，不构成对归一化输入正确性的证明。历史上的 `PASS(rel)` 口径正是掩盖了这些
绝对误差不达标的配置。

另有 75 项 sanitizer（memcheck/racecheck/synccheck）全部 0 errors、
192/192 幅度与相消压力配置通过、30 项 GPU 负向测试通过。

### 4.2 性能（L40S，131072 tokens，normalize=true，CUDA Event）

变换 kernel（avg μs，粗体为该行最快）：

| dtype | d | baseline | optimized  | warp | tc_fast | tc_split |
|---|---:|---:|---:|---:|---:|---:|
| FP16 | 64 | 73.43 | 21.31 | 21.41 | **18.38** | 21.73 |
| FP16 | 128 | 109.80 | 38.73 | **23.01** | 32.13 | 38.71 |
| FP16 | 256 | 240.66 | 193.00 | **185.31** | 190.17 | 190.98 |
| BF16 | 64 | 73.36 | 21.38 | 21.40 | **19.44** | 23.27 |
| BF16 | 128 | 109.55 | 38.72 | **23.02** | 34.42 | 42.05 |
| BF16 | 256 | 240.89 | 192.91 | **185.18** | 190.17 | 191.23 |

融合 INT4 端到端（avg μs；`unfused` = warp FHT + 独立 quantize 两个 kernel）：

| dtype | d | unfused | fused_opt  | fused_warp | fused_tc_fast | fused_warp |
|---|---:|---:|---:|---:|---:|---:|
| FP16 | 64 | 92.81 | 73.38 | 22.09 | **18.58** | 3.32× |
| FP16 | 128 | 94.37 | 73.55 | **28.82** | 32.85 | 2.55× |
| FP16 | 256 | 237.67 | 103.85 | **48.82** | 62.01 | 2.13× |
| BF16 | 64 | 92.99 | 73.41 | 21.99 | **19.88** | 3.34× |
| BF16 | 128 | 94.17 | 73.66 | **28.68** | 35.16 | 2.57× |
| BF16 | 256 | 237.65 | 103.45 | **48.32** | 66.58 | 2.14× |

TC 融合不支持 d>256（此时一个 token 跨多个 tile，per-token max 无法只靠 sub-warp
归约，接口显式回落到非 TC 融合 kernel，是刻意限制）。INT4 含 scale 的输出压缩率为
3.56×–3.88×，反量化 MAE 0.0912–0.1084。

![优化与融合性能](results/optimization_16970010/figures/advanced_performance.png)

同一环境下 warp / Tensor Core 各路径的时间与误差对照：

![warp 与 Tensor Core 对比](tensor_core/results/tc_17083405/figures/tc_performance.png)

### 4.3 与外部参考库的对照（H200，Graph 回放摊销）

FP16、131072 tokens、normalize=true，中位数 μs。两侧同 GPU、
同输入地址、同 stream，各捕获 16 次调用为 CUDA Graph 后摊销：

| d | Dao v1.1.0 | warp | Dao/warp | Dao+量化 | fused_warp | pipeline/fused |
|---:|---:|---:|---:|---:|---:|---:|
| 64 | 80.568 | 13.770 | **5.85×** | 160.710 | 18.858 | **8.52×** |
| 128 | 80.626 | 22.380 | 3.60× | 160.957 | 27.882 | 5.77× |
| 256 | 80.710 | 36.421 | 2.22× | 161.736 | 47.818 | 3.38× |
| 512 | 81.630 | 70.481 | 1.16× | 219.084 | 78.128 | 2.80× |
| 1024 | 249.396 | 169.180 | 1.47× | 544.036 | 169.388 | 3.21× |

**小规模收益要小得多**：FP16 d=128 在 tokens=1 时为 1.546→1.371 μs，
不能把 3.60× 推广给所有规模。融合对照是"Dao 变换 + 本项目 INT4 量化器"，
Dao 本身不提供融合量化。

![参考库相对性能随输入规模变化](tensor_core/results/libperf_17289704/benchmark/library_comparison.png)

### 4.4 关键结论：Tensor Core 为什么只在 d=64 有效

这是本项目最实质的发现。归因由**两条互相独立的证据**支撑，另有一条修改后的
验证性证据。

**证据一： launch 结构（nsys）**：WMMA 的 accumulator 与 matrix_b fragment 内部
布局不同，第 1 次 mma 的结果必须经 shared memory 中转。这块 scratch 占
9.7–12.8 KB/block，把 resource-limit occupancy 从 100% 压到 83.3%/66.7%，
寄存器从 21 涨到 30–34。而 warp 路径的 shared memory 用量是 **0**。

**证据二： stall 构成（H200 NCU，FP16/d=128）**：

| 指标 | warp | tc_fast | tc_split |
|---|---:|---:|---:|
| dynamic shared KB/block | **0.000** | 9.728 | 12.800 |
| DRAM GB/s | **841.1** | 544.5 | 454.2 |
| stall barrier | **0.00** | 1.06 | 1.32 |
| stall long scoreboard | 10.64 | 4.10 | 2.97 |
| stall short scoreboard | 1.90 | **12.94** | **11.07** |
| stall mio throttle | 1.34 | **10.52** | **12.82** |

warp 路径的 `stall barrier` **精确为 0**，主导 stall 是 global memory 延迟 ——
一个访存受限算子应有的样子。两条 TC kernel 的主导 stall 换成了 `short scoreboard`
与 `mio throttle`：**它们在等 shared memory，不是在等显存**。

**证据三（验证性： 显式 PTX**：用 `mma.sync` + warp shuffle 直接完成
accumulator → operand 重排后，H200 D=128 的 dynamic shared 从 9728 B 降到 **0**，
`short scoreboard / active issue` 从 6.164 降到 0.976，NCU 时间 6.304 → 4.992 μs。
这闭合了整条推理链 —— 但 PTX 与 WMMA 输出逐位一致，**也就继承了同样的精度失败**。

因此 Tensor Core 的收益窗口只在"蝶形级数相对访存量足够多"的小 d 区间。
d ≥ 256 时所有路径都撞到 HBM 上限：d=256 的最小逻辑 I/O 按 864 GB/s 也需 155 μs，
实测 185 μs，五条路径收敛到 1%–3% 以内。

这个访存受限的判断在 baseline 阶段就已成立。下图横轴使用**逻辑最小 I/O**
（一次 16-bit 读 + 一次 16-bit 写）而非实测 DRAM 流量，因为该次采集未取得
NCU 的实际 DRAM bytes：

![L40S baseline 逻辑 I/O Roofline](results/profile_16967596/figures/roofline.png)

**加速比必须对最好的非 TC 实现计算。** 若只对教学 baseline 比较，`tc_fast` 在
d=64 上显示 4.0×；对 warp-per-token 只有 1.16×。本项目一律采用后者。

## 5. 目录结构

```text
张根豪/
├── include/                  # 稳定 host API、dtype、CUDA 错误检查宏
├── src/
│   ├── hadamard.cu           # baseline + optimized + fused INT4
│   ├── hadamard_warp.cu      # warp-per-token FHT（0 shared / 0 barrier）
│   ├── main.cu               # 计时、dump、CSV
│   ├── advanced_bench.cu     # A/B 与三类 bit-exact 检查
│   └── reference_cpu.h       # 开发期 FP32 自检
├── tensor_core/              # WMMA / PTX / 精度诊断实验分支 + 对照基准
├── musa/ muxi/ tianshu/      # 国产平台适配
├── tests/                    # 参考库验收入口与 CPU 工具测试
├── scripts/ slurm/           # 一键测试、profiler 采集、集群作业模板
├── results/                  # 主项目的实测 CSV、图表与 profiler 证据
├── docs/
│   ├── final_report.md       # 唯一技术报告
│   ├── repository_cleanup.md # 归档与恢复说明
│   └── cleanup_manifest.csv  # 归档清单
├── CMakeLists.txt  Makefile
```

## 6. 编译期可选参数

所有优化候选都以编译参数提供，**默认值一律保持原有行为**，不因某次 A/B 的最快
结果自动切换。完整取值与采纳/拒绝理由见报告。

| Make 变量 | CMake 选项 | 说明 |
|---|---|---|
| `WARPS` | `HADAMARD_WARPS_PER_BLOCK` | 非 TC 变换 warps/block；`0` = 按形状自适应 |
| `FUSED_WARPS` | `HADAMARD_FUSED_WARPS_PER_BLOCK` | 融合 INT4 的 warps/block，与变换解耦 |
| `QUANT_VECTOR` | `HADAMARD_QUANT_VECTOR_STORE` | INT4 宽写回；`2` = **仅 d=1024** |
| `TC_WARPS` / `TC_VECTOR` | — | TC 的 warps/block 与 shared 中转向量化 |
| `MMA_PRECISION` | — | 仅作用于独立 PTX split 的精度模式 |
| `OUT` | — | 输出目录。**每组参数必须用独立 `OUT`**，否则 make 会复用旧编译参数 |

```bash
make -C tensor_core h200-tuned                                  # 可选 TC 调优组合
make CUDA_ARCH=90 WARPS=0 QUANT_VECTOR=2 OUT=build/h200-selective # 仅 d=1024 宽写回
```

两个被**拒绝**的候选值得记录，因为它们说明了本项目的取舍标准：
`TC_VECTOR=2`（裁剪末尾同步）变化在 ±0.55% 内，无稳定收益；
`QUANT_VECTOR=1`（全维度宽写回）d=1024 快约 6% 但 **d=512 回退约 2.5%**，
经三个独立进程复测后改为只对 d=1024 开启。**单次最快不等于收益。**

## 7. 国产平台适配

三个适配都不改变 CUDA 主工程的默认构建，不移植 NVIDIA 专有的 WMMA 分支，
覆盖 baseline / optimized / warp / 融合 INT4 四条非 TC 路径：

| 目录 | 平台 | 工具链 | 验证硬件 |
|---|---|---|---|
| [`musa/`](musa/README.md) | 摩尔线程 MUSA | MUSA 5.1.0 / `mcc` | S4000 |
| [`muxi/`](muxi/README.md) | 沐曦 MACA | MACA 3.5.3 / `mxcc` | MetaX C500 |
| [`tianshu/`](tianshu/README.md) | 天数智芯 CoreX | CoreX 4.4 / `ivcore` clang++ | Iluvatar MR-V100 |

```bash
cd musa    && source env.sh && make -j2 && make smoke
cd muxi    && source env.sh && make -j2 && make smoke
cd tianshu && source env.sh && make COREX_CXX=/path/to/corex/clang++ && make smoke
```


## 许可

见 [LICENSE](LICENSE)。
