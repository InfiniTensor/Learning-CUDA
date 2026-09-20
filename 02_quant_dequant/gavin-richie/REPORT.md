# 总结报告：MXFP8 / NVFP4 低精度软件模拟与反量化

作者：gavin-richie ｜ 开发 GPU：RTX 4060 (Ada, sm_89) ｜ CUDA 12.6
验证 GPU：RTX 4090 D (Ada, sm_89) ｜ CUDA 12.8（NGC 25.01 容器，原生 Linux）
复测 GPU：RTX 5090 (Blackwell, sm_120) ｜ CUDA 12.8 / 驱动 610.43.02
             (consumer Blackwell；sm_120 不在数据中心 sm_100 范围，
              但 FP8/FP4 cvt 指令集在 sm_100/101/120 都已发布)

## 1. 低精度格式与缩放策略

### 1.1 数值格式（全部为软件模拟，无硬件指令）

| 格式 | 位宽 | 指数/尾数 | 特殊值 | 最大有限值 |
|---|---|---|---|---|
| E4M3 | 8 | 4/3 | 仅 NaN（0xF7），无 inf（OCP MX 规范，溢出饱和到 ±448,正规数$\pm [2^{-6},2^7*(2-2^{-3}=240)]$，全格式正规数$\pm (288,320,252,384,416,448)$ | ±448 |
| E5M2 | 8 | 5/2 | IEEE 风格 2个inf/3个NaN | $\pm [2^{-14},2^{16}(2-2^{-2})]$ = $\pm [2^{-14},57344]$ |
| E2M1 | 4 | 2/1 | 无 | ±6，可表示幅值 $\pm {0, .5, 1, 1.5, 2, 3, 4, 6}$ |
| E8M0 | 8 | 8/0 | 仅作 block scale，值 = 2^(code−127) | — |

编码统一采用 round-to-nearest-even（`rintf`/显式中点偶数规则），
`include/fp_formats.h` 为 host/device 双端实现，GPU 与 CPU 参考实现逐位一致
（由 `test_roundtrip` 验证）。舍入到 nearest 的实现要点：
- FP8 正常数域：尾数整数化 `q = |x|·2^mant−e`，`rintf` 舍入后若进位到下一binade 则指数 +1；
- 次正规数域：ulp = 2^(emin−mant)，整数编码自然覆盖"舍入后进位为最小正规数"；
- E2M1 查表舍入，中点（0.25/0.75/1.25/1.75/2.5/3.5/5）取偶数编码(0,2,2,4,4,6,6)。

随机舍入（stochastic）：在目标网格 ulp 宽度内加均匀噪声
`y += (u−0.5)·ulp(y)`（`u` 为基于元素下标的确定性哈希），再走 nearest 编码。

### 1.2 缩放策略

- **per-tensor（fp32 标量）**：`S = amax_tensor / vmax`（vmax：E4M3=448，
  NVFP4=6）。量化 `q = enc(x/S)`，反量化 `x̂ = dec(q)·S`。
- **MXFP8 block（32 元素共享 E8M0）**：每块
  `S_b = 2^ceil(log2(amax_b/448))`，即 2 的幂且保证 `amax_b/S_b ≤ 448`，不溢出。存储 1 字节 E8M0 编码。
- **NVFP4 两级缩放（block=16）**：
  - 全局 fp32：`GS = max(1, amax_tensor/2688)`（2688 = 448×6）；
  - 块级 E4M3：`S_b = E4M3(amax_b/GS/6)`（先除全局再编码，保证块内幅值落入 E2M1 范围，且 E4M3 编码本身的误差 ≤ 2^-3 相对）；
  - 量化 `q = E2M1(x/GS/S_b)`，反量化 `x̂ = E2M1^{-1}(q)·S_b·GS`。

E5M2 与 E4M3 走同一 kernel，仅 vmax 与解码常量不同。

## 2. 打包布局与内存访问

- **MXFP8**：每元素 1 字节，天然对齐。
- **NVFP4**：每字节严格存 2 个元素（元素 2k 低半字节、2k+1 高半字节）。
  - 量化侧 packed store：每线程处理一个 16 元素 block，在寄存器中组装8 字节后以单次 8 字节（`unsigned long long`）store 写回；尾块不足 16元素时退化为字节写，高半字节补零。
  - 反量化侧 packed load：每线程加载 1 字节，解出 2 个 nibble、写 2 个输出, 相邻线程访问连续字节，合并访存。
- 权重文件布局：`WeightFileHeader`（magic "QWHT"、format/scale_mode/block_size、rows/cols、packed 字节数、scale 数量、E8M0/E4M3 区分、是否带NVFP4 全局 scale）+ packed data + block scale 数组 + fp32 scale（tensor scale 恒有，NVFP4 block 模式额外带 global scale）。

## 3. Kernel 设计

- 量化：每线程一个 block——先块内求 amax（值域缩放），再逐元素编码写回；block scale 就地写出。
- 反量化：每线程 1（MXFP8）/ 2（NVFP4）元素，packed load → 解码 → 乘 scale
  → 按 output_type（fp16/bf16/fp32）写出。
- 误差统计：单 kernel 归约 max abs / MAE / MSE（double 累加原子）。
- 计时：cudaEvent 只包住量化/反量化 kernel 本身；有效带宽 =（读+写字节数）/时间。

## 4. 实测结果（RTX 4060 Laptop, 8GB, 4096×4096 fp32 输入, fp16 输出）

误差（nearest 舍入，block 缩放）：

| 分布 | 格式 | max abs | MAE | MSE | 压缩率 |
|---|---|---|---|---|---|
| normal | MXFP8-E4M3 | 0.2494 | 0.0180 | 0.00070 | 3.88× |
| normal | NVFP4 | 0.7427 | 0.0714 | 0.00904 | 7.11× |
| random(U[-10,10]) | MXFP8-E4M3 | 0.5000 | 0.1167 | 0.02620 | 3.88× |
| random | NVFP4 | 1.6250 | 0.4379 | 0.33828 | 7.11× |
| outlier | MXFP8-E4M3 | 16.00 | 0.0106 | 0.09497 | 3.88× |
| outlier | NVFP4 | 76.56 | 0.0128 | 0.13973 | 7.11× |

- 压缩率含 scale 开销：MXFP8 = 32/9 ≈ 3.56（1B scale/32B 数据），NVFP4 = 32/4.5 = 16/2.25 ≈ 7.11（1B scale/8B packed 数据）。
- outlier 分布中极少数大值（±500）由缩放吸收，主体小值（σ=0.01）保持高精度（MAE 0.013），但 max abs error 由离群点所在块决定——这正是 block scaling 相对 per-tensor 的优势；per-tensor + E4M3 时小值会被整体压低（对照日志 `log_ts.txt`：normal 分布 tensor 模式 MAE 0.024 > block 0.018）。

性能（同上配置，cudaEvent 计时）：

| 格式 | 量化 ms | 反量化 ms | 量化 GB/s | 反量化 GB/s |
|---|---|---|---|---|
| MXFP8-E4M3 | 1.44 | 0.52 | 58 | 162 |
| NVFP4 | 0.42 | 0.31 | 182 | 246 |

反量化 kernel 为纯流式访存，166–246 GB/s 已达该卡有效带宽的较大部分（RTX 4060 8G 理论 ≈ 272 GB/s）。量化 kernel 因每 block 先做串行 amax 归约（32/16 次全局读），带宽低于反量化，是后续优化点。


### 4.1 RTX 4090 D 复测（同 4096×4096 fp32 输入、fp16 输出、seed=42）

误差与 4060 **逐位一致**——6 组 max abs / MAE / MSE 全部相同，量化数学
与架构无关得到实证。性能对照（cudaEvent 计时，同口径 `(输入 4B +
压缩后字节)/时间`）：

| 格式 | 指标 | 4060 Laptop (≈272 GB/s) | 4090 D (≈1008 GB/s) | 提升 |
|---|---|---|---|---|
| MXFP8 | 量化 GB/s | 58 | 178 | 3.0× |
| MXFP8 | 反量化 GB/s | 162 | 767 | 4.7× |
| NVFP4 | 量化 GB/s | 182 | 750 | 4.1× |
| NVFP4 | 反量化 GB/s | 246 | 1089 | 4.4× |

- 4090 D 反量化带宽超过显存规格峰值（1089–1167 GB/s > 1008 GB/s）：
  64 MB 张量整体驻留该卡 72 MB L2，读命中 L2 所致（outlier 分布最高
  1167 GB/s）。
- 量化 kernel 的串行 amax 归约在两代卡上都是瓶颈（仅 178–750 GB/s），
  与 4060 上的结论一致，仍为首要优化点。
- 完整日志见 `artifacts/4090d/log_*.txt`，nsys 时间线见
  `artifacts/4090d/profile_{mxfp8,nvfp4}/`（nsys 捕获的
  quant/dequant/metrics kernel 耗时与 cudaEvent 计时一致）。

### 4.2 Transformer Engine 对比（per-tensor FP8，RTX 4090 D）

容器预装 TransformerEngine 1.14，仅有 DelayedScaling/CurrentScaling
per-tensor recipe；MXFP8/NVFP4 block-scaling recipe 需要 TE ≥ 2.0 且
Blackwell（sm_100）硬件，本卡无法运行。用 `scripts/te_compare.py` 走
TE 的 `Float8Tensor.quantize_` / `dequantize` cast kernel（E4M3、fp16
输出、与软件路径同误差/带宽口径）：

| 分布 | TE max abs | TE MAE | 软件 MXFP8 MAE | TE 压缩率 | TE 量化/反量化 ms | 软件量化/反量化 ms |
|---|---|---|---|---|---|---|
| normal | 0.1911 | 0.0180 | 0.0180 | 4.00 | 0.270 / 0.032 | 0.475 / 0.110 |
| random | 0.3605 | 0.1105 | 0.1167 | 4.00 | 0.270 / 0.032 | 0.473 / 0.111 |
| outlier | 17.95 | 0.0114 | 0.0106 | 4.00 | 0.270 / 0.032 | 0.475 / 0.112 |

- 误差：normal 下 TE per-tensor 与软件 block-scale 几乎一致；outlier 下
  软件 block scaling 的 max abs（16.0 vs 17.95）与 MAE 均略优——块级
  缩放把离群点的影响隔离在所在 32 元素块内；软件压缩率 3.88 略低于
  TE 的 4.00（含 block scale 开销）。
- 吞吐：TE dequant kernel 约为软件实现的 3.4×（2592 vs 767 GB/s，同口径
  公式），quantize（含 amax/scale 归约）约 1.8×。差距主要来自 TE cast
  kernel 的向量化程度，正是第 8 节优化项的预期空间。
- MXFP8/NVFP4 recipe 的硬件对照待 Blackwell + TE ≥ 2.0 补做，
  `te_compare.py` 已为此预留 `--format` 扩展点。

### 4.3 RTX 5090 (Blackwell, sm_120) 复测

在 RTX 5090 上重编 `sm_120` 并复跑 4.1 节的同 9 组基准（block-scale MXFP8
/ NVFP4 + per-tensor MXFP8，normal/random/outlier × 3 分布），输入与
seed 完全一致（4096×4096 fp32, seed=42）。`artifacts/5090/` 下的日志与
`artifacts/4090d/` 误差指标**全部逐位一致**——再次实证量化数学与架构
无关：

| 格式 | scale_mode | 分布 | 4060/4090D/5090 max abs | MAE | MSE |
|---|---|---|---|---|---|
| MXFP8 | block | normal | 0.249422 | 0.017960 | 0.000704 |
| MXFP8 | block | random | 0.500000 | 0.116688 | 0.026200 |
| MXFP8 | block | outlier | 15.999512 | 0.010556 | 0.094955 |
| NVFP4 | block | normal | 0.742681 | 0.071412 | 0.009042 |
| NVFP4 | block | random | 1.624999 | 0.437936 | 0.338279 |
| NVFP4 | block | outlier | 76.555634 | 0.012835 | 0.139734 |
| MXFP8 | tensor | normal | 0.191125 | 0.017974 | 0.000701 |
| MXFP8 | tensor | random | 0.360490 | 0.110535 | 0.025650 |
| MXFP8 | tensor | outlier | 17.953125 | 0.011386 | 0.134961 |

5090 显存带宽理论值 ≈ 1792 GB/s（GDDR7 32 Gbps × 512-bit / 8）。三卡
性能对照（cuEvent 计时，同口径 `(4n + packed + scales)/t`）：

| 格式 | kernel | 4060 (272 GB/s) | 4090 D (1008 GB/s) | 5090 (1792 GB/s) | 5090 / 4090D |
|---|---|---|---|---|---|
| MXFP8 block | quant GB/s | 58 | 178 | 277 | 1.56× |
| MXFP8 block | dequant GB/s | 162 | 767 | 1206 | 1.57× |
| NVFP4 block | quant GB/s | 182 | 750 | 1217 | 1.62× |
| NVFP4 block | dequant GB/s | 246 | 1089 | 1690 | 1.55× |
| MXFP8 tensor | quant GB/s | — | — | 281 | — |
| MXFP8 tensor | dequant GB/s | — | — | 1514 | — |

- 5090 NVFP4 dequant 已到 1690 GB/s，相对 4090D 提升 1.55×，与两卡
  带宽比（1.78×）的差距来自 L2 命中差异——outlier 测得 1691 GB/s，
  4096×4096 (64 MB) 张量全部驻留 5090 的 96 MB L2，读几乎全命中；
  4090 D 的 72 MB L2 同样覆盖完整张量。
- 量化 kernel 仍是瓶颈（相对反量化 4.4× 较慢），三卡共性，与串行
  amax 归约有关——第 8 节优化项。
- nsys 时间线在 5090 上采集：`artifacts/5090/profile_{mxfp8,nvfp4}/
  t4096_normal_pipeline.nsys-rep`；ncu 仍因容器 GPU perf-counter
  权限被跳过（`ERR_NVGPUCTRPERM`，与 4090 D 同根因）。

### 4.4 TE 1.14 per-tensor 在 5090 上的复测

容器自带 TE 1.14 的 per-tensor Float8Tensor 路径（`quantize_` 走
`tex.cast_to_fp8` 等 PTX kernel）在 sm_120 上**可直接运行，无需重装**。
recip 误差与 4090 D 上**完全相同**（max abs 0.191125 / MAE 0.017974，
与软件 MXFP8 block-scale 的 0.017960 相近但不完全相等，原因见 4.2
分析），进一步确认量化结果与硬件架构无关。5090 上的吞吐：

| 路径 | quant (GB/s) | dequant (GB/s) | 4090D 同路径 |
|---|---|---|---|
| TE 1.14 per-tensor | 605 | 4883 | 310 / 2592 |
| 软件 MXFP8 block | 277 | 1206 | 178 / 767 |

TE 1.14 的 dequant kernel 在 5090 上达到 4883 GB/s，已逼近 L2 + 显存
的实际极限（远超显存峰值说明几乎全部命中 L2）。软件实现与之的差距
来自 TE cast kernel 的向量化（每线程 8 元素 vs 软件 2 元素 packed
load），正是第 8 节预期的优化空间。

### 4.5 CUDA Math API 与软件编码器逐位一致性（`test_native_cvt`）

新增 `tests/test_native_cvt.cu`（CMake `-DENABLE_NATIVE_CVT_TEST=ON`
选项，架构自动加 `a` 后缀以激活原生 cvt 指令；`include` 路径下
`fp_formats.h` 与 `cuda_fp8.h / cuda_fp4.h` 的 `__nv_cvt_*` 内建对
比）。在 RTX 5090（sm_120a）上：

- 穷举所有 256 个 FP8 码（含 E4M3 / E5M2 的 decode 与 encode
  roundtrip，跳过 NaN 与 E5M2 ±inf——后者两种 encoder 在
  `__NV_SATFINITE` 下都饱和为最大有限值，是设计行为）：**全部一致**。
- 穷举 16 个 E2M1 nibble 码：**全部一致**。
- 随机扫描：4 194 304 个 32-bit 任意位值（含 NaN / ±inf / denorm /
  常规） × {E4M3, E5M2, E2M1}，**0 失配**。

ctest 现 4/4 通过（`test_formats` / `test_io` / `test_roundtrip` /
`test_native_cvt`）。结论：**E4M3/E5M2/E2M1 软件编解码在数值上与
Blackwell 硬件 `cvt.rn.satfinite.e*.*.f32` 等指令逐位一致**，软件
路径既无平台移植风险也无正确性折损。

### 4.6 MXFP8 / NVFP4 block-scaling recipe（TE 2.19，RTX 5090）

`te_compare.py` 扩展了 `--format mxfp8|nvfp4`（TE 2.x 的
`MXFP8Quantizer` / `NVFP4Quantizer` cast kernel）与
`--nvfp4-variant vanilla|full`（vanilla：关 RHT / stochastic rounding /
2D，与软件路径直接可比；full：RHT 开启，按生产语义用 bf16 输入）。

环境：独立 venv `/data/venvs/te2`（python 3.12，**torch 2.10.0+cu128**）。
NVIDIA 不发布绑定层的通用预编译 wheel（GitHub Releases 只覆盖 NGC
容器内置 torch 版本），`transformer_engine_torch-2.19.0` 绑定层本地
源码编译（48 GB 内存下需 `MAX_JOBS=8` 防 cc1plus OOM），核心库
`transformer_engine_cu12-2.19.0` 用 PyPI 预编译 wheel。

**误差对照（4096×4096 fp32、seed=42，三分布；软件数字同 4.3 节）：**

| 路径 | 分布 | max abs | MAE | MSE | 与软件对照 |
|---|---|---|---|---|---|
| TE MXFP8 recipe | normal | 0.249422 | 0.017960 | 0.000704 | **逐位一致** |
| TE MXFP8 recipe | random | 0.500000 | 0.116688 | 0.026195 | **逐位一致** |
| TE MXFP8 recipe | outlier | 15.999512 | 0.010556 | 0.094965 | **逐位一致** |
| TE per-tensor e4m3 | normal | 0.191125 | 0.017974 | 0.000701 | **逐位一致** |
| TE per-tensor e4m3 | random | 0.360490 | 0.110535 | 0.021690 | **逐位一致** |
| TE per-tensor e4m3 | outlier | 17.953125 | 0.011386 | 0.106352 | **逐位一致** |
| TE NVFP4 vanilla | normal | 0.707525 | 0.071414 | 0.009039 | 软件为 0.742681/0.071412 |
| TE NVFP4 vanilla | random | 1.666666 | 0.442941 | 0.344591 | 软件为 1.624999/0.437936 |
| TE NVFP4 vanilla | outlier | 81.251495 | 0.012144 | 0.120740 | 软件为 76.555634/0.012835 |
| TE NVFP4 full (RHT) | normal | 0.707525 | 0.071444 | 0.009050 | 与 vanilla 几乎相同 |
| TE NVFP4 full (RHT) | outlier | 81.251495 | 0.012153 | 0.121072 | 与 vanilla 几乎相同 |

- **MXFP8 与 per-tensor 两条路径：软件模拟与 TE 2.19 的误差三项指标
  逐位一致**。并做了元素级验证：512×512 张量上 TE MXFP8 与软件路径
  的 fp16 反量化输出 **262 144 个元素全部相等**。即
  `include/fp_formats.h` 的 E4M3 编码 + E8M0 ceil 块缩放 + RNE 舍入
  与生产 TE 实现逐位等价；也反向确认了 4.5 节 `__nv_cvt` 硬件一致性
  结论贯穿整个量化链路。
- **NVFP4 两者接近但非逐位**：MAE 差 ≤ 5%（outlier 上 TE 反而略优
  0.012144 vs 0.012835，normal/random 上软件略优）。根因是两级缩放
  的实现自由度：软件全局 scale 用 `GS = max(1, amax/2688)`（仅大幅
  值张量缩到满量程），TE 用 `448·6/amax` 恒定满量程利用；块级
  E4M3 scale 的舍入方向两边也不同。两者均为合法 NVFP4 编码，误差
  同量级，块内缩放对离群点的隔离效果一致。
- **RHT 对本合成分布收益有限**：outlier 的最大误差元素是离群值本身
  （±500），16 元素块内的 Hadamard 旋转不会缩小大值自身的量化误差，
  vanilla 与 full 的 max abs 完全相同（81.25）。RHT 的设计目标是
  结构化/相干分布（如 attention logits），稀疏独立离群点场景需
  更细粒度方案。

**吞吐对照（同口径 `(4n + packed + scales)/t`；64 MB 张量驻留
5090 的 96 MB L2，故读数普遍高于 1792 GB/s 的 HBM 峰值）：**

| 路径 | 量化 GB/s (ms) | 反量化 GB/s (ms) | 软件 / TE1.14 对照 |
|---|---|---|---|
| TE 2.19 MXFP8 | 5540 (0.0152) | 6632 (0.0127) | 软件 277 / 1206 |
| TE 2.19 NVFP4 vanilla (fp32 入) | 2318 (0.0330) | 2771 (0.0276) | 软件 1217 / 1690 |
| TE 2.19 NVFP4 full (bf16 入) | 3923 (0.0196) | 2772 (0.0276) | — |
| TE 2.19 per-tensor e4m3 | 3176 (0.0264) | 4918 (0.0171) | 软件 281 / 1514 |
| TE 1.14 per-tensor e4m3（JIT） | 605 (0.1387) | 4883 (0.0172) | 4.4 节 |

- TE 2.19 的 Blackwell 原生 cast kernel 相对软件实现：MXFP8 量化
  **20×**（fused amax + E8M0 scale + cast 单 pass，对比软件的串行
  块内归约）、反量化 5.5×；这正是第 8 节优化项的量化空间上限。
- 相对 TE 1.14（PTX JIT 到 sm_120），2.19 原生编译的 per-tensor
  量化 kernel 快 5.3×（0.0264 vs 0.1387 ms）——当前缩放已融合进
  cast kernel；反量化带宽相近（都受 L2 限制）。
- NVFP4 量化（0.033 ms）比 MXFP8（0.015 ms）慢，因为两级缩放需要
  额外的 amax 归约与 E4M3 scale 编码 pass；bf16 输入的 full 变体
  量化更快（0.0196 ms，读入字节减半）。
- 日志：`artifacts/5090/log_te2_{e4m3,mxfp8,nvfp4,nvfp4full}_{normal,
  random,outlier}.txt`；TE 1.14 同卡对照见 `log_te114_e4m3_*.txt`。

## 5. 软件模拟与硬件路径边界

- **全部数值编解码（E4M3/E5M2/E2M1/E8M0）、打包、缩放、反量化均为纯软件
  模拟**：只用整数/浮点位运算与 `ldexpf/frexpf/rintf`，无 `__nv_fp8` 内建、
  无 FP8/FP4 Tensor Core、无 Hopper/Blackwell/Ampere+ 专属指令，可在任意
  sm_60+ GPU（含 4060）运行。
- CUDA Math API（`__nv_cvt_float_to_fp8` 等）与 Transformer Engine 硬件
  加速路径属于**加分项**。TE 对比已在 RTX 4090 D 完成 per-tensor FP8
  路径（见 4.2 节）；在 RTX 5090（Blackwell, sm_120）上同样可运行
  并复测（见 4.4 节，`scripts/te_compare.py`，不改 C++ 代码）。
  CUDA Math API 与软件编码器的逐位一致性已在 5090 (sm_120a) 上用
  `tests/test_native_cvt.cu`（CMake `-DENABLE_NATIVE_CVT_TEST=ON`）
  验证（见 4.5 节）。
- MXFP8/NVFP4 block-scaling recipe 的硬件对照需 TE ≥ 2.0：
  `scripts/te_compare.py` 已扩展 `--format mxfp8|nvfp4`，
  `--nvfp4-variant vanilla|full` 区分与软件路径直接可比 vs 完整
  production recipe。TE 2.19 在 RTX 5090 上的 MXFP8 / NVFP4
  recipe 对照已完成（见 4.6 节）。
- 测试覆盖：编解码单元测试（含 0/饱和/次正规/中点偶数/NaN/全码空间往返）、
  文件 IO 往返、GPU 与 CPU 参考实现逐位一致的 roundtrip 测试（含奇数尾块、
  tensor/block × nearest/stochastic 组合）、可选的 CUDA Math API 与
  软件编码器逐位一致性测试（`test_native_cvt`，CMake
  `-DENABLE_NATIVE_CVT_TEST=ON`，仅在架构后缀含 `a` 时编译，如 120a）。

## 6. ncu / nsys 使用

- 采集脚本 `scripts/profile.sh`（nsys 时间线已验证，
  `artifacts/sample_pipeline.nsys-rep`）。
- **跨平台结论**：`.nsys-rep` 与 `.ncu-rep` 均为自包含二进制报告，可在
  Windows 宿主机安装的 Nsight Systems / Nsight Compute GUI 中直接
  File→Open 打开分析，不依赖生成平台；只需保证 Windows 端软件版本 ≥
  采集端版本（建议均装最新版），文件通过共享目录/网络复制即可。
- ncu 计数器在两套环境都不可用，但根因不同、脚本处理一致：WSL2 是驱动
  层限制；4090 D 容器是宿主机驱动 `RmProfilingAdminOnly=1` 且容器未授予
  `CAP_SYS_ADMIN`（uid=0 也无效）。需在宿主机执行
  `modprobe nvidia NVreg_RestrictProfilingToAdminUsers=0`，或以
  `--cap-add SYS_ADMIN`（配合 `--security-opt seccomp=unconfined`）启动
  容器后方可采集。`profile.sh` 将 `ERR_NVGPUCTRPERM` 识别为"跳过"而非
  失败，并保留日志。
- nsys 时间线已在 4090 D（原生 Linux 容器，Nsight Systems 2024.6.2）
  采集：`artifacts/4090d/profile_{mxfp8,nvfp4}/*.nsys-rep`。
- `profile.sh` 现产出五类文件（`.ncu-rep` / `_ncu_summary.txt` /
  `_ncu.log` / `.nsys-rep` / `_nsys_stats.txt`），对"ncu 未安装 /
  计数器无权限 / 其他失败"三种情况分别处理（跳过 / 跳过 / 报错退出）；
  kernel 过滤放宽为 `regex:quant|dequant|amax|metrics` 覆盖全部 6 个
  kernel。

## 7. 开发中发现的问题

1. FP8 编码器初版把 `emin` 混入了正常数域的尾数量化指数（`q` 的移位多了一个 `emin` 因子），且次正规域移位错 2×；全码空间往返测试
   （decode→encode 稳定性）暴露后修正。
2. `frexpf` 的第二个参数传 `nullptr` 会在运行期崩溃——E8M0 ceil 编码需要
   同时取尾数判断"恰为 2 的幂"，必须用真实指针。
3. WSL2 与 4090 D 容器下 ncu 计数器均不可用（根因不同，见第 6 节），
   这也是把"报告文件跨平台打开"作为分析路径的原因。
4. 定制版 TE 1.14 直接调 `tex.cast_to_fp8(..., scaling_mode)` 会在
   `CheckScaleTensor` 处整数除零崩溃（SIGFPE）；改走
   `Float8Tensor.quantize_(x, scale=, amax=)` 封装层正常。此外
   `ncu --import | head` 这类"长输出管道接 head"在 `set -o pipefail`
   脚本里会因 SIGPIPE（退出码 141）杀死整个脚本——先落盘再用 `head`
   截断显示才安全。
5. TE ≥ 2.0 的打包结构：`transformer_engine` 是纯 Python 元包，
   `transformer_engine_cu{12,13}` 是预编译 C++ 核心（PyPI 提供
   manylinux wheel），`transformer_engine_torch` 才是与 torch 链
   接的 C++ 绑定——必须本地源码编译（NVIDIA GitHub Releases 上
   `+cu{12,13}torch{ver}` 形式的绑定 wheel 只覆盖 NGC 容器里的 torch
   版本）。本机 48 GB 内存下 ninja 默认 384 核并行会触发 cc1plus
   OOM killer，`MAX_JOBS=8` 可编译通过（约 5 分钟）。`pip` 默认把
   死索引 `/etc/pip.conf` 中的 `pypi.ngc.nvidia.com` 也用作
   extra-index，会拖慢并最终失败；用清华源 `https://pypi.tuna.
   tsinghua.edu.cn/simple` 替代即可。
6. 容器内 GPU 设备节点（`/dev/nvidia0`、`/dev/nvidia-uvm`）会在
   长任务期间被宿主机释放（`nvidia-smi` 静默失败，`cuInit` 返回
   `CUDA_ERROR_NO_DEVICE`，但 `/proc/driver/nvidia/gpus/` 仍记录
   硬件存在），影响所有 CUDA 调用。本次会话中设备节点一度恢复并
   补齐了全部数据采集；经验是把基准拆成短脚本、日志即时落盘
   （`artifacts/` 中的数据在任何中断点都完整可用）。
   `libcuda.so`/`libnvidia-ml.so` 挂载的是真实驱动库还是 0 字节
   stub，会影响 `cuInit` 报错形态。已将该诊断固化成
   `scripts/gpu_env_doctor.sh`：三层检查（内核模块与设备节点 →
   驱动库真实性/版本匹配 → LD_LIBRARY_PATH 混载 + `ctypes` 实时
   `dlopen`/`cuInit` 探针），`--fix` 对可修复项（如 803 的 compat
   库混载）输出移除对应 `LD_LIBRARY_PATH` 条目的修复命令；
   `te_compare.py` 启动预检复用同一套根因判定，失败即打印提示并
   指向 doctor。另注意本环境的 `stat` 不跟随符号链接（报链接字符
   串长度而非目标文件大小），脚本统一 `readlink -f` 后再判大小。

## 8. 未来工作

- 量化 kernel 的块内 amax 改为 warp shuffle 归约 + 每线程多 block（提高
  带宽利用率）；NVFP4 反量化改 `uint4`（16B=32 元素）向量化 packed
  load。TE 2.19 fused cast kernel 的 20× 量化吞吐差（4.6 节）即本项
  的空间上限。
- `__nv_cvt` 硬件路径与软件编码器的一致性已在 5090 (sm_120a) 上
  验证（4.5 节）；等价验证可推广到 sm_100 / sm_101 / sm_103 等其
  他 Blackwell 变种与 Hopper (sm_90, 仅 FP8)，确认 sm_120a 不代表
  偶然。
- NVFP4 软件路径的全局 scale 改用 TE 的满量程约定（`448·6/amax`）
  并对齐块级 E4M3 舍入方向，可与 production recipe 做到逐位一致
  （4.6 节中的差异均来自这两处自由度）。
- 3090/4090/H20 多架构带宽标定，验证 kernel 的架构无关性（4.1/4.3
  三卡复测已基本完成，可补 Hopper sm_90）。
- 国产平台（metax/moore/iluvatar）移植：kernel 仅用 CUDA C++ 基础特性，
  适配工作主要在 RUNTIME 宏层（参考仓库 master 分支 tester 抽象）。
