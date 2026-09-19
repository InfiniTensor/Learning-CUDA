# 完整接口与开发记录（归档说明）

本页来自固定档案 3676727 的接口与历史验证说明，所有命令从项目根目录执行。当前精简提交和归档边界见 [README](../README.md) 与 [证据索引](../EVIDENCE.md)。

在输入最后一维执行 Sylvester Hadamard 变换，提供普通 CUDA 基线、warp 寄存器实现、Tensor Core 对照实现，以及变换与 INT4 量化融合。支持 FP16/BF16 和四维输入形状；验证与性能测量命令在同一可执行程序中。

本目录的命令行程序只需要 C++17 和 CUDA Toolkit。另提供 PyTorch 前向接口，用于接收真实张量，并与第三方 CUDA 库在同一张 GPU 上比较。

实施与后续优化的固定输入、计时和验收标准见 [实施与优化方案](../reports/implementation-optimization-plan.md)。

最新采用的N32–256对齐读写见[最终验收报告](../reports/wide-pairs-io-20260918.md)，以下旧平台计数只适用于原归档版本。

## 原始归档时的状态

原始归档源码为 `24dfef776ad73a4128cb6138674c5886c21e49c0`。N=1/2/4/8/16支持显式packed/auto，默认仍original；新增A100-SXM4-40GB经过独立筛选和留出验证的9条自动规则，已有A800/4090规则不变。详见[A100规则与调用方式报告](../reports/a100-calling-validation.md)。

最终源码在A100与4090分别通过同一1876项CLI矩阵的七种模式、原1800项Dao对照、280项大行数/偏移/stream条件及接口/CSV检查。A100另外完成160条原生安全夹具、memcheck和synccheck零错误；两卡nsys执行路径核验完成。全部原始数据、失败与哈希见[本轮档案](https://github.com/a962695448-rgb/Learning-CUDA/blob/3676727fc21ea27bda668d743d10e52d7e138f68/03_hadamard_tc/a962695448-rgb/results/nvidia_a100_calling_20260908/README.md)。

最终Graph中，A100的64个启用配置、4090的62个启用配置三轮均减少至少5%耗时，相对同轮更快的原128/256线程路径，范围分别为5.24%～68.56%、5.51%～66.74%。两卡各24个回退控制的波动范围在约±0.31%内。这些是已测配置的均摊Graph结果，不是应用整体收益。

普通Python调用可使用已有位置参数写法：相对同一auto的关键字写法，12配置三轮的耗时减少范围为A100 6.22%～9.49%、4090 5.95%～12.31%。相对原默认调用仍有最高约3.45%的额外开销，默认original保留。旧A800/4090负例、A100首次PATH配置失败和时间线检查脚本的范围误判均独立保留。

此前N256显式 `fused_layout="contiguous256"` 的[4090记录](../reports/fused-layout-validation.md)和[A100记录](../reports/fused-layout-a100-validation.md)保留各自版本与负例。PR与正式提交待项目所有者验收。

## 已完成的基线与平台验证

- 2026-09-05 已在租赁的 **NVIDIA GeForce RTX 4090 24 GB（sm89）** 上编译命令行程序和 PyTorch 扩展；原有 RTX 4060 Laptop / WSL2 记录也保留。
- 4090 的 **1,876 组自测通过**，覆盖 FP16/BF16、连续随机、正态和异常值输入，最大绝对误差为 `0.0078125`。35 个 warp 输出元素与已舍入的稠密矩阵参考存在容差内差异，原始日志完整保留。
- CPU 对实际变换输出的量化、GPU 分步量化、GPU 融合量化的 packed bytes 与 scales 全量一致；15 项非法命令行参数检查通过。
- 4090 的 16 组形状/精度/scale 配置已测量，共 110 行性能数据，包含 CPU、GPU 含复制时间及各 CUDA 路径。
- 与固定版本 `fast_hadamard_transform` 的 **1,800 组真实 GPU 对照通过**，FP16 和 BF16 的最大绝对差均为 0；10 项非法张量输入检查、非默认 CUDA stream 检查通过。单卡环境没有验证多 GPU 切换。
- 第三方比较另完成 12 组 CUDA Graph 性能测量，且每个捕获输出均与 eager 输出一致。本轮大批量 6 组的 `Dao 时间 / 本项目时间` 为 `1.200～1.926`；`[17,256]` 两种精度下本项目稍慢，负例完整保留。
- Nsight Systems 已成功导出时间线和 kernel 统计。Nsight Compute 因 `ERR_NVGPUCTRPERM` 无法读取 GPU 性能计数器，失败日志保留，没有虚构带宽利用率或占用率结论。
- NVIDIA 新增显式 `block_threads=128/256`（CLI 为 `--block-threads`），**默认仍为128**。两种选择均通过同一1,876组CLI矩阵；原1,800组Dao矩阵逐输入确认旧默认、显式128、显式256位一致，测试重复不累计为新用例。
- 256线程的原24个候选与48个邻近配置，在64份独立输出的CUDA Graph条件下三轮均减少耗时至少5%，实测范围为6.53%～25.58%；仅覆盖下文明确的N=16/64范围，不自动派发、不保证其他形状更快。
- A100-SXM4-40GB 已重新按 sm80 编译并完成同一1,876组CLI矩阵、1,800组固定Dao对照及线程兼容检查；预定72配置三轮也已完成。CUDA 与九齿的 A100 结果分别记录，当前尚无上游合并结果。
- 天数 MR-V100 已完成独立 COREX 原生适配：三种实现通过 1,504 组变换、180 项接口及 14 项命令行检查，三轮性能原始样本完整保存。源码和原始结果见下方平台入口，不与 NVIDIA 的测试数量相加。
- 沐曦 C500 的 25% sGPU / 16000 MiB 配额也完成独立 MACA 原生验证：相同 1,504 组矩阵、180 项接口和 14 项命令行检查通过，三轮共 4,050 条计时样本。重复同一矩阵的跨平台执行不累计为更多独立算法用例。
- 壁仞 106M 完成原生 SUPA 适配，显式处理旧 SDK 的 BF16 截断与项目最近偶数舍入约定的差异。完整矩阵、量化一致性、三轮基准及小批量留出验证通过；同时保留 Warp32 比共享实现更慢的形状和未推广的实验。
- 昇腾 910B1 完成原生 Ascend C/CANN 9.0 适配：主矩阵1,496组、多核专项128组和有限大行数专项2组通过，分步/融合INT4与CPU参考精确一致。可选矢量缩放的三轮交错对照完整保留8,400条观测；不使用会改变量化中点结果的矢量除法。

下文初轮命令行结果对应实现基线 `6f8e15a2db63a1816c2da6632848a1945380cf21`；初轮 eager 比较使用修复本地安装来源识别后的脚本 `766283ba7352250d5def06fbe62428c74e546917`；CUDA Graph 与 profiler 记录对应 `0b29fcf9031193f49319b2d4132df4d1ef6a4a74`。原始文件及 SHA-256 清单见[实测证据](#实测证据)。

## 平台入口

- 摩尔线程 MTT S4000 / MUSA 5.1.0：[原生 API 与优化报告](../platforms/moore/REPORT.zh-CN.md)。精确 INT4 整数量化通过 1,504 组完整输入、192 项 API 与 14 项 CLI 检查；三轮 6,750 条配对计时中，默认融合相对上一版加速 2.012～2.898×。17,527,208 项 CPU 与 2,074,568 项重叠 GPU 量化检查通过。本轮 77 份文本证据及测试夹具已本地校验。

- NVIDIA A100：[跨卡验收报告](../reports/a100-validation.md)与[完整档案](https://github.com/a962695448-rgb/Learning-CUDA/blob/3676727fc21ea27bda668d743d10e52d7e138f68/03_hadamard_tc/a962695448-rgb/results/nvidia_a100_20260906/README.md)。同轮128/256线程配对的72配置，每轮均减少至少5%耗时，范围6.19%～26.09%；仅限预定N=16/64形状。默认128的原12组Graph仍保留两个较慢的`[17,256]`配置，不把eager比值当作纯kernel收益。

| 平台 | 交付入口与实测边界 |
|---|---|
| NVIDIA RTX 4090 / 4060 | 本页的 CUDA 命令行、PyTorch 前向扩展与参考库对照；下方 `sm80+` 构建要求针对这条路径 |
| 天数 Iluvatar MR-V100 / COREX 4.4.0 | [原生 C++ API、Warp64 复现与完整报告](../platforms/iluvatar/README.md)，[固定源码与原始结果清单](https://github.com/a962695448-rgb/Learning-CUDA/blob/3676727fc21ea27bda668d743d10e52d7e138f68/03_hadamard_tc/a962695448-rgb/results/iluvatar/warp64_a387db3/manifest.json)；独立构建，不使用 NVIDIA `sm_89` 目标 |
| 沐曦 MetaX C500 / MACA 3.0 / sGPU 25% | [原生 C++ API、配额与复现报告](../platforms/metax/README.md)，[固定源码与原始结果清单](https://github.com/a962695448-rgb/Learning-CUDA/blob/3676727fc21ea27bda668d743d10e52d7e138f68/03_hadamard_tc/a962695448-rgb/results/metax/fe44aa3/manifest.json)；cu-bridge 使用 CUDA 命名生成 MACA 设备代码，实际为分区算力 |
| 壁仞 Biren106M / SUPA 1.10 | [原生 API、BF16 舍入、Warp32 及毫秒性能表](../platforms/biren/README.md)，[完整矩阵与原始基准](https://github.com/a962695448-rgb/Learning-CUDA/blob/3676727fc21ea27bda668d743d10e52d7e138f68/03_hadamard_tc/a962695448-rgb/results/biren/2cbaf41/manifest.json)，[交错留出实验](https://github.com/a962695448-rgb/Learning-CUDA/blob/3676727fc21ea27bda668d743d10e52d7e138f68/03_hadamard_tc/a962695448-rgb/results/biren/holdout_8f75553/manifest.json)；使用 BRCC 和原生 `su*` 接口，不依赖 CUDA runtime |
| 昇腾 Ascend910B1 / CANN 9.0 | [原生异步 API、量化中点问题与完整复现](../platforms/ascend/README.md)，[原始探针、完整验证与交错A/B](https://github.com/a962695448-rgb/Learning-CUDA/tree/3676727fc21ea27bda668d743d10e52d7e138f68/03_hadamard_tc/a962695448-rgb/results/ascend)；使用真实 NPU、ACL 与 Ascend C，16位存储接口独立于CUDA类型 |

天数 Warp64 在实测 N=64/128/256、五档行数、两种精度的三轮中，变换相对本平台基础实现为 1.1907～3.3867 倍，融合路径为 1.5245～7.3599 倍。它们是同设备、同输入语义的事件区间比较；不与 NVIDIA 的 Graph 或端到端时间相除，也不宣称所有形状或其他国产设备具备同样收益。平台报告保留共享内存候选的退化样例和初始化故障修复记录。

沐曦在自己的相同规模矩阵中，Warp64 变换相对本平台基础实现为 1.0697～2.5394 倍，融合路径为 1.2900～6.0572 倍；Warp64 内部融合相比分步降低耗时 21.40%～41.46%。所有分母均来自同轮沐曦实测，不比较不同芯片或不同计算配额的绝对快慢；报告保留跨轮波动及共享内存候选的退化。

壁仞保留共享实现作为默认；小批量单 warp 发射在已测 rows=32/63/64、N=64、两种精度的融合路径中，相对共享实现每轮减少耗时 10.09%～12.68%。这不是全范围保证：N=256 和较大行数仍有显著负例，量化线程均衡实验也只作为可选消融。具体方法选择、原始失败、阈值外控制和计时边界均在该平台报告中说明。

昇腾的`--vector-scale`仅改变VectorGather的缩放计算，量化保持NPU标量除法；默认关闭。120个向量路径/形状/精度组合中97个在三轮每轮均减少耗时至少5%，其余含未达门槛与混合方向的结果。对已测M=257/4096/16384、N=64/128/256，两种精度的变换耗时减少40.89%～79.02%，融合减少21.97%～29.33%。这些是同block_dim=32的OFF/ON构建比较，不能解释为相对CPU或NVIDIA的加速；小输入与Vector对Scalar的负例均保留。

## 数学与接口约定

输入形状为 `[batch, seq, heads, dim]`，按行连续存储，前三维展平成 `rows=batch*seq*heads`。当前支持 `dim ∈ {1,2,4,8,16,32,64,128,256}`。

变换定义为：`y[row,j] = scale * sum_i H[j,i] * x[row,i]`，其中 `H[j,i] = (-1)^popcount(j & i)`。

- 默认 `scale=1`，为未归一化变换，和参考库默认 scale 对齐。
- `--normalize` 或 `--scale normalized` 使用 `scale=1/sqrt(dim)`。
- 输入和输出均为所选 FP16/BF16，内部加减与 Tensor Core 累加使用 FP32。
- 当前输入由确定性生成器按形状和精度构造；自测包含边界形状、全零、脉冲、交替符号、随机、正态和异常值模式。该命令行程序不解析外部二进制张量文件。
- 当前构建要求 sm80+，以同时验证 BF16 Tensor Core；RTX 4060/4090 为 sm89，A100 为 sm80。

## 量化约定

首版明确采用 per-row 对称 INT4，量化整数范围 `[-7,7]`：

1. Hadamard 结果先舍入为输出 FP16/BF16。
2. 每行 `s=max(abs(y))/7`；全零行固定 `s=1`。
3. `q=clamp(round_to_nearest_even(y/s),-7,7)`。
4. 每个字节放两个 4-bit 补码数：偶数元素在低半字节，奇数元素在高半字节。N=1 时高半字节为零。

融合路径刻意保留第 1 步的舍入，使其与“先变换写回，再量化”采用相同数值语义；不能为了减少一次舍入而破坏融合一致性。每行 scale 单独输出，packed 存储确实每字节两个数。

FP8、随机符号旋转和更长维度属于当前未实现的扩展，不计入已完成能力。

## 实现路径

- `naive_global`：先转 FP32，每个蝶形阶段启动一个 kernel，最后舍入写回。用于可读基线。
- `warp`：每个 warp 处理一行；短距离蝶形使用 shuffle，跨 32 个元素的蝶形在每个线程的寄存器数组中完成。
- `tensor_core`：WMMA 执行稠密 Hadamard 矩阵乘，作为不同算法的 Tensor Core 对照；只用于 dim>=16。
- `split_int4`：warp 变换后再调用独立量化 kernel。
- `fused_int4`：一条 warp kernel 完成变换、输出精度舍入、行最大值归约、量化及打包。

当前 Tensor Core 方法做了比 FWHT 更多的运算，而且有共享内存中转。在 4090 的大批量、dim=256 配置中明显慢于 warp；小配置的差距较小。该结果只适用于当前 WMMA 实现，不能外推到所有 Tensor Core Hadamard 算法。

## 构建和运行

Linux 或 WSL2，已配置 NVIDIA 驱动和 CUDA Toolkit：

```bash
# 在本目录执行。4090 服务器使用 CUDA 12.8。
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
make CUDA_HOME="$CUDA_HOME" ARCH=89
make cpu-test
./build/hadamard --self-test

# 单一配置。日志文件记录实际硬件与测量范围。
./build/hadamard --benchmark --batch 4 --seq 128 --heads 8 \
  --dim 256 --dtype fp16 --csv results/my-run.csv

# 完整自测、非法参数检查和 16 组基准；label 不得和已有结果混用。
python3 scripts/run_validation.py --label rtx4090-new-run --benchmark
```

A100 上重编：

```bash
make CUDA_HOME=/usr/local/cuda ARCH=80
./build/hadamard --self-test
python3 scripts/run_validation.py --label a100-first-run --benchmark
```

Makefile 会记录 NVCC 路径和编译参数；修改 ARCH 后自动重编，避免把 sm89 二进制直接当成 A100 构建。

## PyTorch 张量接口

实测环境为 Python `3.12.3`、PyTorch `2.6.0a0+ecf3bae40a.nv25.01`、PyTorch CUDA `12.8`。服务器已有可用 PyTorch 时，使用它构建扩展；无需为了本项目替换系统 PyTorch。

```bash
# 在本目录、已启用 CUDA PyTorch 的 Python 环境内执行。
export MAX_JOBS=1
export NVCC_THREADS=1
export TORCH_CUDA_ARCH_LIST=8.9
python scripts/build_torch_extension.py --verbose
```

第一次调用会编译，后续复用本目录 `build/torch_extension/` 的产物。以下例子可以直接保存为 Python 文件运行：

```python
import torch
from scripts.build_torch_extension import load_extension

op = load_extension()
x = torch.randn((2, 16), device="cuda", dtype=torch.float16)
y = op.hadamard(x, 1.0)                    # 只做变换，保持 x 的形状和 dtype
packed, row_scales = op.hadamard_int4(x, 1.0)  # 融合变换和 INT4 量化
split_packed, split_scales = op.quantize_int4(y)
assert torch.equal(packed, split_packed)
assert torch.equal(row_scales, split_scales)
```

输入必须是连续、非空的 CUDA FP16/BF16 张量，形状为 `[rows, dim]` 或 `[batch, seq, heads, dim]`，最后一维取前述 1～256 的二次幂。接口是前向计算，不支持 `requires_grad=True`。`packed` 的 dtype 为 `uint8`，最后一维为 `ceil(dim/2)`；`row_scales` 为 FP32，形状是输入去掉最后一维。接口使用调用者当前 CUDA stream。

带有 PyTorch 惰性负号标记的视图会明确拒绝，请先调用 `x.resolve_neg()` 将逻辑值物化，再传入接口。普通负数张量保持支持。

## N256 显式融合布局

```python
# x 的最后一维必须为256；旧三参数调用保持兼容。
packed_original, scales_original = op.hadamard_int4(x, 1.0, 128)
packed, scales = op.hadamard_int4(x, 1.0, 128, fused_layout="contiguous256")
```

```bash
./build/hadamard --benchmark --batch 1 --seq 17 --heads 1 --dim 256 \
  --dtype fp16 --block-threads 128 --fused-layout contiguous256 --csv results/new-layout.csv
./build/hadamard --self-test --fused-layout contiguous256
```

该选项只改变融合路径；独立变换和独立量化继续使用原 kernel。自测仍执行全矩阵，仅 N256 的融合项切换布局。CSV 增加第 21 列 `fused_layout`，仅融合行填写实际选项；旧表头拒绝混写，复现时使用新文件名。

## 小维度显式行布局

```python
# x 的最后一维为1/2/4/8/16；旧调用默认仍是original。
y = op.hadamard(x, 1.0, 128, "auto")
packed, row_scales = op.hadamard_int4(x, 1.0, 128, "original", "auto")
y_packed = op.hadamard(x, 1.0, block_threads=256, row_layout="packed")
```

```bash
./build/hadamard --self-test --row-layout auto
./build/hadamard --benchmark --batch 1 --seq 4096 --heads 1 \
  --dim 8 --dtype fp16 --row-layout auto --csv results/rows-auto-new.csv
```

`packed`明确要求N≤16；auto可在已有规则中选择256线程，否则回退到调用者指定的128/256。自测的packed模式只切换N≤16，其他维度仍执行原实现。非original行布局不能与contiguous256组合。A100-SXM4-40GB也已完成独立规则验证。关键字写法仍支持，位置参数可减少所测调用开销；普通调用相对原默认仍有少量成本。自动阈值、未采用的N16融合和有限覆盖范围见[最新报告](../reports/a100-calling-validation.md)。

CSV默认original仍为21列；packed/auto采用24列，在末尾增加`requested_row_layout`、`resolved_row_layout`、`quantize_block_threads`。`warp_block_threads`记录实际选择的变换/融合线程，split的独立量化保留调用者线程并单列记录。不同表头拒绝混写，使用新的输出文件。

## 显式选择 NVIDIA 线程数

默认row_layout="original"时，省略线程参数仍使用128；显式auto的选择规则另见上一节。以下为早期128/256线程实验： 本次接口源码可定位至[`24849f6`](https://github.com/a962695448-rgb/Learning-CUDA/commit/24849f61ef06350f4e8bcd224ef93d97622c9744)；实机原字节与提交后的LF内容核查分别保存在下方归档。正确性已覆盖原全部dim=1～256的二次幂，性能证据仅覆盖N=16/64：变换M=4096/16384及各自M±1，融合INT4为M=4096及M±1。更一般的 M 范围和其他 GPU 仍需分别验证；已测 A100 的范围与结果见 [A100 报告](../reports/a100-validation.md)。

```python
# 保留原调用；x仍须满足前述CUDA/形状/连续存储约束。
y_default = op.hadamard(x)                  # 默认128
# scale仍为原来的第二个参数；block_threads追加为可选参数。
y_256 = op.hadamard(x, 1.0, block_threads=256)
packed, row_scales = op.hadamard_int4(x, 1.0, block_threads=256)
split_packed, split_scales = op.quantize_int4(y_256, block_threads=256)
```

```bash
# 同一1,876组完整矩阵，明确选择256线程；默认命令仍使用128。
./build/hadamard --self-test --block-threads 256
./build/hadamard --benchmark --batch 4 --seq 128 --heads 8 \
  --dim 64 --dtype fp16 --block-threads 256 --csv results/threads256-new-run.csv

# 独立构建缓存；复用原1,800组Dao矩阵比较默认/128/256，输出文件须为新文件。
python scripts/verify_block_threads.py \
  --reference-repo /data/infinitensor-2026/fast-hadamard-transform \
  --build-directory /tmp/hadamard-thread-check-new-run \
  --json results/thread-api-check-new-run.json
```

CLI选项只影响warp变换、独立量化、split/fused和warp含复制路径；naive、Tensor Core和CPU路径不变。不支持的线程值会明确拒绝。输入检查、设备guard、当前CUDA stream和原来的输出精度/INT4舍入语义均保留。

计时日志同时显示us和ms。初次线程接口扩展时，CSV在原18列后追加`warp_block_threads`和`mean_ms`；当前original还含第21列`fused_layout`，packed/auto的24列另见上节。`mean_ms`严格按`mean_us/1000`换算；不涉及warp的行将线程字段留空。已有旧表头的CSV会被拒绝追加，请使用新的结果文件，避免不同格式混写。

[256线程独立Graph复核与原始源码](https://github.com/a962695448-rgb/Learning-CUDA/blob/3676727fc21ea27bda668d743d10e52d7e138f68/03_hadamard_tc/a962695448-rgb/results/nvidia_thread_promotion_20260905/README.md)保存三轮全部72配置及原始采样；[生产接口集成验证](https://github.com/a962695448-rgb/Learning-CUDA/blob/3676727fc21ea27bda668d743d10e52d7e138f68/03_hadamard_tc/a962695448-rgb/results/nvidia_api_integration_20260906/README.md)保存28个原始文件及对应源码SHA。集成后只复测了六个既有代表配置，256线程相对128减少耗时7.12%～25.34%，没有扩大性能搜索。Graph均摊时间仍不等于独立单kernel延迟或端到端时间；未知驻留CUDA上下文及单卡限制均在归档中说明。

## 固定第三方版本并复现比较

参考库固定为 Dao-AILab 的提交 `e7706faf8d1c3b9f241e36860640ad1dac644ede`。下面使用实测服务器的源码目录；已有干净的固定版本目录时，直接复用，不重复 clone。

```bash
REFERENCE_ROOT=/data/infinitensor-2026/fast-hadamard-transform
git clone https://github.com/Dao-AILab/fast-hadamard-transform.git "$REFERENCE_ROOT"
git -C "$REFERENCE_ROOT" checkout --detach e7706faf8d1c3b9f241e36860640ad1dac644ede

# 使用同一个 Python/PyTorch 环境，保留其 C++ ABI 设置。
export MAX_JOBS=1
export NVCC_THREADS=1
export FAST_HADAMARD_TRANSFORM_FORCE_BUILD=TRUE
export FAST_HADAMARD_TRANSFORM_SKIP_CUDA_BUILD=FALSE
python -m pip install --no-deps --no-build-isolation --no-cache-dir "$REFERENCE_ROOT"

# 回到本项目目录；每轮结果使用新的文件名。
python scripts/compare_reference.py --reference-repo "$REFERENCE_ROOT" \
  --benchmark --json results/third_party_rtx4090-new-run.json
```

参考库有自己的多架构编译参数；`TORCH_CUDA_ARCH_LIST=8.9` 只保证本项目扩展的目标，不会覆盖参考库写死的 `-gencode`。`MAX_JOBS=1` 与 `NVCC_THREADS=1` 限制编译并发，适合本次 20 GB 内存服务器。不要人为改成与已安装 PyTorch 不一致的 C++ ABI。

比较脚本检查固定源码提交、受跟踪文件是否干净、实际安装来源及 `.so` 哈希。两边使用完全相同的输入、dtype 和 FP32 `scale`：默认均为 1，归一化时均为 `1/sqrt(dim)`。本次 1,800 组对照包含两种精度、两种 scale、多种形状、4 种输入模式和 3 个随机种子，同时检查融合与分步 INT4 结果完全一致。

`third_party_rtx4090_eager.json` 保留初轮 eager 测量。增加 CUDA Graph 后重新验证了同一组 1,800 个正确性用例，再将 eager 和 Graph 两种口径保存为新的 `third_party_rtx4090_graph.json`；这不是新增 1,800 个独立用例。Graph 数据不会覆盖或混入初轮记录。

## 如何判定正确

- 独立 CPU 参考采用 FP64 稠密矩阵求和，没有复用 GPU 的蝶形实现；再按公开输出 dtype 舍入。
- FP16 最大绝对误差严格 `<1e-2`，BF16 严格 `<5e-2`。
- 小用例对所有元素作稠密比较；大批量选 32 行作独立稠密比较，但分步/融合量化仍覆盖全部元素。
- 量化独立 oracle 对实际 warp 输出进行 CPU 量化，并与两个 GPU 路径逐字节、逐 scale 比较。
- 若变换仅在允许的误差内接近稠密参考，不会把恰在量化阈值附近的离散差异误称为融合错误；此类差异单独记录。只有 warp 变换与稠密参考完全相同时，才强制要求稠密量化也精确相等。
- CPU 专用测试还覆盖 Hadamard 二次变换还原、最近偶数舍入、补码打包及非法长度。

## 性能测量口径

- `kernel_only`：CUDA events，先预热、再多次执行取均值；排除分配、输入复制和 Hadamard 矩阵准备。
- `cpu_compute`：优化编译的单线程 FP32 FWHT，使用主机时钟，不包含重置输入的复制。
- `host_e2e`：已有缓冲区上的 pageable H2D + warp 变换 + D2H，用主机时钟计时。
- 吞吐量为输入元素数/秒，不等于 FLOP/s，也不假称物理内存带宽。
- 命令行扫描是单轮重复均值：每条 CUDA 路径 300 次，CPU 与含复制路径各 20 次。它不是跨次运行的置信区间。
- 第三方 eager 比较在 CUDA event 区间内反复调用会分配输出的 PyTorch API，每组 200 次、共 5 组，取 5 个组均值的中位数。它排除了 H2D/D2H、编译和校验，但可能包含 Python 调用与 kernel 发射之间的 GPU 空闲时间，不能与命令行 `kernel_only` 列直接相除。
- CUDA Graph 将每条路径的 64 次调用捕获到各自的图中，保留全部输出；每组重放 20 次，共 5 组，并交替两条路径的测量顺序。图重放使用固定输入与输出地址，计时除以每组 1,280 次调用，再取组间中位数。它去除了逐次 Python 发射和分配，但仍含捕获的 GPU 工作与均摊的图调度开销，**不是单个 kernel 的独立延迟**。
- 旧 RTX 4060 Laptop 日志受 WSL/WDDM 调度影响，与本次 Linux 4090 结果分开保存。

4090 代表结果如下，单位均为 **微秒（us）**，`scale=1`。CPU 与 GPU 含复制两列用于观察把主机数据送入 GPU 是否划算；其余列用于比较 GPU 内部实现。

| 输入形状 | dtype | CPU FWHT | GPU 含复制 | warp | WMMA | 分步 INT4 | 融合 INT4 |
|---|---|---:|---:|---:|---:|---:|---:|
| `[4,128,8,256]` | FP16 | 2246.584 | 401.763 | 3.157 | 34.434 | 6.246 | 3.956 |
| `[4,128,8,256]` | BF16 | 2222.205 | 432.524 | 3.164 | 33.888 | 6.274 | 3.959 |
| `[1,1,17,256]` | FP16 | 9.244 | 11.065 | 2.068 | 7.772 | 3.993 | 2.270 |

在 FP16 `[4,128,8,256]` 中，普通多次 kernel 基线 `naive_global` 为 `39.059 us`，warp 为 `3.157 us`；融合 INT4 相比本项目分步 INT4 约快 `1.58 倍`。当输入原本在 CPU 上时，含复制的 GPU 总时间约为 CPU FWHT 的 `1/5.59`，远小于仅看 GPU kernel 能算出的表面加速比。

小规模并不保证端到端收益：`[1,1,17,256]` FP16 的 CPU 为 `9.244 us`，GPU 含复制为 `11.065 us`。这组负例保留在同一 CSV 中；大规模收益不能外推到所有形状。

初轮第三方 eager PyTorch API 的 12 组 `Dao 时间 / 本项目时间` 为 `2.151～2.747`。例如 FP16 `[4,128,8,256]` 为本项目 `7.357 us`、Dao `15.846 us`。这说明当前接口及发射方式在该测量区间内更快，**不能证明纯 CUDA kernel 快了 2 倍**。

以下列出第二轮完整的 12 组 CUDA Graph 结果，单位为每次调用均摊的 **us**，`scale=1`。比值大于 1 表示本项目用时较少，小于 1 表示 Dao 用时较少。

| 输入形状 | dtype | 本项目 | Dao | Dao / 本项目 |
|---|---|---:|---:|---:|
| `[17,16]` | FP16 | 0.9912 | 1.0072 | 1.0161 |
| `[4,128,8,16]` | FP16 | 1.6976 | 3.2696 | 1.9260 |
| `[17,64]` | FP16 | 1.0152 | 1.0320 | 1.0165 |
| `[4,128,8,64]` | FP16 | 1.7383 | 3.3008 | 1.8989 |
| `[17,256]` | FP16 | 1.1256 | 1.0656 | **0.9467** |
| `[4,128,8,256]` | FP16 | 2.7920 | 3.3664 | 1.2057 |
| `[17,16]` | BF16 | 0.9912 | 1.0096 | 1.0186 |
| `[4,128,8,16]` | BF16 | 1.6984 | 3.2696 | 1.9251 |
| `[17,64]` | BF16 | 1.0144 | 1.0296 | 1.0150 |
| `[4,128,8,64]` | BF16 | 1.7368 | 3.3000 | 1.9000 |
| `[17,256]` | BF16 | 1.1464 | 1.0687 | **0.9322** |
| `[4,128,8,256]` | BF16 | 2.8104 | 3.3712 | 1.1995 |

本轮大批量配置有明确的时间差，小规模 dim=16/64 的结果接近，不能据约 1% 的差距推广出稳定优势。`[17,256]` 的两种精度均是本项目更慢。更严格的 Graph 口径缩小了 eager 比较中的优势，说明前面的两倍左右比值不能简单归因于蝶形计算本身。上述12组仍是单卡、一次运行中的分组重复测量，没有跨设备或跨日期置信区间。

## Profiler 复现与结论边界

下面沿用实测日志的配置，只将输出改成新的文件名前缀，避免覆盖证据。在本项目目录、程序已编译后执行：

```bash
nsys profile --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none \
  --output results/profile_rtx4090-new-run \
  ./build/hadamard --benchmark --batch 4 --seq 128 --heads 8 \
  --dim 256 --dtype fp16 --warmup 1 --repetitions 3
nsys stats --report cuda_gpu_kern_sum,cuda_api_sum --format csv \
  results/profile_rtx4090-new-run.nsys-rep > results/profile_rtx4090-new-run_stats.txt

ncu --set basic --kernel-name 'regex:warp_kernel' --launch-count 1 \
  --export results/profile_rtx4090-new-run_ncu \
  ./build/hadamard --benchmark --batch 4 --seq 128 --heads 8 \
  --dim 256 --dtype fp16 --warmup 1 --repetitions 3
```

Nsight Systems 退出码为 0，原始 `.nsys-rep` 时间线保存在本地。此格式可能包含运行环境信息，因此公开仓库提供统计和日志，原始二进制不随代码发布。该次 **profiler 运行内部**的 kernel 统计如下：

| 实际执行的 kernel | 实例数 | 中位数（us） |
|---|---:|---:|
| WMMA `tensor_core_kernel` | 5 | 34.400 |
| 只做变换的 `warp_kernel<...,true,false>` | 14 | 2.624 |
| 只做量化的 `warp_kernel<...,false,true>` | 5 | 2.496 |
| 融合变换量化的 `warp_kernel<...,true,true>` | 5 | 3.424 |

统计证明上述路径均实际执行，也显示当前 WMMA 路径在该配置下耗时较长。这里聚合了整段 trace 中的同名实例，各路径实例数不同；不能将两个中位数相加当作一次分步调用延迟。这些数据受 profiler 与短测量配置影响，与前面的常规基准分开解读；日志里 profiler 附加后的程序计时也不能替代常规 CSV。

Nsight Compute 退出码为 1，明确报 `ERR_NVGPUCTRPERM`。当前容器没有目标 GPU 的性能计数器权限，需要算力平台开放权限后再测。尚未获得 ncu 的带宽利用率、occupancy 等硬件计数器结果；当前报告不据时间线推断这些指标已经测得。

## 实测证据

以下 4090 文件从服务器结果逐字节保存；文件大小、SHA-256 和对应提交见 [rtx4090_initial_manifest.json](https://github.com/a962695448-rgb/Learning-CUDA/blob/3676727fc21ea27bda668d743d10e52d7e138f68/03_hadamard_tc/a962695448-rgb/results/rtx4090_initial_manifest.json) 与 [rtx4090_graph_profile_manifest.json](https://github.com/a962695448-rgb/Learning-CUDA/blob/3676727fc21ea27bda668d743d10e52d7e138f68/03_hadamard_tc/a962695448-rgb/results/rtx4090_graph_profile_manifest.json)。保留完整数据，包括负例、容差内差异及 profiler 权限失败。

| 文件 | 内容 |
|---|---|
| [validation_rtx4090_initial.log](https://github.com/a962695448-rgb/Learning-CUDA/blob/3676727fc21ea27bda668d743d10e52d7e138f68/03_hadamard_tc/a962695448-rgb/results/validation_rtx4090_initial.log) | 1,876 组自测、15 项非法参数及全部基准命令输出 |
| [benchmark_rtx4090_initial.csv](https://github.com/a962695448-rgb/Learning-CUDA/blob/3676727fc21ea27bda668d743d10e52d7e138f68/03_hadamard_tc/a962695448-rgb/results/benchmark_rtx4090_initial.csv) | 16 组配置、110 行 CPU/GPU 原始均值 |
| [third_party_rtx4090_eager.json](https://github.com/a962695448-rgb/Learning-CUDA/blob/3676727fc21ea27bda668d743d10e52d7e138f68/03_hadamard_tc/a962695448-rgb/results/third_party_rtx4090_eager.json) | 1,800 组第三方对照、环境与二进制来源、12 组 eager 原始采样 |
| [third_party_rtx4090_graph.json](https://github.com/a962695448-rgb/Learning-CUDA/blob/3676727fc21ea27bda668d743d10e52d7e138f68/03_hadamard_tc/a962695448-rgb/results/third_party_rtx4090_graph.json) | 同组正确性复测、第二轮 eager 与 12 组 CUDA Graph 原始区间及采样 |
| [profile_rtx4090_nsys.log](https://github.com/a962695448-rgb/Learning-CUDA/blob/3676727fc21ea27bda668d743d10e52d7e138f68/03_hadamard_tc/a962695448-rgb/results/profile_rtx4090_nsys.log) | nsys 完整命令、退出码及程序输出 |
| [profile_rtx4090_nsys_stats.txt](https://github.com/a962695448-rgb/Learning-CUDA/blob/3676727fc21ea27bda668d743d10e52d7e138f68/03_hadamard_tc/a962695448-rgb/results/profile_rtx4090_nsys_stats.txt) | CUDA kernel 与 API 调用统计 |
| [profile_rtx4090_ncu.log](https://github.com/a962695448-rgb/Learning-CUDA/blob/3676727fc21ea27bda668d743d10e52d7e138f68/03_hadamard_tc/a962695448-rgb/results/profile_rtx4090_ncu.log) | ncu 完整命令与 `ERR_NVGPUCTRPERM` 失败证据 |

早期 4060 证据仍在 `results/validation_sm89_expanded.log`、`results/benchmark_sm89_expanded.csv`、`results/build_sm89_dryrun.log`；初始 580 组 exact-grid 日志为 `results/self_test_sm89.log`。这些历史结果不能标成 4090 或 A100 数据。

## 后续验收

- 本轮A100-SXM4-40GB与4090的新版验证已完成；其他设备型号、更广形状与真实应用仍按新实验分别验证，不能继承本轮有限范围。九齿的A100记录在九齿仓库独立保存。
- 在平台允许硬件计数器采集后补充 ncu；结合已取得的 nsys 时间线，进一步检查访存、寄存器与当前 WMMA 路径，评估分解 Hadamard 的 Tensor Core 算法。
- 提交前核对训练营“包含测试”与通用“无测试代码”的措辞冲突；保留完整开发验证证据。

## 参考

- [本季度官方题目](https://github.com/InfiniTensor/Learning-CUDA/tree/2026-summer-project/03_hadamard_tc)
- [参考库及其 scale 约定](https://github.com/Dao-AILab/fast-hadamard-transform/blob/e7706faf8d1c3b9f241e36860640ad1dac644ede/README.md)

本实现自行编写；参考库作为固定版本的独立 GPU 对照，不将其公开 A100 数据计作本项目实测。
