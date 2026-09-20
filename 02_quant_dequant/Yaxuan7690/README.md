# MXFP8 / NVFP4 / MXFP4 低精度格式软件模拟与反量化（CUDA）

2026 夏季训练营 CUDA 方向项目 · **选题二**：MXFP8 / NVFP4 低精度模拟与反量化（另补齐了 **MXFP4** 与 **4/6 自适应块缩放**，见 `总结报告.md` 2.2 / 2.3.1 / 6.8）。

本程序在**普通 CUDA GPU（不依赖 Hopper / Blackwell / Ampere 以上架构特性，不使用
FP8 / FP4 Tensor Core，也不包含 `cuda_fp8` / `cuda_fp4` 头文件）**上，用纯软件方式
模拟 MXFP8、NVFP4 与 MXFP4 三种低精度格式的编码、打包、缩放、解包、反量化与误差评估。

## 交付内容

本次提交按选题二的「需提交内容」组织：**整个程序（包含测试）** + **总结报告**。

| 交付物 | 位置 |
| --- | --- |
| 总结报告（格式与公式、实现思路、优化历程、结果分析、软件模拟与架构依赖的区分、未来方向） | `总结报告.md` |
| 程序：9 个头文件 + 7 个 `.cu` 实现 | `include/lpq/`、`src/` |
| 测试：3 个可执行测试（合计 314205 条断言）+ 公共测试头 | `tests/` |
| 构建入口 | `Makefile`（四个平台分支）、`.clang-format`、`.gitignore` |
| 辅助工具与实验脚本 | `tools/`、`scripts/` |
| 测量数据与证据（报告全部数字的原始出处） | `data/`、`docs/measurements.md`、`docs/profile/` |
| 过程与证据索引（非交付要求，供评审参考） | `项目状态与交接记录.md` |

「题目要求 → 本实现」的逐条对照表见 `总结报告.md` §1；提交路径与打包命令见
`项目状态与交接记录.md` §7。

## 功能

- 读取 FP32 / FP16 输入矩阵；
- 按配置量化为 **MXFP8（E4M3 / E5M2 + E8M0 共享指数）**、
  **MXFP4（E2M1 + E8M0 共享指数，block 32）** 或
  **NVFP4（E2M1 + E4M3 块缩放 + FP32 全局缩放）**；
- 保存低精度数据与全部缩放因子；
- 用 CUDA kernel 反量化为 **FP16 / BF16 / FP32**；
- 输出误差统计（最大绝对误差、MAE、MSE、RMSE、相对 MSE、带符号偏置）与性能日志
  （各阶段 kernel 耗时、有效带宽、压缩率）；
- 支持 **per-tensor** 与 **block-wise** 两种缩放策略，支持**就近舍入（RNE）**与
  **随机舍入（SR）**；
- NVFP4 另支持 **4/6 自适应块缩放**（5 条 `scale_rule` 规则，见下）；4/6 的两个候选取舍已并进
  量化 kernel，输入遍数与标准 NVFP4 相同。
- 内置**完全独立的 CPU 参考实现**（查表 + 二分 + 串行流水线），可逐位对拍。

## 目录结构

```
include/lpq/          公共头文件
  platform.hpp        平台探测与跨平台差异抽象（换平台唯一需要改的文件）
  common.hpp          位运算辅助、随机数、主机/设备共用工具
  formats.hpp         低精度格式定义与转换（E2M1 / E4M3 / E5M2 / E8M0 / FP16 / BF16）
  config.hpp          量化配置与参数文件解析
  lowp_tensor.hpp     输入张量 / 低精度权重文件读写
  pipeline.hpp        GPU 流水线接口、误差统计接口
  reference.hpp       独立 CPU 参考实现接口
  vec_io.cuh          设备侧向量化访存与元素 traits
src/                  实现（main.cu 为命令行主程序）
tests/                单元测试与端到端测试
tools/                gen_data.py（造数据） / verify_numpy.py（NumPy 独立实现对拍）
                       / verify_e2m1_rne.cu、verify_fp8_hw_{encode,decode}.cu（穷举对拍）
                       / bw_ceiling.cu（逐阶段显存带宽上限标定）
scripts/              build.ps1（Windows 构建） / run_domestic.sh（国产平台一键跑通）
                      / run_experiments.py（全矩阵实验） / benchmark.py（规模扫描）
                      / cpu_compare.py（GPU vs CPU 对比） / perf_probe.py（优化迭代台架）
                      / analyze.py（汇总成 Markdown 表格） / pack_submission.py（整理提交副本）
                      / make_snapshot.py（把整个目录打成归档快照 + md5 校验文件）
data/                 测试数据、参数、日志、汇总 CSV
  domestic_probes/    国产平台排查用的一次性探针（见该目录 README，不参与构建）
docs/                 测量结果与 nsys 剖析记录
```

## 构建

Linux（正式提交使用，见 `Makefile`）：

```bash
make            # 编译主程序与测试，并运行全部测试
make build      # 只编译
make test       # 只运行测试
make app        # 只编译 build/lpq_quant
make PLATFORM=iluvatar   # 切换国产平台（iluvatar / moore / metax）
```

Windows（本地开发，见 `scripts/build.ps1`）：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\build.ps1
```

> 源码统一为 UTF-8 **无 BOM**。在 MSVC 上构建时脚本会自动追加 `-Xcompiler /utf-8`，
> 否则 MSVC 会按本地代码页解释非 ASCII 字节。
>
> **已知坑（仅 Windows + nvcc）**：`-Xcompiler /utf-8` 只作用于主机编译器，管不到 CUDA
> 前端（cudafe++）。`.cu` 里的**中文字符串字面量**会被该前端按本地代码页误解析，随机
> 触发 `missing closing quote`；中文**注释**不受影响。因此新增的
> `tests/test_portability.cu` 刻意只使用 ASCII 输出字符串 —— 它是用来在未知国产编译器
> 上做诊断的，必须自己先能编过。其余源文件保留中文注释与既有的中文字符串。

## 运行

```bash
build/lpq_quant --input <tensor> --params <params> [选项]
```

| 选项 | 说明 |
| --- | --- |
| `--out-prefix <p>` | 输出文件前缀（默认取输入文件名去掉扩展名） |
| `--out-dir <d>` | 输出目录（默认与输入同目录） |
| `--seed <n>` | 随机舍入种子（覆盖参数文件） |
| `--repeat <n>` | 计时重复次数（默认 20） |
| `--warmup <n>` | 预热次数（默认 5） |
| `--device <n>` | 指定 GPU |
| `--cpu-check` | 额外运行 CPU 参考实现并逐位比对（同时给出 GPU/CPU 加速比） |
| `--quiet` | 不向 stdout 打印摘要 |

示例：

```bash
python tools/gen_data.py --out-dir data/tensors --name normal_512 \
       --rows 512 --cols 512 --dist normal --dtype fp32
build/lpq_quant --input data/tensors/normal_512.tensor \
                --params data/params/nvfp4_block_fp16_nearest.params \
                --out-dir data/out --cpu-check
```

### 输入张量文件格式（小端）

```
int64  num_rows
int64  num_cols
uint8  dtype_len
char   dtype[dtype_len]        # "fp32" 或 "fp16"
<data> num_rows*num_cols 个元素，行主序紧排
```

题目里「张量数据文件」的 `[header]` 段（`num_rows: int64`、`num_cols: int64`、
`dtype: string`）对应的就是上面这个紧凑二进制头，`[data]` 段即紧随其后的裸数组。没有采用字面的
文本段标记，是为了避免逐元素解析文本 —— 256 MiB 的输入解析一次就要几百毫秒，会盖过 kernel
本身的耗时。读取端同时接受"数据紧跟 `dtype` 串"与"8 字节对齐"两种偏移，方便用别的工具造数据。

### 量化参数文件格式

```
format       = "mxfp8" | "nvfp4" | "mxfp4"
elem_format  = "e4m3" | "e5m2" | "e2m1"  # mxfp8 用 e4m3/e5m2, mxfp4 固定 e2m1, nvfp4 不写
block_size   = 32                 # mxfp8 / mxfp4 默认 32, nvfp4 默认 16
scale_mode   = "tensor" | "block"
output_type  = "fp16" | "bf16" | "fp32"
rounding     = "nearest" | "stochastic"
scale_rule   = "static_6" | "static_4" | "abs_max" | "mae" | "mse"
                                  # 仅 nvfp4; 默认 static_6 (= 规范公式); 后三条为 4/6 自适应
target_gpu   = "T4"                # 仅用于报告说明
seed         = 20260915            # 可选，随机舍入种子
```

题目「量化参数文件」里定义的那 6 项（`format` / `block_size` / `scale_mode` / `output_type` /
`rounding` / `target_gpu`）就是全部必需项 —— 只写这 6 行即可运行（`mxfp8` 的元素格式默认
`e4m3`、`nvfp4` 的 `scale_rule` 默认 `static_6`），其余都是本实现的可选扩展。

### 输出文件

| 文件 | 内容 |
| --- | --- |
| `<prefix>_lowp.bin` | 128 字节固定头 `LowpHeader` + packed 数据 + scale 数组 |
| `<prefix>_dequant.bin` | 反量化结果，纯二进制、行主序，类型由 `output_type` 决定 |
| `<prefix>_log.txt` | 误差与性能日志（`key = value` 分区，便于脚本解析） |

4 bit 格式（NVFP4 与 MXFP4）的打包布局：每字节 2 个元素，**下标为偶数的元素放在低半字节**。

## 测试

关于「无测试代码」：项目代码要求第 1 条写的是"无测试代码"，而本选题的「需提交内容」写的是
"整个完成上述功能的程序（包含测试）"。这里按选题二的要求执行：`tests/` 是随 `make all` 一起
构建的正式验证套件（就是下面这三套），不是临时调试残留 —— 正式代码里没有任何打印或调试分支
（`src/` 除 `main.cu` 之外不出现 `printf` / `std::cout`）。

```bash
build/test_portability  # 平台能力探针：宏探测、warp 交换、原子、位模式、半精度、向量化访存
build/test_formats      # 格式转换：全编码枚举、特殊值、RNE/平局、MX/NV 缩放、4bit 译码穷举、GPU-vs-CPU 模糊测试
build/test_pipeline     # 流水线：全配置矩阵、打包布局、随机舍入、无偏性、误差统计、4/6 逐规则对拍
```

当前状态（RTX 5060 Ti / sm_120 / CUDA 12.9）：`test_formats` 292651 条、`test_pipeline` 21532 条、
`test_portability` 22 条断言，**全部通过**（合计 314205 条）。

### 架构下限验证

选题二要求「基础版本不得依赖 Hopper、Blackwell 或 Ampere 以上架构特性」。用
`-arch=compute_75 -code=compute_75`（只生成 PTX，再由驱动 JIT 到本机 sm_120）重新
编译并运行三套测试，结果与 `-arch=native` 完全一致（292651 / 21532 / 22 条断言全过），
说明 PTX 层面不含任何 sm_80 及以上的指令：

```bash
make ARCH=compute_75 && build/test_formats && build/test_pipeline
```

## 复现实验

```bash
python scripts/run_experiments.py    # 3 种分布 x 52 种配置 = 260 次运行 -> data/summary.csv
python scripts/benchmark.py          # 256..8192 规模扫描 -> data/benchmark.csv
python scripts/cpu_compare.py --n 8192   # GPU vs CPU 对比 -> data/cpu_compare.csv
python scripts/perf_probe.py --n 8192 --repeat 50   # 8192^2 快速性能台架
python scripts/analyze.py            # 汇总成 Markdown 表格 -> docs/measurements.md
python scripts/pack_submission.py --id <报名ID> --dest <fork>/02_quant_dequant/<报名ID>
                                     # 整理出不含 build/、data/tensors/、*.bin 的提交副本
python tools/verify_numpy.py --input data/tensors/random_512.tensor \
                             --lowp data/out/random_512__nvfp4_block_fp16_nearest_lowp.bin
```

详细结果、公式推导、优化历程与剖析分析见 **`总结报告.md`**；
自动生成的测量表格见 `docs/measurements.md`；nsys 剖析记录见 `docs/profile/`；
**项目当前状态、验证证据、复现命令与国产平台上机方案见 `项目状态与交接记录.md`。**

## 平台适配（国产 GPU）

选题二要求「默认需在英伟达平台上支持，每在一款国产平台上支持额外加分」。本实现只使用最基础的
CUDA C++，换平台原则上只需换编译器；确实存在六处需要感知平台的**源码级**差异，它们**全部集中在
`include/lpq/platform.hpp`** —— 逐条明细、抽象名与回退开关见 `总结报告.md` 6.6；
`tests/test_portability.cu` 会把它们拆成独立小测试逐项报告，并在失败时直接给出该加哪个 `-D`。

```bash
make PLATFORM=iluvatar   # 天数智芯 CoreX：clang++
make PLATFORM=moore      # 摩尔线程 MUSA：mcc（另见下面的 MUSA_MODE）
make PLATFORM=metax      # 沐曦 MACA：mxcc
```

| 平台 | 编译器 | 追加参数 |
| --- | --- | --- |
| nvidia | `nvcc` | `-arch=native -lineinfo` |
| iluvatar | `clang++` | `-I/usr/local/corex/include -fPIC -L/usr/local/corex/lib64 -lcudart` |
| moore | `mcc` | `-mtgpu -cuda_wrapper --offload-arch=mp_XX -I/usr/local/musa/include -L/usr/local/musa/lib -lcuda2musa`（另有 `mapping` / `plain` 两种模式） |
| metax | `mxcc` | `-lmxcudart` |

MUSA 多一个构建变量 `MUSA_MODE`：`wrapper`（`mcc -cuda_wrapper` + `-lcuda2musa`，源码一行不改）、
`mapping`（SDK ≥ 5.2 的编译期插件）、`plain`（源码先用 `musify-text` 迁移过）；默认 `auto` 按可用性
自上而下选。目标架构代号必须显式给 `--offload-arch=mp_XX`（MUSA 3.1.0 的有效值只有
`mp_10 / mp_21 / mp_22`），而官方没有「卡型号 → 代号」对照表，所以 `scripts/run_domestic.sh`
会**用与正式构建完全相同的模式与库参数**逐个候选编一个极小 kernel 并真的跑一次，自动试出可用代号。

### 上机流程

```bash
# 1. 先跑探针：它逐项报告平台缺什么，并直接给出该加哪个 -D
bash scripts/run_domestic.sh moore
#    日志落在 data/out/domestic_moore.log

# 2. 若探针报某项缺失，用 LPQ_EXTRA 传开关后重跑
LPQ_EXTRA="-DLPQ_NO_ATOMIC_ADD_F64=1" bash scripts/run_domestic.sh metax
```

脚本依次完成：环境检查 → 目标架构代号自动探测 → `make -n` 干跑 → 编译 → 探针 → 两套正确性
测试 → 小规模端到端实验 → 带宽天花板标定，并刻意不在失败时退出（租用机器按时计费，一次跑完
拿到尽可能多的信息比中途停下更有价值）。

### 支持状态

- **已实测：NVIDIA（RTX 5060 Ti，sm_120 / CUDA 12.9）** —— 三套测试 22 / 292651 / 21532 条断言
  全过；另用 `-arch=compute_75`（只出 PTX、由驱动 JIT）复跑，断言数与结果完全一致，证明不依赖
  Ampere 以上架构特性。
- **已实测：摩尔线程 MTT S4000（MUSA 3.1.0，mp_22）** —— 同一份源码经 `musify-text` 迁移后，
  三套测试断言数与英伟达侧**完全一致**（22 / 292651 / 21532，0 失败），端到端（含 NVFP4 + 4/6）
  与 CPU 参考实现逐位一致；该镜像里既没有 `libcuda2musa.so` 也没有 Mapping 插件，实际可用的是
  `make PLATFORM=moore MUSA_MODE=plain MUSA_ARCH=mp_22`。移植过程中修掉的三处数值语义差异见
  `总结报告.md` 6.7。
- **未实测：天数智芯（Iluvatar）/ 沐曦（MetaX）** —— 两条分支与抽象层按官方仓库的可用配置写好，
  并带自诊断探针；在真机跑通之前只是一份**已准备好、带诊断能力的移植方案**。
