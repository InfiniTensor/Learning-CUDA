# 沐曦 C500 MACA 后端

本目录提供 Hadamard 项目的独立沐曦 MACA/cu-bridge 后端。固定源码 **`fe44aa33a865e27a9d52120c94084994dbbcb8de`** 已在 **C500 25% 计算配额、16000 MiB 显存配额的 sGPU** 上完成 1504 组变换验证、180 项 API 契约检查、14 项 CLI 拒绝检查，以及三次独立进程的 4050 条基准组样本。所有请求阶段成功退出；真实编译弃用告警保留在原始日志，不能称为“编译无警告”。

算法起点是本项目天数后端固定提交 `a387db3332c6f9b01f128dd681848260c9691281`，沐曦代码使用独立命名空间、编译宏和结果记录。天数与本次沐曦设备都报告 warp64，不代表它们具有相同的编译、同步、舍入或性能行为；天数的 1504 组结果不能计为沐曦通过用例。

## 已核验的开发环境与实际配额

| 项目 | 当前实机/镜像信息 |
|---|---|
| 设备 | 沐曦 C500，sGPU 计算配额 **25%** |
| 可用资源配额 | `mx-smi` 明确显示 **16000 MiB** 显存配额；租赁界面标为 16GB |
| 物理设备属性 | 65536 MiB 显存、104 个 SM；这是物理属性，不能当成当前实例独占资源 |
| MACA/驱动工具报告 | MACA `3.0.0.8`，Kernel Mode `3.8.30`，`mx-smi 2.2.6` |
| 供应商镜像标签 | PyTorch `2.4`、Python `3.10.10`、MACA `3.0.0.4`；镜像标签与驱动工具报告分别记录，不合并成一个版本 |
| 编译器 | `cucc` / `mxcc 1.0.0`，版本标识 `df29922f9c` |
| 兼容运行时返回值 | runtime API 和 driver API 均返回 `11060`；这是兼容 API 的数值，不证明正在使用 NVIDIA CUDA 11.6 |
| warp | 设备报告 64；已对两个完整 warp 执行最小交换探针 |

**本实例是 25% 计算配额的 C500 sGPU，不是独占完整 C500。**后续性能必须标注该配额，不能与整卡结果直接比较，也不能简单将吞吐量乘以四外推整卡性能。尚未取得干扰隔离、固定时钟和性能计数器权限的证据。

记录环境时可运行：

```bash
mx-smi
/opt/maca/tools/cu-bridge/bin/cucc --version
/opt/maca/mxgpu_llvm/bin/mxcc --version
python3 --version
```

供应商的镜像和宿主机驱动可以分别提供组件。当前版本组合只以已运行探针为初步兼容证据，不据此承诺其他 MACA 版本组合可用。

## MACA 原生接口与本项目的 cu-bridge 路径

MACA 原生开发使用 `mxcc`、MACA 头文件与运行时接口，例如 `mc_runtime.h`、`mc_common.h` 和 `mcMalloc` 等；原生源文件需要按 MACA API 编写。这与保留 CUDA 命名的兼容源文件是两种开发入口，不能把它们的头文件和链接参数随意混用。

当前移植选择 **MACA 的 cu-bridge 编译路径**：`/opt/maca/tools/cu-bridge/bin/cucc` 编译设备程序，源码保留 `cuda_runtime.h`、`cuda_fp16.h`、`cuda_bf16.h`、`__half`、`__nv_bfloat16` 等兼容名称。已在该机器确认这些头文件及转换入口可用。保留 CUDA 名称不等于由 NVIDIA 编译器执行，也不等于 CPU 回退。

不要把天数的 `/usr/local/corex`、`-x ivcore`、`__ILUVATAR__`，或 NVIDIA 的 `-arch=sm_89` 参数带入沐曦构建。沐曦 Warp64 分支使用 `HADAMARD_METAX_WARP64` 和真实 MACA 编译环境的 `__MACACC__`；不手动伪造编译器识别宏。

## 最小实机探针已经证明什么

- FP16、BF16 各 257 个输入，包含正负值、正负零、舍入中点及边界样例。设备存储位模式与独立宿主机整数最近偶数舍入参考一致，随后转为 FP32 的简单算术结果位模式一致。
- 输入保持不变、分配区前后各 17 个元素的哨兵检查通过；覆盖只满足 2 字节对齐的半精度存储及不满线程块的尾部。
- 两个完整的 64-lane warp，对 XOR 距离 `1、2、4、8、16、32` 执行整数 `__shfl_xor` 交换，共 768 个观察值全部符合预期，哨兵检查通过。

单独的探针只能证明该有限范围，不证明完整 FWHT、跨寄存器蝶形、多行/尾行同步或融合性能；后文记录了随后独立执行的完整验证。BF16 证明的是存储/转换及 FP32 计算链路，**不宣称 C500 的原生 BF16 算术加速已经由本探针证实**。原始 [探针源码](https://github.com/a962695448-rgb/Learning-CUDA/blob/3676727fc21ea27bda668d743d10e52d7e138f68/03_hadamard_tc/a962695448-rgb/results/metax/fe44aa3/initial/metax_smoke.cu)、[编译日志](https://github.com/a962695448-rgb/Learning-CUDA/blob/3676727fc21ea27bda668d743d10e52d7e138f68/03_hadamard_tc/a962695448-rgb/results/metax/fe44aa3/initial/smoke_build.log)、[运行日志](https://github.com/a962695448-rgb/Learning-CUDA/blob/3676727fc21ea27bda668d743d10e52d7e138f68/03_hadamard_tc/a962695448-rgb/results/metax/fe44aa3/initial/smoke_run.log) 与 [初始环境](https://github.com/a962695448-rgb/Learning-CUDA/blob/3676727fc21ea27bda668d743d10e52d7e138f68/03_hadamard_tc/a962695448-rgb/results/metax/fe44aa3/initial/environment_initial.json) 均单独归档，没有借用天数日志。

## C++ API 与调用契约

接口见 [hadamard_api.h](hadamard_api.h)，实现见 [hadamard_api.cu](hadamard_api.cu)，命名空间为 `hadamard::metax`。现有声明包括：

| 入口 | 目标行为 |
|---|---|
| `transform` | FP16/BF16 Hadamard 变换，内部 FP32，允许完全原位变换 |
| `quantize_int4` | 对已存储的 FP16/BF16 数据执行每行对称 INT4 量化 |
| `transform_int4` | 融合变换与量化，先按 FP16/BF16 输出类型舍入，再量化 |

方法枚举包含 `Baseline`、共享内存 `Optimized` 和条件编译的 `Warp64`，默认仍是 `Optimized`。本次完整沐曦运行检查了三种方法。对当前已经验证的 C500/SDK/配额，可在满足下方编译与设备检查条件后显式选择 `Method::Warp64`；不自动改变其他设备的默认派发。

输入为连续设备内存 `[rows,n]`，N 是 1～256 的 2 次幂；连续 `[batch,seq,heads,head_dim]` 可在检查乘法溢出后令 `rows=batch*seq*heads`、`n=head_dim`。API 不接受 stride 参数，不自动转换非连续视图。scale 应有限且为正，输入应有限，变换后的值应位于输出存储类型的有限范围内。

输入/变换输出各需 `rows*n` 个对应 dtype，INT4 需 `rows*ceil(n/2)` 字节，量化 scale 需 `rows` 个 float。输入及变换输出要求 2 字节对齐，scales 要求 float 对齐；量化涉及的缓冲区互不重叠。transform 只允许完全原位或互不重叠。API 不查询真实分配容量或设备归属，调用方必须保证它们有效。

INT4 约定为 `[-7,7]`，每行 `s=max(abs(x))/7`，全零行取 `s=1`；最近偶数舍入，偶数元素存低四位，N=1 的空高四位为零。本次完整运行中融合路径与分步路径的 packed bytes 及 float scales 已分别对照并精确一致。

操作只在调用方 stream 上发射，不分配、不复制、不等待。检查参数/发射返回值，并在回读前检查 stream 同步的返回值。合法 `rows=0` 不发射内核，仍检查 N、scale 和方法；不能把零行成功当作设备路径验证。

Warp64 必须用真实 MACA/cu-bridge 编译，显式启用 `-DHADAMARD_METAX_WARP64`；独立 API 调用方还须在设备初始化时通过 `cudaGetDeviceProperties` 检查 `prop.warpSize == 64`。未编入该路径时，非空 Warp64 调用返回 `cudaErrorNotSupported`，不静默回退。仅通过该属性检查仍不足以替代本平台完整测试。

## 完整复现入口

本平台使用独立的 [run_platform.py](run_platform.py) 和 [validate_and_benchmark.cu](validate_and_benchmark.cu)。以下命令重现已验证版本的执行方式；从项目 `03_hadamard_tc/a962695448-rgb` 目录执行并固定上述源码提交，结果目录必须尚不存在：

```bash
# 快速检查，不能替代完整矩阵或性能验收。
python3 platforms/metax/run_platform.py --warp64 --quick --no-benchmark \
  --output results/metax/warp64_quick_01

# 完整矩阵与默认基准。
python3 platforms/metax/run_platform.py --warp64 --repeats 100 --groups 5 \
  --output results/metax/warp64_full_01

# 仅共享实现的单独复测入口；本次公开全量记录为上面的 Warp64 构建。
python3 platforms/metax/run_platform.py --no-benchmark \
  --output results/metax/shared_full_01
```

`--maca-root` 默认为 `/opt/maca`，`--compiler` 默认为该目录下 `tools/cu-bridge/bin/cucc`。runner 保留 `cucc` 入口名称，不因解析符号链接而改变兼容模式；实际编译为 cucc、C++17、`-O2`、项目 include 路径和两个 `.cu` 文件，Warp64 模式额外定义 `HADAMARD_METAX_WARP64`，由 cucc 处理兼容运行库链接。

runner 只为本次子进程设置 `MACA_PATH`、指向 cu-bridge 的 `CUDA_PATH`/`CUCC_PATH`、工具搜索路径及存在的 SDK 库目录；保留继承环境，不安装/替换驱动或框架。`--warp64` 是构建选项，生成的程序会在设备运行前检查真实 warp 宽度。不满足条件时应明确失败，不能把该条件静默跳过后写成支持。

本次 Warp64 完整模式实际执行 1504 个参数组合、180 项 API 契约检查，另有 runner 的 14 项 CLI 拒绝检查；基准为九条路径，每轮 1350 组样本。不启用 Warp64 的独立构建具有 1500 个参数组合、122 项 API 契约检查、12 项未编入路径检查和六条基准路径；该构建入口不与本次启用 Warp64 的实测数量相加。

每个新目录保存 `build.log`、`invalid_*.log`、`validation.log/json`、可选 `benchmark.log/csv` 和 `run_summary.json`。摘要包含本次沐曦源码 hash、Git HEAD/状态、MACA 环境、编译器、各阶段命令/退出码及产物 hash。`adapted_from` 是算法来源的天数提交，不能误认成当前沐曦实际测试源码版本。后续多轮须使用新的文件/目录；同一正确性矩阵重复运行不累计成新用例覆盖。

## 完整实机结果

`fe44aa33a865e27a9d52120c94084994dbbcb8de` 在测试时工作树干净，五个参与构建/运行的源文件 hash 已与该 Git 提交逐一核对。`run_summary.json` 为 `PASS`，`quick=false`、`warp64_enabled=true`、`full_matrix=true`。三种方法和原位路径的实际运行证据与阈值都记录在本平台，未把其他平台结果改名使用。

| 验收项 | 当前状态 | 原始记录 |
|---|---|---|
| 最小存储/舍入/算术、整数 shuffle | 通过上述有限范围 | [initial](https://github.com/a962695448-rgb/Learning-CUDA/blob/3676727fc21ea27bda668d743d10e52d7e138f68/03_hadamard_tc/a962695448-rgb/results/metax/fe44aa3/initial/smoke_run.log) |
| 三种方法 cucc 构建 | 退出 0，有 `__shfl_xor` 弃用告警 | [build.log](https://github.com/a962695448-rgb/Learning-CUDA/blob/3676727fc21ea27bda668d743d10e52d7e138f68/03_hadamard_tc/a962695448-rgb/results/metax/fe44aa3/build.log) |
| 两 dtype 完整矩阵、API、内存和 INT4 | 1504 组、180 项 API、14 项 CLI 通过 | [validation.json](https://github.com/a962695448-rgb/Learning-CUDA/blob/3676727fc21ea27bda668d743d10e52d7e138f68/03_hadamard_tc/a962695448-rgb/results/metax/fe44aa3/validation.json)、[运行日志](https://github.com/a962695448-rgb/Learning-CUDA/blob/3676727fc21ea27bda668d743d10e52d7e138f68/03_hadamard_tc/a962695448-rgb/results/metax/fe44aa3/validation.log) |
| 多行、尾行与网格复用 | 覆盖 65537 行及 262145 行边界路径 | 同一完整验证记录，不重复计数 |
| 同配额性能 | 三进程均退出 0，4050 条原始样本 | [repeat_runs.json](https://github.com/a962695448-rgb/Learning-CUDA/blob/3676727fc21ea27bda668d743d10e52d7e138f68/03_hadamard_tc/a962695448-rgb/results/metax/fe44aa3/repeat_runs.json) 与后文 CSV |
| 版本与归档 | 源码和原始文本均按 hash 固定 | [run_summary.json](https://github.com/a962695448-rgb/Learning-CUDA/blob/3676727fc21ea27bda668d743d10e52d7e138f68/03_hadamard_tc/a962695448-rgb/results/metax/fe44aa3/run_summary.json)、[公开清单](https://github.com/a962695448-rgb/Learning-CUDA/blob/3676727fc21ea27bda668d743d10e52d7e138f68/03_hadamard_tc/a962695448-rgb/results/metax/fe44aa3/manifest.json) |

### 矩阵、误差和量化一致性

N 覆盖 `{1,2,4,8,16,32,64,128,256}`，普通矩阵行数为 `{1,3,17,257}`，scale 为 1 和 `1/sqrt(N)`，N=1 不重复计相同 scale。输入含均匀 `[-1,1)`、标准差 0.5 的正态分布、幅度 8 的离群值、全零和单位脉冲；随机分布使用 `123、8042、15961` 三个种子，确定性零值/脉冲只计一次。另有 N=1/2 的 65537 行和 262145 行用例，分别覆盖共享实现及每 CTA 四行的 Warp64 网格复用/尾行。

| 精度 | 实际参数组合 | 检查元素 | API 契约检查 | 已舍入参考最大绝对误差 | 未舍入参考最大绝对误差 |
|---|---:|---:|---:|---:|---:|
| FP16 | 752 | 4,105,264 | 90 | 0.00390625 | 0.015620231628418 |
| BF16 | 752 | 4,105,264 | 90 | 0.000003814697265625 | 0.124984741210938 |

所有三种方法的变换和原位结果逐字节一致；FP16/BF16 两类下，CPU 对实际设备变换输出量化、三种方法的分步量化和融合量化，packed bytes 和 scale 全部一致。非法参数、对齐、重叠、溢出、空指针、零行、输入保持、前后哨兵及手写正负半整数 RNE 预期均在本次运行检查。每 dtype 的 262145 行 N=1/2 用例共 786,435 个元素，已计入上表。

正确性沿用项目已舍入参考定义：独立 FP64 稠密结果经 FP32 转换后舍入到输出 dtype，与实际输出比较，FP16 绝对误差严格小于 `1e-2`、BF16 严格小于 `5e-2`。**阈值判定只对应已舍入列；未舍入列包含输出存储精度的舍入误差，不能也说它小于同一阈值。**两列均公开保留，没有放宽阈值、缩小测试分布或跳过 BF16。

### 编译告警与兼容范围

cucc 编译使用 `mxcc 1.0.0 (df29922f9c)`，SDK 实际解析为 `/opt/maca-3.0.0`。完整 [build.log](https://github.com/a962695448-rgb/Learning-CUDA/blob/3676727fc21ea27bda668d743d10e52d7e138f68/03_hadamard_tc/a962695448-rgb/results/metax/fe44aa3/build.log) 保留 219034 字节原始输出，包括 legacy `__shfl_xor` 被 SDK 标记为 deprecated 的警告及模板实例化信息；最小探针的编译也有相应告警。构建和真实运行退出 0 不等于无告警。

本次版本继续使用经该 MACA/设备组合独立验证的 legacy shuffle。它可能在其他或未来 SDK 中变化；升级或替换原语时必须独立核验 lane 参与、width/mask 语义、全量正确性和性能。当前报告既不删除告警，也不声称已经证明所有未来版本兼容。

### 三轮性能和原始数据

三次独立基准进程使用相同二进制 SHA256 `8f7ac15ea904a47de0b26635094cdc0740a49b9bc4039490d5dde0e52cbf35e1`，均退出 0。每轮覆盖两 dtype、N=64/128/256、rows=1/17/257/4096/16384，三方法各有 transform/split/fused 三种操作，共 30 个 shape/dtype 条件和九条路径。各路径先预热 10 次，五组、每组重复 100 次，组间轮换方法次序；每轮 1350 条组样本，合计 4050。

原始样本见 [第一轮](https://github.com/a962695448-rgb/Learning-CUDA/blob/3676727fc21ea27bda668d743d10e52d7e138f68/03_hadamard_tc/a962695448-rgb/results/metax/fe44aa3/benchmark.csv)、[第二轮](https://github.com/a962695448-rgb/Learning-CUDA/blob/3676727fc21ea27bda668d743d10e52d7e138f68/03_hadamard_tc/a962695448-rgb/results/metax/fe44aa3/benchmark_run2.csv)、[第三轮](https://github.com/a962695448-rgb/Learning-CUDA/blob/3676727fc21ea27bda668d743d10e52d7e138f68/03_hadamard_tc/a962695448-rgb/results/metax/fe44aa3/benchmark_run3.csv)，完整统计见 [analysis.json](https://github.com/a962695448-rgb/Learning-CUDA/blob/3676727fc21ea27bda668d743d10e52d7e138f68/03_hadamard_tc/a962695448-rgb/results/metax/fe44aa3/analysis/analysis.json) 和 [method_summary.csv](https://github.com/a962695448-rgb/Learning-CUDA/blob/3676727fc21ea27bda668d743d10e52d7e138f68/03_hadamard_tc/a962695448-rgb/results/metax/fe44aa3/analysis/method_summary.csv)。每轮内部先取五组中位数，再对相同 shape/dtype/scale/操作匹配比较；不与天数或旧实验的测量相除。

Warp64 的 90/90 组匹配比较在每一轮均比本机基线减少至少 5% 耗时，没有相对基线退化的样例。共享内存 `Optimized` 则有 55/90 组每轮减少至少 5%，30 组至少一轮退化、29 组每轮退化、14 组每轮退化超过 3%。分析总数 145 是 **共享版 55 + Warp64 90**，不能全算成 Warp64 收益；共享版负例完整保留。

下表是本次全部对应形状和三轮的“对照中位耗时 / Warp64 中位耗时”范围：

| 操作 | 基线 / Warp64 | 共享内存优化版 / Warp64 |
|---|---:|---:|
| transform | 1.06974554～2.53939018 倍 | 1.11071430～1.69060521 倍 |
| split | 1.07633461～5.29428683 倍 | 1.06692802～1.44593072 倍 |
| fused | 1.28999182～6.05722033 倍 | 1.02615774～1.43636359 倍 |

相对共享内存优化版，Warp64 在本轮矩阵中没有退化，但只有 **88/90** 组每轮都减少至少 5% 耗时。两个例外是 FP16/BF16 的 `[16,64,16,128]` fused，改善只有约 2.55%～3.84%；不能写成相对共享版也是 90/90 达到 5%。

只比较 Warp64 自身分步与融合路径，在同一轮相同 shape/dtype/scale 下，`split/fused` 为 **1.27218808～1.70823860 倍**，对应融合耗时减少 **21.3953%～41.4602%**；30/30 个形状条件在每轮都降低至少 5%。这些量化收益使用当前沐曦九路径实验，与其他平台或旧版本的时间无关。

负例和波动同样保留：共享版 FP16 `[1,17,1,128]` transform 比基线慢约 14.50%、5.57%、6.64%；Warp64 相对基线的最小收益来自 BF16 `[1,1,1,128]` transform，其三轮降时为约 6.52%、22.66%、21.92%。少数重复进程无法证明任意共享负载下的性能稳定性；均值/标准差/CV 描述采样，不是置信区间。

本次固定分配与只读输入，复用输入可能受缓存影响。设备事件区间排除分配、H2D/D2H、预热和验证，可能包含主机发射之间的设备空闲间隔；它不是端到端时间，也不是隔离测得的纯单内核执行时间。逻辑读写量换算的 GB/s 不是实测物理显存带宽。速度比仅覆盖 N=64/128/256 的性能矩阵，N=1～32 虽通过正确性，不能外推相同收益。天数和沐曦处于不同硬件/配额，不直接用两张表相除宣称平台优劣。

## 交付范围、公开清单与局限

- 本次独立验证对象是上述 MACA 3.0 系列/cu-bridge 和 C500 **25% sGPU**，不是物理整卡；没有证据证明同宿主机其他租户始终空闲或设备时钟固定。前后 `mx-smi` 快照随结果保留，只反映采样时刻。
- 支持的 C++ 接口为连续 FP16/BF16、N=1～256 的 2 次幂、调用方 stream、原位变换、每行 INT4 分步与融合。非连续视图、任意长度、容量不足、跨设备指针、非有限输入及超出存储范围不在当前契约内。
- 默认方法仍为共享内存 `Optimized`，MR-V100 的结果未用于决定 C500 通过。对本次已经验证的 C500 环境推荐显式选择 `hadamard::metax::Method::Warp64`；调用方必须满足编译宏及实际 `warpSize==64` 预检查。
- 未提供沐曦 PyTorch Python 扩展，没有证明原生 BF16 算术或矩阵单元加速，没有硬件性能计数器或主机端到端延迟数据。其他 SDK/设备/配额应重新验证；deprecated shuffle 的未来兼容性不由本次结果保证。

[结果目录清单](https://github.com/a962695448-rgb/Learning-CUDA/blob/3676727fc21ea27bda668d743d10e52d7e138f68/03_hadamard_tc/a962695448-rgb/results/metax/fe44aa3/manifest.json) 按字节数与 SHA256 列出原始运行、三轮 CSV、派生分析和初始探针/环境。29 份完整运行原始文件通过传输清单校验，五个实际源码 hash 与固定 Git 提交匹配；目录 `.gitattributes` 使用 `-text`，避免 Git 换行转换破坏日志和证据 hash。原始编译告警保持完整，没有复制本机二进制和私人租赁收据。

公开材料排除实例访问地址、租赁编号和凭据；保留足以复现的设备型号、配额、软件版本、命令和日志。本平台扩展独立记录构建、数值与性能结果，后续优化保持同一量化契约和可复现测量；PR 与课程登记在项目所有者验收后单独执行。
