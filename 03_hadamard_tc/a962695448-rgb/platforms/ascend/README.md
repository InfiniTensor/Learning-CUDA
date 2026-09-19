# 昇腾 910B1 原生 Ascend C 后端

本目录实现原生 Ascend C/CANN Hadamard 变换和融合 INT4。固定源码 `f614762a69e75524db65b47fbf7d6d01836db438` 的 vector-scale OFF/ON 两种构建，均已在真实 Ascend910B1 完成主矩阵、grid 专项、大行数变换专项和接口检查；源码、日志和配置分别保存，重复矩阵不相加。

严格交错 OFF/ON 对照和独立统计复核已完成。120 个向量路径/形状/精度条件中，97 个在三轮每轮都减少至少 5% 耗时，11 个三轮均更快但未全部达到 5%，12 个方向混合；不能宣称所有条件都更快或应用整体提速。`--vector-scale` 默认保持 OFF，建议只在已测适用范围显式启用并复核。本文是实验与交付记录，不是设备释放回执。

## 实机环境与原生执行路径

| 项目 | 已核实信息 |
|---|---|
| 实际 SoC | `Ascend910B1`，ACL SoC 查询及 NPU 日志相符 |
| CANN | `/usr/local/Ascend/cann-9.0.0` |
| 主机 | aarch64；Python 3.11.15，GCC 12.3.1/openEuler 工具链 |
| 设备工具 | `npu-smi 25.5.1`，初始设备状态 Health=OK |
| 初始 HBM 报告 | 65536 MB 总量；设备状态快照不证明独占、固定频率或无其他负载 |
| 构建 | CMake + CANN 自带 Ascend C kernel CMake，目标 `Ascend910B1`、`RUN_MODE=npu` |

[初始环境与源码清单](https://github.com/a962695448-rgb/Learning-CUDA/blob/3676727fc21ea27bda668d743d10e52d7e138f68/03_hadamard_tc/a962695448-rgb/results/ascend/initial_probes/manifest.json) 保留设备查询、目录信息及自写探针。实现直接使用 ACL caller stream 与 NPU kernel；`ScalarButterfly` 中的“标量”同样在 NPU 上执行，**不是 CPU 回退**。CPU 代码只负责输入、独立参考、检查和控制流程，不代替设备输出。

项目只引用已安装 CANN 的 `kernel_operator.h`、ACL 和 Ascend C CMake 接口，不复制私人 SDK 头文件实现。SDK 产生的内核对象、launch 头文件和构建缓存位于忽略的 `build/ascend/`，不混进公开源代码或结果档案。

## 两种变换与量化的数值约定

- `Method::ScalarButterfly`：NPU 标量蝶形基线。
- `Method::VectorGather`：使用 NPU Gather/矢量操作实现蝶形，内部 FP32。
- `--vector-scale`：仅将 `VectorGather` 变换后的乘 scale 步骤改为矢量 `Muls`；默认关闭。它不改变下面的量化除法，也不是重新启用矢量 `Div`。

输入/输出为 FP16/BF16，内部计算使用 FP32。Hadamard 支持 N=1～256 的 2 次幂，常用 transform scale 为 1 或 `1/sqrt(N)`。每行 INT4 为 `[-7,7]`，量化 scale 为 `max(abs(x))/7`、全零行取 1；采用最近偶数舍入，偶数元素存低四位，N=1 的空高四位为零。融合路径先舍入到公开 FP16/BF16 存储类型，再读回 FP32 量化，保持分步语义。

### 为什么量化没有使用矢量 Div

独立除法探针在同一实机上发现：矢量 `Div` 的 256 个输入中有 64 个结果与项目 FP32 位级参考不同，而 NPU 标量 C++ `/` 的 256 个结果全部逐位相同。输入保持与哨兵检查通过。原始 [矢量日志](https://github.com/a962695448-rgb/Learning-CUDA/blob/3676727fc21ea27bda668d743d10e52d7e138f68/03_hadamard_tc/a962695448-rgb/results/ascend/division_probe/div_vector_run.log)、[两路径日志](https://github.com/a962695448-rgb/Learning-CUDA/blob/3676727fc21ea27bda668d743d10e52d7e138f68/03_hadamard_tc/a962695448-rgb/results/ascend/division_probe/div_scalar_run.log)、[配置/源码与清单](https://github.com/a962695448-rgb/Learning-CUDA/blob/3676727fc21ea27bda668d743d10e52d7e138f68/03_hadamard_tc/a962695448-rgb/results/ascend/division_probe/manifest.json) 完整保留。

其中一个影响 INT4 的边界是：参考除法得到 `3.49999976`，矢量结果变成 `3.5`。随后最近偶数舍入会分别得到 3 和 4，不能只因浮点误差很小就判定量化等价。生产实现因此使用已经单独核验的 **NPU 标量除法**计算量化 scale 和 `x/scale`，不采用该矢量 Div 或近似倒数替代。

探针控制器的 `PROBE_FINISHED` 只表示流程结束；矢量运行退出 1，启用标量的 `--mode both` 运行也因仍包含矢量失败而退出 1。必须分别读取子路径结果，不能把整批称为 PASS，也不能将该精度契约差异直接称为厂商 bug。

`f614762` 增加固定串联回归：输入 `[0.75,0.25]` 的未归一化变换为 `[1,0.5]`，量化 scale 的 FP32 位模式应为 `0x3e124925`，packed 输出应为 `0x37`。两 dtype、两 Method 各一例，共四个合同检查，分步/融合及哨兵均通过；它们增加在 API 检查计数中，不把例内的多次调用拆成多个用例。

## 公共 C++ API

接口与实现分别见 [hadamard_api.h](hadamard_api.h)、[hadamard_api.cpp](hadamard_api.cpp) 和 [hadamard_kernel.cpp](hadamard_kernel.cpp)，命名空间为 `hadamard::ascend`。

| 入口 | 行为 |
|---|---|
| `transform` | 变换至同 dtype 输出，允许完全原位 |
| `quantize_int4` | 量化已经存储为 FP16/BF16 的输出；两种 Method 共用同一个 NPU 量化内核 |
| `transform_int4` | 融合变换与量化，输出 packed bytes 和 float scales |

`StorageType::FP16/BF16` 指定类型，输入/变换输出使用 `std::uint16_t*` 承载对应原始位模式。连续 `[batch,seq,heads,head_dim]` 经溢出检查后展平为 `[rows,n]`，`rows=batch*seq*heads`、`n=head_dim`；不接受任意 stride 或自动重排非连续视图。

调用方分配当前设备上容量足够的缓冲区：输入/变换输出各 `rows*n` 个 16 位元素，packed 为 `rows*ceil(n/2)` 字节，scale 为 `rows` 个 float。输入及变换输出至少 2 字节对齐、scales 至少 4 字节，packed 无额外地址对齐要求。transform 只允许完全原位或互不重叠；量化涉及的各缓冲区必须互不重叠。

返回值为 `aclError`，参数/发射错误需立即检查。非空调用只在传入的 `aclrtStream` 发射，不分配、不复制、不等待；异步执行错误在调用方同步 stream/event 时检查。`rows=0` 在检查枚举、N、scale、block_dim 后直接返回，允许空指针/空 stream，不发射内核。

`block_dim` 支持 1～32，默认 1。它是本实现暴露的启动参数，不代表最优值，也不能直接等同其他 CUDA 平台的线程块含义。调用方保证 stream、设备归属、容量和生命周期；API 不扫描设备数据，输入与 FP32 中间值应有限，结果应在目标类型的有限范围内。

## 构建和复现命令

从项目 `03_hadamard_tc/a962695448-rgb` 目录、配套 aarch64 CANN 环境执行。结果目录和关联构建目录必须是新目录。runner 仅在子进程中加载已安装 CANN `set_env.sh` 并设置必要搜索路径，不安装或替换驱动/框架，也不将完整环境变量写进结果。

```bash
# 快速检查与 M17/N128 的单点计时，只用于试跑。
python3 platforms/ascend/run_platform.py \
  --cann-root /usr/local/Ascend/cann-9.0.0 --quick --pilot-benchmark \
  --output results/ascend/reproduce_quick_01

# 固定 f614762 源码，vector-scale OFF：完整验证，默认不做基准。
python3 platforms/ascend/run_platform.py \
  --cann-root /usr/local/Ascend/cann-9.0.0 \
  --output results/ascend/reproduce_scale_off_01

# 相同源码，vector-scale ON：单独完整验证。
python3 platforms/ascend/run_platform.py \
  --cann-root /usr/local/Ascend/cann-9.0.0 --vector-scale \
  --output results/ascend/reproduce_scale_on_01
```

[CMakeLists.txt](CMakeLists.txt) 要求 `RUN_MODE=npu`，默认 `SOC_VERSION=Ascend910B1`；使用实际安装的 `aarch64-linux/tikcpp/ascendc_kernel_cmake/ascendc.cmake`。`ENABLE_VECTOR_SCALE` 默认 OFF，runner 的 `--vector-scale` 会将它设为 ON，并核对生成 CMakeCache 与结果 JSON 的开关状态，防止测试到另一种构建。

runner [run_platform.py](run_platform.py) 的主验证 block_dim 默认 1，可用 `--block-dim` 指定；只有显式加 `--benchmark` 或 `--pilot-benchmark` 才计时。普通小基准为 rows=1/17/257、N=64/128/256、两 dtype；pilot 为 rows=17/N=128。默认预热 3 次、每组重复 5 次、五组，可显式调整 `--warmup`、`--repeats`、`--groups`。两方法各有 transform/split/fused 六路径，加一个共用的 `quant_only`，不是两套独立量化算法。

`--quick` 不是完整矩阵；`--skip-stress` 会令 `full_suite_complete=false`，不能跳过大行数专项后继续宣称完整套件通过。SDK 生成物和可执行文件在 `build/ascend/<本次目录>`；公开结果目录保存配置/构建日志、CLI 拒绝日志、验证 JSON/日志、可选原始 CSV 及运行摘要。

## 已归档的真实正确性结果

| 固定版本/构建 | 主矩阵 | grid 专项 | 大行数专项 | API 检查 | CLI 拒绝检查 |
|---|---:|---:|---:|---:|---:|
| `1b305d3e3a07b28b6879596babf47aadbf84ba0c` | 1496 | 128 | 2 | 184 | 17 |
| `f614762a69e75524db65b47fbf7d6d01836db438` OFF | 1496 | 128 | 2 | 188 | 17 |
| 同一 `f614762` ON | 1496 | 128 | 2 | 188 | 17 |

三套完整记录均为 `execution=npu`、`full_matrix=true`、`full_suite_complete=true`。主矩阵与专项分列，避免把不同测试范围混成一个数字；相同矩阵跨构建不累计成新覆盖。旧 `1b305` 是 184 个 API 检查，新的 188 是增加四个固定除法串联回归后的计数。

主矩阵覆盖 N=1～256 的全部 2 次幂、rows=1/3/17/257、两种 transform scale（N=1 不重复）、零值、单位脉冲、均匀/正态/离群值分布及多个种子。两 dtype 各 748 组、3,122,218 个元素，ScalarButterfly/VectorGather/原位路径及 CPU FP32 FWHT 输出位模式一致；全部元素另与已舍入 dtype 的独立 FP64 稠密参考比较。

| 精度 | 主矩阵最大绝对误差：已舍入 FP64 | 主矩阵最大绝对误差：未舍入 FP64 |
|---|---:|---:|
| FP16 | 0.00390625 | 0.015620231628418 |
| BF16 | 0.000003814697265625 | 0.124984741210938 |

FP16 `<1e-2`、BF16 `<5e-2` 的判定只对应已舍入参考；未舍入列包含输出存储精度误差，不声称它也低于相同阈值。CPU 对实际 NPU 变换输出量化、设备分步及融合的 packed bytes 和 float scales 精确一致。

grid 专项在 rows=33、N=1/256、block_dim=1～32 上运行，每 dtype 64 组、271,392 元素，合计 128 组；其误差单独记录。大行数专项为每 dtype 一例 rows=262145、N=1、block_dim=32，只覆盖两种 transform 和 VectorGather 原位变换；**不声称大行数量化已测，也不声称已经覆盖大于 2^32 的索引**。

正式输出分别见 [1b305 全量](https://github.com/a962695448-rgb/Learning-CUDA/blob/3676727fc21ea27bda668d743d10e52d7e138f68/03_hadamard_tc/a962695448-rgb/results/ascend/production_runs/full_1b305/validation.json)、[f614762 OFF](https://github.com/a962695448-rgb/Learning-CUDA/blob/3676727fc21ea27bda668d743d10e52d7e138f68/03_hadamard_tc/a962695448-rgb/results/ascend/production_runs/scale_off_f614762/validation.json)、[f614762 ON](https://github.com/a962695448-rgb/Learning-CUDA/blob/3676727fc21ea27bda668d743d10e52d7e138f68/03_hadamard_tc/a962695448-rgb/results/ascend/production_runs/scale_on_f614762/validation.json)。四份完整/quick 运行摘要的七个源码 hash 已与对应 Git 对象核对，产物 hash 与下载原件一致；旧 quick/pilot 不当作额外完整验证。

最新 OFF 二进制 SHA256 为 `fe0b0cd7d3998e61ce8f6ac6a1d8dd17f9964bc41d1ea68d68011ff7010b70d3`；ON 为 `b738db0bb242945b9e16b9c919c9a6378c26c854edeac0caace3034b447b99fa`。它们来自同一源码、不同明确 CMake 开关，独立构建与测试；本机二进制不放入公开档案。

## 交错 OFF/ON 性能对照与独立复算

已保留旧版单 block 和 block_dim=32 的 pilot，以及最新 OFF/ON 的阶段试跑 CSV；它们位于 [production_runs](https://github.com/a962695448-rgb/Learning-CUDA/blob/3676727fc21ea27bda668d743d10e52d7e138f68/03_hadamard_tc/a962695448-rgb/results/ascend/production_runs/manifest.json)，采样方法和版本不能混合。不同阶段的 32.6 微秒与 83.7 微秒，或单点标量/矢量比值，都不足以替代严格控制的交错结果，本文暂不将它们写成最终优化结论。

固定对照包含 rows=1/17/257/4096/16384 与 N=16/64/128/256 的 20 个形状，每个独立进程同时测两 dtype，即 40 个 shape/dtype 条件。OFF/ON 交错、三轮，实际 120 个独立进程均退出 0、8400 条原始组样本通过核验，block_dim=32、五组、每组五次重复、预热三次。旧/新执行顺序按预先确定的轮次和形状奇偶交替，不作事后挑选。它不是同进程 A/B。

完整 [ab_summary.json](https://github.com/a962695448-rgb/Learning-CUDA/blob/3676727fc21ea27bda668d743d10e52d7e138f68/03_hadamard_tc/a962695448-rgb/results/ascend/ab_f614762/ab_summary.json) 保留固定参数、执行顺序、每进程原始 CSV/log、二进制前后 hash 和设备快照。两个二进制运行前后均未变化，并与前述 OFF/ON 全量构建记录匹配；外部还核对了源码 commit 与七个源码 hash。因此没有将脚本 `--source-id` 的调用方标签本身当成源码证明。

每个变体每轮合并 CSV 有 1400 行，共六份：[OFF 第一轮](https://github.com/a962695448-rgb/Learning-CUDA/blob/3676727fc21ea27bda668d743d10e52d7e138f68/03_hadamard_tc/a962695448-rgb/results/ascend/ab_f614762/old_run1.csv)、[第二轮](https://github.com/a962695448-rgb/Learning-CUDA/blob/3676727fc21ea27bda668d743d10e52d7e138f68/03_hadamard_tc/a962695448-rgb/results/ascend/ab_f614762/old_run2.csv)、[第三轮](https://github.com/a962695448-rgb/Learning-CUDA/blob/3676727fc21ea27bda668d743d10e52d7e138f68/03_hadamard_tc/a962695448-rgb/results/ascend/ab_f614762/old_run3.csv)，以及 [ON 第一轮](https://github.com/a962695448-rgb/Learning-CUDA/blob/3676727fc21ea27bda668d743d10e52d7e138f68/03_hadamard_tc/a962695448-rgb/results/ascend/ab_f614762/new_run1.csv)、[第二轮](https://github.com/a962695448-rgb/Learning-CUDA/blob/3676727fc21ea27bda668d743d10e52d7e138f68/03_hadamard_tc/a962695448-rgb/results/ascend/ab_f614762/new_run2.csv)、[第三轮](https://github.com/a962695448-rgb/Learning-CUDA/blob/3676727fc21ea27bda668d743d10e52d7e138f68/03_hadamard_tc/a962695448-rgb/results/ascend/ab_f614762/new_run3.csv)。它们只是各 cell 原始 CSV 的拼接，**不能将合并表再计为新的 8400 个观察值**。`quant_only` 和标量路径均完整保留作控制项，没有按结果筛选。

自写 [ascend_ab_benchmark.py](https://github.com/a962695448-rgb/Learning-CUDA/blob/3676727fc21ea27bda668d743d10e52d7e138f68/03_hadamard_tc/a962695448-rgb/results/ascend/ab_f614762/ascend_ab_benchmark.py) 的 SHA256 为 `ce49e0bb9f3c03db4a8b747030e92c6e543c23269da6e892d70d9416e3c0342d`，与实机摘要和本地脚本一致。它不编译、不修改源码、不重设环境，只运行已经构建且完整验证过的两份程序。复现时先在相应主机加载配套 CANN 环境，再把 `BINARY_OFF`、`BINARY_ON` 设置为两份 `run_summary.json` 中 `binary.path` 指向的真实文件，使用全新的输出目录：

```bash
python3 results/ascend/ab_f614762/ascend_ab_benchmark.py \
  --old-binary "$BINARY_OFF" --new-binary "$BINARY_ON" \
  --source-id f614762a69e75524db65b47fbf7d6d01836db438 \
  --block-dim 32 --repeats 5 --warmup 3 --groups 5 \
  --output results/ascend/reproduce_interleaved_ab_01
```

本脚本运行的是各形状基准附带检查，完整正确性来自分开的 production 记录；不把这 120 个基准进程算作新的完整测试套件。也不应在对照过程中并发编译、运行其他 NPU 作业或修改输入/源码。

### 全部正负例与控制条件

独立复算逐项核对六份合并 CSV 和 120 份 cell CSV，重建合并内容并与原件逐字段相符；120 份进程日志 hash 也一致。每轮使用五组原始毫秒读数的中位数，同轮相同形状、dtype、method 配对。耗时下降百分比定义为 `(OFF_ms-ON_ms)/OFF_ms*100`，“稳定达到 5%”严格要求三轮每一轮都达到该阈值。

| 分类 | 120 个向量路径条件 | 160 个控制条件 |
|---|---:|---:|
| 三轮每轮均减少至少 5% 耗时 | 97 | 0 |
| 三轮均更快，但未全部达到 5% | 11 | 105 |
| 三轮均更慢，但未全部达到 5% | 0 | 7 |
| 方向混合或不稳定 | 12 | 48 |
| 三轮每轮均增加至少 5% 耗时 | 0 | 0 |

这里的控制条件是 scalar transform/split/fused 和 `quant_only`。控制没有三轮都达到 5% 的稳定变化，**不等于控制完全不受编译布局、发射空档或系统状态影响**；单轮仍可有较大变化。不能将向量路径的全部变化都归因于这一条 Muls 指令。

所有负例保留。例如 FP16 rows=17、N=16 的 vector_split，三轮降时约为 `+4.518%、+4.571%、-4.507%`；负值表示该轮 ON 更慢。向量配置没有三轮全部更慢的情况，不代表没有单轮退化。三轮 5% 规则是描述性稳定标准，不是统计显著性检验；未计算置信区间。

### 建议显式启用的已测范围

在 **本机 Ascend910B1/CANN9.0.0、block_dim=32、transform scale=1、FP16/BF16，以及 rows∈{257,4096,16384}、N∈{64,128,256}** 的离散受测范围中，各操作有 18 个 shape/dtype 条件。ON 相对 OFF 的全部三轮耗时下降范围为：

| VectorGather 操作 | 所有对应形状和三轮的耗时下降 |
|---|---:|
| transform | 40.885385%～79.021993% |
| split | 19.237522%～29.758525% |
| fused | 21.967081%～29.329016% |

该范围支持在匹配条件下显式尝试 `--vector-scale`，**不能写成任意 rows≥257、任意 dtype/scale 或其他 NPU 都有同样收益**。生产构建默认保持 OFF，未根据有限基准更改全局派发。

N≥64 的完整受测范围共 90 个向量条件，其中 79 个三轮每轮至少降时 5%，最差单轮约退化 2.950976%。N=16 的 30 个条件只有 18 个满足该稳定规则，最差单轮约退化 4.506625%；小 rows 和短 N 应按实际输入重新测量。上述百分比是逐配置比较，没有应用工作负载权重，**不能推导未测量的应用级总提速或端到端收益**。

### 可复算的毫秒表和分析来源

所有派生时间表使用毫秒，保留全部正负例：

| 表 | 行数 | 内容 |
|---|---:|---|
| [timings_median_ms.csv](https://github.com/a962695448-rgb/Learning-CUDA/blob/3676727fc21ea27bda668d743d10e52d7e138f68/03_hadamard_tc/a962695448-rgb/results/ascend/ab_f614762/analysis/timings_median_ms.csv) | 1680 | 变体/轮次/形状/精度/方法的五组统计 |
| [off_on_paired_rounds_ms.csv](https://github.com/a962695448-rgb/Learning-CUDA/blob/3676727fc21ea27bda668d743d10e52d7e138f68/03_hadamard_tc/a962695448-rgb/results/ascend/ab_f614762/analysis/off_on_paired_rounds_ms.csv) | 840 | 每轮 OFF/ON 配对及耗时变化 |
| [off_on_three_round_stability.csv](https://github.com/a962695448-rgb/Learning-CUDA/blob/3676727fc21ea27bda668d743d10e52d7e138f68/03_hadamard_tc/a962695448-rgb/results/ascend/ab_f614762/analysis/off_on_three_round_stability.csv) | 280 | 三轮逐配置稳定性分类 |
| [within_variant_scalar_vector_ms.csv](https://github.com/a962695448-rgb/Learning-CUDA/blob/3676727fc21ea27bda668d743d10e52d7e138f68/03_hadamard_tc/a962695448-rgb/results/ascend/ab_f614762/analysis/within_variant_scalar_vector_ms.csv) | 720 | 同一变体、同一 block_dim 内 Scalar/Vector 对照 |
| [within_variant_vector_fusion_ms.csv](https://github.com/a962695448-rgb/Learning-CUDA/blob/3676727fc21ea27bda668d743d10e52d7e138f68/03_hadamard_tc/a962695448-rgb/results/ascend/ab_f614762/analysis/within_variant_vector_fusion_ms.csv) | 240 | 同一变体、相同条件内 vector split/fused 对照 |

这些表重复表达同一批 8400 条原始观察的统计与配对，不能把表行数相加为新增测量。变体内部 Scalar/Vector 和融合表也不能与 OFF/ON 开关效果混为一类比较。

自写 [analyze_ascend_ab.py](https://github.com/a962695448-rgb/Learning-CUDA/blob/3676727fc21ea27bda668d743d10e52d7e138f68/03_hadamard_tc/a962695448-rgb/results/ascend/ab_f614762/analysis/analyze_ascend_ab.py) 的 SHA256 为 `11a4c17b4378b18dce74b3bb92062d18afae80005cb86e32a694bdd4da3c4551`，只读原始证据、不运行 NPU。固定审计记录见 [analysis.json](https://github.com/a962695448-rgb/Learning-CUDA/blob/3676727fc21ea27bda668d743d10e52d7e138f68/03_hadamard_tc/a962695448-rgb/results/ascend/ab_f614762/analysis/analysis.json) 和 [结论.md](https://github.com/a962695448-rgb/Learning-CUDA/blob/3676727fc21ea27bda668d743d10e52d7e138f68/03_hadamard_tc/a962695448-rgb/results/ascend/ab_f614762/analysis/结论.md)。可在项目目录复算到尚不存在的新目录：

```bash
python3 results/ascend/ab_f614762/analysis/analyze_ascend_ab.py \
  --input results/ascend/ab_f614762 \
  --expected-source-id f614762 \
  --output results/ascend/recomputed_ab_analysis
```

公开 `analysis.json` 只将派生元数据的 `input_directory` 从私有本地绝对路径改为 `..`（相对于该 JSON 所在目录，指向原始 A/B 目录）；其他 JSON 字段和五份 CSV 字节均未修改。原始/公开 JSON 的双 SHA256、修改字段和范围保存在 [publication_provenance.json](https://github.com/a962695448-rgb/Learning-CUDA/blob/3676727fc21ea27bda668d743d10e52d7e138f68/03_hadamard_tc/a962695448-rgb/results/ascend/ab_f614762/analysis/publication_provenance.json)。复算脚本会记录执行者实际解析的输入路径，因此该路径字段及 JSON hash 可随目录变化；这不改变数值结果、输入 hash 或样本数。

最新原始 CSV 同时记录 `kernel_ms` 和 `kernel_us`，它们是同一 ACL timeline event 读数的两种单位，不是新增样本或新增有效位数。旧探索 CSV 如只有微秒，原件保持不变；需要毫秒时只能新增明确标注的派生表，不能改写原始证据。

计时区间应明确：NPU event 不包含分配、CPU 参考或主机/设备复制，但可能包含主机发射间的设备空闲，不能冒充隔离测得的纯内核时间或端到端延迟。只读输入复用存在缓存条件，逻辑 I/O 换算 GB/s 不是物理带宽计数器读数；没有独占/固定频率证据时不作相应宣称。

## 归档与当前局限

- [initial_probes 清单](https://github.com/a962695448-rgb/Learning-CUDA/blob/3676727fc21ea27bda668d743d10e52d7e138f68/03_hadamard_tc/a962695448-rgb/results/ascend/initial_probes/manifest.json)：25 份原始文件及自写 Add/短拷贝/RNE 探针，Add 256 元素通过、Pad/RNE 各 34 组合通过。重复单点不累计。
- [division_probe 清单](https://github.com/a962695448-rgb/Learning-CUDA/blob/3676727fc21ea27bda668d743d10e52d7e138f68/03_hadamard_tc/a962695448-rgb/results/ascend/division_probe/manifest.json)：14 份原始文件及自写两路径探针，明确矢量失败、NPU 标量子路径通过。
- [production_runs 清单](https://github.com/a962695448-rgb/Learning-CUDA/blob/3676727fc21ea27bda668d743d10e52d7e138f68/03_hadamard_tc/a962695448-rgb/results/ascend/production_runs/manifest.json)：110 份正式/探索/controller 原件，按版本与开关区分。逐文件 SHA256、源码对应关系和历史范围保留。
- [ab_f614762 清单](https://github.com/a962695448-rgb/Learning-CUDA/blob/3676727fc21ea27bda668d743d10e52d7e138f68/03_hadamard_tc/a962695448-rgb/results/ascend/ab_f614762/manifest.json)：251 份完成的交错对照原件，包含自写脚本、120 个进程数据与设备快照；另收录已独立复核的五份派生表、分析器、结论和单字段路径修改来源说明。

所有公开原件都按传输清单校验字节数与 SHA256，目录 `.gitattributes` 使用 `-text` 保留原始字节。不复制私有 CANN 头文件、SDK 生成物、租赁凭据、算力券号、SSH/Jupyter 访问资料。当前交付为原生 C++ API，不宣称已完成 PyTorch GPU 框架扩展、矩阵单元版本、其他昇腾型号适配或最优 block_dim。

正式提交及双份公开档案核验尚是单独步骤；设备释放必须在工作完成、原始结果取回并核验后执行。不同项目的验证记录分别维护；PR 与课程登记在项目所有者验收后单独执行。
