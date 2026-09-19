# Hadamard 变换加速与 INT4 融合

2026 夏季训练营 Hadamard 项目的精简提交版本，提供 NVIDIA CUDA 实现、PyTorch
前向接口，以及五种国产平台的独立入口。完整开发记录见 [证据索引](EVIDENCE.md)。

## 功能与数值约定

- 对最后一维执行 Sylvester Hadamard 变换，支持 FP16/BF16 存储、FP32 内部计算。
- 最后一维支持 1～256 的二次幂；CLI 使用 batch、seq、heads、dim 描述四维输入。
- 同时提供普通 CUDA 基线、warp 寄存器实现和稠密 WMMA Tensor Core 对照。
- 支持分步与融合 INT4：先将变换结果舍入到公开存储类型，再按行缩放、最近偶数舍入、
  截断到 [-7,7] 并打包；分步与融合的字节及 scales 要求精确一致。
- 默认 scale=1；归一化为 1/sqrt(dim)。全零行的量化 scale 为 1。

PyTorch 接受连续、非空的二维或四维 CUDA 张量，输出变换保持输入形状和 dtype。
打包结果为 uint8，最后一维为 ceil(dim/2)；行 scales 为 FP32。
接口使用调用方当前 stream，提供前向计算；不支持自动求导、FP8、任意步长和 dim>256。

## 快速开始

以下命令从本目录执行。CPU reference 检查只需要 C++17：

~~~bash
make cpu-test
~~~

NVIDIA 构建需要 CUDA Toolkit 和 sm80+ GPU。按实际设备设置 ARCH：

~~~bash
# A100 使用 ARCH=80；RTX 4090/4060 使用 ARCH=89。
make CUDA_HOME=/usr/local/cuda ARCH=89
./build/hadamard --self-test
python3 scripts/run_validation.py --label validation-new --benchmark
~~~

PyTorch 扩展通过当前 PyTorch ABI 编译：

~~~bash
python3 scripts/build_torch_extension.py --verbose
~~~

~~~python
import torch
from scripts.build_torch_extension import load_extension

op = load_extension()
x = torch.randn((2, 16), device="cuda", dtype=torch.float16)
y = op.hadamard(x, 1.0)
packed, row_scales = op.hadamard_int4(x, 1.0)
split_packed, split_scales = op.quantize_int4(y)
assert torch.equal(packed, split_packed)
assert torch.equal(row_scales, split_scales)
~~~

Hadamard 与融合接口默认 row_layout 为 original、线程数为 128。packed/auto、256 线程和 contiguous256
融合布局均为显式选项，其设备及形状范围见 [完整接口与开发记录](docs/DEVELOPMENT.md)。
Tensor Core 对照在部分已测场景中慢于 warp，相关结果完整保留。

## 平台入口

| 平台 | 构建与验收说明 |
|---|---|
| NVIDIA CUDA | 本页、[A100/4090 生产验证](reports/a100-calling-validation.md) |
| 天数 MR-V100 / COREX | [原生 API、Warp64 与结果](platforms/iluvatar/README.md) |
| 沐曦 C500 / MACA，25% sGPU 配额 | [原生 API 与结果](platforms/metax/README.md) |
| 壁仞 106M / SUPA | [原生 API、BF16 舍入与结果](platforms/biren/README.md) |
| 昇腾 910B1 / CANN | [Ascend C API 与结果](platforms/ascend/README.md) |
| 摩尔 S4000 / MUSA | [报告](platforms/moore/REPORT.zh-CN.md)、[公开复现材料](platforms/moore/repro/README.zh-CN.md) |

这些后端使用各自的 SDK、编译器和设备 API，按对应 README 独立构建。

## 验证与性能范围

- NVIDIA 生产版本在 A100 与 4090 分别通过 1,876 组 CLI 输入的多种模式、
  1,800 组固定版本第三方对照及接口/stream/偏移条件检查。
- 分步、融合 INT4 的输出字节和行 scales 精确一致；原始失败与负例均可追溯。
- 摩尔完整验证为 1,504 组输入、192 项 API 和 14 项 CLI 检查；精确量化优化的
  三轮配对计时共 6,750 条，默认融合相对上一版在已测配置中为 2.012～2.898×。
- 各平台记录各自的硬件、SDK、版本和计时方式；Graph、event 与端到端数据分别解释。

以上为首轮跨平台基线的归档结果，版本、计数与限制见 [证据索引](EVIDENCE.md)。
后续增量优化与补测见下方专题报告，各次结论只覆盖对应源码、硬件和测试范围。

## 建议审查顺序

1. include/reference.hpp：独立参考、舍入和打包约定。
2. include/kernels.cuh 及其他内核头文件：CUDA 实现与显式调优策略。
3. src/main.cu、tests/cpu_reference_test.cpp：验证与基准入口。
4. src/torch_binding.cu：张量契约和前向接口。
5. 各国产平台目录与对应报告。

## 2026-09-16 执行上下文补测

新增 [120 组流与 CUDA Graph 回归](reports/execution-context-20260916.md)：独立 CPU 参考、输入变化、输入不可变性，以及默认流错误/旧输出负例。RTX 4090 D 实测全部通过；原计算内核和性能结论保持各自的历史验证范围。

## RTX 4090 D 自动布局优化

[限定范围的 auto 路由](reports/4090d-routing-20260916.md) 在 dim 1/2/4/8/16、4096～65536 行上选用已有 packed kernel。180 组配置经过三轮配对及正确性验证，受测 auto 路径的 Graph 设备执行加速为 1.31～7.53×。默认 original 保持不变；该数字不代表普通 Python 调用或模型端到端收益。

## 复用输出缓冲区

新增 [三个 out 前向接口](reports/out-buffers-20260917.md)，可复用预分配的输出。受测小、中批量的普通 Python 调用平均加速约 1.75～2.05×；输出由调用者持有，接口返回 None。原接口、默认参数及设备核函数源码保持不变。

## 小维度独立量化

新增显式 `quantize_int4_packed` 与 `quantize_int4_packed_out`，支持 N≤16。RTX4090D 的三轮分组设备加速约 2.87–3.17×，普通 allocating/out 调用整体几何平均约 1.36×/1.89×；原接口签名与默认路径保留。见 [使用方式、完整范围与原始实验取舍](reports/quantize-packed-20260917.md)。原始结果见[固定验证档案](https://github.com/a962695448-rgb/Learning-CUDA/tree/29bb34a0b2817e282a9554bd92c3911a3c010762/03_hadamard_tc/a962695448-rgb/results/quantize-packed-20260917)。

## 单元素量化穷举

新增 [FP16/BF16 单元素量化穷举](reports/singleton-quantization-20260917.md)，覆盖 131,072 种存储编码和 128,768 个有限值的独立参考。计算简化候选未达到预设性能门槛，已保留实验记录并继续使用原生产内核。

## 成对独立量化

N=2/4/8/16 的显式 packed 量化改为每线程处理相邻两值，保持原 API。RTX4090 最终分组设备几何平均加速约 1.36–1.43×，普通 allocating/out 目标约 1.07×/1.12×；所有对照及 A/A 校准通过。见 [完整范围、失败记录与复现入口](reports/paired-quantization-20260917.md)。

## 打包变换边界验证与负实验

新增 [packed Hadamard 验证器及两版实验记录](reports/paired-hadamard-20260917.md)。基线通过 128 项有限分量/偏移配置与 24 项流/Graph 检查；两版成对变换候选均未通过普通调用门槛，生产计算源码保持 b49fff4。

## 普通调用计时诊断

[三进程交叉诊断](reports/host-attribution-20260917.md) 保存同源双模块、共享/轮换输出池和测量顺序的3240个计时块。四种配置未复现跨进程稳定超过5%的候选差异；它不替代完整矩阵，原来的两次未采用判定和生产计算保持原有范围。

## 整合入口候选的完整验证

[完整CLI/接口/性能验证](reports/integrated-pairs-20260917.md) 的正确性与所有相对性能限值通过，但一个共同波动区间未过预设绝对稳定性判据，最终仍为REJECT。生产内核保持原版，完整日志和时间线已归档。

## 相对稳定性前瞻确认

[三轮新采样确认](reports/relative-stability-confirm-20260918.md) 仍有一项配对比值波动超出预登记门槛，最终未采用。全部原始结果已归档；生产计算保持原有版本，停止对该候选重复确认。

## N32–256 半warp变换验证

[完整三轮记录](reports/wide-pairs-20260918.md) 包含416项有限分量、512项溢出兼容及全部入口回归。678设备与1356主机记录完整，部分配置退化超出门槛，因此保留原生产实现；融合目标分组的收益也按未采用候选记录。

## 最终采用：N32–256 对齐成对读写

[最终三轮验收](reports/wide-pairs-io-20260918.md)通过全部1050设备与2064主机记录、边界及数值回归。rows>=4096且满足4B对齐的warp变换使用独立半warp实现，其余回退原warp；设备目标分组Hadamard为1.303–1.387×，融合为1.386–1.408×。性能只覆盖本轮RTX4090D范围。
